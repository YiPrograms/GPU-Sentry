package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"sync"
	"time"

	"gpu-sentry/native/protocol"

	"google.golang.org/protobuf/proto"
)

type collector struct {
	root     string
	mu       sync.Mutex
	online   *onlineClient
	batcher  *launchBatcher
	sessions map[string]*clientSession
	cfg      onlineConfig
}

type clientSession struct {
	conn net.Conn
	mu   sync.Mutex
}

func main() {
	configPath := flag.String("config", defaultOnlineConfigPath, "project config")
	captureDir := flag.String("captures", "artifacts/captures", "capture output directory")
	flag.Parse()

	cfg, err := loadOnlineConfig(*configPath)
	if err != nil {
		fatal(err)
	}
	cfg.CaptureDir = *captureDir

	if err := os.MkdirAll(cfg.CaptureDir, 0o755); err != nil {
		fatal(err)
	}

	ln, err := net.Listen("tcp", cfg.ListenAddress)
	if err != nil {
		fatal(err)
	}
	defer ln.Close()

	c := &collector{root: cfg.CaptureDir, sessions: make(map[string]*clientSession), cfg: cfg}
	c.online = newOnlineClient(cfg)
	c.online.start()
	c.batcher = newLaunchBatcher(c.online, cfg)
	go c.handleVerdicts()
	logf("online detection enabled processor_socket=%s", cfg.ProcessorSocket)
	logf("GPU-Sentry collector listening on %s, output %s", cfg.ListenAddress, cfg.CaptureDir)

	for {
		conn, err := ln.Accept()
		if err != nil {
			fatal(err)
		}
		logf("hook client connected remote=%s", conn.RemoteAddr())
		go c.handleConn(conn)
	}
}

func (c *collector) handleConn(conn net.Conn) {
	defer conn.Close()
	var sessionID string
	defer func() {
		if sessionID != "" {
			c.batcher.endSession(sessionID)
			c.online.endSession(sessionID, "hook_connection_closed")
			c.unregisterSessionConn(sessionID, conn)
		}
	}()

	for {
		payload, err := readFrame(conn)
		if err != nil {
			if err != io.EOF && err != io.ErrUnexpectedEOF {
				logf("hook connection error session=%s error=%v", shortSession(sessionID), err)
			}
			return
		}

		var env protocol.Envelope
		if err := proto.Unmarshal(payload, &env); err != nil {
			logf("decode envelope failed: %v", err)
			return
		}
		if env.SessionId != "" {
			sessionID = env.SessionId
			c.registerSessionConn(env.SessionId, conn)
		}
		if err := c.store(&env); err != nil {
			logf("store event failed session=%s error=%v", shortSession(env.SessionId), err)
			return
		}
	}
}

func readFrame(r io.Reader) ([]byte, error) {
	var hdr [4]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return nil, err
	}
	n := binary.BigEndian.Uint32(hdr[:])
	buf := make([]byte, n)
	_, err := io.ReadFull(r, buf)
	return buf, err
}

func (c *collector) store(env *protocol.Envelope) error {
	if env.SessionId == "" {
		return fmt.Errorf("missing session_id")
	}

	sessionDir := filepath.Join(c.root, safeName(env.SessionId))

	c.mu.Lock()
	defer c.mu.Unlock()

	if err := os.MkdirAll(filepath.Join(sessionDir, "code"), 0o755); err != nil {
		return err
	}

	switch event := env.Event.(type) {
	case *protocol.Envelope_ProcessInfo:
		return c.storeProcessInfo(sessionDir, env, event.ProcessInfo)
	case *protocol.Envelope_ClientHello:
		return c.appendJSONL(sessionDir, "events.jsonl", map[string]any{
			"type":         "client_hello",
			"received_at":  time.Now().Format(time.RFC3339Nano),
			"session_id":   env.SessionId,
			"sequence":     env.Sequence,
			"timestamp_ns": env.TimestampNs,
			"pid":          env.Pid,
			"hello_pid":    event.ClientHello.Pid,
		})
	case *protocol.Envelope_Code:
		return c.storeCode(sessionDir, env, event.Code)
	case *protocol.Envelope_KernelLaunch:
		var codeID any
		if event.KernelLaunch.CodeId != nil {
			codeID = event.KernelLaunch.GetCodeId()
		}
		row := map[string]any{
			"type":              "kernel_launch",
			"received_at":       time.Now().Format(time.RFC3339Nano),
			"session_id":        env.SessionId,
			"sequence":          env.Sequence,
			"timestamp_ns":      env.TimestampNs,
			"pid":               env.Pid,
			"tid":               env.Tid,
			"kernel_name":       event.KernelLaunch.KernelName,
			"code_id":           codeID,
			"grid_dim":          []uint32{event.KernelLaunch.GridDimX, event.KernelLaunch.GridDimY, event.KernelLaunch.GridDimZ},
			"block_dim":         []uint32{event.KernelLaunch.BlockDimX, event.KernelLaunch.BlockDimY, event.KernelLaunch.BlockDimZ},
			"shared_mem_bytes":  event.KernelLaunch.SharedMemBytes,
			"stream":            event.KernelLaunch.Stream,
			"device_pci_bus_id": event.KernelLaunch.DevicePciBusId,
		}
		c.batcher.add(env.SessionId, row)
		return c.appendJSONL(sessionDir, "events.jsonl", row)
	case *protocol.Envelope_Stats:
		row := map[string]any{
			"type":           "stats",
			"received_at":    time.Now().Format(time.RFC3339Nano),
			"session_id":     env.SessionId,
			"sequence":       env.Sequence,
			"timestamp_ns":   env.TimestampNs,
			"pid":            env.Pid,
			"dropped_events": event.Stats.DroppedEvents,
			"dropped_bytes":  event.Stats.DroppedBytes,
			"queued_events":  event.Stats.QueuedEvents,
		}
		c.online.send(row)
		return c.appendJSONL(sessionDir, "events.jsonl", row)
	default:
		return c.appendJSONL(sessionDir, "events.jsonl", map[string]any{
			"type":         "unknown",
			"received_at":  time.Now().Format(time.RFC3339Nano),
			"session_id":   env.SessionId,
			"sequence":     env.Sequence,
			"timestamp_ns": env.TimestampNs,
			"pid":          env.Pid,
			"tid":          env.Tid,
		})
	}
}

func (c *collector) storeProcessInfo(sessionDir string, env *protocol.Envelope, info *protocol.ProcessInfoEvent) error {
	path := filepath.Join(sessionDir, "process.json")
	if _, err := os.Stat(path); err == nil {
		return c.appendJSONL(sessionDir, "events.jsonl", map[string]any{
			"type":         "process_info_duplicate",
			"received_at":  time.Now().Format(time.RFC3339Nano),
			"session_id":   env.SessionId,
			"sequence":     env.Sequence,
			"timestamp_ns": env.TimestampNs,
			"pid":          env.Pid,
		})
	}

	doc := map[string]any{
		"session_id":            env.SessionId,
		"received_at":           time.Now().Format(time.RFC3339Nano),
		"pid":                   env.Pid,
		"ppid":                  info.Ppid,
		"uid":                   info.Uid,
		"gid":                   info.Gid,
		"hostname":              info.Hostname,
		"exe_path":              info.ExePath,
		"cwd":                   info.Cwd,
		"argv":                  info.Argv,
		"process_start_time_ns": info.ProcessStartTimeNs,
		"hook_version":          info.HookVersion,
		"client_version":        info.ClientVersion,
	}
	data, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return err
	}
	logf("process info stored session=%s pid=%d exe=%s", shortSession(env.SessionId), env.Pid, info.ExePath)
	c.online.send(map[string]any{
		"type":         "process_info",
		"session_id":   env.SessionId,
		"sequence":     env.Sequence,
		"timestamp_ns": env.TimestampNs,
		"pid":          env.Pid,
		"process_info": doc,
	})
	return nil
}

func (c *collector) storeCode(sessionDir string, env *protocol.Envelope, code *protocol.CodeEvent) error {
	digest := sha256.Sum256(code.Data)
	expectedID := binary.BigEndian.Uint64(digest[:8])
	if code.CodeId != expectedID {
		return fmt.Errorf("code_id %016x does not match content hash %016x", code.CodeId, expectedID)
	}
	hash := hex.EncodeToString(digest[:])
	name := fmt.Sprintf("code_%016x.bin", code.CodeId)
	path := filepath.Join(sessionDir, "code", safeName(name))
	if err := os.WriteFile(path, code.Data, 0o644); err != nil {
		return err
	}
	logf("code stored session=%s code_id=%016x size=%d sha256=%s", shortSession(env.SessionId), code.CodeId, len(code.Data), hash)
	c.online.send(map[string]any{
		"type":         "code_object",
		"session_id":   env.SessionId,
		"sequence":     env.Sequence,
		"timestamp_ns": env.TimestampNs,
		"pid":          env.Pid,
		"tid":          env.Tid,
		"code_id":      code.CodeId,
		"code_type":    code.CodeType,
		"sha256":       hash,
		"size":         len(code.Data),
		"path":         absPath(path),
	})
	return c.appendJSONL(sessionDir, "events.jsonl", map[string]any{
		"type":         "code",
		"received_at":  time.Now().Format(time.RFC3339Nano),
		"session_id":   env.SessionId,
		"sequence":     env.Sequence,
		"timestamp_ns": env.TimestampNs,
		"pid":          env.Pid,
		"tid":          env.Tid,
		"code_id":      code.CodeId,
		"code_type":    code.CodeType,
		"sha256":       hash,
		"size":         len(code.Data),
		"path":         filepath.ToSlash(filepath.Join("code", safeName(name))),
	})
}

func (c *collector) registerSessionConn(sessionID string, conn net.Conn) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if current := c.sessions[sessionID]; current == nil || current.conn != conn {
		logf("session registered session=%s", shortSession(sessionID))
		c.sessions[sessionID] = &clientSession{conn: conn}
	}
}

func (c *collector) unregisterSessionConn(sessionID string, conn net.Conn) {
	c.mu.Lock()
	defer c.mu.Unlock()
	current := c.sessions[sessionID]
	if current != nil && current.conn == conn {
		delete(c.sessions, sessionID)
		logf("session unregistered session=%s", shortSession(sessionID))
	}
}

func (c *collector) sendControl(sessionID string, verdict processorVerdict) error {
	c.mu.Lock()
	session := c.sessions[sessionID]
	c.mu.Unlock()
	if session == nil {
		return fmt.Errorf("no active client connection for session %s", sessionID)
	}
	action := protocol.ControlAction_CONTROL_ACTION_LOG
	if verdict.Action == "terminate" {
		action = protocol.ControlAction_CONTROL_ACTION_TERMINATE
	}
	message := verdict.Message
	if message == "" {
		message = defaultControlMessage
	}
	miningProbability := 0.0
	if prediction := verdict.Prediction; prediction != nil {
		if value, ok := prediction["mining_probability_max"].(float64); ok {
			miningProbability = value
		}
	}
	env := &protocol.ServerEnvelope{
		SessionId:   sessionID,
		TimestampNs: uint64(time.Now().UnixNano()),
		Command: &protocol.ServerEnvelope_Control{Control: &protocol.ControlCommand{
			Action:            action,
			Message:           message,
			Reason:            verdict.Reason,
			MiningProbability: miningProbability,
			WindowId:          verdict.WindowID,
		}},
	}
	payload, err := proto.Marshal(env)
	if err != nil {
		return err
	}
	session.mu.Lock()
	defer session.mu.Unlock()
	var hdr [4]byte
	binary.BigEndian.PutUint32(hdr[:], uint32(len(payload)))
	if _, err := session.conn.Write(hdr[:]); err != nil {
		return err
	}
	_, err = session.conn.Write(payload)
	if err == nil {
		logf("hook control sent session=%s action=%s window=%s reason=%s", shortSession(sessionID), action.String(), verdict.WindowID, verdict.Reason)
	}
	return err
}

func (c *collector) handleVerdicts() {
	for verdict := range c.online.verdictCh {
		if verdict.SessionID == "" {
			continue
		}
		logf("online verdict session=%s window=%s suspicious=%v reason=%s", shortSession(verdict.SessionID), verdict.WindowID, verdict.Suspicious, verdict.Reason)
		if !verdict.Suspicious && verdict.Action != "terminate" {
			continue
		}
		if err := c.sendControl(verdict.SessionID, verdict); err != nil {
			logf("send hook control failed session=%s error=%v", shortSession(verdict.SessionID), err)
		}
	}
}

func (c *collector) appendJSONL(sessionDir string, name string, value any) error {
	path := filepath.Join(sessionDir, name)
	file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer file.Close()

	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	data = append(data, '\n')
	_, err = file.Write(data)
	return err
}

func safeName(s string) string {
	out := []byte(s)
	for i, c := range out {
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
			(c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-' {
			continue
		}
		out[i] = '_'
	}
	return string(out)
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
}
