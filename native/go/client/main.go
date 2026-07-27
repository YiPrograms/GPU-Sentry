package main

/*
#cgo CFLAGS: -I../../include
#include <stdlib.h>
#include <string.h>
#include "telemetry.h"
*/
import "C"

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
	"unsafe"

	"gpu-sentry/native/internal/codeid"
	"gpu-sentry/native/protocol"

	"google.golang.org/protobuf/proto"
)

const clientVersion = "0.1.0"

var runtimeState struct {
	mu      sync.Mutex
	started bool
	stop    chan struct{}
	done    chan struct{}
}

type clientConfig struct {
	serverAddr         string
	batchSize          int
	reconnectBackoff   time.Duration
	hookVersion        string
	sessionID          string
	processInfo        *protocol.ProcessInfoEvent
	pid                uint32
	processInfoWasSent bool
}

//export sg_go_start
func sg_go_start(raw *C.SGClientConfig) {
	runtimeState.mu.Lock()
	defer runtimeState.mu.Unlock()
	if runtimeState.started {
		return
	}

	cfg := loadConfig(raw)
	runtimeState.started = true
	runtimeState.stop = make(chan struct{})
	runtimeState.done = make(chan struct{})

	go runClient(cfg, runtimeState.stop, runtimeState.done)
}

//export sg_go_stop
func sg_go_stop() {
	runtimeState.mu.Lock()
	if !runtimeState.started {
		runtimeState.mu.Unlock()
		return
	}
	stop := runtimeState.stop
	done := runtimeState.done
	runtimeState.started = false
	runtimeState.mu.Unlock()

	close(stop)
	<-done
}

func main() {}

//export sg_code_id
func sg_code_id(data unsafe.Pointer, size C.uint64_t) C.uint64_t {
	if data == nil || size == 0 {
		return 0
	}
	bytes := unsafe.Slice((*byte)(data), int(size))
	return C.uint64_t(codeid.FromBytes(bytes))
}

func loadConfig(raw *C.SGClientConfig) clientConfig {
	cfg := clientConfig{
		serverAddr:       "0.0.0.0:59400",
		batchSize:        128,
		reconnectBackoff: time.Second,
		hookVersion:      "unknown",
	}
	if raw != nil {
		if raw.server_addr != nil {
			cfg.serverAddr = C.GoString(raw.server_addr)
		}
		if raw.hook_version != nil {
			cfg.hookVersion = C.GoString(raw.hook_version)
		}
	}

	cfg.pid = uint32(os.Getpid())
	cfg.processInfo = collectProcessInfo(cfg.hookVersion)
	cfg.sessionID = makeSessionID(cfg.processInfo)
	return cfg
}

func runClient(cfg clientConfig, stop <-chan struct{}, done chan<- struct{}) {
	defer close(done)

	for {
		select {
		case <-stop:
			return
		default:
		}

		conn, err := (&net.Dialer{Timeout: cfg.reconnectBackoff}).Dial("tcp", cfg.serverAddr)
		if err != nil {
			if !sleepOrStop(cfg.reconnectBackoff, stop) {
				return
			}
			continue
		}

		err = sendHandshake(conn, &cfg)
		if err == nil {
			go controlLoop(conn, cfg.sessionID)
			err = drainLoop(conn, &cfg, stop)
		}
		conn.Close()

		if !sleepOrStop(cfg.reconnectBackoff, stop) {
			return
		}
	}
}

func sendHandshake(conn net.Conn, cfg *clientConfig) error {
	now := uint64(time.Now().UnixNano())
	if !cfg.processInfoWasSent {
		env := &protocol.Envelope{
			SessionId:   cfg.sessionID,
			Sequence:    0,
			TimestampNs: now,
			Pid:         cfg.pid,
			Event:       &protocol.Envelope_ProcessInfo{ProcessInfo: cfg.processInfo},
		}
		if err := writeEnvelope(conn, env); err != nil {
			return err
		}
		cfg.processInfoWasSent = true
		return nil
	}

	env := &protocol.Envelope{
		SessionId:   cfg.sessionID,
		Sequence:    0,
		TimestampNs: now,
		Pid:         cfg.pid,
		Event: &protocol.Envelope_ClientHello{ClientHello: &protocol.ClientHello{
			SessionId: cfg.sessionID,
			Pid:       cfg.pid,
		}},
	}
	return writeEnvelope(conn, env)
}

func drainLoop(conn net.Conn, cfg *clientConfig, stop <-chan struct{}) error {
	if cfg.batchSize <= 0 {
		cfg.batchSize = 128
	}
	eventSize := int(unsafe.Sizeof(C.SGEvent{}))
	buf := C.malloc(C.size_t(eventSize * cfg.batchSize))
	if buf == nil {
		return errors.New("failed to allocate event drain buffer")
	}
	defer C.free(buf)

	idle := 5 * time.Millisecond
	nextStats := time.Now().Add(time.Second)
	for {
		select {
		case <-stop:
			return nil
		default:
		}

		if time.Now().After(nextStats) {
			if err := writeEnvelope(conn, statsEnvelope(cfg.sessionID, cfg.pid)); err != nil {
				return err
			}
			nextStats = time.Now().Add(time.Second)
		}

		n := C.sg_ring_pop_batch((*C.SGEvent)(buf), C.size_t(cfg.batchSize))
		if n == 0 {
			time.Sleep(idle)
			continue
		}

		events := unsafe.Slice((*C.SGEvent)(buf), int(n))
		for i := range events {
			env := convertEvent(&events[i], cfg.sessionID)
			if env == nil {
				continue
			}
			if err := writeEnvelope(conn, env); err != nil {
				C.sg_ring_release_batch((*C.SGEvent)(buf), n)
				return err
			}
		}
		C.sg_ring_release_batch((*C.SGEvent)(buf), n)
	}
}

func convertEvent(ev *C.SGEvent, sessionID string) *protocol.Envelope {
	env := &protocol.Envelope{
		SessionId:   sessionID,
		Sequence:    uint64(ev.sequence),
		TimestampNs: uint64(ev.timestamp_ns),
		Pid:         uint32(ev.pid),
		Tid:         uint32(ev.tid),
	}

	switch uint32(ev.kind) {
	case C.SG_EVENT_CODE:
		data := C.GoBytes(unsafe.Pointer(ev.code.data), C.int(ev.code.data_size))
		env.Event = &protocol.Envelope_Code{Code: &protocol.CodeEvent{
			CodeId:   uint64(ev.code.code_id),
			CodeType: uint32(ev.code.code_type),
			Data:     data,
		}}
	case C.SG_EVENT_KERNEL_LAUNCH:
		launch := &protocol.KernelLaunchEvent{
			KernelName:     cStringFromArray(unsafe.Pointer(&ev.launch.kernel_name[0])),
			GridDimX:       uint32(ev.launch.grid_dim_x),
			GridDimY:       uint32(ev.launch.grid_dim_y),
			GridDimZ:       uint32(ev.launch.grid_dim_z),
			BlockDimX:      uint32(ev.launch.block_dim_x),
			BlockDimY:      uint32(ev.launch.block_dim_y),
			BlockDimZ:      uint32(ev.launch.block_dim_z),
			SharedMemBytes: uint32(ev.launch.shared_mem_bytes),
			Stream:         uint64(ev.launch.stream_handle),
			DevicePciBusId: cStringFromArray(unsafe.Pointer(&ev.launch.device_pci_bus_id[0])),
		}
		if ev.launch.code_id_found != 0 {
			launch.CodeId = proto.Uint64(uint64(ev.launch.code_id))
		}
		env.Event = &protocol.Envelope_KernelLaunch{KernelLaunch: launch}
	default:
		return nil
	}
	return env
}

func statsEnvelope(sessionID string, pid uint32) *protocol.Envelope {
	var stats C.SGTelemetryStats
	C.sg_ring_stats(&stats)
	return &protocol.Envelope{
		SessionId:   sessionID,
		Sequence:    0,
		TimestampNs: uint64(time.Now().UnixNano()),
		Pid:         pid,
		Event: &protocol.Envelope_Stats{Stats: &protocol.StatsEvent{
			DroppedEvents: uint64(stats.dropped_events),
			DroppedBytes:  uint64(stats.dropped_bytes),
			QueuedEvents:  uint64(stats.queued_events),
		}},
	}
}

func writeEnvelope(w io.Writer, env *protocol.Envelope) error {
	payload, err := proto.Marshal(env)
	if err != nil {
		return err
	}
	var hdr [4]byte
	binary.BigEndian.PutUint32(hdr[:], uint32(len(payload)))
	if _, err := w.Write(hdr[:]); err != nil {
		return err
	}
	_, err = w.Write(payload)
	return err
}

func controlLoop(conn net.Conn, sessionID string) {
	for {
		payload, err := readControlFrame(conn)
		if err != nil {
			_ = conn.Close()
			return
		}
		var env protocol.ServerEnvelope
		if err := proto.Unmarshal(payload, &env); err != nil {
			continue
		}
		if env.SessionId != "" && env.SessionId != sessionID {
			continue
		}
		control := env.GetControl()
		if control == nil {
			continue
		}
		fmt.Fprintln(os.Stderr, controlMessageLine(control))
		if control.Action == protocol.ControlAction_CONTROL_ACTION_TERMINATE {
			_ = conn.Close()
			os.Exit(101)
		}
	}
}

func controlMessageLine(control *protocol.ControlCommand) string {
	if control != nil && control.Message != "" {
		return control.Message
	}
	return "[GPU-Sentry] Control command received."
}

func readControlFrame(r io.Reader) ([]byte, error) {
	var hdr [4]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return nil, err
	}
	n := binary.BigEndian.Uint32(hdr[:])
	buf := make([]byte, n)
	_, err := io.ReadFull(r, buf)
	return buf, err
}

func sleepOrStop(d time.Duration, stop <-chan struct{}) bool {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-stop:
		return false
	case <-timer.C:
		return true
	}
}

func collectProcessInfo(hookVersion string) *protocol.ProcessInfoEvent {
	hostname, _ := os.Hostname()
	exe, _ := os.Executable()
	cwd, _ := os.Getwd()
	return &protocol.ProcessInfoEvent{
		Ppid:               uint32(os.Getppid()),
		Uid:                uint32(os.Getuid()),
		Gid:                uint32(os.Getgid()),
		Hostname:           hostname,
		ExePath:            exe,
		Cwd:                cwd,
		Argv:               append([]string(nil), os.Args...),
		ProcessStartTimeNs: readProcessStartTimeNS(),
		HookVersion:        hookVersion,
		ClientVersion:      clientVersion,
	}
}

func makeSessionID(info *protocol.ProcessInfoEvent) string {
	var nonce [16]byte
	_, _ = rand.Read(nonce[:])
	seed := strings.Join([]string{
		info.Hostname,
		strconv.Itoa(os.Getpid()),
		strconv.FormatUint(info.ProcessStartTimeNs, 10),
		hex.EncodeToString(nonce[:]),
	}, ":")
	sum := sha256.Sum256([]byte(seed))
	return hex.EncodeToString(sum[:16])
}

func readProcessStartTimeNS() uint64 {
	data, err := os.ReadFile("/proc/self/stat")
	if err != nil {
		return uint64(time.Now().UnixNano())
	}
	line := string(data)
	end := strings.LastIndex(line, ")")
	if end < 0 || end+2 >= len(line) {
		return uint64(time.Now().UnixNano())
	}
	fields := strings.Fields(line[end+2:])
	if len(fields) < 20 {
		return uint64(time.Now().UnixNano())
	}
	startTicks, err := strconv.ParseUint(fields[19], 10, 64)
	if err != nil {
		return uint64(time.Now().UnixNano())
	}
	const clockTicksPerSecond = 100
	startOffsetNS := startTicks * uint64(time.Second) / clockTicksPerSecond
	if bootTimeNS := readBootTimeNS(); bootTimeNS != 0 {
		return bootTimeNS + startOffsetNS
	}
	return startOffsetNS
}

func readBootTimeNS() uint64 {
	data, err := os.ReadFile("/proc/stat")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(data), "\n") {
		if !strings.HasPrefix(line, "btime ") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) != 2 {
			return 0
		}
		sec, err := strconv.ParseUint(fields[1], 10, 64)
		if err != nil {
			return 0
		}
		return sec * uint64(time.Second)
	}
	return 0
}

func cStringFromArray(ptr unsafe.Pointer) string {
	return C.GoString((*C.char)(ptr))
}
