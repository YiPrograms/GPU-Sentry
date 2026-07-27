package main

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"sync"
	"time"
)

const defaultOnlineConfigPath = "config.json"
const defaultControlMessage = "GPU-Sentry stopped a suspected mining process"
const socketTimeout = time.Second
const reconnectBackoff = time.Second

type onlineConfig struct {
	ListenAddress     string
	ProcessorSocket   string
	CaptureDir        string
	FrameMaxBytes     int
	LaunchBatchSize   int
	FlushIntervalMs   int
	MaxQueuedLaunches int
}

type projectConfig struct {
	Collector struct {
		ListenAddress     string `json:"listen_address"`
		ProcessorSocket   string `json:"processor_socket"`
		FrameMaxBytes     int    `json:"frame_max_bytes"`
		LaunchBatchSize   int    `json:"launch_batch_size"`
		FlushIntervalMs   int    `json:"flush_interval_ms"`
		MaxQueuedLaunches int    `json:"max_queued_launches"`
	} `json:"collector"`
}

type processorVerdict struct {
	Type       string         `json:"type"`
	SessionID  string         `json:"session_id"`
	WindowID   string         `json:"window_id"`
	Action     string         `json:"action"`
	Suspicious bool           `json:"suspicious"`
	Reason     string         `json:"reason"`
	Message    string         `json:"message"`
	Prediction map[string]any `json:"prediction"`
}

type onlineClient struct {
	cfg       onlineConfig
	sendCh    chan map[string]any
	verdictCh chan processorVerdict
	stopCh    chan struct{}
	doneCh    chan struct{}
}

type launchBatcher struct {
	client      *onlineClient
	cfg         onlineConfig
	mu          sync.Mutex
	batches     map[string][]map[string]any
	batchCounts map[string]int
	ticker      *time.Ticker
	stopCh      chan struct{}
}

func loadOnlineConfig(path string) (onlineConfig, error) {
	file, err := os.Open(path)
	if err != nil {
		return onlineConfig{}, err
	}
	defer file.Close()
	var project projectConfig
	if err := json.NewDecoder(file).Decode(&project); err != nil {
		return onlineConfig{}, err
	}
	if project.Collector.ListenAddress == "" || project.Collector.ProcessorSocket == "" {
		return onlineConfig{}, fmt.Errorf("collector.listen_address and collector.processor_socket are required")
	}
	if project.Collector.FrameMaxBytes <= 0 ||
		project.Collector.LaunchBatchSize <= 0 ||
		project.Collector.FlushIntervalMs <= 0 ||
		project.Collector.MaxQueuedLaunches <= 0 {
		return onlineConfig{}, fmt.Errorf("collector size and timing settings must be positive")
	}
	var cfg onlineConfig
	cfg.ListenAddress = project.Collector.ListenAddress
	cfg.ProcessorSocket = project.Collector.ProcessorSocket
	cfg.FrameMaxBytes = project.Collector.FrameMaxBytes
	cfg.LaunchBatchSize = project.Collector.LaunchBatchSize
	cfg.FlushIntervalMs = project.Collector.FlushIntervalMs
	cfg.MaxQueuedLaunches = project.Collector.MaxQueuedLaunches
	cfg.CaptureDir = "artifacts/captures"
	return cfg, nil
}

func newOnlineClient(cfg onlineConfig) *onlineClient {
	return &onlineClient{
		cfg:       cfg,
		sendCh:    make(chan map[string]any, 4096),
		verdictCh: make(chan processorVerdict, 128),
		stopCh:    make(chan struct{}),
		doneCh:    make(chan struct{}),
	}
}

func (c *onlineClient) start() {
	logf("online processor client starting socket=%s", c.cfg.ProcessorSocket)
	go c.run()
}

func (c *onlineClient) stop() {
	close(c.stopCh)
	<-c.doneCh
}

func (c *onlineClient) send(message map[string]any) {
	select {
	case c.sendCh <- message:
	default:
		logf("online processor queue full; dropping type=%v session=%v", message["type"], message["session_id"])
	}
}

func (c *onlineClient) endSession(sessionID string, reason string) {
	if sessionID == "" {
		return
	}
	message := map[string]any{
		"type":       "session_end",
		"session_id": sessionID,
		"reason":     reason,
	}
	select {
	case c.sendCh <- message:
		logf("online session end queued session=%s reason=%s", shortSession(sessionID), reason)
	default:
		logf("online processor queue full; dropping session_end session=%s", shortSession(sessionID))
	}
}

func (c *onlineClient) run() {
	defer close(c.doneCh)
	for {
		select {
		case <-c.stopCh:
			return
		default:
		}
		conn, err := net.DialTimeout(
			"unix",
			c.cfg.ProcessorSocket,
			socketTimeout,
		)
		if err != nil {
			logf("online processor connect failed socket=%s error=%v", c.cfg.ProcessorSocket, err)
			if !sleepOrDone(reconnectBackoff, c.stopCh) {
				return
			}
			continue
		}
		logf("online processor connected socket=%s", c.cfg.ProcessorSocket)
		c.runConn(conn)
		_ = conn.Close()
		logf("online processor disconnected; reconnecting after %s", reconnectBackoff)
		if !sleepOrDone(reconnectBackoff, c.stopCh) {
			return
		}
	}
}

func (c *onlineClient) runConn(conn net.Conn) {
	errCh := make(chan error, 1)
	go c.readLoop(conn, errCh)
	_ = writeJSONFrame(conn, map[string]any{"type": "collector_hello", "version": 1}, socketTimeout)
	for {
		select {
		case <-c.stopCh:
			_ = writeJSONFrame(conn, map[string]any{"type": "collector_shutdown"}, socketTimeout)
			return
		case err := <-errCh:
			if err != nil && !errors.Is(err, io.EOF) {
				logf("online processor read error: %v", err)
			}
			return
		case message := <-c.sendCh:
			logProcessorSend(message)
			if err := writeJSONFrame(conn, message, socketTimeout); err != nil {
				logf("online processor write error type=%v session=%v error=%v", message["type"], message["session_id"], err)
				return
			}
		}
	}
}

func (c *onlineClient) readLoop(conn net.Conn, errCh chan<- error) {
	for {
		message, err := readJSONFrame(conn, c.cfg.FrameMaxBytes, socketTimeout)
		if err != nil {
			errCh <- err
			return
		}
		switch message["type"] {
		case "detection_verdict":
			data, _ := json.Marshal(message)
			var verdict processorVerdict
			if err := json.Unmarshal(data, &verdict); err == nil {
				logf("online verdict received session=%s window=%s suspicious=%v reason=%s", shortSession(verdict.SessionID), verdict.WindowID, verdict.Suspicious, verdict.Reason)
				select {
				case c.verdictCh <- verdict:
				default:
					logf("online verdict queue full; dropping verdict session=%s", shortSession(verdict.SessionID))
				}
			}
		case "processor_error":
			logf("online processor error: %v", message["error"])
		}
	}
}

func newLaunchBatcher(client *onlineClient, cfg onlineConfig) *launchBatcher {
	b := &launchBatcher{
		client:      client,
		cfg:         cfg,
		batches:     make(map[string][]map[string]any),
		batchCounts: make(map[string]int),
		ticker:      time.NewTicker(time.Duration(cfg.FlushIntervalMs) * time.Millisecond),
		stopCh:      make(chan struct{}),
	}
	go b.run()
	return b
}

func (b *launchBatcher) add(sessionID string, launch map[string]any) {
	b.mu.Lock()
	defer b.mu.Unlock()
	rows := b.batches[sessionID]
	if len(rows) >= b.cfg.MaxQueuedLaunches {
		logf("online launch queue full session=%s; dropping launch", shortSession(sessionID))
		return
	}
	rows = append(rows, launch)
	b.batches[sessionID] = rows
	if len(rows) >= b.cfg.LaunchBatchSize {
		b.flushLocked(sessionID)
	}
}

func (b *launchBatcher) stop() {
	close(b.stopCh)
	b.ticker.Stop()
	b.flushAll()
}

func (b *launchBatcher) endSession(sessionID string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	rows := len(b.batches[sessionID])
	delete(b.batches, sessionID)
	delete(b.batchCounts, sessionID)
	if rows > 0 {
		logf("launch batch dropped for ended session=%s count=%d", shortSession(sessionID), rows)
	}
}

func (b *launchBatcher) run() {
	for {
		select {
		case <-b.stopCh:
			return
		case <-b.ticker.C:
			b.flushAll()
		}
	}
}

func (b *launchBatcher) flushAll() {
	b.mu.Lock()
	defer b.mu.Unlock()
	for sessionID := range b.batches {
		b.flushLocked(sessionID)
	}
}

func (b *launchBatcher) flushLocked(sessionID string) {
	rows := b.batches[sessionID]
	if len(rows) == 0 {
		return
	}
	delete(b.batches, sessionID)
	b.batchCounts[sessionID]++
	batchIndex := b.batchCounts[sessionID]
	b.client.send(map[string]any{
		"type":        "kernel_launch_batch",
		"session_id":  sessionID,
		"launches":    rows,
		"batch_index": batchIndex,
	})
	if shouldLogLaunchBatch(batchIndex) {
		logf("launch batch queued session=%s batch=%d count=%d", shortSession(sessionID), batchIndex, len(rows))
	}
}

func writeJSONFrame(conn net.Conn, message map[string]any, timeout time.Duration) error {
	if timeout > 0 {
		_ = conn.SetWriteDeadline(time.Now().Add(timeout))
	}
	payload, err := json.Marshal(message)
	if err != nil {
		return err
	}
	var hdr [4]byte
	binary.BigEndian.PutUint32(hdr[:], uint32(len(payload)))
	writer := bufio.NewWriter(conn)
	if _, err := writer.Write(hdr[:]); err != nil {
		return err
	}
	if _, err := writer.Write(payload); err != nil {
		return err
	}
	return writer.Flush()
}

func readJSONFrame(conn net.Conn, maxBytes int, timeout time.Duration) (map[string]any, error) {
	if timeout > 0 {
		_ = conn.SetReadDeadline(time.Now().Add(timeout))
	}
	var hdr [4]byte
	if _, err := io.ReadFull(conn, hdr[:]); err != nil {
		if netErr, ok := err.(net.Error); ok && netErr.Timeout() {
			return map[string]any{"type": "timeout"}, nil
		}
		return nil, err
	}
	n := binary.BigEndian.Uint32(hdr[:])
	if int(n) > maxBytes {
		return nil, fmt.Errorf("processor frame too large: %d > %d", n, maxBytes)
	}
	buf := make([]byte, n)
	if _, err := io.ReadFull(conn, buf); err != nil {
		return nil, err
	}
	var message map[string]any
	if err := json.Unmarshal(buf, &message); err != nil {
		return nil, err
	}
	return message, nil
}

func sleepOrDone(d time.Duration, done <-chan struct{}) bool {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-done:
		return false
	case <-timer.C:
		return true
	}
}

func absPath(path string) string {
	abs, err := filepath.Abs(path)
	if err != nil {
		return path
	}
	return abs
}

func logf(format string, args ...any) {
	timestamp := time.Now().Format(time.RFC3339)
	fmt.Fprintf(os.Stderr, "[%s] [collector] %s\n", timestamp, fmt.Sprintf(format, args...))
}

func shortSession(sessionID string) string {
	if len(sessionID) <= 12 {
		return sessionID
	}
	return sessionID[:12]
}

func logProcessorSend(message map[string]any) {
	msgType := fmt.Sprint(message["type"])
	sessionID := shortSession(fmt.Sprint(message["session_id"]))
	switch msgType {
	case "kernel_launch_batch":
		count := 0
		if launches, ok := message["launches"].([]map[string]any); ok {
			count = len(launches)
		}
		batchIndex := intFromAny(message["batch_index"])
		if shouldLogLaunchBatch(batchIndex) {
			logf("online send type=%s session=%s batch=%d launches=%d", msgType, sessionID, batchIndex, count)
		}
	case "code_object":
		logf("online send type=%s session=%s code_id=%v size=%v", msgType, sessionID, message["code_id"], message["size"])
	case "process_info", "stats", "session_end", "collector_shutdown":
		logf("online send type=%s session=%s", msgType, sessionID)
	default:
		logf("online send type=%s session=%s", msgType, sessionID)
	}
}

func shouldLogLaunchBatch(batchIndex int) bool {
	return batchIndex <= 3 || batchIndex%100 == 0
}

func intFromAny(value any) int {
	switch v := value.(type) {
	case int:
		return v
	case int64:
		return int(v)
	case float64:
		return int(v)
	default:
		return 0
	}
}
