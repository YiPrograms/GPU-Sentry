package main

import (
	"encoding/json"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestLoadOnlineConfig(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.json")
	data, err := json.Marshal(map[string]any{
		"collector": map[string]any{
			"listen_address":      "127.0.0.1:59400",
			"processor_socket":    "/tmp/gpu-sentry.sock",
			"frame_max_bytes":     1024,
			"launch_batch_size":   16,
			"flush_interval_ms":   10,
			"max_queued_launches": 128,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}

	cfg, err := loadOnlineConfig(path)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.ListenAddress != "127.0.0.1:59400" {
		t.Fatalf("unexpected listen address: %q", cfg.ListenAddress)
	}
	if cfg.LaunchBatchSize != 16 {
		t.Fatalf("unexpected batch size: %d", cfg.LaunchBatchSize)
	}
}

func TestJSONFrameRoundTrip(t *testing.T) {
	left, right := net.Pipe()
	defer left.Close()
	defer right.Close()

	want := map[string]any{"type": "kernel_launch_batch", "session_id": "abc"}
	errCh := make(chan error, 1)
	go func() {
		errCh <- writeJSONFrame(left, want, time.Second)
	}()
	got, err := readJSONFrame(right, 1024, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if err := <-errCh; err != nil {
		t.Fatal(err)
	}
	if got["type"] != want["type"] || got["session_id"] != want["session_id"] {
		t.Fatalf("unexpected frame: %#v", got)
	}
}

func TestLaunchBatcherFlushByCount(t *testing.T) {
	client := &onlineClient{sendCh: make(chan map[string]any, 1)}
	cfg := onlineConfig{}
	cfg.LaunchBatchSize = 2
	cfg.MaxQueuedLaunches = 8
	cfg.FlushIntervalMs = 1000
	batcher := newLaunchBatcher(client, cfg)
	defer batcher.stop()

	batcher.add("s1", map[string]any{"sequence": 1})
	batcher.add("s1", map[string]any{"sequence": 2})
	message := <-client.sendCh
	launches, ok := message["launches"].([]map[string]any)
	if message["type"] != "kernel_launch_batch" || !ok || len(launches) != 2 {
		t.Fatalf("unexpected batch: %#v", message)
	}
}
