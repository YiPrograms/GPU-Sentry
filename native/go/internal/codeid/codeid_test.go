package codeid

import "testing"

func TestFromBytes(t *testing.T) {
	const want = uint64(0xba7816bf8f01cfea)
	if got := FromBytes([]byte("abc")); got != want {
		t.Fatalf("FromBytes() = %016x, want %016x", got, want)
	}
}
