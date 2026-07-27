package codeindex

import "testing"

func TestIndexDeduplicatesByTypeAndContent(t *testing.T) {
	index := New()

	firstDigest, duplicate := index.Register(4, 2, []byte("cubin"))
	if duplicate {
		t.Fatal("first code object marked as duplicate")
	}

	secondDigest, duplicate := index.Register(9, 2, []byte("cubin"))
	if !duplicate {
		t.Fatal("identical code object was not deduplicated")
	}
	if firstDigest != secondDigest {
		t.Fatal("identical code objects produced different digests")
	}
	if got := index.Canonical(9); got != 4 {
		t.Fatalf("canonical code ID = %d, want 4", got)
	}

	if _, duplicate := index.Register(12, 3, []byte("cubin")); duplicate {
		t.Fatal("different code types must remain separate")
	}
	if got := index.Canonical(99); got != 99 {
		t.Fatalf("unknown code ID = %d, want 99", got)
	}
}
