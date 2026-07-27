package codeid

import (
	"crypto/sha256"
	"encoding/binary"
)

func FromBytes(data []byte) uint64 {
	digest := sha256.Sum256(data)
	return binary.BigEndian.Uint64(digest[:8])
}
