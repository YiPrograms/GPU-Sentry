package codeindex

import "crypto/sha256"

type identity struct {
	codeType uint32
	digest   [sha256.Size]byte
}

type Index struct {
	idsByContent map[identity]uint32
	canonicalIDs map[uint32]uint32
}

func New() *Index {
	return &Index{
		idsByContent: make(map[identity]uint32),
		canonicalIDs: make(map[uint32]uint32),
	}
}

func (i *Index) Register(codeID, codeType uint32, data []byte) ([sha256.Size]byte, bool) {
	digest := sha256.Sum256(data)
	key := identity{codeType: codeType, digest: digest}
	canonicalID, duplicate := i.idsByContent[key]
	if !duplicate {
		canonicalID = codeID
		i.idsByContent[key] = codeID
	}
	i.canonicalIDs[codeID] = canonicalID
	return digest, duplicate
}

func (i *Index) Canonical(codeID uint32) uint32 {
	if canonicalID, ok := i.canonicalIDs[codeID]; ok {
		return canonicalID
	}
	return codeID
}
