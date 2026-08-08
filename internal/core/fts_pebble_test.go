// Copyright 2026-present the zvec-go project
//
// Licensed under the Apache License, Version 2.0 (the "License");
package core

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/gorse-io/zvec/internal/indexstore"
	"github.com/stretchr/testify/require"
)

func TestFTSTermDictionaryPebbleRoundTripAndCorruption(t *testing.T) {
	dictionary := buildFTSTestDictionary(t, [][]Token{
		{{Text: "", Position: 0}, {Text: "alpha", Position: 1}, {Text: "alpha", Position: 2}},
		nil,
		{{Text: "alphabet", Position: 0}, {Text: string([]byte{0xff, 'x'}), Position: 1}},
	})
	path := filepath.Join(t.TempDir(), "fts.pebble")
	require.NoError(t, dictionary.Save(context.Background(), path))
	info, err := os.Stat(path)
	require.NoError(t, err)
	require.True(t, info.IsDir())

	reopened, err := OpenFTSTermDictionary(context.Background(), path)
	require.NoError(t, err)
	assertFTSDictionariesEqual(t, reopened, dictionary)

	require.NoError(t, os.WriteFile(filepath.Join(path, "ZVEC-INDEX"), []byte("corrupt\n"), 0o600))
	_, err = OpenFTSTermDictionary(context.Background(), path)
	require.ErrorIs(t, err, ErrCorruptFTSDictionary)
}

func TestFTSTermDictionaryPebblePostingChunks(t *testing.T) {
	documents := make([][]Token, 10_000)
	for index := range documents {
		documents[index] = []Token{{Text: "common", Position: 0}, {Text: "common", Position: 1}}
	}
	dictionary := buildFTSTestDictionary(t, documents)
	path := filepath.Join(t.TempDir(), "chunked.pebble")
	require.NoError(t, dictionary.Save(context.Background(), path))
	keys, err := inspectFTSStoreKeys(path)
	require.NoError(t, err)
	var chunks int
	for _, key := range keys {
		if len(key) > 0 && key[0] == 'p' {
			chunks++
		}
	}
	require.Greater(t, chunks, dictionary.TermCount(), "posting lists were stored as one value per term")
}

func TestValidateFTSPostingKeysHonorsCancellation(t *testing.T) {
	dictionary := buildFTSTestDictionary(t, [][]Token{{{Text: "alpha", Position: 0}}})
	path := filepath.Join(t.TempDir(), "cancel.pebble")
	require.NoError(t, dictionary.Save(context.Background(), path))
	store, err := indexstore.Open(path, indexstore.Options{ReadOnly: true})
	require.NoError(t, err)
	defer store.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	require.ErrorIs(t, validateFTSPostingKeys(ctx, store, 1), context.Canceled)
}
