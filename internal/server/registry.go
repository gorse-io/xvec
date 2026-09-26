// Copyright 2026-present the xvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

// Package server provides server-side ownership of xvec collection handles.
package server

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"sync"

	"github.com/gorse-io/xvec"
)

var (
	ErrInvalidCollectionName = errors.New("invalid collection name")
	ErrSymlinkCollection     = errors.New("collection path is a symlink")
	ErrCollectionExists      = errors.New("collection already exists")
	ErrCollectionUnavailable = errors.New("collection is unavailable")
	ErrRegistryClosed        = errors.New("collection registry is closed")
)

var collectionNamePattern = regexp.MustCompile(`^[a-zA-Z0-9_-]{3,64}$`)

// CreateOptions controls persisted options that may be selected per collection.
// Runtime handle options always come from the registry configuration.
type CreateOptions struct {
	EnableMmap *bool
}

type collectionEntry struct {
	collection *xvec.Collection
	leases     int
	destroying bool
}

// Registry owns the sole writable handle for every loaded collection.
type Registry struct {
	dataDir string
	options xvec.CollectionOptions

	mu        sync.Mutex
	cond      *sync.Cond
	entries   map[string]*collectionEntry
	closed    bool
	closeDone chan struct{}
	closeErr  error
}

// CollectionRegistry is the descriptive form of Registry.
type CollectionRegistry = Registry

// NewRegistry creates a registry rooted at the canonical form of dataDir.
func NewRegistry(dataDir string, options xvec.CollectionOptions) (*Registry, error) {
	if dataDir == "" {
		return nil, fmt.Errorf("create collection registry: data directory is empty")
	}
	if err := os.MkdirAll(dataDir, 0o700); err != nil {
		return nil, fmt.Errorf("create collection registry data directory: %w", err)
	}
	absolute, err := filepath.Abs(dataDir)
	if err != nil {
		return nil, fmt.Errorf("resolve collection registry data directory: %w", err)
	}
	canonical, err := filepath.EvalSymlinks(absolute)
	if err != nil {
		return nil, fmt.Errorf("canonicalize collection registry data directory: %w", err)
	}
	info, err := os.Stat(canonical)
	if err != nil {
		return nil, fmt.Errorf("stat collection registry data directory: %w", err)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("create collection registry: data path is not a directory")
	}
	options.ReadOnly = false
	registry := &Registry{
		dataDir: canonical,
		options: options,
		entries: make(map[string]*collectionEntry),
	}
	registry.cond = sync.NewCond(&registry.mu)
	return registry, nil
}

// NewCollectionRegistry creates a registry rooted at dataDir.
func NewCollectionRegistry(dataDir string, options xvec.CollectionOptions) (*CollectionRegistry, error) {
	return NewRegistry(dataDir, options)
}

// DataDir returns the registry's absolute, symlink-resolved data directory.
func (r *Registry) DataDir() string {
	if r == nil {
		return ""
	}
	return r.dataDir
}

// Create creates and registers the collection named by schema.Name.
func (r *Registry) Create(ctx context.Context, schema xvec.CollectionSchema, createOptions CreateOptions) (*xvec.Collection, error) {
	if ctx == nil {
		return nil, fmt.Errorf("create collection: context is nil")
	}
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("create collection: %w", err)
	}
	path, err := r.collectionPath(schema.Name)
	if err != nil {
		return nil, err
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil, ErrRegistryClosed
	}
	if _, exists := r.entries[schema.Name]; exists {
		return nil, fmt.Errorf("create collection %q: %w", schema.Name, ErrCollectionExists)
	}
	if err := rejectSymlink(path); err != nil {
		return nil, err
	}
	if _, err := os.Lstat(path); err == nil {
		return nil, fmt.Errorf("create collection %q: %w", schema.Name, ErrCollectionExists)
	} else if !errors.Is(err, os.ErrNotExist) {
		return nil, fmt.Errorf("inspect collection %q: %w", schema.Name, err)
	}

	options := r.options
	if createOptions.EnableMmap != nil {
		options.EnableMmap = *createOptions.EnableMmap
	}
	collection, err := xvec.CreateAndOpen(ctx, path, schema, options)
	if err != nil {
		return nil, err
	}
	r.entries[schema.Name] = &collectionEntry{collection: collection}
	return collection, nil
}

// Acquire leases the shared collection handle. The returned release function is
// safe to call more than once.
func (r *Registry) Acquire(ctx context.Context, name string) (*xvec.Collection, func(), error) {
	if ctx == nil {
		return nil, nil, fmt.Errorf("acquire collection: context is nil")
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, fmt.Errorf("acquire collection: %w", err)
	}
	path, err := r.collectionPath(name)
	if err != nil {
		return nil, nil, err
	}

	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return nil, nil, ErrRegistryClosed
	}
	entry := r.entries[name]
	if entry != nil && entry.destroying {
		r.mu.Unlock()
		return nil, nil, fmt.Errorf("acquire collection %q: %w", name, ErrCollectionUnavailable)
	}
	if entry == nil {
		if err := rejectSymlink(path); err != nil {
			r.mu.Unlock()
			return nil, nil, err
		}
		collection, openErr := xvec.Open(ctx, path, r.options)
		if openErr != nil {
			r.mu.Unlock()
			return nil, nil, openErr
		}
		entry = &collectionEntry{collection: collection}
		r.entries[name] = entry
	}
	entry.leases++
	collection := entry.collection
	r.mu.Unlock()

	var once sync.Once
	release := func() {
		once.Do(func() {
			r.mu.Lock()
			entry.leases--
			r.cond.Broadcast()
			r.mu.Unlock()
		})
	}
	return collection, release, nil
}

// Destroy prevents new leases, waits for existing leases, and removes the
// collection from disk.
func (r *Registry) Destroy(ctx context.Context, name string) error {
	if ctx == nil {
		return fmt.Errorf("destroy collection: context is nil")
	}
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("destroy collection: %w", err)
	}
	path, err := r.collectionPath(name)
	if err != nil {
		return err
	}

	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return ErrRegistryClosed
	}
	entry := r.entries[name]
	if entry != nil && entry.destroying {
		r.mu.Unlock()
		return fmt.Errorf("destroy collection %q: %w", name, ErrCollectionUnavailable)
	}
	if err := rejectSymlink(path); err != nil {
		r.mu.Unlock()
		return err
	}
	if entry == nil {
		collection, openErr := xvec.Open(ctx, path, r.options)
		if openErr != nil {
			r.mu.Unlock()
			return openErr
		}
		entry = &collectionEntry{collection: collection}
		r.entries[name] = entry
	}
	entry.destroying = true
	stopWakeup := context.AfterFunc(ctx, func() {
		r.mu.Lock()
		r.cond.Broadcast()
		r.mu.Unlock()
	})
	defer stopWakeup()
	for entry.leases > 0 && ctx.Err() == nil {
		r.cond.Wait()
	}
	if err := ctx.Err(); err != nil {
		entry.destroying = false
		r.cond.Broadcast()
		r.mu.Unlock()
		return fmt.Errorf("destroy collection %q: %w", name, err)
	}
	r.mu.Unlock()

	destroyErr := entry.collection.Destroy(ctx)

	r.mu.Lock()
	if destroyErr == nil {
		delete(r.entries, name)
	} else {
		entry.destroying = false
	}
	r.cond.Broadcast()
	r.mu.Unlock()
	return destroyErr
}

// List returns sorted collection names from loaded handles and valid-looking
// persisted collection directories.
func (r *Registry) List() ([]string, error) {
	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return nil, ErrRegistryClosed
	}
	names := make(map[string]struct{}, len(r.entries))
	for name := range r.entries {
		names[name] = struct{}{}
	}
	r.mu.Unlock()

	children, err := os.ReadDir(r.dataDir)
	if err != nil {
		return nil, fmt.Errorf("list collection registry data directory: %w", err)
	}
	for _, child := range children {
		name := child.Name()
		if !collectionNamePattern.MatchString(name) || child.Type()&os.ModeSymlink != 0 || !child.IsDir() {
			continue
		}
		current, err := os.Lstat(filepath.Join(r.dataDir, name, "CURRENT"))
		if err != nil || !current.Mode().IsRegular() || current.Mode()&os.ModeSymlink != 0 {
			continue
		}
		names[name] = struct{}{}
	}

	result := make([]string, 0, len(names))
	for name := range names {
		result = append(result, name)
	}
	sort.Strings(result)
	return result, nil
}

// Close prevents new work, waits for all leases and destroys in progress, and
// closes every loaded handle.
func (r *Registry) Close() error {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	if r.closed {
		done := r.closeDone
		r.mu.Unlock()
		if done != nil {
			<-done
		}
		r.mu.Lock()
		err := r.closeErr
		r.mu.Unlock()
		return err
	}
	r.closed = true
	r.closeDone = make(chan struct{})
	for r.hasActiveWorkLocked() {
		r.cond.Wait()
	}
	entries := r.entries
	r.entries = make(map[string]*collectionEntry)
	r.mu.Unlock()

	errs := make([]error, 0, len(entries))
	for name, entry := range entries {
		if err := entry.collection.Close(); err != nil {
			errs = append(errs, fmt.Errorf("close collection %q: %w", name, err))
		}
	}
	closeErr := errors.Join(errs...)

	r.mu.Lock()
	r.closeErr = closeErr
	close(r.closeDone)
	r.mu.Unlock()
	return closeErr
}

func (r *Registry) hasActiveWorkLocked() bool {
	for _, entry := range r.entries {
		if entry.leases > 0 || entry.destroying {
			return true
		}
	}
	return false
}

func (r *Registry) collectionPath(name string) (string, error) {
	if r == nil {
		return "", fmt.Errorf("collection registry is nil")
	}
	if !collectionNamePattern.MatchString(name) {
		return "", fmt.Errorf("collection name %q: %w", name, ErrInvalidCollectionName)
	}
	path := filepath.Join(r.dataDir, name)
	if filepath.Dir(path) != r.dataDir {
		return "", fmt.Errorf("collection name %q escapes data directory: %w", name, ErrInvalidCollectionName)
	}
	return path, nil
}

func rejectSymlink(path string) error {
	info, err := os.Lstat(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return fmt.Errorf("inspect collection path: %w", err)
	}
	if info.Mode()&os.ModeSymlink != 0 {
		return fmt.Errorf("inspect collection path %q: %w", path, ErrSymlinkCollection)
	}
	return nil
}
