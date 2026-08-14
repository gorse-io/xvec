// Copyright 2026-present the xvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package common

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"
)

const FileLockRetryDelay = 10 * time.Millisecond

var ErrReadOnly = errors.New("db: collection is read-only")

func IDMapCheckpointName(generation uint64) string {
	return fmt.Sprintf("idmap/idmap-%020d.pebble", generation)
}

func DeleteSnapshotName(generation uint64) string {
	return fmt.Sprintf("snapshots/delete-%020d.snap", generation)
}

func IsOwnedIDMapName(name string) bool {
	if strings.HasPrefix(name, "idmap-") && strings.HasSuffix(name, ".pebble") {
		digits := strings.TrimSuffix(strings.TrimPrefix(name, "idmap-"), ".pebble")
		if len(digits) == 20 {
			generation, err := strconv.ParseUint(digits, 10, 64)
			return err == nil && generation > 0
		}
		return false
	}
	if !strings.HasPrefix(name, ".working-") || !strings.HasSuffix(name, ".pebble") {
		return false
	}
	parts := strings.Split(strings.TrimSuffix(strings.TrimPrefix(name, ".working-"), ".pebble"), "-")
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return false
	}
	_, firstErr := strconv.ParseUint(parts[0], 10, 64)
	_, secondErr := strconv.ParseUint(parts[1], 10, 64)
	return firstErr == nil && secondErr == nil
}

func SyncDirectory(path string) error { return syncDirectory(path) }
