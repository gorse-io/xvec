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

package hashutil

import "hash/crc32"

var castagnoliTable = crc32.MakeTable(crc32.Castagnoli)

// CRC32C computes the Castagnoli CRC-32 checksum used by xvec disk records.
func CRC32C(data []byte) uint32 { return crc32.Checksum(data, castagnoliTable) }

// UpdateCRC32C adds data to a previously computed CRC32C value.
func UpdateCRC32C(crc uint32, data []byte) uint32 {
	return crc32.Update(crc, castagnoliTable, data)
}
