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

package core_test

import (
	"fmt"

	"github.com/gorse-io/xvec/internal/core"
)

func ExampleNewDiskANNLayout() {
	layout, err := core.NewDiskANNLayout(core.MetricL2, 1000, 128, 64)
	if err != nil {
		panic(err)
	}
	fmt.Println(layout.RecordSize(), layout.NodesPerSector(), layout.SectorsPerNode(), layout.DataLength())
	// Output: 776 5 1 819200
}
