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

package sqlengine

import "fmt"

// AnalysisError identifies a semantically invalid filter at its source
// position after syntax parsing succeeded.
type AnalysisError struct {
	Position Position
	Message  string
}

func (e *AnalysisError) Error() string {
	if e == nil {
		return "<nil>"
	}
	return fmt.Sprintf("sql filter analysis error at %d:%d (byte %d): %s", e.Position.Line, e.Position.Column, e.Position.Offset, e.Message)
}

func analysisError(position Position, format string, args ...any) error {
	return &AnalysisError{Position: position, Message: fmt.Sprintf(format, args...)}
}

var _ error = (*AnalysisError)(nil)
