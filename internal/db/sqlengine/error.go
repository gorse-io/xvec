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

// ParseError reports the exact token or byte at which lexing/parsing failed.
type ParseError struct {
	Position Position
	Message  string
}

func (e *ParseError) Error() string {
	if e == nil {
		return "sql filter: parse error"
	}
	return fmt.Sprintf("sql filter at %d:%d (byte %d): %s", e.Position.Line, e.Position.Column, e.Position.Offset, e.Message)
}

func parseError(position Position, format string, arguments ...any) error {
	return &ParseError{Position: position, Message: fmt.Sprintf(format, arguments...)}
}
