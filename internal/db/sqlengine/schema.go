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

// Field describes the scalar runtime shape needed by filter analysis. A field
// with Filterable=false remains visible so diagnostics can distinguish an
// unsupported vector/FTS target from a missing field.
type Field struct {
	Name             string
	Kind             ValueKind
	Array            bool
	Nullable         bool
	Filterable       bool
	Indexed          bool
	RangeOptimized   bool
	ExtendedWildcard bool
}

// Schema is an immutable filter-analysis schema.
type Schema struct {
	fields map[string]Field
	order  []string
}

func NewSchema(fields []Field) (Schema, error) {
	schema := Schema{fields: make(map[string]Field, len(fields)), order: make([]string, 0, len(fields))}
	for index, field := range fields {
		if field.Name == "" {
			return Schema{}, fmt.Errorf("sql: field %d has an empty name", index)
		}
		if _, exists := schema.fields[field.Name]; exists {
			return Schema{}, fmt.Errorf("sql: duplicate field %q", field.Name)
		}
		if field.Filterable && !field.Kind.valid() {
			return Schema{}, fmt.Errorf("sql: filterable field %q has invalid kind %d", field.Name, field.Kind)
		}
		if field.Indexed && !field.Filterable {
			return Schema{}, fmt.Errorf("sql: indexed field %q is not filterable", field.Name)
		}
		schema.fields[field.Name] = field
		schema.order = append(schema.order, field.Name)
	}
	return schema, nil
}

func (s Schema) Field(name string) (Field, bool) {
	field, found := s.fields[name]
	return field, found
}

// Fields returns independent descriptors in schema order.
func (s Schema) Fields() []Field {
	fields := make([]Field, 0, len(s.order))
	for _, name := range s.order {
		fields = append(fields, s.fields[name])
	}
	return fields
}
