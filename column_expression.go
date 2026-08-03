// Copyright 2026-present the zvec-go project
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

package zvec

import (
	"fmt"
	"math"
	"math/big"
	"strconv"
)

type columnExpression struct {
	root *columnExpressionNode
}

type columnExpressionNode struct {
	op          byte
	name        string
	literal     columnNumber
	left, right *columnExpressionNode
}

type columnNumber struct {
	null     bool
	floating bool
	integer  *big.Int
	float    float64
}

type columnExpressionParser struct {
	input  string
	pos    int
	schema CollectionSchema
}

func parseColumnExpression(input string, schema CollectionSchema) (*columnExpression, error) {
	parser := &columnExpressionParser{input: input, schema: schema}
	parser.skipWhitespace()
	root, err := parser.parseExpression()
	if err != nil {
		return nil, err
	}
	parser.skipWhitespace()
	if parser.pos != len(parser.input) {
		return nil, fmt.Errorf("unexpected character %q at byte %d", parser.input[parser.pos], parser.pos)
	}
	return &columnExpression{root: root}, nil
}

func (p *columnExpressionParser) parseExpression() (*columnExpressionNode, error) {
	left, err := p.parseTerm()
	if err != nil {
		return nil, err
	}
	for {
		p.skipWhitespace()
		if !p.consume('+') && !p.consume('-') {
			return left, nil
		}
		op := p.input[p.pos-1]
		right, err := p.parseTerm()
		if err != nil {
			return nil, err
		}
		left = &columnExpressionNode{op: op, left: left, right: right}
	}
}

func (p *columnExpressionParser) parseTerm() (*columnExpressionNode, error) {
	left, err := p.parseFactor()
	if err != nil {
		return nil, err
	}
	for {
		p.skipWhitespace()
		if !p.consume('*') && !p.consume('/') {
			return left, nil
		}
		op := p.input[p.pos-1]
		right, err := p.parseFactor()
		if err != nil {
			return nil, err
		}
		left = &columnExpressionNode{op: op, left: left, right: right}
	}
}

func (p *columnExpressionParser) parseFactor() (*columnExpressionNode, error) {
	p.skipWhitespace()
	if p.pos >= len(p.input) {
		return nil, fmt.Errorf("unexpected end of expression at byte %d", p.pos)
	}
	switch current := p.input[p.pos]; {
	case current == '(':
		p.pos++
		inner, err := p.parseExpression()
		if err != nil {
			return nil, err
		}
		p.skipWhitespace()
		if !p.consume(')') {
			return nil, fmt.Errorf("missing closing parenthesis at byte %d", p.pos)
		}
		return inner, nil
	case current == '+' || current == '-':
		p.pos++
		operand, err := p.parseFactor()
		if err != nil {
			return nil, err
		}
		if current == '+' {
			return operand, nil
		}
		return &columnExpressionNode{op: '~', left: operand}, nil
	case current >= '0' && current <= '9':
		return p.parseNumber()
	case isColumnIdentifierStart(current):
		return p.parseIdentifier()
	default:
		return nil, fmt.Errorf("unexpected character %q at byte %d", current, p.pos)
	}
}

func (p *columnExpressionParser) parseNumber() (*columnExpressionNode, error) {
	start := p.pos
	for p.pos < len(p.input) && p.input[p.pos] >= '0' && p.input[p.pos] <= '9' {
		p.pos++
	}
	floating := false
	if p.pos < len(p.input) && p.input[p.pos] == '.' {
		floating = true
		p.pos++
		for p.pos < len(p.input) && p.input[p.pos] >= '0' && p.input[p.pos] <= '9' {
			p.pos++
		}
	}
	if p.pos < len(p.input) && (p.input[p.pos] == 'e' || p.input[p.pos] == 'E') {
		floating = true
		p.pos++
		if p.pos < len(p.input) && (p.input[p.pos] == '+' || p.input[p.pos] == '-') {
			p.pos++
		}
		exponentStart := p.pos
		for p.pos < len(p.input) && p.input[p.pos] >= '0' && p.input[p.pos] <= '9' {
			p.pos++
		}
		if exponentStart == p.pos {
			return nil, fmt.Errorf("invalid exponent at byte %d", exponentStart)
		}
	}
	text := p.input[start:p.pos]
	if !floating {
		integer, err := strconv.ParseInt(text, 10, 64)
		if err == nil {
			return &columnExpressionNode{op: '#', literal: columnNumber{integer: big.NewInt(integer)}}, nil
		}
	}
	value, err := strconv.ParseFloat(text, 64)
	if err != nil || math.IsInf(value, 0) || math.IsNaN(value) {
		return nil, fmt.Errorf("invalid number %q at byte %d", text, start)
	}
	return &columnExpressionNode{op: '#', literal: columnNumber{floating: true, float: value}}, nil
}

func (p *columnExpressionParser) parseIdentifier() (*columnExpressionNode, error) {
	start := p.pos
	p.pos++
	for p.pos < len(p.input) && isColumnIdentifierContinue(p.input[p.pos]) {
		p.pos++
	}
	name := p.input[start:p.pos]
	field, found := p.schema.Field(name)
	if !found {
		return nil, fmt.Errorf("column %q does not exist", name)
	}
	if !addColumnDataTypeSupported(field.DataType) {
		return nil, fmt.Errorf("column %q is not numeric", name)
	}
	return &columnExpressionNode{op: '@', name: name}, nil
}

func (p *columnExpressionParser) skipWhitespace() {
	for p.pos < len(p.input) {
		switch p.input[p.pos] {
		case ' ', '\t', '\n', '\r', '\f', '\v':
			p.pos++
		default:
			return
		}
	}
}

func (p *columnExpressionParser) consume(token byte) bool {
	if p.pos < len(p.input) && p.input[p.pos] == token {
		p.pos++
		return true
	}
	return false
}

func isColumnIdentifierStart(value byte) bool {
	return value >= 'a' && value <= 'z' || value >= 'A' && value <= 'Z' || value == '_'
}

func isColumnIdentifierContinue(value byte) bool {
	return isColumnIdentifierStart(value) || value >= '0' && value <= '9'
}

func (e *columnExpression) evaluate(fields map[string]any, target DataType) (any, error) {
	if e == nil || e.root == nil {
		return nil, fmt.Errorf("expression is empty")
	}
	value, err := e.root.evaluate(fields)
	if err != nil {
		return nil, err
	}
	return value.cast(target)
}

func (n *columnExpressionNode) evaluate(fields map[string]any) (columnNumber, error) {
	if n == nil {
		return columnNumber{}, fmt.Errorf("expression node is nil")
	}
	switch n.op {
	case '#':
		return n.literal.clone(), nil
	case '@':
		value, found := fields[n.name]
		if !found || value == nil {
			return columnNumber{null: true}, nil
		}
		return columnNumberFromValue(value)
	case '~':
		value, err := n.left.evaluate(fields)
		if err != nil || value.null {
			return value, err
		}
		if value.floating {
			value.float = -value.float
			return value, nil
		}
		return columnNumber{integer: new(big.Int).Neg(value.integer)}, nil
	case '+', '-', '*', '/':
		left, err := n.left.evaluate(fields)
		if err != nil {
			return columnNumber{}, err
		}
		right, err := n.right.evaluate(fields)
		if err != nil {
			return columnNumber{}, err
		}
		return applyColumnArithmetic(n.op, left, right)
	default:
		return columnNumber{}, fmt.Errorf("unknown expression operation %q", n.op)
	}
}

func columnNumberFromValue(value any) (columnNumber, error) {
	integer := new(big.Int)
	switch value := value.(type) {
	case int32:
		return columnNumber{integer: integer.SetInt64(int64(value))}, nil
	case int64:
		return columnNumber{integer: integer.SetInt64(value)}, nil
	case uint32:
		return columnNumber{integer: integer.SetUint64(uint64(value))}, nil
	case uint64:
		return columnNumber{integer: integer.SetUint64(value)}, nil
	case float32:
		return columnNumber{floating: true, float: float64(value)}, nil
	case float64:
		return columnNumber{floating: true, float: value}, nil
	default:
		return columnNumber{}, fmt.Errorf("value %T is not numeric", value)
	}
}

func applyColumnArithmetic(operator byte, left, right columnNumber) (columnNumber, error) {
	if left.null || right.null {
		return columnNumber{null: true}, nil
	}
	if left.floating || right.floating {
		leftFloat, err := left.float64()
		if err != nil {
			return columnNumber{}, err
		}
		rightFloat, err := right.float64()
		if err != nil {
			return columnNumber{}, err
		}
		var result float64
		switch operator {
		case '+':
			result = leftFloat + rightFloat
		case '-':
			result = leftFloat - rightFloat
		case '*':
			result = leftFloat * rightFloat
		case '/':
			if rightFloat == 0 {
				return columnNumber{}, fmt.Errorf("division by zero")
			}
			result = leftFloat / rightFloat
		}
		if math.IsInf(result, 0) || math.IsNaN(result) {
			return columnNumber{}, fmt.Errorf("arithmetic result is not finite")
		}
		return columnNumber{floating: true, float: result}, nil
	}
	result := new(big.Int)
	switch operator {
	case '+':
		result.Add(left.integer, right.integer)
	case '-':
		result.Sub(left.integer, right.integer)
	case '*':
		result.Mul(left.integer, right.integer)
	case '/':
		if right.integer.Sign() == 0 {
			return columnNumber{}, fmt.Errorf("division by zero")
		}
		result.Quo(left.integer, right.integer)
	}
	return columnNumber{integer: result}, nil
}

func (n columnNumber) clone() columnNumber {
	if n.integer != nil {
		n.integer = new(big.Int).Set(n.integer)
	}
	return n
}

func (n columnNumber) float64() (float64, error) {
	if n.floating {
		return n.float, nil
	}
	value, _ := new(big.Float).SetInt(n.integer).Float64()
	if math.IsInf(value, 0) || math.IsNaN(value) {
		return 0, fmt.Errorf("integer is outside the floating-point range")
	}
	return value, nil
}

func (n columnNumber) cast(target DataType) (any, error) {
	if n.null {
		return nil, nil
	}
	switch target {
	case DataTypeFloat:
		value, err := n.float64()
		if err != nil || math.IsInf(float64(float32(value)), 0) {
			return nil, fmt.Errorf("result cannot be represented as FLOAT")
		}
		return float32(value), nil
	case DataTypeDouble:
		value, err := n.float64()
		if err != nil {
			return nil, fmt.Errorf("result cannot be represented as DOUBLE")
		}
		return value, nil
	}
	integer, err := n.asInteger()
	if err != nil {
		return nil, err
	}
	switch target {
	case DataTypeInt32:
		return int32(wrapColumnInteger(integer, 32, true).Int64()), nil
	case DataTypeInt64:
		return wrapColumnInteger(integer, 64, true).Int64(), nil
	case DataTypeUint32:
		return uint32(wrapColumnInteger(integer, 32, false).Uint64()), nil
	case DataTypeUint64:
		return wrapColumnInteger(integer, 64, false).Uint64(), nil
	default:
		return nil, fmt.Errorf("unsupported expression result type %s", target)
	}
}

func (n columnNumber) asInteger() (*big.Int, error) {
	if !n.floating {
		return new(big.Int).Set(n.integer), nil
	}
	if math.IsInf(n.float, 0) || math.IsNaN(n.float) {
		return nil, fmt.Errorf("non-finite value cannot be converted to an integer")
	}
	integer, _ := new(big.Float).SetFloat64(n.float).Int(nil)
	return integer, nil
}

func wrapColumnInteger(value *big.Int, bits uint, signed bool) *big.Int {
	modulus := new(big.Int).Lsh(big.NewInt(1), bits)
	result := new(big.Int).Mod(new(big.Int).Set(value), modulus)
	if signed && result.Bit(int(bits-1)) != 0 {
		result.Sub(result, modulus)
	}
	return result
}

func addColumnDataTypeSupported(dataType DataType) bool {
	switch dataType {
	case DataTypeInt32, DataTypeInt64, DataTypeUint32, DataTypeUint64, DataTypeFloat, DataTypeDouble:
		return true
	default:
		return false
	}
}
