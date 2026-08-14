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

package sql

import (
	"fmt"

	"github.com/gorse-io/xvec/internal/ailego/container"
)

// Resolver returns one typed field value. Missing nullable fields should be
// returned as typed NULL by the storage adapter.
type Resolver func(field Field) (Value, error)

// PlanStats describes deterministic rewrites and optimizations applied while
// building an executable filter plan.
type PlanStats struct {
	EqualitySets         int
	EmptyContainRewrites int
	ConstantFolds        int
}

// Plan is an immutable, concurrency-safe forward filter execution plan.
type Plan struct {
	root       planNode
	fields     []Field
	normalized string
	stats      PlanStats
}

func (p *Plan) Evaluate(resolve Resolver) (Truth, error) {
	if p == nil || p.root == nil {
		return TruthUnknown, fmt.Errorf("sql: nil filter plan")
	}
	if resolve == nil {
		return TruthUnknown, fmt.Errorf("sql: nil field resolver")
	}
	return p.root.evaluate(resolve)
}

func (p *Plan) Match(resolve Resolver) (bool, error) {
	result, err := p.Evaluate(resolve)
	return result.Match(), err
}

func (p *Plan) Fields() []Field {
	if p == nil {
		return nil
	}
	return append([]Field(nil), p.fields...)
}

func (p *Plan) Normalized() string {
	if p == nil {
		return ""
	}
	return p.normalized
}

func (p *Plan) Explain() string {
	if p == nil || p.root == nil {
		return "<nil>"
	}
	return p.root.explain()
}

func (p *Plan) Stats() PlanStats {
	if p == nil {
		return PlanStats{}
	}
	return p.stats
}

func (p *Plan) AlwaysFalse() bool {
	if p == nil || p.root == nil {
		return false
	}
	constant, ok := p.root.(*constantPlanNode)
	return ok && constant.value == TruthFalse
}

// IndexSet maps schema field names to sealed snapshot-local indexes.
type IndexSet map[string]*InvertedIndex

// Candidates returns a safe row-ordinal prefilter. used=false means no safe
// index prefilter exists. exact reports whether the bitmap itself represents
// the complete SQL match set; callers may always forward-verify candidates.
func (p *Plan) Candidates(indexes IndexSet, totalRows uint64) (bitmap *container.Bitmap, used, exact bool, err error) {
	if p == nil || p.root == nil {
		return nil, false, false, fmt.Errorf("sql: nil filter plan")
	}
	return candidatesForNode(p.root, indexes, totalRows)
}

func candidatesForNode(node planNode, indexes IndexSet, totalRows uint64) (*container.Bitmap, bool, bool, error) {
	switch node := node.(type) {
	case *constantPlanNode:
		bitmap := container.NewBitmap(totalRows)
		if node.value == TruthTrue {
			for row := uint64(0); row < totalRows; row++ {
				bitmap.Set(row)
			}
		}
		return bitmap, true, true, nil
	case *predicatePlanNode:
		var field Field
		var arrayLength bool
		switch source := node.source.(type) {
		case fieldSource:
			field = source.field
		case arrayLengthSource:
			field = source.field
			arrayLength = true
		default:
			return nil, false, false, nil
		}
		if !field.Indexed {
			return nil, false, false, nil
		}
		index := indexes[field.Name]
		if index == nil {
			return nil, false, false, nil
		}
		var result InvertedResult
		var err error
		if arrayLength {
			result, err = index.SearchArrayLength(node.predicate)
		} else {
			result, err = index.Search(node.predicate)
		}
		if err != nil {
			return nil, false, false, err
		}
		if !result.Supported {
			return nil, false, false, nil
		}
		return result.Bitmap, true, true, nil
	case *logicalPlanNode:
		left, leftUsed, leftExact, err := candidatesForNode(node.left, indexes, totalRows)
		if err != nil {
			return nil, false, false, err
		}
		right, rightUsed, rightExact, err := candidatesForNode(node.right, indexes, totalRows)
		if err != nil {
			return nil, false, false, err
		}
		switch node.operator {
		case LogicalAnd:
			if leftUsed && rightUsed {
				left.And(right)
				return left, true, leftExact && rightExact, nil
			}
			if leftUsed {
				return left, true, false, nil
			}
			if rightUsed {
				return right, true, false, nil
			}
			return nil, false, false, nil
		case LogicalOr:
			if !leftUsed || !rightUsed {
				return nil, false, false, nil
			}
			left.Or(right)
			return left, true, leftExact && rightExact, nil
		default:
			return nil, false, false, fmt.Errorf("sql: invalid logical plan operator %d", node.operator)
		}
	default:
		return nil, false, false, nil
	}
}

type planNode interface {
	evaluate(Resolver) (Truth, error)
	explain() string
}

type constantPlanNode struct {
	value Truth
}

func (n *constantPlanNode) evaluate(Resolver) (Truth, error) { return n.value, nil }
func (n *constantPlanNode) explain() string                  { return n.value.String() }

type logicalPlanNode struct {
	operator LogicalOperator
	left     planNode
	right    planNode
}

func (n *logicalPlanNode) evaluate(resolve Resolver) (Truth, error) {
	left, err := n.left.evaluate(resolve)
	if err != nil {
		return TruthUnknown, err
	}
	if n.operator == LogicalAnd && left == TruthFalse {
		return TruthFalse, nil
	}
	if n.operator == LogicalOr && left == TruthTrue {
		return TruthTrue, nil
	}
	right, err := n.right.evaluate(resolve)
	if err != nil {
		return TruthUnknown, err
	}
	if n.operator == LogicalAnd {
		return left.And(right), nil
	}
	if n.operator == LogicalOr {
		return left.Or(right), nil
	}
	return TruthUnknown, fmt.Errorf("sql: invalid logical plan operator %d", n.operator)
}

func (n *logicalPlanNode) explain() string {
	return "(" + n.left.explain() + " " + n.operator.String() + " " + n.right.explain() + ")"
}

type valueSource interface {
	resolve(Resolver) (Value, error)
	explain() string
}

type fieldSource struct {
	field Field
}

func (s fieldSource) resolve(resolve Resolver) (Value, error) { return resolve(s.field) }
func (s fieldSource) explain() string                         { return s.field.Name }

type arrayLengthSource struct {
	field Field
}

func (s arrayLengthSource) resolve(resolve Resolver) (Value, error) {
	value, err := resolve(s.field)
	if err != nil {
		return Value{}, err
	}
	if value.IsNull() {
		return NullValue(ValueUint32, false)
	}
	length, ok := value.Len()
	if !ok {
		return Value{}, fmt.Errorf("sql: array_length resolver returned %s for %q", value.describe(), s.field.Name)
	}
	if uint64(length) > uint64(^uint32(0)) {
		return Value{}, fmt.Errorf("sql: array length for %q exceeds UINT32", s.field.Name)
	}
	return Uint32Value(uint32(length)), nil
}

func (s arrayLengthSource) explain() string { return "array_length(" + s.field.Name + ")" }

type predicatePlanNode struct {
	source    valueSource
	predicate BoundPredicate
	text      string
}

func (n *predicatePlanNode) evaluate(resolve Resolver) (Truth, error) {
	value, err := n.source.resolve(resolve)
	if err != nil {
		return TruthUnknown, err
	}
	return n.predicate.Evaluate(value)
}

func (n *predicatePlanNode) explain() string { return n.text }

type filterOptimizer interface {
	optimize(planNode, *PlanStats) planNode
}

type forwardFilterOptimizer struct{}

func (forwardFilterOptimizer) optimize(node planNode, stats *PlanStats) planNode {
	node = rewriteBoundPlan(node, stats)
	return optimizePlan(node, stats)
}

func rewriteBoundPlan(node planNode, stats *PlanStats) planNode {
	switch node := node.(type) {
	case *logicalPlanNode:
		node.left = rewriteBoundPlan(node.left, stats)
		node.right = rewriteBoundPlan(node.right, stats)
		return node
	case *predicatePlanNode:
		operator := node.predicate.Operator()
		if node.predicate.SetSize() != 0 || operator != PredicateContainAll && operator != PredicateContainAny {
			return node
		}
		stats.EmptyContainRewrites++
		if operator == PredicateContainAll && !node.predicate.Negated() ||
			operator == PredicateContainAny && node.predicate.Negated() {
			node.predicate = NewNullPredicate(true)
			node.text = node.source.explain() + " IS NOT NULL"
			return node
		}
		return &constantPlanNode{value: TruthFalse}
	default:
		return node
	}
}

func optimizePlan(node planNode, stats *PlanStats) planNode {
	logical, ok := node.(*logicalPlanNode)
	if !ok {
		return node
	}
	logical.left = optimizePlan(logical.left, stats)
	logical.right = optimizePlan(logical.right, stats)
	left, leftConstant := logical.left.(*constantPlanNode)
	right, rightConstant := logical.right.(*constantPlanNode)
	if logical.operator == LogicalAnd {
		if leftConstant && left.value == TruthFalse || rightConstant && right.value == TruthFalse {
			stats.ConstantFolds++
			return &constantPlanNode{value: TruthFalse}
		}
		if leftConstant && left.value == TruthTrue {
			stats.ConstantFolds++
			return logical.right
		}
		if rightConstant && right.value == TruthTrue {
			stats.ConstantFolds++
			return logical.left
		}
	}
	if logical.operator == LogicalOr {
		if leftConstant && left.value == TruthTrue || rightConstant && right.value == TruthTrue {
			stats.ConstantFolds++
			return &constantPlanNode{value: TruthTrue}
		}
		if leftConstant && left.value == TruthFalse {
			stats.ConstantFolds++
			return logical.right
		}
		if rightConstant && right.value == TruthFalse {
			stats.ConstantFolds++
			return logical.left
		}
	}
	return logical
}
