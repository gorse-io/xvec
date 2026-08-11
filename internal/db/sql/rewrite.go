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

// RewriteStats records semantics-preserving syntax rewrites.
type RewriteStats struct {
	EqualitySets int
}

type filterRewriter interface {
	rewrite(Expr, *RewriteStats) Expr
}

type simpleFilterRewriter struct{}

func (simpleFilterRewriter) rewrite(expression Expr, stats *RewriteStats) Expr {
	return rewriteExpression(expression, stats)
}

// RewriteFilter returns a new expression tree. OR-connected equality and IN
// predicates on the same identifier are coalesced into one IN predicate.
// Unlike the pinned implementation, inequality under OR is deliberately not
// rewritten because doing so would change Boolean semantics.
func RewriteFilter(expression Expr) (Expr, RewriteStats) {
	stats := RewriteStats{}
	var rewriter filterRewriter = simpleFilterRewriter{}
	return rewriter.rewrite(expression, &stats), stats
}

func rewriteExpression(expression Expr, stats *RewriteStats) Expr {
	logical, ok := expression.(*LogicalExpr)
	if !ok {
		return expression
	}
	if logical.Operator == LogicalOr {
		terms := flattenLogical(logical, LogicalOr)
		for index := range terms {
			terms[index] = rewriteExpression(terms[index], stats)
		}
		return rewriteEqualitySets(terms, stats)
	}
	rewritten := &LogicalExpr{
		Operator: logical.Operator,
		Left:     rewriteExpression(logical.Left, stats),
		Right:    rewriteExpression(logical.Right, stats),
		Range:    logical.Range,
	}
	return rewritten
}

func rewriteEqualitySets(terms []Expr, stats *RewriteStats) Expr {
	type equalityGroup struct {
		index  int
		field  *IdentifierExpr
		values []*LiteralExpr
		terms  int
		range_ Span
	}
	groups := make(map[string]*equalityGroup)
	kept := make([]Expr, 0, len(terms))
	for _, term := range terms {
		field, values, ok := equalitySetCandidate(term)
		if !ok {
			kept = append(kept, term)
			continue
		}
		group, exists := groups[field.Name]
		if !exists {
			group = &equalityGroup{
				index: len(kept), field: field, values: append([]*LiteralExpr(nil), values...), terms: 1, range_: term.NodeSpan(),
			}
			groups[field.Name] = group
			kept = append(kept, term)
			continue
		}
		group.values = append(group.values, values...)
		group.terms++
		group.range_.End = term.NodeSpan().End
	}
	for _, group := range groups {
		if group.terms < 2 {
			continue
		}
		listRange := Span{Start: group.values[0].Range.Start, End: group.values[len(group.values)-1].Range.End}
		kept[group.index] = &PredicateExpr{
			Operator: PredicateIn,
			Left:     group.field,
			Right:    &ListExpr{Values: append([]*LiteralExpr(nil), group.values...), Range: listRange},
			Range:    group.range_,
		}
		stats.EqualitySets++
	}
	return joinLogical(kept, LogicalOr)
}

func flattenLogical(expression Expr, operator LogicalOperator) []Expr {
	logical, ok := expression.(*LogicalExpr)
	if !ok || logical.Operator != operator {
		return []Expr{expression}
	}
	left := flattenLogical(logical.Left, operator)
	return append(left, flattenLogical(logical.Right, operator)...)
}

func joinLogical(expressions []Expr, operator LogicalOperator) Expr {
	if len(expressions) == 0 {
		return nil
	}
	joined := expressions[0]
	for _, expression := range expressions[1:] {
		joined = &LogicalExpr{
			Operator: operator, Left: joined, Right: expression,
			Range: Span{Start: joined.NodeSpan().Start, End: expression.NodeSpan().End},
		}
	}
	return joined
}

func equalitySetCandidate(expression Expr) (*IdentifierExpr, []*LiteralExpr, bool) {
	predicate, ok := expression.(*PredicateExpr)
	if !ok || predicate.Negated {
		return nil, nil, false
	}
	field, ok := predicate.Left.(*IdentifierExpr)
	if !ok {
		return nil, nil, false
	}
	switch predicate.Operator {
	case PredicateEQ:
		literal, ok := predicate.Right.(*LiteralExpr)
		if !ok {
			return nil, nil, false
		}
		return field, []*LiteralExpr{literal}, true
	case PredicateIn:
		list, ok := predicate.Right.(*ListExpr)
		if !ok {
			return nil, nil, false
		}
		return field, list.Values, true
	default:
		return nil, nil, false
	}
}
