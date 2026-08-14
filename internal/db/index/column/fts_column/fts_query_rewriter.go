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

package ftscolumn

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
)

// ErrInvalidFTSQueryAST identifies a nil, cyclic, oversized, or otherwise
// malformed caller-supplied query tree.
var ErrInvalidFTSQueryAST = errors.New("core: invalid FTS query AST")

// SimplifyFTSQuery clones and canonicalizes node without mutating the caller's
// tree. It flattens safe same-kind composites, merges duplicate scored leaves,
// propagates empty nodes, detects contradictions, and converts OR occurrence
// modifiers into the AND/must/must-not/should form used by execution.
func SimplifyFTSQuery(ctx context.Context, node FTSQueryNode) (FTSQueryNode, error) {
	if ctx == nil {
		return nil, fmt.Errorf("%w: nil context", ErrInvalidFTSQueryAST)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if ftsNilInterface(node) {
		return nil, fmt.Errorf("%w: root is nil", ErrInvalidFTSQueryAST)
	}
	state := ftsASTCloneState{ctx: ctx, active: make(map[FTSQueryNode]bool)}
	cloned, err := state.clone(node, 0)
	if err != nil {
		return nil, err
	}
	return simplifyFTSQueryNode(ctx, cloned)
}

type ftsASTCloneState struct {
	ctx    context.Context
	active map[FTSQueryNode]bool
	nodes  int
}

func (s *ftsASTCloneState) clone(node FTSQueryNode, depth int) (FTSQueryNode, error) {
	if ftsNilInterface(node) {
		return nil, nil
	}
	if depth > MaxFTSQueryDepth {
		return nil, fmt.Errorf("%w: depth exceeds %d", ErrInvalidFTSQueryAST, MaxFTSQueryDepth)
	}
	if s.nodes&4095 == 0 {
		if err := s.ctx.Err(); err != nil {
			return nil, err
		}
	}
	if s.nodes >= MaxFTSQueryTokens {
		return nil, fmt.Errorf("%w: node count exceeds %d", ErrInvalidFTSQueryAST, MaxFTSQueryTokens)
	}
	if s.active[node] {
		return nil, fmt.Errorf("%w: query tree contains a cycle", ErrInvalidFTSQueryAST)
	}
	modifier := node.Modifier()
	if math.IsNaN(float64(modifier.Boost)) || math.IsInf(float64(modifier.Boost), 0) {
		return nil, fmt.Errorf("%w: node boost is not finite", ErrInvalidFTSQueryAST)
	}
	s.nodes++
	s.active[node] = true
	defer delete(s.active, node)

	switch typed := node.(type) {
	case *FTSTermQueryNode:
		return &FTSTermQueryNode{Flags: typed.Flags, Term: typed.Term}, nil
	case *FTSPhraseQueryNode:
		if len(typed.Terms) > MaxFTSQueryTokens {
			return nil, fmt.Errorf("%w: phrase term count exceeds %d", ErrInvalidFTSQueryAST, MaxFTSQueryTokens)
		}
		return &FTSPhraseQueryNode{Flags: typed.Flags, Terms: append([]string(nil), typed.Terms...)}, nil
	case *FTSAndQueryNode:
		children, err := s.cloneChildren(typed.Children, depth)
		if err != nil {
			return nil, err
		}
		return &FTSAndQueryNode{Flags: typed.Flags, Children: children}, nil
	case *FTSOrQueryNode:
		children, err := s.cloneChildren(typed.Children, depth)
		if err != nil {
			return nil, err
		}
		return &FTSOrQueryNode{Flags: typed.Flags, Children: children}, nil
	case *FTSEmptyQueryNode:
		return &FTSEmptyQueryNode{Flags: typed.Flags}, nil
	default:
		return nil, fmt.Errorf("%w: unknown node type %T", ErrInvalidFTSQueryAST, node)
	}
}

func (s *ftsASTCloneState) cloneChildren(children []FTSQueryNode, depth int) ([]FTSQueryNode, error) {
	if len(children) > MaxFTSQueryTokens-s.nodes {
		return nil, fmt.Errorf("%w: child count exceeds remaining node budget", ErrInvalidFTSQueryAST)
	}
	cloned := make([]FTSQueryNode, 0, len(children))
	for _, child := range children {
		copy, err := s.clone(child, depth+1)
		if err != nil {
			return nil, err
		}
		if copy != nil {
			cloned = append(cloned, copy)
		}
	}
	return cloned, nil
}

func simplifyFTSQueryNode(ctx context.Context, node FTSQueryNode) (FTSQueryNode, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	switch typed := node.(type) {
	case *FTSTermQueryNode, *FTSPhraseQueryNode, *FTSEmptyQueryNode:
		return node, nil
	case *FTSAndQueryNode:
		return simplifyFTSAnd(ctx, typed)
	case *FTSOrQueryNode:
		return simplifyFTSOr(ctx, typed)
	default:
		return nil, fmt.Errorf("%w: unknown node type %T", ErrInvalidFTSQueryAST, node)
	}
}

func simplifyFTSAnd(ctx context.Context, node *FTSAndQueryNode) (FTSQueryNode, error) {
	children := make([]FTSQueryNode, 0, len(node.Children))
	for index, child := range node.Children {
		if index&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		simplified, err := simplifyFTSQueryNode(ctx, child)
		if err != nil {
			return nil, err
		}
		if simplified != nil {
			children = append(children, simplified)
		}
	}
	node.Children = children

	children = make([]FTSQueryNode, 0, len(node.Children))
	for _, child := range node.Children {
		if child.Type() == FTSQueryNodeEmpty {
			if child.Modifier().MustNot {
				continue
			}
			return emptyFTSQueryLike(node), nil
		}
		children = append(children, child)
	}
	node.Children = flattenFTSAndChildren(children)
	if err := mergeDuplicateFTSQueryChildren(ctx, &node.Children); err != nil {
		return nil, err
	}
	conflict, err := ftsAndHasMustNotConflict(ctx, node.Children)
	if err != nil {
		return nil, err
	}
	if conflict {
		return emptyFTSQueryLike(node), nil
	}
	anyPositive := false
	for _, child := range node.Children {
		if !child.Modifier().MustNot {
			anyPositive = true
			break
		}
	}
	if !anyPositive {
		return emptyFTSQueryLike(node), nil
	}
	if len(node.Children) == 1 {
		child := node.Children[0]
		applyFTSQueryModifier(child, node.Flags.Must, node.Flags.MustNot)
		if modifier := child.Modifier(); modifier.Must && modifier.MustNot {
			return &FTSEmptyQueryNode{Flags: FTSQueryModifier{
				Must: node.Flags.Must, MustNot: node.Flags.MustNot, Boost: 1,
			}}, nil
		}
		return child, nil
	}
	return node, nil
}

func simplifyFTSOr(ctx context.Context, node *FTSOrQueryNode) (FTSQueryNode, error) {
	children := make([]FTSQueryNode, 0, len(node.Children))
	for index, child := range node.Children {
		if index&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		simplified, err := simplifyFTSQueryNode(ctx, child)
		if err != nil {
			return nil, err
		}
		if simplified != nil && simplified.Type() != FTSQueryNodeEmpty {
			children = append(children, simplified)
		}
	}
	node.Children = flattenFTSOrChildren(children)
	if err := mergeDuplicateFTSQueryChildren(ctx, &node.Children); err != nil {
		return nil, err
	}

	mustNotCount, mustCount := 0, 0
	for _, child := range node.Children {
		modifier := child.Modifier()
		if modifier.MustNot {
			mustNotCount++
		} else if modifier.Must {
			mustCount++
		}
	}
	if mustNotCount == len(node.Children) {
		return emptyFTSQueryLike(node), nil
	}
	if mustNotCount > 0 || mustCount > 0 {
		must := make([]FTSQueryNode, 0, mustCount)
		mustNot := make([]FTSQueryNode, 0, mustNotCount)
		plain := make([]FTSQueryNode, 0, len(node.Children)-mustCount-mustNotCount)
		for _, child := range node.Children {
			modifier := child.ftsQueryModifier()
			switch {
			case modifier.MustNot:
				mustNot = append(mustNot, child)
			case modifier.Must:
				modifier.Must = false
				must = append(must, child)
			default:
				plain = append(plain, child)
			}
		}
		wrapped := &FTSAndQueryNode{Flags: FTSQueryModifier{
			Must: node.Flags.Must, MustNot: node.Flags.MustNot, Boost: node.Flags.Boost,
		}, Children: must}
		if len(plain) > 0 {
			var plainPart FTSQueryNode
			if len(plain) == 1 {
				plainPart = plain[0]
			} else {
				plainPart = &FTSOrQueryNode{Flags: defaultFTSQueryModifier(), Children: plain}
			}
			if mustCount > 0 {
				plainPart.ftsQueryModifier().Should = true
			}
			wrapped.Children = append(wrapped.Children, plainPart)
		}
		wrapped.Children = append(wrapped.Children, mustNot...)
		return simplifyFTSAnd(ctx, wrapped)
	}
	if len(node.Children) == 1 {
		child := node.Children[0]
		applyFTSQueryModifier(child, node.Flags.Must, node.Flags.MustNot)
		if modifier := child.Modifier(); modifier.Must && modifier.MustNot {
			return &FTSEmptyQueryNode{Flags: FTSQueryModifier{
				Must: node.Flags.Must, MustNot: node.Flags.MustNot, Boost: 1,
			}}, nil
		}
		return child, nil
	}
	return node, nil
}

func flattenFTSAndChildren(children []FTSQueryNode) []FTSQueryNode {
	result := make([]FTSQueryNode, 0, len(children))
	for _, child := range children {
		inner, ok := child.(*FTSAndQueryNode)
		if !ok || inner.Flags.Must || inner.Flags.MustNot || ftsAnyMustNot(inner.Children) {
			result = append(result, child)
			continue
		}
		result = append(result, inner.Children...)
	}
	return result
}

func flattenFTSOrChildren(children []FTSQueryNode) []FTSQueryNode {
	result := make([]FTSQueryNode, 0, len(children))
	for _, child := range children {
		inner, ok := child.(*FTSOrQueryNode)
		if !ok || inner.Flags.Must || inner.Flags.MustNot || ftsAnyMustOrMustNot(inner.Children) {
			result = append(result, child)
			continue
		}
		result = append(result, inner.Children...)
	}
	return result
}

func ftsAnyMustNot(children []FTSQueryNode) bool {
	for _, child := range children {
		if child.Modifier().MustNot {
			return true
		}
	}
	return false
}

func ftsAnyMustOrMustNot(children []FTSQueryNode) bool {
	for _, child := range children {
		modifier := child.Modifier()
		if modifier.Must || modifier.MustNot {
			return true
		}
	}
	return false
}

func mergeDuplicateFTSQueryChildren(ctx context.Context, children *[]FTSQueryNode) error {
	result := make([]FTSQueryNode, 0, len(*children))
	buckets := make(map[uint64][]int)
	for index, child := range *children {
		if index&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		if child.Type() != FTSQueryNodeTerm && child.Type() != FTSQueryNodePhrase {
			result = append(result, child)
			continue
		}
		hash := hashFTSQueryLeaf(child, true)
		merged := false
		for _, resultIndex := range buckets[hash] {
			if !sameFTSQueryDedupKey(result[resultIndex], child) {
				continue
			}
			modifier := result[resultIndex].ftsQueryModifier()
			modifier.Boost += child.Modifier().Boost
			if math.IsInf(float64(modifier.Boost), 0) || math.IsNaN(float64(modifier.Boost)) {
				return fmt.Errorf("%w: merged boost is not finite", ErrInvalidFTSQueryAST)
			}
			merged = true
			break
		}
		if merged {
			continue
		}
		buckets[hash] = append(buckets[hash], len(result))
		result = append(result, child)
	}
	*children = result
	return nil
}

func sameFTSQueryDedupKey(left, right FTSQueryNode) bool {
	if left.Type() != right.Type() {
		return false
	}
	leftModifier, rightModifier := left.Modifier(), right.Modifier()
	if leftModifier.Must != rightModifier.Must || leftModifier.MustNot != rightModifier.MustNot {
		return false
	}
	switch typed := left.(type) {
	case *FTSTermQueryNode:
		return typed.Term == right.(*FTSTermQueryNode).Term
	case *FTSPhraseQueryNode:
		return slices.Equal(typed.Terms, right.(*FTSPhraseQueryNode).Terms)
	default:
		return false
	}
}

func sameFTSQueryLeafText(left, right FTSQueryNode) bool {
	if left.Type() != right.Type() {
		return false
	}
	switch typed := left.(type) {
	case *FTSTermQueryNode:
		return typed.Term == right.(*FTSTermQueryNode).Term
	case *FTSPhraseQueryNode:
		return slices.Equal(typed.Terms, right.(*FTSPhraseQueryNode).Terms)
	default:
		return false
	}
}

func ftsAndHasMustNotConflict(ctx context.Context, children []FTSQueryNode) (bool, error) {
	positive := make(map[uint64][]FTSQueryNode)
	negative := make(map[uint64][]FTSQueryNode)
	for index, child := range children {
		if index&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return false, err
			}
		}
		if child.Type() != FTSQueryNodeTerm && child.Type() != FTSQueryNodePhrase {
			continue
		}
		hash := hashFTSQueryLeaf(child, false)
		if child.Modifier().MustNot {
			for _, other := range positive[hash] {
				if sameFTSQueryLeafText(other, child) {
					return true, nil
				}
			}
			negative[hash] = append(negative[hash], child)
			continue
		}
		for _, other := range negative[hash] {
			if sameFTSQueryLeafText(other, child) {
				return true, nil
			}
		}
		positive[hash] = append(positive[hash], child)
	}
	return false, nil
}

func hashFTSQueryLeaf(node FTSQueryNode, includeOccurrence bool) uint64 {
	const (
		offsetBasis = uint64(14695981039346656037)
		prime       = uint64(1099511628211)
	)
	hash := offsetBasis
	appendByte := func(value byte) { hash = (hash ^ uint64(value)) * prime }
	appendUint64 := func(value uint64) {
		for shift := 0; shift < 64; shift += 8 {
			appendByte(byte(value >> shift))
		}
	}
	appendString := func(value string) {
		appendUint64(uint64(len(value)))
		for index := 0; index < len(value); index++ {
			appendByte(value[index])
		}
	}
	appendByte(byte(node.Type()))
	if includeOccurrence {
		modifier := node.Modifier()
		if modifier.Must {
			appendByte(1)
		} else {
			appendByte(0)
		}
		if modifier.MustNot {
			appendByte(1)
		} else {
			appendByte(0)
		}
	}
	switch typed := node.(type) {
	case *FTSTermQueryNode:
		appendString(typed.Term)
	case *FTSPhraseQueryNode:
		appendUint64(uint64(len(typed.Terms)))
		for _, term := range typed.Terms {
			appendString(term)
		}
	}
	return hash
}

func emptyFTSQueryLike(node FTSQueryNode) FTSQueryNode {
	modifier := node.Modifier()
	return &FTSEmptyQueryNode{Flags: FTSQueryModifier{
		Must: modifier.Must, MustNot: modifier.MustNot, Boost: modifier.Boost,
	}}
}
