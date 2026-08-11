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

package core

import (
	"math"
	"strconv"
	"strings"
)

// FTSQueryNodeType identifies one full-text query AST node.
type FTSQueryNodeType uint8

const (
	FTSQueryNodeTerm FTSQueryNodeType = iota + 1
	FTSQueryNodePhrase
	FTSQueryNodeAnd
	FTSQueryNodeOr
	FTSQueryNodeEmpty
)

var ftsQueryNodeTypeNames = map[FTSQueryNodeType]string{
	FTSQueryNodeTerm:   "TERM",
	FTSQueryNodePhrase: "PHRASE",
	FTSQueryNodeAnd:    "AND",
	FTSQueryNodeOr:     "OR",
	FTSQueryNodeEmpty:  "EMPTY",
}

func (t FTSQueryNodeType) String() string {
	if name, ok := ftsQueryNodeTypeNames[t]; ok {
		return name
	}
	return "UNKNOWN(" + strconv.FormatUint(uint64(t), 10) + ")"
}

// FTSQueryModifier carries Lucene-style occurrence flags. Boost is initialized
// to 1 for parser-produced nodes; field boosts remain unsupported at parse
// time. Should is reserved for the later AST rewrite/scoring stage.
type FTSQueryModifier struct {
	Must    bool
	MustNot bool
	Should  bool
	Boost   float32
}

func defaultFTSQueryModifier() FTSQueryModifier {
	return FTSQueryModifier{Boost: 1}
}

// FTSQueryNode is one immutable-by-convention query AST node. Parser results
// own all strings and slices.
type FTSQueryNode interface {
	Type() FTSQueryNodeType
	Modifier() FTSQueryModifier
	String() string
	ftsQueryNode()
	ftsQueryModifier() *FTSQueryModifier
}

// FTSTermQueryNode represents one analyzed term.
type FTSTermQueryNode struct {
	Flags FTSQueryModifier
	Term  string
}

func (*FTSTermQueryNode) Type() FTSQueryNodeType { return FTSQueryNodeTerm }
func (n *FTSTermQueryNode) Modifier() FTSQueryModifier {
	if n == nil {
		return FTSQueryModifier{}
	}
	return n.Flags
}
func (n *FTSTermQueryNode) String() string {
	if n == nil {
		return "<nil>"
	}
	return ftsModifierPrefix(n.Flags) + n.Term + ftsBoostSuffix(n.Flags)
}
func (*FTSTermQueryNode) ftsQueryNode() {}
func (n *FTSTermQueryNode) ftsQueryModifier() *FTSQueryModifier {
	return &n.Flags
}

// FTSPhraseQueryNode represents adjacent analyzed terms in exact order.
type FTSPhraseQueryNode struct {
	Flags FTSQueryModifier
	Terms []string
}

func (*FTSPhraseQueryNode) Type() FTSQueryNodeType { return FTSQueryNodePhrase }
func (n *FTSPhraseQueryNode) Modifier() FTSQueryModifier {
	if n == nil {
		return FTSQueryModifier{}
	}
	return n.Flags
}
func (n *FTSPhraseQueryNode) String() string {
	if n == nil {
		return "<nil>"
	}
	return ftsModifierPrefix(n.Flags) + `"` + strings.Join(n.Terms, " ") + `"` + ftsBoostSuffix(n.Flags)
}
func (*FTSPhraseQueryNode) ftsQueryNode() {}
func (n *FTSPhraseQueryNode) ftsQueryModifier() *FTSQueryModifier {
	return &n.Flags
}

// FTSAndQueryNode requires its positive children and excludes MustNot
// children.
type FTSAndQueryNode struct {
	Flags    FTSQueryModifier
	Children []FTSQueryNode
}

func (*FTSAndQueryNode) Type() FTSQueryNodeType { return FTSQueryNodeAnd }
func (n *FTSAndQueryNode) Modifier() FTSQueryModifier {
	if n == nil {
		return FTSQueryModifier{}
	}
	return n.Flags
}
func (n *FTSAndQueryNode) String() string {
	if n == nil {
		return "<nil>"
	}
	return ftsCompositeText("AND", n.Flags, n.Children)
}
func (*FTSAndQueryNode) ftsQueryNode() {}
func (n *FTSAndQueryNode) ftsQueryModifier() *FTSQueryModifier {
	return &n.Flags
}

// FTSOrQueryNode matches any positive child while preserving occurrence flags
// for the later AST rewrite/execution stage.
type FTSOrQueryNode struct {
	Flags    FTSQueryModifier
	Children []FTSQueryNode
}

func (*FTSOrQueryNode) Type() FTSQueryNodeType { return FTSQueryNodeOr }
func (n *FTSOrQueryNode) Modifier() FTSQueryModifier {
	if n == nil {
		return FTSQueryModifier{}
	}
	return n.Flags
}
func (n *FTSOrQueryNode) String() string {
	if n == nil {
		return "<nil>"
	}
	return ftsCompositeText("OR", n.Flags, n.Children)
}
func (*FTSOrQueryNode) ftsQueryNode() {}
func (n *FTSOrQueryNode) ftsQueryModifier() *FTSQueryModifier {
	return &n.Flags
}

// FTSEmptyQueryNode matches no documents. It is returned when syntax is valid
// but query analysis removes every bare term.
type FTSEmptyQueryNode struct {
	Flags FTSQueryModifier
}

func (*FTSEmptyQueryNode) Type() FTSQueryNodeType { return FTSQueryNodeEmpty }
func (n *FTSEmptyQueryNode) Modifier() FTSQueryModifier {
	if n == nil {
		return FTSQueryModifier{}
	}
	return n.Flags
}
func (n *FTSEmptyQueryNode) String() string {
	if n == nil {
		return "<nil>"
	}
	return ftsModifierPrefix(n.Flags) + "<empty>"
}
func (*FTSEmptyQueryNode) ftsQueryNode() {}
func (n *FTSEmptyQueryNode) ftsQueryModifier() *FTSQueryModifier {
	return &n.Flags
}

func ftsModifierPrefix(modifier FTSQueryModifier) string {
	switch {
	case modifier.Must:
		return "+"
	case modifier.MustNot:
		return "-"
	case modifier.Should:
		return "?"
	default:
		return ""
	}
}

func ftsBoostSuffix(modifier FTSQueryModifier) string {
	if math.Abs(float64(modifier.Boost-1)) < 1e-6 {
		return ""
	}
	return "^" + strconv.FormatFloat(float64(modifier.Boost), 'f', 6, 32)
}

func ftsCompositeText(name string, modifier FTSQueryModifier, children []FTSQueryNode) string {
	var builder strings.Builder
	builder.WriteString(ftsModifierPrefix(modifier))
	builder.WriteString(name)
	builder.WriteByte('(')
	for index, child := range children {
		if index > 0 {
			builder.WriteByte(' ')
		}
		if child == nil {
			builder.WriteString("<nil>")
		} else {
			builder.WriteString(child.String())
		}
	}
	builder.WriteByte(')')
	return builder.String()
}

func applyFTSQueryModifier(node FTSQueryNode, must, mustNot bool) {
	if node == nil || !must && !mustNot {
		return
	}
	modifier := node.ftsQueryModifier()
	modifier.Must = modifier.Must || must
	modifier.MustNot = modifier.MustNot || mustNot
}

var (
	_ FTSQueryNode = (*FTSTermQueryNode)(nil)
	_ FTSQueryNode = (*FTSPhraseQueryNode)(nil)
	_ FTSQueryNode = (*FTSAndQueryNode)(nil)
	_ FTSQueryNode = (*FTSOrQueryNode)(nil)
	_ FTSQueryNode = (*FTSEmptyQueryNode)(nil)
)
