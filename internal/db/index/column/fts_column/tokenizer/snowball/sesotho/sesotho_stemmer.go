//! Generated from sesotho.sbl by Snowball 3.1.1 - https://snowballstem.org/

package sesotho

import (
	snowballRuntime "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball"
)

var A_0 = []*snowballRuntime.Among{
	{Str: "ba", A: -1, B: -1, F: nil},
	{Str: "boi", A: -1, B: -1, F: nil},
	{Str: "le", A: -1, B: -1, F: nil},
	{Str: "li", A: -1, B: -1, F: nil},
	{Str: "ma", A: -1, B: -1, F: nil},
	{Str: "me", A: -1, B: -1, F: nil},
	{Str: "mo", A: -1, B: -1, F: nil},
	{Str: "se", A: -1, B: -1, F: nil},
}

var A_1 = []*snowballRuntime.Among{
	{Str: "a", A: -1, B: 1, F: nil},
	{Str: "ela", A: 0, B: 1, F: nil},
	{Str: "isa", A: 0, B: 1, F: nil},
	{Str: "wa", A: 0, B: 1, F: nil},
	{Str: "ile", A: -1, B: 1, F: nil},
	{Str: "etse", A: -1, B: 1, F: nil},
	{Str: "ang", A: -1, B: 1, F: nil},
	{Str: "eng", A: -1, B: 1, F: nil},
	{Str: "ong", A: -1, B: 1, F: nil},
}

var A_2 = []*snowballRuntime.Among{
	{Str: "ana", A: -1, B: 1, F: nil},
	{Str: "nyana", A: 0, B: 1, F: nil},
	{Str: "oa", A: -1, B: 1, F: nil},
	{Str: "i", A: -1, B: 1, F: nil},
	{Str: "ano", A: -1, B: 1, F: nil},
}

var G_v = []byte{17, 65, 16}

type Context struct {
	i_pV int
}

func r_mark_regions(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	var v_1 = env.Cursor
	if !env.GoOutGrouping(G_v, 97, 117) {
		return false
	}
	env.NextChar()
	context.i_pV = env.Cursor
	env.Cursor = v_1
	var v_2 = env.Cursor
	if !env.Hop(2) {
		return false
	}
lab0:
	for {
		if env.Cursor <= context.i_pV {
			break lab0
		}
		context.i_pV = env.Cursor
		break lab0
	}
	env.Cursor = v_2
	return true
}

func r_remove_noun_prefixes(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	env.Bra = env.Cursor
	if env.FindAmong(A_0, context) == 0 {
		return false
	}
	env.Ket = env.Cursor
	var v_1 = env.Cursor
	if env.Cursor >= env.Limit {
		return false
	}
	env.NextChar()
	if env.Cursor >= env.Limit {
		return false
	}
	env.Cursor = v_1
	if !env.GoOutGrouping(G_v, 97, 117) {
		return false
	}
	env.NextChar()
	env.SliceDel()
	return true
}

func r_remove_verb_suffixes(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	if env.Cursor < context.i_pV {
		return false
	}
	var v_1 = env.LimitBackward
	env.LimitBackward = context.i_pV
	env.Ket = env.Cursor
	if env.FindAmongB(A_1, context) == 0 {
		env.LimitBackward = v_1
		return false
	}
	env.Bra = env.Cursor
	env.SliceDel()
	env.LimitBackward = v_1
	return true
}

func r_remove_nominal_suffixes(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	if env.Cursor < context.i_pV {
		return false
	}
	var v_1 = env.LimitBackward
	env.LimitBackward = context.i_pV
	env.Ket = env.Cursor
	if env.FindAmongB(A_2, context) == 0 {
		env.LimitBackward = v_1
		return false
	}
	env.Bra = env.Cursor
	env.SliceDel()
	env.LimitBackward = v_1
	return true
}

func Stem(env *snowballRuntime.Env) bool {
	var context = &Context{
		i_pV: 0,
	}
	_ = context
	if !r_mark_regions(env, context) {
		return false
	}
	env.LimitBackward = env.Cursor
	env.Cursor = env.Limit
	var v_1 = env.Limit - env.Cursor
	r_remove_nominal_suffixes(env, context)
	env.Cursor = env.Limit - v_1
	var v_2 = env.Limit - env.Cursor
	r_remove_verb_suffixes(env, context)
	env.Cursor = env.Limit - v_2
	env.Cursor = env.LimitBackward
	var v_3 = env.Cursor
	r_remove_noun_prefixes(env, context)
	env.Cursor = v_3
	return true
}
