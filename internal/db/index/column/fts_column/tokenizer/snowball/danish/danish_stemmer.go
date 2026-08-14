//! Generated from danish.sbl by Snowball 3.1.1 - https://snowballstem.org/

package danish

import (
	snowballRuntime "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball"
)

var A_0 = []*snowballRuntime.Among{
	{Str: "hed", A: -1, B: 1, F: nil},
	{Str: "ethed", A: 0, B: 1, F: nil},
	{Str: "ered", A: -1, B: 1, F: nil},
	{Str: "e", A: -1, B: 1, F: nil},
	{Str: "erede", A: 3, B: 1, F: nil},
	{Str: "ende", A: 3, B: 1, F: nil},
	{Str: "erende", A: 5, B: 1, F: nil},
	{Str: "ene", A: 3, B: 1, F: nil},
	{Str: "erne", A: 3, B: 1, F: nil},
	{Str: "ere", A: 3, B: 1, F: nil},
	{Str: "en", A: -1, B: 1, F: nil},
	{Str: "heden", A: 10, B: 1, F: nil},
	{Str: "eren", A: 10, B: 1, F: nil},
	{Str: "er", A: -1, B: 1, F: nil},
	{Str: "heder", A: 13, B: 1, F: nil},
	{Str: "erer", A: 13, B: 1, F: nil},
	{Str: "s", A: -1, B: 2, F: nil},
	{Str: "heds", A: 16, B: 1, F: nil},
	{Str: "es", A: 16, B: 1, F: nil},
	{Str: "endes", A: 18, B: 1, F: nil},
	{Str: "erendes", A: 19, B: 1, F: nil},
	{Str: "enes", A: 18, B: 1, F: nil},
	{Str: "ernes", A: 18, B: 1, F: nil},
	{Str: "eres", A: 18, B: 1, F: nil},
	{Str: "ens", A: 16, B: 1, F: nil},
	{Str: "hedens", A: 24, B: 1, F: nil},
	{Str: "erens", A: 24, B: 1, F: nil},
	{Str: "ers", A: 16, B: 1, F: nil},
	{Str: "ets", A: 16, B: 1, F: nil},
	{Str: "erets", A: 28, B: 1, F: nil},
	{Str: "et", A: -1, B: 1, F: nil},
	{Str: "eret", A: 30, B: 1, F: nil},
}

var A_1 = []*snowballRuntime.Among{
	{Str: "gd", A: -1, B: -1, F: nil},
	{Str: "dt", A: -1, B: -1, F: nil},
	{Str: "gt", A: -1, B: -1, F: nil},
	{Str: "kt", A: -1, B: -1, F: nil},
}

var A_2 = []*snowballRuntime.Among{
	{Str: "ig", A: -1, B: 1, F: nil},
	{Str: "lig", A: 0, B: 1, F: nil},
	{Str: "elig", A: 1, B: 1, F: nil},
	{Str: "els", A: -1, B: 1, F: nil},
	{Str: "løst", A: -1, B: 2, F: nil},
}

var G_undouble_c = []byte{53, 94, 7}

var G_v = []byte{17, 65, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 48, 0, 128}

var G_s_ending = []byte{1, 0, 0, 0, 0, 0, 0, 188, 251, 171, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64}

type Context struct {
	i_p1 int
}

func r_mark_regions(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	context.i_p1 = env.Limit
	var v_1 = env.Cursor
lab0:
	for {
	lab1:
		for {
			var v_2 = env.Cursor
		lab2:
			for {
			golab3:
				for {
				lab4:
					for {
						if !env.EqS("'") {
							break lab4
						}
						break golab3
					}
					if env.Cursor >= env.Limit {
						break lab2
					}
					env.NextChar()
				}
				break lab1
			}
			env.Cursor = v_2
			if !env.GoOutGrouping(G_v, 97, 248) {
				break lab0
			}
			env.NextChar()
			if !env.GoInGrouping(G_v, 97, 248) {
				break lab0
			}
			env.NextChar()
			break lab1
		}
		context.i_p1 = env.Cursor
		break lab0
	}
	env.Cursor = v_1
	var v_3 = env.Cursor
	if !env.Hop(3) {
		return false
	}
lab5:
	for {
		if context.i_p1 >= env.Cursor {
			break lab5
		}
		context.i_p1 = env.Cursor
		break lab5
	}
	env.Cursor = v_3
	return true
}

func r_main_suffix(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	var among_var int32
	if env.Cursor < context.i_p1 {
		return false
	}
	var v_1 = env.LimitBackward
	env.LimitBackward = context.i_p1
	env.Ket = env.Cursor
	among_var = env.FindAmongB(A_0, context)
	if among_var == 0 {
		env.LimitBackward = v_1
		return false
	}
	env.Bra = env.Cursor
	env.LimitBackward = v_1
	switch among_var {
	case 1:
		env.SliceDel()
	case 2:
		if !env.InGroupingB(G_s_ending, 39, 229) {
			return false
		}
		env.SliceDel()
	}
	return true
}

func r_consonant_pair(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	var v_1 = env.Limit - env.Cursor
	if env.Cursor < context.i_p1 {
		return false
	}
	var v_2 = env.LimitBackward
	env.LimitBackward = context.i_p1
	env.Ket = env.Cursor
	if env.FindAmongB(A_1, context) == 0 {
		env.LimitBackward = v_2
		return false
	}
	env.Bra = env.Cursor
	env.LimitBackward = v_2
	env.Cursor = env.Limit - v_1
	if env.Cursor <= env.LimitBackward {
		return false
	}
	env.PrevChar()
	env.Bra = env.Cursor
	env.SliceDel()
	return true
}

func r_other_suffix(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	var among_var int32
	var v_1 = env.Limit - env.Cursor
lab0:
	for {
		env.Ket = env.Cursor
		if !env.EqSB("st") {
			break lab0
		}
		env.Bra = env.Cursor
		if !env.EqSB("ig") {
			break lab0
		}
		env.SliceDel()
		break lab0
	}
	env.Cursor = env.Limit - v_1
	if env.Cursor < context.i_p1 {
		return false
	}
	var v_2 = env.LimitBackward
	env.LimitBackward = context.i_p1
	env.Ket = env.Cursor
	among_var = env.FindAmongB(A_2, context)
	if among_var == 0 {
		env.LimitBackward = v_2
		return false
	}
	env.Bra = env.Cursor
	env.LimitBackward = v_2
	switch among_var {
	case 1:
		env.SliceDel()
		var v_3 = env.Limit - env.Cursor
		r_consonant_pair(env, context)
		env.Cursor = env.Limit - v_3
	case 2:
		env.SliceFrom("løs")
	}
	return true
}

func r_undouble(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	var S_ch string
	if env.Cursor < context.i_p1 {
		return false
	}
	var v_1 = env.LimitBackward
	env.LimitBackward = context.i_p1
	env.Ket = env.Cursor
	if !env.InGroupingB(G_undouble_c, 98, 116) {
		env.LimitBackward = v_1
		return false
	}
	env.Bra = env.Cursor
	S_ch = env.SliceTo()
	env.LimitBackward = v_1
	if !env.EqSB(S_ch) {
		return false
	}
	env.SliceDel()
	return true
}

func Stem(env *snowballRuntime.Env) bool {
	var context = &Context{
		i_p1: 0,
	}
	_ = context
	if !r_mark_regions(env, context) {
		return false
	}
	env.LimitBackward = env.Cursor
	env.Cursor = env.Limit
	var v_1 = env.Limit - env.Cursor
	r_main_suffix(env, context)
	env.Cursor = env.Limit - v_1
	var v_2 = env.Limit - env.Cursor
	r_consonant_pair(env, context)
	env.Cursor = env.Limit - v_2
	var v_3 = env.Limit - env.Cursor
	r_other_suffix(env, context)
	env.Cursor = env.Limit - v_3
	var v_4 = env.Limit - env.Cursor
	r_undouble(env, context)
	env.Cursor = env.Limit - v_4
	env.Ket = env.Cursor
	if !env.EqSB("'") {
		return false
	}
	env.Bra = env.Cursor
	env.SliceDel()
	env.Cursor = env.LimitBackward
	return true
}
