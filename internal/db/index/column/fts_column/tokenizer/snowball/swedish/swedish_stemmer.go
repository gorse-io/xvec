//! Generated from swedish.sbl by Snowball 3.1.1 - https://snowballstem.org/

package swedish

import (
	snowballRuntime "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball"
)

var A_0 = []*snowballRuntime.Among{
	{Str: "fab", A: -1, B: -1, F: nil},
	{Str: "h", A: -1, B: -1, F: nil},
	{Str: "pak", A: -1, B: -1, F: nil},
	{Str: "rak", A: -1, B: -1, F: nil},
	{Str: "stak", A: -1, B: -1, F: nil},
	{Str: "kom", A: -1, B: -1, F: nil},
	{Str: "iet", A: -1, B: -1, F: nil},
	{Str: "cit", A: -1, B: -1, F: nil},
	{Str: "dit", A: -1, B: -1, F: nil},
	{Str: "alit", A: -1, B: -1, F: nil},
	{Str: "ilit", A: -1, B: -1, F: nil},
	{Str: "mit", A: -1, B: -1, F: nil},
	{Str: "nit", A: -1, B: -1, F: nil},
	{Str: "pit", A: -1, B: -1, F: nil},
	{Str: "rit", A: -1, B: -1, F: nil},
	{Str: "sit", A: -1, B: -1, F: nil},
	{Str: "tit", A: -1, B: -1, F: nil},
	{Str: "uit", A: -1, B: -1, F: nil},
	{Str: "ivit", A: -1, B: -1, F: nil},
	{Str: "kvit", A: -1, B: -1, F: nil},
	{Str: "xit", A: -1, B: -1, F: nil},
}

var A_1 = []*snowballRuntime.Among{
	{Str: "a", A: -1, B: 1, F: nil},
	{Str: "arna", A: 0, B: 1, F: nil},
	{Str: "erna", A: 0, B: 1, F: nil},
	{Str: "heterna", A: 2, B: 1, F: nil},
	{Str: "orna", A: 0, B: 1, F: nil},
	{Str: "ad", A: -1, B: 1, F: nil},
	{Str: "e", A: -1, B: 1, F: nil},
	{Str: "ade", A: 6, B: 1, F: nil},
	{Str: "ande", A: 6, B: 1, F: nil},
	{Str: "arne", A: 6, B: 1, F: nil},
	{Str: "are", A: 6, B: 1, F: nil},
	{Str: "aste", A: 6, B: 1, F: nil},
	{Str: "en", A: -1, B: 1, F: nil},
	{Str: "anden", A: 12, B: 1, F: nil},
	{Str: "aren", A: 12, B: 1, F: nil},
	{Str: "heten", A: 12, B: 1, F: nil},
	{Str: "ern", A: -1, B: 1, F: nil},
	{Str: "ar", A: -1, B: 1, F: nil},
	{Str: "er", A: -1, B: 1, F: nil},
	{Str: "heter", A: 18, B: 1, F: nil},
	{Str: "or", A: -1, B: 1, F: nil},
	{Str: "s", A: -1, B: 2, F: nil},
	{Str: "as", A: 21, B: 1, F: nil},
	{Str: "arnas", A: 22, B: 1, F: nil},
	{Str: "ernas", A: 22, B: 1, F: nil},
	{Str: "ornas", A: 22, B: 1, F: nil},
	{Str: "es", A: 21, B: 1, F: nil},
	{Str: "ades", A: 26, B: 1, F: nil},
	{Str: "andes", A: 26, B: 1, F: nil},
	{Str: "ens", A: 21, B: 1, F: nil},
	{Str: "arens", A: 29, B: 1, F: nil},
	{Str: "hetens", A: 29, B: 1, F: nil},
	{Str: "erns", A: 21, B: 1, F: nil},
	{Str: "at", A: -1, B: 1, F: nil},
	{Str: "et", A: -1, B: 3, F: nil},
	{Str: "andet", A: 34, B: 1, F: nil},
	{Str: "het", A: 34, B: 1, F: nil},
	{Str: "ast", A: -1, B: 1, F: nil},
}

var A_2 = []*snowballRuntime.Among{
	{Str: "dd", A: -1, B: -1, F: nil},
	{Str: "gd", A: -1, B: -1, F: nil},
	{Str: "nn", A: -1, B: -1, F: nil},
	{Str: "dt", A: -1, B: -1, F: nil},
	{Str: "gt", A: -1, B: -1, F: nil},
	{Str: "kt", A: -1, B: -1, F: nil},
	{Str: "tt", A: -1, B: -1, F: nil},
}

var A_3 = []*snowballRuntime.Among{
	{Str: "ig", A: -1, B: 1, F: nil},
	{Str: "lig", A: 0, B: 1, F: nil},
	{Str: "els", A: -1, B: 1, F: nil},
	{Str: "fullt", A: -1, B: 3, F: nil},
	{Str: "öst", A: -1, B: 2, F: nil},
}

var G_v = []byte{17, 65, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 32}

var G_s_ending = []byte{119, 127, 149}

var G_ost_ending = []byte{173, 58}

type Context struct {
	i_p1 int
}

func r_mark_regions(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	var i_x int
	context.i_p1 = env.Limit
	var v_1 = env.Cursor
	if !env.Hop(3) {
		return false
	}
	i_x = env.Cursor
	env.Cursor = v_1
	if !env.GoOutGrouping(G_v, 97, 246) {
		return false
	}
	env.NextChar()
	if !env.GoInGrouping(G_v, 97, 246) {
		return false
	}
	env.NextChar()
	context.i_p1 = env.Cursor
lab0:
	for {
		if context.i_p1 >= i_x {
			break lab0
		}
		context.i_p1 = i_x
		break lab0
	}
	return true
}

func r_et_condition(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	var v_1 = env.Limit - env.Cursor
	if !env.OutGroupingB(G_v, 97, 246) {
		return false
	}
	if !env.InGroupingB(G_v, 97, 246) {
		return false
	}
	if env.Cursor <= env.LimitBackward {
		return false
	}
	env.Cursor = env.Limit - v_1
	var v_2 = env.Limit - env.Cursor
lab0:
	for {
		if env.FindAmongB(A_0, context) == 0 {
			break lab0
		}
		return false
	}
	env.Cursor = env.Limit - v_2
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
	among_var = env.FindAmongB(A_1, context)
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
	lab0:
		for {
			var v_2 = env.Limit - env.Cursor
		lab1:
			for {
				if !env.EqSB("et") {
					break lab1
				}
				if !r_et_condition(env, context) {
					break lab1
				}
				env.Bra = env.Cursor
				break lab0
			}
			env.Cursor = env.Limit - v_2
			if !env.InGroupingB(G_s_ending, 98, 121) {
				return false
			}
			break lab0
		}
		env.SliceDel()
	case 3:
		if !r_et_condition(env, context) {
			return false
		}
		env.SliceDel()
	}
	return true
}

func r_consonant_pair(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	if env.Cursor < context.i_p1 {
		return false
	}
	var v_1 = env.LimitBackward
	env.LimitBackward = context.i_p1
	var v_2 = env.Limit - env.Cursor
	if env.FindAmongB(A_2, context) == 0 {
		env.LimitBackward = v_1
		return false
	}
	env.Cursor = env.Limit - v_2
	env.Ket = env.Cursor
	if env.Cursor <= env.LimitBackward {
		env.LimitBackward = v_1
		return false
	}
	env.PrevChar()
	env.Bra = env.Cursor
	env.SliceDel()
	env.LimitBackward = v_1
	return true
}

func r_other_suffix(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	var among_var int32
	if env.Cursor < context.i_p1 {
		return false
	}
	var v_1 = env.LimitBackward
	env.LimitBackward = context.i_p1
	env.Ket = env.Cursor
	among_var = env.FindAmongB(A_3, context)
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
		if !env.InGroupingB(G_ost_ending, 105, 118) {
			return false
		}
		env.SliceFrom("ös")
	case 3:
		env.SliceFrom("full")
	}
	return true
}

func Stem(env *snowballRuntime.Env) bool {
	var context = &Context{
		i_p1: 0,
	}
	_ = context
	var v_1 = env.Cursor
	r_mark_regions(env, context)
	env.Cursor = v_1
	env.LimitBackward = env.Cursor
	env.Cursor = env.Limit
	var v_2 = env.Limit - env.Cursor
	r_main_suffix(env, context)
	env.Cursor = env.Limit - v_2
	var v_3 = env.Limit - env.Cursor
	r_consonant_pair(env, context)
	env.Cursor = env.Limit - v_3
	var v_4 = env.Limit - env.Cursor
	r_other_suffix(env, context)
	env.Cursor = env.Limit - v_4
	env.Cursor = env.LimitBackward
	return true
}
