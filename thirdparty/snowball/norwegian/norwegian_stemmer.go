//! Generated from norwegian.sbl by Snowball 3.1.1 - https://snowballstem.org/

package norwegian

import (
	snowballRuntime "github.com/gorse-io/xvec/thirdparty/snowball"
)

var A_0 = []*snowballRuntime.Among{
	{Str: "", A: -1, B: 1, F: nil},
	{Str: "ind", A: 0, B: -1, F: nil},
	{Str: "kk", A: 0, B: -1, F: nil},
	{Str: "nk", A: 0, B: -1, F: nil},
	{Str: "amm", A: 0, B: -1, F: nil},
	{Str: "omm", A: 0, B: -1, F: nil},
	{Str: "kap", A: 0, B: -1, F: nil},
	{Str: "skap", A: 6, B: 1, F: nil},
	{Str: "pp", A: 0, B: -1, F: nil},
	{Str: "lt", A: 0, B: -1, F: nil},
	{Str: "ast", A: 0, B: -1, F: nil},
	{Str: "øst", A: 0, B: -1, F: nil},
	{Str: "v", A: 0, B: -1, F: nil},
	{Str: "hav", A: 12, B: 1, F: nil},
	{Str: "giv", A: 12, B: 1, F: nil},
}

var A_1 = []*snowballRuntime.Among{
	{Str: "a", A: -1, B: 1, F: nil},
	{Str: "e", A: -1, B: 1, F: nil},
	{Str: "ede", A: 1, B: 1, F: nil},
	{Str: "ande", A: 1, B: 1, F: nil},
	{Str: "ende", A: 1, B: 1, F: nil},
	{Str: "ane", A: 1, B: 1, F: nil},
	{Str: "ene", A: 1, B: 1, F: nil},
	{Str: "hetene", A: 6, B: 1, F: nil},
	{Str: "erte", A: 1, B: 4, F: nil},
	{Str: "en", A: -1, B: 1, F: nil},
	{Str: "heten", A: 9, B: 1, F: nil},
	{Str: "ar", A: -1, B: 1, F: nil},
	{Str: "er", A: -1, B: 1, F: nil},
	{Str: "heter", A: 12, B: 1, F: nil},
	{Str: "s", A: -1, B: 3, F: nil},
	{Str: "as", A: 14, B: 1, F: nil},
	{Str: "es", A: 14, B: 1, F: nil},
	{Str: "edes", A: 16, B: 1, F: nil},
	{Str: "endes", A: 16, B: 1, F: nil},
	{Str: "enes", A: 16, B: 1, F: nil},
	{Str: "hetenes", A: 19, B: 1, F: nil},
	{Str: "ens", A: 14, B: 1, F: nil},
	{Str: "hetens", A: 21, B: 1, F: nil},
	{Str: "ers", A: 14, B: 2, F: nil},
	{Str: "ets", A: 14, B: 1, F: nil},
	{Str: "et", A: -1, B: 1, F: nil},
	{Str: "het", A: 25, B: 1, F: nil},
	{Str: "ert", A: -1, B: 4, F: nil},
	{Str: "ast", A: -1, B: 1, F: nil},
}

var A_2 = []*snowballRuntime.Among{
	{Str: "dt", A: -1, B: -1, F: nil},
	{Str: "vt", A: -1, B: -1, F: nil},
}

var A_3 = []*snowballRuntime.Among{
	{Str: "leg", A: -1, B: 1, F: nil},
	{Str: "eleg", A: 0, B: 1, F: nil},
	{Str: "ig", A: -1, B: 1, F: nil},
	{Str: "eig", A: 2, B: 1, F: nil},
	{Str: "lig", A: 2, B: 1, F: nil},
	{Str: "elig", A: 4, B: 1, F: nil},
	{Str: "els", A: -1, B: 1, F: nil},
	{Str: "lov", A: -1, B: 1, F: nil},
	{Str: "elov", A: 7, B: 1, F: nil},
	{Str: "slov", A: 7, B: 1, F: nil},
	{Str: "hetslov", A: 9, B: 1, F: nil},
}

var G_v = []byte{17, 65, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 48, 2, 142}

var G_s_ending = []byte{119, 125, 148, 1}

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
		among_var = env.FindAmongB(A_0, context)
		switch among_var {
		case 1:
			env.SliceDel()
		}
	case 3:
	lab0:
		for {
			var v_2 = env.Limit - env.Cursor
		lab1:
			for {
				if !env.InGroupingB(G_s_ending, 98, 122) {
					break lab1
				}
				break lab0
			}
			env.Cursor = env.Limit - v_2
		lab2:
			for {
				if !env.EqSB("r") {
					break lab2
				}
			lab3:
				for {
					if !env.EqSB("e") {
						break lab3
					}
					break lab2
				}
				break lab0
			}
			env.Cursor = env.Limit - v_2
			if !env.EqSB("k") {
				return false
			}
			if !env.OutGroupingB(G_v, 97, 248) {
				return false
			}
			break lab0
		}
		env.SliceDel()
	case 4:
		env.SliceFrom("er")
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
	if env.FindAmongB(A_2, context) == 0 {
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
	if env.Cursor < context.i_p1 {
		return false
	}
	var v_1 = env.LimitBackward
	env.LimitBackward = context.i_p1
	env.Ket = env.Cursor
	if env.FindAmongB(A_3, context) == 0 {
		env.LimitBackward = v_1
		return false
	}
	env.Bra = env.Cursor
	env.LimitBackward = v_1
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
	env.Ket = env.Cursor
	if !env.EqSB("'") {
		return false
	}
	env.Bra = env.Cursor
	env.SliceDel()
	env.Cursor = env.LimitBackward
	return true
}
