//! Generated from dutch_porter.sbl by Snowball 3.1.1 - https://snowballstem.org/

package dutch_porter

import (
	snowballRuntime "github.com/gorse-io/zvec/internal/core/snowball"
)

var A_0 = []*snowballRuntime.Among{
	{Str: "", A: -1, B: 6, F: nil},
	{Str: "á", A: 0, B: 1, F: nil},
	{Str: "ä", A: 0, B: 1, F: nil},
	{Str: "é", A: 0, B: 2, F: nil},
	{Str: "ë", A: 0, B: 2, F: nil},
	{Str: "í", A: 0, B: 3, F: nil},
	{Str: "ï", A: 0, B: 3, F: nil},
	{Str: "ó", A: 0, B: 4, F: nil},
	{Str: "ö", A: 0, B: 4, F: nil},
	{Str: "ú", A: 0, B: 5, F: nil},
	{Str: "ü", A: 0, B: 5, F: nil},
}

var A_1 = []*snowballRuntime.Among{
	{Str: "", A: -1, B: 3, F: nil},
	{Str: "I", A: 0, B: 2, F: nil},
	{Str: "Y", A: 0, B: 1, F: nil},
}

var A_2 = []*snowballRuntime.Among{
	{Str: "dd", A: -1, B: -1, F: nil},
	{Str: "kk", A: -1, B: -1, F: nil},
	{Str: "tt", A: -1, B: -1, F: nil},
}

var A_3 = []*snowballRuntime.Among{
	{Str: "ene", A: -1, B: 2, F: nil},
	{Str: "se", A: -1, B: 3, F: nil},
	{Str: "en", A: -1, B: 2, F: nil},
	{Str: "heden", A: 2, B: 1, F: nil},
	{Str: "s", A: -1, B: 3, F: nil},
}

var A_4 = []*snowballRuntime.Among{
	{Str: "end", A: -1, B: 1, F: nil},
	{Str: "ig", A: -1, B: 2, F: nil},
	{Str: "ing", A: -1, B: 1, F: nil},
	{Str: "lijk", A: -1, B: 3, F: nil},
	{Str: "baar", A: -1, B: 4, F: nil},
	{Str: "bar", A: -1, B: 5, F: nil},
}

var A_5 = []*snowballRuntime.Among{
	{Str: "aa", A: -1, B: -1, F: nil},
	{Str: "ee", A: -1, B: -1, F: nil},
	{Str: "oo", A: -1, B: -1, F: nil},
	{Str: "uu", A: -1, B: -1, F: nil},
}

var G_v = []byte{17, 65, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128}

var G_v_I = []byte{1, 0, 0, 17, 65, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128}

var G_v_j = []byte{17, 67, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128}

type Context struct {
	i_p2      int
	i_p1      int
	b_e_found bool
}

func r_prelude(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	var among_var int32
	var v_1 = env.Cursor
replab0:
	for {
		var v_2 = env.Cursor
	lab1:
		for range [2]struct{}{} {
			env.Bra = env.Cursor
			among_var = env.FindAmong(A_0, context)
			env.Ket = env.Cursor
			switch among_var {
			case 1:
				env.SliceFrom("a")
			case 2:
				env.SliceFrom("e")
			case 3:
				env.SliceFrom("i")
			case 4:
				env.SliceFrom("o")
			case 5:
				env.SliceFrom("u")
			case 6:
				if env.Cursor >= env.Limit {
					break lab1
				}
				env.NextChar()
			}
			continue replab0
		}
		env.Cursor = v_2
		break replab0
	}
	env.Cursor = v_1
	var v_3 = env.Cursor
lab2:
	for {
		env.Bra = env.Cursor
		if !env.EqS("y") {
			env.Cursor = v_3
			break lab2
		}
		env.Ket = env.Cursor
		env.SliceFrom("Y")
		break lab2
	}
replab3:
	for {
		var v_4 = env.Cursor
	lab4:
		for range [2]struct{}{} {
			if !env.GoOutGrouping(G_v, 97, 232) {
				break lab4
			}
			env.NextChar()
			var v_5 = env.Cursor
		lab5:
			for {
				env.Bra = env.Cursor
			lab6:
				for {
					var v_6 = env.Cursor
				lab7:
					for {
						if !env.EqS("i") {
							break lab7
						}
						env.Ket = env.Cursor
						var v_7 = env.Cursor
					lab8:
						for {
							if !env.InGrouping(G_v, 97, 232) {
								break lab8
							}
							env.SliceFrom("I")
							break lab8
						}
						env.Cursor = v_7
						break lab6
					}
					env.Cursor = v_6
					if !env.EqS("y") {
						env.Cursor = v_5
						break lab5
					}
					env.Ket = env.Cursor
					env.SliceFrom("Y")
					break lab6
				}
				break lab5
			}
			continue replab3
		}
		env.Cursor = v_4
		break replab3
	}
	return true
}

func r_mark_regions(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	var i_x int
	context.i_p1 = env.Limit
	context.i_p2 = env.Limit
	var v_1 = env.Cursor
	if !env.Hop(3) {
		return false
	}
	i_x = env.Cursor
	env.Cursor = v_1
	if !env.GoOutGrouping(G_v, 97, 232) {
		return false
	}
	env.NextChar()
	if !env.GoInGrouping(G_v, 97, 232) {
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
	if !env.GoOutGrouping(G_v, 97, 232) {
		return false
	}
	env.NextChar()
	if !env.GoInGrouping(G_v, 97, 232) {
		return false
	}
	env.NextChar()
	context.i_p2 = env.Cursor
	return true
}

func r_postlude(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	var among_var int32
replab0:
	for {
		var v_1 = env.Cursor
	lab1:
		for range [2]struct{}{} {
			env.Bra = env.Cursor
			among_var = env.FindAmong(A_1, context)
			env.Ket = env.Cursor
			switch among_var {
			case 1:
				env.SliceFrom("y")
			case 2:
				env.SliceFrom("i")
			case 3:
				if env.Cursor >= env.Limit {
					break lab1
				}
				env.NextChar()
			}
			continue replab0
		}
		env.Cursor = v_1
		break replab0
	}
	return true
}

func r_R1(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	return context.i_p1 <= env.Cursor
}

func r_R2(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	return context.i_p2 <= env.Cursor
}

func r_undouble(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	var v_1 = env.Limit - env.Cursor
	if env.FindAmongB(A_2, context) == 0 {
		return false
	}
	env.Cursor = env.Limit - v_1
	env.Ket = env.Cursor
	if env.Cursor <= env.LimitBackward {
		return false
	}
	env.PrevChar()
	env.Bra = env.Cursor
	env.SliceDel()
	return true
}

func r_e_ending(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	context.b_e_found = false
	env.Ket = env.Cursor
	if !env.EqSB("e") {
		return false
	}
	env.Bra = env.Cursor
	if !r_R1(env, context) {
		return false
	}
	var v_1 = env.Limit - env.Cursor
	if !env.OutGroupingB(G_v, 97, 232) {
		return false
	}
	env.Cursor = env.Limit - v_1
	env.SliceDel()
	context.b_e_found = true
	return r_undouble(env, context)
}

func r_en_ending(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	if !r_R1(env, context) {
		return false
	}
	var v_1 = env.Limit - env.Cursor
	if !env.OutGroupingB(G_v, 97, 232) {
		return false
	}
	env.Cursor = env.Limit - v_1
lab0:
	for {
		if !env.EqSB("gem") {
			break lab0
		}
		return false
	}
	env.SliceDel()
	return r_undouble(env, context)
}

func r_standard_suffix(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	var among_var int32
	var v_1 = env.Limit - env.Cursor
lab0:
	for {
		env.Ket = env.Cursor
		among_var = env.FindAmongB(A_3, context)
		if among_var == 0 {
			break lab0
		}
		env.Bra = env.Cursor
		switch among_var {
		case 1:
			if !r_R1(env, context) {
				break lab0
			}
			env.SliceFrom("heid")
		case 2:
			if !r_en_ending(env, context) {
				break lab0
			}
		case 3:
			if !r_R1(env, context) {
				break lab0
			}
			if !env.OutGroupingB(G_v_j, 97, 232) {
				break lab0
			}
			env.SliceDel()
		}
		break lab0
	}
	env.Cursor = env.Limit - v_1
	var v_2 = env.Limit - env.Cursor
	r_e_ending(env, context)
	env.Cursor = env.Limit - v_2
	var v_3 = env.Limit - env.Cursor
lab1:
	for {
		env.Ket = env.Cursor
		if !env.EqSB("heid") {
			break lab1
		}
		env.Bra = env.Cursor
		if !r_R2(env, context) {
			break lab1
		}
	lab2:
		for {
			if !env.EqSB("c") {
				break lab2
			}
			break lab1
		}
		env.SliceDel()
		env.Ket = env.Cursor
		if !env.EqSB("en") {
			break lab1
		}
		env.Bra = env.Cursor
		if !r_en_ending(env, context) {
			break lab1
		}
		break lab1
	}
	env.Cursor = env.Limit - v_3
	var v_4 = env.Limit - env.Cursor
lab3:
	for {
		env.Ket = env.Cursor
		among_var = env.FindAmongB(A_4, context)
		if among_var == 0 {
			break lab3
		}
		env.Bra = env.Cursor
		switch among_var {
		case 1:
			if !r_R2(env, context) {
				break lab3
			}
			env.SliceDel()
		lab4:
			for {
				var v_5 = env.Limit - env.Cursor
			lab5:
				for {
					env.Ket = env.Cursor
					if !env.EqSB("ig") {
						break lab5
					}
					env.Bra = env.Cursor
					if !r_R2(env, context) {
						break lab5
					}
				lab6:
					for {
						if !env.EqSB("e") {
							break lab6
						}
						break lab5
					}
					env.SliceDel()
					break lab4
				}
				env.Cursor = env.Limit - v_5
				if !r_undouble(env, context) {
					break lab3
				}
				break lab4
			}
		case 2:
			if !r_R2(env, context) {
				break lab3
			}
		lab7:
			for {
				if !env.EqSB("e") {
					break lab7
				}
				break lab3
			}
			env.SliceDel()
		case 3:
			if !r_R2(env, context) {
				break lab3
			}
			env.SliceDel()
			if !r_e_ending(env, context) {
				break lab3
			}
		case 4:
			if !r_R2(env, context) {
				break lab3
			}
			env.SliceDel()
		case 5:
			if !r_R2(env, context) {
				break lab3
			}
			if !context.b_e_found {
				break lab3
			}
			env.SliceDel()
		}
		break lab3
	}
	env.Cursor = env.Limit - v_4
	var v_6 = env.Limit - env.Cursor
lab8:
	for {
		if !env.OutGroupingB(G_v_I, 73, 232) {
			break lab8
		}
		var v_7 = env.Limit - env.Cursor
		if env.FindAmongB(A_5, context) == 0 {
			break lab8
		}
		if !env.OutGroupingB(G_v, 97, 232) {
			break lab8
		}
		env.Cursor = env.Limit - v_7
		env.Ket = env.Cursor
		if env.Cursor <= env.LimitBackward {
			break lab8
		}
		env.PrevChar()
		env.Bra = env.Cursor
		env.SliceDel()
		break lab8
	}
	env.Cursor = env.Limit - v_6
	return true
}

func Stem(env *snowballRuntime.Env) bool {
	var context = &Context{
		i_p2:      0,
		i_p1:      0,
		b_e_found: false,
	}
	_ = context
	var v_1 = env.Cursor
	r_prelude(env, context)
	env.Cursor = v_1
	var v_2 = env.Cursor
	r_mark_regions(env, context)
	env.Cursor = v_2
	env.LimitBackward = env.Cursor
	env.Cursor = env.Limit
	r_standard_suffix(env, context)
	env.Cursor = env.LimitBackward
	var v_3 = env.Cursor
	r_postlude(env, context)
	env.Cursor = v_3
	return true
}
