//! Generated from german.sbl by Snowball 3.1.1 - https://snowballstem.org/

package german

import (
	snowballRuntime "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball"
)

var A_0 = []*snowballRuntime.Among{
	{Str: "", A: -1, B: 5, F: nil},
	{Str: "ae", A: 0, B: 2, F: nil},
	{Str: "oe", A: 0, B: 3, F: nil},
	{Str: "qu", A: 0, B: -1, F: nil},
	{Str: "ue", A: 0, B: 4, F: nil},
	{Str: "ß", A: 0, B: 1, F: nil},
}

var A_1 = []*snowballRuntime.Among{
	{Str: "", A: -1, B: 5, F: nil},
	{Str: "U", A: 0, B: 2, F: nil},
	{Str: "Y", A: 0, B: 1, F: nil},
	{Str: "ä", A: 0, B: 3, F: nil},
	{Str: "ö", A: 0, B: 4, F: nil},
	{Str: "ü", A: 0, B: 2, F: nil},
}

var A_2 = []*snowballRuntime.Among{
	{Str: "e", A: -1, B: 3, F: nil},
	{Str: "em", A: -1, B: 1, F: nil},
	{Str: "en", A: -1, B: 3, F: nil},
	{Str: "erinnen", A: 2, B: 2, F: nil},
	{Str: "erin", A: -1, B: 2, F: nil},
	{Str: "ln", A: -1, B: 5, F: nil},
	{Str: "ern", A: -1, B: 2, F: nil},
	{Str: "er", A: -1, B: 2, F: nil},
	{Str: "s", A: -1, B: 4, F: nil},
	{Str: "es", A: 8, B: 3, F: nil},
	{Str: "lns", A: 8, B: 5, F: nil},
}

var A_3 = []*snowballRuntime.Among{
	{Str: "tick", A: -1, B: -1, F: nil},
	{Str: "plan", A: -1, B: -1, F: nil},
	{Str: "geordn", A: -1, B: -1, F: nil},
	{Str: "intern", A: -1, B: -1, F: nil},
	{Str: "tr", A: -1, B: -1, F: nil},
}

var A_4 = []*snowballRuntime.Among{
	{Str: "en", A: -1, B: 1, F: nil},
	{Str: "er", A: -1, B: 1, F: nil},
	{Str: "et", A: -1, B: 3, F: nil},
	{Str: "st", A: -1, B: 2, F: nil},
	{Str: "est", A: 3, B: 1, F: nil},
}

var A_5 = []*snowballRuntime.Among{
	{Str: "ig", A: -1, B: 1, F: nil},
	{Str: "lich", A: -1, B: 1, F: nil},
}

var A_6 = []*snowballRuntime.Among{
	{Str: "end", A: -1, B: 1, F: nil},
	{Str: "ig", A: -1, B: 2, F: nil},
	{Str: "ung", A: -1, B: 1, F: nil},
	{Str: "lich", A: -1, B: 3, F: nil},
	{Str: "isch", A: -1, B: 2, F: nil},
	{Str: "ik", A: -1, B: 2, F: nil},
	{Str: "heit", A: -1, B: 3, F: nil},
	{Str: "keit", A: -1, B: 4, F: nil},
}

var A_7 = []*snowballRuntime.Among{
	{Str: "'", A: -1, B: 1, F: nil},
	{Str: "'sch", A: -1, B: 1, F: nil},
	{Str: "'s", A: -1, B: 1, F: nil},
}

var G_v = []byte{17, 65, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 32, 8}

var G_et_ending = []byte{1, 128, 198, 227, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128}

var G_s_ending = []byte{117, 30, 5}

var G_st_ending = []byte{117, 30, 4}

type Context struct {
	i_p2 int
	i_p1 int
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
		golab2:
			for {
				var v_3 = env.Cursor
			lab3:
				for {
					if !env.InGrouping(G_v, 97, 252) {
						break lab3
					}
					env.Bra = env.Cursor
				lab4:
					for {
						var v_4 = env.Cursor
					lab5:
						for {
							if !env.EqS("u") {
								break lab5
							}
							env.Ket = env.Cursor
							if !env.InGrouping(G_v, 97, 252) {
								break lab5
							}
							env.SliceFrom("U")
							break lab4
						}
						env.Cursor = v_4
						if !env.EqS("y") {
							break lab3
						}
						env.Ket = env.Cursor
						if !env.InGrouping(G_v, 97, 252) {
							break lab3
						}
						env.SliceFrom("Y")
						break lab4
					}
					env.Cursor = v_3
					break golab2
				}
				env.Cursor = v_3
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
replab6:
	for {
		var v_5 = env.Cursor
	lab7:
		for range [2]struct{}{} {
			env.Bra = env.Cursor
			among_var = env.FindAmong(A_0, context)
			env.Ket = env.Cursor
			switch among_var {
			case 1:
				env.SliceFrom("ss")
			case 2:
				env.SliceFrom("ä")
			case 3:
				env.SliceFrom("ö")
			case 4:
				env.SliceFrom("ü")
			case 5:
				if env.Cursor >= env.Limit {
					break lab7
				}
				env.NextChar()
			}
			continue replab6
		}
		env.Cursor = v_5
		break replab6
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
	if !env.GoOutGrouping(G_v, 97, 252) {
		return false
	}
	env.NextChar()
	if !env.GoInGrouping(G_v, 97, 252) {
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
	if !env.GoOutGrouping(G_v, 97, 252) {
		return false
	}
	env.NextChar()
	if !env.GoInGrouping(G_v, 97, 252) {
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
				env.SliceFrom("u")
			case 3:
				env.SliceFrom("a")
			case 4:
				env.SliceFrom("o")
			case 5:
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

func r_standard_suffix(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	var among_var int32
	var v_1 = env.Limit - env.Cursor
lab0:
	for {
		env.Ket = env.Cursor
		among_var = env.FindAmongB(A_2, context)
		if among_var == 0 {
			break lab0
		}
		env.Bra = env.Cursor
		if !r_R1(env, context) {
			break lab0
		}
		switch among_var {
		case 1:
		lab1:
			for {
				if !env.EqSB("syst") {
					break lab1
				}
				break lab0
			}
			env.SliceDel()
		case 2:
			env.SliceDel()
		case 3:
			env.SliceDel()
			var v_2 = env.Limit - env.Cursor
		lab2:
			for {
				env.Ket = env.Cursor
				if !env.EqSB("s") {
					env.Cursor = env.Limit - v_2
					break lab2
				}
				env.Bra = env.Cursor
				if !env.EqSB("nis") {
					env.Cursor = env.Limit - v_2
					break lab2
				}
				env.SliceDel()
				break lab2
			}
		case 4:
			if !env.InGroupingB(G_s_ending, 98, 116) {
				break lab0
			}
			env.SliceDel()
		case 5:
			env.SliceFrom("l")
		}
		break lab0
	}
	env.Cursor = env.Limit - v_1
	var v_3 = env.Limit - env.Cursor
lab3:
	for {
		env.Ket = env.Cursor
		among_var = env.FindAmongB(A_4, context)
		if among_var == 0 {
			break lab3
		}
		env.Bra = env.Cursor
		if !r_R1(env, context) {
			break lab3
		}
		switch among_var {
		case 1:
			env.SliceDel()
		case 2:
			if !env.InGroupingB(G_st_ending, 98, 116) {
				break lab3
			}
			if !env.HopBack(3) {
				break lab3
			}
			env.SliceDel()
		case 3:
			var v_4 = env.Limit - env.Cursor
			if !env.InGroupingB(G_et_ending, 85, 228) {
				break lab3
			}
			env.Cursor = env.Limit - v_4
			var v_5 = env.Limit - env.Cursor
		lab4:
			for {
				if env.FindAmongB(A_3, context) == 0 {
					break lab4
				}
				break lab3
			}
			env.Cursor = env.Limit - v_5
			env.SliceDel()
		}
		break lab3
	}
	env.Cursor = env.Limit - v_3
	var v_6 = env.Limit - env.Cursor
lab5:
	for {
		env.Ket = env.Cursor
		among_var = env.FindAmongB(A_6, context)
		if among_var == 0 {
			break lab5
		}
		env.Bra = env.Cursor
		if !r_R2(env, context) {
			break lab5
		}
		switch among_var {
		case 1:
			env.SliceDel()
			var v_7 = env.Limit - env.Cursor
		lab6:
			for {
				env.Ket = env.Cursor
				if !env.EqSB("ig") {
					env.Cursor = env.Limit - v_7
					break lab6
				}
				env.Bra = env.Cursor
			lab7:
				for {
					if !env.EqSB("e") {
						break lab7
					}
					env.Cursor = env.Limit - v_7
					break lab6
				}
				if !r_R2(env, context) {
					env.Cursor = env.Limit - v_7
					break lab6
				}
				env.SliceDel()
				break lab6
			}
		case 2:
		lab8:
			for {
				if !env.EqSB("e") {
					break lab8
				}
				break lab5
			}
			env.SliceDel()
		case 3:
			env.SliceDel()
			var v_8 = env.Limit - env.Cursor
		lab9:
			for {
				env.Ket = env.Cursor
			lab10:
				for {
				lab11:
					for {
						if !env.EqSB("er") {
							break lab11
						}
						break lab10
					}
					if !env.EqSB("en") {
						env.Cursor = env.Limit - v_8
						break lab9
					}
					break lab10
				}
				env.Bra = env.Cursor
				if !r_R1(env, context) {
					env.Cursor = env.Limit - v_8
					break lab9
				}
				env.SliceDel()
				break lab9
			}
		case 4:
			env.SliceDel()
			var v_9 = env.Limit - env.Cursor
		lab12:
			for {
				env.Ket = env.Cursor
				if env.FindAmongB(A_5, context) == 0 {
					env.Cursor = env.Limit - v_9
					break lab12
				}
				env.Bra = env.Cursor
				if !r_R2(env, context) {
					env.Cursor = env.Limit - v_9
					break lab12
				}
				env.SliceDel()
				break lab12
			}
		}
		break lab5
	}
	env.Cursor = env.Limit - v_6
	var v_10 = env.Limit - env.Cursor
lab13:
	for {
		env.Ket = env.Cursor
		if env.FindAmongB(A_7, context) == 0 {
			break lab13
		}
		env.Bra = env.Cursor
		if env.Cursor <= env.LimitBackward {
			break lab13
		}
		env.PrevChar()
		if env.Cursor <= env.LimitBackward {
			break lab13
		}
		env.SliceDel()
		break lab13
	}
	env.Cursor = env.Limit - v_10
	return true
}

func Stem(env *snowballRuntime.Env) bool {
	var context = &Context{
		i_p2: 0,
		i_p1: 0,
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
