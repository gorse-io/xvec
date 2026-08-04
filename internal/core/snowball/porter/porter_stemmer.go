//! Generated from porter.sbl by Snowball 3.1.1 - https://snowballstem.org/

package porter

import (
	snowballRuntime "github.com/gorse-io/zvec/internal/core/snowball"
)

var A_0 = []*snowballRuntime.Among{
	{Str: "s", A: -1, B: 3, F: nil},
	{Str: "ies", A: 0, B: 2, F: nil},
	{Str: "sses", A: 0, B: 1, F: nil},
	{Str: "ss", A: 0, B: -1, F: nil},
}

var A_1 = []*snowballRuntime.Among{
	{Str: "", A: -1, B: 3, F: nil},
	{Str: "bb", A: 0, B: 2, F: nil},
	{Str: "dd", A: 0, B: 2, F: nil},
	{Str: "ff", A: 0, B: 2, F: nil},
	{Str: "gg", A: 0, B: 2, F: nil},
	{Str: "bl", A: 0, B: 1, F: nil},
	{Str: "mm", A: 0, B: 2, F: nil},
	{Str: "nn", A: 0, B: 2, F: nil},
	{Str: "pp", A: 0, B: 2, F: nil},
	{Str: "rr", A: 0, B: 2, F: nil},
	{Str: "at", A: 0, B: 1, F: nil},
	{Str: "tt", A: 0, B: 2, F: nil},
	{Str: "iz", A: 0, B: 1, F: nil},
}

var A_2 = []*snowballRuntime.Among{
	{Str: "ed", A: -1, B: 2, F: nil},
	{Str: "eed", A: 0, B: 1, F: nil},
	{Str: "ing", A: -1, B: 2, F: nil},
}

var A_3 = []*snowballRuntime.Among{
	{Str: "anci", A: -1, B: 3, F: nil},
	{Str: "enci", A: -1, B: 2, F: nil},
	{Str: "abli", A: -1, B: 4, F: nil},
	{Str: "eli", A: -1, B: 6, F: nil},
	{Str: "alli", A: -1, B: 9, F: nil},
	{Str: "ousli", A: -1, B: 11, F: nil},
	{Str: "entli", A: -1, B: 5, F: nil},
	{Str: "aliti", A: -1, B: 9, F: nil},
	{Str: "biliti", A: -1, B: 13, F: nil},
	{Str: "iviti", A: -1, B: 12, F: nil},
	{Str: "tional", A: -1, B: 1, F: nil},
	{Str: "ational", A: 10, B: 8, F: nil},
	{Str: "alism", A: -1, B: 9, F: nil},
	{Str: "ation", A: -1, B: 8, F: nil},
	{Str: "ization", A: 13, B: 7, F: nil},
	{Str: "izer", A: -1, B: 7, F: nil},
	{Str: "ator", A: -1, B: 8, F: nil},
	{Str: "iveness", A: -1, B: 12, F: nil},
	{Str: "fulness", A: -1, B: 10, F: nil},
	{Str: "ousness", A: -1, B: 11, F: nil},
}

var A_4 = []*snowballRuntime.Among{
	{Str: "icate", A: -1, B: 2, F: nil},
	{Str: "ative", A: -1, B: 3, F: nil},
	{Str: "alize", A: -1, B: 1, F: nil},
	{Str: "iciti", A: -1, B: 2, F: nil},
	{Str: "ical", A: -1, B: 2, F: nil},
	{Str: "ful", A: -1, B: 3, F: nil},
	{Str: "ness", A: -1, B: 3, F: nil},
}

var A_5 = []*snowballRuntime.Among{
	{Str: "ic", A: -1, B: 1, F: nil},
	{Str: "ance", A: -1, B: 1, F: nil},
	{Str: "ence", A: -1, B: 1, F: nil},
	{Str: "able", A: -1, B: 1, F: nil},
	{Str: "ible", A: -1, B: 1, F: nil},
	{Str: "ate", A: -1, B: 1, F: nil},
	{Str: "ive", A: -1, B: 1, F: nil},
	{Str: "ize", A: -1, B: 1, F: nil},
	{Str: "iti", A: -1, B: 1, F: nil},
	{Str: "al", A: -1, B: 1, F: nil},
	{Str: "ism", A: -1, B: 1, F: nil},
	{Str: "ion", A: -1, B: 2, F: nil},
	{Str: "er", A: -1, B: 1, F: nil},
	{Str: "ous", A: -1, B: 1, F: nil},
	{Str: "ant", A: -1, B: 1, F: nil},
	{Str: "ent", A: -1, B: 1, F: nil},
	{Str: "ment", A: 15, B: 1, F: nil},
	{Str: "ement", A: 16, B: 1, F: nil},
	{Str: "ou", A: -1, B: 1, F: nil},
}

var G_v = []byte{17, 65, 16, 1}

var G_v_WXY = []byte{1, 17, 65, 208, 1}

type Context struct {
	i_p2 int
	i_p1 int
}

func r_shortv(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	if !env.OutGroupingB(G_v_WXY, 89, 121) {
		return false
	}
	if !env.InGroupingB(G_v, 97, 121) {
		return false
	}
	return env.OutGroupingB(G_v, 97, 121)
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

func r_Step_1a(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	var among_var int32
	env.Ket = env.Cursor
	among_var = env.FindAmongB(A_0, context)
	if among_var == 0 {
		return false
	}
	env.Bra = env.Cursor
	switch among_var {
	case 1:
		env.SliceFrom("ss")
	case 2:
		env.SliceFrom("i")
	case 3:
		env.SliceDel()
	}
	return true
}

func r_Step_1b(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	var among_var int32
	env.Ket = env.Cursor
	among_var = env.FindAmongB(A_2, context)
	if among_var == 0 {
		return false
	}
	env.Bra = env.Cursor
	switch among_var {
	case 1:
		if !r_R1(env, context) {
			return false
		}
		env.SliceFrom("ee")
	case 2:
		var v_1 = env.Limit - env.Cursor
		if !env.GoOutGroupingB(G_v, 97, 121) {
			return false
		}
		env.PrevChar()
		env.Cursor = env.Limit - v_1
		env.SliceDel()
		var v_2 = env.Limit - env.Cursor
		among_var = env.FindAmongB(A_1, context)
		env.Cursor = env.Limit - v_2
		switch among_var {
		case 1:
			{
				var c = env.Cursor
				bra, ket := env.Cursor, env.Cursor
				env.Insert(bra, ket, "e")
				env.Cursor = c
			}
		case 2:
			env.Ket = env.Cursor
			if env.Cursor <= env.LimitBackward {
				return false
			}
			env.PrevChar()
			env.Bra = env.Cursor
			env.SliceDel()
		case 3:
			if env.Cursor != context.i_p1 {
				return false
			}
			var v_3 = env.Limit - env.Cursor
			if !r_shortv(env, context) {
				return false
			}
			env.Cursor = env.Limit - v_3
			{
				var c = env.Cursor
				bra, ket := env.Cursor, env.Cursor
				env.Insert(bra, ket, "e")
				env.Cursor = c
			}
		}
	}
	return true
}

func r_Step_1c(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	env.Ket = env.Cursor
lab0:
	for {
	lab1:
		for {
			if !env.EqSB("y") {
				break lab1
			}
			break lab0
		}
		if !env.EqSB("Y") {
			return false
		}
		break lab0
	}
	env.Bra = env.Cursor
	if !env.GoOutGroupingB(G_v, 97, 121) {
		return false
	}
	env.PrevChar()
	env.SliceFrom("i")
	return true
}

func r_Step_2(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	var among_var int32
	env.Ket = env.Cursor
	among_var = env.FindAmongB(A_3, context)
	if among_var == 0 {
		return false
	}
	env.Bra = env.Cursor
	if !r_R1(env, context) {
		return false
	}
	switch among_var {
	case 1:
		env.SliceFrom("tion")
	case 2:
		env.SliceFrom("ence")
	case 3:
		env.SliceFrom("ance")
	case 4:
		env.SliceFrom("able")
	case 5:
		env.SliceFrom("ent")
	case 6:
		env.SliceFrom("e")
	case 7:
		env.SliceFrom("ize")
	case 8:
		env.SliceFrom("ate")
	case 9:
		env.SliceFrom("al")
	case 10:
		env.SliceFrom("ful")
	case 11:
		env.SliceFrom("ous")
	case 12:
		env.SliceFrom("ive")
	case 13:
		env.SliceFrom("ble")
	}
	return true
}

func r_Step_3(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	var among_var int32
	env.Ket = env.Cursor
	among_var = env.FindAmongB(A_4, context)
	if among_var == 0 {
		return false
	}
	env.Bra = env.Cursor
	if !r_R1(env, context) {
		return false
	}
	switch among_var {
	case 1:
		env.SliceFrom("al")
	case 2:
		env.SliceFrom("ic")
	case 3:
		env.SliceDel()
	}
	return true
}

func r_Step_4(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	var among_var int32
	env.Ket = env.Cursor
	among_var = env.FindAmongB(A_5, context)
	if among_var == 0 {
		return false
	}
	env.Bra = env.Cursor
	if !r_R2(env, context) {
		return false
	}
	switch among_var {
	case 1:
		env.SliceDel()
	case 2:
	lab0:
		for {
		lab1:
			for {
				if !env.EqSB("s") {
					break lab1
				}
				break lab0
			}
			if !env.EqSB("t") {
				return false
			}
			break lab0
		}
		env.SliceDel()
	}
	return true
}

func r_Step_5a(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	env.Ket = env.Cursor
	if !env.EqSB("e") {
		return false
	}
	env.Bra = env.Cursor
lab0:
	for {
	lab1:
		for {
			if !r_R2(env, context) {
				break lab1
			}
			break lab0
		}
		if !r_R1(env, context) {
			return false
		}
		var v_1 = env.Limit - env.Cursor
	lab2:
		for {
			if !r_shortv(env, context) {
				break lab2
			}
			return false
		}
		env.Cursor = env.Limit - v_1
		break lab0
	}
	env.SliceDel()
	return true
}

func r_Step_5b(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	env.Ket = env.Cursor
	if !env.EqSB("l") {
		return false
	}
	env.Bra = env.Cursor
	if !r_R2(env, context) {
		return false
	}
	if !env.EqSB("l") {
		return false
	}
	env.SliceDel()
	return true
}

func Stem(env *snowballRuntime.Env) bool {
	var context = &Context{
		i_p2: 0,
		i_p1: 0,
	}
	_ = context
	var b_Y_found bool
	b_Y_found = false
	var v_1 = env.Cursor
lab0:
	for {
		env.Bra = env.Cursor
		if !env.EqS("y") {
			break lab0
		}
		env.Ket = env.Cursor
		env.SliceFrom("Y")
		b_Y_found = true
		break lab0
	}
	env.Cursor = v_1
	var v_2 = env.Cursor
lab1:
	for {
	replab2:
		for {
			var v_3 = env.Cursor
		lab3:
			for range [2]struct{}{} {
			golab4:
				for {
					var v_4 = env.Cursor
				lab5:
					for {
						if !env.InGrouping(G_v, 97, 121) {
							break lab5
						}
						env.Bra = env.Cursor
						if !env.EqS("y") {
							break lab5
						}
						env.Ket = env.Cursor
						env.Cursor = v_4
						break golab4
					}
					env.Cursor = v_4
					if env.Cursor >= env.Limit {
						break lab3
					}
					env.NextChar()
				}
				env.SliceFrom("Y")
				b_Y_found = true
				continue replab2
			}
			env.Cursor = v_3
			break replab2
		}
		break lab1
	}
	env.Cursor = v_2
	context.i_p1 = env.Limit
	context.i_p2 = env.Limit
	var v_5 = env.Cursor
lab6:
	for {
		if !env.GoOutGrouping(G_v, 97, 121) {
			break lab6
		}
		env.NextChar()
		if !env.GoInGrouping(G_v, 97, 121) {
			break lab6
		}
		env.NextChar()
		context.i_p1 = env.Cursor
		if !env.GoOutGrouping(G_v, 97, 121) {
			break lab6
		}
		env.NextChar()
		if !env.GoInGrouping(G_v, 97, 121) {
			break lab6
		}
		env.NextChar()
		context.i_p2 = env.Cursor
		break lab6
	}
	env.Cursor = v_5
	env.LimitBackward = env.Cursor
	env.Cursor = env.Limit
	var v_6 = env.Limit - env.Cursor
	r_Step_1a(env, context)
	env.Cursor = env.Limit - v_6
	var v_7 = env.Limit - env.Cursor
	r_Step_1b(env, context)
	env.Cursor = env.Limit - v_7
	var v_8 = env.Limit - env.Cursor
	r_Step_1c(env, context)
	env.Cursor = env.Limit - v_8
	var v_9 = env.Limit - env.Cursor
	r_Step_2(env, context)
	env.Cursor = env.Limit - v_9
	var v_10 = env.Limit - env.Cursor
	r_Step_3(env, context)
	env.Cursor = env.Limit - v_10
	var v_11 = env.Limit - env.Cursor
	r_Step_4(env, context)
	env.Cursor = env.Limit - v_11
	var v_12 = env.Limit - env.Cursor
	r_Step_5a(env, context)
	env.Cursor = env.Limit - v_12
	var v_13 = env.Limit - env.Cursor
	r_Step_5b(env, context)
	env.Cursor = env.Limit - v_13
	env.Cursor = env.LimitBackward
	var v_14 = env.Cursor
lab7:
	for {
		if !b_Y_found {
			break lab7
		}
	replab8:
		for {
			var v_15 = env.Cursor
		lab9:
			for range [2]struct{}{} {
			golab10:
				for {
					var v_16 = env.Cursor
				lab11:
					for {
						env.Bra = env.Cursor
						if !env.EqS("Y") {
							break lab11
						}
						env.Ket = env.Cursor
						env.Cursor = v_16
						break golab10
					}
					env.Cursor = v_16
					if env.Cursor >= env.Limit {
						break lab9
					}
					env.NextChar()
				}
				env.SliceFrom("y")
				continue replab8
			}
			env.Cursor = v_15
			break replab8
		}
		break lab7
	}
	env.Cursor = v_14
	return true
}
