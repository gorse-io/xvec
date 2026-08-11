//! Generated from indonesian.sbl by Snowball 3.1.1 - https://snowballstem.org/

package indonesian

import (
	snowballRuntime "github.com/gorse-io/xvec/internal/core/snowball"
)

var A_0 = []*snowballRuntime.Among{
	{Str: "kah", A: -1, B: 1, F: nil},
	{Str: "lah", A: -1, B: 1, F: nil},
	{Str: "pun", A: -1, B: 1, F: nil},
}

var A_1 = []*snowballRuntime.Among{
	{Str: "nya", A: -1, B: 1, F: nil},
	{Str: "ku", A: -1, B: 1, F: nil},
	{Str: "mu", A: -1, B: 1, F: nil},
}

var A_2 = []*snowballRuntime.Among{
	{Str: "i", A: -1, B: 2, F: nil},
	{Str: "an", A: -1, B: 1, F: nil},
}

var A_3 = []*snowballRuntime.Among{
	{Str: "di", A: -1, B: 1, F: nil},
	{Str: "ke", A: -1, B: 3, F: nil},
	{Str: "me", A: -1, B: 1, F: nil},
	{Str: "mem", A: 2, B: 5, F: nil},
	{Str: "men", A: 2, B: 2, F: nil},
	{Str: "meng", A: 4, B: 1, F: nil},
	{Str: "pem", A: -1, B: 6, F: nil},
	{Str: "pen", A: -1, B: 4, F: nil},
	{Str: "peng", A: 7, B: 3, F: nil},
	{Str: "ter", A: -1, B: 1, F: nil},
}

var A_4 = []*snowballRuntime.Among{
	{Str: "be", A: -1, B: 2, F: nil},
	{Str: "pe", A: -1, B: 1, F: nil},
}

var G_vowel = []byte{17, 65, 16}

type Context struct {
	i_prefix  int
	i_measure int
}

func r_remove_particle(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	env.Ket = env.Cursor
	if env.FindAmongB(A_0, context) == 0 {
		return false
	}
	env.Bra = env.Cursor
	env.SliceDel()
	context.i_measure -= 1
	return true
}

func r_remove_possessive_pronoun(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	env.Ket = env.Cursor
	if env.FindAmongB(A_1, context) == 0 {
		return false
	}
	env.Bra = env.Cursor
	env.SliceDel()
	context.i_measure -= 1
	return true
}

func r_remove_suffix(env *snowballRuntime.Env, ctx interface{}) bool {
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
	lab0:
		for {
			var v_1 = env.Limit - env.Cursor
		lab1:
			for {
				if context.i_prefix == 3 {
					break lab1
				}
				if context.i_prefix == 2 {
					break lab1
				}
				if !env.EqSB("k") {
					break lab1
				}
				env.Bra = env.Cursor
				break lab0
			}
			env.Cursor = env.Limit - v_1
			if context.i_prefix == 1 {
				return false
			}
			break lab0
		}
	case 2:
		if context.i_prefix > 2 {
			return false
		}
	lab2:
		for {
			if !env.EqSB("s") {
				break lab2
			}
			return false
		}
	}
	env.SliceDel()
	context.i_measure -= 1
	return true
}

func r_remove_first_order_prefix(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	var among_var int32
	env.Bra = env.Cursor
	among_var = env.FindAmong(A_3, context)
	if among_var == 0 {
		return false
	}
	env.Ket = env.Cursor
	switch among_var {
	case 1:
		env.SliceDel()
		context.i_prefix = 1
		context.i_measure -= 1
	case 2:
	lab0:
		for {
			var v_1 = env.Cursor
		lab1:
			for {
				if !env.EqS("y") {
					break lab1
				}
				var v_2 = env.Cursor
				if !env.InGrouping(G_vowel, 97, 117) {
					break lab1
				}
				env.Cursor = v_2
				env.Ket = env.Cursor
				env.SliceFrom("s")
				context.i_prefix = 1
				context.i_measure -= 1
				break lab0
			}
			env.Cursor = v_1
			env.SliceDel()
			context.i_prefix = 1
			context.i_measure -= 1
			break lab0
		}
	case 3:
		env.SliceDel()
		context.i_prefix = 3
		context.i_measure -= 1
	case 4:
	lab2:
		for {
			var v_3 = env.Cursor
		lab3:
			for {
				if !env.EqS("y") {
					break lab3
				}
				var v_4 = env.Cursor
				if !env.InGrouping(G_vowel, 97, 117) {
					break lab3
				}
				env.Cursor = v_4
				env.Ket = env.Cursor
				env.SliceFrom("s")
				context.i_prefix = 3
				context.i_measure -= 1
				break lab2
			}
			env.Cursor = v_3
			env.SliceDel()
			context.i_prefix = 3
			context.i_measure -= 1
			break lab2
		}
	case 5:
		context.i_prefix = 1
		context.i_measure -= 1
	lab4:
		for {
			var v_5 = env.Cursor
		lab5:
			for {
				var v_6 = env.Cursor
				if !env.InGrouping(G_vowel, 97, 117) {
					break lab5
				}
				env.Cursor = v_6
				env.SliceFrom("p")
				break lab4
			}
			env.Cursor = v_5
			env.SliceDel()
			break lab4
		}
	case 6:
		context.i_prefix = 3
		context.i_measure -= 1
	lab6:
		for {
			var v_7 = env.Cursor
		lab7:
			for {
				var v_8 = env.Cursor
				if !env.InGrouping(G_vowel, 97, 117) {
					break lab7
				}
				env.Cursor = v_8
				env.SliceFrom("p")
				break lab6
			}
			env.Cursor = v_7
			env.SliceDel()
			break lab6
		}
	}
	return true
}

func r_remove_second_order_prefix(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	var among_var int32
	env.Bra = env.Cursor
	among_var = env.FindAmong(A_4, context)
	if among_var == 0 {
		return false
	}
	switch among_var {
	case 1:
	lab0:
		for {
			var v_1 = env.Cursor
		lab1:
			for {
				if !env.EqS("r") {
					break lab1
				}
				env.Ket = env.Cursor
				context.i_prefix = 2
				break lab0
			}
			env.Cursor = v_1
		lab2:
			for {
				if !env.EqS("l") {
					break lab2
				}
				env.Ket = env.Cursor
				if !env.EqS("ajar") {
					break lab2
				}
				break lab0
			}
			env.Cursor = v_1
			env.Ket = env.Cursor
			context.i_prefix = 2
			break lab0
		}
	case 2:
	lab3:
		for {
			var v_2 = env.Cursor
		lab4:
			for {
				if !env.EqS("r") {
					break lab4
				}
				env.Ket = env.Cursor
				break lab3
			}
			env.Cursor = v_2
		lab5:
			for {
				if !env.EqS("l") {
					break lab5
				}
				env.Ket = env.Cursor
				if !env.EqS("ajar") {
					break lab5
				}
				break lab3
			}
			env.Cursor = v_2
			env.Ket = env.Cursor
			if !env.OutGrouping(G_vowel, 97, 117) {
				return false
			}
			if !env.EqS("er") {
				return false
			}
			break lab3
		}
		context.i_prefix = 4
	}
	context.i_measure -= 1
	env.SliceDel()
	return true
}

func Stem(env *snowballRuntime.Env) bool {
	var context = &Context{
		i_prefix:  0,
		i_measure: 0,
	}
	_ = context
	context.i_measure = 0
	var v_1 = env.Cursor
lab0:
	for {
	replab1:
		for {
			var v_2 = env.Cursor
		lab2:
			for range [2]struct{}{} {
				if !env.GoOutGrouping(G_vowel, 97, 117) {
					break lab2
				}
				env.NextChar()
				context.i_measure += 1
				continue replab1
			}
			env.Cursor = v_2
			break replab1
		}
		break lab0
	}
	env.Cursor = v_1
	if context.i_measure <= 2 {
		return false
	}
	context.i_prefix = 0
	env.LimitBackward = env.Cursor
	env.Cursor = env.Limit
	var v_3 = env.Limit - env.Cursor
	r_remove_particle(env, context)
	env.Cursor = env.Limit - v_3
	if context.i_measure <= 2 {
		return false
	}
	var v_4 = env.Limit - env.Cursor
	r_remove_possessive_pronoun(env, context)
	env.Cursor = env.Limit - v_4
	env.Cursor = env.LimitBackward
	if context.i_measure <= 2 {
		return false
	}
lab3:
	for {
		var v_5 = env.Cursor
	lab4:
		for {
			var v_6 = env.Cursor
			if !r_remove_first_order_prefix(env, context) {
				break lab4
			}
			var v_7 = env.Cursor
		lab5:
			for {
				var v_8 = env.Cursor
				if context.i_measure <= 2 {
					break lab5
				}
				env.LimitBackward = env.Cursor
				env.Cursor = env.Limit
				if !r_remove_suffix(env, context) {
					break lab5
				}
				env.Cursor = env.LimitBackward
				env.Cursor = v_8
				if context.i_measure <= 2 {
					break lab5
				}
				if !r_remove_second_order_prefix(env, context) {
					break lab5
				}
				break lab5
			}
			env.Cursor = v_7
			env.Cursor = v_6
			break lab3
		}
		env.Cursor = v_5
		var v_9 = env.Cursor
		r_remove_second_order_prefix(env, context)
		env.Cursor = v_9
		var v_10 = env.Cursor
	lab6:
		for {
			if context.i_measure <= 2 {
				break lab6
			}
			env.LimitBackward = env.Cursor
			env.Cursor = env.Limit
			if !r_remove_suffix(env, context) {
				break lab6
			}
			env.Cursor = env.LimitBackward
			break lab6
		}
		env.Cursor = v_10
		break lab3
	}
	return true
}
