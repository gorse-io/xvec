//! Generated from polish.sbl by Snowball 3.1.1 - https://snowballstem.org/

package polish

import (
	snowballRuntime "github.com/gorse-io/xvec/internal/core/snowball"
)

var A_0 = []*snowballRuntime.Among{
	{Str: "byście", A: -1, B: 1, F: nil},
	{Str: "bym", A: -1, B: 1, F: nil},
	{Str: "by", A: -1, B: 1, F: nil},
	{Str: "byśmy", A: -1, B: 1, F: nil},
	{Str: "byś", A: -1, B: 1, F: nil},
}

var A_1 = []*snowballRuntime.Among{
	{Str: "ąc", A: -1, B: 1, F: nil},
	{Str: "ając", A: 0, B: 1, F: nil},
	{Str: "sząc", A: 0, B: 2, F: nil},
	{Str: "sz", A: -1, B: 1, F: nil},
	{Str: "iejsz", A: 3, B: 1, F: nil},
}

var A_2 = []*snowballRuntime.Among{
	{Str: "a", A: -1, B: 1, F: r_R1},
	{Str: "ąca", A: 0, B: 1, F: nil},
	{Str: "ająca", A: 1, B: 1, F: nil},
	{Str: "sząca", A: 1, B: 2, F: nil},
	{Str: "ia", A: 0, B: 1, F: r_R1},
	{Str: "sza", A: 0, B: 1, F: nil},
	{Str: "iejsza", A: 5, B: 1, F: nil},
	{Str: "ała", A: 0, B: 1, F: nil},
	{Str: "iała", A: 7, B: 1, F: nil},
	{Str: "iła", A: 0, B: 1, F: nil},
	{Str: "ąc", A: -1, B: 1, F: nil},
	{Str: "ając", A: 10, B: 1, F: nil},
	{Str: "e", A: -1, B: 1, F: r_R1},
	{Str: "ące", A: 12, B: 1, F: nil},
	{Str: "ające", A: 13, B: 1, F: nil},
	{Str: "szące", A: 13, B: 2, F: nil},
	{Str: "ie", A: 12, B: 1, F: r_R1},
	{Str: "cie", A: 16, B: 1, F: nil},
	{Str: "acie", A: 17, B: 1, F: nil},
	{Str: "ecie", A: 17, B: 1, F: nil},
	{Str: "icie", A: 17, B: 1, F: nil},
	{Str: "ajcie", A: 17, B: 1, F: nil},
	{Str: "liście", A: 17, B: 4, F: nil},
	{Str: "aliście", A: 22, B: 1, F: nil},
	{Str: "ieliście", A: 22, B: 1, F: nil},
	{Str: "iliście", A: 22, B: 1, F: nil},
	{Str: "łyście", A: 17, B: 4, F: nil},
	{Str: "ałyście", A: 26, B: 1, F: nil},
	{Str: "iałyście", A: 27, B: 1, F: nil},
	{Str: "iłyście", A: 26, B: 1, F: nil},
	{Str: "sze", A: 12, B: 1, F: nil},
	{Str: "iejsze", A: 30, B: 1, F: nil},
	{Str: "ach", A: -1, B: 1, F: r_R1},
	{Str: "iach", A: 32, B: 1, F: r_R1},
	{Str: "ich", A: -1, B: 5, F: nil},
	{Str: "ych", A: -1, B: 5, F: nil},
	{Str: "i", A: -1, B: 1, F: r_R1},
	{Str: "ali", A: 36, B: 1, F: nil},
	{Str: "ieli", A: 36, B: 1, F: nil},
	{Str: "ili", A: 36, B: 1, F: nil},
	{Str: "ami", A: 36, B: 1, F: r_R1},
	{Str: "iami", A: 40, B: 1, F: r_R1},
	{Str: "imi", A: 36, B: 5, F: nil},
	{Str: "ymi", A: 36, B: 5, F: nil},
	{Str: "owi", A: 36, B: 1, F: r_R1},
	{Str: "iowi", A: 44, B: 1, F: r_R1},
	{Str: "aj", A: -1, B: 1, F: nil},
	{Str: "ej", A: -1, B: 5, F: nil},
	{Str: "iej", A: 47, B: 5, F: nil},
	{Str: "am", A: -1, B: 1, F: nil},
	{Str: "ałam", A: 49, B: 1, F: nil},
	{Str: "iałam", A: 50, B: 1, F: nil},
	{Str: "iłam", A: 49, B: 1, F: nil},
	{Str: "em", A: -1, B: 1, F: r_R1},
	{Str: "iem", A: 53, B: 1, F: r_R1},
	{Str: "ałem", A: 53, B: 1, F: nil},
	{Str: "iałem", A: 55, B: 1, F: nil},
	{Str: "iłem", A: 53, B: 1, F: nil},
	{Str: "im", A: -1, B: 5, F: nil},
	{Str: "om", A: -1, B: 1, F: r_R1},
	{Str: "iom", A: 59, B: 1, F: r_R1},
	{Str: "ym", A: -1, B: 5, F: nil},
	{Str: "o", A: -1, B: 1, F: r_R1},
	{Str: "ego", A: 62, B: 5, F: nil},
	{Str: "iego", A: 63, B: 5, F: nil},
	{Str: "ało", A: 62, B: 1, F: nil},
	{Str: "iało", A: 65, B: 1, F: nil},
	{Str: "iło", A: 62, B: 1, F: nil},
	{Str: "u", A: -1, B: 1, F: r_R1},
	{Str: "iu", A: 68, B: 1, F: r_R1},
	{Str: "emu", A: 68, B: 5, F: nil},
	{Str: "iemu", A: 70, B: 5, F: nil},
	{Str: "ów", A: -1, B: 1, F: r_R1},
	{Str: "y", A: -1, B: 5, F: nil},
	{Str: "amy", A: 73, B: 1, F: nil},
	{Str: "emy", A: 73, B: 1, F: nil},
	{Str: "imy", A: 73, B: 1, F: nil},
	{Str: "liśmy", A: 73, B: 4, F: nil},
	{Str: "aliśmy", A: 77, B: 1, F: nil},
	{Str: "ieliśmy", A: 77, B: 1, F: nil},
	{Str: "iliśmy", A: 77, B: 1, F: nil},
	{Str: "łyśmy", A: 73, B: 4, F: nil},
	{Str: "ałyśmy", A: 81, B: 1, F: nil},
	{Str: "iałyśmy", A: 82, B: 1, F: nil},
	{Str: "iłyśmy", A: 81, B: 1, F: nil},
	{Str: "ały", A: 73, B: 1, F: nil},
	{Str: "iały", A: 85, B: 1, F: nil},
	{Str: "iły", A: 73, B: 1, F: nil},
	{Str: "asz", A: -1, B: 1, F: nil},
	{Str: "esz", A: -1, B: 1, F: nil},
	{Str: "isz", A: -1, B: 1, F: nil},
	{Str: "ał", A: -1, B: 1, F: nil},
	{Str: "iał", A: 91, B: 1, F: nil},
	{Str: "ił", A: -1, B: 1, F: nil},
	{Str: "ą", A: -1, B: 1, F: r_R1},
	{Str: "ącą", A: 94, B: 1, F: nil},
	{Str: "ającą", A: 95, B: 1, F: nil},
	{Str: "szącą", A: 95, B: 2, F: nil},
	{Str: "ią", A: 94, B: 1, F: r_R1},
	{Str: "ają", A: 94, B: 1, F: nil},
	{Str: "szą", A: 94, B: 3, F: nil},
	{Str: "iejszą", A: 100, B: 1, F: nil},
	{Str: "ać", A: -1, B: 1, F: nil},
	{Str: "ieć", A: -1, B: 1, F: nil},
	{Str: "ić", A: -1, B: 1, F: nil},
	{Str: "ąć", A: -1, B: 1, F: nil},
	{Str: "aść", A: -1, B: 1, F: nil},
	{Str: "eść", A: -1, B: 1, F: nil},
	{Str: "ę", A: -1, B: 1, F: nil},
	{Str: "szę", A: 108, B: 2, F: nil},
	{Str: "łaś", A: -1, B: 4, F: nil},
	{Str: "ałaś", A: 110, B: 1, F: nil},
	{Str: "iałaś", A: 111, B: 1, F: nil},
	{Str: "iłaś", A: 110, B: 1, F: nil},
	{Str: "łeś", A: -1, B: 4, F: nil},
	{Str: "ałeś", A: 114, B: 1, F: nil},
	{Str: "iałeś", A: 115, B: 1, F: nil},
	{Str: "iłeś", A: 114, B: 1, F: nil},
}

var A_3 = []*snowballRuntime.Among{
	{Str: "ń", A: -1, B: 2, F: nil},
	{Str: "ć", A: -1, B: 1, F: nil},
	{Str: "ś", A: -1, B: 3, F: nil},
	{Str: "ź", A: -1, B: 4, F: nil},
}

var G_v = []byte{17, 65, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 16, 0, 0, 1}

type Context struct {
	i_p1 int
}

func r_mark_regions(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	context.i_p1 = env.Limit
	if !env.GoOutGrouping(G_v, 97, 281) {
		return false
	}
	env.NextChar()
	if !env.GoInGrouping(G_v, 97, 281) {
		return false
	}
	env.NextChar()
	context.i_p1 = env.Cursor
	return true
}

func r_R1(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	return context.i_p1 <= env.Cursor
}

func r_remove_endings(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	var among_var int32
	var v_1 = env.Limit - env.Cursor
lab0:
	for {
		if env.Cursor < context.i_p1 {
			break lab0
		}
		var v_2 = env.LimitBackward
		env.LimitBackward = context.i_p1
		env.Ket = env.Cursor
		if env.FindAmongB(A_0, context) == 0 {
			env.LimitBackward = v_2
			break lab0
		}
		env.Bra = env.Cursor
		env.LimitBackward = v_2
		env.SliceDel()
		break lab0
	}
	env.Cursor = env.Limit - v_1
	env.Ket = env.Cursor
	among_var = env.FindAmongB(A_2, context)
	if among_var == 0 {
		return false
	}
	env.Bra = env.Cursor
	switch among_var {
	case 1:
		env.SliceDel()
	case 2:
		env.SliceFrom("s")
	case 3:
	lab1:
		for {
			var v_3 = env.Limit - env.Cursor
		lab2:
			for {
				if !r_R1(env, context) {
					break lab2
				}
				env.SliceDel()
				break lab1
			}
			env.Cursor = env.Limit - v_3
			env.SliceFrom("s")
			break lab1
		}
	case 4:
		env.SliceFrom("ł")
	case 5:
		env.SliceDel()
		var v_4 = env.Limit - env.Cursor
	lab3:
		for {
			env.Ket = env.Cursor
			among_var = env.FindAmongB(A_1, context)
			if among_var == 0 {
				env.Cursor = env.Limit - v_4
				break lab3
			}
			env.Bra = env.Cursor
			switch among_var {
			case 1:
				env.SliceDel()
			case 2:
				env.SliceFrom("s")
			}
			break lab3
		}
	}
	var v_5 = env.Limit - env.Cursor
lab4:
	for {
		env.Ket = env.Cursor
		if !env.EqSB("'") {
			env.Cursor = env.Limit - v_5
			break lab4
		}
		env.Bra = env.Cursor
		env.SliceDel()
		break lab4
	}
	return true
}

func r_normalize_consonant(env *snowballRuntime.Env, ctx interface{}) bool {
	context := ctx.(*Context)
	_ = context
	var among_var int32
	env.Ket = env.Cursor
	among_var = env.FindAmongB(A_3, context)
	if among_var == 0 {
		return false
	}
	env.Bra = env.Cursor
	if env.Cursor <= env.LimitBackward {
		return false
	}
	switch among_var {
	case 1:
		env.SliceFrom("c")
	case 2:
		env.SliceFrom("n")
	case 3:
		env.SliceFrom("s")
	case 4:
		env.SliceFrom("z")
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
lab0:
	for {
		var v_2 = env.Cursor
	lab1:
		for {
			if !env.Hop(2) {
				break lab1
			}
			env.LimitBackward = env.Cursor
			env.Cursor = env.Limit
			if !r_remove_endings(env, context) {
				break lab1
			}
			env.Cursor = env.LimitBackward
			break lab0
		}
		env.Cursor = v_2
		env.LimitBackward = env.Cursor
		env.Cursor = env.Limit
		if !r_normalize_consonant(env, context) {
			return false
		}
		env.Cursor = env.LimitBackward
		break lab0
	}
	return true
}
