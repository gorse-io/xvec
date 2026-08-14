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
//
// Code generated from Snowball 3.1.1 libstemmer/modules.txt; DO NOT EDIT.

package stemmers

import (
	"sort"

	snowball "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball"
	arabic "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball/arabic"
	armenian "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball/armenian"
	basque "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball/basque"
	catalan "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball/catalan"
	czech "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball/czech"
	danish "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball/danish"
	dutch "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball/dutch"
	dutch_porter "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball/dutch_porter"
	english "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball/english"
	esperanto "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball/esperanto"
	estonian "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball/estonian"
	finnish "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball/finnish"
	french "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball/french"
	german "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball/german"
	greek "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball/greek"
	hindi "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball/hindi"
	hungarian "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball/hungarian"
	indonesian "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball/indonesian"
	irish "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball/irish"
	italian "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball/italian"
	lithuanian "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball/lithuanian"
	nepali "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball/nepali"
	norwegian "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball/norwegian"
	persian "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball/persian"
	polish "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball/polish"
	porter "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball/porter"
	portuguese "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball/portuguese"
	romanian "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball/romanian"
	russian "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball/russian"
	serbian "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball/serbian"
	sesotho "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball/sesotho"
	spanish "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball/spanish"
	swedish "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball/swedish"
	tamil "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball/tamil"
	turkish "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball/turkish"
	yiddish "github.com/gorse-io/xvec/internal/db/index/column/fts_column/tokenizer/snowball/yiddish"
)

// Func applies one generated Snowball algorithm to env.
type Func func(env *snowball.Env) bool

var aliases = map[string]Func{
	"ar":              arabic.Stem,
	"ara":             arabic.Stem,
	"arabic":          arabic.Stem,
	"arm":             armenian.Stem,
	"armenian":        armenian.Stem,
	"baq":             basque.Stem,
	"basque":          basque.Stem,
	"ca":              catalan.Stem,
	"cat":             catalan.Stem,
	"catalan":         catalan.Stem,
	"ces":             czech.Stem,
	"cs":              czech.Stem,
	"cze":             czech.Stem,
	"czech":           czech.Stem,
	"da":              danish.Stem,
	"dan":             danish.Stem,
	"danish":          danish.Stem,
	"de":              german.Stem,
	"deu":             german.Stem,
	"dut":             dutch.Stem,
	"dutch":           dutch.Stem,
	"dutch_porter":    dutch_porter.Stem,
	"el":              greek.Stem,
	"ell":             greek.Stem,
	"en":              english.Stem,
	"eng":             english.Stem,
	"english":         english.Stem,
	"eo":              esperanto.Stem,
	"epo":             esperanto.Stem,
	"es":              spanish.Stem,
	"esl":             spanish.Stem,
	"esperanto":       esperanto.Stem,
	"est":             estonian.Stem,
	"estonian":        estonian.Stem,
	"et":              estonian.Stem,
	"eu":              basque.Stem,
	"eus":             basque.Stem,
	"fa":              persian.Stem,
	"fas":             persian.Stem,
	"fi":              finnish.Stem,
	"fin":             finnish.Stem,
	"finnish":         finnish.Stem,
	"fr":              french.Stem,
	"fra":             french.Stem,
	"fre":             french.Stem,
	"french":          french.Stem,
	"ga":              irish.Stem,
	"ger":             german.Stem,
	"german":          german.Stem,
	"gle":             irish.Stem,
	"gre":             greek.Stem,
	"greek":           greek.Stem,
	"hi":              hindi.Stem,
	"hin":             hindi.Stem,
	"hindi":           hindi.Stem,
	"hu":              hungarian.Stem,
	"hun":             hungarian.Stem,
	"hungarian":       hungarian.Stem,
	"hy":              armenian.Stem,
	"hye":             armenian.Stem,
	"id":              indonesian.Stem,
	"ind":             indonesian.Stem,
	"indonesian":      indonesian.Stem,
	"irish":           irish.Stem,
	"it":              italian.Stem,
	"ita":             italian.Stem,
	"italian":         italian.Stem,
	"kraaij_pohlmann": dutch.Stem,
	"lit":             lithuanian.Stem,
	"lithuanian":      lithuanian.Stem,
	"lt":              lithuanian.Stem,
	"ne":              nepali.Stem,
	"nep":             nepali.Stem,
	"nepali":          nepali.Stem,
	"nl":              dutch.Stem,
	"nld":             dutch.Stem,
	"no":              norwegian.Stem,
	"nor":             norwegian.Stem,
	"norwegian":       norwegian.Stem,
	"pers":            persian.Stem,
	"persian":         persian.Stem,
	"pl":              polish.Stem,
	"pol":             polish.Stem,
	"polish":          polish.Stem,
	"por":             portuguese.Stem,
	"porter":          porter.Stem,
	"portuguese":      portuguese.Stem,
	"pt":              portuguese.Stem,
	"ro":              romanian.Stem,
	"romanian":        romanian.Stem,
	"ron":             romanian.Stem,
	"ru":              russian.Stem,
	"rum":             romanian.Stem,
	"rus":             russian.Stem,
	"russian":         russian.Stem,
	"serbian":         serbian.Stem,
	"sesotho":         sesotho.Stem,
	"sot":             sesotho.Stem,
	"spa":             spanish.Stem,
	"spanish":         spanish.Stem,
	"sr":              serbian.Stem,
	"srp":             serbian.Stem,
	"st":              sesotho.Stem,
	"sv":              swedish.Stem,
	"swe":             swedish.Stem,
	"swedish":         swedish.Stem,
	"ta":              tamil.Stem,
	"tam":             tamil.Stem,
	"tamil":           tamil.Stem,
	"tr":              turkish.Stem,
	"tur":             turkish.Stem,
	"turkish":         turkish.Stem,
	"yi":              yiddish.Stem,
	"yid":             yiddish.Stem,
	"yiddish":         yiddish.Stem,
}

// Lookup resolves a case-sensitive libstemmer language name or alias.
func Lookup(language string) (Func, bool) {
	stem, found := aliases[language]
	return stem, found
}

// Languages returns every supported libstemmer name and alias in lexical order.
func Languages() []string {
	languages := make([]string, 0, len(aliases))
	for language := range aliases {
		languages = append(languages, language)
	}
	sort.Strings(languages)
	return languages
}
