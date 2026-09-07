package snowball

// The generated stemmers use Snowball 3.1.1 (commit
// cd195b51e948a902a4312f023f4a14392516a543). Set SNOWBALL to the root of a
// checkout built at that revision before running go generate.

//go:generate $SNOWBALL/snowball $SNOWBALL/algorithms/arabic.sbl -go -o arabic/arabic_stemmer -P arabic -gor github.com/gorse-io/xvec/thirdparty/snowball
//go:generate $SNOWBALL/snowball $SNOWBALL/algorithms/armenian.sbl -go -o armenian/armenian_stemmer -P armenian -gor github.com/gorse-io/xvec/thirdparty/snowball
//go:generate $SNOWBALL/snowball $SNOWBALL/algorithms/basque.sbl -go -o basque/basque_stemmer -P basque -gor github.com/gorse-io/xvec/thirdparty/snowball
//go:generate $SNOWBALL/snowball $SNOWBALL/algorithms/catalan.sbl -go -o catalan/catalan_stemmer -P catalan -gor github.com/gorse-io/xvec/thirdparty/snowball
//go:generate $SNOWBALL/snowball $SNOWBALL/algorithms/czech.sbl -go -o czech/czech_stemmer -P czech -gor github.com/gorse-io/xvec/thirdparty/snowball
//go:generate $SNOWBALL/snowball $SNOWBALL/algorithms/danish.sbl -go -o danish/danish_stemmer -P danish -gor github.com/gorse-io/xvec/thirdparty/snowball
//go:generate $SNOWBALL/snowball $SNOWBALL/algorithms/dutch.sbl -go -o dutch/dutch_stemmer -P dutch -gor github.com/gorse-io/xvec/thirdparty/snowball
//go:generate $SNOWBALL/snowball $SNOWBALL/algorithms/dutch_porter.sbl -go -o dutch_porter/dutch_porter_stemmer -P dutch_porter -gor github.com/gorse-io/xvec/thirdparty/snowball
//go:generate $SNOWBALL/snowball $SNOWBALL/algorithms/english.sbl -go -o english/english_stemmer -P english -gor github.com/gorse-io/xvec/thirdparty/snowball
//go:generate $SNOWBALL/snowball $SNOWBALL/algorithms/esperanto.sbl -go -o esperanto/esperanto_stemmer -P esperanto -gor github.com/gorse-io/xvec/thirdparty/snowball
//go:generate $SNOWBALL/snowball $SNOWBALL/algorithms/estonian.sbl -go -o estonian/estonian_stemmer -P estonian -gor github.com/gorse-io/xvec/thirdparty/snowball
//go:generate $SNOWBALL/snowball $SNOWBALL/algorithms/finnish.sbl -go -o finnish/finnish_stemmer -P finnish -gor github.com/gorse-io/xvec/thirdparty/snowball
//go:generate $SNOWBALL/snowball $SNOWBALL/algorithms/french.sbl -go -o french/french_stemmer -P french -gor github.com/gorse-io/xvec/thirdparty/snowball
//go:generate $SNOWBALL/snowball $SNOWBALL/algorithms/german.sbl -go -o german/german_stemmer -P german -gor github.com/gorse-io/xvec/thirdparty/snowball
//go:generate $SNOWBALL/snowball $SNOWBALL/algorithms/greek.sbl -go -o greek/greek_stemmer -P greek -gor github.com/gorse-io/xvec/thirdparty/snowball
//go:generate $SNOWBALL/snowball $SNOWBALL/algorithms/hindi.sbl -go -o hindi/hindi_stemmer -P hindi -gor github.com/gorse-io/xvec/thirdparty/snowball
//go:generate $SNOWBALL/snowball $SNOWBALL/algorithms/hungarian.sbl -go -o hungarian/hungarian_stemmer -P hungarian -gor github.com/gorse-io/xvec/thirdparty/snowball
//go:generate $SNOWBALL/snowball $SNOWBALL/algorithms/indonesian.sbl -go -o indonesian/indonesian_stemmer -P indonesian -gor github.com/gorse-io/xvec/thirdparty/snowball
//go:generate $SNOWBALL/snowball $SNOWBALL/algorithms/irish.sbl -go -o irish/irish_stemmer -P irish -gor github.com/gorse-io/xvec/thirdparty/snowball
//go:generate $SNOWBALL/snowball $SNOWBALL/algorithms/italian.sbl -go -o italian/italian_stemmer -P italian -gor github.com/gorse-io/xvec/thirdparty/snowball
//go:generate $SNOWBALL/snowball $SNOWBALL/algorithms/lithuanian.sbl -go -o lithuanian/lithuanian_stemmer -P lithuanian -gor github.com/gorse-io/xvec/thirdparty/snowball
//go:generate $SNOWBALL/snowball $SNOWBALL/algorithms/nepali.sbl -go -o nepali/nepali_stemmer -P nepali -gor github.com/gorse-io/xvec/thirdparty/snowball
//go:generate $SNOWBALL/snowball $SNOWBALL/algorithms/norwegian.sbl -go -o norwegian/norwegian_stemmer -P norwegian -gor github.com/gorse-io/xvec/thirdparty/snowball
//go:generate $SNOWBALL/snowball $SNOWBALL/algorithms/persian.sbl -go -o persian/persian_stemmer -P persian -gor github.com/gorse-io/xvec/thirdparty/snowball
//go:generate $SNOWBALL/snowball $SNOWBALL/algorithms/polish.sbl -go -o polish/polish_stemmer -P polish -gor github.com/gorse-io/xvec/thirdparty/snowball
//go:generate $SNOWBALL/snowball $SNOWBALL/algorithms/porter.sbl -go -o porter/porter_stemmer -P porter -gor github.com/gorse-io/xvec/thirdparty/snowball
//go:generate $SNOWBALL/snowball $SNOWBALL/algorithms/portuguese.sbl -go -o portuguese/portuguese_stemmer -P portuguese -gor github.com/gorse-io/xvec/thirdparty/snowball
//go:generate $SNOWBALL/snowball $SNOWBALL/algorithms/romanian.sbl -go -o romanian/romanian_stemmer -P romanian -gor github.com/gorse-io/xvec/thirdparty/snowball
//go:generate $SNOWBALL/snowball $SNOWBALL/algorithms/russian.sbl -go -o russian/russian_stemmer -P russian -gor github.com/gorse-io/xvec/thirdparty/snowball
//go:generate $SNOWBALL/snowball $SNOWBALL/algorithms/serbian.sbl -go -o serbian/serbian_stemmer -P serbian -gor github.com/gorse-io/xvec/thirdparty/snowball
//go:generate $SNOWBALL/snowball $SNOWBALL/algorithms/sesotho.sbl -go -o sesotho/sesotho_stemmer -P sesotho -gor github.com/gorse-io/xvec/thirdparty/snowball
//go:generate $SNOWBALL/snowball $SNOWBALL/algorithms/spanish.sbl -go -o spanish/spanish_stemmer -P spanish -gor github.com/gorse-io/xvec/thirdparty/snowball
//go:generate $SNOWBALL/snowball $SNOWBALL/algorithms/swedish.sbl -go -o swedish/swedish_stemmer -P swedish -gor github.com/gorse-io/xvec/thirdparty/snowball
//go:generate $SNOWBALL/snowball $SNOWBALL/algorithms/tamil.sbl -go -o tamil/tamil_stemmer -P tamil -gor github.com/gorse-io/xvec/thirdparty/snowball
//go:generate $SNOWBALL/snowball $SNOWBALL/algorithms/turkish.sbl -go -o turkish/turkish_stemmer -P turkish -gor github.com/gorse-io/xvec/thirdparty/snowball
//go:generate $SNOWBALL/snowball $SNOWBALL/algorithms/yiddish.sbl -go -o yiddish/yiddish_stemmer -P yiddish -gor github.com/gorse-io/xvec/thirdparty/snowball
//go:generate gofmt -s -w arabic/arabic_stemmer.go armenian/armenian_stemmer.go basque/basque_stemmer.go catalan/catalan_stemmer.go czech/czech_stemmer.go danish/danish_stemmer.go dutch/dutch_stemmer.go dutch_porter/dutch_porter_stemmer.go english/english_stemmer.go esperanto/esperanto_stemmer.go estonian/estonian_stemmer.go finnish/finnish_stemmer.go french/french_stemmer.go german/german_stemmer.go greek/greek_stemmer.go hindi/hindi_stemmer.go hungarian/hungarian_stemmer.go indonesian/indonesian_stemmer.go irish/irish_stemmer.go italian/italian_stemmer.go lithuanian/lithuanian_stemmer.go nepali/nepali_stemmer.go norwegian/norwegian_stemmer.go persian/persian_stemmer.go polish/polish_stemmer.go porter/porter_stemmer.go portuguese/portuguese_stemmer.go romanian/romanian_stemmer.go russian/russian_stemmer.go serbian/serbian_stemmer.go sesotho/sesotho_stemmer.go spanish/spanish_stemmer.go swedish/swedish_stemmer.go tamil/tamil_stemmer.go turkish/turkish_stemmer.go yiddish/yiddish_stemmer.go
