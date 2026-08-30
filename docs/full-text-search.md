# Full-text search

A string field becomes searchable when its schema uses `FTSIndexParams`.
`NewFTSIndexParams` selects the standard tokenizer and lowercase filter.
Tokenization and filters are used consistently for document indexing, natural
language matching, and parsed query terms.

```go
field := xvec.FieldSchema{
    Name: "body",
    DataType: xvec.DataTypeString,
    Index: xvec.FTSIndexParams{
        Tokenizer: "standard",
        Filters: []string{"lowercase", "ascii_folding"},
    },
}
```

`FTSClause.Match` analyzes the whole input as natural language and combines
terms with the selected default operator. `FTSClause.Query` interprets the FTS
expression grammar. Exactly one must be non-empty.

## Tokenizers and filters

Supported tokenizers are:

- `standard`: Unicode-aware word segmentation with deterministic pinned tables;
- `whitespace`: splits on Unicode whitespace while preserving byte offsets;
- `ngram`: emits overlapping codepoint n-grams, defaulting to length 2;
- `jieba`: pure-Go cppjieba-compatible Chinese segmentation using external
  dictionary/HMM resources.

N-gram minimum and maximum lengths must be positive, ordered, and differ by at
most one. An optional character-class mask limits segments to letters, digits,
whitespace, punctuation, and/or symbols. Malformed UTF-8 bytes terminate a
segment rather than being merged into surrounding n-grams.

Jieba supports search, mix, full, and HMM modes. Search is the default. Search
and mix need dictionary plus HMM model; full needs the dictionary; HMM needs the
model. The dictionary directory resolves from field configuration, then
`ZVEC_JIEBA_DICT_DIR`, then the process fallback. Production language resources
are not embedded in the module and must be provided separately.

Supported filters include lowercase, ASCII folding, and Snowball stemming.
Filters run in declaration order. Lowercase and character categories use pinned
Unicode data so behavior does not depend on the host Go release. Stemming
requires a supported language and fails explicitly for an unknown language.
Tokenizer/filter configuration is validated before publication.

`ExtraParams` is a JSON object shared by tokenizer and filter configuration:

| Key | Applies to | Values |
| --- | --- | --- |
| `max_token_length` | standard | integer in `[1, 1048576]` |
| `ngram_min`, `ngram_max` | ngram | positive uint32 values, ordered and differing by at most one |
| `token_chars` | ngram | array containing `letter`, `digit`, `whitespace`, `punctuation`, and/or `symbol` |
| `jieba_dict_dir` | jieba | resource directory string |
| `user_dict_path` | jieba | user dictionary path string |
| `cut_mode` | jieba | `search`, `mix`, `full`, or `hmm` |
| `stemmer_lang` | stemmer filter | a supported Snowball language |

Unknown tokenizer/filter names and invalid JSON values are rejected. Custom
n-gram character definitions are not supported.

## Query language

The parser recognizes case-insensitive `OR`, `AND`, and binary `NOT`; unary `+`
and `-`; parentheses; double-quoted phrases; ordinary and escaped terms. Its
precedence is:

1. `OR`;
2. `AND` and binary `NOT`;
3. adjacent atoms using the configured default operator; and
4. unary modifiers and atoms.

For example:

```text
+vector -slow "exact phrase"
```

`a NOT b` and `a AND NOT b` are equivalent. Leading `NOT` is invalid because
NOT is binary. A quoted phrase requires exact adjacent analyzed positions.
Field prefixes such as `title:vector` and boosts such as `vector^2` are parsed
but return an unsupported-query error rather than being treated as text.

The parser validates complete syntax before semantic support and analysis, so a
later syntax error is not hidden by an earlier tokenizer error. Token and nesting
bounds return explicit complexity errors.

## Dictionary, postings, and BM25

Each immutable segment has its own term dictionary and compressed posting lists
with document frequency, term frequency, positions, and document lengths.
Dictionaries and posting chunks are checksummed and validated on reopen. Flush
builds artifacts for newly sealed segments; Optimize merges current live state.

Collection-level BM25 aggregates corpus statistics across all live segments and
excludes deletions. It uses Robertson-Sparck Jones smoothing:

```text
idf(t) = ln((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
```

Scored boolean execution uses block-max WAND to skip blocks whose upper bound
cannot enter the current top-k. Required and prohibited clauses determine
matching; optional clauses contribute score. Phrase candidates are first
intersected by term and then verified by positions.

A scalar filter masks FTS candidates without changing corpus size, IDF, or
average document length. For a sufficiently selective filter, the runtime may
seek only candidate ordinals; this is an exact planner choice.

## Persistence and consistency

FTS state is cached per immutable segment. Native artifacts use Pebble
byte-ordered keys and bounded posting chunks with contiguous ordinal suffixes.
Each artifact has a synchronized wrapper marker and is referenced only after it
is complete. Missing metadata may be rebuilt from durable documents; malformed
metadata, posting domains, chunk order, or statistics are treated as corruption.

All query work runs against an immutable collection snapshot. Tokenizers,
filters, dictionaries, and query plans are safe for concurrent use when shared
through that snapshot, and cancellation returns no partial result.
