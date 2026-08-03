# SQL filter parser

The v0.2 filter frontend is a dependency-free lexer, recursive-descent parser,
and typed AST in `internal/db/sql`. It follows the pinned baseline grammar and
parses one complete expression:

```text
expression  = or-expression
or          = and { OR and }
and         = primary { AND primary }
primary     = predicate | "(" expression ")"
predicate   = identifier comparison value
            | function-call comparison value
            | identifier LIKE value
            | identifier [NOT] IN "(" literal { "," literal } ")"
            | identifier [NOT] (CONTAIN_ALL | CONTAIN_ANY)
                "(" [ literal { "," literal } ] ")"
            | identifier IS [NOT] NULL
```

Keywords are case-insensitive; identifiers and string content preserve source
case. The baseline identifier alphabet is ASCII letters, digits, `_`, and `-`,
including identifiers that begin with a digit or hyphen. Its keyword whitelist
also permits names such as `and`, `not`, and `where`, while `from`, Boolean/NULL
literals, and contain operators remain reserved.

Integer text is not prematurely narrowed to `int64`, so schema analysis can
accept the full unsigned range. Floats preserve exponent/suffix syntax.
Single- and double-quoted strings accept a backslash plus any following
character and reproduce the baseline normalization of escaped quotes. Boolean
literals normalize to lowercase. Function arguments may contain literals,
identifiers, nested calls, and numeric vector/matrix literals.

`AND` binds tighter than `OR`; both associate left. `IN` requires at least one
literal, while the contain predicates intentionally accept an empty list.
`BETWEEN`, unary Boolean `NOT`, and `NOT LIKE` are not filter productions in the
pinned grammar and are rejected.

Every token and AST node has a half-open source span. `ParseError` reports a
zero-based UTF-8 byte offset plus one-based line and Unicode column, including
lexer failures, unexpected tokens, incomplete expressions, and a guarded
nesting-depth limit. SQL line and block comments are skipped.

The typed predicate runtime is documented in
[Scalar filter evaluation](filter-evaluation.md), and its schema binding,
rewrites, and live-query integration are documented in
[Schema analysis and filter plans](filter-plans.md).
