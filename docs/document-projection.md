# Documents and projection

`zvec.Document` contains a primary key, named fields, a query score, and the
internal global document ID. Field values use explicit Go types: primitive
scalar types, named array types, `Binary`, the dense vector types, or one of the
two sparse vector structs. Plain `int`, `[]byte`, and `[]float32` are rejected
because they would be ambiguous on disk. Documents clone all slice-backed
values and canonicalize sparse coordinates.

Full document validation checks the collection schema, required and nullable
fields, exact data types, and dense dimensions. The native document payload is
deterministic and checksummed so reopen never relies on Go map iteration order
or JSON's generic numeric representation.

Projection preserves the baseline's nil-versus-empty distinction:

- `OutputFields == nil` selects every scalar and array field.
- an allocated empty `OutputFields` selects no scalar fields;
- a non-empty list selects exactly those scalar fields;
- `[]string{"*"}` is an explicit select-all;
- `IncludeVectors` independently includes every dense and sparse vector.

Projected documents retain primary key, score, and internal document ID, and
own copies of every selected value. Unknown fields, duplicate names, vector
names in `OutputFields`, and a wildcard mixed with other names are rejected.
