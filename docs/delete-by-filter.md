# Delete by filter

`Collection.DeleteByFilter(ctx, filter)` removes every current live document
for which the v0.2 SQL filter evaluates to `TRUE`. It returns nil when the
filter is valid but matches nothing. An empty, malformed, type-invalid, or
unknown-field expression returns `ErrInvalidArgument`; read-only handles return
`ErrPermissionDenied`.

Selection runs against one consistent live-document snapshot while the
collection write lock is held. Only the current primary-key version is
considered: obsolete versions in immutable segments cannot cause a newer
version to be deleted. INVERT fields use the same conservative candidate
planner as vector queries, and every candidate is forward-verified before its
key is selected.

Each selected primary key is deleted through the checksummed WAL before its
in-memory mapping and deletion bitmap are changed. The result therefore
survives Close or process restart without requiring Flush, including matches
from immutable and current writing segments. Flush later publishes the normal
deletion snapshot.

```go
err := collection.DeleteByFilter(ctx,
    "rating >= 4 AND tags CONTAIN_ANY ('archived')",
)
```

Deletion follows the library's existing batch-write cancellation model. The
context is checked during selection and before each selected key; if it is
canceled after some WAL records have committed, those completed deletions
remain durable and the method returns the cancellation error.
