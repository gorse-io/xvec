# DropColumn

`Collection.DropColumn(ctx, column)` atomically removes one existing basic
numeric field. The installed v0.2 unit supports `INT32`, `INT64`, `UINT32`,
`UINT64`, `FLOAT`, and `DOUBLE`, including fields with INVERT parameters. Other
scalar, array, vector, and FTS fields are rejected until their owning index
milestones can migrate auxiliary state without leaving an inconsistent schema.
The last field cannot be dropped because a collection schema must remain
nonempty.

```go
if err := collection.DropColumn(ctx, "legacy_score"); err != nil {
    return err
}
```

DropColumn holds the collection write lock and removes the key from every live
document payload; result projection does not merely hide it. It then writes new
immutable segments, an IDMap checkpoint, an empty delete snapshot, and a new WAL before
publishing the reduced schema through `CURRENT`. Live document IDs and the next
writable ID remain unchanged, while superseded versions and deletion records
are reclaimed. A failure before the manifest commit leaves the old schema and
payloads authoritative, and a successful drop survives reopen without Flush.

Queries, filters, projections, and subsequent writes that still name the
dropped field fail schema validation. Dropping a field from an empty collection
publishes only the schema generation because there are no payloads to rewrite.
