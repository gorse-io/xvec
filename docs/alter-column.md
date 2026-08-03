# AlterColumn

`Collection.AlterColumn(ctx, column, rename, field, options)` atomically changes
one existing basic numeric column. It has two mutually exclusive forms:

- Set `rename` and pass a nil `field` to change only the name while retaining
  the data type, nullability, and index parameters.
- Leave `rename` empty and pass `field` to replace the field definition. The
  replacement may keep or change the name, convert among `INT32`, `INT64`,
  `UINT32`, `UINT64`, `FLOAT`, and `DOUBLE`, change a non-nullable field to
  nullable, and add, remove, or reconfigure an implemented INVERT index.

```go
replacement := zvec.FieldSchema{
    Name: "adjusted", DataType: zvec.DataTypeInt64,
    Index: zvec.NewInvertIndexParams(),
}
err := collection.AlterColumn(ctx, "rating", "", &replacement,
    zvec.AlterColumnOptions{Concurrency: 4},
)
```

Changing a nullable field to non-nullable is rejected even if the current
snapshot happens not to contain NULL. Nonnumeric source or destination fields,
duplicate or invalid new names, and specifying both forms are also rejected.
Numeric conversion uses the pinned baseline's permissive cast: floating values
truncate toward zero when converted to integers, and integer overflow wraps to
the destination width. NULL remains NULL, while an omitted nullable value
remains omitted.

`Concurrency` bounds parallel value conversion; zero selects the library
default and a negative value is invalid. AlterColumn holds the collection write
lock and publishes the converted live snapshot, primary-key state, empty delete
snapshot, new WAL, and schema through one manifest commit. Live document IDs
and the next writable ID are preserved. A failure before `CURRENT` changes
leaves the prior name, types, values, and manifest active; a successful change
survives reopen without Flush. Later Insert and Upsert calls must use the new
field definition.
