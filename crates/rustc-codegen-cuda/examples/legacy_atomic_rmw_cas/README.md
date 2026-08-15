# legacy_atomic_rmw_cas

Regression coverage for integer atomic read-modify-write and compare-exchange
operations on the legacy LLVM 7 NVVM path.

The fixture intentionally covers the subset legalized by this change:

- `DeviceAtomicU32` and `DeviceAtomicU64`;
- device synchronization scope;
- integer `fetch_add` with a strong source ordering, exercising the existing
  fence-splitting path;
- `compare_exchange` with `AcqRel` success ordering and `Relaxed` failure
  ordering;
- successful and failed CAS old-value semantics.

The legacy legalizer remains fail-closed for forms whose semantics are not
proven representable by the supported legacy NVVM dialect, including block and
system scopes, stronger CAS failure orderings, and integer widths other than
32 and 64 bits.
