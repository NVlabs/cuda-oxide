# cuda-gdb debug-info tests

This suite exercises cuda-oxide debug information with `cuda-gdb`. For now,
these tests are intended to be run locally; they are not plugged into CI yet.

## Expected failures

Tests disabled by a known limitation are marked with strict `pytest.xfail`
markers, so an unexpected pass is reported and the marker can be removed when
the underlying behavior is fixed. The current list is:

- `test_execution.py::test_mixed_physical_and_inlined_callstack`
  The generated debug information omits locals for an inline frame.

There are no other disabled tests.

The tests require an NVIDIA GPU and driver, a CUDA toolkit containing
`cuda-gdb`, and the Python packages in `requirements.txt`. Run them from the
repository root with:

```sh
pytest scripts/debug-tests
```

Use `-s` to show the annotated cuda-gdb transcript and `-k NAME` to select a
test. `CUDA_OXIDE_CUDA_GDB` overrides the cuda-gdb executable, while
`CUDA_OXIDE_TARGET` overrides the detected GPU architecture (for example,
`sm_90`).
