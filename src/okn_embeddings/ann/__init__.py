"""ANN sidecar artifacts: building, reading, and (eventually) evaluating
usearch indexes over embed Parquet files."""

# torch must load before usearch. Both torch and numkong (usearch >= 2.26's
# SIMD dependency) bundle their own private libomp.dylib, and two live
# OpenMP runtimes cannot share a process: whichever loads first wins the
# interposable symbol bindings, and torch's threaded CPU inference then
# runs its fork barriers against thread structures the other runtime
# initialized -- SIGSEGV in __kmp_fork_barrier on the first forward pass.
# torch-first keeps torch's copy primary, which both libraries tolerate.
# (GPU inference never enters the OpenMP thread-team path, which is why
# interactive MPS use never crashed while the CPU-only test suite did.)
# Known upstream: https://github.com/ashvardanian/NumKong/issues/373 --
# remove this pin once numkong stops bundling its own OpenMP runtime.
import torch  # noqa: F401  (imported for its side effect, see above)
