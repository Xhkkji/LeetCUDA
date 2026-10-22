---
name: create-kernel-practice
description: Create safe, answer-free CUDA operator practice files from a completed LeetCUDA kernel and its Python harness. Use when preparing *_practice.cu and *_practice.py exercises, preserving reference files and non-exercise scaffolding while adding independent compilation, correctness checks, and boundary coverage.
---

# Create CUDA Kernel Practice Files

Convert one completed operator directory under `kernels/` into a progressive exercise without modifying the reference implementation.

## Inspect before editing

1. Read the operator's `.cu`, `.py`, and `README.md` completely.
2. Check Git status and preserve existing user edits.
3. Identify:
   - CUDA kernels the learner should implement;
   - helper macros and includes;
   - host launch/binding code;
   - Python compilation and benchmark code;
   - mathematical definition and input/output shapes documented in comments;
   - assumptions about dtype, shape, contiguity, alignment, and vector width.
4. Audit the reference launch and boundary behavior. Do not copy a known under-launch or out-of-bounds bug into the exercise scaffold.

## Create the CUDA exercise

Create `<operator>_practice.cu` and leave the reference `.cu` unchanged.

- Preserve kernel signatures so the binding remains usable.
- Replace only learner-owned kernel bodies with concise TODOs.
- Order TODOs from scalar to vectorized and packed implementations.
- State the goal and constraints without embedding the answer.
- Preserve concise reference-comment context in each TODO: dtype, input/output shapes, mathematical formula, and intended grid/block relationship. Normalize launch notation such as `grid(ceil(N/256))` when needed, and do not preserve an unsafe boundary assumption.
- Preserve required includes, type-punning helpers, bindings, and module registration.
- Keep launch configuration outside learner-owned TODOs unless launch design is itself the exercise.
- For vector widths `W`, ensure the scaffold launches `ceil(N/W)` logical workers and makes tail paths reachable.
- Validate CUDA device, dtype, contiguity, equal element counts, empty input, and supported index range in the host wrapper when practical.

## Create the Python exercise harness

Create `<operator>_practice.py` and leave the reference `.py` unchanged.

1. Set repository-local compiler and architecture defaults before importing PyTorch. Use `setdefault` so callers may override them.
2. Give the extension a unique name such as `<operator>_practice_lib`.
3. Compile `<operator>_practice.cu`, never the reference source.
4. Compare every custom implementation against a trusted PyTorch reference with `torch.testing.assert_close`.
5. Initialize output with `NaN` or a sentinel so missing writes cannot silently pass.
6. Run small correctness cases before long benchmarks.
7. Include lengths around every vector width, for example `1, 2, 3, 4, 5, 7, 8, 9, 17, 257`.
8. Synchronize CUDA before evaluating results and around wall-clock timing.
9. Use an `if __name__ == "__main__"` guard so the extension can be imported independently.
10. Emit an error naming the failed TODO without revealing its implementation.
11. Order correctness execution by TODO first and test size second: run every boundary case for TODO 1 before starting TODO 2, and so on.
12. Print an explicit `PASS TODO N` plus a representative output immediately after each implementation passes all its cases.
13. Fail fast at the first incomplete or incorrect TODO, but preserve the PASS output already produced for every preceding TODO. Never require all TODOs to be complete before reporting earlier success.

## Preserve validation integrity

- Do not copy reference kernel bodies into TODO comments.
- Do not weaken assertions merely to make an unfinished exercise pass.
- Do not rely only on large dimensions divisible by all vector widths.
- Do not treat successful compilation as proof of correctness.
- Do not treat correct output as proof that vectorized machine instructions were generated.
- Do not modify system CUDA links, GPU drivers, compiler alternatives, shell profiles, or global Conda configuration.

## Validate the generated exercise

1. Run Python syntax checking.
2. Compile the empty exercise and confirm the extension can load.
3. Run it and confirm it fails clearly at the first unimplemented TODO.
4. Check that reference files remain unchanged.
5. Check that only the intended practice files and this workflow document appear in Git status.
6. After the learner completes all TODOs, rerun boundary correctness tests and benchmarks separately.
7. During progressive completion, confirm each completed TODO prints PASS after all of its boundary cases, followed by a clear failure at the next unfinished TODO.
8. Confirm every TODO retains the operator formula, tensor-shape contract, dtype/vector width, and grid/block hint without revealing its implementation.

## Expected outputs

For an operator named `relu`, produce:

```text
kernels/relu/relu.cu              unchanged reference
kernels/relu/relu.py              unchanged reference
kernels/relu/relu_practice.cu     learner-owned kernel TODOs
kernels/relu/relu_practice.py     independent build, tests, and benchmark
```
