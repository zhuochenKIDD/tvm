# Apache TVM Architecture Reference

## Project Identity
- **Name**: Apache TVM v0.24.dev0
- **Purpose**: Open ML compilation framework — transforms ML models into optimized deployable code
- **Design**: Python-first, cross-level IR (Relax graph + TIRx tensor), universal hardware deployment
- **Repo**: https://github.com/apache/tvm

## Codebase Scale
- C++ core: ~274K lines, 644 .cc/.h files (`src/`, `include/tvm/`)
- Python API: ~179K lines, 744 .py files (`python/tvm/`)
- Tests: `tests/python/` mirrors source structure

## Tech Stack
| Component | Technology |
|-----------|-----------|
| Core | C++ (CMake build) |
| Python API | Python 3.10+, tvm_ffi bridge |
| Build | CMake + scikit-build-core |
| Deps | numpy, scipy, cloudpickle, ml_dtypes, tornado |
| Testing | pytest (xdist, cov, benchmark, timeout) |
| Linting | ruff (100 char lines), mypy |
| CI | Jenkins (primary) + GitHub Actions (lint, bots) |

## Architecture Layers

```
Frontends (PyTorch, ONNX, TFLite) → python/tvm/relax/frontend/
         ↓
Relax IR (graph-level) → src/relax/, python/tvm/relax/
         ↓
TIRx / S-TIR (tensor-level) → src/tirx/, src/s_tir/, python/tvm/tirx/, python/tvm/s_tir/
         ↓
Target / Codegen → src/target/ (LLVM, CUDA, SPIRV, Metal, Vulkan)
         ↓
Runtime / VM → src/runtime/ (VM execution, device mgmt, RPC)
```

## Module Map

| Module | Dir (C++ / Python) | Lines (C++ / Py) | Purpose |
|--------|---------------------|-------------------|---------|
| **relax** | `src/relax/` / `python/tvm/relax/` | 63K / 65K | Graph-level IR, ops, transforms, backend, frontends |
| **s_tir** | `src/s_tir/` / `python/tvm/s_tir/` | 63K / 33K | Schedulable TIR: Schedule API, meta_schedule, DLight |
| **runtime** | `src/runtime/` / `python/tvm/runtime/` | 56K / 4K | VM, device APIs, RPC, KV cache, disco |
| **tirx** | `src/tirx/` / `python/tvm/tirx/` | 27K / 10K | Core Tensor IR: exprs, stmts, buffers, PrimFunc |
| **target** | `src/target/` / `python/tvm/target/` | 27K / 2K | Codegen backends: LLVM, SPIRV, source |
| **arith** | `src/arith/` / `python/tvm/arith/` | 17K / 1K | Arithmetic analysis, simplification |
| **script** | `src/script/` / `python/tvm/script/` | 9K / 12K | TVMScript parser & printer |
| **ir** | `src/ir/` / `python/tvm/ir/` | 4K / 3K | Shared IR infra: IRModule, Op, transform |
| **topi** | `src/topi/` / `python/tvm/topi/` | 1K / 25K | Tensor operator inventory (compute defs) |
| **contrib** | `src/contrib/` / `python/tvm/contrib/` | ~0 / 14K | External libs: cuBLAS, cuDNN, CUTLASS, etc. |

## Key Abstractions

1. **`IRModule`** — Container: `GlobalVar → BaseFunc`. Central to all passes.
   - C++: `include/tvm/ir/module.h`
   - Py: `python/tvm/ir/module.py`

2. **`PrimFunc`** — Tensor-level function (loop nests, buffers).
   - C++: `include/tvm/tirx/function.h`
   - Py: `python/tvm/tirx/function.py`

3. **`relax.Function`** — Graph-level function (dataflow, calls, control flow).
   - Py: `python/tvm/relax/expr.py`

4. **`Schedule`** — Scheduling interface for TIR transformations.
   - Py: `python/tvm/s_tir/schedule/schedule.py`

5. **`BlockBuilder`** — Programmatic Relax IR construction (used by frontends).
   - Py: `python/tvm/relax/block_builder.py`

6. **`runtime.Module`** — Compiled module with generated code.
   - Py: `python/tvm/runtime/module.py`

## Data Flow
1. Frontend import → IRModule (Relax functions)
2. Relax transforms (fusion, legalization, optimization)
3. TIR scheduling (manual / meta_schedule / DLight)
4. Codegen (LLVM/CUDA/SPIRV/etc.)
5. VM build → runtime.Module / VMExecutable
6. Execution via Relax VM

## Build & Dev

```bash
# Build
mkdir build && cd build
cp ../cmake/config.cmake .  # edit to enable CUDA/LLVM/etc.
cmake .. && make -j$(nproc)

# Python
export TVM_HOME=/path/to/tvm
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH

# Test
python -m pytest tests/python/relax/ -x -v --timeout=300

# Lint
ruff check python/ tests/
```

## Gotchas
- `tirx` vs `s_tir`: tirx = core IR defs, s_tir = scheduling layer
- No Relay — Relax replaced it
- `_ffi_api.py` in each module = auto FFI bridge
- `_RUNTIME_ONLY` guards compiler-only imports
- `config.cmake` in build dir for options (not CLI flags)
- TVMScript: `@T.prim_func` (TIR), `@R.function` (Relax)
- ruff enforces 100-char lines; `__init__.py` has relaxed import rules

## Important Subdirectories
- `src/relax/backend/` — Relax compilation backend (codegen dispatch)
- `src/relax/transform/` — Graph-level optimization passes
- `src/relax/op/` — Relax operator definitions and type inference
- `src/s_tir/schedule/` — Schedule primitives implementation
- `src/s_tir/meta_schedule/` — Auto-tuning search framework
- `python/tvm/s_tir/dlight/` — Rule-based auto-scheduling (GPU focus)
- `src/runtime/vm/` — Relax VM implementation (vm.cc, bytecode, paged_kv_cache)
- `python/tvm/relax/frontend/torch/` — PyTorch → Relax importer
- `docs/arch/` — Architecture documentation (fusion, VM, BYOC, etc.)
