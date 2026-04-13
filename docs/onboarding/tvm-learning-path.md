# TVM 深入学习路线（中文 TODO 教程）

## 阶段一：理解全局架构（~2小时）

### TODO 1：理解 IRModule — TVM 的核心容器
- **读**：`include/tvm/ir/module.h` (L44-L100)
- **读**：`python/tvm/ir/module.py`
- **目标**：理解 `IRModule` 是 `GlobalVar → BaseFunc` 的映射容器，所有编译 pass 都以它为输入输出。

### TODO 2：理解 Python ↔ C++ 的 FFI 桥接
- **读**：`python/tvm/__init__.py` — 看顶层 `from tvm_ffi import ...`
- **读**：任意一个 `_ffi_api.py`（如 `python/tvm/relax/_ffi_api.py`）
- **读**：`src/ir/module.cc` 中的 `TVM_REGISTER_GLOBAL` 调用
- **目标**：理解 C++ 函数如何通过 `TVM_REGISTER_GLOBAL` 注册，Python 端通过 `_ffi_api` 调用。这是 TVM 架构的骨架。

### TODO 3：理解 TVMScript 语法
- **读**：`python/tvm/script/tirx.py` 和 `python/tvm/script/relax.py`
- **跑一下**：找一个测试用例，如 `tests/python/tvmscript/` 下的文件
- **目标**：理解 `@T.prim_func` 和 `@R.function` 装饰器如何将 Python 语法解析为 IR 节点。

---

## 阶段二：深入 Tensor IR（TIRx / S-TIR）（~4小时）

### TODO 4：理解 TIR 表达式和语句
- **读**：`python/tvm/tirx/__init__.py` — 列出所有 TIR 类型
- **读**：`python/tvm/tirx/expr.py` (Var, Buffer, Add, Mul, Call 等)
- **读**：`python/tvm/tirx/stmt.py` (For, BufferStore, SBlock, SeqStmt 等)
- **目标**：理解 TIR 是一个类 C 的低级 IR，用 `For` 循环 + `BufferStore` 表示计算。`SBlock`（Schedule Block）是调度的基本单元。

### TODO 5：理解 PrimFunc
- **读**：`python/tvm/tirx/function.py`
- **读**：`include/tvm/tirx/function.h`
- **目标**：`PrimFunc` 封装了一个 TIR 计算体（参数列表 + buffer 映射 + 语句体）。它是从 Relax 调用的最小计算单元。

### TODO 6：理解 Schedule API — 如何变换循环
- **读**：`python/tvm/s_tir/schedule/schedule.py` — 重点看 `split`, `reorder`, `bind`, `compute_at`, `vectorize`, `unroll`
- **跑一下**：`tests/python/s_tir/test_schedule_*.py` 中的一个简单用例
- **目标**：Schedule 不改变计算语义，只改变循环结构和数据布局。理解"分离计算定义和调度"这一核心设计理念。

### TODO 7：理解 meta_schedule 自动调优
- **读**：`python/tvm/s_tir/meta_schedule/tune.py` — 入口
- **读**：`python/tvm/s_tir/meta_schedule/schedule_rule/` — 调度规则
- **读**：`python/tvm/s_tir/meta_schedule/search_strategy/` — 搜索策略
- **目标**：meta_schedule 通过搜索调度空间自动找到最优调度。理解 "Task → SearchSpace → Measure → CostModel" 的闭环。

### TODO 8：理解 DLight 自动调度
- **读**：`python/tvm/s_tir/dlight/__init__.py`
- **读**：`python/tvm/s_tir/dlight/gpu/` — GPU 调度规则
- **目标**：DLight 是基于规则的轻量调度器（不搜索），为常见 GPU pattern 提供默认调度。对比理解它与 meta_schedule 的区别。

---

## 阶段三：深入 Relax 图级 IR（~4小时）

### TODO 9：理解 Relax 表达式体系
- **读**：`python/tvm/relax/expr.py` — `Var`, `Call`, `Function`, `SeqExpr`, `BindingBlock`, `DataflowBlock`
- **读**：`python/tvm/relax/struct_info.py` — StructInfo 类型系统（Tensor 形状/dtype 推断）
- **目标**：Relax 用 A-Normal Form + DataflowBlock 表示计算图。`StructInfo` 替代了传统的 Type 系统来做形状推断。

### TODO 10：理解 BlockBuilder — 如何构建 Relax 程序
- **读**：`python/tvm/relax/block_builder.py`
- **目标**：所有前端（PyTorch、ONNX 等）通过 `BlockBuilder` 发射 Relax IR。理解 `bb.emit()`, `bb.emit_func_output()`, `bb.call_te()` 的语义。

### TODO 11：理解 Relax 算子定义
- **读**：`python/tvm/relax/op/__init__.py` — 算子列表
- **读**：`src/relax/op/` 下任意一个算子（如 `nn/convolution.cc`）
- **目标**：高级算子在 Relax 层定义（类型推断 + shape 推断），最终通过 legalization 降级到 TIR PrimFunc。

### TODO 12：理解 Relax Transform Pass
- **读**：`python/tvm/relax/transform/transform.py` — Pass 列表
- **重点关注**：`FuseOps`, `LegalizeOps`, `FuseTIR`, `DeadCodeElimination`
- **读**：`docs/arch/fusion.rst` — 算子融合架构文档
- **目标**：理解 Relax 编译管线中的关键 pass。`LegalizeOps` 将 high-level op 降级为 TIR，`FuseOps` 将多个 op 融合为一个 kernel。

---

## 阶段四：前端和端到端流程（~3小时）

### TODO 13：跟踪 PyTorch 模型导入流程
- **读**：`python/tvm/relax/frontend/torch/` — 入口文件
- **跑一下**：找一个 `tests/python/relax/test_frontend_*.py` 测试
- **目标**：跟踪一个 PyTorch 模型如何变成 IRModule。理解 FX graph → Relax IR 的翻译过程。

### TODO 14：跟踪完整编译流程 — relax.build()
- **读**：`python/tvm/relax/vm_build.py` — `build()` 函数
- **读**：`python/tvm/driver/build_module.py` — `tvm.build()`
- **目标**：理解从 IRModule → Relax transform pipeline → TIR scheduling → codegen → VMExecutable 的完整链路。

### TODO 15：理解 Relax VM 执行
- **读**：`src/runtime/vm/vm.cc` — VM 主循环
- **读**：`src/runtime/vm/bytecode.cc` — 字节码定义
- **读**：`docs/arch/relax_vm.rst` — VM 架构文档
- **目标**：理解 VM 的指令集（Call, Ret, If, Goto 等），以及如何调度 kernel 执行。

---

## 阶段五：Codegen 和运行时（~3小时）

### TODO 16：理解 Target 和代码生成
- **读**：`python/tvm/target/target.py`
- **读**：`src/target/codegen.cc` — codegen 调度
- **读**：`src/target/llvm/` — LLVM codegen（选读几个文件）
- **目标**：Target 描述硬件特征。Codegen 将 TIR 翻译为目标代码。理解 "Target → CodegenFactory → 目标代码" 流程。

### TODO 17：理解 Runtime Module 和部署
- **读**：`python/tvm/runtime/module.py` — Module 加载/导出
- **读**：`src/runtime/module.cc`
- **目标**：编译产物是 `runtime.Module`，可序列化为 `.so` / `.tar`。理解 `export_library()` 和 `load_module()` 的部署流程。

### TODO 18：理解 BYOC（Bring Your Own Codegen）
- **读**：`docs/arch/external_library_dispatch.rst`
- **读**：`python/tvm/contrib/cutlass/` — CUTLASS 集成示例
- **目标**：BYOC 允许将子图 offload 到外部库（cuBLAS、cuDNN、CUTLASS）。理解 pattern matching → partition → extern codegen 的流程。

---

## 阶段六：进阶专题（按兴趣选学）

### TODO 19：Arithmetic 分析
- **读**：`src/arith/analyzer.cc`, `src/arith/rewrite_simplify.cc`
- **目标**：理解 TVM 如何做符号表达式简化和边界推断（用于循环优化和内存分配）。

### TODO 20：Disco 分布式执行
- **读**：`src/runtime/disco/`, `python/tvm/runtime/disco/`
- **目标**：Disco 是 TVM 的分布式 runtime，支持多 GPU 执行。

### TODO 21：Paged KV Cache
- **读**：`src/runtime/vm/paged_kv_cache.cc`
- **目标**：理解 LLM serving 场景下的 paged attention KV cache 实现。

---

## 学习方法建议

1. **先跑通再读源码**：每个 TODO 先找对应的 `tests/python/` 测试跑一遍，建立直觉。
2. **TVMScript 是最好的学习工具**：用 `@T.prim_func` 和 `@R.function` 写小例子，print 出来看 IR 结构。
3. **善用 `tvm.ir.save_json()` / Script Printer**：把 IRModule 打印出来看中间状态。
4. **从 Python → C++**：先理解 Python API 语义，再看 C++ 实现细节。Python 层是"文档"，C++ 层是"实现"。
5. **关注数据流而非控制流**：理解"数据怎么从 PyTorch 模型变成 GPU kernel"比理解每个类的继承关系更重要。
