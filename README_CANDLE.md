# MyGPT - Candle Version

这是使用 Candle (0.9.1) 实现的 GPT 模型，从 Burn 框架移植而来。

## 特性

- 完整的 Transformer 架构实现
- 多头自注意力机制
- 前馈神经网络
- Layer Normalization
- Dropout 正则化
- AdamW 优化器
- 支持 CUDA 加速
- 流式文本生成

## 项目结构

```
src_candle/
├── main.rs              # 主程序入口
├── modeling/            # 模型定义
│   └── mod.rs          # BigramModel, MultiHead, Block 等
├── training/            # 训练逻辑
│   └── mod.rs          # 训练循环和配置
├── generating/          # 文本生成
│   └── mod.rs          # 流式生成实现
├── tokenizing/          # 分词器
│   └── mod.rs          # 简单字符级分词器
├── dataset/             # 数据集
│   └── mod.rs          # TokenPairDataset
└── batching/            # 批处理
    └── mod.rs          # TokenPairBatch
```

## 模型架构

- **词汇表大小**: 自动从训练文本推断
- **Block Size**: 256
- **嵌入维度**: 384
- **注意力头数**: 6
- **Transformer 层数**: 6
- **Dropout**: 0.2

## 安装

1. 确保已安装 Rust (1.70+)
2. 克隆仓库并进入目录
3. 将 `Cargo_candle.toml` 重命名为 `Cargo.toml` 或直接使用：

```bash
cargo build --release
```

## 使用方法

### 训练模型

```bash
cargo run --release --bin mygpt-candle
```

首次运行会自动训练模型并保存到 `artifact_candle/` 目录。

### 使用已训练模型生成文本

如果 `artifact_candle/config.json` 和 `artifact_candle/model.safetensors` 存在，
程序会自动加载模型并生成文本。

### 自定义配置

在 `src_candle/main.rs` 中修改以下常量：

```rust
pub const MAX_BLOCK_SIZE: usize = 256;  // 上下文长度
pub const N_EMBD: usize = 384;          // 嵌入维度
pub const N_HEADS: usize = 6;           // 注意力头数
pub const N_LAYER: usize = 6;           // Transformer 层数
pub const DROPOUT: f64 = 0.2;           // Dropout 概率
pub const NUM_EPOCHS: usize = 1;        // 训练轮数
pub const BATCH_SIZE: usize = 64;       // 批大小
pub const LEARNING_RATE: f64 = 3.0e-4;  // 学习率
```

## 训练数据

默认使用 `src/4in1.txt` 作为训练数据（四大名著合集）。
可以修改 `main.rs` 中的 `include_str!` 路径来使用其他文本文件。

## 与 Burn 版本的区别

1. **后端**: 
   - Burn: 使用 Wgpu 后端
   - Candle: 支持 CUDA/CPU/Metal

2. **API 风格**:
   - Burn: 更高级的抽象，类似 PyTorch
   - Candle: 更底层，需要手动管理更多细节

3. **训练循环**:
   - Burn: 使用 `Learner` 和 `TrainStep`/`ValidStep` trait
   - Candle: 手动实现训练循环

4. **模型保存**:
   - Burn: 使用 `CompactRecorder`
   - Candle: 使用 SafeTensors 格式

5. **Dropout**:
   - Burn: 内置 Dropout 层
   - Candle: 需要手动实现

## 性能

- 训练速度取决于硬件和 CUDA 可用性
- 推理时使用流式生成，逐 token 输出
- 支持 GPU 加速（如果 CUDA 可用）

## 依赖

主要依赖：
- `candle-core = "0.9.1"` - 核心张量操作
- `candle-nn = "0.9.1"` - 神经网络层
- `tokio` - 异步运行时
- `async-stream` - 流式生成
- `serde` + `serde_json` - 配置序列化
- `anyhow` - 错误处理

## 许可证

与原项目相同的许可证。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 已知问题

1. 训练过程中的指标显示较为简单
2. 没有实现学习率调度器
3. 没有实现梯度裁剪
4. Dropout 实现可能不如框架内置版本高效

## 改进建议

1. 添加 TensorBoard 或其他可视化工具
2. 实现更复杂的采样策略（top-k, top-p, temperature）
3. 添加梯度累积支持更大的有效批大小
4. 实现混合精度训练
5. 添加分布式训练支持
