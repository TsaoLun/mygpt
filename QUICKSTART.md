# MyGPT Candle 版本 - 快速入门

## 🚀 快速开始

### 前置要求

- Rust 1.70 或更高版本
- (可选) CUDA 11.8+ 用于 GPU 加速

### 方法一：使用构建脚本 (推荐)

```bash
# 直接运行构建脚本
./build_candle.sh
```

这个脚本会：

1. 备份原始的 Cargo.toml
2. 使用 Candle 版本的配置
3. 构建并运行程序
4. 自动恢复原始配置

### 方法二：手动构建

```bash
# 1. 复制 Candle 配置
cp Cargo_candle.toml Cargo.toml

# 2. 构建项目
cargo build --release --bin mygpt-candle

# 3. 运行
cargo run --release --bin mygpt-candle

# 4. (可选) 恢复 Burn 配置
# 如果需要切换回 Burn 版本，恢复原始 Cargo.toml
```

### 方法三：直接指定配置文件

```bash
# 不修改 Cargo.toml，直接使用 Cargo_candle.toml
cargo build --release --bin mygpt-candle --manifest-path Cargo_candle.toml
cargo run --release --bin mygpt-candle --manifest-path Cargo_candle.toml
```

## 📁 项目结构

```
mygpt/
├── src/                    # Burn 版本 (原始)
│   ├── main.rs
│   ├── modeling/
│   ├── training/
│   └── ...
├── src_candle/            # Candle 版本 (新增)
│   ├── main.rs
│   ├── modeling/
│   ├── training/
│   └── ...
├── Cargo.toml             # Burn 配置 (原始)
├── Cargo_candle.toml      # Candle 配置 (新增)
├── artifact/              # Burn 模型输出
├── artifact_candle/       # Candle 模型输出
├── build_candle.sh        # 构建脚本
├── README_CANDLE.md       # Candle 文档
└── COMPARISON.md          # 对比文档
```

## 🎯 使用流程

### 首次运行 (训练模型)

```bash
./build_candle.sh
```

程序会：

1. 读取训练文本 (`src/4in1.txt`)
2. 创建分词器
3. 训练模型 (默认 1 个 epoch)
4. 保存模型到 `artifact_candle/`
5. 使用训练好的模型生成文本

输出示例：

```
text chars len: 2847195
vocab size: 6355
Training new model...
Creating datasets...
Train dataset size: 2562275
Test dataset size: 284698
Initializing model...

=== Epoch 1/1 ===
Train [40001/40001] Loss: 1.2345
Train Loss: 1.3456
Val [4449/4449] Loss: 1.4567
Val Loss: 1.5678
Checkpoint saved to artifact_candle/checkpoint_epoch_1.safetensors

Model saved to artifact_candle/model.safetensors

Generating text...

大师兄，你怎么了？...
```

### 后续运行 (加载已训练模型)

如果 `artifact_candle/` 目录已存在模型文件，程序会自动加载：

```bash
./build_candle.sh
```

输出：

```
text chars len: 2847195
vocab size: 6355
Loaded existing model from artifact_candle

Generating text...

大师兄，你怎么了？...
```

## ⚙️ 自定义配置

编辑 `src_candle/main.rs`：

```rust
// 修改这些常量
pub const MAX_BLOCK_SIZE: usize = 256;  // 上下文窗口大小
pub const N_EMBD: usize = 384;          // 嵌入维度
pub const N_HEADS: usize = 6;           // 注意力头数
pub const N_LAYER: usize = 6;           // Transformer 层数
pub const DROPOUT: f64 = 0.2;           // Dropout 概率
pub const NUM_EPOCHS: usize = 1;        // 训练轮数
pub const BATCH_SIZE: usize = 64;       // 批大小
pub const LEARNING_RATE: f64 = 3.0e-4;  // 学习率
```

然后重新构建：

```bash
cargo clean
./build_candle.sh
```

## 🔧 更换训练数据

### 方法一：修改代码

编辑 `src_candle/main.rs`：

```rust
// 原来
let text = include_str!("../src/4in1.txt");

// 改为
let text = include_str!("../src/my_data.txt");
```

### 方法二：使用其他文件

将你的训练数据放在项目中，例如：

- `src/红楼梦.txt` (单本书)
- `src/my_corpus.txt` (自定义语料)

然后修改 `main.rs` 中的路径。

## 📊 训练输出说明

```
=== Epoch 1/1 ===
Train [40001/40001] Loss: 1.2345  ← 训练批次进度和损失
Train Loss: 1.3456                ← 平均训练损失
Val [4449/4449] Loss: 1.4567      ← 验证批次进度和损失
Val Loss: 1.5678                  ← 平均验证损失
```

- **Train Loss**: 越低越好，表示模型在训练集上的性能
- **Val Loss**: 越低越好，表示模型的泛化能力
- 如果 Val Loss > Train Loss 太多，可能过拟合

## 🐛 常见问题

### 1. 编译错误：找不到 `candle-core`

```bash
cargo clean
cargo update
cargo build --release --bin mygpt-candle
```

### 2. 运行时错误：CUDA 不可用

程序会自动降级到 CPU：

```rust
let device = Device::cuda_if_available(0)?;
```

如果想强制使用 CPU：

```rust
let device = Device::Cpu;
```

### 3. 内存不足

减小批大小或模型大小：

```rust
pub const BATCH_SIZE: usize = 32;  // 从 64 减到 32
pub const N_EMBD: usize = 256;     // 从 384 减到 256
```

### 4. 生成的文本质量不好

- 增加训练轮数：`pub const NUM_EPOCHS: usize = 5;`
- 增加模型大小：`pub const N_EMBD: usize = 512;`
- 检查训练数据质量
- 调整学习率：`pub const LEARNING_RATE: f64 = 1.0e-4;`

### 5. 训练太慢

- 减小模型大小
- 增大批大小（如果内存足够）
- 使用 GPU：安装 CUDA
- 减少训练数据量

## 🔄 在 Burn 和 Candle 之间切换

### 切换到 Candle

```bash
./build_candle.sh
```

### 切换回 Burn

```bash
# 如果有备份
cp Cargo_burn.toml.bak Cargo.toml

# 或者手动恢复原始配置
cargo build --release
cargo run --release
```

## 📈 性能优化建议

1. **使用 GPU**：安装 CUDA 可以获得 10-50x 加速
2. **增大批大小**：在内存允许的情况下
3. **使用 release 模式**：`--release` 标志是必须的
4. **减少日志输出**：注释掉不必要的 `println!`
5. **优化数据加载**：考虑使用内存映射

## 🎨 生成参数调整

在 `src_candle/generating/mod.rs` 中可以调整：

- **Temperature**: 控制随机性（未实现，可添加）
- **Top-k**: 只从概率最高的 k 个词中采样（未实现）
- **Top-p**: 核采样（未实现）

示例实现 temperature：

```rust
// 在 softmax 之前
let logits = logits.affine(1.0 / temperature, 0.0)?;
```

## 📚 相关文档

- [README_CANDLE.md](README_CANDLE.md) - 详细说明
- [COMPARISON.md](COMPARISON.md) - Burn vs Candle 对比
- [Candle 官方文档](https://github.com/huggingface/candle)

## 💡 提示

1. 第一次训练会花费较长时间（取决于数据量和硬件）
2. 模型文件会保存在 `artifact_candle/` 目录
3. 可以随时中断程序 (Ctrl+C)，下次运行会从头开始
4. 建议在 GPU 上训练以获得更好的性能
5. 生成的文本会直接打印到终端

## 🎉 完成！

现在你已经成功设置了 Candle 版本的 MyGPT！ 尝试运行 `./build_candle.sh`
开始训练和生成吧！
