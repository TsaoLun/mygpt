# Candle 版本对比文档

## Burn vs Candle 实现对比

### 1. 核心概念映射

| Burn              | Candle                     | 说明       |
| ----------------- | -------------------------- | ---------- |
| `Backend` trait   | `Device`                   | 设备抽象   |
| `Tensor<B, D, K>` | `Tensor`                   | 张量类型   |
| `Module` trait    | 手动实现                   | 模块抽象   |
| `Config` trait    | `serde`                    | 配置序列化 |
| `Autodiff<B>`     | `VarMap` + `backward_step` | 自动微分   |
| `CompactRecorder` | `safetensors`              | 模型保存   |

### 2. 层实现对比

#### Linear Layer

```rust
// Burn
let linear = LinearConfig::new(in_features, out_features).init(device);
let output = linear.forward(input);

// Candle
let linear = candle_nn::linear(in_features, out_features, vb)?;
let output = linear.forward(&input)?;
```

#### Embedding Layer

```rust
// Burn
let embedding = EmbeddingConfig::new(vocab_size, embed_dim).init(device);
let output = embedding.forward(input);

// Candle
let embedding = candle_nn::embedding(vocab_size, embed_dim, vb)?;
let output = embedding.forward(&input)?;
```

#### LayerNorm

```rust
// Burn
let ln = LayerNormConfig::new(normalized_shape).init(device);
let output = ln.forward(input);

// Candle
let ln = candle_nn::layer_norm(normalized_shape, eps, vb)?;
let output = ln.forward(&input)?;
```

#### Dropout

```rust
// Burn
let dropout = DropoutConfig::new(prob).init();
let output = dropout.forward(input);

// Candle (手动实现)
fn dropout(x: &Tensor, prob: f64) -> Result<Tensor> {
    let random = Tensor::rand(0.0, 1.0, x.shape(), x.device())?;
    let mask = random.gt(prob)?;
    x.broadcast_mul(&mask)?.affine(1.0/(1.0-prob), 0.0)
}
```

### 3. 训练循环对比

#### Burn 方式

```rust
impl<B: AutodiffBackend> TrainStep<Batch<B>, Output<B>> for Model<B> {
    fn step(&self, batch: Batch<B>) -> TrainOutput<Output<B>> {
        let output = self.forward_classification(batch.inputs, batch.targets);
        TrainOutput::new(self, output.loss.backward(), output)
    }
}

let learner = LearnerBuilder::new(artifact_dir)
    .metric_train_numeric(AccuracyMetric::new())
    .build(model, optimizer, learning_rate);
    
let model = learner.fit(train_loader, val_loader);
```

#### Candle 方式

```rust
let mut optimizer = AdamW::new(varmap.all_vars(), params)?;

for epoch in 0..num_epochs {
    for batch in batches {
        let loss = model.loss(&batch.inputs, &batch.targets, true)?;
        optimizer.backward_step(&loss)?;
    }
}

varmap.save("model.safetensors")?;
```

### 4. 张量操作对比

#### 形状操作

```rust
// Burn
let x = x.reshape([b * t, c]);
let x = x.repeat_dim(0, batch_size);

// Candle
let x = x.reshape((b * t, c))?;
let x = x.repeat((batch_size, 1))?;
```

#### 索引操作

```rust
// Burn
let x = x.slice([0..1, t-1..t]);

// Candle
let x = x.i((0, t-1))?;  // 单个元素
let x = x.narrow(1, t-1, 1)?;  // 切片
```

#### 掩码操作

```rust
// Burn
let mask = Tensor::tril_mask([b, t, t], 0, device);
let x = x.mask_fill(mask, f32::NEG_INFINITY);

// Candle (手动实现)
let mask_data: Vec<f32> = (0..size)
    .flat_map(|i| (0..size).map(move |j| 
        if j > i { f32::NEG_INFINITY } else { 0.0 }
    ))
    .collect();
let mask = Tensor::from_vec(mask_data, (size, size), device)?;
let x = x.broadcast_add(&mask)?;
```

### 5. 模型保存和加载

#### Burn

```rust
// 保存
model.save_file(path, &CompactRecorder::new())?;

// 加载
let record = CompactRecorder::new()
    .load(path.into(), device)?;
let model = config.init::<B>(device).load_record(record);
```

#### Candle

```rust
// 保存
varmap.save("model.safetensors")?;

// 加载
let vb = unsafe { 
    VarBuilder::from_mmaped_safetensors(
        &["model.safetensors"], 
        DType::F32, 
        device
    )? 
};
let model = Model::new(&config, vb)?;
```

### 6. 错误处理

#### Burn

- 使用 `expect()` 和 `unwrap()` 较多
- 部分操作是 infallible 的

#### Candle

- 所有操作都返回 `Result`
- 使用 `anyhow` 进行错误传播
- 需要显式处理所有错误

### 7. 性能考虑

| 方面       | Burn          | Candle     |
| ---------- | ------------- | ---------- |
| 编译速度   | 较慢          | 较快       |
| 运行时性能 | 优秀          | 优秀       |
| 内存使用   | 自动优化      | 需手动注意 |
| GPU 支持   | Wgpu (跨平台) | CUDA/Metal |
| 易用性     | 高            | 中         |

### 8. 主要挑战

从 Burn 迁移到 Candle 的主要挑战：

1. **Dropout 实现**: Candle 没有内置 Dropout 层，需要手动实现
2. **训练循环**: 需要手动实现整个训练循环，包括前向、后向传播
3. **批处理**: 需要手动管理批次数据的创建和传输
4. **错误处理**: 所有操作都可能失败，需要仔细处理 `Result`
5. **类型系统**: Candle 的类型系统不如 Burn 严格，容易出现运行时错误
6. **形状推断**: 需要手动跟踪张量形状

### 9. 优势

Candle 相对于 Burn 的优势：

1. **更轻量**: 依赖更少，编译更快
2. **更底层**: 更接近硬件，可以做更多优化
3. **SafeTensors**: 原生支持 SafeTensors 格式
4. **互操作性**: 更容易与其他 Rust 库集成
5. **社区**: HuggingFace 支持，生态更成熟

### 10. 使用建议

**选择 Burn 如果**:

- 需要快速原型开发
- 喜欢高级抽象
- 需要跨平台 GPU 支持
- 刚开始学习深度学习

**选择 Candle 如果**:

- 需要更好的性能控制
- 需要与 HuggingFace 生态集成
- 喜欢底层控制
- 有深度学习框架经验

### 11. 迁移清单

- [x] 模型定义 (modeling/mod.rs)
- [x] 训练循环 (training/mod.rs)
- [x] 文本生成 (generating/mod.rs)
- [x] 分词器 (tokenizing/mod.rs)
- [x] 数据集 (dataset/mod.rs)
- [x] 批处理 (batching/mod.rs)
- [x] 主程序 (main.rs)
- [x] Dropout 实现
- [x] 配置序列化
- [x] 模型保存/加载
- [x] 错误处理
- [x] 文档

### 12. 代码行数对比

| 模块       | Burn   | Candle | 差异 |
| ---------- | ------ | ------ | ---- |
| modeling   | ~280行 | ~350行 | +25% |
| training   | ~80行  | ~140行 | +75% |
| generating | ~40行  | ~70行  | +75% |
| 总计       | ~500行 | ~700行 | +40% |

Candle 版本代码量增加主要原因：

- 需要手动实现 Dropout
- 需要手动实现训练循环
- 需要更多错误处理代码
- 需要手动管理批处理逻辑
