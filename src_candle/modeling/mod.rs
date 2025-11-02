use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{linear, Linear, VarBuilder, VarMap};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BigramModelConfig {
    pub vocab_size: usize,
    pub block_size: usize,
    pub n_embd: usize,
    pub n_heads: usize,
    pub n_layer: usize,
    pub dropout: f64,
}

impl BigramModelConfig {
    pub fn new(
        vocab_size: usize,
        block_size: usize,
        n_embd: usize,
        n_heads: usize,
        n_layer: usize,
        dropout: f64,
    ) -> Self {
        Self {
            vocab_size,
            block_size,
            n_embd,
            n_heads,
            n_layer,
            dropout,
        }
    }
}

pub struct Head {
    query: Linear,
    key: Linear,
    value: Linear,
    dropout_prob: f64,
}

impl Head {
    pub fn new(n_embd: usize, head_size: usize, dropout: f64, vb: VarBuilder) -> Result<Self> {
        let query = linear(n_embd, head_size, vb.pp("query"))?;
        let key = linear(n_embd, head_size, vb.pp("key"))?;
        let value = linear(n_embd, head_size, vb.pp("value"))?;

        Ok(Self {
            query,
            key,
            value,
            dropout_prob: dropout,
        })
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let (_b, t, _c) = x.dims3()?;

        let q = self.query.forward(x)?;
        let k = self.key.forward(x)?;
        let v = self.value.forward(x)?;

        let c = k.dim(D::Minus1)?;
        let scale = (c as f64).sqrt();

        // wei = q @ k.T / sqrt(c)
        let k_t = k.transpose(1, 2)?;
        let mut wei = q.matmul(&k_t)?.affine(1.0 / scale, 0.0)?;

        // Apply causal mask
        let mask = Self::create_causal_mask(t, x.device())?;
        wei = wei.broadcast_add(&mask)?;

        // Softmax
        let wei = candle_nn::ops::softmax_last_dim(&wei)?;

        // Apply dropout during training
        let wei = if train {
            dropout(&wei, self.dropout_prob)?
        } else {
            wei
        };

        // out = wei @ v
        wei.matmul(&v).map_err(|e| anyhow::anyhow!("{}", e))
    }

    fn create_causal_mask(size: usize, device: &Device) -> Result<Tensor> {
        let mask_data: Vec<f32> = (0..size)
            .flat_map(|i| (0..size).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
            .collect();
        let tensor = Tensor::from_vec(mask_data, (size, size), device)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        tensor.unsqueeze(0).map_err(|e| anyhow::anyhow!("{}", e))
    }
}

pub struct MultiHead {
    heads: Vec<Head>,
    proj: Linear,
    dropout_prob: f64,
}

impl MultiHead {
    pub fn new(n_embd: usize, n_heads: usize, dropout: f64, vb: VarBuilder) -> Result<Self> {
        let head_size = n_embd / n_heads;
        let mut heads = Vec::new();

        for i in 0..n_heads {
            heads.push(Head::new(
                n_embd,
                head_size,
                dropout,
                vb.pp(format!("head_{}", i)),
            )?);
        }

        let proj = linear(n_embd, n_embd, vb.pp("proj"))?;

        Ok(Self {
            heads,
            proj,
            dropout_prob: dropout,
        })
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let mut outputs = Vec::new();
        for head in &self.heads {
            outputs.push(head.forward(x, train)?);
        }

        let x = Tensor::cat(&outputs, D::Minus1)?;
        let x = self.proj.forward(&x)?;

        if train {
            dropout(&x, self.dropout_prob)
        } else {
            Ok(x)
        }
    }
}

pub struct FeedForward {
    linear: Linear,
    proj: Linear,
    dropout_prob: f64,
}

impl FeedForward {
    pub fn new(n_embd: usize, dropout: f64, vb: VarBuilder) -> Result<Self> {
        let lin1 = linear(n_embd, n_embd * 4, vb.pp("linear"))?;
        let proj = linear(n_embd * 4, n_embd, vb.pp("proj"))?;

        Ok(Self {
            linear: lin1,
            proj,
            dropout_prob: dropout,
        })
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let x = self.linear.forward(x)?;
        let x = x.relu()?;
        let x = self.proj.forward(&x)?;

        if train {
            dropout(&x, self.dropout_prob)
        } else {
            Ok(x)
        }
    }
}

pub struct Block {
    sa_heads: MultiHead,
    ffwd: FeedForward,
    ln1: candle_nn::LayerNorm,
    ln2: candle_nn::LayerNorm,
}

impl Block {
    pub fn new(n_embd: usize, n_heads: usize, dropout: f64, vb: VarBuilder) -> Result<Self> {
        let sa_heads = MultiHead::new(n_embd, n_heads, dropout, vb.pp("sa_heads"))?;
        let ffwd = FeedForward::new(n_embd, dropout, vb.pp("ffwd"))?;
        let ln1 = candle_nn::layer_norm(n_embd, 1e-5, vb.pp("ln1"))?;
        let ln2 = candle_nn::layer_norm(n_embd, 1e-5, vb.pp("ln2"))?;

        Ok(Self {
            sa_heads,
            ffwd,
            ln1,
            ln2,
        })
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let x_norm = self.ln1.forward(x)?;
        let attn_out = self.sa_heads.forward(&x_norm, train)?;
        let x = (x + attn_out)?;

        let x_norm = self.ln2.forward(&x)?;
        let ff_out = self.ffwd.forward(&x_norm, train)?;
        (x + ff_out).map_err(|e| anyhow::anyhow!("{}", e))
    }
}

pub struct BigramModel {
    token_embedding_table: candle_nn::Embedding,
    position_embedding_table: candle_nn::Embedding,
    blocks: Vec<Block>,
    lm_head: Linear,
    config: BigramModelConfig,
}

impl BigramModel {
    pub fn new(config: &BigramModelConfig, vb: VarBuilder) -> Result<Self> {
        let token_embedding_table =
            candle_nn::embedding(config.vocab_size, config.n_embd, vb.pp("token_emb"))?;
        let position_embedding_table =
            candle_nn::embedding(config.block_size, config.n_embd, vb.pp("pos_emb"))?;

        let mut blocks = Vec::new();
        for i in 0..config.n_layer {
            blocks.push(Block::new(
                config.n_embd,
                config.n_heads,
                config.dropout,
                vb.pp(format!("block_{}", i)),
            )?);
        }

        let lm_head = linear(config.n_embd, config.vocab_size, vb.pp("lm_head"))?;

        Ok(Self {
            token_embedding_table,
            position_embedding_table,
            blocks,
            lm_head,
            config: config.clone(),
        })
    }

    pub fn forward(&self, idx: &Tensor, train: bool) -> Result<Tensor> {
        let (_b, t) = idx.dims2()?;
        let device = idx.device();

        let tok_emb = self.token_embedding_table.forward(idx)?;

        let pos_idx = Tensor::arange(0u32, t as u32, device)?.unsqueeze(0)?;
        let pos_emb = self.position_embedding_table.forward(&pos_idx)?;

        let mut x = tok_emb.broadcast_add(&pos_emb)?;

        for block in &self.blocks {
            x = block.forward(&x, train)?;
        }

        self.lm_head
            .forward(&x)
            .map_err(|e| anyhow::anyhow!("{}", e))
    }

    pub fn loss(&self, idx: &Tensor, targets: &Tensor, train: bool) -> Result<Tensor> {
        let logits = self.forward(idx, train)?;

        let (b, t, c) = logits.dims3()?;
        let logits = logits.reshape((b * t, c))?;
        let targets = targets.reshape(b * t)?;

        candle_nn::loss::cross_entropy(&logits, &targets).map_err(|e| anyhow::anyhow!("{}", e))
    }

    pub fn save(&self, path: &str, varmap: &VarMap) -> Result<()> {
        varmap.save(path)?;
        Ok(())
    }

    pub fn load(config: &BigramModelConfig, path: String, device: &Device) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? };
        Self::new(config, vb)
    }

    pub fn config(&self) -> &BigramModelConfig {
        &self.config
    }
}

// Helper function for dropout
fn dropout(x: &Tensor, prob: f64) -> Result<Tensor> {
    if prob == 0.0 {
        return Ok(x.clone());
    }

    let device = x.device();
    let shape = x.shape();

    // Create random mask
    let random_vals =
        Tensor::rand(0.0f32, 1.0f32, shape, device).map_err(|e| anyhow::anyhow!("{}", e))?;
    let keep_mask = random_vals.gt(prob).map_err(|e| anyhow::anyhow!("{}", e))?;

    // Scale by 1/(1-prob) to maintain expected value
    let scale = 1.0 / (1.0 - prob);
    let result = x
        .broadcast_mul(
            &keep_mask
                .to_dtype(x.dtype())
                .map_err(|e| anyhow::anyhow!("{}", e))?,
        )
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    result
        .affine(scale, 0.0)
        .map_err(|e| anyhow::anyhow!("{}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation() -> Result<()> {
        let device = Device::Cpu;
        let config = BigramModelConfig::new(100, 256, 384, 6, 6, 0.2);

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let model = BigramModel::new(&config, vb)?;

        // Test forward pass
        let input = Tensor::zeros((2, 8), DType::U32, &device)?;
        let output = model.forward(&input, false)?;

        assert_eq!(output.dims3()?, (2, 8, 100));

        Ok(())
    }
}
