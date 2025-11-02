use anyhow::Result;
use async_stream::stream;
use candle_core::{Device, IndexOp, Tensor};
use futures_core::Stream;
use rand::distributions::{Distribution, WeightedIndex};

use crate::{modeling::BigramModel, MAX_BLOCK_SIZE};

pub fn generate<'a>(
    model: &'a BigramModel,
    prompt: Vec<i32>,
    max_new_tokens: usize,
    device: &'a Device,
) -> impl Stream<Item = i32> + 'a {
    stream! {
        let mut tokens = prompt;
        let mut rng = rand::thread_rng();

        for _ in 0..max_new_tokens {
            // Get the last block_size tokens
            let start = tokens.len().saturating_sub(MAX_BLOCK_SIZE);
            let context = &tokens[start..];

            // Convert to tensor
            let input_tensor = match create_input_tensor(context, device) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Error creating input tensor: {}", e);
                    break;
                }
            };

            // Get logits from model
            let logits = match model.forward(&input_tensor, false) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("Error in forward pass: {}", e);
                    break;
                }
            };

            // Get logits for the last position
            let last_logits = match logits.i((0, context.len() - 1)) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("Error indexing logits: {}", e);
                    break;
                }
            };

            // Apply softmax to get probabilities
            let last_logits_unsqueezed = match last_logits.unsqueeze(0) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Error unsqueezing tensor: {}", e);
                    break;
                }
            };

            let probs = match candle_nn::ops::softmax_last_dim(&last_logits_unsqueezed) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("Error computing softmax: {}", e);
                    break;
                }
            };

            // Sample from the distribution
            let probs_vec = match probs.to_vec1::<f32>() {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("Error converting probs to vec: {}", e);
                    break;
                }
            };

            let next_token = match sample_from_probs(&probs_vec, &mut rng) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Error sampling: {}", e);
                    break;
                }
            };

            yield next_token;
            tokens.push(next_token);
        }
    }
}

fn create_input_tensor(tokens: &[i32], device: &Device) -> Result<Tensor> {
    let len = tokens.len();
    let data: Vec<u32> = tokens.iter().map(|&x| x as u32).collect();
    Tensor::from_vec(data, (1, len), device).map_err(|e| anyhow::anyhow!("{}", e))
}

fn sample_from_probs(probs: &[f32], rng: &mut impl rand::Rng) -> Result<i32> {
    let dist = WeightedIndex::new(probs)
        .map_err(|e| anyhow::anyhow!("Failed to create weighted distribution: {}", e))?;

    Ok(dist.sample(rng) as i32)
}
