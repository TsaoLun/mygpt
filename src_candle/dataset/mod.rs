use anyhow::Result;
use candle_core::{Device, Tensor};

use crate::batching::TokenPairBatch;

#[derive(Debug, Clone)]
pub struct TokenPair {
    pub input: Vec<i32>,
    pub target: Vec<i32>,
}

#[derive(Debug)]
pub struct TokenPairDataset {
    data: Vec<TokenPair>,
}

impl TokenPairDataset {
    pub fn from_tokens(tokens: Vec<i32>, block_size: usize) -> (Self, Self) {
        let mut train_data = tokens;

        let split_at = (train_data.len() as f64 * 0.9) as usize;
        let test_data = train_data.split_off(split_at);

        let train_dataset = Self {
            data: train_data
                .windows(block_size)
                .collect::<Vec<_>>()
                .windows(2)
                .map(|v| TokenPair {
                    input: v[0].to_vec(),
                    target: v[1].to_vec(),
                })
                .collect(),
        };

        let test_dataset = Self {
            data: test_data
                .windows(block_size)
                .collect::<Vec<_>>()
                .windows(2)
                .map(|v| TokenPair {
                    input: v[0].to_vec(),
                    target: v[1].to_vec(),
                })
                .collect(),
        };

        (train_dataset, test_dataset)
    }

    pub fn get(&self, index: usize) -> Option<&TokenPair> {
        self.data.get(index)
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn get_batch(
        &self,
        start_idx: usize,
        end_idx: usize,
        device: &Device,
    ) -> Result<TokenPairBatch> {
        let items: Vec<TokenPair> = self.data[start_idx..end_idx].to_vec();

        let batch_size = items.len();
        let seq_len = items[0].input.len();

        // Create input tensor
        let input_data: Vec<u32> = items
            .iter()
            .flat_map(|item| item.input.iter().map(|&x| x as u32))
            .collect();
        let inputs = Tensor::from_vec(input_data, (batch_size, seq_len), device)?;

        // Create target tensor
        let target_data: Vec<u32> = items
            .iter()
            .flat_map(|item| item.target.iter().map(|&x| x as u32))
            .collect();
        let targets = Tensor::from_vec(target_data, (batch_size, seq_len), device)?;

        Ok(TokenPairBatch { inputs, targets })
    }
}
