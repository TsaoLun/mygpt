use anyhow::Result;
use candle_core::Device;
use futures_util::{pin_mut, stream::StreamExt};

use crate::{
    generating::generate,
    modeling::{BigramModel, BigramModelConfig},
    tokenizing::Tokenizer,
    training::{train, TrainingConfig},
};

pub mod batching;
pub mod dataset;
pub mod generating;
pub mod modeling;
pub mod tokenizing;
pub mod training;

pub const MAX_BLOCK_SIZE: usize = 256;
pub const N_EMBD: usize = 384;
pub const N_HEADS: usize = 6;
pub const N_LAYER: usize = 6;
pub const DROPOUT: f64 = 0.2;
pub const NUM_EPOCHS: usize = 1;
pub const BATCH_SIZE: usize = 64;
pub const LEARNING_RATE: f64 = 3.0e-4;

#[tokio::main]
async fn main() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    let artifact_dir = "artifact_candle";

    let text = include_str!("../src/4in1.txt");
    println!("text chars len: {}", text.chars().count());

    let tokenizer = Tokenizer::from_text(text);
    println!("vocab size: {}", tokenizer.vocab_size());

    let model = if let Ok(config) = TrainingConfig::load(format!("{artifact_dir}/config.json")) {
        let model = BigramModel::load(
            &config.model,
            format!("{artifact_dir}/model.safetensors"),
            &device,
        )?;
        println!("Loaded existing model from {}", artifact_dir);
        model
    } else {
        println!("Training new model...");
        let config = TrainingConfig::new(
            BigramModelConfig::new(
                tokenizer.vocab_size(),
                MAX_BLOCK_SIZE,
                N_EMBD,
                N_HEADS,
                N_LAYER,
                DROPOUT,
            ),
            NUM_EPOCHS,
            BATCH_SIZE,
            LEARNING_RATE,
        );

        train(artifact_dir, tokenizer.encode(&text), config, &device)?;

        let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))?;
        BigramModel::load(
            &config.model,
            format!("{artifact_dir}/model.safetensors"),
            &device,
        )?
    };

    println!("\nGenerating text...\n");
    let stream = generate(&model, tokenizer.encode("大师兄"), 10000, &device);
    pin_mut!(stream);

    while let Some(token) = stream.next().await {
        print!("{}", tokenizer.decode(&[token]));
    }
    println!();

    Ok(())
}
