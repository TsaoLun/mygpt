use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use serde::{Deserialize, Serialize};
use std::fs;

use crate::{
    dataset::TokenPairDataset, modeling::{BigramModel, BigramModelConfig}
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub model: BigramModelConfig,
    pub num_epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
}

impl TrainingConfig {
    pub fn new(
        model: BigramModelConfig,
        num_epochs: usize,
        batch_size: usize,
        learning_rate: f64,
    ) -> Self {
        Self {
            model,
            num_epochs,
            batch_size,
            learning_rate,
        }
    }

    pub fn save(&self, path: String) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    pub fn load(path: String) -> Result<Self> {
        let json = fs::read_to_string(path)?;
        let config: TrainingConfig = serde_json::from_str(&json)?;
        Ok(config)
    }
}

fn create_artifact_dir(artifact_dir: &str) {
    fs::remove_dir_all(artifact_dir).ok();
    fs::create_dir_all(artifact_dir).ok();
}

pub fn train(
    artifact_dir: &str,
    tokens: Vec<i32>,
    config: TrainingConfig,
    device: &Device,
) -> Result<()> {
    create_artifact_dir(artifact_dir);
    config.save(format!("{artifact_dir}/config.json"))?;

    println!("Creating datasets...");
    let (dataset_train, dataset_test) =
        TokenPairDataset::from_tokens(tokens, config.model.block_size);

    println!("Train dataset size: {}", dataset_train.len());
    println!("Test dataset size: {}", dataset_test.len());

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

    println!("Initializing model...");
    let model = BigramModel::new(&config.model, vb.clone())?;

    let params = ParamsAdamW {
        lr: config.learning_rate,
        ..Default::default()
    };
    let mut optimizer = AdamW::new(varmap.all_vars(), params)?;

    println!("\nStarting training...");
    for epoch in 0..config.num_epochs {
        println!("\n=== Epoch {}/{} ===", epoch + 1, config.num_epochs);

        // Training phase
        let train_loss = train_epoch(
            &model,
            &dataset_train,
            &mut optimizer,
            config.batch_size,
            device,
            true,
        )?;
        println!("Train Loss: {:.4}", train_loss);

        // Validation phase
        let val_loss = train_epoch(
            &model,
            &dataset_test,
            &mut optimizer,
            config.batch_size,
            device,
            false,
        )?;
        println!("Val Loss: {:.4}", val_loss);

        // Save checkpoint
        let checkpoint_path = format!("{}/checkpoint_epoch_{}.safetensors", artifact_dir, epoch + 1);
        varmap.save(&checkpoint_path)?;
        println!("Checkpoint saved to {}", checkpoint_path);
    }

    // Save final model
    let model_path = format!("{}/model.safetensors", artifact_dir);
    model.save(&model_path, &varmap)?;
    println!("\nModel saved to {}", model_path);

    Ok(())
}

fn train_epoch(
    model: &BigramModel,
    dataset: &TokenPairDataset,
    optimizer: &mut AdamW,
    batch_size: usize,
    device: &Device,
    is_training: bool,
) -> Result<f64> {
    let num_batches = (dataset.len() + batch_size - 1) / batch_size;
    let mut total_loss = 0.0;
    let mut num_samples = 0;

    let phase = if is_training { "Train" } else { "Val" };

    for batch_idx in 0..num_batches {
        let start_idx = batch_idx * batch_size;
        let end_idx = (start_idx + batch_size).min(dataset.len());

        let batch = dataset.get_batch(start_idx, end_idx, device)?;

        let loss = model.loss(&batch.inputs, &batch.targets, is_training)?;
        let loss_val = loss.to_vec0::<f32>()?;

        if is_training {
            optimizer.backward_step(&loss)?;
        }

        total_loss += loss_val as f64 * (end_idx - start_idx) as f64;
        num_samples += end_idx - start_idx;

        if (batch_idx + 1) % 50 == 0 || batch_idx == num_batches - 1 {
            print!(
                "\r{} [{}/{}] Loss: {:.4}",
                phase,
                batch_idx + 1,
                num_batches,
                loss_val
            );
            use std::io::Write;
            std::io::stdout().flush().ok();
        }
    }
    println!();

    Ok(total_loss / num_samples as f64)
}
