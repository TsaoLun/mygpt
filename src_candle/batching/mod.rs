use candle_core::Tensor;

#[derive(Debug, Clone)]
pub struct TokenPairBatch {
    pub inputs: Tensor,
    pub targets: Tensor,
}
