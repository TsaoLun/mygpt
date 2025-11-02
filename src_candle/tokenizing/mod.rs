use std::collections::{BTreeSet, HashMap};

pub struct Tokenizer {
    stoi: HashMap<char, i32>,
    itos: HashMap<i32, char>,
}

impl Tokenizer {
    pub fn from_text(text: &str) -> Self {
        let chars = text.chars().collect::<BTreeSet<char>>();

        let stoi = chars
            .iter()
            .enumerate()
            .map(|(i, ch)| (*ch, i as i32))
            .collect::<HashMap<char, i32>>();

        let itos = chars
            .iter()
            .enumerate()
            .map(|(i, ch)| (i as i32, *ch))
            .collect::<HashMap<i32, char>>();

        Self { stoi, itos }
    }

    pub fn encode(&self, text: &str) -> Vec<i32> {
        text.chars()
            .map(|ch| *self.stoi.get(&ch).unwrap_or(&0))
            .collect()
    }

    pub fn decode(&self, tokens: &[i32]) -> String {
        tokens
            .iter()
            .filter_map(|i| self.itos.get(i))
            .collect()
    }

    pub fn vocab_size(&self) -> usize {
        self.itos.len()
    }
}
