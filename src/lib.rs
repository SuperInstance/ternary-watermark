//! # ternary-watermark
//!
//! Ternary watermarking for neural model provenance.
//! Embed fingerprints in weights that survive quantization.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Watermark {
    pub owner: String,
    pub pattern: Vec<i8>,
    pub positions: Vec<usize>,
}

impl Watermark {
    pub fn new(owner: &str, pattern: Vec<i8>, positions: Vec<usize>) -> Self {
        Self { owner: owner.into(), pattern, positions }
    }

    /// Generate a deterministic watermark from owner string.
    pub fn generate(owner: &str, length: usize, seed_positions: &[usize]) -> Self {
        let mut pattern = Vec::with_capacity(length);
        let mut h: u64 = 5381;
        for b in owner.bytes() { h = h.wrapping_mul(33).wrapping_add(b as u64); }
        for i in 0..length {
            let bit = ((h.wrapping_add(i as u64)) >> (i % 8)) & 1;
            let val = if bit == 0 { 1i8 } else { -1i8 };
            pattern.push(val);
        }
        Self { owner: owner.into(), pattern, positions: seed_positions.to_vec() }
    }

    /// Embed watermark into weights at specified positions.
    pub fn embed(&self, weights: &mut [i8]) -> Result<usize, String> {
        let mut embedded = 0;
        for (i, &pos) in self.positions.iter().enumerate() {
            if pos < weights.len() && i < self.pattern.len() {
                weights[pos] = self.pattern[i];
                embedded += 1;
            }
        }
        if embedded == 0 { Err("no positions embedded".into()) } else { Ok(embedded) }
    }

    /// Detect watermark in weights.
    pub fn detect(&self, weights: &[i8]) -> f64 {
        if self.positions.is_empty() { return 0.0; }
        let matches = self.positions.iter().zip(self.pattern.iter())
            .filter(|(&pos, &pat)| pos < weights.len() && weights[pos] == pat)
            .count();
        matches as f64 / self.positions.len() as f64
    }
}

/// Watermark registry for multiple owners.
pub struct WatermarkRegistry {
    watermarks: HashMap<String, Watermark>,
}

impl WatermarkRegistry {
    pub fn new() -> Self { Self { watermarks: HashMap::new() } }

    pub fn register(&mut self, wm: Watermark) {
        self.watermarks.insert(wm.owner.clone(), wm);
    }

    pub fn verify(&self, weights: &[i8], threshold: f64) -> Vec<(String, f64)> {
        self.watermarks.iter()
            .map(|(owner, wm)| (owner.clone(), wm.detect(weights)))
            .filter(|(_, score)| *score >= threshold)
            .collect()
    }

    pub fn watermark_count(&self) -> usize { self.watermarks.len() }
}

impl Default for WatermarkRegistry { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate() {
        let wm = Watermark::generate("alice", 8, &[0, 1, 2, 3]);
        assert_eq!(wm.pattern.len(), 8);
        assert!(wm.pattern.iter().all(|&v| v == -1 || v == 1));
    }

    #[test]
    fn test_embed_and_detect() {
        let wm = Watermark::new("bob", vec![1, -1, 1], vec![0, 5, 10]);
        let mut weights = vec![0i8; 16];
        wm.embed(&mut weights).unwrap();
        assert_eq!(weights[0], 1);
        assert_eq!(weights[5], -1);
        assert_eq!(weights[10], 1);
        let score = wm.detect(&weights);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn test_partial_detect() {
        let wm = Watermark::new("carol", vec![1, -1, 1, -1], vec![0, 1, 2, 3]);
        let mut weights = vec![0i8; 8];
        weights[0] = 1; weights[1] = -1; weights[2] = 0; weights[3] = -1;
        let score = wm.detect(&weights);
        assert!((score - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_registry_verify() {
        let mut reg = WatermarkRegistry::new();
        let wm1 = Watermark::new("alice", vec![1, -1], vec![0, 1]);
        let wm2 = Watermark::new("bob", vec![-1, 1], vec![2, 3]);
        reg.register(wm1);
        reg.register(wm2);
        let mut weights = vec![0i8; 8];
        weights[0] = 1; weights[1] = -1; // alice's watermark
        let results = reg.verify(&weights, 0.9);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "alice");
    }

    #[test]
    fn test_deterministic() {
        let wm1 = Watermark::generate("test", 10, &[0, 1, 2]);
        let wm2 = Watermark::generate("test", 10, &[0, 1, 2]);
        assert_eq!(wm1.pattern, wm2.pattern);
    }

    #[test]
    fn test_different_owners() {
        let wm1 = Watermark::generate("alice", 10, &[0]);
        let wm2 = Watermark::generate("bob", 10, &[0]);
        assert_ne!(wm1.pattern, wm2.pattern);
    }
}
