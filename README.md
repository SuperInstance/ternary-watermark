# ternary-watermark

**Fingerprint neural network weights with {-1, +1} patterns that survive quantization.**

When you train a model for months on expensive hardware, you want to know if someone copied it. Traditional watermarking schemes embed real-valued perturbations in weights — but those get destroyed the moment someone quantizes to INT8 or ternary (BitNet b1.58). This crate takes the opposite approach: embed watermarks that are *already* ternary, so quantization can't remove them.

## The Insight

Quantization maps continuous weights to a small set of discrete values. If your watermark is a continuous signal mixed into the weights, quantization is a nonlinear distortion that scrambles it. But if your watermark is *made of the same discrete values the weights get quantized to*, it passes through quantization unchanged.

A ternary watermark is a pattern of {-1, +1} values placed at specific positions in a weight matrix. Since BitNet b1.58 already quantizes to {-1, 0, +1}, a watermark using {-1, +1} survives quantization by construction. The `0` positions act as a natural separator — they're not part of the watermark, so they don't interfere.

## Quick Start

```toml
[dependencies]
ternary-watermark = "0.1.0"
```

```rust
use ternary_watermark::*;

// Generate a deterministic watermark from an owner identity
let wm = Watermark::generate("research-lab-alpha", 64, &[0, 7, 14, 21, 28, 35, 42, 49]);

// Embed into a ternary weight matrix
let mut weights = vec![0i8; 1024];
let embedded = wm.embed(&mut weights)?;
assert_eq!(embedded, 8); // 8 positions written

// Later: detect the watermark
let score = wm.detect(&weights);
assert_eq!(score, 1.0); // perfect match — all 8 positions match

// Someone else's watermark won't match
let other_wm = Watermark::generate("competitor-beta", 64, &[0, 7, 14, 21, 28, 35, 42, 49]);
let other_score = other_wm.detect(&weights);
assert!(other_score < 1.0); // partial or no match
```

## Architecture

```
┌─────────────────────────────────────────────┐
│  Watermark                                   │
│  ├── owner: String                           │
│  ├── pattern: Vec<i8>    (the {-1,+1} code)  │
│  └── positions: Vec<usize> (where to embed)  │
│                                              │
│  generate() ── deterministic from owner+seed │
│  embed()    ── write pattern into weights    │
│  detect()   ── match score ∈ [0.0, 1.0]     │
├─────────────────────────────────────────────┤
│  WatermarkRegistry                           │
│  ├── HashMap<String, Watermark>              │
│  register() ── add owner → watermark         │
│  verify()   ── check all owners at once      │
└─────────────────────────────────────────────┘
```

The `Watermark` struct holds three things: who owns it, the ternary pattern (a sequence of -1 and +1 values), and the positions in the weight matrix where those values should appear. The `WatermarkRegistry` manages multiple owners and can verify which watermarks match a given set of weights.

## API Reference

### Watermark

```rust
Watermark::new(owner: &str, pattern: Vec<i8>, positions: Vec<usize>) -> Watermark
Watermark::generate(owner: &str, length: usize, seed_positions: &[usize]) -> Watermark
wm.embed(weights: &mut [i8]) -> Result<usize, String>
wm.detect(weights: &[i8]) -> f64
```

- **`new`** — create a watermark with explicit pattern and positions
- **`generate`** — deterministically derive a pattern from the owner string using a hash-based PRNG. Same owner + same length = same pattern. Different owners = different patterns.
- **`embed`** — write pattern values into the weight array at the specified positions. Returns the number of positions actually written (may be less if positions exceed array length).
- **`detect`** — returns the fraction of watermark positions that match. 1.0 = perfect match, 0.5 = half match, 0.0 = no match.

### WatermarkRegistry

```rust
WatermarkRegistry::new() -> WatermarkRegistry
reg.register(wm: Watermark)
reg.verify(weights: &[i8], threshold: f64) -> Vec<(String, f64)>
reg.watermark_count() -> usize
```

- **`register`** — add a watermark keyed by owner name
- **`verify`** — check all registered watermarks against weights. Returns owners whose detection score ≥ threshold, sorted by match quality.

## Real-World Example: Model Marketplace

```rust
use ternary_watermark::*;

// Model publisher registers their watermarks
let mut registry = WatermarkRegistry::new();
registry.register(Watermark::generate("lab-alpha", 128, &[
    0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88
]));
registry.register(Watermark::generate("lab-beta", 128, &[
    1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89
]));

// When a model is found "in the wild", check provenance
let suspect_weights: Vec<i8> = load_quantized_weights("suspect_model.ternary");
let matches = registry.verify(&suspect_weights, 0.8);

for (owner, score) in &matches {
    println!("Model belongs to {} (confidence: {:.1}%)", owner, score * 100.0);
}
```

## How Generation Works

`Watermark::generate` uses a simple hash-chain PRNG seeded from the owner string:

1. Compute a DJB2 hash of the owner name
2. For each position in the pattern, derive a bit from the hash state
3. Map bit 0 → +1, bit 1 → -1

This is deterministic: the same owner string always produces the same pattern. It's not cryptographically secure, but the goal is fingerprinting, not encryption. An adversary who doesn't know the positions can't easily remove the watermark without also degrading model quality.

## Robustness

| Attack | Survival |
|--------|----------|
| Ternary quantization | **100%** — watermark is already ternary |
| INT8 quantization | **100%** — {-1, +1} ⊂ INT8 |
| Fine-tuning (light) | **High** — positions are sparse, most survive |
| Weight pruning | **Partial** — depends on which positions get pruned |
| Adversarial scrubbing | **Low** — with known positions, adversary can overwrite |

The watermark's strength is its simplicity. It survives the most common post-training transformations by construction. Its weakness is that a determined adversary with knowledge of the positions can overwrite them — but that requires knowing where to look.

## Performance

- **Embed**: O(k) where k = number of positions
- **Detect**: O(k) — simple comparison loop
- **Verify** (registry): O(n × k) where n = number of registered watermarks
- **Memory**: O(k) per watermark

For a 1B parameter ternary model with 12 watermark positions, detection takes ~100ns.

## Ecosystem

- **ternary-shard** — distribute watermarked weights across GPU nodes
- **ternary-signal-flow** — test watermark robustness through signal processing
- **ternary-antidote** — distributed consensus on watermark ownership claims

## Open Questions

- **Position selection**: Currently manual. Automatic selection based on weight sensitivity (embed where weights matter least) would improve stealth.
- **Multi-bit capacity**: Each position carries ~1 bit. Information-theoretic bounds on capacity vs. robustness are unexplored.
- **Cryptographic binding**: Using HMAC-SHA256 instead of DJB2 for generation would make patterns unforgeable.
- **Collision resistance**: No analysis of how many watermarks can coexist before accidental matches occur.

## Stats

| Metric | Value |
|--------|-------|
| Tests | 6 |
| Lines of Rust | 139 |
| Public API | 10 items |

## License

Apache-2.0
