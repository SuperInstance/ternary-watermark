# ternary-watermark

Ternary watermarking for neural model provenance. Embed {-1,0,+1} fingerprints in weight matrices that survive quantization. Verify model ownership without full weight access.

## Why This Matters

# ternary-watermark
Ternary watermarking for neural model provenance.
Embed fingerprints in weights that survive quantization.

## The Five-Layer Stack

This crate is part of the **Oxide Stack** — a distributed GPU runtime built on five layers:

```
┌─────────────────┐
│  cudaclaw        │  Persistent GPU kernels, warp consensus, SmartCRDT
├─────────────────┤
│  cuda-oxide      │  Flux → MIR → Pliron → NVVM → PTX compiler
├─────────────────┤
│  flux-core       │  Bytecode VM + A2A agent protocol
├─────────────────┤
│  pincher         │  "Vector DB as runtime, LLM as compiler"
├─────────────────┤
│  open-parallel   │  Async runtime (tokio fork)
└─────────────────┘
```

The key insight: **ternary values {-1, 0, +1} map directly to GPU compute**. They pack 16× denser than FP32, enable XNOR+popcount matmul, and conservation laws become compile-time checks.

## Design

Every value in this crate follows **ternary algebra** (Z₃):

| Value | Meaning | GPU Analog |
|-------|---------|------------|
| +1 | Positive / Active / Healthy | Warp vote yes |
| 0 | Neutral / Pending / Balanced | Warp vote abstain |
| -1 | Negative / Failed / Overloaded | Warp vote no |

This isn't arbitrary — ternary is the natural encoding for:
1. **BitNet b1.58** (Microsoft) — ternary LLMs at 60% less power
2. **GPU warp voting** — hardware ballot returns ternary consensus
3. **Conservation laws** — {-1, 0, +1} preserves quantity

## Key Types

```rust
pub struct Watermark
pub fn new
pub fn generate
pub fn embed
pub fn detect
pub struct WatermarkRegistry
pub fn new
pub fn register
pub fn verify
pub fn watermark_count
```

## Usage

```toml
[dependencies]
ternary-watermark = "0.1.0"
```

```rust
use ternary_watermark::*;
// See src/lib.rs tests for complete working examples
```

## Testing

```bash
git clone https://github.com/SuperInstance/ternary-watermark.git
cd ternary-watermark
cargo test    # 6 tests
```

## Stats

| Metric | Value |
|--------|-------|
| Tests | 6 |
| Lines of Rust | 139 |
| Public API | 10 items |

## License

Apache-2.0
