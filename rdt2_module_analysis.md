# RDT2 Model-Folder Deep Analysis for Diagram Construction

Scope: RDT2 model code (official repo + fetched model files)

Primary analyzed files:
- `models/rdt_runner.py`
- `models/rdt_inferencer.py`
- `models/rdt/model.py`
- `models/rdt/blocks.py`
- `models/rdt/attention.py`
- `models/rdt/norm.py`
- `models/rdt/pos_emb.py`

## 1) File-by-File Module Map

| File | Purpose | Main Symbols | Diagram Role |
|---|---|---|---|
| `models/rdt_inferencer.py` | Inference wrapper: condition extraction + policy call | `RDTInferencer`, `encode_image_and_instruction`, `step` | Condition construction stage |
| `models/rdt_runner.py` | Training/inference orchestration around RDT core | `RDTRunner`, `adapt_conditions`, `conditional_sample`, `predict_action` | Condition packaging + FM loop panel |
| `models/rdt/model.py` | Core Robotics Diffusion Transformer | `RDT` | Backbone panel |
| `models/rdt/blocks.py` | Transformer block internals + adaLN modulation | `RDTBlock`, `FinalLayer`, `TimestepEmbedder` | Layer-level detailed inset |
| `models/rdt/attention.py` | Self-attention and cross-attention implementation | `Attention`, `CrossAttention`, `repeat_kv` | Attention mechanism inset |
| `models/rdt/norm.py` | RMSNorm | `RMSNorm` | Utility box (shared norm op) |
| `models/rdt/pos_emb.py` | 1D/ND multimodal sinusoidal position embedding | `get_multimodal_pos_embed` | Positional embedding utility box |

## 2) Condition Construction Stage (Before Injection)

From `rdt_inferencer.py`:
- Build VLM prompt with image + instruction.
- Forward pass through Qwen2.5-VL model.
- Extract selected layer `past_key_values` as `vlang_kv_cache`.
- Build language attention mask `vlang_attn_mask`.
- Optionally compute separate image embeddings when dedicated vision encoder branch is used.
- Build proprio token from robot state.

Diagram separation:
- **Condition Stream A**: VLM KV cache (`lang_kv_cache`, per selected layers).
- **Condition Stream B**: Image condition tokens (`img_tokens`) when enabled.
- **Condition Stream C**: State/proprio token (`state_tokens`).

## 3) Condition Packaging and Adaptation Stage

From `rdt_runner.py`:
- `adapt_conditions(...)` maps raw condition/action/state tokens into hidden space via adaptors.
- Lang/img adaptors may be bypassed when direct VLM features/KV are used.
- `state_adaptor` and `act_adaptor` are core path for policy width alignment.
- `_prepare_condition_inputs(...)` enforces either `lang_c` or `lang_c_kv` and packages masks.

## 4) Injection Stage in RDT Backbone

From `rdt/model.py` + `rdt/blocks.py` + `rdt/attention.py`:
- `x` (action trajectory) receives learned register tokens and action positional embeddings.
- `timestep` -> `TimestepEmbedder`, concatenated with `state_c` for modulation source.
- Each `RDTBlock` executes:
  1. self-attention on action stream,
  2. cross-attention using condition stream (`lang_c_kv` or `lang_c`, optionally `img_c` depending on schedule),
  3. feed-forward.
- AdaLN modulation (`shift/scale/gate`) controls attention/cross-attn/ffn using `[timestep + state]` context.
- Final layer maps hidden trajectory back to action dimension.

## 5) Flow-Matching Loop Stage

From `rdt_runner.py`:
- `conditional_sample(...)` initializes noisy actions (if none provided).
- Iterative integration over `num_inference_timesteps`:
  - project noisy actions -> model width,
  - predict velocity/model output,
  - update action sample with Euler-style step.
- `predict_action(...)` is inference entrypoint.

## 6) Edge List for Diagram Arrows (RDT2)

| Source | Target | Payload |
|---|---|---|
| image + instruction | VLM forward | multimodal prompt inputs |
| VLM forward | `lang_kv_cache` | selected-layer KV tuples |
| VLM forward | `lang_attn_mask` | valid-token mask |
| optional vision encoder | `img_tokens` | image condition embeddings |
| robot state | `state_tokens` | proprio token |
| noisy action | `act_adaptor` | action trajectory hidden tokens |
| (`lang_kv_cache`,`img_tokens`,`state_tokens`) | condition packaging | model condition dict |
| packaged conditions | RDT block cross-attention | key/value or condition sequence |
| timestep + state token | adaLN modulation | shift/scale/gates |
| RDT stack output | FM integration step | velocity / denoising direction |
| integration loop | action output | predicted action chunk |

## 7) Diagram Box/Arrow Guidance

- Use two major panels: **Condition Construction** and **Condition Injection + FM Loop**.
- Explicitly draw three incoming condition arrows into packaging node (KV/image/state).
- Inside RDT block, show two orthogonal arrows:
  - condition arrow into cross-attention,
  - modulation arrow into adaLN/gating path.
- Show loopback arrow in flow-matching integrator to emphasize iterative denoising.
- Label all arrows by payload type (token embeddings, KV cache, mask, noisy action, velocity).

## 8) Known Unknowns / Cautions

- Exact selected layer set and dimensions are config-dependent.
- Whether lang/img adaptors are active is config-dependent.
- Condition scheduling across layers (alternating or fixed) should be treated as implementation-config behavior.
