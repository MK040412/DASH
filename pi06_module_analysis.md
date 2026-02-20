# pi0.6 (openpi) Model-Folder Deep Analysis for Diagram Construction

Scope: `qwen_review/pi06/openpi/src/openpi/models`

## 1) File-by-File Module Map

| File | Purpose | Main Symbols | Diagram Role |
|---|---|---|---|
| `model.py` | Core data contracts and base model interface | `Observation`, `BaseModelConfig`, `BaseModel`, `preprocess_observation` | Global IO adapter and preprocessing entry box |
| `pi0_config.py` | pi0/pi0.5 config and model factory | `Pi0Config` | Config box for pi0 branch |
| `pi0.py` | Flow-matching policy model (continuous action trajectory) | `Pi0`, `embed_prefix`, `embed_suffix`, `compute_loss`, `sample_actions` | Main pi0 architecture panel |
| `pi0_fast.py` | FAST autoregressive token policy variant | `Pi0FASTConfig`, `Pi0FAST`, `embed_inputs` | Alternate pi0-fast branch panel |
| `tokenizer.py` | Prompt/state/action tokenizers for pi0/fast/baselines | `PaligemmaTokenizer`, `FASTTokenizer`, `BinningTokenizer`, `FSQTokenizer` | Text/action tokenization panel |
| `gemma.py` | Multi-expert Gemma transformer used by pi0 | `Module`, `Block`, `Attention`, `RMSNorm`, `Embedder` | LLM core panel (pi0) |
| `gemma_fast.py` | Gemma implementation with decode/KV-cache loop | `Module`, `Block`, `Attention`, `RMSNorm` | LLM core panel (pi0-fast) |
| `siglip.py` | Vision encoder backbone for images | `_Module`/`Module`, `Encoder`, `Encoder1DBlock` | Vision encoder panel |
| `vit.py` | General ViT implementation | `VisionTransformer`, `Encoder`, `Encoder1DBlock` | Optional/auxiliary vision architecture reference |
| `lora.py` | LoRA adapters for einsum/FFN | `LoRAConfig`, `Einsum`, `FeedForward` | Parameter-efficient fine-tuning callout |
| `utils/fsq_tokenizer.py` | FSQ/LFQ action-tokenizer model internals | `FsqCodebook`, `FsqAttentionTokenizer`, `TokenizerEncoderDecoder`, `CrossAttentionLayer` | FSQ tokenizer subgraph |
| `*_test.py` | Validation tests for model/tokenizer/lora | various tests | Exclude from main architecture figure (optional appendix) |

## 2) pi0 Main Forward/Training Flow (Continuous Flow Matching)

### 2.1 Prefix construction (`Pi0.embed_prefix`)
- Inputs: multi-camera RGB images (`obs.images`), image validity masks (`obs.image_masks`), optional tokenized language (`obs.tokenized_prompt`).
- Image path: each image -> `self.PaliGemma.img` (SigLIP) -> image tokens.
- Language path: prompt tokens -> `self.PaliGemma.llm(..., method="embed")` -> language embeddings.
- Merge: concatenate image tokens and language embeddings into one prefix token sequence.
- Attention regime: prefix uses bidirectional-style shared mask block (`ar_mask=False` over prefix tokens).

### 2.2 Suffix construction (`Pi0.embed_suffix`)
- Inputs: state vector, noisy action trajectory `x_t`, scalar timestep `t`.
- Non-pi0.5 mode:
  - `state -> state_proj -> state token`.
  - `x_t -> action_in_proj -> action tokens`.
  - `t -> sincos posemb -> time token`, mixed with action tokens via `action_time_mlp_*`.
- pi0.5 mode:
  - Uses `time_mlp_*` and adaRMS conditioning path (`adarms_cond`) for action expert.
- Suffix attention regime: state/action are causal-segmented relative to prefix (via `ar_mask`).

### 2.3 Loss path (`Pi0.compute_loss`)
- Sample `noise` and `time`.
- Construct flow interpolation: `x_t = t * noise + (1 - t) * actions`.
- Target velocity: `u_t = noise - actions`.
- One pass through Gemma experts with combined prefix+suffix attention mask.
- Predict `v_t` from suffix outputs via `action_out_proj`.
- Optimize `MSE(v_t, u_t)`.

### 2.4 Inference path (`Pi0.sample_actions`)
- Precompute prefix forward and KV cache from image+language prefix.
- Iterative integration loop over time:
  - rebuild suffix from current `x_t` and current `t`
  - run suffix-only forward with prefix KV cache
  - predict velocity `v_t`
  - Euler-like update `x_t <- x_t + dt * v_t`.
- Output: denoised action trajectory `x_0`.

## 3) pi0-fast Main Flow (Autoregressive FAST)

- Inputs: images + tokenized prompt fields including `token_ar_mask` and `token_loss_mask`.
- `embed_inputs`: image embeddings + prompt token embeddings are concatenated.
- `compute_loss`:
  - build AR-aware attention mask,
  - next-token objective on shifted targets,
  - logits only for target suffix to reduce memory.
- `sample_actions`:
  - prefill KV cache with prefix,
  - autoregressive decode loop (argmax or temperature sampling),
  - terminate on EOS or max steps,
  - output token sequence.

## 4) Tokenizer Subsystem Notes

- `PaligemmaTokenizer`: text-only (or text+discretized-state for pi0.5) discrete prompt format.
- `FASTTokenizer`: maps action tokens into reserved PaliGemma token range; defines prefix/postfix and masks.
- `FSQTokenizer` in `tokenizer.py`: wrapper over checkpointed `utils/fsq_tokenizer.py` model for detokenization paths.

## 5) Edge List for Diagram Arrows (pi0.6)

| Source | Target | Payload |
|---|---|---|
| Observation images | SigLIP (`PaliGemma.img`) | image tensor per camera |
| SigLIP | Prefix token concat | image tokens |
| Tokenized prompt | Gemma embed | language embeddings |
| Gemma embed | Prefix token concat | language tokens |
| Observation state | `state_proj` | state token |
| Noisy actions `x_t` | `action_in_proj` | action tokens |
| Timestep `t` | `posemb_sincos` + MLP | time embedding |
| (action,time,state) suffix pieces | Suffix token concat | suffix tokens |
| Prefix+Suffix tokens + masks | Gemma experts | hidden states |
| Suffix hidden states | `action_out_proj` | velocity prediction `v_t` |
| `v_t`, `x_t`, `dt` | Integration step | updated `x_t` |
| Final `x_t` | Action output | predicted trajectory |

## 6) Box/Arrow Rules for Draw Diagram

- Separate **Prefix Path** and **Suffix/Action Path** into parallel vertical lanes.
- Use a cache arrow from prefix Gemma forward to iterative suffix loop.
- Distinguish training-only arrows (noise/time sampling, MSE target) vs inference-only arrows (while-loop integration).
- Show two model families in separate panels: `pi0` (flow-matching continuous) and `pi0-fast` (autoregressive token decoding).
- Keep tokenizer as upstream side panel feeding prompt/state/action token flows.
