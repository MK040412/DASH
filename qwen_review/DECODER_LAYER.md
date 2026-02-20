# Qwen3-VL Decoder Layer — Detailed Analysis

## Source
- **File**: `transformers/models/qwen3_vl/modeling_qwen3_vl.py`
- **Classes**: `Qwen3VLTextDecoderLayer` (L476), `Qwen3VLTextAttention` (L385), `Qwen3VLTextMLP` (L460)
- **Diagram**: `decoder_layer_detail.drawio` (39KB, 132 elements)
- **Generator**: `gen_decoder_layer.py`

## Architecture: Pre-LN Transformer Block

```
Input: hidden_states (B, S, 3584)
  │
  ├── residual₁ = x
  ├── input_layernorm: RMSNorm(3584)
  ├── self_attn: GQA (28Q / 4KV heads, head_dim=128)
  │   ├── q_proj(3584→3584) → q_norm(RMSNorm(128)) → reshape(B,28,S,128)
  │   ├── k_proj(3584→512) → k_norm(RMSNorm(128)) → reshape(B,4,S,128)
  │   ├── v_proj(3584→512) → reshape(B,4,S,128)
  │   ├── M-RoPE(Q, K, cos, sin)  — 3D position (t, h, w)
  │   ├── KV Cache update
  │   ├── SDPA: softmax(QK^T / √128) · V  — GQA 7:1
  │   └── o_proj(3584→3584)
  ├── x = residual₁ ⊕ attn_output
  │
  ├── residual₂ = x
  ├── post_attention_layernorm: RMSNorm(3584)
  ├── mlp: SwiGLU
  │   ├── gate_proj(3584→18944, no bias) → SiLU
  │   ├── up_proj(3584→18944, no bias)
  │   ├── SiLU(gate) ⊙ up
  │   └── down_proj(18944→3584, no bias)
  └── x = residual₂ ⊕ mlp_output

Output: hidden_states (B, S, 3584)

[After layers 0, 1, 2: DeepStack injection]
  h[vis_mask, :] += visual_embeds_from_PatchMerger
```

## Key Design Choices

### 1. Grouped Query Attention (GQA) — 7:1 Ratio
- 28 query heads share 4 KV head groups
- Each KV head serves 7 query heads
- Reduces KV cache by 7× vs MHA
- All projections have **bias=True** (unlike many LLMs)

### 2. Per-Head Q/K Normalization
- `q_norm = RMSNorm(head_dim=128)` applied AFTER projection, BEFORE RoPE
- `k_norm = RMSNorm(head_dim=128)` same
- Stabilizes attention logits — prevents extreme values
- Applied per-head (dim=128), NOT per-token (dim=3584)

### 3. M-RoPE (Multimodal Rotary Position Embedding)
- 3D position IDs: temporal (t), height (h), width (w)
- Interleaved assignment: different sections of head_dim encode different axes
- Enables spatial-temporal awareness for multimodal inputs
- Text tokens use uniform position; image/video tokens use 3D grid positions

### 4. SwiGLU MLP
```python
output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
```
- **No bias** on gate/up/down projections (unlike attention)
- Intermediate size 18944 ≈ 5.29× hidden (close to 8/3 × hidden ≈ 9557, but larger)
- SiLU = x · σ(x), smooth approximation of ReLU

### 5. DeepStack Visual Injection
- `_deepstack_process()` called AFTER decoder layer forward, BEFORE next layer
- Only on layers 0, 1, 2 (first 3 layers)
- Operation: `h[visual_positions, :] += visual_embeds`
- Additive — does NOT replace, adds to existing hidden states
- visual_embeds come from `deepstack_merger_list[i]` (see PATCH_MERGER.md)

## Qwen3-VL-7B Configuration

| Parameter | Value |
|-----------|-------|
| hidden_size | 3584 |
| num_hidden_layers | 28 |
| num_attention_heads | 28 |
| num_key_value_heads | 4 |
| head_dim | 128 |
| intermediate_size | 18944 |
| hidden_act | silu |
| attention_bias | True |
| rms_norm_eps | 1e-6 |

## Diagram Panels

- **(a)** Overall Decoder Layer — Pre-LN block with residual connections + DeepStack
- **(b)** GQA Attention — Q/K/V parallel projections, per-head norm, M-RoPE, KV Cache, SDPA
- **(c)** SwiGLU MLP — gate_proj ‖ up_proj → SiLU ⊙ multiply → down_proj
- **(d)** DeepStack Injection — _deepstack_process() flow with mapping table
- **(e)** Config Table — 13 key parameters with notes
