# Qwen3-VL as VLA Prior — Diagram Project Plan

## Objective
Create detailed TikZ diagrams showing how Qwen3-VL can be integrated as a VLA (Vision-Language-Action) prior, following the π0/π0.5 approach from Physical Intelligence.

## Architecture Research Summary

### Qwen3-VL (arxiv:2511.21631)
- **ViT Encoder**: Dynamic resolution, Window Attention (8×8 windows) + Full Attention (4 layers only)
  - PatchEmbed: 3D Conv (temporal_patch_size=2 × patch_size=14 × patch_size=14)
  - VisionBlocks: RMSNorm → Attention → RMSNorm → MLP(SwiGLU)
  - PatchMerger: spatial_merge_size² → linear projection to LLM dim
  - RoPE: 2D (height, width) rotary position embeddings for vision
- **LLM Backbone**: Qwen3 (Dense: 2B/4B/8B/32B; MoE: 30B-A3B, 235B-A22B)
  - SwiGLU activation, RMSNorm, GQA (grouped query attention)
  - Interleaved-MRoPE: full-frequency allocation over time, width, height
- **DeepStack**: Multi-level ViT features fused into LLM (NEW in Qwen3-VL)
- **Text-Timestamp Alignment**: Absolute time encoding for video temporal modeling
- Sizes from 2B (edge) to 235B MoE (cloud)

### π0 VLA (arxiv:2410.24164)
- **VLM Backbone**: PaliGemma (3B params, ViT + Gemma LLM)
- **Action Expert**: Separate 300M param transformer weights for action/state tokens
  - Mixture of Experts style: VLM weights for text/image, Action Expert for robot tokens
  - Flow matching loss on action tokens, cross-entropy on text tokens
- **Flow Matching**: Conditional flow matching for continuous action distributions
  - Action chunks: H=50 future actions at up to 50Hz
  - Linear-Gaussian path: A_τ = τ·A + (1-τ)·ε
  - Vector field: v_θ(A_τ, o_t) → matches u(A_τ|A) = A - ε
  - Inference: 10 Euler steps (δ=0.1) from noise to actions
- **Inputs**: Multiple RGB images + language command + proprioceptive state
- **Training**: Pre-training (diverse robot data + OXE) → Post-training (task-specific high-quality data)

### π0.5 VLA
- Same base as π0 with co-training on heterogeneous data
- High-level text reasoning + low-level motor commands (chain of thought)
- Discrete autoregressive for high-level, flow matching for low-level
- Co-training: robot demos, web data, verbal instructions, cross-embodiment

### Proposed Integration: Qwen3-VL → VLA
Replace PaliGemma with Qwen3-VL:
1. Qwen3-VL ViT + DeepStack → rich visual features
2. Qwen3-VL LLM backbone → semantic understanding + reasoning
3. Add Action Expert (flow matching) → continuous action generation
4. Keep VLM weights frozen or LoRA, train Action Expert from scratch
5. Benefits: stronger visual grounding (3D), agent capabilities, longer context

## Diagram List

### D1: Qwen3-VL Overall Architecture
- ViT Encoder → DeepStack → LLM Backbone → Output
- Show Interleaved-MRoPE, dynamic resolution, Window/Full attention
- From actual code: `Qwen2_5_VisionTransformerPretrainedModel` structure

### D2: Qwen3-VL ViT Encoder Detail
- PatchEmbed3D → [VisionBlock × N] → PatchMerger
- VisionBlock: RMSNorm → VisionAttention → RMSNorm → MLP(SwiGLU)
- Window vs Full attention layers (fullatt_block_indexes)
- Rotary position encoding (2D height/width)

### D3: π0 VLA Architecture
- PaliGemma VLM backbone (frozen/fine-tuned)
- Action Expert (separate weights for action tokens)
- Flow matching pipeline: noise → Euler integration → action chunks
- Input: images + language + proprioceptive state

### D4: Proposed Qwen3-VL VLA Integration
- Qwen3-VL as VLM backbone (replacing PaliGemma)
- DeepStack feeding multi-level features
- Action Expert with flow matching
- High-level text + low-level motor (π0.5 style)
- Training stages: VLM pre-train → Robot co-train → Task fine-tune

### D5: Training & Inference Pipeline
- 3-stage training: (1) VLM pre-training, (2) robot data co-training, (3) task fine-tuning
- Inference: image encoding → high-level planning → action chunk generation
- KV-cache optimization for real-time control

### D6: Flow Matching Action Generation
- Noise sampling → Euler integration steps → denoised action chunk
- Action Expert architecture detail
- β-distribution timestep sampling during training

## Agent Assignment
- **Agent 1 (Opus)**: D1 — Qwen3-VL Overall Architecture
- **Agent 2 (Opus)**: D2 — Qwen3-VL ViT Encoder Detail  
- **Agent 3 (Opus)**: D3 + D6 — π0 VLA + Flow Matching
- **Agent 4 (Opus)**: D4 + D5 — Proposed Integration + Training Pipeline

## Output
- Each diagram: standalone `.tex` file + compiled `.pdf` + `.png`
- All in `/home/perelman/.openclaw/workspace/qwen_review/`
- Final compilation into overview document
