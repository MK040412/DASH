# Qwen3-VL PatchMerger — Detailed Analysis

## Source
- **File**: `transformers/models/qwen3_vl/modeling_qwen3_vl.py`
- **Class**: `Qwen3VLVisionPatchMerger` (lines 93–108)
- **Diagram**: `patch_merger_detail.drawio` (53KB, 186 elements)
- **Generator**: `gen_patch_merger.py`

## Architecture Summary

The PatchMerger reduces visual token count by **4×** via spatial merge (2×2 block concatenation),
then projects to the LLM's hidden dimension.

### Two Variants

#### 1. Final Merger (`self.merger`)
- **Used at**: After VisionBlock₂₆ (last block)
- **`use_postshuffle_norm=False`**
- **Flow**: LN(1152) → view(-1, 4608) → Linear(4608→4608) → GELU → Linear(4608→3584)
- **Output**: Feeds directly into LLM token sequence

#### 2. DeepStack Merger (`deepstack_merger_list[i]`, i=0,1,2)
- **Used at**: After VisionBlocks 8, 16, 24 (intermediate features)
- **`use_postshuffle_norm=True`**
- **Flow**: view(-1, 4608) → LN(4608) → Linear(4608→4608) → GELU → Linear(4608→3584)
- **Output**: Injected into LLM Decoder layers 0, 1, 2 via DeepStack

### Key Difference: Normalization Placement

| Property | Final Merger | DeepStack Merger |
|----------|-------------|-----------------|
| Norm order | **Norm → Merge → MLP** | **Merge → Norm → MLP** |
| Norm dim | 1152 (per-token, pre-merge) | 4608 (per-merged-token, post-merge) |

**Why the difference?** DeepStack mergers operate on intermediate ViT features (blocks 8, 16, 24)
where the 4 tokens in each 2×2 block may have different magnitudes across layers.
Normalizing AFTER concatenation ensures the merged 4608-dim vector is properly scaled before the MLP.
The final merger normalizes BEFORE merge because all 27 blocks have already refined the features.

## Spatial Merge Operation

```python
# spatial_merge_size = 2
# Tokens are pre-ordered in block-major layout (2×2 groups adjacent in memory)
x = self.norm(x)        # LN(1152) per token  [Final variant]
# or:
# x.view(-1, 4608)      # Merge first          [DeepStack variant]
# x = self.norm(x)      # LN(4608)

x = x.view(-1, self.hidden_size)  # (N, 1152) → (N/4, 4608)
x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))  # MLP: 4608 → 4608 → 3584
```

**Token reordering**: Position embeddings are computed in block-major order via
`fast_pos_embed_interpolate()` with a `.permute(0, 1, 3, 2, 4, 5)` that groups
each 2×2 spatial block's tokens adjacent in memory. The `.view(-1, 4608)` then
naturally concatenates the 4 tokens per block.

## Qwen2.5-VL vs Qwen3-VL Comparison

| | Qwen2.5-VL | Qwen3-VL |
|---|---|---|
| Class | `Qwen2_5_VLPatchMerger` | `Qwen3VLVisionPatchMerger` |
| Norm | RMSNorm(1280) | LayerNorm(1152 or 4608) |
| ViT width | 1280 | 1152 |
| Merged dim | 5120 (4×1280) | 4608 (4×1152) |
| Mergers | 1 (final only) | 1 final + 3 DeepStack |
| Output dim | 3584 | 3584 |

## Diagram Panels

The `patch_merger_detail.drawio` contains 5 panels:

- **(a)** Spatial Merge 2×2 — Visual: 6×6 grid → 3×3 merged, with concatenation detail
- **(b)** Final PatchMerger — Bottom-to-top block diagram with PyTorch layer names
- **(c)** DeepStack PatchMerger — Same structure but different norm placement
- **(d)** ViT Pipeline — 27 blocks with 3 DeepStack tap points + final merger
- **(e)** Comparison Table — Qwen2.5-VL vs Qwen3-VL Final vs DeepStack (11 properties)

## Configuration (Qwen3VLVisionConfig)

```python
hidden_size = 1152          # ViT feature dimension
out_hidden_size = 3584      # LLM hidden dimension
spatial_merge_size = 2      # 2×2 spatial merge
deepstack_visual_indexes = [8, 16, 24]  # DeepStack extraction points
depth = 27                  # Total VisionBlocks
```
