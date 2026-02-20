# Qwen3-VL Vision Encoder — 전체 파이프라인 상세 분석

> **Source**: `transformers/models/qwen3_vl/modeling_qwen3_vl.py`  
> **Config**: `Qwen3VLVisionConfig` (defaults: Qwen3-VL-4B-Instruct)

---

## 0. 핵심 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `depth` | 27 | Vision Transformer 블록 수 |
| `hidden_size` ($d$) | 1152 | ViT 내부 hidden dimension |
| `intermediate_size` ($d_{ff}$) | 4304 | MLP 중간 차원 |
| `num_heads` ($H$) | 16 | Multi-head attention 헤드 수 |
| `head_dim` ($d_h$) | 72 | $= d / H = 1152 / 16$ |
| `patch_size` ($P$) | 16 | 공간 패치 크기 |
| `temporal_patch_size` ($T_p$) | 2 | 시간 패치 크기 |
| `spatial_merge_size` ($M$) | 2 | PatchMerger 병합 비율 |
| `out_hidden_size` ($d_{out}$) | 3584 | LLM에 전달되는 최종 차원 |
| `in_channels` ($C$) | 3 | RGB |
| `num_position_embeddings` | 2304 | $= 48^2$, 절대 위치 임베딩 사이즈 |
| `deepstack_visual_indexes` | [8, 16, 24] | DeepStack 추출 레이어 |
| `hidden_act` | `gelu_pytorch_tanh` | MLP 활성화 함수 |

---

## 1. 입력 전처리 — PatchEmbed

**클래스**: `Qwen3VLVisionPatchEmbed`

입력 이미지/비디오를 3D 패치로 분할하고 선형 임베딩한다.

### 입력

$$\mathbf{X} \in \mathbb{R}^{T \times H \times W \times C}$$

여기서 $T$는 프레임 수 (이미지: $T=1 \to$ 패딩하여 $T_p=2$로 맞춤), $H \times W$는 해상도, $C=3$.

### 패치 분할

입력을 $(T_p, P, P) = (2, 16, 16)$ 크기 3D 패치로 분할:

$$N = \frac{T}{T_p} \times \frac{H}{P} \times \frac{W}{P}$$

### Conv3d 임베딩

$$\mathbf{E}_{\text{patch}} = \text{Conv3d}(\mathbf{X}) \in \mathbb{R}^{N \times d}$$

```python
# kernel_size = stride = [T_p, P, P] = [2, 16, 16]
self.proj = nn.Conv3d(in_channels=3, out_channels=1152,
                      kernel_size=[2, 16, 16], stride=[2, 16, 16], bias=True)
```

내부적으로 입력을 `(-1, C, T_p, P, P)`로 reshape 후 Conv3d 적용, 출력을 `(-1, d)`로 flatten:

$$\mathbf{X}_{\text{reshaped}} \in \mathbb{R}^{N \times C \times T_p \times P \times P} \xrightarrow{\text{Conv3d}} \mathbb{R}^{N \times d \times 1 \times 1 \times 1} \xrightarrow{\text{view}} \mathbb{R}^{N \times d}$$

### 예시

224×224 이미지 ($T=1$):
$$N = 1 \times \frac{224}{16} \times \frac{224}{16} = 1 \times 14 \times 14 = 196 \text{ patches}$$

> **Note**: 이미지는 $T=1$이지만 내부적으로 temporal 축을 $T_p=2$에 맞게 처리됨 (padding 또는 반복).

---

## 2. 위치 임베딩 — 이중 구조

Qwen3-VL은 **두 가지** 위치 임베딩을 동시에 사용한다:

### 2.1 절대 위치 임베딩 (Learnable)

$$\mathbf{h}^{(0)} = \mathbf{E}_{\text{patch}} + \mathbf{E}_{\text{pos}}$$

```python
self.pos_embed = nn.Embedding(num_position_embeddings=2304, hidden_size=1152)
# 2304 = 48 × 48 그리드
```

**Bilinear Interpolation** (`fast_pos_embed_interpolate`):

임의 해상도 입력을 지원하기 위해, $48 \times 48$ 그리드의 학습된 임베딩을 실제 $(H/P) \times (W/P)$ 그리드로 bilinear 보간한다:

$$\mathbf{E}_{\text{pos}}(i, j) = \sum_{k \in \{0,1\}^2} w_k \cdot \mathbf{E}_{\text{table}}[\text{floor}(i') + k_0, \text{floor}(j') + k_1]$$

여기서 $(i', j')$는 원래 그리드에서 보간된 연속 좌표, $w_k$는 bilinear 가중치.

**Spatial Merge 순서 재배치**: 보간 후 `spatial_merge_size=2`에 맞게 토큰 순서를 재배열:

$$\text{reshape}(T, H/M, M, W/M, M, d) \to \text{permute}(0, 1, 3, 2, 4, 5) \to \text{flatten}(0, 4)$$

이는 나중에 PatchMerger에서 인접 $2 \times 2$ 패치를 연속으로 배치하기 위함.

### 2.2 2D Rotary Position Embedding (RoPE)

Attention 내부의 Q, K에 적용되는 **상대 위치** 인코딩.

```python
head_dim = 1152 // 16 = 72
self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(dim=72 // 2 = 36)
```

**주파수 테이블 생성**:

$$f_k = \frac{1}{\theta^{2k/d_r}}, \quad k = 0, 1, \ldots, d_r/2 - 1$$

여기서 $\theta = 10000$, $d_r = d_h / 2 = 36$.

**2D 좌표**: 각 패치의 $(h_{\text{row}}, w_{\text{col}})$ 좌표를 계산하고, 각각 주파수 테이블로 변환:

$$\text{emb}_{\text{row}} = [\cos(h \cdot f_0), \cos(h \cdot f_1), \ldots] \in \mathbb{R}^{N \times 36}$$
$$\text{emb}_{\text{col}} = [\cos(w \cdot f_0), \cos(w \cdot f_1), \ldots] \in \mathbb{R}^{N \times 36}$$

$$\text{rotary\_emb} = [\text{emb}_{\text{row}}; \text{emb}_{\text{col}}] \in \mathbb{R}^{N \times 72}$$

이를 $(\cos, \sin)$ 형태로 변환하여 Q, K에 적용:

$$\text{position\_embeddings} = (\cos(\text{rotary\_emb}), \sin(\text{rotary\_emb}))$$

---

## 3. Vision Transformer Blocks (×27)

**클래스**: `Qwen3VLVisionBlock`

Pre-LayerNorm Transformer 구조. 총 27개 블록을 순차 실행한다.

### 3.1 전체 수식

$$\mathbf{h}^{(l)}_{\text{mid}} = \mathbf{h}^{(l)} + \text{Attn}\bigl(\text{LN}_1(\mathbf{h}^{(l)})\bigr)$$

$$\mathbf{h}^{(l+1)} = \mathbf{h}^{(l)}_{\text{mid}} + \text{MLP}\bigl(\text{LN}_2(\mathbf{h}^{(l)}_{\text{mid}})\bigr)$$

```python
def forward(self, hidden_states, cu_seqlens, position_embeddings, **kwargs):
    hidden_states = hidden_states + self.attn(
        self.norm1(hidden_states), cu_seqlens=cu_seqlens,
        position_embeddings=position_embeddings, **kwargs)
    hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
    return hidden_states
```

### 3.2 LayerNorm

$$\text{LN}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

- $\epsilon = 10^{-6}$
- `nn.LayerNorm(1152, eps=1e-6)` — learnable $\gamma, \beta$

> **Note**: Qwen3-VL의 **Vision** 부분은 LayerNorm 사용 (RMSNorm이 아님). Text 부분만 RMSNorm.

---

## 4. VisionAttention — 상세

**클래스**: `Qwen3VLVisionAttention`

### 4.1 QKV 프로젝션

단일 Linear로 Q, K, V를 한번에 생성:

$$[\mathbf{Q}; \mathbf{K}; \mathbf{V}] = \mathbf{h} \cdot \mathbf{W}_{qkv} + \mathbf{b}_{qkv}$$

$$\mathbf{W}_{qkv} \in \mathbb{R}^{d \times 3d}, \quad d = 1152$$

```python
self.qkv = nn.Linear(1152, 1152 * 3, bias=True)
```

Split & Reshape:

$$\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times H \times d_h}$$

```python
query, key, value = self.qkv(hidden_states) \
    .reshape(seq_length, 3, num_heads, head_dim) \
    .permute(1, 0, 2, 3).unbind(0)
# 각각: (N, 16, 72)
```

### 4.2 Rotary Position Embedding 적용

Q와 K에만 2D RoPE 적용 (V에는 적용하지 않음):

$$\mathbf{Q}' = \text{RoPE}(\mathbf{Q}, \cos, \sin)$$
$$\mathbf{K}' = \text{RoPE}(\mathbf{K}, \cos, \sin)$$

RoPE 수식:

$$\text{RoPE}(\mathbf{x}, \cos, \sin) = \mathbf{x} \odot \cos + \text{rotate\_half}(\mathbf{x}) \odot \sin$$

여기서:

$$\text{rotate\_half}([x_1, x_2, \ldots, x_{d/2}, x_{d/2+1}, \ldots, x_d]) = [-x_{d/2+1}, \ldots, -x_d, x_1, \ldots, x_{d/2}]$$

### 4.3 Window Attention (cu_seqlens)

**핵심**: Qwen3-VL Vision은 **가변 길이 시퀀스**를 배치로 처리한다. 각 이미지/비디오 프레임의 패치들은 독립적인 attention window를 형성한다.

`cu_seqlens` (cumulative sequence lengths)로 시퀀스 경계를 정의:

$$\text{cu\_seqlens} = [0, n_1, n_1 + n_2, \ldots, \sum_i n_i]$$

여기서 $n_i = H_i \times W_i$ (각 프레임의 패치 수).

**Flash Attention 2** 사용 시:

$$\text{Attn}_i = \text{softmax}\left(\frac{\mathbf{Q}_i \mathbf{K}_i^T}{\sqrt{d_h}}\right) \mathbf{V}_i, \quad \text{for each window } i$$

**Eager mode** (FA2 미사용):

```python
# 각 window를 개별적으로 split 후 attention 계산
lengths = cu_seqlens[1:] - cu_seqlens[:-1]
splits = [torch.split(tensor, lengths.tolist(), dim=2)
          for tensor in (Q, K, V)]
attn_outputs = [attention(q, k, v) for q, k, v in zip(*splits)]
attn_output = torch.cat(attn_outputs, dim=1)
```

### 4.4 Scaled Dot-Product Attention

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_h}}\right) \mathbf{V}$$

- Scaling: $\alpha = d_h^{-0.5} = 72^{-0.5} \approx 0.1179$
- **Non-causal** (`is_causal=False`): Vision은 양방향 attention
- No attention mask (window 내부 전체 attend)

### 4.5 출력 프로젝션

$$\mathbf{o} = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) \cdot \mathbf{W}_{\text{proj}}$$

```python
self.proj = nn.Linear(1152, 1152)  # bias=False (default)
```

Reshape: `(N, H, d_h) → (N, d)`:

```python
attn_output = attn_output.reshape(seq_length, -1).contiguous()
attn_output = self.proj(attn_output)
```

---

## 5. VisionMLP — 상세

**클래스**: `Qwen3VLVisionMLP`

**표준 2-layer MLP** (SwiGLU가 아님):

$$\text{MLP}(\mathbf{x}) = \mathbf{W}_2 \cdot \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2$$

여기서 $\sigma = \text{GELU}_{\text{tanh}}$ (`gelu_pytorch_tanh`).

$$\text{GELU}_{\text{tanh}}(x) = 0.5 \cdot x \cdot \left(1 + \tanh\left[\sqrt{2/\pi}\left(x + 0.044715 x^3\right)\right]\right)$$

```python
self.linear_fc1 = nn.Linear(1152, 4304, bias=True)   # d → d_ff
self.act_fn = gelu_pytorch_tanh
self.linear_fc2 = nn.Linear(4304, 1152, bias=True)   # d_ff → d
```

차원 흐름:

$$\mathbb{R}^{N \times 1152} \xrightarrow{W_1} \mathbb{R}^{N \times 4304} \xrightarrow{\sigma} \mathbb{R}^{N \times 4304} \xrightarrow{W_2} \mathbb{R}^{N \times 1152}$$

> **Note**: Vision MLP는 **bias=True**. Text MLP는 bias=False이고 SwiGLU 사용. Vision ≠ Text 구조.

---

## 6. PatchMerger — 공간 토큰 축소

**클래스**: `Qwen3VLVisionPatchMerger`

ViT 출력 토큰 수를 $1/M^2 = 1/4$로 줄여서 LLM에 전달한다.

### 6.1 Final Merger (use_postshuffle_norm=False)

순서:

$$\mathbf{x}_{\text{norm}} = \text{LN}(\mathbf{x}) \quad \text{(per-token, dim=1152)}$$

$$\mathbf{x}_{\text{merged}} = \text{Concat}(\mathbf{x}_{\text{norm}}[i:i+4]) \quad \text{(2×2 인접 패치 병합)}$$

$$\mathbf{x}_{\text{merged}} \in \mathbb{R}^{N/4 \times 4608} \quad (4608 = 1152 \times 2^2)$$

$$\mathbf{y} = \mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \cdot \mathbf{x}_{\text{merged}} + \mathbf{b}_1) + \mathbf{b}_2$$

```python
# Final merger
self.norm = nn.LayerNorm(1152, eps=1e-6)     # norm BEFORE merge
self.linear_fc1 = nn.Linear(4608, 4608)      # hidden_size * M²
self.act_fn = nn.GELU()
self.linear_fc2 = nn.Linear(4608, 3584)      # → out_hidden_size
```

차원 흐름:

$$\mathbb{R}^{N \times 1152} \xrightarrow{\text{LN}} \mathbb{R}^{N \times 1152} \xrightarrow{\text{merge 2×2}} \mathbb{R}^{N/4 \times 4608} \xrightarrow{W_1} \mathbb{R}^{N/4 \times 4608} \xrightarrow{\text{GELU}} \xrightarrow{W_2} \mathbb{R}^{N/4 \times 3584}$$

### 6.2 DeepStack Merger (use_postshuffle_norm=True)

DeepStack용 PatchMerger는 **norm 순서가 다르다**:

```python
# DeepStack merger (postshuffle)
self.norm = nn.LayerNorm(4608, eps=1e-6)     # norm AFTER merge (dim=4608)
```

$$\mathbf{x}_{\text{merged}} = \text{Concat}(\mathbf{x}[i:i+4]) \in \mathbb{R}^{N/4 \times 4608}$$
$$\mathbf{x}_{\text{norm}} = \text{LN}(\mathbf{x}_{\text{merged}}) \quad \text{(dim=4608)}$$
$$\mathbf{y} = \mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \cdot \mathbf{x}_{\text{norm}})$$

> **차이점**: Final merger는 merge 전에 LN (dim=1152), DeepStack merger는 merge 후에 LN (dim=4608).

### 6.3 "Merge" 연산의 실체

코드상 "merge"는 실제로 `.view(-1, hidden_size)`로 구현됨:

```python
def forward(self, x):
    x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm
                  else x).view(-1, self.hidden_size)
    x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
    return x
```

**핵심**: 토큰이 이미 `fast_pos_embed_interpolate`에서 spatial merge 순서로 재배열되어 있으므로, 연속 $M^2 = 4$개 토큰을 concat하면 자연스럽게 $2 \times 2$ 공간 병합이 된다.

---

## 7. DeepStack — 멀티스케일 비전 피처 주입

**Qwen3-VL의 핵심 혁신.** ViT 중간 레이어의 출력을 LLM 디코더의 초기 레이어에 직접 주입한다.

### 7.1 피처 추출

블록 순회 중 지정된 레이어 (`[8, 16, 24]`)에서 중간 특성을 추출:

```python
deepstack_feature_lists = []
for layer_num, blk in enumerate(self.blocks):
    hidden_states = blk(hidden_states, ...)
    if layer_num in [8, 16, 24]:
        feat = self.deepstack_merger_list[idx](hidden_states)
        deepstack_feature_lists.append(feat)
```

각 DeepStack 피처:

$$\mathbf{F}^{(k)} = \text{PatchMerger}_k(\mathbf{h}^{(l_k)}) \in \mathbb{R}^{N/4 \times 3584}$$

여기서 $l_k \in \{8, 16, 24\}$, $k = 0, 1, 2$.

### 7.2 LLM 디코더 주입

Vision Encoder의 **return**:

```python
return hidden_states_merged, deepstack_feature_lists
# hidden_states_merged: (N/4, 3584) — final visual tokens
# deepstack_feature_lists: [3 × (N/4, 3584)] — multi-scale features
```

LLM Text Decoder의 초기 레이어에서 (별도 코드):

$$\mathbf{h}_{\text{LLM}}^{(l)}[\text{vis\_pos}] \mathrel{+}= \mathbf{F}^{(k)}$$

즉, 비전 토큰 위치의 hidden state에 **가산** (additive injection).

### 7.3 직관

| DeepStack 레이어 | 추출 시점 | 의미 |
|-----------------|---------|------|
| Block 8 (초기) | 27블록 중 ~30% | 저수준 시각 특성 (edge, texture) |
| Block 16 (중기) | 27블록 중 ~60% | 중수준 의미 (object parts, patterns) |
| Block 24 (후기) | 27블록 중 ~90% | 고수준 의미 (object, scene semantics) |

LLM의 **초기 레이어**에 이를 주입함으로써:
- LLM이 텍스트 생성 초반부터 풍부한 시각 정보를 활용
- 단일 레이어가 아닌 **멀티스케일** 정보로 시각적 기반(grounding) 강화

---

## 8. 전체 Forward Pass 요약

```
Input: X ∈ ℝ^{T×H×W×3}, grid_thw ∈ ℝ^{B×3}
```

### Step 1: Patch Embedding
$$\mathbf{E} = \text{Conv3d}(\mathbf{X}) \in \mathbb{R}^{N \times 1152}$$

### Step 2: Absolute Position Embedding
$$\mathbf{h}^{(0)} = \mathbf{E} + \text{BilinearInterp}(\mathbf{E}_{\text{pos}}) \in \mathbb{R}^{N \times 1152}$$

### Step 3: Rotary Position Embedding 준비
$$\text{rot} = \text{2D\_RoPE}(\text{grid\_thw}) \in \mathbb{R}^{N \times 72}$$
$$(\cos\_\text{emb}, \sin\_\text{emb}) = (\cos([\text{rot}; \text{rot}]),\ \sin([\text{rot}; \text{rot}]))$$

### Step 4: cu_seqlens 계산
$$\text{cu\_seqlens} = \text{pad}(\text{cumsum}(H_i \times W_i \text{ repeated } T_i \text{ times}), (1, 0))$$

### Step 5: Vision Transformer (×27)
$$\text{for } l = 0, 1, \ldots, 26:$$
$$\quad \mathbf{h}^{(l+1)} = \mathbf{h}^{(l)} + \text{Attn}(\text{LN}(\mathbf{h}^{(l)}), \text{cu\_seqlens}, \text{RoPE})$$
$$\quad \mathbf{h}^{(l+1)} = \mathbf{h}^{(l+1)} + \text{MLP}(\text{LN}(\mathbf{h}^{(l+1)}))$$
$$\quad \text{if } l \in \{8, 16, 24\}: \quad \mathbf{F}^{(k)} = \text{PatchMerger}_k(\mathbf{h}^{(l+1)})$$

### Step 6: Final PatchMerger
$$\mathbf{v} = \text{PatchMerger}(\mathbf{h}^{(27)}) \in \mathbb{R}^{N/4 \times 3584}$$

### 최종 출력
$$\text{return} \quad (\mathbf{v},\ [\mathbf{F}^{(0)}, \mathbf{F}^{(1)}, \mathbf{F}^{(2)}])$$

- $\mathbf{v}$: **visual_tokens** — LLM 입력의 vision token으로 사용
- $\mathbf{F}^{(k)}$: **deepstack_features** — LLM 초기 레이어에 주입

---

## 9. 차원 흐름 총정리

```
Image (1, 224, 224, 3)
  ↓ PatchEmbed Conv3d[2,16,16]
(196, 1152)                          ← N = 14×14 = 196
  ↓ + AbsPosEmbed (bilinear interp)
(196, 1152)
  ↓ Vision Transformer ×27 blocks
  │   ├─ LN(1152) → Attn(1152→1152, 16 heads, RoPE) → +residual
  │   └─ LN(1152) → MLP(1152→4304→1152, GELU_tanh) → +residual
  │
  │   [Block 8]  → DeepStack Merger → (49, 3584)
  │   [Block 16] → DeepStack Merger → (49, 3584)
  │   [Block 24] → DeepStack Merger → (49, 3584)
  ↓
(196, 1152)
  ↓ PatchMerger: LN → merge(2×2) → FC(4608→4608) → GELU → FC(4608→3584)
(49, 3584)                           ← N/4 = 196/4 = 49

Output:
  visual_tokens:        (49, 3584)   → LLM input tokens
  deepstack_features:   3 × (49, 3584) → LLM early layer injection
```

---

## 10. Qwen2.5-VL과의 주요 차이점

| 항목 | Qwen2.5-VL | Qwen3-VL |
|------|-----------|----------|
| **DeepStack** | ❌ 없음 | ✅ 레이어 [8,16,24]에서 추출 |
| **PatchMerger** | 1개 (final only) | 4개 (1 final + 3 DeepStack) |
| **Postshuffle Norm** | N/A | DeepStack merger에서 사용 |
| **Position Embedding** | `rotary_pos_emb` only | `pos_embed` (abs) + `rotary_pos_emb` (2D RoPE) |
| **Abs Pos Interp** | 없음 | Bilinear interpolation ($48^2$ grid) |
| **Patch Size** | 14 | 16 |
| **hidden_act** | `quick_gelu` | `gelu_pytorch_tanh` |
| **Depth** | 32 | 27 |
| **LLM 연결** | visual_tokens만 | visual_tokens + deepstack_features |

---

*Generated from `transformers==4.x` source code analysis. Config: `Qwen3VLVisionConfig` defaults.*
