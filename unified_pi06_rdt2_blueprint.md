# Unified Blueprint: pi0.6 vs RDT2 Architecture Figures

**Purpose:** Diagram-ready specification for paper figures comparing pi0.6 (OpenPI) and RDT2 architectures.  
**Scope:** Two-panel figure showing training and inference flows for each architecture.  
**Source Files:** `pi06_module_analysis.md`, `rdt2_module_analysis.md`

---

## 1. Canonical Box List

### 1.1 Shared Primitives (Both Architectures)

| Box ID | Box Name | Source Origin | Notes |
|--------|----------|---------------|-------|
| B01 | **Observation (Images + State)** | Both | External input: multi-camera RGB + robot proprioception |
| B02 | **RMSNorm** | Both | Normalization primitive (shared symbol) |
| B03 | **Position Embedding** | Both | Sinusoidal multimodal positional encoding |
| B04 | **MLP / Feed-Forward** | Both | Transformer FFN block |
| B05 | **Integration Step (Euler)** | Both | Flow-matching denoising update |

### 1.2 pi0.6 Architecture Boxes

| Box ID | Box Name | Source File | Diagram Role |
|--------|----------|-------------|--------------|
| P01 | **SigLIP Vision Encoder** | `siglip.py` | Image -> image tokens |
| P02 | **PaliGemma LLM Embedder** | `gemma.py` | Language prompt -> embeddings |
| P03 | **Gemma Transformer (pi0)** | `gemma.py` | Main backbone (flow-matching) |
| P04 | **PaliGemma Tokenizer** | `tokenizer.py` | Text prompt tokenization |
| P05 | **State Projection** | `pi0.py` | `state_proj` - state token creation |
| P06 | **Action Input Projection** | `pi0.py` | `action_in_proj` - noisy action token creation |
| P07 | **Time Embedding (SinCos + MLP)** | `pi0.py` | Timestep conditioning |
| P08 | **Action Output Projection** | `pi0.py` | `action_out_proj` - velocity prediction |
| P09 | **Prefix Concatenation** | `pi0.py` | Merge image tokens + language embeddings |
| P10 | **Suffix Concatenation** | `pi0.py` | Merge state + action + time tokens |
| P11 | **Flow Interpolation** | `pi0.py` | Training: `x_t = t*noise + (1-t)*actions` |
| P12 | **Noise/Time Sampler** | `pi0.py` | Training-only: sample noise and timestep |
| P13 | **Velocity Target** | `pi0.py` | Training: `u_t = noise - actions` |
| P14 | **MSE Loss** | `pi0.py` | Training-only optimization target |
| P15 | **Inference Loop** | `pi0.py` | Iterative denoising (inference-only) |
| P16 | **KV Cache (Prefix)** | `pi0.py` | Cached prefix hidden states |
| P17 | **Gemma Transformer (pi0-fast)** | `gemma_fast.py` | Autoregressive FAST variant backbone |
| P18 | **FAST Tokenizer** | `tokenizer.py` | Action token mapping for pi0-fast |
| P19 | **AR Mask Builder** | `pi0_fast.py` | Autoregressive attention mask construction |

### 1.3 RDT2 Architecture Boxes

| Box ID | Box Name | Source File | Diagram Role |
|--------|----------|-------------|--------------|
| R01 | **Qwen2.5-VL (VLM)** | `rdt_inferencer.py` | Vision-language model for condition |
| R02 | **VLM KV Cache Extractor** | `rdt_inferencer.py` | Extract selected-layer KV tuples |
| R03 | **Language Attention Mask** | `rdt_inferencer.py` | Valid-token mask for VLM features |
| R04 | **Vision Encoder (Optional)** | `rdt_inferencer.py` | Dedicated image embeddings (when used) |
| R05 | **Proprio Token Builder** | `rdt_inferencer.py` | Robot state -> state_tokens |
| R06 | **Condition Adaptors** | `rdt_runner.py` | `lang_adaptor`, `img_adaptor`, `state_adaptor` |
| R07 | **Condition Packaging** | `rdt_runner.py` | `adapt_conditions` - merge condition streams |
| R08 | **RDT Backbone** | `rdt/model.py` | Core Robotics Diffusion Transformer |
| R09 | **RDT Block** | `rdt/blocks.py` | Single transformer block with cross-attn |
| R10 | **Self-Attention** | `rdt/attention.py` | Action stream self-attention |
| R11 | **Cross-Attention** | `rdt/attention.py` | Condition injection via cross-attn |
| R12 | **AdaLN Modulation** | `rdt/blocks.py` | `shift/scale/gate` from timestep+state |
| R13 | **Timestep Embedder** | `rdt/blocks.py` | Timestep -> modulation features |
| R14 | **Action Adaptor** | `rdt_runner.py` | `act_adaptor` - action width alignment |
| R15 | **Register Tokens** | `rdt/model.py` | Learned tokens prepended to action sequence |
| R16 | **Final Layer** | `rdt/blocks.py` | Hidden -> action dimension projection |
| R17 | **Flow Sample Init** | `rdt_runner.py` | Initialize noisy actions for inference |
| R18 | **Conditional Sample Loop** | `rdt_runner.py` | FM iterative integration |
| R19 | **Velocity Predictor** | `rdt_runner.py` | Model output as velocity/direction |

---

## 2. Canonical Arrow List

### 2.1 Training Arrows (pi0.6)

| Arrow ID | Source | Target | Payload | Flow Type |
|----------|--------|--------|---------|-----------|
| A01 | B01 (images) | P01 (SigLIP) | raw RGB tensors (per camera) | Data |
| A02 | P01 (SigLIP) | P09 (Prefix Concat) | image tokens | Data |
| A03 | B01 (prompt) | P04 (Tokenizer) | language text | Data |
| A04 | P04 (Tokenizer) | P02 (PaliGemma) | token IDs | Data |
| A05 | P02 (PaliGemma) | P09 (Prefix Concat) | language embeddings | Data |
| A06 | B01 (state) | P05 (State Proj) | state vector | Data |
| A07 | P05 (State Proj) | P10 (Suffix Concat) | state token | Data |
| A08 | P12 (Noise Sampler) | P11 (Flow Interp) | noise sample | Training |
| A09 | P12 (Time Sampler) | P07 (Time Embed) | scalar timestep `t` | Training |
| A10 | P12 (Time Sampler) | P11 (Flow Interp) | scalar timestep `t` | Training |
| A11 | B01 (actions) | P11 (Flow Interp) | clean action trajectory | Training |
| A12 | P11 (Flow Interp) | P06 (Action Proj) | noisy actions `x_t` | Training |
| A13 | P07 (Time Embed) | P10 (Suffix Concat) | time embedding | Training |
| A14 | P09 (Prefix Concat) | P03 (Gemma pi0) | prefix tokens | Data |
| A15 | P10 (Suffix Concat) | P03 (Gemma pi0) | suffix tokens | Data |
| A16 | P03 (Gemma) | P08 (Action Out Proj) | suffix hidden states | Data |
| A17 | P08 (Action Out Proj) | P13 (Velocity Target) | predicted velocity `v_t` | Training |
| A18 | P13 (Velocity Target) | P14 (MSE Loss) | `(v_t, u_t)` pair | Training |
| A19 | P11 (Flow Interp) | P13 (Velocity Target) | target velocity `u_t = noise - actions` | Training |

### 2.2 Inference Arrows (pi0.6)

| Arrow ID | Source | Target | Payload | Flow Type |
|----------|--------|--------|---------|-----------|
| AI01 | B01 (images) | P01 (SigLIP) | raw RGB tensors | Inference |
| AI02 | P01 (SigLIP) | P09 (Prefix Concat) | image tokens | Inference |
| AI03 | B01 (prompt) | P04 (Tokenizer) | language text | Inference |
| AI04 | P04 (Tokenizer) | P02 (PaliGemma) | token IDs | Inference |
| AI05 | P02 (PaliGemma) | P09 (Prefix Concat) | language embeddings | Inference |
| AI06 | B01 (state) | P05 (State Proj) | state vector | Inference |
| AI07 | P05 (State Proj) | P10 (Suffix Concat) | state token | Inference |
| AI08 | P09 (Prefix Concat) | P03 (Gemma) | prefix tokens | Inference |
| AI09 | P03 (Gemma) | P16 (KV Cache) | prefix hidden states + KV | Inference |
| AI10 | P16 (KV Cache) | P03 (Gemma) | cached KV (suffix forward) | Inference |
| AI11 | B01 (noisy actions) | P06 (Action Proj) | initial `x_T` | Inference |
| AI12 | P06 (Action Proj) | P10 (Suffix Concat) | action tokens | Inference |
| AI13 | AI Loop: P10 | P03 (Gemma) | suffix tokens + KV | Inference |
| AI14 | P03 (Gemma) | P08 (Action Out Proj) | hidden states | Inference |
| AI15 | P08 (Action Out Proj) | P05 (Integration) | predicted velocity `v_t` | Inference |
| AI16 | P05 (Integration) | P10 (Suffix Concat) | updated `x_t` | Inference |
| AI17 | P05 (Integration) | B01 (Action Output) | final `x_0` | Inference |

### 2.3 Training Arrows (RDT2)

| Arrow ID | Source | Target | Payload | Flow Type |
|----------|--------|--------|---------|-----------|
| R-A01 | B01 (image + instruction) | R01 (Qwen2.5-VL) | multimodal prompt | Data |
| R-A02 | R01 (VLM) | R02 (KV Extractor) | VLM hidden states | Data |
| R-A03 | R02 (KV Extractor) | R07 (Condition Pkg) | `lang_kv_cache` | Data |
| R-A04 | R02 (KV Extractor) | R07 (Condition Pkg) | `lang_attn_mask` | Data |
| R-A05 | R01 (VLM) | R04 (Vision Enc) | image features (when used) | Data |
| R-A06 | R04 (Vision Enc) | R07 (Condition Pkg) | `img_tokens` | Data |
| R-A07 | B01 (robot state) | R05 (Proprio) | state vector | Data |
| R-A08 | R05 (Proprio) | R07 (Condition Pkg) | `state_tokens` | Data |
| R-A09 | B01 (actions) | R14 (Action Adaptor) | action trajectory | Training |
| R-A10 | R14 (Action Adaptor) | R08 (RDT Backbone) | adapted action tokens | Training |
| R-A11 | R07 (Condition Pkg) | R08 (RDT Backbone) | condition dict (KV/img/state) | Training |
| R-A12 | R-A10 (action tokens) | R15 (Register Tokens) | action + register concat | Training |
| R-A13 | P12 (Time Sampler) | R13 (Timestep Emb) | timestep `t` | Training |
| R-A14 | R-A08 (state tokens) | R13 (Timestep Emb) | state context | Training |
| R-A15 | R13 (Timestep Emb) | R12 (AdaLN) | modulation features | Training |
| R-A16 | R12 (AdaLN) | R09 (RDT Block) | shift/scale/gate params | Training |
| R-A17 | R09 (RDT Block) | R16 (Final Layer) | hidden trajectory | Training |
| R-A18 | R16 (Final Layer) | R19 (Velocity Pred) | velocity / denoising direction | Training |
| R-A19 | R19 (Velocity Pred) | P14 (MSE Loss) | `(v_t, u_t)` pair | Training |

### 2.4 Inference Arrows (RDT2)

| Arrow ID | Source | Target | Payload | Flow Type |
|----------|--------|--------|---------|-----------|
| R-AI01 | B01 (image + instruction) | R01 (Qwen2.5-VL) | multimodal prompt | Inference |
| R-AI02 | R01 (VLM) | R02 (KV Extractor) | VLM hidden states | Inference |
| R-AI03 | R02 (KV Extractor) | R07 (Condition Pkg) | `lang_kv_cache` | Inference |
| R-AI04 | R02 (KV Extractor) | R07 (Condition Pkg) | `lang_attn_mask` | Inference |
| R-AI05 | B01 (robot state) | R05 (Proprio) | state vector | Inference |
| R-AI06 | R05 (Proprio) | R07 (Condition Pkg) | `state_tokens` | Inference |
| R-AI07 | R07 (Condition Pkg) | R08 (RDT Backbone) | packaged conditions | Inference |
| R-AI08 | R17 (Flow Init) | R14 (Action Adaptor) | random noise `x_T` | Inference |
| R-AI09 | R14 (Action Adaptor) | R08 (RDT Backbone) | adapted noisy actions | Inference |
| R-AI10 | R15 (Register Tokens) | R08 (RDT Backbone) | learned register tokens | Inference |
| R-AI11 | R13 (Timestep Emb) | R12 (AdaLN) | timestep embedding | Inference |
| R-AI12 | R08 (state) | R12 (AdaLN) | state context | Inference |
| R-AI13 | R12 (AdaLN) | R09 (RDT Block) | modulation params | Inference |
| R-AI14 | R09 (RDT Block) | R16 (Final Layer) | hidden states | Inference |
| R-AI15 | R16 (Final Layer) | R19 (Velocity Pred) | velocity prediction | Inference |
| R-AI16 | R19 (Velocity Pred) | R18 (Sample Loop) | `v_t` | Inference |
| R-AI17 | R18 (Sample Loop) | R14 (Action Adaptor) | updated `x_t` | Inference |
| R-AI18 | R18 (Sample Loop) | B01 (Action Output) | final denoised actions | Inference |

---

## 3. Panel Layout Specification

### 3.1 Figure Structure (Two-Panel Comparison)

```
+---------------------------+---------------------------+
|      pi0.6 (OpenPI)       |         RDT2              |
+---------------------------+---------------------------+
| [TOP] Training Flow       | [TOP] Training Flow       |
| - Prefix Lane (images)    | - Condition Construction  |
| - Suffix Lane (actions)   | - Condition Packaging     |
| - Flow-Matching Loss      | - RDT Backbone + FM Loss  |
+---------------------------+---------------------------+
| [BOTTOM] Inference Flow   | [BOTTOM] Inference Flow   |
| - Prefix KV Cache         | - Condition Extraction    |
| - Iterative Denoising     | - Flow-Matching Loop      |
| - Action Output           | - Action Output           |
+---------------------------+---------------------------+
```

### 3.2 pi0.6 Panel Internal Layout

```
+----------------------------------------------------------+
|  pi0.6 (OpenPI) - Training Flow                         |
|                                                          |
|  [Observation]----->[SigLIP]----->[Prefix Concat]       |
|      |                                        |          |
|      +----->[Tokenizer]---->[PaliGemma]----->+          |
|      |                                        |          |
|      +----->[State Proj]--------------------->+          |
|                           [Suffix Concat]----->[Gemma]   |
|                                              |          |
|  [Noise/Time]---->[Flow Interp]---->[Action Proj]        |
|                        |                                  |
|                        v                                  |
|                   [MSE Loss]<----[Velocity Target]       |
+----------------------------------------------------------+

+----------------------------------------------------------+
|  pi0.6 (OpenPI) - Inference Flow                         |
|                                                          |
|  [Observation]----->[SigLIP]----->[Prefix Concat]       |
|      |                                        |          |
|      +----->[Tokenizer]---->[PaliGemma]----->+          |
|      |                                        v          |
|      +----->[State Proj]---->[Suffix Concat]<-[KV Cache] |
|                                              |          |
|                              [Gemma]----->[Action Out]   |
|                                   ^            |          |
|                                   |            v          |
|                          +--------+--------+   |          |
|                          | Inference Loop|<--+          |
|                          +----------------+              |
|                                   |                       |
|                                   v                       |
|                          [Action Output]                 |
+----------------------------------------------------------+
```

### 3.3 RDT2 Panel Internal Layout

```
+----------------------------------------------------------+
|  RDT2 - Training Flow                                    |
|                                                          |
|  [Image+Instruction]-->[Qwen2.5-VL]                      |
|                              |                           |
|                              +---->[KV Extractor]       |
|                              |         |                |
|                              |         v                |
|                              |  [Condition Pkg]<--[Img] |
|                              |         |                |
|                              |         v                |
|  [State]----------------->[Proprio]----->+             |
|                                              |          |
|  [Actions]---->[Action Adaptor]---->[Register Tokens]   |
|                                              |          |
|                                              v          |
|  [Timestep]---->[Timestep Emb]---->[AdaLN]              |
|                                              |          |
|                                    [RDT Block Stack]    |
|                                              |          |
|                                              v          |
|                                   [MSE Loss]<-[Velocity]|
+----------------------------------------------------------+

+----------------------------------------------------------+
|  RDT2 - Inference Flow                                   |
|                                                          |
|  [Image+Instruction]-->[Qwen2.5-VL]---->[KV Extractor]  |
|                              |                |          |
|                              |                v          |
|  [State]----------------->[Proprio]---->[Condition Pkg]  |
|                                              |           |
|                                              v           |
|  [Noise Init]---->[Action Adaptor]---->[RDT Backbone]   |
|                          ^                |              |
|                          |                v              |
|                   +------+-------+  [AdaLN Modulation]  |
|                   | Sample Loop |<----[Timestep Emb]    |
|                   +-------------+                       |
|                          |                              |
|                          v                              |
|                   [Action Output]                        |
+----------------------------------------------------------+
```

---

## 4. Style Rules (Reference Image Conventions)

### 4.1 Box Styling

| Element | Style Rule |
|---------|------------|
| **Model Backbone** (Gemma, RDT) | Rounded rectangle, solid fill, bold label |
| **Input/Output** (Observation, Actions) | Sharp rectangle, dashed border |
| **Tokenizer/Projection** | Smaller rounded box, lighter fill |
| **Loss/Computation** | Diamond or hexagon shape |
| **Loops** (Inference, Sample) | Rounded container with curved arrows |
| **Condition Streams** | Parallel vertical lanes with distinct colors |

### 4.2 Arrow Styling

| Element | Style Rule |
|---------|------------|
| **Training Data Flow** | Solid arrow, black |
| **Training Loss Gradient** | Dashed arrow, red |
| **Inference Cache** | Dotted arrow with label "KV Cache" |
| **Inference Loopback** | Curved arrow returning to input |
| **Condition Injection** | Colored arrow (e.g., blue) distinct from data flow |
| **Modulation Signal** | Thin arrow with "mod" label |

### 4.3 Color Coding (Recommended)

| Stream | Color | Hex |
|--------|-------|-----|
| Image/Visual | Blue | `#4285F4` |
| Language/Text | Green | `#34A853` |
| State/Proprio | Orange | `#FBBC04` |
| Action/Trajectory | Red | `#EA4335` |
| Time/Modulation | Purple | `#9334E6` |
| Condition (RDT) | Teal | `#00ACC1` |

### 4.4 Text Labels

- Box labels: Uppercase, 10-12pt font
- Arrow payloads: Italic, 8-9pt font alongside arrow
- Panel titles: Bold, 14pt font
- Distinguish training vs inference with suffix: `(train)` / `(infer)`

---

## 5. Consistency Checks

### 5.1 Arrow-Box Validation

| Check | Validation Rule |
|-------|-----------------|
| C01 | Every arrow source must exist in Box List (B##, P##, R##) |
| C02 | Every arrow target must exist in Box List |
| C03 | Training-only boxes (Noise Sampler, MSE Loss) must not appear in inference flow |
| C04 | Inference-only boxes (KV Cache, Inference Loop) must not appear in training flow |
| C05 | All loops must have a clear entry and exit point |

### 5.2 Payload Semantic Integrity

| Check | Validation Rule |
|-------|-----------------|
| C06 | Arrow payloads must match exact semantics from source analyses |
| C07 | pi0.6: prefix = image tokens + language embeddings (bidirectional) |
| C08 | pi0.6: suffix = state token + action tokens + time embedding (causal) |
| C09 | RDT2: condition streams must show KV OR embeddings, not both without adaptor |
| C10 | RDT2: adaLN receives [timestep + state] concatenated |
| C11 | Flow-matching velocity: `v_t` predicted, `u_t = noise - actions` target |

### 5.3 Condition Stream Distinctions

| Check | Validation Rule |
|-------|-----------------|
| C12 | pi0.6: Prefix uses `ar_mask=False` (bidirectional) |
| C13 | pi0.6: Suffix uses `ar_mask=True` (causal segment) |
| C14 | RDT2: Show three separate condition arrows into packaging (KV/img/state) |
| C15 | RDT2: Cross-attention receives condition stream, not action stream |
| C16 | RDT2: AdaLN modulation arrow orthogonal to cross-attention arrow |

### 5.4 Module Presence Verification

| Check | Validation Rule |
|-------|-----------------|
| C17 | Do NOT include LoRA boxes (pi0.6 analysis mentions but marks as "callout" only) |
| C18 | Do NOT include ViT if SigLIP is the primary encoder (pi0.6 uses SigLIP) |
| C19 | Do NOT include FSQ tokenizer subgraph unless specifically showing detokenization path |
| C20 | Do NOT include both pi0 and pi0-fast in same panel unless showing variant relationship |

---

## 6. Draw.io Author Checklist

### 6.1 Pre-Draft Checklist

- [ ] Read both source analyses in full
- [ ] Confirm figure scope: training + inference for each architecture
- [ ] Select panel layout (side-by-side or top-bottom)
- [ ] Verify all box IDs are assigned from canonical list

### 6.2 Drafting Checklist

- [ ] Draw TOP panel: pi0.6 Training Flow
  - [ ] Prefix lane: images -> SigLIP -> tokenizer -> prefix concat
  - [ ] Suffix lane: state -> state_proj -> suffix concat
  - [ ] Action lane: noise/time -> flow_interp -> action_proj -> suffix concat
  - [ ] Gemma forward: prefix+suffix -> backbone -> action_out_proj
  - [ ] Loss path: velocity prediction -> MSE target
- [ ] Draw BOTTOM panel: pi0.6 Inference Flow
  - [ ] Prefix with KV cache extraction
  - [ ] Iterative loop with curved arrow back to suffix
  - [ ] Final action output
- [ ] Draw TOP panel: RDT2 Training Flow
  - [ ] VLM forward with KV extraction
  - [ ] Three condition streams (KV/img/state) into packaging
  - [ ] Action adaptor + register tokens
  - [ ] RDT backbone with self-attn + cross-attn + adaLN
  - [ ] Velocity output to loss
- [ ] Draw BOTTOM panel: RDT2 Inference Flow
  - [ ] Condition extraction (no training VLM)
  - [ ] Flow initialization
  - [ ] Sample loop with iteration arrow
  - [ ] Final action output

### 6.3 Validation Checklist

- [ ] Run all C01-C20 consistency checks
- [ ] Verify no invented modules appear (C17-C20)
- [ ] Confirm all payloads match source semantics (C06-C11)
- [ ] Check color coding applied consistently
- [ ] Verify training/inference path separation

### 6.4 Final Review Checklist

- [ ] Panel titles clearly labeled
- [ ] Training vs inference flows visually distinguishable
- [ ] Condition streams (KV/cache, image, state) explicitly shown
- [ ] Flow-matching velocity targets clearly indicated
- [ ] Arrow payloads annotated on diagram
- [ ] No orphan boxes without connections

---

## 7. Summary: Key Architectural Distinctions

| Aspect | pi0.6 (OpenPI) | RDT2 |
|--------|---------------|------|
| **Vision** | SigLIP encoder (separate) | Qwen2.5-VL (joint) |
| **Language** | PaliGemma embedder | Qwen2.5-VL |
| **Condition Form** | Prefix tokens (visible to transformer) | KV cache (cross-attention) |
| **State Handling** | Projected to token, attends with actions | Projected to token, modulates via adaLN |
| **Action Encoding** | Continuous flow-matching | Continuous flow-matching |
| **Modulation** | Time MLP in suffix path | AdaLN from [timestep + state] |
| **Inference Cache** | Prefix KV cached | Full condition cached |

---

*Blueprint generated from aggregation of `pi06_module_analysis.md` and `rdt2_module_analysis.md`. No invented modules included. All payload semantics preserved from source.*
