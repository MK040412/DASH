#!/usr/bin/env python3
"""Qwen3-VL Vision Encoder — EXTREMELY DETAILED draw.io diagram.

Every operation, tensor shape, formula, and code-level detail is shown.
Paper-figure quality. Every element editable in draw.io.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom

# Palette
C_IN  = "#F5F5F5";  C_CONV = "#E3F2FD"; C_POS  = "#F3E5F5"
C_ROPE= "#EDE7F6";  C_ATTN = "#E3F2FD"; C_NORM = "#FFFDE7"
C_MLP = "#FFF3E0";  C_MERGE= "#E8F5E9"; C_DS   = "#FCE4EC"
C_RES = "#ECEFF1";  C_OUT  = "#E0F2F1"; C_W    = "#FFFFFF"
C_FORMULA = "#FFF8E1"

S_D = "#37474F"; S_B = "#1565C0"; S_P = "#6A1B9A"; S_O = "#E65100"
S_G = "#2E7D32"; S_R = "#C62828"; S_GR = "#9E9E9E"; S_Y = "#F9A825"


class D:
    def __init__(self, name, pw=1600, ph=4400):
        self.name, self.pw, self.ph = name, pw, ph
        self.cells = []; self._id = 2
    def _n(self):
        i = self._id; self._id += 1; return str(i)

    def box(self, x, y, w, h, label, fill=C_W, stroke=S_D, fs=10, bold=False, r=True):
        cid = self._n()
        s = (f"rounded={'1' if r else '0'};whiteSpace=wrap;html=1;fillColor={fill};"
             f"strokeColor={stroke};fontSize={fs};fontStyle={'1' if bold else '0'};arcSize=8;")
        self.cells.append(dict(id=cid,value=label,style=s,vertex=True,x=x,y=y,w=w,h=h))
        return cid

    def lbl(self, x, y, w, h, text, fs=9, color="#616161", align="center", bold=False):
        cid = self._n()
        s = (f"text;html=1;align={align};verticalAlign=middle;resizable=1;points=[];"
             f"autosize=0;strokeColor=none;fillColor=none;fontSize={fs};fontColor={color};"
             f"fontStyle={'1' if bold else '0'};")
        self.cells.append(dict(id=cid,value=text,style=s,vertex=True,x=x,y=y,w=w,h=h))
        return cid

    def grp(self, x, y, w, h, label, fill="#FAFAFA", stroke=S_GR, dash=True, fs=11):
        cid = self._n()
        s = (f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
             f"dashed={'1' if dash else '0'};dashPattern=8 4;verticalAlign=top;fontSize={fs};"
             f"fontStyle=1;fontColor=#424242;opacity=60;arcSize=6;")
        self.cells.append(dict(id=cid,value=label,style=s,vertex=True,x=x,y=y,w=w,h=h))
        return cid

    def arr(self, s, t, label="", dash=False, color=S_D, sw=1.5):
        cid = self._n()
        dd = "dashed=1;dashPattern=6 3;" if dash else ""
        st = (f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;html=1;{dd}"
              f"strokeColor={color};fontSize=9;fontColor=#616161;endArrow=blockThin;"
              f"endFill=1;strokeWidth={sw};")
        self.cells.append(dict(id=cid,value=label,style=st,edge=True,source=s,target=t))
        return cid

    def to_xml(self):
        root = ET.Element("mxfile", host="app.diagrams.net")
        diag = ET.SubElement(root, "diagram", name=self.name, id="ve2")
        m = ET.SubElement(diag, "mxGraphModel",
            dx="1422",dy="900",grid="1",gridSize="10",guides="1",tooltips="1",
            connect="1",arrows="1",fold="1",page="1",pageScale="1",
            pageWidth=str(self.pw),pageHeight=str(self.ph))
        rc = ET.SubElement(m, "root")
        ET.SubElement(rc, "mxCell", id="0")
        ET.SubElement(rc, "mxCell", id="1", parent="0")
        for c in self.cells:
            a = {"id":c["id"],"style":c["style"],"parent":"1"}
            if c.get("value"): a["value"]=c["value"]
            if c.get("vertex"): a["vertex"]="1"
            if c.get("edge"):
                a["edge"]="1"
                if c.get("source"): a["source"]=c["source"]
                if c.get("target"): a["target"]=c["target"]
            cl = ET.SubElement(rc, "mxCell", **a)
            if c.get("vertex") and "x" in c:
                ET.SubElement(cl, "mxGeometry",
                    x=str(c["x"]),y=str(c["y"]),width=str(c["w"]),height=str(c["h"]),
                    **{"as":"geometry"})
            elif c.get("edge"):
                ET.SubElement(cl, "mxGeometry", relative="1", **{"as":"geometry"})
        return minidom.parseString(ET.tostring(root, encoding="unicode")).toprettyxml(indent="  ")


def build():
    d = D("Qwen3-VL Vision Encoder (Detailed)", pw=1600, ph=4800)

    # Layout constants
    LX = 80       # left column X (main flow)
    MX = 350      # main column X
    RX = 780      # right column X (expanded details)
    W = 300       # main box width
    RW = 420      # right detail width
    sp = 8        # vertical spacing

    y = 20
    # ═══════════════════════════════════════════════
    # TITLE
    # ═══════════════════════════════════════════════
    d.lbl(100, y, 1000, 30, "<b>Qwen3-VL Vision Encoder</b> — Full Architecture Detail", fs=18, color="#212121")
    y += 30
    d.lbl(100, y, 1000, 20,
        "Source: <code>transformers/models/qwen3_vl/modeling_qwen3_vl.py</code> → class <code>Qwen3VLVisionModel</code>",
        fs=10, color="#757575")

    # ═══════════════════════════════════════════════
    # SECTION 1: INPUT
    # ═══════════════════════════════════════════════
    y += 40
    d.lbl(LX, y, 200, 18, "① Input", fs=12, color=S_B, bold=True, align="left")
    y += 22

    inp = d.box(MX, y, W, 40,
        "<b>pixel_values</b>", fill=C_IN, stroke=S_GR, fs=12, bold=True)
    d.lbl(MX+W+10, y+5, 250, 30,
        "shape: <b>(batch_pixels, C·T_p·P·P)</b><br>"
        "where C=3, T_p=2, P=16 (patch_size)",
        fs=9, color="#757575", align="left")

    # ═══════════════════════════════════════════════
    # SECTION 2: PATCH EMBED
    # ═══════════════════════════════════════════════
    y += 60
    d.lbl(LX, y, 200, 18, "② PatchEmbed", fs=12, color=S_B, bold=True, align="left")
    d.lbl(LX, y+16, 250, 14, "class Qwen3VLVisionPatchEmbed", fs=8, color="#9E9E9E", align="left")
    y += 35

    pe_grp = d.grp(MX-20, y, W+50, 200, "", fill="#EBF5FB", stroke=S_B)

    step1 = d.box(MX, y+15, W, 38,
        "<b>Step 1: Reshape input</b><br>"
        "<code>view(-1, C=3, T_p=2, P=16, P=16)</code>",
        fill=C_W, stroke=S_B, fs=9)
    d.arr(inp, step1)
    d.lbl(MX+W+10, y+15, 200, 38,
        "Unflatten pixels into<br>3D patches per temporal_patch_size",
        fs=8, color="#757575", align="left")

    step2 = d.box(MX, y+65, W, 55,
        "<b>Step 2: Conv3d projection</b><br>"
        "<code>nn.Conv3d(</code><br>"
        "<code>  in=3, out=1152,</code><br>"
        "<code>  kernel=[2,16,16], stride=[2,16,16], bias=True)</code>",
        fill=C_CONV, stroke=S_B, fs=9)
    d.arr(step1, step2)
    d.lbl(MX+W+10, y+70, 220, 45,
        "<b>Each 3D patch → 1 token</b><br>"
        "kernel = stride (non-overlapping)<br>"
        "Output: (batch_pixels, 1152, 1, 1, 1)",
        fs=8, color=S_B, align="left")

    step3 = d.box(MX, y+135, W, 32,
        "<b>Step 3: Flatten</b>  <code>view(-1, 1152)</code>",
        fill=C_W, stroke=S_B, fs=9)
    d.arr(step2, step3)
    d.lbl(MX+W+10, y+137, 180, 28,
        "Output: <b>(N, 1152)</b><br>"
        "N = total patch tokens", fs=9, color=S_B, align="left")

    # ═══════════════════════════════════════════════
    # SECTION 3: ABSOLUTE POSITION EMBEDDING
    # ═══════════════════════════════════════════════
    y += 250
    d.lbl(LX, y, 300, 18, "③ Absolute Position Embedding", fs=12, color=S_P, bold=True, align="left")
    d.lbl(LX, y+16, 300, 14, "fast_pos_embed_interpolate()", fs=8, color="#9E9E9E", align="left")
    y += 35

    pos_grp = d.grp(MX-20, y, W+50, 210, "", fill="#F8F0FC", stroke=S_P)

    pos_emb = d.box(MX, y+15, W, 40,
        "<b>nn.Embedding(2304, 1152)</b><br>"
        "Learnable 2D grid: 48×48 = 2304 positions",
        fill=C_POS, stroke=S_P, fs=9)
    d.arr(step3, pos_emb)

    pos_interp = d.box(MX, y+68, W, 65,
        "<b>Bilinear Interpolation</b><br>"
        "For any input resolution (H, W):<br>"
        "① Compute fractional grid indices<br>"
        "② 4-point bilinear weights (floor/ceil)<br>"
        "③ pos = Σ<sub>i</sub> w<sub>i</sub> · Embedding[idx<sub>i</sub>]",
        fill=C_W, stroke=S_P, fs=9)
    d.arr(pos_emb, pos_interp)
    d.lbl(MX+W+10, y+70, 250, 60,
        "<b>Resolution-agnostic</b><br>"
        "Unlike Qwen2.5-VL (RoPE only),<br>"
        "Qwen3-VL adds absolute position<br>"
        "on TOP of rotary encoding.<br>"
        "Handles any H,W via interpolation.",
        fs=8, color=S_P, align="left")

    pos_add = d.box(MX, y+148, W, 35,
        "<b>hidden_states = hidden_states + pos_embeds</b>",
        fill=C_POS, stroke=S_P, fs=10, bold=True)
    d.arr(pos_interp, pos_add)
    d.lbl(MX+W+10, y+150, 150, 28,
        "Element-wise addition<br>Output: <b>(N, 1152)</b>", fs=9, color="#757575", align="left")

    # ═══════════════════════════════════════════════
    # SECTION 4: 2D ROTARY POSITION EMBEDDING
    # ═══════════════════════════════════════════════
    y += 260
    d.lbl(LX, y, 300, 18, "④ 2D Rotary Position Embedding", fs=12, color=S_P, bold=True, align="left")
    d.lbl(LX, y+16, 300, 14, "class Qwen3VLVisionRotaryEmbedding + rot_pos_emb()", fs=8, color="#9E9E9E", align="left")
    y += 35

    rope_grp = d.grp(MX-20, y, W+50, 250, "", fill="#F0EBF8", stroke=S_P)

    rope_freq = d.box(MX, y+15, W, 45,
        "<b>Inverse frequency table</b><br>"
        "<code>inv_freq = 1/(θ^(2i/d))</code><br>"
        "θ=10000, dim=head_dim//2=36",
        fill=C_ROPE, stroke=S_P, fs=9)

    rope_pos = d.box(MX, y+72, W, 55,
        "<b>Position ID computation</b><br>"
        "For each image/video in batch:<br>"
        "① Compute 2D grid: (row_idx, col_idx)<br>"
        "② Respect spatial_merge_size=2 ordering<br>"
        "③ Repeat for num_frames if video",
        fill=C_W, stroke=S_P, fs=9)
    d.arr(rope_freq, rope_pos)

    rope_embed = d.box(MX, y+140, W, 40,
        "<b>Lookup + concat</b><br>"
        "<code>freqs = freq_table[pos_ids]  # (N, 2, 36)</code><br>"
        "<code>emb = cat(freqs, freqs) → (N, 72)</code>",
        fill=C_ROPE, stroke=S_P, fs=9)
    d.arr(rope_pos, rope_embed)

    rope_cossin = d.box(MX, y+195, W, 30,
        "<b>cos, sin = emb.cos(), emb.sin()</b>",
        fill=C_ROPE, stroke=S_P, fs=10, bold=True)
    d.arr(rope_embed, rope_cossin)
    d.lbl(MX+W+10, y+195, 200, 28,
        "Output: <b>(N, 72)</b> each<br>"
        "Passed to VisionAttention", fs=9, color=S_P, align="left")

    # ═══════════════════════════════════════════════
    # SECTION 5: cu_seqlens
    # ═══════════════════════════════════════════════
    y += 300
    d.lbl(LX, y, 300, 18, "⑤ cu_seqlens (Window Boundaries)", fs=12, color=S_D, bold=True, align="left")
    y += 22
    cuseq = d.box(MX, y, W, 55,
        "<b>Cumulative sequence lengths</b><br>"
        "<code>cu_seqlens = (grid_h × grid_w).repeat(grid_t).cumsum()</code><br>"
        "<code>cu_seqlens = F.pad(cu_seqlens, (1,0), value=0)</code><br>"
        "dtype: int32 (FlashAttention2 requirement)",
        fill=C_IN, stroke=S_GR, fs=9)
    d.lbl(MX+W+10, y+5, 250, 50,
        "Defines window boundaries<br>"
        "for variable-length attention.<br>"
        "Each window = one spatial frame<br>"
        "(H/P × W/P tokens per window)",
        fs=8, color="#616161", align="left")

    # ═══════════════════════════════════════════════
    # SECTION 6: VISION BLOCK × 27
    # ═══════════════════════════════════════════════
    y += 80
    d.lbl(LX, y, 300, 18, "⑥ VisionBlock × 27 (Main Loop)", fs=12, color=S_B, bold=True, align="left")
    d.lbl(LX, y+16, 350, 14, "class Qwen3VLVisionBlock — Pre-Norm Residual Transformer Block", fs=8, color="#9E9E9E", align="left")
    y += 38

    vb_grp = d.grp(MX-40, y, W+100+RW+20, 1250, "for layer_num in range(27):", fill="#F5F7FA", stroke=S_B, fs=10)

    by = y + 30

    # --- 6a: LayerNorm 1 ---
    norm1 = d.box(MX, by, W, 40,
        "<b>norm1 = LayerNorm(1152, eps=1e-6)</b><br>"
        "<code>normalized = (x - μ) / √(σ² + ε) · γ + β</code>",
        fill=C_NORM, stroke=S_Y, fs=9)
    d.arr(pos_add, norm1, label="(N, 1152)")
    d.lbl(MX+W+10, by+5, 200, 30,
        "Pre-normalization<br>"
        "Learnable γ, β ∈ ℝ<sup>1152</sup>",
        fs=8, color="#616161", align="left")

    # --- 6b: VisionAttention (EXPANDED) ---
    by += 55
    attn_grp = d.grp(MX-20, by, W+50+RW, 550,
        "VisionAttention (class Qwen3VLVisionAttention)", fill="#E8EAF6", stroke=S_B, fs=10)

    aqy = by + 30

    # QKV projection
    qkv = d.box(MX, aqy, W, 45,
        "<b>QKV = self.qkv(x)</b><br>"
        "<code>nn.Linear(1152 → 3456, bias=True)</code><br>"
        "Combined Q, K, V in single projection",
        fill=C_ATTN, stroke=S_B, fs=9)
    d.arr(norm1, qkv)
    d.lbl(MX+W+10, aqy+5, 200, 40,
        "3456 = 1152 × 3<br>"
        "= 3 × (num_heads × head_dim)<br>"
        "= 3 × (16 × 72)", fs=8, color=S_B, align="left")

    # Reshape + split
    aqy += 60
    reshape_qkv = d.box(MX, aqy, W, 45,
        "<b>Reshape + Unbind</b><br>"
        "<code>reshape(N, 3, 16, 72)</code><br>"
        "<code>permute(1, 0, 2, 3).unbind(0) → Q, K, V</code>",
        fill=C_W, stroke=S_B, fs=9)
    d.arr(qkv, reshape_qkv)

    # Q, K, V boxes
    aqy += 60
    q = d.box(MX, aqy, 85, 40,
        "<b>Q</b><br>[N, 16, 72]", fill="#BBDEFB", stroke=S_B, fs=9)
    k = d.box(MX+105, aqy, 85, 40,
        "<b>K</b><br>[N, 16, 72]", fill="#BBDEFB", stroke=S_B, fs=9)
    v = d.box(MX+210, aqy, 85, 40,
        "<b>V</b><br>[N, 16, 72]", fill="#BBDEFB", stroke=S_B, fs=9)
    d.arr(reshape_qkv, q)
    d.arr(reshape_qkv, k)
    d.arr(reshape_qkv, v)

    # RoPE application
    aqy += 55
    rope_apply = d.box(MX, aqy, W, 65,
        "<b>apply_rotary_pos_emb_vision(Q, K, cos, sin)</b><br><br>"
        "<code>Q' = Q·cos + rotate_half(Q)·sin</code><br>"
        "<code>K' = K·cos + rotate_half(K)·sin</code><br>"
        "<code>rotate_half([x1,x2]) = [-x2, x1]</code>",
        fill=C_ROPE, stroke=S_P, fs=9)
    d.arr(q, rope_apply)
    d.arr(k, rope_apply)
    d.arr(rope_cossin, rope_apply, label="cos, sin", dash=True, color=S_P)
    d.lbl(MX+W+10, aqy+5, 200, 60,
        "<b>V is NOT rotated</b><br>"
        "Only Q, K get positional info.<br>"
        "2D spatial encoding:<br>"
        "(height, width) positions<br>"
        "No temporal dim in vision RoPE",
        fs=8, color=S_P, align="left")

    # Transpose for attention
    aqy += 80
    transpose = d.box(MX, aqy, W, 38,
        "<b>Transpose for batch attention</b><br>"
        "<code>Q',K',V: transpose(0,1).unsqueeze(0) → [1, 16, N, 72]</code>",
        fill=C_W, stroke=S_B, fs=9)
    d.arr(rope_apply, transpose)
    d.arr(v, transpose)

    # Window Attention
    aqy += 55
    win_attn = d.box(MX, aqy, W, 80,
        "<b>Window Attention (via cu_seqlens)</b><br><br>"
        "<b>FlashAttention2 path:</b><br>"
        "<code>flash_attn(Q', K', V', cu_seqlens, max_seqlen)</code><br><br>"
        "<b>Eager path:</b><br>"
        "Split by cu_seqlens → per-window SDPA → concat",
        fill=C_ATTN, stroke=S_B, fs=9)
    d.arr(transpose, win_attn)
    d.arr(cuseq, win_attn, label="cu_seqlens", dash=True, color=S_GR)
    d.lbl(MX+W+10, aqy+5, 250, 75,
        "<b>is_causal = False</b><br>"
        "(bidirectional, not autoregressive)<br><br>"
        "attention_dropout = 0.0<br>"
        "scaling = head_dim<sup>-0.5</sup> = 72<sup>-0.5</sup><br><br>"
        "Each window attends independently<br>"
        "(no cross-window interaction)",
        fs=8, color=S_B, align="left")

    # Output projection
    aqy += 95
    o_proj = d.box(MX, aqy, W, 38,
        "<b>Output Projection</b><br>"
        "<code>reshape(N, -1) → self.proj: Linear(1152→1152)</code>",
        fill=C_ATTN, stroke=S_B, fs=9)
    d.arr(win_attn, o_proj)
    d.lbl(MX+W+10, aqy+5, 180, 30,
        "No bias on output proj<br>"
        "Output: <b>(N, 1152)</b>", fs=8, color=S_B, align="left")

    # --- 6c: Residual 1 ---
    aqy += 55
    res1 = d.box(MX, aqy, W, 38,
        "<b>Residual Connection 1</b><br>"
        "<code>hidden_states = hidden_states + attn(norm1(hidden_states))</code>",
        fill=C_RES, stroke=S_GR, fs=9)
    d.arr(o_proj, res1)
    d.lbl(MX+W+10, aqy+8, 180, 22,
        "Pre-norm residual pattern", fs=8, color="#616161", align="left")

    # --- 6d: LayerNorm 2 ---
    aqy += 52
    norm2 = d.box(MX, aqy, W, 40,
        "<b>norm2 = LayerNorm(1152, eps=1e-6)</b><br>"
        "<code>normalized = (x - μ) / √(σ² + ε) · γ + β</code>",
        fill=C_NORM, stroke=S_Y, fs=9)
    d.arr(res1, norm2)

    # --- 6e: VisionMLP (EXPANDED) ---
    aqy += 55
    mlp_grp = d.grp(MX-20, aqy, W+50, 230,
        "VisionMLP (class Qwen3VLVisionMLP) — Simple 2-layer, NOT SwiGLU",
        fill="#FFF8E1", stroke=S_O, fs=9)

    mqy = aqy + 30
    fc1 = d.box(MX, mqy, W, 42,
        "<b>linear_fc1 = Linear(1152 → 4304, bias=True)</b><br>"
        "<code>h = self.linear_fc1(x)</code>",
        fill=C_MLP, stroke=S_O, fs=9)
    d.arr(norm2, fc1)
    d.lbl(MX+W+10, mqy+5, 220, 35,
        "intermediate_size = 4304<br>"
        "Expansion ratio ≈ 3.73×<br>"
        "(not 4× like Qwen2.5-VL's 4608)", fs=8, color=S_O, align="left")

    mqy += 55
    act = d.box(MX, mqy, W, 42,
        "<b>act_fn: GELU (gelu_pytorch_tanh)</b><br>"
        "<code>GELU(x) = x · Φ(x) ≈ 0.5x(1+tanh(√(2/π)(x+0.044715x³)))</code>",
        fill=C_FORMULA, stroke=S_O, fs=9)
    d.arr(fc1, act)
    d.lbl(MX+W+10, mqy+5, 200, 35,
        "Tanh approximation variant<br>"
        "Same as GPT-2 / LLaMA<br>"
        "Output: (N, 4304)", fs=8, color=S_O, align="left")

    mqy += 55
    fc2 = d.box(MX, mqy, W, 42,
        "<b>linear_fc2 = Linear(4304 → 1152, bias=True)</b><br>"
        "<code>output = self.linear_fc2(h)</code>",
        fill=C_MLP, stroke=S_O, fs=9)
    d.arr(act, fc2)
    d.lbl(MX+W+10, mqy+8, 220, 28,
        "Compress back to hidden_size<br>"
        "Output: <b>(N, 1152)</b>", fs=8, color=S_O, align="left")

    # --- 6f: Residual 2 ---
    mqy += 58
    res2 = d.box(MX, mqy, W, 38,
        "<b>Residual Connection 2</b><br>"
        "<code>hidden_states = hidden_states + mlp(norm2(hidden_states))</code>",
        fill=C_RES, stroke=S_GR, fs=9)
    d.arr(fc2, res2)

    # --- 6g: DeepStack Branch ---
    mqy += 55
    ds_grp = d.grp(MX-20, mqy, W+50+RW, 200,
        "⑦ DeepStack Feature Extraction (NEW in Qwen3-VL)", fill="#FCE4EC", stroke=S_R, fs=10)

    dsy = mqy + 28
    ds_check = d.box(MX, dsy, W, 35,
        "<b>if layer_num in [8, 16, 24]:</b>",
        fill=C_W, stroke=S_R, fs=10, bold=True)
    d.arr(res2, ds_check)

    dsy += 48
    ds_merger = d.box(MX, dsy, W, 75,
        "<b>DeepStack PatchMerger</b> (per-level, postshuffle_norm=True)<br><br>"
        "① <code>norm = LayerNorm(4608, eps=1e-6)</code>  ← post-shuffle<br>"
        "② <code>view(-1, 1152×4) → norm</code>  (2×2 spatial merge)<br>"
        "③ <code>Linear(4608→4608) → GELU → Linear(4608→3584)</code>",
        fill="#FFCDD2", stroke=S_R, fs=9)
    d.arr(ds_check, ds_merger)

    d.lbl(MX+W+10, dsy, 380, 75,
        "<b>3 separate PatchMerger instances</b><br>"
        "(one per deepstack_visual_indexes entry)<br><br>"
        "Key difference vs final PatchMerger:<br>"
        "<b>use_postshuffle_norm=True</b><br>"
        "→ LayerNorm on (4608) after spatial merge<br>"
        "→ Final merger uses pre-shuffle norm on (1152)<br><br>"
        "Output per level: <b>(N/4, 3584)</b><br>"
        "→ Collected into deepstack_feature_lists[]",
        fs=8, color=S_R, align="left")

    dsy += 85
    ds_out = d.box(MX, dsy, W, 30,
        "<code>deepstack_feature_lists.append(feature)</code>",
        fill="#FFCDD2", stroke=S_R, fs=9)
    d.arr(ds_merger, ds_out)

    # ═══════════════════════════════════════════════
    # SECTION 7: FINAL PATCH MERGER
    # ═══════════════════════════════════════════════
    block_end_y = mqy + 220
    y = block_end_y + 20
    d.lbl(LX, y, 300, 18, "⑧ Final PatchMerger", fs=12, color=S_G, bold=True, align="left")
    d.lbl(LX, y+16, 350, 14,
        "class Qwen3VLVisionPatchMerger (use_postshuffle_norm=False)", fs=8, color="#9E9E9E", align="left")
    y += 38

    pm_grp = d.grp(MX-20, y, W+50, 290, "", fill="#E8F5E9", stroke=S_G)

    pmy = y + 15
    pm_norm = d.box(MX, pmy, W, 42,
        "<b>LayerNorm (pre-shuffle)</b><br>"
        "<code>norm = LayerNorm(1152, eps=1e-6)</code><br>"
        "Applied BEFORE spatial merge (unlike DeepStack)",
        fill=C_NORM, stroke=S_Y, fs=9)
    d.arr(res2, pm_norm, label="after all 27 blocks: (N, 1152)")

    pmy += 55
    pm_reshape = d.box(MX, pmy, W, 42,
        "<b>2×2 Spatial Merge</b><br>"
        "<code>view(-1, 1152 × 4) = view(-1, 4608)</code><br>"
        "Groups 4 adjacent tokens into 1",
        fill=C_W, stroke=S_G, fs=9)
    d.arr(pm_norm, pm_reshape)
    d.lbl(MX+W+10, pmy+5, 200, 35,
        "spatial_merge_size = 2<br>"
        "Token count: N → N/4<br>"
        "Hidden dim: 1152 → 4608", fs=8, color=S_G, align="left")

    pmy += 55
    pm_fc1 = d.box(MX, pmy, W, 32,
        "<b>linear_fc1 = Linear(4608 → 4608)</b>",
        fill=C_MERGE, stroke=S_G, fs=10)
    d.arr(pm_reshape, pm_fc1)

    pmy += 42
    pm_gelu = d.box(MX, pmy, W, 28,
        "<b>GELU()</b>  <code>nn.GELU()</code>",
        fill="#C8E6C9", stroke=S_G, fs=10)
    d.arr(pm_fc1, pm_gelu)

    pmy += 38
    pm_fc2 = d.box(MX, pmy, W, 35,
        "<b>linear_fc2 = Linear(4608 → 3584)</b><br>"
        "out_hidden_size = d<sub>LLM</sub>",
        fill=C_MERGE, stroke=S_G, fs=10)
    d.arr(pm_gelu, pm_fc2)
    d.lbl(MX+W+10, pmy+5, 200, 28,
        "Projects to LLM dimension<br>"
        "Output: <b>(N/4, 3584)</b>", fs=9, color=S_G, align="left")

    # ═══════════════════════════════════════════════
    # SECTION 8: OUTPUTS
    # ═══════════════════════════════════════════════
    y = pmy + 60
    d.lbl(LX, y, 200, 18, "⑨ Outputs", fs=12, color=S_G, bold=True, align="left")
    y += 25

    out1 = d.box(MX, y, W, 42,
        "<b>visual_tokens (hidden_states)</b><br>"
        "shape: <b>(N/4, 3584)</b>",
        fill=C_OUT, stroke=S_G, fs=11, bold=True)
    d.arr(pm_fc2, out1)
    d.lbl(MX+W+10, y+5, 260, 35,
        "→ Inserted at &lt;|vision|&gt; token positions<br>"
        "   in LLM input embeddings<br>"
        "→ Replaces text embeddings at vision slots",
        fs=9, color="#616161", align="left")

    y += 55
    out2 = d.box(MX, y, W, 50,
        "<b>deepstack_feature_lists</b><br>"
        "List of 3 tensors, each <b>(N/4, 3584)</b><br>"
        "from layers [8, 16, 24]",
        fill="#FFCDD2", stroke=S_R, fs=10, bold=True)
    d.lbl(MX+W+10, y+5, 280, 45,
        "→ Injected into early LLM decoder layers:<br>"
        "   <code>hidden_states[vis_pos] += deepstack_feat[i]</code><br>"
        "→ Provides multi-level visual features to LLM<br>"
        "→ Fine-grained detail preserved across depths",
        fs=9, color=S_R, align="left")

    # ═══════════════════════════════════════════════
    # CONFIG + COMPARISON TABLE
    # ═══════════════════════════════════════════════
    y += 80
    d.box(MX-20, y, 350, 220,
        "<b>Qwen3VLVisionConfig</b><br>"
        "<hr size='1'>"
        "<div style='text-align:left;padding:5px;font-size:9px;'>"
        "depth = 27<br>"
        "hidden_size = 1152<br>"
        "intermediate_size = 4304<br>"
        "num_heads = 16<br>"
        "head_dim = 72 (= 1152/16)<br>"
        "in_channels = 3<br>"
        "patch_size = 16<br>"
        "temporal_patch_size = 2<br>"
        "spatial_merge_size = 2<br>"
        "out_hidden_size = 3584 (= d<sub>LLM</sub>)<br>"
        "num_position_embeddings = 2304 (48²)<br>"
        "deepstack_visual_indexes = [8, 16, 24]<br>"
        "hidden_act = gelu_pytorch_tanh<br>"
        "initializer_range = 0.02"
        "</div>",
        fill="#FAFAFA", stroke=S_GR, fs=10)

    d.box(MX+360, y, 350, 220,
        "<b>Qwen3-VL vs Qwen2.5-VL (Vision)</b><br>"
        "<hr size='1'>"
        "<div style='text-align:left;padding:5px;font-size:9px;'>"
        "✦ <b>DeepStack</b>: NEW — multi-level feat injection<br>"
        "✦ <b>Norm</b>: LayerNorm (was RMSNorm)<br>"
        "✦ <b>MLP</b>: 2-linear fc1→act→fc2 (was SwiGLU 3-linear)<br>"
        "✦ <b>MLP size</b>: 4304 (was 4608)<br>"
        "✦ <b>Position</b>: Abs embed + RoPE (was RoPE only)<br>"
        "✦ <b>Conv3d</b>: bias=True (was False)<br>"
        "✦ <b>patch_size</b>: 16 (was 14)<br>"
        "✦ <b>depth</b>: 27 (was 32)<br>"
        "✦ <b>PatchMerger norm</b>: pre vs post shuffle variants<br>"
        "✦ <b>Abs pos embed</b>: learnable 2D grid with interp<br>"
        "✦ <b>Hidden act</b>: gelu_pytorch_tanh (was quick_gelu)"
        "</div>",
        fill="#FFF9C4", stroke=S_Y, fs=10)

    return d


if __name__ == "__main__":
    import os
    path = "/home/perelman/.openclaw/workspace/qwen_review/d2_vision_encoder_detail.drawio"
    xml = build().to_xml()
    with open(path, "w") as f:
        f.write(xml)
    print(f"✅ {path} ({os.path.getsize(path)} bytes)")
