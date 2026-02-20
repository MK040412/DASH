#!/usr/bin/env python3
"""Generate a publication-quality draw.io diagram of the Qwen3-VL Vision Encoder.

Paper figure style: muted colors, thin strokes, clean layout, dimension annotations.
Every box/arrow/label is individually editable in draw.io.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom

# Paper-style muted palette
C_INPUT = "#F5F5F5"       # light gray
C_CONV = "#E3F2FD"        # light blue
C_POS_ABS = "#F3E5F5"     # light purple (absolute pos)
C_POS_ROT = "#EDE7F6"     # deeper purple (rotary)
C_ATTN = "#E3F2FD"        # light blue
C_NORM = "#FFFDE7"        # light yellow
C_MLP = "#FFF3E0"         # light orange
C_MERGE = "#E8F5E9"       # light green
C_DEEPSTACK = "#FCE4EC"   # light pink/rose
C_RESIDUAL = "#ECEFF1"    # blue-gray
C_OUTPUT = "#E0F2F1"      # light teal
C_WHITE = "#FFFFFF"

S_DARK = "#37474F"        # dark blue-gray stroke
S_BLUE = "#1565C0"
S_PURPLE = "#6A1B9A"
S_ORANGE = "#E65100"
S_GREEN = "#2E7D32"
S_PINK = "#C62828"
S_GRAY = "#9E9E9E"


class D:
    """Draw.io builder — paper figure style."""
    def __init__(self, name, pw=1400, ph=2200):
        self.name = name
        self.pw = pw
        self.ph = ph
        self.cells = []
        self._id = 2

    def _nid(self):
        i = self._id; self._id += 1; return str(i)

    def box(self, x, y, w, h, label, fill=C_WHITE, stroke=S_DARK, fs=11, bold=False, rounded=True, opacity=100):
        cid = self._nid()
        r = "1" if rounded else "0"
        style = (f"rounded={r};whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
                 f"fontSize={fs};fontStyle={'1' if bold else '0'};opacity={opacity};arcSize=8;")
        self.cells.append(dict(id=cid, value=label, style=style, vertex=True, x=x, y=y, w=w, h=h))
        return cid

    def label(self, x, y, w, h, text, fs=9, color="#616161", align="center", bold=False):
        cid = self._nid()
        style = (f"text;html=1;align={align};verticalAlign=middle;resizable=1;points=[];"
                 f"autosize=0;strokeColor=none;fillColor=none;fontSize={fs};fontColor={color};"
                 f"fontStyle={'1' if bold else '0'};")
        self.cells.append(dict(id=cid, value=text, style=style, vertex=True, x=x, y=y, w=w, h=h))
        return cid

    def group(self, x, y, w, h, label, fill="#FAFAFA", stroke=S_GRAY, dashed=True, fs=11):
        cid = self._nid()
        d = "1" if dashed else "0"
        style = (f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
                 f"dashed={d};dashPattern=8 4;verticalAlign=top;fontSize={fs};"
                 f"fontStyle=1;fontColor=#424242;opacity=60;arcSize=6;")
        self.cells.append(dict(id=cid, value=label, style=style, vertex=True, x=x, y=y, w=w, h=h))
        return cid

    def arrow(self, src, tgt, label="", dashed=False, color=S_DARK, curved=False):
        cid = self._nid()
        d = "dashed=1;dashPattern=6 3;" if dashed else ""
        edge = "edgeStyle=entityRelationEdgeStyle;" if curved else "edgeStyle=orthogonalEdgeStyle;"
        style = (f"{edge}rounded=1;orthogonalLoop=1;html=1;{d}strokeColor={color};"
                 f"fontSize=9;fontColor=#616161;endArrow=blockThin;endFill=1;strokeWidth=1.5;")
        self.cells.append(dict(id=cid, value=label, style=style, edge=True, source=src, target=tgt))
        return cid

    def to_xml(self):
        root = ET.Element("mxfile", host="app.diagrams.net")
        diag = ET.SubElement(root, "diagram", name=self.name, id="ve1")
        model = ET.SubElement(diag, "mxGraphModel",
                              dx="1422", dy="800", grid="1", gridSize="10",
                              guides="1", tooltips="1", connect="1", arrows="1",
                              fold="1", page="1", pageScale="1",
                              pageWidth=str(self.pw), pageHeight=str(self.ph))
        rc = ET.SubElement(model, "root")
        ET.SubElement(rc, "mxCell", id="0")
        ET.SubElement(rc, "mxCell", id="1", parent="0")
        for c in self.cells:
            attrs = {"id": c["id"], "style": c["style"], "parent": "1"}
            if c.get("value"): attrs["value"] = c["value"]
            if c.get("vertex"): attrs["vertex"] = "1"
            if c.get("edge"):
                attrs["edge"] = "1"
                if c.get("source"): attrs["source"] = c["source"]
                if c.get("target"): attrs["target"] = c["target"]
            cell = ET.SubElement(rc, "mxCell", **attrs)
            if c.get("vertex") and "x" in c:
                ET.SubElement(cell, "mxGeometry",
                              x=str(c["x"]), y=str(c["y"]),
                              width=str(c["w"]), height=str(c["h"]),
                              **{"as": "geometry"})
            elif c.get("edge"):
                ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})
        return minidom.parseString(ET.tostring(root, encoding="unicode")).toprettyxml(indent="  ")


def build():
    d = D("Qwen3-VL Vision Encoder", pw=1400, ph=2400)

    # ════════════════════════════════════════
    # Title
    # ════════════════════════════════════════
    d.label(200, 15, 800, 35, "<b>Qwen3-VL Vision Encoder</b>  (Qwen3VLVisionModel)", fs=16, color="#212121")
    d.label(200, 45, 800, 22, "depth=27 · hidden_size=1152 · num_heads=16 · head_dim=72 · patch_size=16 · temporal_patch_size=2", fs=9, color="#757575")

    # ════════════════════════════════════════
    # 1. Input
    # ════════════════════════════════════════
    CX = 400  # center X for main flow
    y = 85
    inp = d.box(CX, y, 240, 36, "<b>Input: pixel_values</b>", fill=C_INPUT, stroke=S_GRAY, fs=11, bold=True)
    d.label(CX, y+36, 240, 18, "<i>(B, 3, T, H, W)</i>  video/image frames", fs=9)

    # ════════════════════════════════════════
    # 2. PatchEmbed
    # ════════════════════════════════════════
    y += 72
    pe_grp = d.group(CX-30, y, 300, 110, "PatchEmbed", fill="#EBF5FB", stroke=S_BLUE)

    pe_reshape = d.box(CX, y+28, 240, 30,
        "reshape → (-1, 3, 2, 16, 16)", fill=C_WHITE, stroke=S_BLUE, fs=9)
    pe_conv = d.box(CX, y+66, 240, 34,
        "<b>Conv3d</b>(3 → 1152, k=[2,16,16], s=[2,16,16], bias=True)", fill=C_CONV, stroke=S_BLUE, fs=9, bold=True)
    d.arrow(inp, pe_reshape)
    d.arrow(pe_reshape, pe_conv)
    d.label(CX+245, y+70, 120, 20, "view(-1, 1152)", fs=8, color=S_BLUE)

    # ════════════════════════════════════════
    # 3. Absolute Position Embedding
    # ════════════════════════════════════════
    y += 130
    abs_pos = d.box(CX, y, 240, 50,
        "<b>+ Absolute Pos Embed</b><br>"
        "nn.Embedding(2304, 1152)<br>"
        "bilinear interpolation for any resolution",
        fill=C_POS_ABS, stroke=S_PURPLE, fs=9)
    d.arrow(pe_conv, abs_pos)
    d.label(CX-35, y+10, 30, 18, "<b>+</b>", fs=14, color=S_PURPLE)
    d.label(CX+245, y+8, 100, 18, "(N, 1152)", fs=9, color="#757575")

    # ════════════════════════════════════════
    # 4. 2D Rotary Position Embedding (side)
    # ════════════════════════════════════════
    rope_x = 750
    rope = d.box(rope_x, y-15, 220, 80,
        "<b>2D RoPE</b><br>"
        "VisionRotaryEmbedding<br>"
        "dim = head_dim//2 = 36<br>"
        "θ = 10000<br>"
        "applied to Q, K only",
        fill=C_POS_ROT, stroke=S_PURPLE, fs=9)
    d.label(rope_x, y+65, 220, 18, "cos/sin → (N, head_dim)", fs=8, color=S_PURPLE)

    # ════════════════════════════════════════
    # 5. cu_seqlens (side)
    # ════════════════════════════════════════
    cu_seq = d.box(rope_x, y+100, 220, 45,
        "<b>cu_seqlens</b><br>"
        "cumulative sequence lengths<br>"
        "for variable-length window attention",
        fill=C_INPUT, stroke=S_GRAY, fs=9)

    # ════════════════════════════════════════
    # 6. VisionBlock × 27 (MAIN SECTION)
    # ════════════════════════════════════════
    y += 75
    vb_top = y + 15
    vb_grp = d.group(CX-80, vb_top, 570, 680,
        "VisionBlock × 27  (depth iterations)", fill="#F5F7FA", stroke=S_BLUE)

    # ── Single block detail ──
    by = vb_top + 35

    # norm1
    norm1 = d.box(CX, by, 240, 32,
        "<b>LayerNorm</b> (norm1, eps=1e-6)", fill=C_NORM, stroke="#F9A825", fs=10)
    d.arrow(abs_pos, norm1)

    # ── Attention sub-group ──
    by += 48
    attn_grp = d.group(CX-60, by, 520, 340,
        "VisionAttention", fill="#E8EAF6", stroke=S_BLUE)

    # QKV
    aqy = by + 32
    qkv = d.box(CX, aqy, 240, 34,
        "<b>QKV Linear</b>(1152 → 3456, bias=True)", fill=C_ATTN, stroke=S_BLUE, fs=10)
    d.arrow(norm1, qkv)

    # reshape + unbind
    aqy += 46
    reshape = d.box(CX, aqy, 240, 26,
        "reshape(seq, 3, 16, 72) → permute → unbind", fill=C_WHITE, stroke=S_BLUE, fs=9)
    d.arrow(qkv, reshape)

    # Q, K, V split
    aqy += 40
    q_box = d.box(CX-30, aqy, 80, 36, "<b>Q</b><br>[N,16,72]", fill="#BBDEFB", stroke=S_BLUE, fs=9)
    k_box = d.box(CX+70, aqy, 80, 36, "<b>K</b><br>[N,16,72]", fill="#BBDEFB", stroke=S_BLUE, fs=9)
    v_box = d.box(CX+170, aqy, 80, 36, "<b>V</b><br>[N,16,72]", fill="#BBDEFB", stroke=S_BLUE, fs=9)
    d.arrow(reshape, q_box)
    d.arrow(reshape, k_box)
    d.arrow(reshape, v_box)

    # RoPE application
    aqy += 50
    rope_apply = d.box(CX, aqy, 180, 32,
        "<b>apply_rotary_pos_emb</b><br>Q·cos + rot(Q)·sin", fill=C_POS_ROT, stroke=S_PURPLE, fs=9)
    d.arrow(q_box, rope_apply)
    d.arrow(k_box, rope_apply)
    d.arrow(rope, rope_apply, label="cos, sin", dashed=True, color=S_PURPLE)
    d.label(CX+185, aqy+4, 80, 20, "V: no RoPE", fs=8, color=S_GRAY)

    # Window Attention
    aqy += 48
    win_attn = d.box(CX-20, aqy, 280, 42,
        "<b>Window Attention</b><br>"
        "heads=16 · head_dim=72 · is_causal=False<br>"
        "FlashAttention2 / chunked by cu_seqlens",
        fill=C_ATTN, stroke=S_BLUE, fs=9)
    d.arrow(rope_apply, win_attn)
    d.arrow(v_box, win_attn)
    d.arrow(cu_seq, win_attn, label="cu_seqlens", dashed=True, color=S_GRAY)

    # Output proj
    aqy += 58
    o_proj = d.box(CX, aqy, 240, 30,
        "<b>Output Proj</b> Linear(1152 → 1152)", fill=C_ATTN, stroke=S_BLUE, fs=10)
    d.arrow(win_attn, o_proj)

    # Residual 1
    aqy += 48
    res1 = d.box(CX+20, aqy, 200, 26,
        "<b>+ Residual</b>  (h = h + attn(norm1(h)))", fill=C_RESIDUAL, stroke=S_GRAY, fs=9)
    d.arrow(o_proj, res1)

    # norm2
    aqy += 42
    norm2 = d.box(CX, aqy, 240, 32,
        "<b>LayerNorm</b> (norm2, eps=1e-6)", fill=C_NORM, stroke="#F9A825", fs=10)
    d.arrow(res1, norm2)

    # ── MLP sub-group ──
    aqy += 48
    mlp_grp = d.group(CX-60, aqy, 520, 160,
        "VisionMLP  (simple 2-layer, NOT SwiGLU)", fill="#FFF8E1", stroke=S_ORANGE)

    mqy = aqy + 30
    fc1 = d.box(CX, mqy, 260, 30,
        "<b>linear_fc1</b>  Linear(1152 → 4304, bias=True)", fill=C_MLP, stroke=S_ORANGE, fs=10)
    d.arrow(norm2, fc1)

    mqy += 42
    act = d.box(CX+30, mqy, 200, 28,
        "<b>GELU</b> (gelu_pytorch_tanh)", fill="#FFE0B2", stroke=S_ORANGE, fs=10)
    d.arrow(fc1, act)

    mqy += 40
    fc2 = d.box(CX, mqy, 260, 30,
        "<b>linear_fc2</b>  Linear(4304 → 1152, bias=True)", fill=C_MLP, stroke=S_ORANGE, fs=10)
    d.arrow(act, fc2)

    # Residual 2
    mqy += 44
    res2 = d.box(CX+20, mqy, 200, 26,
        "<b>+ Residual</b>  (h = h + mlp(norm2(h)))", fill=C_RESIDUAL, stroke=S_GRAY, fs=9)
    d.arrow(fc2, res2)

    # ── DeepStack branching ──
    # Show on the left side
    ds_y = vb_top + 200
    ds_grp = d.group(40, ds_y, 260, 200,
        "DeepStack Branch", fill="#FCE4EC", stroke=S_PINK)

    ds_label = d.label(55, ds_y+30, 230, 50,
        "At layers <b>[8, 16, 24]</b>:<br>"
        "hidden_states → DeepStack PatchMerger<br>"
        "(separate merger per level)",
        fs=9, color="#B71C1C")

    ds_pm = d.box(60, ds_y+100, 220, 65,
        "<b>DeepStack PatchMerger</b><br>"
        "LayerNorm(4608, postshuffle)<br>"
        "view(-1, 1152×4) → 2×2 merge<br>"
        "Linear(4608→4608) → GELU<br>"
        "Linear(4608→3584)",
        fill="#FFCDD2", stroke=S_PINK, fs=9)

    # Arrow from VisionBlock group to DeepStack
    d.arrow(res2, ds_pm, label="layers 8,16,24", dashed=True, color=S_PINK, curved=True)

    ds_out = d.label(60, ds_y+170, 220, 25,
        "→ <b>deepstack_features</b>  [3 × (N/4, 3584)]", fs=9, color="#B71C1C")

    # ════════════════════════════════════════
    # 7. Final PatchMerger
    # ════════════════════════════════════════
    pm_y = vb_top + 700
    pm_grp = d.group(CX-30, pm_y, 300, 210,
        "PatchMerger (final)", fill="#E8F5E9", stroke=S_GREEN)

    pmy = pm_y + 30
    pm_norm = d.box(CX, pmy, 240, 28,
        "<b>LayerNorm</b> (pre-shuffle, 1152, eps=1e-6)", fill=C_NORM, stroke="#F9A825", fs=9)
    d.arrow(res2, pm_norm, label="(N, 1152)")

    pmy += 38
    pm_reshape = d.box(CX, pmy, 240, 28,
        "view(-1, 1152 × 4) → <b>2×2 spatial merge</b>", fill=C_WHITE, stroke=S_GREEN, fs=9)
    d.arrow(pm_norm, pm_reshape)
    d.label(CX+245, pmy+2, 100, 18, "(N/4, 4608)", fs=8, color=S_GREEN)

    pmy += 38
    pm_fc1 = d.box(CX, pmy, 240, 28,
        "<b>linear_fc1</b>  Linear(4608 → 4608)", fill=C_MERGE, stroke=S_GREEN, fs=9)
    d.arrow(pm_reshape, pm_fc1)

    pmy += 34
    pm_gelu = d.box(CX+30, pmy, 180, 24, "<b>GELU</b>", fill="#C8E6C9", stroke=S_GREEN, fs=10)
    d.arrow(pm_fc1, pm_gelu)

    pmy += 32
    pm_fc2 = d.box(CX, pmy, 240, 28,
        "<b>linear_fc2</b>  Linear(4608 → 3584)", fill=C_MERGE, stroke=S_GREEN, fs=9)
    d.arrow(pm_gelu, pm_fc2)
    d.label(CX+245, pmy+2, 130, 18, "out_hidden_size=d<sub>LLM</sub>", fs=8, color=S_GREEN)

    # ════════════════════════════════════════
    # 8. Output
    # ════════════════════════════════════════
    out_y = pm_y + 230
    out_vis = d.box(CX, out_y, 240, 36,
        "<b>visual_tokens</b>", fill=C_OUTPUT, stroke=S_GREEN, fs=12, bold=True)
    d.arrow(pm_fc2, out_vis, label="(N/4, 3584)")
    d.label(CX, out_y+36, 240, 18, "→ inserted at &lt;|vision|&gt; positions in LLM input", fs=9, color="#616161")

    ds_arrow_out = d.box(CX-250, out_y, 200, 36,
        "<b>deepstack_features</b>", fill="#FFCDD2", stroke=S_PINK, fs=10, bold=True)
    d.label(CX-250, out_y+36, 200, 18, "→ injected into early LLM decoder layers", fs=9, color="#B71C1C")

    # ════════════════════════════════════════
    # Config summary box (bottom right)
    # ════════════════════════════════════════
    cfg_y = pm_y + 50
    d.box(750, cfg_y, 280, 200,
        "<b>Qwen3VLVisionConfig (defaults)</b><br><br>"
        "<div style='text-align:left;padding-left:10px;'>"
        "depth = 27<br>"
        "hidden_size = 1152<br>"
        "intermediate_size = 4304<br>"
        "num_heads = 16<br>"
        "head_dim = 72<br>"
        "patch_size = 16<br>"
        "temporal_patch_size = 2<br>"
        "spatial_merge_size = 2<br>"
        "out_hidden_size = 3584<br>"
        "num_position_embeddings = 2304<br>"
        "deepstack_visual_indexes = [8,16,24]<br>"
        "hidden_act = gelu_pytorch_tanh"
        "</div>",
        fill="#FAFAFA", stroke=S_GRAY, fs=9, rounded=True)

    # ════════════════════════════════════════
    # Key differences box (bottom right)
    # ════════════════════════════════════════
    d.box(750, cfg_y + 215, 280, 95,
        "<b>vs Qwen2.5-VL Vision Encoder</b><br><br>"
        "<div style='text-align:left;padding-left:10px;'>"
        "✦ DeepStack (NEW) — multi-level injection<br>"
        "✦ LayerNorm (was RMSNorm)<br>"
        "✦ Simple MLP (was SwiGLU 3-linear)<br>"
        "✦ Abs pos embed (was RoPE only)<br>"
        "✦ Conv3d bias=True (was False)"
        "</div>",
        fill="#FFF9C4", stroke="#F9A825", fs=9, rounded=True)

    return d


if __name__ == "__main__":
    import os
    outdir = "/home/perelman/.openclaw/workspace/qwen_review"
    d = build()
    path = os.path.join(outdir, "d2_vision_encoder_detail.drawio")
    with open(path, "w") as f:
        f.write(d.to_xml())
    print(f"✅ {path} ({os.path.getsize(path)} bytes)")
