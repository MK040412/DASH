#!/usr/bin/env python3
"""Generate editable .drawio XML files for Qwen3-VL architecture diagrams."""

import xml.etree.ElementTree as ET
from xml.dom import minidom

# Color scheme
BLUE_VISION = "#dae8fc"       # Vision encoder
BLUE_ATTN = "#b3cde3"         # Attention
GREEN_LLM = "#d5e8d4"         # LLM / decoder
ORANGE_MLP = "#ffe6cc"        # MLP
YELLOW_NORM = "#fff2cc"       # Normalization
PURPLE_ROPE = "#e1d5e7"       # Positional encoding
RED_OUTPUT = "#f8cecc"        # Output
CYAN_MERGE = "#d4e1f5"        # Token merging
GREEN_CACHE = "#b9e0a5"       # KV Cache
DEEPSTACK_ORANGE = "#fce5cd"  # DeepStack

STROKE_BLUE = "#6c8ebf"
STROKE_GREEN = "#82b366"
STROKE_ORANGE = "#d6b656"
STROKE_YELLOW = "#d6b656"
STROKE_PURPLE = "#9673a6"
STROKE_RED = "#b85450"


class DrawioBuilder:
    def __init__(self, name="Page-1"):
        self.name = name
        self.cells = []
        self._id = 2  # 0 and 1 are reserved

    def _next_id(self):
        i = self._id
        self._id += 1
        return str(i)

    def add_box(self, x, y, w, h, label, fill="#ffffff", stroke="#000000",
                rounded=True, style_extra="", font_size=12, bold=False):
        cid = self._next_id()
        r = "1" if rounded else "0"
        b = "1" if bold else "0"
        style = (f"rounded={r};whiteSpace=wrap;html=1;fillColor={fill};"
                 f"strokeColor={stroke};fontSize={font_size};fontStyle={1 if bold else 0};"
                 f"{style_extra}")
        self.cells.append({
            "id": cid, "value": label, "style": style, "vertex": True,
            "x": x, "y": y, "w": w, "h": h
        })
        return cid

    def add_text(self, x, y, w, h, label, font_size=10, color="#666666", align="center"):
        cid = self._next_id()
        style = (f"text;html=1;align={align};verticalAlign=middle;resizable=0;"
                 f"points=[];autosize=1;strokeColor=none;fillColor=none;"
                 f"fontSize={font_size};fontColor={color};")
        self.cells.append({
            "id": cid, "value": label, "style": style, "vertex": True,
            "x": x, "y": y, "w": w, "h": h
        })
        return cid

    def add_group_box(self, x, y, w, h, label, fill="#f5f5f5", stroke="#666666",
                      dashed=True):
        cid = self._next_id()
        dash = "1" if dashed else "0"
        style = (f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};"
                 f"strokeColor={stroke};dashed={dash};dashPattern=5 5;"
                 f"opacity=50;verticalAlign=top;fontSize=11;fontStyle=1;")
        self.cells.append({
            "id": cid, "value": label, "style": style, "vertex": True,
            "x": x, "y": y, "w": w, "h": h
        })
        return cid

    def add_arrow(self, src, tgt, label="", style_extra="", dashed=False):
        cid = self._next_id()
        dash = "dashed=1;dashPattern=5 5;" if dashed else ""
        style = (f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;"
                 f"jettySize=auto;html=1;{dash}{style_extra}")
        self.cells.append({
            "id": cid, "value": label, "style": style, "edge": True,
            "source": src, "target": tgt
        })
        return cid

    def add_curved_arrow(self, src, tgt, label="", style_extra="", dashed=False):
        cid = self._next_id()
        dash = "dashed=1;dashPattern=5 5;" if dashed else ""
        style = (f"edgeStyle=entityRelationEdgeStyle;rounded=1;orthogonalLoop=1;"
                 f"html=1;{dash}{style_extra}")
        self.cells.append({
            "id": cid, "value": label, "style": style, "edge": True,
            "source": src, "target": tgt
        })
        return cid

    def to_xml(self):
        root = ET.Element("mxfile", host="app.diagrams.net")
        diagram = ET.SubElement(root, "diagram", name=self.name, id="d1")
        model = ET.SubElement(diagram, "mxGraphModel",
                              dx="1422", dy="762", grid="1", gridSize="10",
                              guides="1", tooltips="1", connect="1",
                              arrows="1", fold="1", page="1",
                              pageScale="1", pageWidth="1600", pageHeight="1200")
        rootcell = ET.SubElement(model, "root")
        ET.SubElement(rootcell, "mxCell", id="0")
        ET.SubElement(rootcell, "mxCell", id="1", parent="0")

        for c in self.cells:
            attrs = {"id": c["id"], "style": c["style"], "parent": "1"}
            if c.get("value"):
                attrs["value"] = c["value"]
            if c.get("vertex"):
                attrs["vertex"] = "1"
            if c.get("edge"):
                attrs["edge"] = "1"
                if c.get("source"):
                    attrs["source"] = c["source"]
                if c.get("target"):
                    attrs["target"] = c["target"]

            cell = ET.SubElement(rootcell, "mxCell", **attrs)

            if c.get("vertex") and "x" in c:
                ET.SubElement(cell, "mxGeometry",
                              x=str(c["x"]), y=str(c["y"]),
                              width=str(c["w"]), height=str(c["h"]),
                              **{"as": "geometry"})
            elif c.get("edge"):
                ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})

        raw = ET.tostring(root, encoding="unicode")
        return minidom.parseString(raw).toprettyxml(indent="  ")


def build_d1_overall():
    """D1: Qwen3-VL Overall Architecture with DeepStack"""
    b = DrawioBuilder("D1: Qwen3-VL Overall")

    # ── Vision Encoder group ──
    vg = b.add_group_box(20, 30, 340, 720, "Vision Encoder", fill="#eef3ff", stroke=STROKE_BLUE)

    px = b.add_box(90, 80, 200, 40, "pixel_values", fill="#ffffff", stroke="#999999", font_size=11)
    b.add_text(90, 120, 200, 20, "(B×T×3×H×W)", font_size=9)

    pe = b.add_box(90, 160, 200, 50, "<b>PatchEmbed</b><br>Conv3d(3→1152, k=[2,14,14])", fill=BLUE_VISION, stroke=STROKE_BLUE, font_size=10)
    b.add_arrow(px, pe)

    pe_pos = b.add_box(90, 240, 200, 40, "<b>+ Absolute Pos Embed</b><br>nn.Embedding + bilinear interp", fill=PURPLE_ROPE, stroke=STROKE_PURPLE, font_size=9)
    b.add_arrow(pe, pe_pos)

    vb = b.add_box(90, 310, 200, 50, "<b>VisionBlock × 32</b><br>LayerNorm + Attention + MLP", fill=BLUE_VISION, stroke=STROKE_BLUE, font_size=10)
    b.add_arrow(pe_pos, vb)

    rope2d = b.add_box(30, 310, 50, 50, "<b>2D<br>RoPE</b>", fill=PURPLE_ROPE, stroke=STROKE_PURPLE, font_size=9)
    b.add_arrow(rope2d, vb, dashed=True)

    # DeepStack extraction
    ds_extract = b.add_box(90, 390, 200, 45, "<b>DeepStack Feature Extract</b><br>intermediate layer outputs", fill=DEEPSTACK_ORANGE, stroke=STROKE_ORANGE, font_size=9)
    b.add_curved_arrow(vb, ds_extract, style_extra=f"strokeColor={STROKE_ORANGE};")

    pm = b.add_box(90, 465, 200, 50, "<b>PatchMerger</b><br>2×2 spatial merge → d_LLM", fill="#c8e6c9", stroke=STROKE_GREEN, font_size=10)
    b.add_arrow(vb, pm, label="(N, 1152)")

    ds_merger = b.add_box(90, 545, 200, 45, "<b>DeepStack PatchMergers</b><br>separate merger per level", fill=DEEPSTACK_ORANGE, stroke=STROKE_ORANGE, font_size=9)
    b.add_arrow(ds_extract, ds_merger)

    vt = b.add_box(90, 620, 200, 40, "visual_tokens", fill="#ffffff", stroke="#999999", font_size=11)
    b.add_arrow(pm, vt, label="(N/4, d)")
    b.add_text(90, 660, 200, 20, "deepstack_features", font_size=9, color=STROKE_ORANGE)
    b.add_arrow(ds_merger, vt, dashed=True, style_extra=f"strokeColor={STROKE_ORANGE};")

    # ── Token Merging ──
    mg = b.add_group_box(400, 300, 280, 160, "Token Merging", fill="#fffde7", stroke=STROKE_YELLOW)

    inp = b.add_box(430, 340, 220, 40, "input_ids → embed_tokens", fill=YELLOW_NORM, stroke=STROKE_YELLOW, font_size=10)
    merge = b.add_box(430, 400, 220, 40, "<b>Merge Embeddings</b><br>insert visual @ &lt;|vision|&gt;", fill=CYAN_MERGE, stroke=STROKE_BLUE, font_size=10)
    b.add_arrow(inp, merge)
    b.add_arrow(vt, merge, label="visual_tokens")

    # ── LLM Decoder ──
    lg = b.add_group_box(740, 30, 360, 720, "Language Model (Decoder)", fill="#e8f5e9", stroke=STROKE_GREEN)

    mrope = b.add_box(800, 70, 240, 45, "<b>M-RoPE</b><br>3D: temporal | height | width", fill=PURPLE_ROPE, stroke=STROKE_PURPLE, font_size=10)

    dl = b.add_box(800, 300, 240, 50, "<b>DecoderLayer × N</b><br>RMSNorm + GQA + SwiGLU", fill=GREEN_LLM, stroke=STROKE_GREEN, font_size=10)
    b.add_arrow(merge, dl, label="merged_hidden_states (B×L×d)")
    b.add_arrow(mrope, dl, label="position_ids (3×L)", dashed=True)

    # DeepStack injection into decoder
    ds_inject = b.add_box(800, 200, 240, 50, "<b>DeepStack Injection</b><br>hidden += visual_embeds<br>(early decoder layers)", fill=DEEPSTACK_ORANGE, stroke=STROKE_ORANGE, font_size=9)
    b.add_arrow(ds_merger, ds_inject, label="deepstack features", dashed=True, style_extra=f"strokeColor={STROKE_ORANGE};")
    b.add_arrow(ds_inject, dl, dashed=True, style_extra=f"strokeColor={STROKE_ORANGE};")

    norm = b.add_box(800, 400, 240, 40, "<b>RMSNorm</b>", fill=YELLOW_NORM, stroke=STROKE_YELLOW, font_size=11)
    b.add_arrow(dl, norm, label="(B×L×d)")

    head = b.add_box(800, 480, 240, 40, "<b>lm_head</b><br>nn.Linear(d, V, bias=False)", fill=RED_OUTPUT, stroke=STROKE_RED, font_size=10)
    b.add_arrow(norm, head)

    logits = b.add_box(800, 560, 240, 40, "logits", fill=RED_OUTPUT, stroke=STROKE_RED, font_size=11, bold=True)
    b.add_arrow(head, logits, label="(B×L×V)")

    # ── Legend ──
    ly = 780
    b.add_text(20, ly, 100, 20, "<b>Legend:</b>", font_size=10, color="#333333", align="left")
    b.add_box(20, ly+25, 15, 15, "", fill=BLUE_VISION, stroke=STROKE_BLUE)
    b.add_text(40, ly+22, 120, 20, "Vision Encoder", font_size=9, color="#333", align="left")
    b.add_box(20, ly+45, 15, 15, "", fill=GREEN_LLM, stroke=STROKE_GREEN)
    b.add_text(40, ly+42, 120, 20, "Language Model", font_size=9, color="#333", align="left")
    b.add_box(20, ly+65, 15, 15, "", fill=PURPLE_ROPE, stroke=STROKE_PURPLE)
    b.add_text(40, ly+62, 120, 20, "Position Encoding", font_size=9, color="#333", align="left")
    b.add_box(20, ly+85, 15, 15, "", fill=DEEPSTACK_ORANGE, stroke=STROKE_ORANGE)
    b.add_text(40, ly+82, 120, 20, "DeepStack (NEW)", font_size=9, color="#333", align="left")
    b.add_box(20, ly+105, 15, 15, "", fill=RED_OUTPUT, stroke=STROKE_RED)
    b.add_text(40, ly+102, 120, 20, "Output", font_size=9, color="#333", align="left")

    # Model variants
    b.add_text(400, ly, 400, 80,
               "<b>Model Variants:</b><br>"
               "2B: d=1536, N=28, V=151936<br>"
               "7B: d=3584, N=28, V=152064<br>"
               "72B: d=8192, N=80, V=152064<br>"
               "Vision: embed_dim=1152, depth=32",
               font_size=9, color="#666", align="left")

    return b


def build_d2_vision_encoder():
    """D2: Vision Encoder Detail"""
    b = DrawioBuilder("D2: Vision Encoder")

    b.add_text(100, 10, 300, 30, "<b>Qwen3-VL Vision Encoder</b>", font_size=14, color="#333")

    inp = b.add_box(120, 50, 220, 40, "<b>Input Image / Video</b>", fill="#ffffff", stroke="#999")
    b.add_text(120, 90, 220, 20, "(B, 3, T, H, W)", font_size=9)

    pe = b.add_box(120, 120, 220, 50, "<b>PatchEmbed</b><br>Conv3d(3→1152, k=[2,14,14], bias=True)<br>reshape → view(-1, 1152)", fill=BLUE_VISION, stroke=STROKE_BLUE, font_size=9)
    b.add_arrow(inp, pe)

    pos = b.add_box(120, 200, 220, 45, "<b>+ Absolute Position Embed</b><br>nn.Embedding + bilinear interpolation", fill=PURPLE_ROPE, stroke=STROKE_PURPLE, font_size=9)
    b.add_arrow(pe, pos, label="(N, 1152)")

    # VisionBlock group
    vbg = b.add_group_box(60, 275, 340, 280, "VisionBlock × 32", fill="#f0f4ff", stroke=STROKE_BLUE)

    ln1 = b.add_box(120, 310, 200, 35, "<b>LayerNorm</b> (norm1)<br>eps=1e-6", fill=YELLOW_NORM, stroke=STROKE_YELLOW, font_size=9)
    b.add_arrow(pos, ln1)

    attn = b.add_box(120, 365, 200, 45, "<b>VisionAttention</b><br>QKV Linear(1152→3456)<br>Window Attn (cu_seqlens)", fill=BLUE_ATTN, stroke=STROKE_BLUE, font_size=9)
    b.add_arrow(ln1, attn)

    rope = b.add_box(340, 365, 50, 45, "<b>2D<br>RoPE</b>", fill=PURPLE_ROPE, stroke=STROKE_PURPLE, font_size=8)
    b.add_arrow(rope, attn, label="Q,K", dashed=True)

    res1 = b.add_box(120, 425, 200, 25, "+ Residual", fill="#f5f5f5", stroke="#999", font_size=9)
    b.add_arrow(attn, res1)

    ln2 = b.add_box(120, 465, 200, 35, "<b>LayerNorm</b> (norm2)<br>eps=1e-6", fill=YELLOW_NORM, stroke=STROKE_YELLOW, font_size=9)
    b.add_arrow(res1, ln2)

    mlp = b.add_box(120, 510, 200, 40, "<b>VisionMLP</b><br>fc1(1152→4608) → QuickGELU → fc2", fill=ORANGE_MLP, stroke=STROKE_ORANGE, font_size=9)
    b.add_arrow(ln2, mlp)

    # DeepStack tap
    ds = b.add_box(120, 590, 220, 45, "<b>DeepStack Extract</b><br>intermediate layers → separate PatchMergers<br>(deepstack_visual_indexes)", fill=DEEPSTACK_ORANGE, stroke=STROKE_ORANGE, font_size=9)
    b.add_curved_arrow(mlp, ds, dashed=True, style_extra=f"strokeColor={STROKE_ORANGE};")
    b.add_text(120, 635, 220, 20, "multi-level features to LLM decoder", font_size=8, color=STROKE_ORANGE)

    # PatchMerger
    pmg = b.add_group_box(60, 670, 340, 120, "PatchMerger", fill="#e8f5e9", stroke=STROKE_GREEN)
    lnq = b.add_box(120, 700, 200, 30, "LayerNorm (ln_q)", fill=YELLOW_NORM, stroke=STROKE_YELLOW, font_size=9)
    b.add_arrow(mlp, lnq, label="(N, 1152)")
    reshape = b.add_box(120, 740, 200, 35, "Reshape 2×2 spatial merge<br>Linear→GELU→Linear(→d_LLM)", fill="#c8e6c9", stroke=STROKE_GREEN, font_size=9)
    b.add_arrow(lnq, reshape)

    out = b.add_box(120, 820, 220, 40, "<b>To LLM (Qwen3-VL Language Model)</b>", fill="#ffffff", stroke="#999", font_size=10, bold=True)
    b.add_arrow(reshape, out, label="(N/4, d_LLM)")

    return b


def build_d3_vision_block():
    """D3: VisionBlock Internal"""
    b = DrawioBuilder("D3: VisionBlock Internal")

    b.add_text(80, 10, 300, 30, "<b>Qwen3-VL VisionBlock Internal</b>", font_size=14, color="#333")

    inp = b.add_box(100, 50, 180, 35, "Input hidden_states", fill="#ffffff", stroke="#999", font_size=10)
    b.add_text(280, 55, 100, 25, "[N, 1152]", font_size=9)

    # Left: main path
    ln1 = b.add_box(100, 110, 180, 35, "<b>LayerNorm</b> (norm1)", fill=YELLOW_NORM, stroke=STROKE_YELLOW, font_size=10)
    b.add_arrow(inp, ln1)

    attn = b.add_box(100, 170, 180, 35, "<b>VisionAttention</b>", fill=BLUE_ATTN, stroke=STROKE_BLUE, font_size=10)
    b.add_arrow(ln1, attn)

    res1 = b.add_box(100, 230, 180, 30, "+ Residual", fill="#f5f5f5", stroke="#999", font_size=10)
    b.add_arrow(attn, res1)

    ln2 = b.add_box(100, 285, 180, 35, "<b>LayerNorm</b> (norm2)", fill=YELLOW_NORM, stroke=STROKE_YELLOW, font_size=10)
    b.add_arrow(res1, ln2)

    mlp = b.add_box(100, 345, 180, 35, "<b>VisionMLP</b>", fill=ORANGE_MLP, stroke=STROKE_ORANGE, font_size=10)
    b.add_arrow(ln2, mlp)

    res2 = b.add_box(100, 405, 180, 30, "+ Residual", fill="#f5f5f5", stroke="#999", font_size=10)
    b.add_arrow(mlp, res2)

    out = b.add_box(100, 460, 180, 35, "Output hidden_states", fill="#ffffff", stroke="#999", font_size=10)
    b.add_arrow(res2, out)

    # Right: Expanded VisionAttention
    attn_g = b.add_group_box(420, 30, 320, 280, "VisionAttention (Expanded)", fill="#eef3ff", stroke=STROKE_BLUE)

    qkv = b.add_box(460, 65, 240, 35, "<b>QKV Linear</b> (1152→3456, bias=True)", fill=BLUE_ATTN, stroke=STROKE_BLUE, font_size=9)

    q = b.add_box(460, 120, 70, 30, "<b>Q</b><br>[N,16,72]", fill="#bbdefb", stroke=STROKE_BLUE, font_size=8)
    k = b.add_box(545, 120, 70, 30, "<b>K</b><br>[N,16,72]", fill="#bbdefb", stroke=STROKE_BLUE, font_size=8)
    v = b.add_box(630, 120, 70, 30, "<b>V</b><br>[N,16,72]", fill="#bbdefb", stroke=STROKE_BLUE, font_size=8)
    b.add_arrow(qkv, q)
    b.add_arrow(qkv, k)
    b.add_arrow(qkv, v)

    rope = b.add_box(480, 170, 110, 30, "<b>2D RoPE</b><br>apply to Q, K", fill=PURPLE_ROPE, stroke=STROKE_PURPLE, font_size=8)
    b.add_arrow(q, rope, dashed=True)
    b.add_arrow(k, rope, dashed=True)
    b.add_text(630, 170, 70, 30, "(no RoPE)", font_size=8, color="#999")

    wattn = b.add_box(460, 220, 240, 35, "<b>Window Attention</b><br>num_heads=16, head_dim=72, cu_seqlens", fill=BLUE_VISION, stroke=STROKE_BLUE, font_size=9)
    b.add_arrow(rope, wattn)
    b.add_arrow(v, wattn)

    oproj = b.add_box(460, 270, 240, 30, "<b>Output Proj</b> (1152→1152)", fill=BLUE_ATTN, stroke=STROKE_BLUE, font_size=9)
    b.add_arrow(wattn, oproj)

    # Right: Expanded VisionMLP
    mlp_g = b.add_group_box(420, 340, 320, 170, "VisionMLP (Expanded)", fill="#fff8e1", stroke=STROKE_ORANGE)

    fc1 = b.add_box(460, 375, 240, 30, "<b>fc1 Linear</b> (1152→4608)", fill=ORANGE_MLP, stroke=STROKE_ORANGE, font_size=9)
    gelu = b.add_box(460, 420, 240, 30, "<b>QuickGELU</b>", fill="#ffecb3", stroke=STROKE_ORANGE, font_size=10)
    fc2 = b.add_box(460, 465, 240, 30, "<b>fc2 Linear</b> (4608→1152)", fill=ORANGE_MLP, stroke=STROKE_ORANGE, font_size=9)
    b.add_arrow(fc1, gelu)
    b.add_arrow(gelu, fc2)
    b.add_text(460, 500, 240, 20, "mlp_ratio=4", font_size=8, color=STROKE_ORANGE)

    # Dotted connectors from left to right
    b.add_arrow(attn, qkv, dashed=True, style_extra="strokeColor=#999999;")
    b.add_arrow(mlp, fc1, dashed=True, style_extra="strokeColor=#999999;")

    return b


def build_d4_decoder():
    """D4: Decoder Layer with DeepStack"""
    b = DrawioBuilder("D4: Decoder Layer")

    b.add_text(80, 10, 400, 30, "<b>Qwen3-VL Decoder Layer (×N)</b>", font_size=14, color="#333")

    inp = b.add_box(120, 55, 220, 35, "Input Hidden States", fill="#ffffff", stroke="#999", font_size=10)
    b.add_text(340, 60, 80, 25, "(B, L, d)", font_size=9)

    ln1 = b.add_box(120, 115, 220, 35, "<b>RMSNorm</b> (input_layernorm)", fill=YELLOW_NORM, stroke=STROKE_YELLOW, font_size=10)
    b.add_arrow(inp, ln1)

    # Attention group
    ag = b.add_group_box(50, 170, 380, 310, "Attention (GQA)", fill="#eef3ff", stroke=STROKE_BLUE)

    qproj = b.add_box(80, 205, 140, 30, "<b>q_proj</b><br>d → n_h·d_h", fill=BLUE_ATTN, stroke=STROKE_BLUE, font_size=9)
    kproj = b.add_box(230, 205, 140, 30, "<b>k_proj</b><br>d → n_kv·d_h", fill=BLUE_ATTN, stroke=STROKE_BLUE, font_size=9)
    vproj = b.add_box(230, 250, 140, 30, "<b>v_proj</b><br>d → n_kv·d_h", fill=BLUE_ATTN, stroke=STROKE_BLUE, font_size=9)
    b.add_arrow(ln1, qproj)
    b.add_arrow(ln1, kproj)
    b.add_arrow(ln1, vproj)

    mrope = b.add_box(80, 260, 140, 40, "<b>M-RoPE (3D)</b><br>temporal | height | width", fill=PURPLE_ROPE, stroke=STROKE_PURPLE, font_size=8)
    b.add_arrow(qproj, mrope)
    b.add_text(30, 305, 60, 20, "→ Q, K", font_size=8)

    kvcache = b.add_box(250, 300, 120, 35, "<b>KV Cache</b><br>update K, V", fill=GREEN_CACHE, stroke=STROKE_GREEN, font_size=9)
    b.add_arrow(kproj, kvcache)
    b.add_arrow(vproj, kvcache)

    repeat = b.add_box(80, 320, 140, 30, "<b>repeat_kv</b><br>n_kv → n_h (GQA)", fill=BLUE_ATTN, stroke=STROKE_BLUE, font_size=8)

    sdpa = b.add_box(100, 370, 260, 40, "<b>Scaled Dot-Product Attention</b><br>softmax(QK^T/√d_h)V + sliding window", fill=BLUE_VISION, stroke=STROKE_BLUE, font_size=9)
    b.add_arrow(mrope, sdpa)
    b.add_arrow(kvcache, repeat)
    b.add_arrow(repeat, sdpa)

    oproj = b.add_box(100, 430, 260, 30, "<b>o_proj</b> (n_h·d_h → d, no bias)", fill=BLUE_ATTN, stroke=STROKE_BLUE, font_size=9)
    b.add_arrow(sdpa, oproj)

    res1 = b.add_box(120, 500, 220, 25, "+ Residual", fill="#f5f5f5", stroke="#999", font_size=10)
    b.add_arrow(oproj, res1)

    # DeepStack injection
    ds = b.add_box(450, 200, 200, 55, "<b>DeepStack Injection</b><br>hidden_states +=<br>deepstack_visual_embeds<br>(early layers only)", fill=DEEPSTACK_ORANGE, stroke=STROKE_ORANGE, font_size=9)
    b.add_arrow(ds, res1, dashed=True, style_extra=f"strokeColor={STROKE_ORANGE};")

    ln2 = b.add_box(120, 545, 220, 35, "<b>RMSNorm</b> (post_attn_ln)", fill=YELLOW_NORM, stroke=STROKE_YELLOW, font_size=10)
    b.add_arrow(res1, ln2)

    # SwiGLU MLP
    mg = b.add_group_box(50, 600, 380, 170, "SwiGLU MLP", fill="#fff8e1", stroke=STROKE_ORANGE)

    gate = b.add_box(80, 635, 140, 30, "<b>gate_proj</b><br>d → d_ff", fill=ORANGE_MLP, stroke=STROKE_ORANGE, font_size=9)
    up = b.add_box(240, 635, 140, 30, "<b>up_proj</b><br>d → d_ff", fill=ORANGE_MLP, stroke=STROKE_ORANGE, font_size=9)
    b.add_arrow(ln2, gate)
    b.add_arrow(ln2, up)

    silu = b.add_box(80, 680, 140, 25, "<b>SiLU (σ)</b>", fill="#ffecb3", stroke=STROKE_ORANGE, font_size=9)
    b.add_arrow(gate, silu)

    mul = b.add_box(160, 715, 80, 25, "<b>×</b>", fill="#ffecb3", stroke=STROKE_ORANGE, font_size=12, bold=True)
    b.add_arrow(silu, mul)
    b.add_arrow(up, mul)

    down = b.add_box(120, 750, 220, 30, "<b>down_proj</b> (d_ff → d)", fill=ORANGE_MLP, stroke=STROKE_ORANGE, font_size=9)
    b.add_arrow(mul, down)

    res2 = b.add_box(120, 800, 220, 25, "+ Residual", fill="#f5f5f5", stroke="#999", font_size=10)
    b.add_arrow(down, res2)

    out = b.add_box(120, 850, 220, 35, "Output Hidden States", fill="#ffffff", stroke="#999", font_size=10)
    b.add_arrow(res2, out)

    # Comparison box
    comp = b.add_box(450, 500, 230, 170,
                     "<b>LLM Decoder vs Vision Encoder</b><br><br>"
                     "• Separate Q/K/V proj → Combined QKV<br>"
                     "• GQA (n_kv &lt; n_h) → MHA (n_kv = n_h)<br>"
                     "• M-RoPE (3D) → 2D RoPE<br>"
                     "• SwiGLU MLP → Simple 2-Linear MLP<br>"
                     "• Sliding Window → Full Window Attn<br>"
                     "• KV Cache → No Cache<br>"
                     "• Causal mask → No causal mask",
                     fill="#fff9c4", stroke="#f9a825", font_size=8, rounded=True)

    return b


if __name__ == "__main__":
    import os
    outdir = "/home/perelman/.openclaw/workspace/qwen_review"

    builders = {
        "d1_overall.drawio": build_d1_overall,
        "d2_vision_encoder.drawio": build_d2_vision_encoder,
        "d3_vision_block.drawio": build_d3_vision_block,
        "d4_decoder_layer.drawio": build_d4_decoder,
    }

    for fname, builder in builders.items():
        path = os.path.join(outdir, fname)
        xml = builder()
        with open(path, "w") as f:
            f.write(xml.to_xml())
        print(f"✅ {fname} ({os.path.getsize(path)} bytes)")
