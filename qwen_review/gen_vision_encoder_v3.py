#!/usr/bin/env python3
"""Qwen3-VL Vision Encoder — Paper figure style diagram.

Matches reference.png style: visual boxes, minimal text, color-coded,
multi-panel (a)(b)(c)(d)(e) zoom levels, ⊕ for residual.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom


# Colors matching reference style
BLUE     = "#D6E4F0"   # attention components
BLUE_S   = "#4472C4"
ORANGE   = "#FBE5D6"   # MLP / feedforward
ORANGE_S = "#ED7D31"
PURPLE   = "#E2D0E8"   # linear projections
PURPLE_S = "#7030A0"
GREEN    = "#E2EFDA"    # output / embedding
GREEN_S  = "#548235"
YELLOW   = "#FFF2CC"    # misc ops (norm, merge)
YELLOW_S = "#BF9000"
PINK     = "#FCE4EC"    # DeepStack
PINK_S   = "#C62828"
GRAY     = "#F2F2F2"
GRAY_S   = "#808080"
WHITE    = "#FFFFFF"
ROPE_CLR = "#E8D5F5"    # RoPE
ROPE_S   = "#7B1FA2"


class D:
    def __init__(self, name, pw=2200, ph=1400):
        self.name, self.pw, self.ph = name, pw, ph
        self.cells = []; self._id = 2

    def _n(self):
        i = self._id; self._id += 1; return str(i)

    def box(self, x, y, w, h, label, fill=WHITE, stroke="#333333", fs=10, bold=False):
        cid = self._n()
        s = (f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
             f"fontSize={fs};fontStyle={'1' if bold else '0'};arcSize=12;strokeWidth=1.5;")
        self.cells.append(dict(id=cid,value=label,style=s,vertex=True,x=x,y=y,w=w,h=h))
        return cid

    def circle(self, x, y, size, label, fill=WHITE, stroke="#333333", fs=14):
        """Circle node (for ⊕ residual)."""
        cid = self._n()
        s = (f"ellipse;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
             f"fontSize={fs};fontStyle=1;aspect=fixed;strokeWidth=1.5;")
        self.cells.append(dict(id=cid,value=label,style=s,vertex=True,x=x,y=y,w=size,h=size))
        return cid

    def lbl(self, x, y, w, h, text, fs=9, color="#333333", align="center", bold=False):
        cid = self._n()
        s = (f"text;html=1;align={align};verticalAlign=middle;resizable=1;points=[];"
             f"autosize=0;strokeColor=none;fillColor=none;fontSize={fs};fontColor={color};"
             f"fontStyle={'1' if bold else '0'};")
        self.cells.append(dict(id=cid,value=text,style=s,vertex=True,x=x,y=y,w=w,h=h))
        return cid

    def grp(self, x, y, w, h, label="", fill="#FAFAFA", stroke="#CCCCCC"):
        cid = self._n()
        s = (f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
             f"dashed=0;verticalAlign=top;fontSize=10;fontStyle=0;fontColor=#666666;"
             f"opacity=40;arcSize=6;strokeWidth=1;")
        self.cells.append(dict(id=cid,value=label,style=s,vertex=True,x=x,y=y,w=w,h=h))
        return cid

    def arr(self, s, t, label="", dash=False, color="#333333", sw=1.5):
        cid = self._n()
        dd = "dashed=1;dashPattern=6 3;" if dash else ""
        st = (f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;html=1;{dd}"
              f"strokeColor={color};fontSize=8;fontColor=#666666;endArrow=blockThin;"
              f"endFill=1;strokeWidth={sw};")
        self.cells.append(dict(id=cid,value=label,style=st,edge=True,source=s,target=t))
        return cid

    def to_xml(self):
        root = ET.Element("mxfile", host="app.diagrams.net")
        diag = ET.SubElement(root, "diagram", name=self.name, id="ve3")
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
    d = D("Qwen3-VL Vision Encoder", pw=2400, ph=1500)

    # ═══════════════════════════════════════════
    # (a) Overall Pipeline — LEFT
    # ═══════════════════════════════════════════
    ax = 30; ay = 40
    d.lbl(ax, ay, 500, 20, "<b>(a) Qwen3-VL Vision Encoder — Overall</b>", fs=12, color="#333", align="left")

    # Image input
    ay_base = ay + 35
    img = d.box(ax+5, ay_base+400, 100, 80, "🖼️<br><b>Image/Video</b>", fill=GRAY, stroke=GRAY_S, fs=9)

    # Flattening arrow label
    d.lbl(ax+110, ay_base+430, 60, 16, "Flatten", fs=8, color="#666")

    # PatchEmbed
    pe = d.box(ax+60, ay_base+320, 120, 40, "<b>Patch Embed</b><br>Conv3d", fill=PURPLE, stroke=PURPLE_S, fs=10)
    d.arr(img, pe)

    # Patch tokens E0..En
    token_y = ay_base + 260
    for i in range(8):
        tx = ax + 10 + i * 50
        tid = d.box(tx, token_y, 42, 28, f"E<sub>{i}</sub>", fill=PINK, stroke=PINK_S, fs=9)
    d.lbl(ax + 10, token_y + 30, 400, 14, "Patch Tokens  (N, 1152)", fs=8, color="#666")
    d.arr(pe, tid)  # just connect to last one visually

    # + Abs Pos
    abs_pos = d.box(ax+130, token_y - 50, 130, 32, "<b>+ Abs Pos Embed</b>", fill=ROPE_CLR, stroke=ROPE_S, fs=9)
    # Just place, user connects

    # Transformer Encoder (big block)
    enc_y = ay_base + 100
    enc = d.grp(ax, enc_y, 420, 120, "")
    enc_label = d.box(ax+10, enc_y+10, 400, 100, "", fill="#E8EAF6", stroke=BLUE_S)
    d.lbl(ax+10, enc_y+15, 400, 20, "<b>Vision Transformer Encoder</b>", fs=11, color=BLUE_S, bold=True)

    # Inside encoder: VisionBlock repeated
    bw = 45; bh = 55
    for i in range(7):
        bx = ax + 20 + i * 55
        by = enc_y + 40
        fill_c = PINK if i in [1, 3, 5] else BLUE  # highlight deepstack layers
        bid = d.box(bx, by, bw, bh, f"Block<br>{i*4}", fill=fill_c, stroke=BLUE_S if i not in [1,3,5] else PINK_S, fs=8)

    d.lbl(ax+20, enc_y+98, 380, 16,
        "×27 blocks  |  <font color='#C62828'>■</font> DeepStack layers [8, 16, 24]  |  "
        "<font color='#4472C4'>■</font> Regular",
        fs=8, color="#666")

    # Output tokens Z0..Zn
    z_y = ay_base + 50
    for i in range(8):
        tx = ax + 10 + i * 50
        d.box(tx, z_y, 42, 28, f"Z<sub>{i}</sub>", fill=GREEN, stroke=GREEN_S, fs=9)
    d.lbl(ax+10, z_y + 30, 400, 14, "Encoded Tokens  (N, 1152)", fs=8, color="#666")

    # PatchMerger
    pm = d.box(ax+130, z_y - 55, 130, 35, "<b>Patch Merger</b><br>2×2 spatial", fill=GREEN, stroke=GREEN_S, fs=9)
    d.lbl(ax+130, z_y - 20, 130, 14, "(N/4, 3584)", fs=8, color=GREEN_S)

    # visual_tokens output
    vt = d.box(ax+145, z_y - 110, 100, 32, "<b>visual_tokens</b>", fill=GREEN, stroke=GREEN_S, fs=10, bold=True)

    # DeepStack output (side)
    ds_out = d.box(ax+310, z_y - 100, 120, 50,
        "<b>DeepStack<br>Features</b><br>3×(N/4, 3584)",
        fill=PINK, stroke=PINK_S, fs=9, bold=True)
    d.lbl(ax+310, z_y - 48, 120, 14, "→ LLM Decoder", fs=8, color=PINK_S)

    # ═══════════════════════════════════════════
    # (b) VisionBlock Detail — CENTER
    # ═══════════════════════════════════════════
    bx_start = 520; by_start = 40
    d.lbl(bx_start, by_start, 300, 20, "<b>(b) VisionBlock (×27)</b>", fs=12, color="#333", align="left")

    # Vertical layout bottom-to-top like reference
    col_x = bx_start + 60
    BW = 160; BH = 36

    # Z^(0) input
    cy = by_start + 480
    z0 = d.box(col_x, cy, BW, 30, "Z<sup>(l)</sup>", fill=GRAY, stroke=GRAY_S, fs=11, bold=True)

    # LayerNorm 1
    cy -= 50
    ln1 = d.box(col_x, cy, BW, BH, "<b>Layer Norm</b>", fill=YELLOW, stroke=YELLOW_S, fs=10)
    d.arr(z0, ln1)

    # Multi-Head Attention
    cy -= 55
    mha = d.box(col_x, cy, BW, BH+8, "<b>Window<br>Self-Attention</b>", fill=BLUE, stroke=BLUE_S, fs=10, bold=True)
    d.arr(ln1, mha)

    # 2D RoPE (side)
    rope_side = d.box(col_x + BW + 30, cy, 80, 36, "<b>2D<br>RoPE</b>", fill=ROPE_CLR, stroke=ROPE_S, fs=9, bold=True)
    d.arr(rope_side, mha, dash=True, color=ROPE_S)

    # ⊕ residual 1
    cy -= 45
    add1 = d.circle(col_x + BW//2 - 14, cy, 28, "⊕", fill=WHITE, stroke="#333")
    d.arr(mha, add1)

    # L× annotation
    d.lbl(col_x - 50, cy - 100, 40, 200, "L×", fs=16, color="#999", bold=True)

    # LayerNorm 2
    cy -= 50
    ln2 = d.box(col_x, cy, BW, BH, "<b>Layer Norm</b>", fill=YELLOW, stroke=YELLOW_S, fs=10)
    d.arr(add1, ln2)

    # MLP
    cy -= 55
    mlp = d.box(col_x, cy, BW, BH, "<b>MLP</b>", fill=ORANGE, stroke=ORANGE_S, fs=11, bold=True)
    d.arr(ln2, mlp)

    # ⊕ residual 2
    cy -= 45
    add2 = d.circle(col_x + BW//2 - 14, cy, 28, "⊕", fill=WHITE, stroke="#333")
    d.arr(mlp, add2)

    # Z^(L) output
    cy -= 40
    zl = d.box(col_x, cy, BW, 30, "Z<sup>(l+1)</sup>", fill=GRAY, stroke=GRAY_S, fs=11, bold=True)
    d.arr(add2, zl)

    # Group box around block
    d.grp(col_x - 25, cy - 10, BW + 120, by_start + 510 - cy + 20, "", fill="#F8F8FF", stroke=BLUE_S)

    # ═══════════════════════════════════════════
    # (c) VisionAttention Detail — RIGHT-CENTER
    # ═══════════════════════════════════════════
    cx_start = 880; cy_start = 40
    d.lbl(cx_start, cy_start, 400, 20, "<b>(c) Vision Attention</b>", fs=12, color="#333", align="left")

    ccy = cy_start + 440
    # Q, K, V inputs at bottom
    q_in = d.box(cx_start + 10, ccy, 55, 28, "<b>Q</b>", fill=PURPLE, stroke=PURPLE_S, fs=10, bold=True)
    k_in = d.box(cx_start + 80, ccy, 55, 28, "<b>K</b>", fill=PURPLE, stroke=PURPLE_S, fs=10, bold=True)
    v_in = d.box(cx_start + 150, ccy, 55, 28, "<b>V</b>", fill=PURPLE, stroke=PURPLE_S, fs=10, bold=True)

    # Linear projections
    ccy -= 48
    q_lin = d.box(cx_start + 10, ccy, 55, 32, "<b>Linear</b>", fill=PURPLE, stroke=PURPLE_S, fs=9)
    k_lin = d.box(cx_start + 80, ccy, 55, 32, "<b>Linear</b>", fill=PURPLE, stroke=PURPLE_S, fs=9)
    v_lin = d.box(cx_start + 150, ccy, 55, 32, "<b>Linear</b>", fill=PURPLE, stroke=PURPLE_S, fs=9)
    d.arr(q_in, q_lin)
    d.arr(k_in, k_lin)
    d.arr(v_in, v_lin)
    d.lbl(cx_start - 5, ccy + 35, 230, 14, "QKV = Linear(1152→3456) → split", fs=7, color="#666")

    # 2D RoPE on Q, K
    ccy -= 42
    q_rope = d.box(cx_start + 10, ccy, 55, 28, "<b>RoPE</b>", fill=ROPE_CLR, stroke=ROPE_S, fs=8)
    k_rope = d.box(cx_start + 80, ccy, 55, 28, "<b>RoPE</b>", fill=ROPE_CLR, stroke=ROPE_S, fs=8)
    d.arr(q_lin, q_rope)
    d.arr(k_lin, k_rope)
    d.lbl(cx_start + 155, ccy+5, 60, 16, "(no RoPE)", fs=7, color="#999")

    # Window Attention
    ccy -= 55
    wattn = d.box(cx_start, ccy, 220, 40,
        "<b>Window Attention</b><br>(cu_seqlens)", fill=BLUE, stroke=BLUE_S, fs=10)
    d.arr(q_rope, wattn)
    d.arr(k_rope, wattn)
    d.arr(v_lin, wattn)

    # Concatenate
    ccy -= 42
    concat = d.box(cx_start + 20, ccy, 180, 30, "<b>Reshape</b>", fill=YELLOW, stroke=YELLOW_S, fs=10)
    d.arr(wattn, concat)

    # Output Linear
    ccy -= 42
    out_lin = d.box(cx_start + 30, ccy, 160, 30, "<b>Output Proj</b>", fill=PURPLE, stroke=PURPLE_S, fs=10)
    d.arr(concat, out_lin)
    d.lbl(cx_start + 195, ccy+5, 100, 18, "Linear(1152)", fs=8, color=PURPLE_S)

    # Group
    d.grp(cx_start - 10, ccy - 10, 245, cy_start + 480 - ccy + 20, "", fill="#F0F0FF", stroke=BLUE_S)

    # ═══════════════════════════════════════════
    # (d) MLP Detail — FAR RIGHT TOP
    # ═══════════════════════════════════════════
    dx_start = 1180; dy_start = 40
    d.lbl(dx_start, dy_start, 200, 20, "<b>(d) Vision MLP</b>", fs=12, color="#333", align="left")

    ddy = dy_start + 280
    d_in = d.box(dx_start + 20, ddy, 120, 28, "input", fill=GRAY, stroke=GRAY_S, fs=10)

    ddy -= 42
    fc1 = d.box(dx_start + 20, ddy, 120, 30, "<b>FC1</b>", fill=ORANGE, stroke=ORANGE_S, fs=11, bold=True)
    d.arr(d_in, fc1)
    d.lbl(dx_start + 145, ddy+5, 100, 18, "1152→4304", fs=8, color=ORANGE_S)

    ddy -= 42
    gelu = d.box(dx_start + 20, ddy, 120, 30, "<b>GELU</b>", fill=YELLOW, stroke=YELLOW_S, fs=11, bold=True)
    d.arr(fc1, gelu)

    ddy -= 42
    fc2 = d.box(dx_start + 20, ddy, 120, 30, "<b>FC2</b>", fill=ORANGE, stroke=ORANGE_S, fs=11, bold=True)
    d.arr(gelu, fc2)
    d.lbl(dx_start + 145, ddy+5, 100, 18, "4304→1152", fs=8, color=ORANGE_S)

    ddy -= 40
    d_out = d.box(dx_start + 20, ddy, 120, 28, "output", fill=GRAY, stroke=GRAY_S, fs=10)
    d.arr(fc2, d_out)

    d.grp(dx_start + 10, ddy - 10, 140, dy_start + 320 - ddy + 10, "", fill="#FFF8F0", stroke=ORANGE_S)

    # ═══════════════════════════════════════════
    # (e) PatchMerger Detail — FAR RIGHT BOTTOM
    # ═══════════════════════════════════════════
    ex_start = 1180; ey_start = 370
    d.lbl(ex_start, ey_start, 250, 20, "<b>(e) Patch Merger</b>", fs=12, color="#333", align="left")

    eey = ey_start + 320
    e_in = d.box(ex_start + 10, eey, 140, 28, "tokens (N, 1152)", fill=GRAY, stroke=GRAY_S, fs=9)

    eey -= 42
    e_ln = d.box(ex_start + 10, eey, 140, 30, "<b>Layer Norm</b>", fill=YELLOW, stroke=YELLOW_S, fs=10)
    d.arr(e_in, e_ln)

    eey -= 45
    e_merge = d.box(ex_start + 10, eey, 140, 32, "<b>2×2 Spatial<br>Merge</b>", fill=GREEN, stroke=GREEN_S, fs=10)
    d.arr(e_ln, e_merge)
    d.lbl(ex_start + 155, eey+5, 100, 20, "N→N/4<br>1152→4608", fs=8, color=GREEN_S)

    eey -= 42
    e_fc1 = d.box(ex_start + 10, eey, 140, 30, "<b>FC1</b>", fill=GREEN, stroke=GREEN_S, fs=10)
    d.arr(e_merge, e_fc1)
    d.lbl(ex_start + 155, eey+5, 90, 18, "4608→4608", fs=8, color=GREEN_S)

    eey -= 40
    e_gelu = d.box(ex_start + 10, eey, 140, 30, "<b>GELU</b>", fill=YELLOW, stroke=YELLOW_S, fs=10)
    d.arr(e_fc1, e_gelu)

    eey -= 40
    e_fc2 = d.box(ex_start + 10, eey, 140, 30, "<b>FC2</b>", fill=GREEN, stroke=GREEN_S, fs=10)
    d.arr(e_gelu, e_fc2)
    d.lbl(ex_start + 155, eey+5, 100, 18, "4608→d<sub>LLM</sub>", fs=8, color=GREEN_S)

    eey -= 40
    e_out = d.box(ex_start + 10, eey, 140, 28, "visual_tokens<br>(N/4, 3584)", fill=GREEN, stroke=GREEN_S, fs=9, bold=True)
    d.arr(e_fc2, e_out)

    d.grp(ex_start, eey - 10, 165, ey_start + 360 - eey + 10, "", fill="#F0FFF0", stroke=GREEN_S)

    # ═══════════════════════════════════════════
    # (f) DeepStack — BOTTOM CENTER
    # ═══════════════════════════════════════════
    fx_start = 520; fy_start = 540
    d.lbl(fx_start, fy_start, 400, 20, "<b>(f) DeepStack Feature Injection (NEW in Qwen3-VL)</b>", fs=12, color="#333", align="left")

    fy = fy_start + 30

    # Vision Encoder blocks
    for i, lnum in enumerate([8, 16, 24]):
        bx = fx_start + i * 150
        blk = d.box(bx, fy, 120, 36,
            f"<b>Block {lnum}</b><br>output", fill=PINK, stroke=PINK_S, fs=9)

    # Arrows down to PatchMergers
    fy += 55
    for i in range(3):
        bx = fx_start + i * 150
        pm = d.box(bx, fy, 120, 36,
            "<b>PatchMerger</b><br>(postshuffle)", fill=PINK, stroke=PINK_S, fs=9)

    # Merge into list
    fy += 55
    feat_list = d.box(fx_start + 100, fy, 250, 32,
        "<b>deepstack_features</b>  [3 × (N/4, 3584)]", fill=PINK, stroke=PINK_S, fs=10, bold=True)

    # Arrow to LLM
    fy += 50
    llm_inject = d.box(fx_start + 60, fy, 330, 40,
        "<b>LLM Decoder (early layers)</b><br>"
        "hidden_states[vis_pos] += deepstack_feat[i]",
        fill="#E8F5E9", stroke=GREEN_S, fs=10)
    d.arr(feat_list, llm_inject)

    # ═══════════════════════════════════════════
    # Legend
    # ═══════════════════════════════════════════
    lx = 30; ly = 540
    d.lbl(lx, ly, 100, 18, "<b>Legend:</b>", fs=10, color="#333", align="left", bold=True)
    items = [
        (BLUE, BLUE_S, "Attention"),
        (ORANGE, ORANGE_S, "MLP / FF"),
        (PURPLE, PURPLE_S, "Linear Proj"),
        (YELLOW, YELLOW_S, "LayerNorm / Op"),
        (GREEN, GREEN_S, "Merger / Output"),
        (ROPE_CLR, ROPE_S, "Pos Encoding"),
        (PINK, PINK_S, "DeepStack"),
    ]
    for i, (fill, stroke, label) in enumerate(items):
        d.box(lx, ly + 22 + i * 26, 18, 18, "", fill=fill, stroke=stroke, fs=8)
        d.lbl(lx + 22, ly + 22 + i * 26, 100, 18, label, fs=9, color="#333", align="left")

    return d


if __name__ == "__main__":
    import os
    path = "/home/perelman/.openclaw/workspace/qwen_review/d2_vision_encoder_detail.drawio"
    xml = build().to_xml()
    with open(path, "w") as f:
        f.write(xml)
    print(f"✅ {path} ({os.path.getsize(path)} bytes)")
