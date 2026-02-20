#!/usr/bin/env python3
"""Qwen3-VL Vision Encoder — Paper figure (bottom-to-top).

4-panel layout matching ViT reference.png:
  (a) Overall Pipeline  (b) VisionBlock ×27  (c) VisionAttention  (d) PatchMerger

Clean pastel boxes, minimal text, ⊕ residual, dimension annotations.
All data flows BOTTOM → TOP.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
import os

# ── Colors (draw.io default pastels) ──
BLUE     = "#DAE8FC"; BLUE_S   = "#6C8EBF"
ORANGE   = "#FFE6CC"; ORANGE_S = "#D79B00"
PURPLE   = "#E1D5E7"; PURPLE_S = "#9673A6"
GREEN    = "#D5E8D4"; GREEN_S  = "#82B366"
YELLOW   = "#FFF2CC"; YELLOW_S = "#D6B656"
PINK     = "#F8CECC"; PINK_S   = "#B85450"
GRAY     = "#F5F5F5"; GRAY_S   = "#999999"
GRAY_D   = "#E6E6E6"; GRAY_DS  = "#666666"
WHITE    = "#FFFFFF"
ROPE     = "#E6D0DE"; ROPE_S   = "#996185"


class D:
    """Lightweight draw.io XML builder."""

    def __init__(self, name, pw=1600, ph=880):
        self.name, self.pw, self.ph = name, pw, ph
        self.cells = []
        self._id = 2

    def _n(self):
        i = self._id; self._id += 1; return str(i)

    # ── primitives ──

    def box(self, x, y, w, h, label, fill=WHITE, stroke="#333333",
            fs=10, bold=False):
        cid = self._n()
        s = (f"rounded=1;arcSize=12;whiteSpace=wrap;html=1;"
             f"fillColor={fill};strokeColor={stroke};"
             f"fontSize={fs};fontStyle={'1' if bold else '0'};"
             f"strokeWidth=1.5;")
        self.cells.append(dict(id=cid, value=label, style=s,
                               vertex=True, x=x, y=y, w=w, h=h))
        return cid

    def circ(self, cx, cy, sz=26, label="⊕"):
        cid = self._n()
        s = (f"ellipse;whiteSpace=wrap;html=1;aspect=fixed;"
             f"fillColor={WHITE};strokeColor=#333333;"
             f"fontSize=13;fontStyle=1;strokeWidth=1.5;")
        self.cells.append(dict(id=cid, value=label, style=s, vertex=True,
                               x=cx - sz // 2, y=cy - sz // 2, w=sz, h=sz))
        return cid

    def lbl(self, x, y, w, h, text, fs=9, color="#333333",
            align="center", bold=False):
        cid = self._n()
        s = (f"text;html=1;align={align};verticalAlign=middle;"
             f"strokeColor=none;fillColor=none;"
             f"fontSize={fs};fontColor={color};"
             f"fontStyle={'1' if bold else '0'};")
        self.cells.append(dict(id=cid, value=text, style=s, vertex=True,
                               x=x, y=y, w=w, h=h))
        return cid

    def grp(self, x, y, w, h, fill=GRAY, stroke="#CCCCCC"):
        cid = self._n()
        s = (f"rounded=1;arcSize=6;whiteSpace=wrap;html=1;"
             f"fillColor={fill};strokeColor={stroke};"
             f"strokeWidth=1;opacity=40;fontSize=0;")
        self.cells.append(dict(id=cid, value="", style=s, vertex=True,
                               x=x, y=y, w=w, h=h))
        return cid

    def arr(self, src, tgt, label="", color="#333333",
            exX=0.5, exY=0, enX=0.5, enY=1):
        """Arrow with configurable anchors. Default: upward (top→bottom)."""
        cid = self._n()
        s = (f"html=1;strokeColor={color};endArrow=blockThin;endFill=1;"
             f"strokeWidth=1.5;fontSize=8;fontColor=#666666;"
             f"exitX={exX};exitY={exY};exitDx=0;exitDy=0;"
             f"entryX={enX};entryY={enY};entryDx=0;entryDy=0;")
        self.cells.append(dict(id=cid, value=label, style=s,
                               edge=True, source=src, target=tgt))
        return cid

    def arr_orth(self, src, tgt, label="", color="#333333", dash=False,
                 exX=0.5, exY=0.5, enX=0.5, enY=0.5):
        """Orthogonal-routed arrow."""
        cid = self._n()
        dd = "dashed=1;dashPattern=6 3;" if dash else ""
        s = (f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;{dd}"
             f"strokeColor={color};endArrow=blockThin;endFill=1;"
             f"strokeWidth=1.5;fontSize=8;fontColor=#666666;"
             f"exitX={exX};exitY={exY};exitDx=0;exitDy=0;"
             f"entryX={enX};entryY={enY};entryDx=0;entryDy=0;")
        self.cells.append(dict(id=cid, value=label, style=s,
                               edge=True, source=src, target=tgt))
        return cid

    def arr_bypass(self, src, tgt, color="#999999"):
        """Left-side bypass for residual connections."""
        cid = self._n()
        s = (f"edgeStyle=orthogonalEdgeStyle;curved=1;html=1;"
             f"strokeColor={color};endArrow=blockThin;endFill=1;"
             f"strokeWidth=1;fontSize=0;"
             f"exitX=0;exitY=0.5;exitDx=0;exitDy=0;"
             f"entryX=0;entryY=0.5;entryDx=0;entryDy=0;")
        self.cells.append(dict(id=cid, value="", style=s,
                               edge=True, source=src, target=tgt))
        return cid

    # ── serialise ──

    def to_xml(self):
        root = ET.Element("mxfile", host="app.diagrams.net")
        diag = ET.SubElement(root, "diagram", name=self.name, id="d")
        m = ET.SubElement(diag, "mxGraphModel",
                          dx="1422", dy="900", grid="1", gridSize="10",
                          guides="1", tooltips="1", connect="1", arrows="1",
                          fold="1", page="1", pageScale="1",
                          pageWidth=str(self.pw), pageHeight=str(self.ph),
                          background="#FFFFFF")
        rc = ET.SubElement(m, "root")
        ET.SubElement(rc, "mxCell", id="0")
        ET.SubElement(rc, "mxCell", id="1", parent="0")

        for c in self.cells:
            a = {"id": c["id"], "style": c["style"], "parent": "1"}
            if c.get("value"):
                a["value"] = c["value"]
            if c.get("vertex"):
                a["vertex"] = "1"
            if c.get("edge"):
                a["edge"] = "1"
                if c.get("source"):
                    a["source"] = c["source"]
                if c.get("target"):
                    a["target"] = c["target"]
            el = ET.SubElement(rc, "mxCell", **a)
            if c.get("vertex") and "x" in c:
                ET.SubElement(el, "mxGeometry",
                              x=str(c["x"]), y=str(c["y"]),
                              width=str(c["w"]), height=str(c["h"]),
                              **{"as": "geometry"})
            elif c.get("edge"):
                ET.SubElement(el, "mxGeometry", relative="1",
                              **{"as": "geometry"})

        return minidom.parseString(
            ET.tostring(root, encoding="unicode")
        ).toprettyxml(indent="  ")


# ═══════════════════════════════════════════════════════════════════
#  BUILD
# ═══════════════════════════════════════════════════════════════════

def build():
    d = D("Qwen3-VL Vision Encoder", pw=1600, ph=900)

    # ───────────────────────────────────────────
    # (a) Overall Vision Encoder Pipeline
    #     x: 25–475   (w=450)
    # ───────────────────────────────────────────
    AX = 25; AW = 450
    ACX = AX + AW // 2          # center = 250
    BW = 180                    # standard box width

    d.lbl(AX, 10, AW, 22,
          "<b>(a) Vision Encoder Pipeline</b>", fs=12, color="#333333")

    # -- Image / Video  (BOTTOM) --
    img = d.box(ACX - BW // 2, 815, BW, 42,
                "<b>Image / Video</b><br>T × H × W × 3",
                fill=GRAY, stroke=GRAY_S, fs=10)

    # -- PatchEmbed --
    pe = d.box(ACX - BW // 2, 745, BW, 36,
               "<b>PatchEmbed</b><br>Conv3d [2, 16, 16]",
               fill=PURPLE, stroke=PURPLE_S, fs=9)
    d.arr(img, pe)

    # -- Patch tokens  E₀ … E₇ --
    N = 8; TW = 40; TH = 22; TG = 3
    row_w = N * TW + (N - 1) * TG
    tx0 = ACX - row_w // 2
    ey = 690
    e_ids = []
    for i in range(N):
        eid = d.box(tx0 + i * (TW + TG), ey, TW, TH,
                    f"E<sub>{i}</sub>", fill=PINK, stroke=PINK_S, fs=9)
        e_ids.append(eid)
    d.arr(pe, e_ids[N // 2])
    d.lbl(tx0 + row_w + 8, ey, 90, TH,
          "+ 2D Pos Embed", fs=8, color=ROPE_S)

    # -- Vision Transformer  (big block) --
    VT_TOP = 400; VT_H = 260
    vt = d.grp(AX + 30, VT_TOP, AW - 60, VT_H,
               fill="#EAF0FA", stroke=BLUE_S)
    d.lbl(AX + 40, VT_TOP + 8, AW - 80, 18,
          "<b>Vision Transformer</b>", fs=11, color=BLUE_S, bold=True)

    # Mini-blocks inside
    mb_n = 9; mbw = 34; mbh = 155
    total_mb = mb_n * mbw + (mb_n - 1) * 6
    mbx0 = ACX - total_mb // 2
    mby = VT_TOP + 35
    ds_map = {2: "8", 5: "16", 7: "24"}
    for i in range(mb_n):
        mbx = mbx0 + i * (mbw + 6)
        is_ds = i in ds_map
        bl = f"<b>{ds_map[i]}</b>" if is_ds else f"{i * 3}"
        d.box(mbx, mby, mbw, mbh, bl,
              fill=PINK if is_ds else BLUE,
              stroke=PINK_S if is_ds else BLUE_S, fs=7)

    d.lbl(AX + 40, VT_TOP + VT_H - 40, AW - 80, 30,
          "×27 blocks  |  "
          "<font color='#B85450'>■</font> DeepStack [8, 16, 24]  |  "
          "<font color='#6C8EBF'>■</font> Regular",
          fs=8, color="#666666")

    # arrow E → VT
    d.arr(e_ids[N // 2], vt, enY=1)

    # -- Z tokens  Z₀ … Z₇ --
    zy = 350
    z_ids = []
    for i in range(N):
        zid = d.box(tx0 + i * (TW + TG), zy, TW, TH,
                    f"Z<sub>{i}</sub>", fill=GREEN, stroke=GREEN_S, fs=9)
        z_ids.append(zid)
    d.arr(vt, z_ids[N // 2], exY=0)

    # -- PatchMerger --
    pm = d.box(ACX - BW // 2, 290, BW, 36,
               "<b>PatchMerger</b><br>2×2 spatial → MLP",
               fill=GREEN, stroke=GREEN_S, fs=9)
    d.arr(z_ids[N // 2], pm)

    # -- visual_tokens  (TOP output) --
    vt_out = d.box(ACX - 95, 225, 190, 32,
                   "<b>visual_tokens</b>  (N/4, 3584)",
                   fill=GREEN, stroke=GREEN_S, fs=10, bold=True)
    d.arr(pm, vt_out)

    # -- DeepStack side branch --
    ds = d.box(AX + AW - 130, 140, 125, 52,
               "<b>DeepStack</b><br>3 × (N/4, 3584)<br>"
               "<font point-size='8'>→ LLM early layers</font>",
               fill=PINK, stroke=PINK_S, fs=9)

    # arrow: VT right side → DeepStack
    d.arr_orth(vt, ds, color=PINK_S,
               exX=0.85, exY=0, enX=0.5, enY=1)

    # "To LLM" label
    d.lbl(ACX - 50, 190, 100, 20,
          "↑ To LLM Decoder", fs=9, color=GREEN_S, bold=True)

    # ───────────────────────────────────────────
    # (b) VisionBlock (×27)
    #     x: 510–750   (w=240)
    # ───────────────────────────────────────────
    BPX = 510; BPW = 240
    BCX = BPX + BPW // 2        # center = 630
    VBW = 155; VBH = 34

    d.lbl(BPX, 10, BPW, 22,
          "<b>(b) VisionBlock (×27)</b>", fs=12, color="#333333")

    # Background
    d.grp(BPX + 20, 220, BPW - 40, 640, fill="#F5F7FF", stroke=BLUE_S)

    # L× bracket
    d.lbl(BPX + 22, 400, 30, 120,
          "<b>L×</b>", fs=18, color="#AAAAAA", bold=True)

    # Z^(l) input  — BOTTOM
    zl_in = d.box(BCX - VBW // 2, 810, VBW, 28,
                  "Z<sup>(l)</sup>", fill=GRAY_D, stroke=GRAY_DS,
                  fs=11, bold=True)

    # LayerNorm 1
    ln1 = d.box(BCX - VBW // 2, 730, VBW, VBH,
                "<b>LayerNorm</b>", fill=YELLOW, stroke=YELLOW_S, fs=10)
    d.arr(zl_in, ln1)

    # Window Self-Attention
    attn = d.box(BCX - VBW // 2, 650, VBW, 42,
                 "<b>Window<br>Self-Attention</b>",
                 fill=BLUE, stroke=BLUE_S, fs=10, bold=True)
    d.arr(ln1, attn)

    # 2D RoPE (side)
    rope_b = d.box(BCX + VBW // 2 + 12, 656, 60, 30,
                   "<b>2D<br>RoPE</b>", fill=ROPE, stroke=ROPE_S,
                   fs=8, bold=True)
    d.arr_orth(rope_b, attn, dash=True, color=ROPE_S,
               exX=0, exY=0.5, enX=1, enY=0.5)

    # ⊕ residual 1
    add1 = d.circ(BCX, 600)
    d.arr(attn, add1)
    d.arr_bypass(zl_in, add1)   # residual skip

    # LayerNorm 2
    ln2 = d.box(BCX - VBW // 2, 530, VBW, VBH,
                "<b>LayerNorm</b>", fill=YELLOW, stroke=YELLOW_S, fs=10)
    d.arr(add1, ln2)

    # MLP
    mlp_b = d.box(BCX - VBW // 2, 455, VBW, 38,
                  "<b>MLP</b><br>FC → act → FC",
                  fill=ORANGE, stroke=ORANGE_S, fs=10, bold=True)
    d.arr(ln2, mlp_b)

    # ⊕ residual 2
    add2 = d.circ(BCX, 408)
    d.arr(mlp_b, add2)
    d.arr_bypass(add1, add2)    # residual skip

    # Z^(l+1) output  — TOP
    zl_out = d.box(BCX - VBW // 2, 340, VBW, 28,
                   "Z<sup>(l+1)</sup>", fill=GRAY_D, stroke=GRAY_DS,
                   fs=11, bold=True)
    d.arr(add2, zl_out)

    # DeepStack annotation
    d.lbl(BCX - VBW // 2, 260, VBW, 40,
          "<font color='#B85450'>At layers [8, 16, 24]:<br>"
          "output → PatchMerger<br>→ deepstack_features</font>",
          fs=8, color=PINK_S)

    # ───────────────────────────────────────────
    # (c) VisionAttention
    #     x: 785–1085   (w=300)
    # ───────────────────────────────────────────
    CX = 785; CW = 300
    CCX = CX + CW // 2          # center = 935

    d.lbl(CX, 10, CW, 22,
          "<b>(c) VisionAttention</b>", fs=12, color="#333333")

    # Background
    d.grp(CX + 10, 380, CW - 20, 470, fill="#F0F0FF", stroke=BLUE_S)

    # Q, K, V inputs  — BOTTOM
    qw = 65; qg = 18
    total_q = 3 * qw + 2 * qg
    qx0 = CCX - total_q // 2

    q_in = d.box(qx0, 810, qw, 25,
                 "<b>Q</b>", fill=PURPLE, stroke=PURPLE_S, fs=10, bold=True)
    k_in = d.box(qx0 + qw + qg, 810, qw, 25,
                 "<b>K</b>", fill=PURPLE, stroke=PURPLE_S, fs=10, bold=True)
    v_in = d.box(qx0 + 2 * (qw + qg), 810, qw, 25,
                 "<b>V</b>", fill=PURPLE, stroke=PURPLE_S, fs=10, bold=True)

    # Linear projections
    q_lin = d.box(qx0, 755, qw, 30,
                  "<b>Linear</b>", fill=PURPLE, stroke=PURPLE_S, fs=9)
    k_lin = d.box(qx0 + qw + qg, 755, qw, 30,
                  "<b>Linear</b>", fill=PURPLE, stroke=PURPLE_S, fs=9)
    v_lin = d.box(qx0 + 2 * (qw + qg), 755, qw, 30,
                  "<b>Linear</b>", fill=PURPLE, stroke=PURPLE_S, fs=9)
    d.arr(q_in, q_lin); d.arr(k_in, k_lin); d.arr(v_in, v_lin)

    # Dimension annotation
    d.lbl(CX + 10, 780, CW - 20, 16,
          "QKV = Linear(1152 → 3×1152)", fs=7, color="#666666")

    # 2D RoPE (Q, K only)
    q_rope = d.box(qx0, 700, qw, 28,
                   "<b>RoPE</b>", fill=ROPE, stroke=ROPE_S, fs=8, bold=True)
    k_rope = d.box(qx0 + qw + qg, 700, qw, 28,
                   "<b>RoPE</b>", fill=ROPE, stroke=ROPE_S, fs=8, bold=True)
    d.arr(q_lin, q_rope)
    d.arr(k_lin, k_rope)

    # V: no RoPE label
    d.lbl(qx0 + 2 * (qw + qg), 700, qw, 28,
          "<i>(pass)</i>", fs=7, color="#999999")

    # Window Attention  — wide box
    wa_w = 210
    wa = d.box(CCX - wa_w // 2, 615, wa_w, 42,
               "<b>Window Attention</b><br>cu_seqlens",
               fill=BLUE, stroke=BLUE_S, fs=10, bold=True)
    # Q → WA (left)
    d.arr(q_rope, wa, enX=0.25)
    # K → WA (center)
    d.arr(k_rope, wa, enX=0.5)
    # V → WA (right, from Linear directly)
    d.arr(v_lin, wa, enX=0.75)

    # Reshape
    rs_w = 160
    rs = d.box(CCX - rs_w // 2, 545, rs_w, 30,
               "<b>Reshape</b>  (num_heads → hidden)",
               fill=YELLOW, stroke=YELLOW_S, fs=9)
    d.arr(wa, rs)

    # Output Projection
    op = d.box(CCX - rs_w // 2, 480, rs_w, 30,
               "<b>Output Proj</b>  Linear(1152)",
               fill=PURPLE, stroke=PURPLE_S, fs=9)
    d.arr(rs, op)

    # output
    c_out = d.box(CCX - 60, 415, 120, 25,
                  "output", fill=GRAY_D, stroke=GRAY_DS, fs=10)
    d.arr(op, c_out)

    # ───────────────────────────────────────────
    # (d) PatchMerger
    #     x: 1120–1320   (w=200)
    # ───────────────────────────────────────────
    DX = 1120; DW = 200
    DCX = DX + DW // 2          # center = 1220
    DBW = 150

    d.lbl(DX, 10, DW, 22,
          "<b>(d) PatchMerger</b>", fs=12, color="#333333")

    # Background
    d.grp(DX + 10, 345, DW - 20, 510, fill="#F0FFF0", stroke=GREEN_S)

    # tokens input  — BOTTOM
    d_in = d.box(DCX - DBW // 2, 810, DBW, 28,
                 "tokens  (N, 1152)", fill=GRAY_D, stroke=GRAY_DS, fs=9)

    # LayerNorm
    d_ln = d.box(DCX - DBW // 2, 740, DBW, 32,
                 "<b>LayerNorm</b>", fill=YELLOW, stroke=YELLOW_S, fs=10)
    d.arr(d_in, d_ln)

    # 2×2 Spatial Merge
    d_merge = d.box(DCX - DBW // 2, 668, DBW, 38,
                    "<b>2×2 Spatial<br>Merge</b>",
                    fill=GREEN, stroke=GREEN_S, fs=10)
    d.arr(d_ln, d_merge)
    d.lbl(DCX + DBW // 2 + 5, 670, 80, 30,
          "N → N/4<br>1152 → 4608", fs=7, color=GREEN_S)

    # FC1
    d_fc1 = d.box(DCX - DBW // 2, 600, DBW, 32,
                  "<b>FC1</b>  (4608 → 4608)",
                  fill=ORANGE, stroke=ORANGE_S, fs=9, bold=True)
    d.arr(d_merge, d_fc1)

    # GELU
    d_gelu = d.box(DCX - DBW // 2, 540, DBW, 28,
                   "<b>GELU</b>", fill=YELLOW, stroke=YELLOW_S, fs=10)
    d.arr(d_fc1, d_gelu)

    # FC2
    d_fc2 = d.box(DCX - DBW // 2, 475, DBW, 32,
                  "<b>FC2</b>  (4608 → 3584)",
                  fill=ORANGE, stroke=ORANGE_S, fs=9, bold=True)
    d.arr(d_gelu, d_fc2)

    # visual_tokens output  — TOP
    d_out = d.box(DCX - DBW // 2, 395, DBW, 32,
                  "<b>visual_tokens</b><br>(N/4, 3584)",
                  fill=GREEN, stroke=GREEN_S, fs=9, bold=True)
    d.arr(d_fc2, d_out)

    # Annotation: where PatchMerger is used
    d.lbl(DX + 10, 355, DW - 20, 30,
          "Used for both final output<br>"
          "and DeepStack post-shuffle",
          fs=7, color="#666666")

    # ───────────────────────────────────────────
    # Legend  — bottom-right
    # ───────────────────────────────────────────
    lx = 1120; ly = 860
    items = [
        (BLUE, BLUE_S, "Attention"),
        (ORANGE, ORANGE_S, "MLP / FC"),
        (PURPLE, PURPLE_S, "Linear / Embed"),
        (YELLOW, YELLOW_S, "Norm / Op"),
        (GREEN, GREEN_S, "Merger / Output"),
        (ROPE, ROPE_S, "Pos Encoding"),
        (PINK, PINK_S, "DeepStack"),
    ]
    d.lbl(lx, ly, 60, 14, "<b>Legend:</b>", fs=8, color="#333333",
          align="left", bold=True)
    for i, (fill, stroke, label) in enumerate(items):
        bx = lx + 65 + i * 65
        d.box(bx, ly, 12, 12, "", fill=fill, stroke=stroke, fs=6)
        d.lbl(bx + 14, ly - 1, 50, 14, label, fs=7, color="#333333",
              align="left")

    return d


if __name__ == "__main__":
    out = "/home/perelman/.openclaw/workspace/qwen_review/d2_vision_encoder_detail.drawio"
    xml = build().to_xml()
    with open(out, "w") as f:
        f.write(xml)
    sz = os.path.getsize(out)
    print(f"✅ {out}  ({sz:,} bytes)")
