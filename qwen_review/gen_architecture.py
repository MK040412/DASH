#!/usr/bin/env python3
"""Qwen3-VL Full Architecture — Paper figure.

Single comprehensive diagram:
  (a) Overall: Vision Encoder + LLM Decoder + DeepStack cross-connections
  (b) VisionBlock detail (×27)
  (c) Decoder Layer detail (×N)

Bottom-to-top flow throughout. Publication-quality.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
import os

# ── Colors (draw.io pastels) ──
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
TEAL     = "#D4E6F1"; TEAL_S   = "#2471A3"
LGREEN   = "#E8F5E9"; LGREEN_S = "#388E3C"


class D:
    def __init__(self, name, pw=1800, ph=950):
        self.name, self.pw, self.ph = name, pw, ph
        self.cells = []; self._id = 2

    def _n(self):
        i = self._id; self._id += 1; return str(i)

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

    def grp(self, x, y, w, h, fill=GRAY, stroke="#CCCCCC", label=""):
        cid = self._n()
        s = (f"rounded=1;arcSize=6;whiteSpace=wrap;html=1;"
             f"fillColor={fill};strokeColor={stroke};"
             f"strokeWidth=1;opacity=40;fontSize=10;"
             f"fontColor=#666666;verticalAlign=top;fontStyle=0;")
        self.cells.append(dict(id=cid, value=label, style=s, vertex=True,
                               x=x, y=y, w=w, h=h))
        return cid

    def arr(self, src, tgt, label="", color="#333333",
            exX=0.5, exY=0, enX=0.5, enY=1, sw=1.5, dash=False):
        cid = self._n()
        dd = "dashed=1;dashPattern=6 3;" if dash else ""
        s = (f"html=1;{dd}strokeColor={color};endArrow=blockThin;endFill=1;"
             f"strokeWidth={sw};fontSize=8;fontColor=#666666;"
             f"exitX={exX};exitY={exY};exitDx=0;exitDy=0;"
             f"entryX={enX};entryY={enY};entryDx=0;entryDy=0;")
        self.cells.append(dict(id=cid, value=label, style=s,
                               edge=True, source=src, target=tgt))
        return cid

    def arr_orth(self, src, tgt, label="", color="#333333",
                 exX=0.5, exY=0.5, enX=0.5, enY=0.5, sw=1.5, dash=False):
        cid = self._n()
        dd = "dashed=1;dashPattern=6 3;" if dash else ""
        s = (f"edgeStyle=orthogonalEdgeStyle;curved=1;html=1;{dd}"
             f"strokeColor={color};endArrow=blockThin;endFill=1;"
             f"strokeWidth={sw};fontSize=8;fontColor=#666666;"
             f"exitX={exX};exitY={exY};exitDx=0;exitDy=0;"
             f"entryX={enX};entryY={enY};entryDx=0;entryDy=0;")
        self.cells.append(dict(id=cid, value=label, style=s,
                               edge=True, source=src, target=tgt))
        return cid

    def arr_bypass(self, src, tgt, color="#999999"):
        cid = self._n()
        s = (f"edgeStyle=orthogonalEdgeStyle;curved=1;html=1;"
             f"strokeColor={color};endArrow=blockThin;endFill=1;"
             f"strokeWidth=1;fontSize=0;"
             f"exitX=0;exitY=0.5;exitDx=0;exitDy=0;"
             f"entryX=0;entryY=0.5;entryDx=0;entryDy=0;")
        self.cells.append(dict(id=cid, value="", style=s,
                               edge=True, source=src, target=tgt))
        return cid

    def to_xml(self):
        root = ET.Element("mxfile", host="app.diagrams.net")
        diag = ET.SubElement(root, "diagram", name=self.name, id="arch")
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
            if c.get("value"): a["value"] = c["value"]
            if c.get("vertex"): a["vertex"] = "1"
            if c.get("edge"):
                a["edge"] = "1"
                if c.get("source"): a["source"] = c["source"]
                if c.get("target"): a["target"] = c["target"]
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


def build():
    d = D("Qwen3-VL Architecture", pw=1800, ph=950)

    # ════════════════════════════════════════════════════════
    #  (a) OVERALL ARCHITECTURE — Vision + LLM + DeepStack
    # ════════════════════════════════════════════════════════

    d.lbl(20, 5, 700, 22,
          "<b>(a) Qwen3-VL Overall Architecture</b>", fs=13, color="#333333",
          align="left")

    # ────── VISION ENCODER (left column) ──────
    VX = 40; VW = 170; VCX = VX + VW // 2    # center = 125

    # Background
    d.grp(VX - 15, 35, VW + 30, 860, fill="#F0F4FF", stroke=BLUE_S,
          label="   Vision Encoder")

    # Image/Video — BOTTOM
    img = d.box(VCX - 80, 845, 160, 36,
                "<b>Image / Video</b><br>T × H × W × 3",
                fill=GRAY_D, stroke=GRAY_DS, fs=9)

    # PatchEmbed
    pe = d.box(VCX - 80, 785, 160, 34,
               "<b>PatchEmbed</b><br>Conv3d [2, 16, 16]",
               fill=PURPLE, stroke=PURPLE_S, fs=9)
    d.arr(img, pe)

    # + Abs Pos Embed
    ape = d.box(VCX - 75, 730, 150, 30,
                "<b>+ Abs Pos Embed</b><br>Bilinear Interp",
                fill=ROPE, stroke=ROPE_S, fs=8)
    d.arr(pe, ape)
    d.lbl(VCX + 80, 730, 50, 30, "(N, 1152)", fs=7, color="#666666")

    # ── ViT Block Stack ──
    # The key abstraction: grouped blocks with DeepStack taps
    vit_top = 280; vit_bot = 710
    d.grp(VX + 5, vit_top, VW - 10, vit_bot - vit_top,
          fill="#E8EEFF", stroke=BLUE_S, label="  ViT ×27")

    # Block groups — bottom to top
    bw = VW - 30; bx = VX + 15
    blocks = {}

    # Block 0–7 (bottom)
    blocks["0-7"] = d.box(bx, 640, bw, 55,
        "<b>Block 0–7</b><br>×8", fill=BLUE, stroke=BLUE_S, fs=9)

    # Block 8 (DeepStack tap)
    blocks["8"] = d.box(bx, 590, bw, 35,
        "<b>Block 8</b>", fill=PINK, stroke=PINK_S, fs=10, bold=True)

    # Block 9–15
    blocks["9-15"] = d.box(bx, 535, bw, 42,
        "<b>Block 9–15</b><br>×7", fill=BLUE, stroke=BLUE_S, fs=9)

    # Block 16 (DeepStack tap)
    blocks["16"] = d.box(bx, 488, bw, 35,
        "<b>Block 16</b>", fill=PINK, stroke=PINK_S, fs=10, bold=True)

    # Block 17–23
    blocks["17-23"] = d.box(bx, 433, bw, 42,
        "<b>Block 17–23</b><br>×7", fill=BLUE, stroke=BLUE_S, fs=9)

    # Block 24 (DeepStack tap)
    blocks["24"] = d.box(bx, 386, bw, 35,
        "<b>Block 24</b>", fill=PINK, stroke=PINK_S, fs=10, bold=True)

    # Block 25–26
    blocks["25-26"] = d.box(bx, 340, bw, 34,
        "<b>Block 25–26</b><br>×2", fill=BLUE, stroke=BLUE_S, fs=8)

    # Arrows within ViT stack
    d.arr(ape, blocks["0-7"])
    for src, tgt in [("0-7","8"), ("8","9-15"), ("9-15","16"),
                     ("16","17-23"), ("17-23","24"), ("24","25-26")]:
        d.arr(blocks[src], blocks[tgt], sw=1, color="#999999")

    # Final PatchMerger
    fpm = d.box(VCX - 80, 238, 160, 34,
                "<b>PatchMerger</b><br>LN → merge(2×2) → MLP",
                fill=GREEN, stroke=GREEN_S, fs=8)
    d.arr(blocks["25-26"], fpm)

    # visual_tokens
    vtok = d.box(VCX - 70, 185, 140, 30,
                 "<b>visual_tokens</b><br>(N/4, 3584)",
                 fill=GREEN, stroke=GREEN_S, fs=9, bold=True)
    d.arr(fpm, vtok)

    # ── Arrow: visual_tokens → LLM input (curved) ──
    # This will connect to the LLM sequence via a curved arrow

    # ────── DEEPSTACK MERGERS (center column) ──────
    MX = 280; MW = 140; MCX = MX + MW // 2

    d.lbl(MX - 10, 310, MW + 20, 20,
          "<b>DeepStack</b>", fs=10, color=PINK_S, bold=True)

    # Three mergers aligned with their source blocks
    m0 = d.box(MX, 585, MW, 40,
               "<b>Merger₀</b><br><font point-size='7'>postshuffle</font>",
               fill=PINK, stroke=PINK_S, fs=9)
    m1 = d.box(MX, 483, MW, 40,
               "<b>Merger₁</b><br><font point-size='7'>postshuffle</font>",
               fill=PINK, stroke=PINK_S, fs=9)
    m2 = d.box(MX, 381, MW, 40,
               "<b>Merger₂</b><br><font point-size='7'>postshuffle</font>",
               fill=PINK, stroke=PINK_S, fs=9)

    # Arrows from ViT blocks → mergers
    d.arr_orth(blocks["8"], m0, color=PINK_S, sw=2,
               exX=1, exY=0.5, enX=0, enY=0.5)
    d.arr_orth(blocks["16"], m1, color=PINK_S, sw=2,
               exX=1, exY=0.5, enX=0, enY=0.5)
    d.arr_orth(blocks["24"], m2, color=PINK_S, sw=2,
               exX=1, exY=0.5, enX=0, enY=0.5)

    # Dimension labels for mergers
    d.lbl(MX, 625, MW, 14,
          "(N, 1152) → (N/4, 3584)", fs=6, color=PINK_S)
    d.lbl(MX, 335, MW, 14,
          "LN(4608) → FC → GELU → FC", fs=6, color="#666666")

    # ────── LLM DECODER (right column) ──────
    LX = 490; LW = 170; LCX = LX + LW // 2   # center = 575

    d.grp(LX - 15, 35, LW + 30, 860, fill="#F0FFF5", stroke=GREEN_S,
          label="   LLM Decoder")

    # Input — BOTTOM
    llm_in = d.box(LCX - 85, 845, 170, 36,
                   "Text tokens + <b>[IMG]</b><br>input_ids",
                   fill=GRAY_D, stroke=GRAY_DS, fs=9)

    # Token Embedding
    tok_emb = d.box(LCX - 80, 790, 160, 30,
                    "<b>Token Embedding</b><br>nn.Embedding",
                    fill=PURPLE, stroke=PURPLE_S, fs=9)
    d.arr(llm_in, tok_emb)

    # masked_scatter: visual_tokens replace [IMG]
    scatter_box = d.box(LCX - 80, 738, 160, 30,
                        "<b>masked_scatter</b><br>[IMG] → visual_tokens",
                        fill=GREEN, stroke=GREEN_S, fs=8)
    d.arr(tok_emb, scatter_box)

    # Arrow from visual_tokens → scatter
    d.arr_orth(vtok, scatter_box, color=GREEN_S, sw=2,
               exX=1, exY=0.5, enX=0, enY=0.5,
               label="visual_tokens")

    # Dimension after scatter
    d.lbl(LCX + 85, 738, 70, 30, "(B, L, 3584)", fs=7, color="#666666")

    # ── Decoder Layer Stack ──
    llm_top = 280; llm_bot = 720
    d.grp(LX + 5, llm_top, LW - 10, llm_bot - llm_top,
          fill="#E8F5E9", stroke=GREEN_S, label="  Decoder ×N")

    layers = {}

    # Layer 0 (DeepStack injection)
    layers["0"] = d.box(LX + 15, 640, LW - 30, 42,
        "<b>Layer 0</b><br>DecoderLayer", fill=LGREEN, stroke=LGREEN_S, fs=9)

    # ⊕ after Layer 0 (DeepStack injection point)
    inj0 = d.circ(LCX, 612, 24, "⊕")
    d.arr(layers["0"], inj0)

    # Layer 1
    layers["1"] = d.box(LX + 15, 545, LW - 30, 42,
        "<b>Layer 1</b><br>DecoderLayer", fill=LGREEN, stroke=LGREEN_S, fs=9)
    d.arr(inj0, layers["1"])

    inj1 = d.circ(LCX, 517, 24, "⊕")
    d.arr(layers["1"], inj1)

    # Layer 2
    layers["2"] = d.box(LX + 15, 450, LW - 30, 42,
        "<b>Layer 2</b><br>DecoderLayer", fill=LGREEN, stroke=LGREEN_S, fs=9)
    d.arr(inj1, layers["2"])

    inj2 = d.circ(LCX, 422, 24, "⊕")
    d.arr(layers["2"], inj2)

    # Layer 3–N
    layers["3-N"] = d.box(LX + 15, 340, LW - 30, 60,
        "<b>Layer 3 – N</b><br>DecoderLayer<br>×(N-3)", fill=TEAL, stroke=TEAL_S, fs=9)
    d.arr(inj2, layers["3-N"])

    # Arrow from scatter → Layer 0
    d.arr(scatter_box, layers["0"])

    # DeepStack injection arrows: merger → ⊕
    d.arr_orth(m0, inj0, color=PINK_S, sw=2,
               exX=1, exY=0.5, enX=0, enY=0.5,
               label="F⁰")
    d.arr_orth(m1, inj1, color=PINK_S, sw=2,
               exX=1, exY=0.5, enX=0, enY=0.5,
               label="F¹")
    d.arr_orth(m2, inj2, color=PINK_S, sw=2,
               exX=1, exY=0.5, enX=0, enY=0.5,
               label="F²")

    # Injection annotation
    d.lbl(LCX + 80, 600, 100, 16,
          "<font color='#B85450'>h[vis] += F<sup>i</sup></font>",
          fs=8, color=PINK_S)

    # RMSNorm
    rms = d.box(LCX - 60, 270, 120, 28,
                "<b>RMSNorm</b>", fill=YELLOW, stroke=YELLOW_S, fs=10)
    d.arr(layers["3-N"], rms)

    # LM Head
    lm_head = d.box(LCX - 60, 215, 120, 28,
                    "<b>LM Head</b><br>Linear", fill=ORANGE, stroke=ORANGE_S, fs=9)
    d.arr(rms, lm_head)

    # Output
    llm_out = d.box(LCX - 50, 165, 100, 26,
                    "<b>logits</b>", fill=GREEN, stroke=GREEN_S, fs=10, bold=True)
    d.arr(lm_head, llm_out)

    # ── 3D M-RoPE annotation ──
    d.lbl(LX + LW + 5, 480, 80, 50,
          "<font point-size='7'>3D M-RoPE<br>(t, h, w)<br>per layer</font>",
          fs=7, color=ROPE_S)

    # ════════════════════════════════════════════════════════
    #  (b) VISION BLOCK DETAIL
    # ════════════════════════════════════════════════════════

    BPX = 770; BPW = 210
    BCX = BPX + BPW // 2
    VBW = 150; VBH = 32

    d.lbl(BPX, 5, BPW, 22,
          "<b>(b) VisionBlock (×27)</b>", fs=12, color="#333333")

    d.grp(BPX + 15, 175, BPW - 30, 610, fill="#F5F7FF", stroke=BLUE_S)

    # L× bracket
    d.lbl(BPX + 18, 380, 25, 100,
          "<b>L×</b>", fs=16, color="#BBBBBB", bold=True)

    # Z^(l) — bottom
    zl = d.box(BCX - VBW//2, 740, VBW, 26,
               "Z<sup>(l)</sup>  (N, 1152)", fill=GRAY_D, stroke=GRAY_DS, fs=9, bold=True)

    # LN1
    ln1 = d.box(BCX - VBW//2, 680, VBW, VBH,
                "<b>LayerNorm</b>", fill=YELLOW, stroke=YELLOW_S, fs=10)
    d.arr(zl, ln1)

    # Window Self-Attention
    attn = d.box(BCX - VBW//2, 615, VBW, 38,
                 "<b>Window Self-Attn</b><br>16 heads, d<sub>h</sub>=72",
                 fill=BLUE, stroke=BLUE_S, fs=9, bold=True)
    d.arr(ln1, attn)

    # 2D RoPE annotation (side)
    rope_v = d.box(BCX + VBW//2 + 8, 620, 52, 28,
                   "<b>2D<br>RoPE</b>", fill=ROPE, stroke=ROPE_S, fs=7, bold=True)
    d.arr_orth(rope_v, attn, dash=True, color=ROPE_S, sw=1,
               exX=0, exY=0.5, enX=1, enY=0.5)

    # ⊕ 1
    add1 = d.circ(BCX, 578, 24, "⊕")
    d.arr(attn, add1)
    d.arr_bypass(zl, add1)

    # LN2
    ln2 = d.box(BCX - VBW//2, 520, VBW, VBH,
                "<b>LayerNorm</b>", fill=YELLOW, stroke=YELLOW_S, fs=10)
    d.arr(add1, ln2)

    # MLP
    mlp_v = d.box(BCX - VBW//2, 458, VBW, 36,
                  "<b>MLP</b><br>FC(1152→4304)→GELU→FC",
                  fill=ORANGE, stroke=ORANGE_S, fs=8, bold=True)
    d.arr(ln2, mlp_v)

    # ⊕ 2
    add2 = d.circ(BCX, 422, 24, "⊕")
    d.arr(mlp_v, add2)
    d.arr_bypass(add1, add2)

    # Z^(l+1)
    zlp1 = d.box(BCX - VBW//2, 370, VBW, 26,
                 "Z<sup>(l+1)</sup>", fill=GRAY_D, stroke=GRAY_DS, fs=9, bold=True)
    d.arr(add2, zlp1)

    # DeepStack tap annotation
    d.lbl(BCX - VBW//2, 280, VBW, 55,
          "<font color='#B85450'>At l ∈ {8, 16, 24}:<br>"
          "Z<sup>(l+1)</sup> → DeepStack<br>"
          "PatchMerger → F<sup>k</sup></font>",
          fs=8, color=PINK_S)

    # Pos encoding summary
    d.lbl(BCX - VBW//2, 200, VBW, 50,
          "<b>Position Encoding:</b><br>"
          "① Abs Embed (1회, 입력에 +)<br>"
          "② 2D RoPE (매 블록, Q·K에)",
          fs=7, color="#666666")

    # ════════════════════════════════════════════════════════
    #  (c) DECODER LAYER DETAIL
    # ════════════════════════════════════════════════════════

    CPX = 1020; CPW = 220
    CCX = CPX + CPW // 2
    DLW = 160; DLH = 32

    d.lbl(CPX, 5, CPW, 22,
          "<b>(c) Decoder Layer (×N)</b>", fs=12, color="#333333")

    d.grp(CPX + 15, 175, CPW - 30, 610, fill="#F5FFF5", stroke=GREEN_S)

    # h input — bottom
    h_in = d.box(CCX - DLW//2, 740, DLW, 26,
                 "h  (B, L, 3584)", fill=GRAY_D, stroke=GRAY_DS, fs=9, bold=True)

    # RMSNorm 1
    rms1 = d.box(CCX - DLW//2, 680, DLW, DLH,
                 "<b>RMSNorm</b>", fill=YELLOW, stroke=YELLOW_S, fs=10)
    d.arr(h_in, rms1)

    # GQA (Grouped Query Attention)
    gqa = d.box(CCX - DLW//2, 610, DLW, 42,
                "<b>GQA</b><br>Q·K Norm + Causal",
                fill=BLUE, stroke=BLUE_S, fs=9, bold=True)
    d.arr(rms1, gqa)

    # 3D M-RoPE (side)
    mrope = d.box(CCX + DLW//2 + 8, 616, 52, 30,
                  "<b>3D<br>M-RoPE</b>", fill=ROPE, stroke=ROPE_S, fs=7, bold=True)
    d.arr_orth(mrope, gqa, dash=True, color=ROPE_S, sw=1,
               exX=0, exY=0.5, enX=1, enY=0.5)

    # ⊕ 1
    d_add1 = d.circ(CCX, 575, 24, "⊕")
    d.arr(gqa, d_add1)
    d.arr_bypass(h_in, d_add1)

    # RMSNorm 2
    rms2 = d.box(CCX - DLW//2, 515, DLW, DLH,
                 "<b>RMSNorm</b>", fill=YELLOW, stroke=YELLOW_S, fs=10)
    d.arr(d_add1, rms2)

    # SwiGLU MLP
    smlp = d.box(CCX - DLW//2, 448, DLW, 40,
                 "<b>SwiGLU MLP</b><br>gate · up → down",
                 fill=ORANGE, stroke=ORANGE_S, fs=9, bold=True)
    d.arr(rms2, smlp)

    # ⊕ 2
    d_add2 = d.circ(CCX, 412, 24, "⊕")
    d.arr(smlp, d_add2)
    d.arr_bypass(d_add1, d_add2)

    # h output
    h_out = d.box(CCX - DLW//2, 365, DLW, 26,
                  "h'  (B, L, 3584)", fill=GRAY_D, stroke=GRAY_DS, fs=9, bold=True)
    d.arr(d_add2, h_out)

    # DeepStack injection annotation
    d.lbl(CCX - DLW//2, 280, DLW, 55,
          "<font color='#B85450'>After layers 0, 1, 2:<br>"
          "h'[vis_mask] += F<sup>k</sup><br>"
          "(only at [IMG] positions)</font>",
          fs=8, color=PINK_S)

    # GQA details
    d.lbl(CCX - DLW//2, 200, DLW, 50,
          "<b>GQA Details:</b><br>"
          "Q: proj → RMSNorm → RoPE<br>"
          "K: proj → RMSNorm → RoPE<br>"
          "V: proj (no norm, no RoPE)",
          fs=7, color="#666666")

    # ════════════════════════════════════════════════════════
    #  (d) KEY DIFFERENCES TABLE — bottom right
    # ════════════════════════════════════════════════════════

    TX = 770; TY = 810

    d.lbl(TX, TY, 500, 20,
          "<b>(d) Vision Block vs Decoder Layer</b>", fs=11, color="#333333",
          align="left")

    # Comparison boxes
    ty = TY + 25
    rows = [
        ("Norm", "LayerNorm", "RMSNorm"),
        ("Attention", "Window Self-Attn", "GQA (Causal)"),
        ("Pos Encoding", "2D RoPE (h, w)", "3D M-RoPE (t, h, w)"),
        ("MLP", "FC→GELU→FC", "SwiGLU (gate·up→down)"),
        ("Q/K Norm", "—", "RMSNorm per head"),
        ("Bias", "Yes (QKV, FC)", "No"),
    ]
    # Headers
    d.box(TX, ty, 100, 20, "<b>Component</b>", fill=GRAY_D, stroke=GRAY_DS, fs=8, bold=True)
    d.box(TX + 102, ty, 130, 20, "<b>Vision Block</b>", fill=BLUE, stroke=BLUE_S, fs=8, bold=True)
    d.box(TX + 234, ty, 130, 20, "<b>Decoder Layer</b>", fill=LGREEN, stroke=LGREEN_S, fs=8, bold=True)
    ty += 22
    for comp, vis, llm in rows:
        d.box(TX, ty, 100, 18, comp, fill=WHITE, stroke=GRAY_DS, fs=7)
        d.box(TX + 102, ty, 130, 18, vis, fill=WHITE, stroke=BLUE_S, fs=7)
        d.box(TX + 234, ty, 130, 18, llm, fill=WHITE, stroke=GREEN_S, fs=7)
        ty += 19

    # ════════════════════════════════════════════════════════
    #  LEGEND
    # ════════════════════════════════════════════════════════

    d.lbl(20, 920, 50, 14, "<b>Legend:</b>", fs=8, color="#333333",
          align="left", bold=True)
    legend = [
        (BLUE, BLUE_S, "ViT / Attn"),
        (PINK, PINK_S, "DeepStack"),
        (LGREEN, LGREEN_S, "Injection Layers"),
        (TEAL, TEAL_S, "Regular Layers"),
        (ORANGE, ORANGE_S, "MLP"),
        (PURPLE, PURPLE_S, "Embed / Proj"),
        (YELLOW, YELLOW_S, "Norm"),
        (GREEN, GREEN_S, "Output"),
        (ROPE, ROPE_S, "Position"),
    ]
    for i, (fill, stroke, label) in enumerate(legend):
        bx = 72 + i * 78
        d.box(bx, 920, 12, 12, "", fill=fill, stroke=stroke, fs=6)
        d.lbl(bx + 14, 919, 62, 14, label, fs=7, color="#333333", align="left")

    return d


if __name__ == "__main__":
    out = "/home/perelman/.openclaw/workspace/qwen_review/qwen3vl_architecture.drawio"
    xml = build().to_xml()
    with open(out, "w") as f:
        f.write(xml)
    sz = os.path.getsize(out)
    print(f"✅ {out}  ({sz:,} bytes)")
