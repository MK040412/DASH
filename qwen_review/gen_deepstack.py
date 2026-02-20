#!/usr/bin/env python3
"""Qwen3-VL DeepStack — Detailed cross-model injection diagram.

Shows how multi-scale ViT features (blocks 8, 16, 24) are extracted,
passed through individual PatchMergers, and injected into LLM decoder
layers 0, 1, 2 at visual token positions.

Bottom-to-top flow, publication-quality.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
import os

# ── Colors ──
BLUE     = "#DAE8FC"; BLUE_S   = "#6C8EBF"
ORANGE   = "#FFE6CC"; ORANGE_S = "#D79B00"
PURPLE   = "#E1D5E7"; PURPLE_S = "#9673A6"
GREEN    = "#D5E8D4"; GREEN_S  = "#82B366"
YELLOW   = "#FFF2CC"; YELLOW_S = "#D6B656"
PINK     = "#F8CECC"; PINK_S   = "#B85450"
GRAY     = "#F5F5F5"; GRAY_S   = "#999999"
GRAY_D   = "#E6E6E6"; GRAY_DS  = "#666666"
WHITE    = "#FFFFFF"
TEAL     = "#D4E6F1"; TEAL_S   = "#2471A3"
LGREEN   = "#E8F5E9"; LGREEN_S = "#388E3C"


class D:
    def __init__(self, name, pw=1500, ph=950):
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

    def circ(self, cx, cy, sz=28, label="⊕"):
        cid = self._n()
        s = (f"ellipse;whiteSpace=wrap;html=1;aspect=fixed;"
             f"fillColor={WHITE};strokeColor=#333333;"
             f"fontSize=14;fontStyle=1;strokeWidth=1.5;")
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

    def to_xml(self):
        root = ET.Element("mxfile", host="app.diagrams.net")
        diag = ET.SubElement(root, "diagram", name=self.name, id="ds")
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
    d = D("DeepStack Detail", pw=1500, ph=950)

    # ═══════════════════════════════════════════
    # TITLE
    # ═══════════════════════════════════════════
    d.lbl(20, 8, 600, 22,
          "<b>Qwen3-VL DeepStack: Multi-Scale Vision Feature Injection</b>",
          fs=14, color="#333333", align="left", bold=True)

    # ═══════════════════════════════════════════
    # (a) Cross-Model Architecture
    # ═══════════════════════════════════════════

    # ── LEFT: Vision Transformer ──
    VX = 40; VW = 140
    VCX = VX + VW // 2

    # Background group
    d.grp(VX - 10, 50, VW + 20, 850, fill="#F0F4FF", stroke=BLUE_S,
          label="  Vision Encoder (ViT)")

    # Blocks — bottom to top (27 blocks compressed)
    # Show: 0-7 (gray), 8 (pink), 9-15 (gray), 16 (pink), 17-23 (gray), 24 (pink), 25-26 (gray)
    block_ids = {}
    block_groups = [
        # (label, y, h, fill, stroke, is_key)
        ("Block 0–7", 800, 70, BLUE, BLUE_S, False),
        ("Block 8", 700, 40, PINK, PINK_S, True),
        ("Block 9–15", 620, 50, BLUE, BLUE_S, False),
        ("Block 16", 540, 40, PINK, PINK_S, True),
        ("Block 17–23", 460, 50, BLUE, BLUE_S, False),
        ("Block 24", 380, 40, PINK, PINK_S, True),
        ("Block 25–26", 310, 40, BLUE, BLUE_S, False),
    ]
    for label, y, h, fill, stroke, is_key in block_groups:
        bid = d.box(VX, y, VW, h, f"<b>{label}</b>",
                    fill=fill, stroke=stroke, fs=9, bold=is_key)
        block_ids[label] = bid
        # arrows between consecutive blocks
    prev = None
    for label, y, h, fill, stroke, is_key in reversed(block_groups):
        if prev is not None:
            d.arr(block_ids[label], prev, color="#999999", sw=1)
        prev = block_ids[label]

    # Input arrow
    vit_in = d.box(VCX - 55, 870, 110, 28,
                   "h<sup>(0)</sup>  (N, 1152)", fill=GRAY_D, stroke=GRAY_DS, fs=9)
    d.arr(vit_in, block_ids["Block 0–7"])

    # Final merger output
    final_merger = d.box(VX, 240, VW, 36,
                         "<b>Final Merger</b><br>PatchMerger",
                         fill=GREEN, stroke=GREEN_S, fs=9)
    d.arr(block_ids["Block 25–26"], final_merger)

    vit_out = d.box(VCX - 60, 175, 120, 28,
                    "<b>visual_tokens</b><br>(N/4, 3584)",
                    fill=GREEN, stroke=GREEN_S, fs=9, bold=True)
    d.arr(final_merger, vit_out)

    # "To LLM" label
    d.lbl(VCX - 40, 145, 80, 18, "→ LLM input", fs=8, color=GREEN_S)

    # ── CENTER: DeepStack Mergers ──
    MX = 280; MW = 170
    MCX = MX + MW // 2

    d.lbl(MX, 50, MW, 22,
          "<b>DeepStack Mergers</b>", fs=11, color=PINK_S, bold=True)

    # 3 PatchMerger boxes — aligned with vision block 8, 16, 24
    merger_data = [
        ("Block 8",  "Merger<sub>0</sub>", 685),
        ("Block 16", "Merger<sub>1</sub>", 525),
        ("Block 24", "Merger<sub>2</sub>", 365),
    ]
    merger_ids = []
    for vblock, mlabel, my in merger_data:
        # Merger box
        mid = d.box(MX, my, MW, 50,
                    f"<b>PatchMerger</b><br>{mlabel}<br>"
                    f"<font point-size='7'>postshuffle_norm=True</font>",
                    fill=PINK, stroke=PINK_S, fs=9)
        merger_ids.append(mid)

        # Arrow from vision block → merger
        d.arr_orth(block_ids[vblock], mid, color=PINK_S, sw=2,
                   exX=1, exY=0.5, enX=0, enY=0.5)

    # Dimension annotations for mergers
    d.lbl(MX + MW + 5, 693, 120, 16,
          "(N, 1152) → (N/4, 3584)", fs=7, color=PINK_S)
    d.lbl(MX + MW + 5, 533, 120, 16,
          "(N, 1152) → (N/4, 3584)", fs=7, color=PINK_S)
    d.lbl(MX + MW + 5, 373, 120, 16,
          "(N, 1152) → (N/4, 3584)", fs=7, color=PINK_S)

    # Merger internal detail (small annotation)
    d.lbl(MX, 740, MW, 40,
          "<font point-size='7'>LN(4608) → FC(4608→4608)<br>"
          "→ GELU → FC(4608→3584)</font>",
          fs=7, color="#666666")

    # ── RIGHT: LLM Decoder ──
    LX = 560; LW = 160
    LCX = LX + LW // 2

    d.grp(LX - 10, 50, LW + 20, 850, fill="#F0FFF5", stroke=GREEN_S,
          label="  LLM Decoder")

    # Decoder layers — bottom to top
    # Show layers 0, 1, 2 (highlighted), then 3–N (gray)
    llm_in = d.box(LCX - 60, 870, 120, 28,
                   "inputs_embeds<br>(B, L, 3584)",
                   fill=GRAY_D, stroke=GRAY_DS, fs=8)

    layer_ids = {}
    llm_layers = [
        ("Layer 0", 770, 50, LGREEN, LGREEN_S, True),
        ("Layer 1", 630, 50, LGREEN, LGREEN_S, True),
        ("Layer 2", 490, 50, LGREEN, LGREEN_S, True),
        ("Layer 3–N", 370, 60, TEAL, TEAL_S, False),
    ]
    for label, y, h, fill, stroke, is_key in llm_layers:
        lid = d.box(LX, y, LW, h, f"<b>{label}</b><br>DecoderLayer",
                    fill=fill, stroke=stroke, fs=9, bold=is_key)
        layer_ids[label] = lid

    d.arr(llm_in, layer_ids["Layer 0"])

    # Arrows between layers
    d.arr(layer_ids["Layer 0"], layer_ids["Layer 1"])
    d.arr(layer_ids["Layer 1"], layer_ids["Layer 2"])
    d.arr(layer_ids["Layer 2"], layer_ids["Layer 3–N"])

    # LLM output
    llm_out = d.box(LCX - 50, 290, 100, 28,
                    "<b>RMSNorm</b>", fill=YELLOW, stroke=YELLOW_S, fs=10)
    d.arr(layer_ids["Layer 3–N"], llm_out)

    llm_final = d.box(LCX - 60, 225, 120, 28,
                      "hidden_states<br>(B, L, 3584)",
                      fill=GREEN, stroke=GREEN_S, fs=9, bold=True)
    d.arr(llm_out, llm_final)

    # ── INJECTION POINTS: ⊕ between merger and layer output ──
    # The injection happens AFTER the decoder layer output
    # hidden_states[vis_pos] += deepstack_feat[i]

    inject_data = [
        (merger_ids[0], "Layer 0", "Layer 1", 740),   # after layer 0
        (merger_ids[1], "Layer 1", "Layer 2", 600),   # after layer 1
        (merger_ids[2], "Layer 2", "Layer 3–N", 460), # after layer 2
    ]
    for mid, src_layer, tgt_layer, iy in inject_data:
        # ⊕ circle to the right of LLM column
        add_id = d.circ(LCX, iy, 28, "⊕")

        # Arrow from merger → ⊕
        d.arr_orth(mid, add_id, color=PINK_S, sw=2,
                   exX=1, exY=0.5, enX=0, enY=0.5)

        # Reconnect: layer output → ⊕ → next layer
        # The ⊕ sits between two layers on the main path

    # Labels for injection points
    d.lbl(LX + LW + 10, 730, 130, 30,
          "<font color='#B85450'><b>h[vis] += F<sup>(0)</sup></b></font><br>"
          "<font point-size='7'>Block 8 features</font>",
          fs=9, color=PINK_S)
    d.lbl(LX + LW + 10, 590, 130, 30,
          "<font color='#B85450'><b>h[vis] += F<sup>(1)</sup></b></font><br>"
          "<font point-size='7'>Block 16 features</font>",
          fs=9, color=PINK_S)
    d.lbl(LX + LW + 10, 450, 130, 30,
          "<font color='#B85450'><b>h[vis] += F<sup>(2)</sup></b></font><br>"
          "<font point-size='7'>Block 24 features</font>",
          fs=9, color=PINK_S)

    # ═══════════════════════════════════════════
    # (b) _deepstack_process Detail — RIGHT SIDE
    # ═══════════════════════════════════════════
    BX = 860; BW = 350

    d.lbl(BX, 50, BW, 22,
          "<b>(b) _deepstack_process()</b>", fs=12, color="#333333")

    d.grp(BX, 80, BW, 420, fill="#FFF8F8", stroke=PINK_S)

    # Input: hidden_states (B, L, 3584)
    hs_in = d.box(BX + 20, 460, 150, 28,
                  "hidden_states<br>(B, L, 3584)", fill=TEAL, stroke=TEAL_S, fs=9)

    # visual_pos_masks
    mask_box = d.box(BX + 190, 460, 140, 28,
                     "visual_pos_masks<br>(B, L) bool",
                     fill=YELLOW, stroke=YELLOW_S, fs=8)

    # Step 1: Index select
    idx_sel = d.box(BX + 40, 395, 180, 30,
                    "<b>h[vis_mask, :]</b><br>select visual positions",
                    fill=PURPLE, stroke=PURPLE_S, fs=9)
    d.arr(hs_in, idx_sel)
    d.arr_orth(mask_box, idx_sel, color=YELLOW_S, sw=1,
               exX=0.3, exY=0, enX=0.8, enY=1)

    # deepstack_visual_embeds[i]
    ds_embed = d.box(BX + 230, 340, 110, 35,
                     "<b>F<sup>(i)</sup></b><br>(N<sub>vis</sub>, 3584)",
                     fill=PINK, stroke=PINK_S, fs=9, bold=True)

    # Step 2: Clone + Add
    clone_add = d.circ(BX + 130, 320, 30, "⊕")
    d.arr(idx_sel, clone_add)
    d.arr_orth(ds_embed, clone_add, color=PINK_S, sw=1.5,
               exX=0, exY=0.5, enX=1, enY=0.5)

    # .clone() annotation
    d.lbl(BX + 20, 345, 80, 16, ".clone()", fs=7, color="#999999")

    # Step 3: Scatter back
    scatter = d.box(BX + 40, 250, 200, 32,
                    "<b>h[vis_mask, :] = result</b><br>scatter back",
                    fill=PURPLE, stroke=PURPLE_S, fs=9)
    d.arr(clone_add, scatter)

    # Output
    hs_out = d.box(BX + 60, 180, 160, 28,
                   "hidden_states (updated)<br>(B, L, 3584)",
                   fill=TEAL, stroke=TEAL_S, fs=9, bold=True)
    d.arr(scatter, hs_out)

    # Code snippet
    d.lbl(BX + 10, 100, BW - 20, 65,
          "<font face='Courier' point-size='8'>"
          "local = h[vis_mask, :].clone()<br>"
          "local = local + visual_embeds<br>"
          "h[vis_mask, :] = local<br>"
          "return h</font>",
          fs=8, color="#666666")

    # ═══════════════════════════════════════════
    # (c) Token Sequence Visualization — BOTTOM RIGHT
    # ═══════════════════════════════════════════
    TX = 860; TY = 530

    d.lbl(TX, TY, 350, 22,
          "<b>(c) LLM Sequence with Visual Positions</b>",
          fs=11, color="#333333")

    # Token sequence bar
    seq_y = TY + 50
    tokens = [
        ("[BOS]", GRAY_D, 50),
        ("User:", TEAL, 50),
        ("Describe", TEAL, 60),
        ("this", TEAL, 40),
        ("[IMG]", PINK, 45),
        ("[IMG]", PINK, 45),
        ("[IMG]", PINK, 45),
        ("...", GRAY_D, 30),
    ]
    tx = TX + 10
    for label, fill, w in tokens:
        stroke = PINK_S if fill == PINK else GRAY_DS
        d.box(tx, seq_y, w, 25, label, fill=fill, stroke=stroke, fs=7)
        tx += w + 3

    # Mask indicator
    d.lbl(TX + 10, seq_y + 30, 350, 20,
          "visual_pos_masks: "
          "<font color='#999'>0  0  0  0</font>  "
          "<font color='#B85450'><b>1  1  1</b></font>  "
          "<font color='#999'>...</font>",
          fs=8, color="#666666")

    # Arrow pointing to IMG tokens
    d.lbl(TX + 10, seq_y + 52, 350, 40,
          "DeepStack features are <b>only added</b> at positions<br>"
          "where <font color='#B85450'><b>visual_pos_masks = True</b></font><br>"
          "(i.e., [IMG] token positions in LLM sequence)",
          fs=8, color="#333333")

    # ═══════════════════════════════════════════
    # (d) Feature Scale Semantics — BOTTOM
    # ═══════════════════════════════════════════
    SX = 860; SY = 700

    d.lbl(SX, SY, 350, 22,
          "<b>(d) Multi-Scale Feature Semantics</b>",
          fs=11, color="#333333")

    scale_data = [
        ("Block 8 → Layer 0", "Low-level: edges, textures",
         "#FCE4EC", PINK_S),
        ("Block 16 → Layer 1", "Mid-level: object parts, patterns",
         "#F8CECC", PINK_S),
        ("Block 24 → Layer 2", "High-level: objects, scene semantics",
         "#EFACAC", PINK_S),
    ]
    for i, (src, desc, fill, stroke) in enumerate(scale_data):
        sy = SY + 30 + i * 45
        d.box(SX + 10, sy, 155, 35, f"<b>{src}</b>",
              fill=fill, stroke=stroke, fs=8, bold=True)
        d.lbl(SX + 175, sy, 170, 35, desc, fs=8, color="#666666",
              align="left")

    # Final note
    d.lbl(SX + 10, SY + 172, 340, 40,
          "Early injection into LLM decoder ensures<br>"
          "rich visual grounding from the <b>first layers</b> of text generation",
          fs=8, color=PINK_S)

    # ═══════════════════════════════════════════
    # Legend
    # ═══════════════════════════════════════════
    d.lbl(40, 920, 60, 14, "<b>Legend:</b>", fs=8, color="#333333",
          align="left", bold=True)
    legend = [
        (BLUE, BLUE_S, "ViT Block"),
        (PINK, PINK_S, "DeepStack"),
        (LGREEN, LGREEN_S, "Injection Layer"),
        (TEAL, TEAL_S, "LLM hidden"),
        (PURPLE, PURPLE_S, "Index Op"),
        (YELLOW, YELLOW_S, "Mask / Norm"),
    ]
    for i, (fill, stroke, label) in enumerate(legend):
        bx = 100 + i * 80
        d.box(bx, 920, 12, 12, "", fill=fill, stroke=stroke, fs=6)
        d.lbl(bx + 14, 919, 64, 14, label, fs=7, color="#333333",
              align="left")

    return d


if __name__ == "__main__":
    out = "/home/perelman/.openclaw/workspace/qwen_review/deepstack_detail.drawio"
    xml = build().to_xml()
    with open(out, "w") as f:
        f.write(xml)
    sz = os.path.getsize(out)
    print(f"✅ {out}  ({sz:,} bytes)")
