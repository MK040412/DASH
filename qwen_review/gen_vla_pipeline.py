#!/usr/bin/env python3
"""Qwen3-VL VLA Pipeline — Full architecture diagram.

Paper figure style: visual blocks with PyTorch class/function names only.
Minimal text, color-coded, multi-panel.

(a) Full VLA Pipeline: Qwen3-VL → Action Expert → Flow Matching → Actions
(b) Action Expert Detail: Cross-Attention to VLM KV Cache
(c) Flow Matching: Training (interpolation + velocity) / Inference (Euler steps)
(d) DeepStack Injection into Action Expert
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom

# Colors
BLUE     = "#D6E4F0"; BLUE_S   = "#4472C4"   # VLM / Attention
ORANGE   = "#FBE5D6"; ORANGE_S = "#ED7D31"   # MLP
PURPLE   = "#E2D0E8"; PURPLE_S = "#7030A0"   # Linear
GREEN    = "#E2EFDA"; GREEN_S  = "#548235"   # Action Expert
YELLOW   = "#FFF2CC"; YELLOW_S = "#BF9000"   # Norm / Op
PINK     = "#FCE4EC"; PINK_S   = "#C62828"   # DeepStack
RED      = "#F8CECC"; RED_S    = "#B85450"   # Flow / Noise
TEAL     = "#D5F5E3"; TEAL_S   = "#1E8449"   # Output
GRAY     = "#F2F2F2"; GRAY_S   = "#808080"
ROPE     = "#E8D5F5"; ROPE_S   = "#7B1FA2"
WHITE    = "#FFFFFF"


class D:
    def __init__(self, name, pw=2400, ph=1600):
        self.name, self.pw, self.ph = name, pw, ph
        self.cells = []; self._id = 2
    def _n(self):
        i = self._id; self._id += 1; return str(i)

    def box(self, x, y, w, h, label, fill=WHITE, stroke="#333", fs=10, bold=False):
        cid = self._n()
        s = (f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
             f"fontSize={fs};fontStyle={'1' if bold else '0'};arcSize=10;strokeWidth=1.5;")
        self.cells.append(dict(id=cid,value=label,style=s,vertex=True,x=x,y=y,w=w,h=h))
        return cid

    def circ(self, x, y, sz, label, fill=WHITE, stroke="#333", fs=14):
        cid = self._n()
        s = (f"ellipse;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
             f"fontSize={fs};fontStyle=1;aspect=fixed;strokeWidth=1.5;")
        self.cells.append(dict(id=cid,value=label,style=s,vertex=True,x=x,y=y,w=sz,h=sz))
        return cid

    def diamond(self, x, y, w, h, label, fill=WHITE, stroke="#333", fs=10):
        cid = self._n()
        s = (f"rhombus;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
             f"fontSize={fs};fontStyle=1;strokeWidth=1.5;")
        self.cells.append(dict(id=cid,value=label,style=s,vertex=True,x=x,y=y,w=w,h=h))
        return cid

    def lbl(self, x, y, w, h, text, fs=9, color="#333", align="center", bold=False):
        cid = self._n()
        s = (f"text;html=1;align={align};verticalAlign=middle;resizable=1;points=[];"
             f"autosize=0;strokeColor=none;fillColor=none;fontSize={fs};fontColor={color};"
             f"fontStyle={'1' if bold else '0'};")
        self.cells.append(dict(id=cid,value=text,style=s,vertex=True,x=x,y=y,w=w,h=h))
        return cid

    def grp(self, x, y, w, h, label="", fill="#FAFAFA", stroke="#CCC"):
        cid = self._n()
        s = (f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
             f"dashed=0;verticalAlign=top;fontSize=10;fontStyle=0;fontColor=#666;"
             f"opacity=40;arcSize=6;strokeWidth=1;")
        self.cells.append(dict(id=cid,value=label,style=s,vertex=True,x=x,y=y,w=w,h=h))
        return cid

    def arr(self, s, t, label="", dash=False, color="#333", sw=1.5):
        cid = self._n()
        dd = "dashed=1;dashPattern=6 3;" if dash else ""
        st = (f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;html=1;{dd}"
              f"strokeColor={color};fontSize=8;fontColor=#666;endArrow=blockThin;"
              f"endFill=1;strokeWidth={sw};")
        self.cells.append(dict(id=cid,value=label,style=st,edge=True,source=s,target=t))
        return cid

    def to_xml(self):
        root = ET.Element("mxfile", host="app.diagrams.net")
        diag = ET.SubElement(root, "diagram", name=self.name, id="vla1")
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
    d = D("Qwen3-VL VLA Architecture", pw=2400, ph=1700)

    BW = 130; BH = 36  # standard box

    # ════════════════════════════════════════════════════════
    # (a) Full VLA Pipeline — LEFT HALF
    # ════════════════════════════════════════════════════════
    ax = 30; ay = 30
    d.lbl(ax, ay, 600, 22, "<b>(a) Full VLA Pipeline</b>", fs=13, color="#333", align="left")

    # --- Inputs (bottom) ---
    iy = ay + 680
    img1 = d.box(ax+10, iy, 90, 65, "🖼️<br><b>cam<sub>1</sub></b>", fill=GRAY, stroke=GRAY_S, fs=9)
    img2 = d.box(ax+110, iy, 90, 65, "🖼️<br><b>cam<sub>2</sub></b>", fill=GRAY, stroke=GRAY_S, fs=9)
    lang = d.box(ax+210, iy, 110, 65, "📝<br><b>lang</b><br><i>instruction</i>", fill=GRAY, stroke=GRAY_S, fs=9)
    prop = d.box(ax+330, iy, 100, 65, "🦾<br><b>proprio</b><br><i>q, gripper</i>", fill=GRAY, stroke=GRAY_S, fs=9)
    noise = d.box(ax+450, iy, 100, 65, "🎲<br><b>a<sub>t</sub></b><br><i>noisy action</i>", fill=RED, stroke=RED_S, fs=9)

    # --- Vision Encoder ---
    iy -= 90
    vis_enc = d.box(ax+10, iy, 190, 55,
        "<b>Qwen3VLVisionModel</b><br>.forward(pixel_values, grid_thw)",
        fill=BLUE, stroke=BLUE_S, fs=9, bold=True)
    d.arr(img1, vis_enc)
    d.arr(img2, vis_enc)

    # Vision outputs
    iy -= 55
    vis_tok = d.box(ax+10, iy, 100, 32, "<b>visual_tokens</b>", fill=BLUE, stroke=BLUE_S, fs=9)
    ds_feat = d.box(ax+120, iy, 100, 32, "<b>deepstack</b>", fill=PINK, stroke=PINK_S, fs=9)
    d.arr(vis_enc, vis_tok, label="(N/4, d)")
    d.arr(vis_enc, ds_feat, label="3×(N/4, d)")

    # --- Tokenizers ---
    tok_y = iy + 55
    lang_tok = d.box(ax+220, tok_y, 100, 32, "<b>nn.Embedding</b>", fill=YELLOW, stroke=YELLOW_S, fs=9)
    d.arr(lang, lang_tok)
    prop_proj = d.box(ax+335, tok_y, 90, 32, "<b>nn.Linear</b>", fill=PURPLE, stroke=PURPLE_S, fs=9)
    d.arr(prop, prop_proj)
    act_proj = d.box(ax+455, tok_y, 90, 32, "<b>nn.Linear</b>", fill=PURPLE, stroke=PURPLE_S, fs=9)
    d.arr(noise, act_proj)

    # --- Token Merge ---
    merge_y = iy - 55
    merge = d.box(ax+60, merge_y, 350, 36,
        "<b>Token Merge</b>  (insert visual @ &lt;|vision|&gt; positions)",
        fill=YELLOW, stroke=YELLOW_S, fs=9)
    d.arr(vis_tok, merge)
    d.arr(lang_tok, merge)
    d.arr(prop_proj, merge)

    # --- Qwen3-VL LLM Decoder ---
    llm_y = merge_y - 70
    d.grp(ax, llm_y, 440, 60, "", fill="#E8EAF6", stroke=BLUE_S)
    llm = d.box(ax+10, llm_y+10, 420, 40,
        "<b>Qwen3VLTextModel</b>.forward()  (frozen / LoRA)<br>"
        "DecoderLayer × N  →  KV Cache",
        fill=BLUE, stroke=BLUE_S, fs=10, bold=True)
    d.arr(merge, llm)
    d.arr(ds_feat, llm, label="deepstack inject", dash=True, color=PINK_S)

    # KV Cache output
    kv_y = llm_y - 45
    kv = d.box(ax+120, kv_y, 160, 32, "<b>KV Cache</b>", fill=BLUE, stroke=BLUE_S, fs=11, bold=True)
    d.arr(llm, kv)

    # --- Action Expert ---
    ae_y = kv_y - 80
    d.grp(ax+40, ae_y, 380, 65, "", fill="#E8F5E9", stroke=GREEN_S)
    ae = d.box(ax+50, ae_y+12, 360, 40,
        "<b>ActionExpert</b>  (~300M params, trained)<br>"
        "TransformerDecoder × 8-12  +  CrossAttention(KV Cache)",
        fill=GREEN, stroke=GREEN_S, fs=10, bold=True)
    d.arr(kv, ae, label="cross-attn")
    d.arr(act_proj, ae, label="action tokens")

    # velocity output
    vel_y = ae_y - 50
    vel = d.box(ax+140, vel_y, 180, 34,
        "<b>velocity v(a<sub>t</sub>, t)</b>",
        fill=GREEN, stroke=GREEN_S, fs=11, bold=True)
    d.arr(ae, vel)

    # Euler integration
    euler_y = vel_y - 50
    euler = d.box(ax+100, euler_y, 260, 36,
        "<b>Euler Step</b>  a<sub>t+dt</sub> = a<sub>t</sub> + dt · v(a<sub>t</sub>, t)",
        fill=RED, stroke=RED_S, fs=10)
    d.arr(vel, euler)

    # loop arrow label
    d.lbl(ax+365, euler_y+5, 80, 24, "×K steps<br>(K=10)", fs=9, color=RED_S, bold=True)

    # Final actions
    act_y = euler_y - 50
    actions = d.box(ax+140, act_y, 180, 36,
        "<b>Action Chunk</b><br>a<sub>1:H</sub>  (H=50)",
        fill=TEAL, stroke=TEAL_S, fs=11, bold=True)
    d.arr(euler, actions)

    # Robot
    robot_y = act_y - 50
    robot = d.box(ax+170, robot_y, 120, 36, "🤖 <b>Robot</b>", fill=TEAL, stroke=TEAL_S, fs=12, bold=True)
    d.arr(actions, robot)

    # ════════════════════════════════════════════════════════
    # (b) Action Expert Detail — CENTER
    # ════════════════════════════════════════════════════════
    bx = 540; by = 30
    d.lbl(bx, by, 400, 22, "<b>(b) ActionExpert (single layer)</b>", fs=13, color="#333", align="left")

    # Bottom-to-top
    cy = by + 550

    # Input
    ae_in = d.box(bx+50, cy, BW+20, 30, "action_tokens", fill=GRAY, stroke=GRAY_S, fs=10)

    # + timestep embed
    cy -= 42
    t_emb = d.box(bx+200, cy+42, 90, 28, "<b>t_embed</b><br>nn.Linear", fill=PURPLE, stroke=PURPLE_S, fs=8)
    add_t = d.circ(bx+100, cy+3, 26, "⊕", fill=WHITE, stroke="#333")
    d.arr(ae_in, add_t)
    d.arr(t_emb, add_t, color=PURPLE_S)

    # LayerNorm
    cy -= 38
    ln1 = d.box(bx+50, cy, BW+20, 30, "<b>RMSNorm</b>", fill=YELLOW, stroke=YELLOW_S, fs=10)
    d.arr(add_t, ln1)

    # Self-Attention
    cy -= 48
    self_attn = d.box(bx+30, cy, BW+60, 36,
        "<b>nn.MultiheadAttention</b><br>Self-Attention (action tokens)",
        fill=GREEN, stroke=GREEN_S, fs=9)
    d.arr(ln1, self_attn)

    # ⊕ residual
    cy -= 38
    res1 = d.circ(bx+100, cy, 26, "⊕")
    d.arr(self_attn, res1)

    # LayerNorm
    cy -= 38
    ln2 = d.box(bx+50, cy, BW+20, 30, "<b>RMSNorm</b>", fill=YELLOW, stroke=YELLOW_S, fs=10)
    d.arr(res1, ln2)

    # Cross-Attention to VLM
    cy -= 55
    cross_attn = d.box(bx+10, cy, BW+100, 42,
        "<b>nn.MultiheadAttention</b><br>Cross-Attention<br>Q=action, KV=VLM Cache",
        fill=BLUE, stroke=BLUE_S, fs=9)
    d.arr(ln2, cross_attn)

    # KV from VLM (side)
    vlm_kv = d.box(bx+250, cy+5, 100, 32, "<b>VLM<br>KV Cache</b>", fill=BLUE, stroke=BLUE_S, fs=9, bold=True)
    d.arr(vlm_kv, cross_attn, dash=True, color=BLUE_S)

    # ⊕ residual
    cy -= 38
    res2 = d.circ(bx+100, cy, 26, "⊕")
    d.arr(cross_attn, res2)

    # LayerNorm
    cy -= 38
    ln3 = d.box(bx+50, cy, BW+20, 30, "<b>RMSNorm</b>", fill=YELLOW, stroke=YELLOW_S, fs=10)
    d.arr(res2, ln3)

    # FFN
    cy -= 42
    ffn = d.box(bx+30, cy, BW+60, 34,
        "<b>SwiGLU</b><br>gate_proj · up_proj → down_proj",
        fill=ORANGE, stroke=ORANGE_S, fs=9)
    d.arr(ln3, ffn)

    # ⊕ residual
    cy -= 38
    res3 = d.circ(bx+100, cy, 26, "⊕")
    d.arr(ffn, res3)

    # output
    cy -= 35
    ae_out = d.box(bx+50, cy, BW+20, 28, "v(a<sub>t</sub>, t)", fill=GREEN, stroke=GREEN_S, fs=10, bold=True)
    d.arr(res3, ae_out)

    # loop annotation
    d.lbl(bx-20, cy+100, 30, 250, "×L", fs=16, color="#999", bold=True)

    # Group
    d.grp(bx, cy-10, 370, by+590-cy+10, "", fill="#F0FFF0", stroke=GREEN_S)

    # ════════════════════════════════════════════════════════
    # (c) Flow Matching — RIGHT TOP
    # ════════════════════════════════════════════════════════
    fx = 960; fy = 30
    d.lbl(fx, fy, 400, 22, "<b>(c) Flow Matching</b>", fs=13, color="#333", align="left")

    # Training
    d.lbl(fx, fy+28, 200, 18, "<b>Training:</b>", fs=11, color=RED_S, align="left", bold=True)

    ty = fy + 52
    # noise
    eps = d.box(fx, ty, 80, 30, "<b>ε</b><br>N(0,I)", fill=RED, stroke=RED_S, fs=10)
    # gt action
    a_gt = d.box(fx+150, ty, 80, 30, "<b>a<sub>gt</sub></b><br>demo", fill=TEAL, stroke=TEAL_S, fs=10)
    # t
    t_samp = d.box(fx+280, ty, 70, 30, "<b>t</b><br>U(0,1)", fill=YELLOW, stroke=YELLOW_S, fs=10)

    ty += 50
    # interpolation
    interp = d.box(fx+20, ty, 200, 34,
        "<b>a<sub>t</sub> = (1-t)·ε + t·a<sub>gt</sub></b>",
        fill=RED, stroke=RED_S, fs=10)
    d.arr(eps, interp)
    d.arr(a_gt, interp)
    d.arr(t_samp, interp, dash=True)

    ty += 52
    # Action Expert
    ae_train = d.box(fx+10, ty, 220, 36,
        "<b>ActionExpert</b>(a<sub>t</sub>, t, ctx)",
        fill=GREEN, stroke=GREEN_S, fs=10, bold=True)
    d.arr(interp, ae_train)

    ty += 50
    # velocity pred
    v_pred = d.box(fx+40, ty, 160, 30, "<b>v<sub>pred</sub></b>", fill=GREEN, stroke=GREEN_S, fs=10)
    d.arr(ae_train, v_pred)

    # target velocity
    v_target = d.box(fx+260, ty, 110, 30, "<b>a<sub>gt</sub> - ε</b>", fill=TEAL, stroke=TEAL_S, fs=10)

    ty += 48
    # MSE Loss
    loss = d.box(fx+80, ty, 160, 34,
        "<b>nn.MSELoss</b><br>‖v<sub>pred</sub> - (a<sub>gt</sub>-ε)‖²",
        fill=RED, stroke=RED_S, fs=10, bold=True)
    d.arr(v_pred, loss)
    d.arr(v_target, loss)

    # Inference
    d.lbl(fx, ty+55, 200, 18, "<b>Inference:</b>", fs=11, color=TEAL_S, align="left", bold=True)

    iy2 = ty + 80
    # a_0
    a0 = d.box(fx, iy2, 80, 30, "<b>a<sub>0</sub></b><br>N(0,I)", fill=RED, stroke=RED_S, fs=10)

    iy2 += 45
    # Euler loop
    euler_box = d.box(fx+10, iy2, 220, 40,
        "<b>for</b> k=1..K:<br>"
        "a<sub>t+dt</sub> = a<sub>t</sub> + dt · ActionExpert(a<sub>t</sub>, t, ctx)",
        fill=GREEN, stroke=GREEN_S, fs=9)
    d.arr(a0, euler_box)
    d.lbl(fx+235, iy2+8, 50, 22, "K=10", fs=9, color=GREEN_S, bold=True)

    iy2 += 55
    # clean action
    a_clean = d.box(fx+30, iy2, 180, 32,
        "<b>a<sub>1.0</sub></b> = denoised actions (H=50)",
        fill=TEAL, stroke=TEAL_S, fs=10, bold=True)
    d.arr(euler_box, a_clean)

    # ════════════════════════════════════════════════════════
    # (d) DeepStack Injection — RIGHT BOTTOM
    # ════════════════════════════════════════════════════════
    dx = 960; dy = 580
    d.lbl(dx, dy, 500, 22, "<b>(d) DeepStack Multi-Level Injection</b>", fs=13, color="#333", align="left")

    # Vision Encoder column
    dy2 = dy + 50
    for i, (lnum, desc) in enumerate([(8, "low-level"), (16, "mid-level"), (24, "high-level")]):
        bx_d = dx + i * 155
        blk = d.box(bx_d, dy2, 130, 36,
            f"<b>VisionBlock[{lnum}]</b><br>{desc}",
            fill=PINK, stroke=PINK_S, fs=9)

    # PatchMerger per level
    dy2 += 55
    for i in range(3):
        bx_d = dx + i * 155
        pm = d.box(bx_d, dy2, 130, 32,
            "<b>PatchMerger</b><br>postshuffle",
            fill=PINK, stroke=PINK_S, fs=9)

    # Features
    dy2 += 52
    feats = d.box(dx+60, dy2, 340, 30,
        "<b>deepstack_features</b>  [3 × (N/4, d<sub>LLM</sub>)]",
        fill=PINK, stroke=PINK_S, fs=10, bold=True)

    # Injection targets
    dy2 += 50
    d.lbl(dx, dy2, 460, 18, "Injection into LLM / Action Expert:", fs=10, color=PINK_S, align="left", bold=True)

    dy2 += 25
    inj_a = d.box(dx, dy2, 220, 36,
        "<b>LLM Decoder (early layers)</b><br>hidden += deepstack_feat[i]",
        fill=BLUE, stroke=BLUE_S, fs=9)
    d.arr(feats, inj_a, color=PINK_S)

    inj_b = d.box(dx+240, dy2, 220, 36,
        "<b>Action Expert cross-attn</b><br>extra KV from multi-level feat",
        fill=GREEN, stroke=GREEN_S, fs=9)
    d.arr(feats, inj_b, color=PINK_S)

    # What each level provides
    dy2 += 55
    d.box(dx, dy2, 140, 50,
        "<b>Layer 8</b><br>edges, textures<br>→ gripper precision",
        fill="#FFCDD2", stroke=PINK_S, fs=8)
    d.box(dx+155, dy2, 140, 50,
        "<b>Layer 16</b><br>shapes, relations<br>→ path planning",
        fill="#FFCDD2", stroke=PINK_S, fs=8)
    d.box(dx+310, dy2, 140, 50,
        "<b>Layer 24</b><br>semantics, intent<br>→ task reasoning",
        fill="#FFCDD2", stroke=PINK_S, fs=8)

    # ════════════════════════════════════════════════════════
    # (e) Training Stages — BOTTOM
    # ════════════════════════════════════════════════════════
    sx = 30; sy = 820
    d.lbl(sx, sy, 400, 22, "<b>(e) 3-Stage Training Pipeline</b>", fs=13, color="#333", align="left")

    sy += 30
    s1 = d.box(sx, sy, 250, 50,
        "<b>Stage 1: VLM Pre-train</b><br>"
        "Qwen3-VL (already done)<br>"
        "Vision-Language tasks",
        fill=BLUE, stroke=BLUE_S, fs=9)

    s2 = d.box(sx+280, sy, 250, 50,
        "<b>Stage 2: Robot Pre-train</b><br>"
        "ActionExpert + LoRA(VLM)<br>"
        "Cross-embodiment data",
        fill=GREEN, stroke=GREEN_S, fs=9)
    d.arr(s1, s2)

    s3 = d.box(sx+560, sy, 250, 50,
        "<b>Stage 3: Task Fine-tune</b><br>"
        "ActionExpert + LoRA(VLM)<br>"
        "Target robot + task",
        fill=TEAL, stroke=TEAL_S, fs=9)
    d.arr(s2, s3)

    # Frozen / Trained labels
    d.lbl(sx, sy+52, 250, 16, "VLM: trained | AE: ✗", fs=8, color=BLUE_S)
    d.lbl(sx+280, sy+52, 250, 16, "VLM: LoRA | AE: trained", fs=8, color=GREEN_S)
    d.lbl(sx+560, sy+52, 250, 16, "VLM: LoRA | AE: fine-tuned", fs=8, color=TEAL_S)

    # ════════════════════════════════════════════════════════
    # Legend
    # ════════════════════════════════════════════════════════
    lx = 30; ly = 950
    d.lbl(lx, ly, 100, 18, "<b>Legend:</b>", fs=10, color="#333", bold=True)
    items = [
        (BLUE, BLUE_S, "VLM (Qwen3-VL)"),
        (GREEN, GREEN_S, "Action Expert"),
        (RED, RED_S, "Noise / Flow"),
        (TEAL, TEAL_S, "Output / GT"),
        (PINK, PINK_S, "DeepStack"),
        (ORANGE, ORANGE_S, "FFN / SwiGLU"),
        (PURPLE, PURPLE_S, "Linear Proj"),
        (YELLOW, YELLOW_S, "Norm / Op"),
    ]
    for i, (fill, stroke, label) in enumerate(items):
        col = i // 4
        row = i % 4
        d.box(lx + col*200, ly+22+row*26, 18, 18, "", fill=fill, stroke=stroke, fs=8)
        d.lbl(lx+22 + col*200, ly+22+row*26, 160, 18, label, fs=9, color="#333", align="left")

    return d


if __name__ == "__main__":
    import os
    path = "/home/perelman/.openclaw/workspace/qwen_review/vla_pipeline.drawio"
    xml = build().to_xml()
    with open(path, "w") as f:
        f.write(xml)
    print(f"✅ {path} ({os.path.getsize(path)} bytes)")
