#!/usr/bin/env python3
"""π0 / π*0.6 Architecture — draw.io diagram from actual openpi source code.

Source: Physical-Intelligence/openpi/src/openpi/models/pi0.py
Paper figure style. PyTorch/Flax function names. Minimal text.
Multi-panel: (a) Overall, (b) Attention mask, (c) Flow matching, (d) Recap RL.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom

# Colors — clean paper palette
VLM     = "#D6E4F0"; VLM_S    = "#4472C4"   # VLM / PaliGemma
ACT     = "#E2EFDA"; ACT_S    = "#548235"   # Action Expert
IMG     = "#E8D5F5"; IMG_S    = "#7B1FA2"   # SigLIP
PROJ    = "#E2D0E8"; PROJ_S   = "#7030A0"   # Projections
FLOW    = "#F8CECC"; FLOW_S   = "#B85450"   # Flow / noise
OUT     = "#D5F5E3"; OUT_S    = "#1E8449"   # Output
NORM    = "#FFF2CC"; NORM_S   = "#BF9000"   # Norm / Ops
MLP     = "#FBE5D6"; MLP_S    = "#ED7D31"   # MLP
GRAY    = "#F2F2F2"; GRAY_S   = "#808080"
RL      = "#FDEBD0"; RL_S     = "#D35400"   # RL / Recap
W       = "#FFFFFF"


class D:
    def __init__(self, name, pw=2200, ph=1500):
        self.name, self.pw, self.ph = name, pw, ph
        self.cells = []; self._id = 2
    def _n(self):
        i = self._id; self._id += 1; return str(i)
    def box(self, x, y, w, h, lbl, fill=W, stroke="#333", fs=10, bold=False):
        cid = self._n()
        s = (f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
             f"fontSize={fs};fontStyle={'1' if bold else '0'};arcSize=10;strokeWidth=1.5;")
        self.cells.append(dict(id=cid,value=lbl,style=s,vertex=True,x=x,y=y,w=w,h=h))
        return cid
    def circ(self, x, y, sz, lbl, fill=W, stroke="#333", fs=14):
        cid = self._n()
        s = (f"ellipse;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
             f"fontSize={fs};fontStyle=1;aspect=fixed;strokeWidth=1.5;")
        self.cells.append(dict(id=cid,value=lbl,style=s,vertex=True,x=x,y=y,w=sz,h=sz))
        return cid
    def lbl(self, x, y, w, h, text, fs=9, color="#333", align="center", bold=False):
        cid = self._n()
        s = (f"text;html=1;align={align};verticalAlign=middle;resizable=1;points=[];"
             f"autosize=0;strokeColor=none;fillColor=none;fontSize={fs};fontColor={color};"
             f"fontStyle={'1' if bold else '0'};")
        self.cells.append(dict(id=cid,value=text,style=s,vertex=True,x=x,y=y,w=w,h=h))
        return cid
    def grp(self, x, y, w, h, lbl="", fill="#FAFAFA", stroke="#CCC"):
        cid = self._n()
        s = (f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
             f"dashed=0;verticalAlign=top;fontSize=10;fontStyle=1;fontColor=#555;"
             f"opacity=40;arcSize=6;strokeWidth=1;")
        self.cells.append(dict(id=cid,value=lbl,style=s,vertex=True,x=x,y=y,w=w,h=h))
        return cid
    def arr(self, s, t, lbl="", dash=False, color="#333", sw=1.5):
        cid = self._n()
        dd = "dashed=1;dashPattern=6 3;" if dash else ""
        st = (f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;html=1;{dd}"
              f"strokeColor={color};fontSize=8;fontColor=#666;endArrow=blockThin;"
              f"endFill=1;strokeWidth={sw};")
        self.cells.append(dict(id=cid,value=lbl,style=st,edge=True,source=s,target=t))
        return cid
    def to_xml(self):
        root = ET.Element("mxfile", host="app.diagrams.net")
        diag = ET.SubElement(root, "diagram", name=self.name, id="pi06")
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
    d = D("π0 / π*0.6 Architecture", pw=2400, ph=1600)

    BW = 120; BH = 34

    # ════════════════════════════════════════════════════════
    # (a) Full Pipeline — LEFT
    # ════════════════════════════════════════════════════════
    ax = 30; ay = 30
    d.lbl(ax, ay, 550, 22, "<b>(a) π0 / π*0.6 — Full Architecture</b>  (from openpi/models/pi0.py)", fs=12, color="#333", align="left")
    d.lbl(ax, ay+20, 550, 16, "PaliGemma (Gemma 2B) + Action Expert (Gemma 300M) + SigLIP (So400m/14)", fs=8, color="#888", align="left")

    # --- Inputs at bottom ---
    iy = ay + 720

    # Camera images
    cam_base = d.box(ax, iy, 85, 50, "🖼️<br><b>base_0</b><br>_rgb", fill=GRAY, stroke=GRAY_S, fs=8)
    cam_left = d.box(ax+95, iy, 85, 50, "🖼️<br><b>left_wrist</b><br>_0_rgb", fill=GRAY, stroke=GRAY_S, fs=8)
    cam_right = d.box(ax+190, iy, 85, 50, "🖼️<br><b>right_wrist</b><br>_0_rgb", fill=GRAY, stroke=GRAY_S, fs=8)

    # Language
    lang = d.box(ax+290, iy, 100, 50, "📝<br><b>tokenized</b><br><b>_prompt</b>", fill=GRAY, stroke=GRAY_S, fs=8)

    # State
    state = d.box(ax+410, iy, 80, 50, "🦾<br><b>state</b><br>(q, grip)", fill=GRAY, stroke=GRAY_S, fs=8)

    # Noisy actions
    noise_act = d.box(ax+510, iy, 80, 50, "🎲<br><b>x<sub>t</sub></b><br>noisy act", fill=FLOW, stroke=FLOW_S, fs=8)

    # Timestep
    timestep = d.box(ax+510, iy-60, 80, 36, "<b>t</b><br>timestep", fill=NORM, stroke=NORM_S, fs=9)

    # ═══ Vision: SigLIP ═══
    iy -= 80
    siglip = d.box(ax+30, iy, 210, 40,
        "<b>SigLIP</b>  (So400m/14)<br>_siglip.Module → image_tokens",
        fill=IMG, stroke=IMG_S, fs=9, bold=True)
    d.arr(cam_base, siglip)
    d.arr(cam_left, siglip)
    d.arr(cam_right, siglip)
    d.lbl(ax+245, iy+8, 80, 22, "(b, s, 2048)", fs=8, color=IMG_S)

    # ═══ Language embed ═══
    lang_emb = d.box(ax+290, iy, 100, 40,
        "<b>Gemma.embed</b><br>llm(tokens)",
        fill=VLM, stroke=VLM_S, fs=9)
    d.arr(lang, lang_emb)

    # ═══ embed_prefix() ═══
    iy -= 60
    prefix_grp = d.grp(ax, iy, 400, 45, "embed_prefix()", fill="#EEF2FF", stroke=VLM_S)
    prefix = d.box(ax+10, iy+8, 380, 28,
        "<b>concat</b>([image_tokens, lang_tokens])  +  ar_mask=[F..F]",
        fill=VLM, stroke=VLM_S, fs=9)
    d.arr(siglip, prefix)
    d.arr(lang_emb, prefix)

    # ═══ State projection ═══
    state_proj = d.box(ax+410, iy-40, 80, 30,
        "<b>state_proj</b><br>nn.Linear", fill=PROJ, stroke=PROJ_S, fs=8)
    d.arr(state, state_proj)
    d.lbl(ax+410, iy-10, 80, 14, "(b, 1, 300M_d)", fs=7, color=PROJ_S)

    # ═══ Action projection ═══
    act_proj = d.box(ax+510, iy-40, 80, 30,
        "<b>action_in</b><br><b>_proj</b><br>nn.Linear", fill=PROJ, stroke=PROJ_S, fs=8)
    d.arr(noise_act, act_proj)

    # ═══ Time embedding ═══
    time_emb = d.box(ax+510, iy-100, 80, 36,
        "<b>posemb</b><br><b>_sincos</b>", fill=NORM, stroke=NORM_S, fs=8)
    d.arr(timestep, time_emb)

    # ═══ Action + Time MLP ═══
    act_time = d.box(ax+430, iy-100, 70, 36,
        "<b>action_time</b><br><b>_mlp</b><br>SiLU", fill=MLP, stroke=MLP_S, fs=7)
    d.arr(act_proj, act_time)
    d.arr(time_emb, act_time)

    # ═══ embed_suffix() ═══
    suf_y = iy - 60
    suffix_grp = d.grp(ax+400, suf_y-45, 200, 40, "embed_suffix()", fill="#F0FFF0", stroke=ACT_S)
    suffix = d.box(ax+410, suf_y-38, 180, 26,
        "<b>concat</b>([state, action_time])  ar_mask=[T..T]",
        fill=ACT, stroke=ACT_S, fs=8)
    d.arr(state_proj, suffix)
    d.arr(act_time, suffix)

    # ═══ PaliGemma LLM (shared backbone) ═══
    llm_y = suf_y - 80
    llm_grp = d.grp(ax, llm_y-10, 600, 65, "", fill="#E8EAF6", stroke=VLM_S)

    d.lbl(ax+10, llm_y-5, 580, 14, "PaliGemma.llm([prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions)", fs=9, color=VLM_S, bold=True)

    # Prefix side (Gemma 2B)
    pali = d.box(ax+10, llm_y+12, 250, 34,
        "<b>Gemma 2B</b>  (PaliGemma backbone)<br>prefix → KV Cache",
        fill=VLM, stroke=VLM_S, fs=9, bold=True)
    d.arr(prefix, pali)

    # Action Expert side (Gemma 300M)
    ae = d.box(ax+280, llm_y+12, 250, 34,
        "<b>Gemma 300M</b>  (Action Expert)<br>suffix → shared attention w/ prefix KV",
        fill=ACT, stroke=ACT_S, fs=9, bold=True)
    d.arr(suffix, ae)

    # Shared attention arrow
    d.arr(pali, ae, lbl="cross-attn via mask", dash=True, color=VLM_S)

    # ═══ Output projection ═══
    out_y = llm_y - 55
    out_proj = d.box(ax+300, out_y, 200, 32,
        "<b>action_out_proj</b>  nn.Linear(300M_d → action_dim)",
        fill=PROJ, stroke=PROJ_S, fs=9)
    d.arr(ae, out_proj, lbl="suffix_out[:, -H:]")

    # ═══ v_t output ═══
    vt = d.box(ax+340, out_y-45, 120, 30,
        "<b>v<sub>t</sub></b> (velocity field)", fill=ACT, stroke=ACT_S, fs=10, bold=True)
    d.arr(out_proj, vt)

    # ═══ Euler integration ═══
    euler = d.box(ax+300, out_y-95, 200, 34,
        "<b>x<sub>t+dt</sub> = x<sub>t</sub> + dt · v<sub>t</sub></b>",
        fill=FLOW, stroke=FLOW_S, fs=10, bold=True)
    d.arr(vt, euler)
    d.lbl(ax+505, out_y-90, 60, 22, "×10 steps", fs=9, color=FLOW_S, bold=True)

    # ═══ Actions output ═══
    actions = d.box(ax+330, out_y-145, 140, 32,
        "<b>actions</b>  (b, H=50, d)",
        fill=OUT, stroke=OUT_S, fs=10, bold=True)
    d.arr(euler, actions)

    # ════════════════════════════════════════════════════════
    # (b) Attention Mask Pattern — RIGHT TOP
    # ════════════════════════════════════════════════════════
    bx = 700; by = 30
    d.lbl(bx, by, 400, 22, "<b>(b) Attention Mask (make_attn_mask)</b>", fs=12, color="#333", align="left")

    by += 30
    # Draw a simplified attention matrix
    d.grp(bx, by, 380, 220, "", fill="#FAFAFA", stroke="#CCC")

    # Token labels (columns = keys, rows = queries)
    segments = [
        ("img<sub>0</sub>", IMG, 60),
        ("img<sub>1</sub>", IMG, 60),
        ("img<sub>2</sub>", IMG, 60),
        ("lang", VLM, 50),
        ("state", PROJ, 40),
        ("act<sub>0..H</sub>", ACT, 70),
    ]

    sx = bx + 55
    for name, clr, w in segments:
        d.box(sx, by+8, w, 22, f"<b>{name}</b>", fill=clr, stroke="#999", fs=7)
        sx += w + 3

    # Matrix visualization description
    d.lbl(bx+10, by+40, 360, 80,
        "<b>Prefix</b> (images + lang): bidirectional (mask_ar=False)<br>"
        "→ all prefix tokens attend to each other<br><br>"
        "<b>Suffix</b> (state + actions): causal (mask_ar=True)<br>"
        "→ suffix attends to prefix + causal within suffix<br><br>"
        "<b>Key insight</b>: prefix CANNOT attend to suffix<br>"
        "(image/lang tokens don't see actions)",
        fs=9, color="#444", align="left")

    d.lbl(bx+10, by+130, 360, 60,
        "<b>make_attn_mask(input_mask, mask_ar):</b><br>"
        "cumsum = cumsum(mask_ar)<br>"
        "attn = cumsum[:, None, :] ≤ cumsum[:, :, None]<br>"
        "return attn ∧ (input_mask[:, None, :] · input_mask[:, :, None])",
        fs=8, color="#666", align="left")

    # ════════════════════════════════════════════════════════
    # (c) Flow Matching Detail — RIGHT MIDDLE
    # ════════════════════════════════════════════════════════
    cx = 700; cy = 290
    d.lbl(cx, cy, 400, 22, "<b>(c) Flow Matching (compute_loss / sample_actions)</b>", fs=12, color="#333", align="left")

    # Training
    cy += 28
    d.lbl(cx, cy, 150, 16, "<b>Training:</b>", fs=10, color=FLOW_S, bold=True, align="left")

    cy += 22
    t_beta = d.box(cx, cy, 130, 26, "<b>t ~ Beta(1.5, 1)</b><br>× 0.999 + 0.001", fill=NORM, stroke=NORM_S, fs=8)
    eps = d.box(cx+145, cy, 90, 26, "<b>ε ~ N(0,I)</b>", fill=FLOW, stroke=FLOW_S, fs=9)
    a_gt = d.box(cx+250, cy, 90, 26, "<b>a<sub>gt</sub></b> (demo)", fill=OUT, stroke=OUT_S, fs=9)

    cy += 38
    interp = d.box(cx+30, cy, 280, 28,
        "<b>x<sub>t</sub> = t · ε + (1-t) · a<sub>gt</sub></b>", fill=FLOW, stroke=FLOW_S, fs=10)
    d.arr(t_beta, interp)
    d.arr(eps, interp)
    d.arr(a_gt, interp)

    cy += 40
    target = d.box(cx+30, cy, 280, 28,
        "<b>u<sub>t</sub> = ε - a<sub>gt</sub></b>  (target velocity)", fill=OUT, stroke=OUT_S, fs=10)

    cy += 40
    loss = d.box(cx+60, cy, 220, 30,
        "<b>loss = mean(‖v<sub>t</sub> - u<sub>t</sub>‖²)</b>", fill=FLOW, stroke=FLOW_S, fs=10, bold=True)
    d.arr(target, loss)

    # Inference
    cy += 50
    d.lbl(cx, cy, 150, 16, "<b>Inference:</b>", fs=10, color=OUT_S, bold=True, align="left")

    cy += 22
    d.box(cx, cy, 350, 28,
        "<b>1.</b> prefix → PaliGemma.llm → <b>kv_cache</b>  (cached once)", fill=VLM, stroke=VLM_S, fs=9)
    cy += 35
    d.box(cx, cy, 350, 28,
        "<b>2.</b> x<sub>1</sub> = noise ~ N(0,I),  dt = -1/K", fill=FLOW, stroke=FLOW_S, fs=9)
    cy += 35
    d.box(cx, cy, 350, 50,
        "<b>3.</b> while t ≥ 0:<br>"
        "    suffix = embed_suffix(x<sub>t</sub>, t)<br>"
        "    v<sub>t</sub> = llm([None, suffix], kv_cache) → action_out_proj<br>"
        "    x<sub>t+dt</sub> = x<sub>t</sub> + dt · v<sub>t</sub>",
        fill=ACT, stroke=ACT_S, fs=9)
    cy += 58
    d.box(cx, cy, 350, 26,
        "<b>4.</b> return x<sub>0</sub> = denoised actions  (b, 50, action_dim)", fill=OUT, stroke=OUT_S, fs=9, bold=True)

    # ════════════════════════════════════════════════════════
    # (d) π*0.6 Recap — BOTTOM RIGHT
    # ════════════════════════════════════════════════════════
    rx = 700; ry = 710
    d.lbl(rx, ry, 450, 22, "<b>(d) π*0.6 — Recap (RL from Experience)</b>", fs=12, color="#333", align="left")

    ry += 28

    # 3 data sources
    demo = d.box(rx, ry, 110, 40, "👤<br><b>Demonstrations</b>", fill=VLM, stroke=VLM_S, fs=9)
    corr = d.box(rx+125, ry, 110, 40, "🔧<br><b>Corrections</b>", fill=MLP, stroke=MLP_S, fs=9)
    exp = d.box(rx+250, ry, 110, 40, "🤖<br><b>Autonomous</b><br><b>Experience</b>", fill=ACT, stroke=ACT_S, fs=9)

    # Value function
    ry += 58
    val_fn = d.box(rx+60, ry, 240, 36,
        "<b>Value Function V(s)</b><br>predicts -(steps to completion)",
        fill=RL, stroke=RL_S, fs=9, bold=True)
    d.arr(exp, val_fn)
    d.arr(corr, val_fn)

    # Advantage
    ry += 50
    adv = d.box(rx+60, ry, 240, 30,
        "<b>Advantage A = V(s') - V(s)</b><br>credit assignment per action",
        fill=RL, stroke=RL_S, fs=9)
    d.arr(val_fn, adv)

    # Advantage-conditioned policy
    ry += 48
    acond = d.box(rx+30, ry, 300, 36,
        "<b>π*0.6 = π0.6 conditioned on A</b><br>train on all data, condition on advantage<br>at test time: set A=high → better actions",
        fill=RL, stroke=RL_S, fs=9, bold=True)
    d.arr(adv, acond)
    d.arr(demo, acond, dash=True)

    # Training stages
    ry += 55
    d.lbl(rx, ry, 360, 16, "<b>Training Pipeline:</b>", fs=10, color=RL_S, align="left", bold=True)
    ry += 20
    s1 = d.box(rx, ry, 110, 36, "<b>Stage 1</b><br>Offline RL<br>(pre-train)", fill=VLM, stroke=VLM_S, fs=8)
    s2 = d.box(rx+125, ry, 110, 36, "<b>Stage 2</b><br>Demo<br>fine-tune", fill=ACT, stroke=ACT_S, fs=8)
    s3 = d.box(rx+250, ry, 110, 36, "<b>Stage 3</b><br>On-robot RL<br>(Recap)", fill=RL, stroke=RL_S, fs=8)
    d.arr(s1, s2)
    d.arr(s2, s3)

    # ════════════════════════════════════════════════════════
    # (e) π0.5 Differences — BOTTOM LEFT
    # ════════════════════════════════════════════════════════
    ex = 30; ey = 810
    d.lbl(ex, ey, 300, 22, "<b>(e) π0 vs π0.5 (config.pi05)</b>", fs=12, color="#333", align="left")

    ey += 28
    d.box(ex, ey, 280, 36,
        "<b>π0</b>: state_proj → suffix, action_time_mlp<br>"
        "concat([action, time]) → MLP → SiLU",
        fill=GRAY, stroke=GRAY_S, fs=8)
    ey += 44
    d.box(ex, ey, 280, 36,
        "<b>π0.5</b>: state as discrete lang tokens<br>"
        "time → time_mlp → <b>adaRMSNorm</b> in Action Expert",
        fill=NORM, stroke=NORM_S, fs=8)

    # ════════════════════════════════════════════════════════
    # Legend
    # ════════════════════════════════════════════════════════
    lx = 30; ly = 940
    d.lbl(lx, ly, 80, 18, "<b>Legend:</b>", fs=10, bold=True)
    items = [
        (IMG, IMG_S, "SigLIP (Vision)"),
        (VLM, VLM_S, "PaliGemma (VLM)"),
        (ACT, ACT_S, "Action Expert"),
        (PROJ, PROJ_S, "Linear Projection"),
        (MLP, MLP_S, "MLP / FFN"),
        (FLOW, FLOW_S, "Noise / Flow"),
        (OUT, OUT_S, "Output / GT"),
        (RL, RL_S, "Recap (RL)"),
        (NORM, NORM_S, "Norm / Embed"),
    ]
    for i, (f, s, l) in enumerate(items):
        col = i // 3; row = i % 3
        d.box(lx+col*180, ly+22+row*24, 16, 16, "", fill=f, stroke=s, fs=7)
        d.lbl(lx+20+col*180, ly+22+row*24, 150, 16, l, fs=9, color="#333", align="left")

    return d


if __name__ == "__main__":
    import os
    path = "/home/perelman/.openclaw/workspace/qwen_review/pi06/pi06_architecture.drawio"
    xml = build().to_xml()
    with open(path, "w") as f:
        f.write(xml)
    print(f"✅ {path} ({os.path.getsize(path)} bytes)")
