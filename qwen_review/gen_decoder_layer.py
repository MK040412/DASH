#!/usr/bin/env python3
"""
gen_decoder_layer.py — Qwen3-VL Decoder Layer Detail Diagram
Output: decoder_layer_detail.drawio

Panels:
  (a) Overall Decoder Layer — Pre-LN Transformer block with residual connections
  (b) GQA Attention Detail — Q/K/V projections, q_norm/k_norm, M-RoPE, KV Cache, GQA
  (c) SwiGLU MLP Detail — gate_proj, up_proj, SiLU, element-wise multiply, down_proj
  (d) DeepStack Injection — _deepstack_process after layers 0,1,2
  (e) Config Table — Qwen3-VL-7B actual parameter values

Source: transformers/models/qwen3_vl/modeling_qwen3_vl.py
  Qwen3VLTextDecoderLayer (line 476)
  Qwen3VLTextAttention (line 385)
  Qwen3VLTextMLP (line 460)
  _deepstack_process (line 875)

Qwen3-VL-7B Config:
  hidden_size=3584, num_attention_heads=28, num_key_value_heads=4
  head_dim=128, intermediate_size=18944, num_hidden_layers=28
  attention_bias=True, hidden_act=silu, rms_norm_eps=1e-6
"""

import os

# ── Colors ──
C = {
    'input':  ('#dae8fc', '#6c8ebf'),   # Blue - input/output
    'norm':   ('#d5e8d4', '#82b366'),   # Green - normalization
    'linear': ('#ffe6cc', '#d6b656'),   # Orange - linear layers
    'act':    ('#f8cecc', '#b85450'),   # Red - activation
    'attn':   ('#e1d5e7', '#9673a6'),   # Purple - attention
    'output': ('#fff2cc', '#d6b656'),   # Yellow - output
    'res':    ('#f5f5f5', '#666666'),   # Gray - residual
    'kv':     ('#e8eaf6', '#3949ab'),   # Indigo - KV cache
    'tap':    ('#fce4ec', '#c62828'),   # Pink - DeepStack
    'white':  ('#ffffff', '#666666'),
    'hdr':    ('#bbdefb', '#1565c0'),   # Header blue
    'rope':   ('#e0f7fa', '#00838f'),   # Cyan - RoPE
    'gqa':    ('#fff3e0', '#e65100'),   # Light orange - GQA
    'swiglu': ('#fff9c4', '#f57f17'),   # Light yellow - SwiGLU
    'mul':    ('#efebe9', '#5d4037'),   # Brown - multiply
}

_id = 1
xml = []


def nid():
    global _id; _id += 1; return _id


def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')


def box(label, x, y, w, h, col='input', fs=11, bold=False, rounded=True):
    fill, stroke = C[col]
    cid = nid()
    xml.append(
        f'<mxCell id="{cid}" value="{esc(label)}" '
        f'style="rounded={"1" if rounded else "0"};whiteSpace=wrap;html=1;'
        f'fillColor={fill};strokeColor={stroke};fontSize={fs};'
        f'fontStyle={"1" if bold else "0"};arcSize=8;verticalAlign=middle;align=center;" '
        f'vertex="1" parent="1">'
        f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>'
        f'</mxCell>')
    return cid


def txt(label, x, y, w=200, h=20, fs=10, bold=False, align='center', color='#333333'):
    cid = nid()
    xml.append(
        f'<mxCell id="{cid}" value="{esc(label)}" '
        f'style="text;html=1;strokeColor=none;fillColor=none;fontColor={color};'
        f'fontSize={fs};fontStyle={"1" if bold else "0"};align={align};verticalAlign=middle;" '
        f'vertex="1" parent="1">'
        f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>'
        f'</mxCell>')
    return cid


def arr_up(src, tgt, label='', color='#333333', width=1):
    cid = nid()
    xml.append(
        f'<mxCell id="{cid}" value="{esc(label)}" '
        f'style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor={color};fontSize=9;'
        f'strokeWidth={width};'
        f'exitX=0.5;exitY=0;exitDx=0;exitDy=0;'
        f'entryX=0.5;entryY=1;entryDx=0;entryDy=0;" '
        f'edge="1" parent="1" source="{src}" target="{tgt}">'
        f'<mxGeometry relative="1" as="geometry"/>'
        f'</mxCell>')
    return cid


def arr_right(src, tgt, label='', color='#333333'):
    cid = nid()
    xml.append(
        f'<mxCell id="{cid}" value="{esc(label)}" '
        f'style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor={color};fontSize=9;'
        f'exitX=1;exitY=0.5;exitDx=0;exitDy=0;'
        f'entryX=0;entryY=0.5;entryDx=0;entryDy=0;" '
        f'edge="1" parent="1" source="{src}" target="{tgt}">'
        f'<mxGeometry relative="1" as="geometry"/>'
        f'</mxCell>')
    return cid


def arr_custom(src, tgt, ex, ey, nx, ny, label='', color='#333333', dashed=False, width=1):
    cid = nid()
    dash = 'dashed=1;' if dashed else ''
    xml.append(
        f'<mxCell id="{cid}" value="{esc(label)}" '
        f'style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor={color};fontSize=9;'
        f'strokeWidth={width};{dash}'
        f'exitX={ex};exitY={ey};exitDx=0;exitDy=0;'
        f'entryX={nx};entryY={ny};entryDx=0;entryDy=0;" '
        f'edge="1" parent="1" source="{src}" target="{tgt}">'
        f'<mxGeometry relative="1" as="geometry"/>'
        f'</mxCell>')
    return cid


def vstack(items, x, y_bottom, w=260, gap=10):
    result = []
    y = y_bottom
    for label, h, col, fs, bld in items:
        y_top = y - h
        cid = box(label, x, y_top, w, h, col=col, fs=fs, bold=bld)
        result.append((cid, y_top))
        y = y_top - gap
    for i in range(len(result) - 1):
        arr_up(result[i][0], result[i + 1][0])
    return result


# ════════════════════════════════════════════════════════════
# (a) Overall Decoder Layer — Pre-LN Transformer Block
# ════════════════════════════════════════════════════════════

def panel_a():
    ox, oy = 20, 10
    BW = 260

    txt('<b>(a) Qwen3VLTextDecoderLayer — Pre-LN Architecture</b>',
        ox, oy, 500, 28, fs=14)
    txt('28 layers × (GQA Attention + SwiGLU MLP) + DeepStack on layers 0,1,2',
        ox, oy + 26, 500, 18, fs=9, color='#666')

    items = [
        ('Input: <b>hidden_states</b>\n(B, S, 3584)', 44, 'input', 10, False),
        ('<b>input_layernorm</b>\nRMSNorm(3584)', 40, 'norm', 10, False),
        ('<b>self_attn</b>\nQwen3VLTextAttention\n(GQA: 28Q / 4KV heads)', 52, 'attn', 10, False),
        ('<b>⊕ Residual</b>', 28, 'res', 10, True),
        ('<b>post_attention_layernorm</b>\nRMSNorm(3584)', 40, 'norm', 10, False),
        ('<b>mlp</b>\nQwen3VLTextMLP\n(SwiGLU: 3584→18944→3584)', 52, 'swiglu', 10, False),
        ('<b>⊕ Residual</b>', 28, 'res', 10, True),
        ('Output: <b>hidden_states</b>\n(B, S, 3584)', 44, 'output', 10, False),
    ]

    y_bottom = oy + 60 + sum(h + 10 for _, h, *_ in items)
    ids = vstack(items, ox + 20, y_bottom, w=BW, gap=10)

    # Residual bypass arrows (input → ⊕ Add)
    # First residual: input → first ⊕
    arr_custom(ids[0][0], ids[3][0], 1, 0.5, 1, 0.5,
               color='#999999', dashed=True, label='residual')
    # Second residual: after first ⊕ → second ⊕
    arr_custom(ids[3][0], ids[6][0], 1, 0.5, 1, 0.5,
               color='#999999', dashed=True, label='residual')

    # DeepStack annotation
    ds_x = ox + 20 + BW + 15
    ds_y = ids[7][1] - 10
    ds = box('<b>DeepStack Injection</b>\n(layers 0, 1, 2 only)\nh[vis_mask] += visual_embeds',
             ds_x, ds_y, 220, 52, col='tap', fs=9)
    arr_custom(ids[7][0], ds, 1, 0.3, 0, 0.5, color='#c62828', width=2)

    # Right-side dim annotations
    rx = ox + 20 + BW + 5
    txt('(B, S, 3584)', rx, ids[0][1] + 12, 110, 16, fs=8, color='#6c8ebf', align='left')
    txt('(B, S, 3584)', rx, ids[2][1] + 16, 110, 16, fs=8, color='#9673a6', align='left')
    txt('(B, S, 3584)', rx, ids[5][1] + 16, 110, 16, fs=8, color='#d6b656', align='left')


panel_a()


# ════════════════════════════════════════════════════════════
# (b) GQA Attention Detail
# ════════════════════════════════════════════════════════════

def panel_b():
    ox, oy = 540, 10
    BW = 120  # narrow columns for Q/K/V parallel

    txt('<b>(b) Qwen3VLTextAttention — Grouped Query Attention</b>',
        ox, oy, 520, 28, fs=14)
    txt('28 query heads, 4 KV heads (7:1 GQA ratio), head_dim=128',
        ox, oy + 26, 520, 18, fs=9, color='#666')

    # Input at bottom
    inp_y = oy + 600
    inp = box('Input: <b>hidden_states</b>\n(B, S, 3584)', ox + 100, inp_y, 300, 40, col='input', fs=10)

    # Q/K/V parallel projections
    proj_y = inp_y - 60
    q_proj = box('<b>q_proj</b>\nLinear(3584→3584)\nbias=True', ox + 10, proj_y, BW, 50, col='linear', fs=9)
    k_proj = box('<b>k_proj</b>\nLinear(3584→512)\nbias=True', ox + 140, proj_y, BW, 50, col='linear', fs=9)
    v_proj = box('<b>v_proj</b>\nLinear(3584→512)\nbias=True', ox + 270, proj_y, BW, 50, col='linear', fs=9)

    arr_up(inp, q_proj)
    arr_up(inp, k_proj)
    arr_up(inp, v_proj)

    # Q/K RMSNorm (per-head)
    norm_y = proj_y - 45
    q_norm = box('<b>q_norm</b>\nRMSNorm(128)', ox + 10, norm_y, BW, 36, col='norm', fs=9)
    k_norm = box('<b>k_norm</b>\nRMSNorm(128)', ox + 140, norm_y, BW, 36, col='norm', fs=9)

    arr_up(q_proj, q_norm)
    arr_up(k_proj, k_norm)

    # Reshape annotation
    txt('→ (B, 28, S, 128)', ox + 10, norm_y - 18, BW, 16, fs=8, color='#666')
    txt('→ (B, 4, S, 128)', ox + 140, norm_y - 18, BW, 16, fs=8, color='#666')
    txt('→ (B, 4, S, 128)', ox + 270, proj_y - 18, BW, 16, fs=8, color='#666')

    # M-RoPE
    rope_y = norm_y - 56
    q_rope = box('<b>M-RoPE</b>\n3D position\n(t, h, w)', ox + 10, rope_y, BW, 44, col='rope', fs=9)
    k_rope = box('<b>M-RoPE</b>\n3D position\n(t, h, w)', ox + 140, rope_y, BW, 44, col='rope', fs=9)

    arr_up(q_norm, q_rope)
    arr_up(k_norm, k_rope)

    # KV Cache
    kvc_y = rope_y - 40
    kv_cache = box('<b>KV Cache</b>\nDynamicCache\nupdate(K, V)', ox + 140, kvc_y, 250, 34, col='kv', fs=9)
    arr_up(k_rope, kv_cache)
    arr_custom(v_proj, kv_cache, 0.5, 0, 0.8, 1, color='#3949ab')

    # Attention computation
    attn_y = kvc_y - 52
    attn = box('<b>Scaled Dot-Product Attention</b>\nsoftmax(QK<sup>T</sup> / √128) · V\n'
               'GQA: 7 Q heads share 1 KV head\ncausal mask applied',
               ox + 10, attn_y, 380, 48, col='attn', fs=9)
    arr_up(q_rope, attn)
    arr_up(kv_cache, attn)

    # O projection
    oproj_y = attn_y - 44
    o_proj = box('<b>o_proj</b>\nLinear(3584→3584, bias=True)', ox + 60, oproj_y, 280, 36, col='linear', fs=9)
    arr_up(attn, o_proj)

    # Output
    out_y = oproj_y - 40
    out = box('Output: <b>attn_output</b>\n(B, S, 3584)', ox + 100, out_y, 300, 36, col='output', fs=10)
    arr_up(o_proj, out)

    # GQA ratio visual
    gqa_x = ox + 410
    gqa_y = attn_y - 5
    box('<b>GQA 7:1</b>\nQ heads: 28\nKV heads: 4\nGroups: 4\n7 Q per KV',
        gqa_x, gqa_y, 120, 80, col='gqa', fs=9)


panel_b()


# ════════════════════════════════════════════════════════════
# (c) SwiGLU MLP Detail
# ════════════════════════════════════════════════════════════

def panel_c():
    ox, oy = 20, 650
    BW = 150

    txt('<b>(c) Qwen3VLTextMLP — SwiGLU Feed-Forward</b>',
        ox, oy, 400, 28, fs=14)
    txt('Gated Linear Unit with SiLU activation, no bias on projections',
        ox, oy + 26, 400, 18, fs=9, color='#666')

    # Input
    inp_y = oy + 410
    inp = box('Input: <b>x</b>\n(B, S, 3584)', ox + 70, inp_y, 240, 36, col='input', fs=10)

    # Parallel gate + up
    par_y = inp_y - 54
    gate = box('<b>gate_proj</b>\nLinear(3584→18944)\nbias=False', ox + 10, par_y, BW, 48, col='linear', fs=9)
    up = box('<b>up_proj</b>\nLinear(3584→18944)\nbias=False', ox + 210, par_y, BW, 48, col='linear', fs=9)

    arr_up(inp, gate)
    arr_up(inp, up)

    # SiLU on gate
    silu_y = par_y - 40
    silu = box('<b>SiLU()</b>\nσ(x) · x', ox + 10, silu_y, BW, 34, col='act', fs=10)
    arr_up(gate, silu)

    # Element-wise multiply
    mul_y = silu_y - 42
    mul = box('<b>⊙ Element-wise Multiply</b>\nSiLU(gate) * up\n(B, S, 18944)',
              ox + 50, mul_y, 260, 40, col='mul', fs=9)
    arr_up(silu, mul)
    arr_up(up, mul)

    # down_proj
    down_y = mul_y - 44
    down = box('<b>down_proj</b>\nLinear(18944→3584)\nbias=False', ox + 80, down_y, 210, 38, col='linear', fs=9)
    arr_up(mul, down)

    # Output
    out_y = down_y - 38
    out = box('Output: <b>mlp_output</b>\n(B, S, 3584)', ox + 70, out_y, 240, 34, col='output', fs=10)
    arr_up(down, out)

    # SwiGLU formula
    txt('<b>SwiGLU(x) = (SiLU(W_gate · x) ⊙ W_up · x) · W_down</b>',
        ox + 10, out_y - 30, 350, 22, fs=10, color='#5d4037')

    # Dim annotations
    txt('(B, S, 18944)', ox + 170, par_y + 14, 100, 16, fs=8, color='#666', align='left')


panel_c()


# ════════════════════════════════════════════════════════════
# (d) DeepStack Injection Detail
# ════════════════════════════════════════════════════════════

def panel_d():
    ox, oy = 440, 650

    txt('<b>(d) DeepStack: _deepstack_process()</b>',
        ox, oy, 400, 28, fs=14)
    txt('Visual features injected into LLM hidden states at layers 0, 1, 2',
        ox, oy + 26, 400, 18, fs=9, color='#666')

    # Flow from bottom to top
    BW = 300
    items = [
        ('Input: <b>hidden_states</b>\n(B, S, 3584) — full sequence', 44, 'input', 10, False),
        ('<b>visual_pos_masks</b>\nBoolean mask → visual token positions', 40, 'kv', 10, False),
        ('<b>h[vis_mask, :].clone()</b>\nExtract visual positions\n(B, N_vis, 3584)', 48, 'attn', 10, False),
        ('<b>⊕ Add visual_embeds</b>\nfrom DeepStack PatchMerger\n(N_vis, 3584)', 48, 'tap', 10, False),
        ('<b>h[vis_mask, :] = result</b>\nWrite back to original positions', 40, 'tap', 10, False),
        ('Output: <b>hidden_states</b>\n(B, S, 3584) — updated', 44, 'output', 10, False),
    ]

    y_bottom = oy + 60 + sum(h + 10 for _, h, *_ in items)
    ids = vstack(items, ox + 20, y_bottom, w=BW, gap=10)

    # Mapping annotation
    rx = ox + 20 + BW + 10
    map_y = ids[3][1]
    box('Mapping:\nLayer 0 ← VisionBlock 8 → Merger₀\n'
        'Layer 1 ← VisionBlock 16 → Merger₁\n'
        'Layer 2 ← VisionBlock 24 → Merger₂',
        rx, map_y - 10, 210, 72, col='tap', fs=9)

    # Important note
    note_y = ids[0][1] + 55
    txt('⚠ Only on layers 0, 1, 2 (len(deepstack_visual_indexes))',
        ox + 20, note_y, BW, 18, fs=9, bold=True, color='#c62828')


panel_d()


# ════════════════════════════════════════════════════════════
# (e) Config Table — Qwen3-VL-7B Parameters
# ════════════════════════════════════════════════════════════

def panel_e():
    ox, oy = 20, 1100

    txt('<b>(e) Qwen3-VL-7B Decoder Configuration</b>',
        ox, oy, 600, 25, fs=13)

    tx, ty = ox + 10, oy + 30
    cw = [170, 150, 280]
    rh = 26

    headers = ['Parameter', 'Value', 'Notes']
    rows = [
        ['hidden_size', '3584', 'LLM hidden dimension'],
        ['num_hidden_layers', '28', 'Decoder layers'],
        ['num_attention_heads', '28', 'Query heads'],
        ['num_key_value_heads', '4', 'KV heads (GQA ratio 7:1)'],
        ['head_dim', '128', '3584 / 28'],
        ['intermediate_size', '18944', 'SwiGLU intermediate (≈5.29× hidden)'],
        ['hidden_act', 'silu', 'SiLU = Swish = x·σ(x)'],
        ['attention_bias', 'True', 'Q, K, V, O all have bias'],
        ['rms_norm_eps', '1e-6', 'RMSNorm epsilon'],
        ['Norm type', 'RMSNorm', 'NOT LayerNorm (Vision uses LayerNorm)'],
        ['Q/K per-head norm', 'RMSNorm(128)', 'q_norm, k_norm applied after projection'],
        ['Position encoding', 'M-RoPE (3D)', 'Interleaved: temporal, height, width'],
        ['DeepStack layers', '0, 1, 2', 'Visual injection from ViT blocks 8, 16, 24'],
    ]

    cx = tx
    for j, h in enumerate(headers):
        box(f'<b>{h}</b>', cx, ty, cw[j], rh, col='hdr', fs=9)
        cx += cw[j] + 2

    for i, row in enumerate(rows):
        cx = tx
        for j, cell in enumerate(row):
            col = 'hdr' if j == 0 else ('res' if i % 2 == 0 else 'white')
            box(cell, cx, ty + (i + 1) * (rh + 2), cw[j], rh, col=col, fs=8,
                bold=(j == 0))
            cx += cw[j] + 2


panel_e()


# ════════════════════════════════════════════════════════════
# Assemble XML
# ════════════════════════════════════════════════════════════

header = """\
<?xml version="1.0" encoding="UTF-8"?>
<mxfile>
<diagram name="Decoder Layer Detail" id="decoder_layer">
<mxGraphModel dx="1400" dy="1800" grid="1" gridSize="10" guides="1">
<root>
<mxCell id="0"/>
<mxCell id="1" parent="0"/>"""

footer = """\
</root>
</mxGraphModel>
</diagram>
</mxfile>"""

output = header + '\n' + '\n'.join(xml) + '\n' + footer

outdir = os.path.dirname(os.path.abspath(__file__))
outpath = os.path.join(outdir, 'decoder_layer_detail.drawio')
with open(outpath, 'w', encoding='utf-8') as f:
    f.write(output)

print(f'Generated: {outpath}')
print(f'Total mxCell elements: {_id}')
