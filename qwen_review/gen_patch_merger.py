#!/usr/bin/env python3
"""
gen_patch_merger.py — Qwen3-VL PatchMerger Detail Diagram
Output: patch_merger_detail.drawio

Panels:
  (a) Spatial Merge 2×2 — visual explanation of token reduction
  (b) Final PatchMerger — use_postshuffle_norm=False (after block 26)
  (c) DeepStack PatchMerger — use_postshuffle_norm=True (at blocks 8,16,24)
  (d) ViT Pipeline — where mergers extract features from 27-block ViT
  (e) Comparison Table — Qwen2.5-VL vs Qwen3-VL variants

Style: Paper-figure, bottom-to-top flow, color-coded, PyTorch class names.
Source: transformers/models/qwen3_vl/modeling_qwen3_vl.py
"""

import os

# ── Colors: (fill, stroke) ──
C = {
    'input':  ('#dae8fc', '#6c8ebf'),   # Blue
    'norm':   ('#d5e8d4', '#82b366'),   # Green
    'linear': ('#ffe6cc', '#d6b656'),   # Orange
    'act':    ('#f8cecc', '#b85450'),   # Red
    'merge':  ('#e1d5e7', '#9673a6'),   # Purple
    'output': ('#fff2cc', '#d6b656'),   # Yellow
    'grid':   ('#f0f0f0', '#999999'),   # Gray
    'hl':     ('#ffcc80', '#e65100'),   # Orange highlight
    'block':  ('#e3f2fd', '#1565c0'),   # Light blue
    'tap':    ('#fce4ec', '#c62828'),   # Pink/red
    'white':  ('#ffffff', '#666666'),   # White
    'hdr':    ('#bbdefb', '#1565c0'),   # Header blue
    'key':    ('#e8eaf6', '#3949ab'),   # Key difference highlight
}

_id = 1
xml = []


def nid():
    global _id
    _id += 1
    return _id


def esc(s):
    """Escape for XML attribute (HTML tags become escaped, draw.io renders them)."""
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


def arr_up(src, tgt, label=''):
    """Upward arrow (bottom-to-top flow)."""
    cid = nid()
    xml.append(
        f'<mxCell id="{cid}" value="{esc(label)}" '
        f'style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#333333;fontSize=9;'
        f'exitX=0.5;exitY=0;exitDx=0;exitDy=0;'
        f'entryX=0.5;entryY=1;entryDx=0;entryDy=0;" '
        f'edge="1" parent="1" source="{src}" target="{tgt}">'
        f'<mxGeometry relative="1" as="geometry"/>'
        f'</mxCell>')
    return cid


def arr_tap(src, tgt, label=''):
    """Side tap arrow (red, thicker)."""
    cid = nid()
    xml.append(
        f'<mxCell id="{cid}" value="{esc(label)}" '
        f'style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#c62828;fontSize=9;'
        f'strokeWidth=2;'
        f'exitX=1;exitY=0.5;exitDx=0;exitDy=0;'
        f'entryX=0;entryY=0.5;entryDx=0;entryDy=0;" '
        f'edge="1" parent="1" source="{src}" target="{tgt}">'
        f'<mxGeometry relative="1" as="geometry"/>'
        f'</mxCell>')
    return cid


def arr_right(src, tgt, label=''):
    """Horizontal arrow left→right."""
    cid = nid()
    xml.append(
        f'<mxCell id="{cid}" value="{esc(label)}" '
        f'style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#666666;fontSize=9;'
        f'exitX=1;exitY=0.5;exitDx=0;exitDy=0;'
        f'entryX=0;entryY=0.5;entryDx=0;entryDy=0;" '
        f'edge="1" parent="1" source="{src}" target="{tgt}">'
        f'<mxGeometry relative="1" as="geometry"/>'
        f'</mxCell>')
    return cid


def vstack(items, x, y_bottom, w=270, gap=10):
    """Build bottom-to-top stack.
    items: [(label, height, color_key, font_size, bold), ...]
    Returns: list of (cell_id, y_top) bottom-to-top.
    """
    result = []
    y = y_bottom  # bottom edge of current item
    for label, h, col, fs, bld in items:
        y_top = y - h
        cid = box(label, x, y_top, w, h, col=col, fs=fs, bold=bld)
        result.append((cid, y_top))
        y = y_top - gap
    # Arrows from lower to higher
    for i in range(len(result) - 1):
        arr_up(result[i][0], result[i + 1][0])
    return result


# ════════════════════════════════════════════════════════════
# (a) Spatial Merge 2×2 Visualization
# ════════════════════════════════════════════════════════════

def panel_a():
    ox, oy = 20, 10

    # Title
    txt('<b>(a) Spatial Merge 2×2 — Token Count Reduction</b>', ox, oy, 950, 28, fs=14)

    # ── Left: Input grid 6×6 ──
    gx, gy = ox + 30, oy + 75
    cs = 36  # cell size
    R, CO = 6, 6

    txt('<b>Input Feature Map</b>', gx - 5, gy - 42, CO * cs, 22, fs=11)
    txt(f'({R}×{CO} patches, dim=1152)', gx - 5, gy - 22, CO * cs, 18, fs=9, color='#666')

    for r in range(R):
        for c in range(CO):
            br, bc = r // 2, c // 2
            ck = 'hl' if (br == 1 and bc == 1) else 'grid'
            box('', gx + c * cs, gy + r * cs, cs - 2, cs - 2, col=ck, fs=7)

    # 2×2 label
    txt('<b>2×2 block</b>', gx + 2 * cs - 5, gy + R * cs + 4, 80, 18, fs=9, color='#e65100')

    # ── Center: Zoom into merge operation ──
    zx = gx + CO * cs + 50
    zy = gy + 20

    txt('<b>Merge Operation (per 2×2 block)</b>', zx - 10, zy - 35, 310, 22, fs=11)

    # Draw 2×2 block
    ps = 44
    patch_labels = [('a', 0, 0), ('b', 0, 1), ('c', 1, 0), ('d', 1, 1)]
    patch_ids = []
    for lb, dr, dc in patch_labels:
        pid = box(f'<b>{lb}</b>', zx + dc * ps, zy + dr * ps, ps - 3, ps - 3, col='hl', fs=14)
        patch_ids.append(pid)

    txt('each: (1, 1152)', zx - 5, zy + 2 * ps + 5, 95, 16, fs=8, color='#666')

    # Arrow →
    ax = zx + 2 * ps + 20
    txt('→', ax, zy + 15, 25, 50, fs=24, bold=True, color='#9673a6')

    # Concatenated result
    cx = ax + 40
    concat = box('<b>[ a ‖ b ‖ c ‖ d ]</b>', cx, zy + 8, 180, 55, col='merge', fs=12)
    txt('4 × 1152 = <b>4608</b>', cx, zy + 66, 180, 18, fs=10, color='#9673a6')
    txt('Tensor.view(-1, 4608)', cx, zy + 84, 180, 16, fs=8, color='#666')

    # ── Right: Output grid 3×3 ──
    ogx = cx + 220
    ogy = gy
    mcs = 62
    MR, MC = 3, 3

    txt('<b>Merged Tokens</b>', ogx, ogy - 42, MC * mcs, 22, fs=11)
    txt(f'({MR}×{MC} tokens, dim=3584)', ogx, ogy - 22, MC * mcs, 18, fs=9, color='#666')

    for r in range(MR):
        for c in range(MC):
            ck = 'hl' if (r == 1 and c == 1) else 'merge'
            box('', ogx + c * mcs, ogy + r * mcs, mcs - 3, mcs - 3, col=ck, fs=8)

    txt('<b>4× fewer tokens</b>', ogx + 10, ogy + MR * mcs + 6, MC * mcs - 20, 20,
        fs=10, color='#9673a6')

    # ── Summary bar ──
    sy = gy + R * cs + 30
    box('spatial_merge_size = 2  ·  N patches → N/4 merged tokens  ·  '
        'Tokens reordered to block-major (2×2 groups adjacent in memory)',
        ox + 20, sy, 900, 34, col='white', fs=10)


panel_a()


# ════════════════════════════════════════════════════════════
# (b) Final PatchMerger (use_postshuffle_norm=False)
# ════════════════════════════════════════════════════════════

def panel_b():
    ox, oy = 20, 400
    BW = 280

    # Title
    txt('<b>(b) Final PatchMerger</b>', ox, oy, 320, 25, fs=13)
    txt('self.merger  ·  use_postshuffle_norm=False', ox, oy + 24, 320, 16, fs=9, color='#666')
    txt('After VisionBlock₂₆ (last block) → LLM input', ox, oy + 40, 320, 16, fs=9, color='#1565c0')

    items = [
        # (label, height, color, fontsize, bold)
        ('Input: <b>(N, 1152)</b>\nfrom VisionBlock₂₆',                48, 'input',  10, False),
        ('<b>nn.LayerNorm(1152)</b>\nself.norm\n❶ Normalize BEFORE merge', 52, 'norm',   10, False),
        ('<b>Tensor.view(-1, 4608)</b>\nSpatial Merge 2×2\n(N, 1152) → (N/4, 4608)',  52, 'merge',  10, False),
        ('<b>nn.Linear(4608, 4608)</b>\nself.linear_fc1',               44, 'linear', 10, False),
        ('<b>nn.GELU()</b>\nself.act_fn',                               36, 'act',    10, False),
        ('<b>nn.Linear(4608, 3584)</b>\nself.linear_fc2',               44, 'linear', 10, False),
        ('Output: <b>(N/4, 3584)</b>\n→ LLM token sequence',           48, 'output', 10, False),
    ]

    y_bottom = oy + 68 + sum(h + 10 for _, h, *_ in items)
    ids = vstack(items, ox + 20, y_bottom, w=BW, gap=10)

    # Right-side dimension annotations
    rx = ox + 20 + BW + 8
    # After norm
    txt('(N, 1152)', rx, ids[1][1] + 15, 100, 16, fs=9, color='#82b366', align='left')
    # After merge
    txt('<b>(N/4, 4608)</b>', rx, ids[2][1] + 15, 110, 16, fs=9, color='#9673a6', align='left')
    # After fc2
    txt('<b>(N/4, 3584)</b>', rx, ids[5][1] + 12, 110, 16, fs=9, color='#d6b656', align='left')

    # Highlight bracket: norm → merge order
    txt('❶ → ❷', rx + 5, ids[1][1] + 45, 60, 20, fs=10, bold=True, color='#2e7d32')
    txt('Norm first,\nthen merge', rx, ids[1][1] + 60, 110, 30, fs=8, color='#2e7d32', align='left')


panel_b()


# ════════════════════════════════════════════════════════════
# (c) DeepStack PatchMerger (use_postshuffle_norm=True)
# ════════════════════════════════════════════════════════════

def panel_c():
    ox, oy = 440, 400
    BW = 280

    txt('<b>(c) DeepStack PatchMerger</b>', ox, oy, 340, 25, fs=13)
    txt('deepstack_merger_list[i]  ·  use_postshuffle_norm=True',
        ox, oy + 24, 340, 16, fs=9, color='#666')
    txt('At VisionBlock 8, 16, 24 → inject into LLM layers 0, 1, 2',
        ox, oy + 40, 340, 16, fs=9, color='#c62828')

    items = [
        ('Input: <b>(N, 1152)</b>\nfrom VisionBlock₈ / ₁₆ / ₂₄',             48, 'input',  10, False),
        ('<b>Tensor.view(-1, 4608)</b>\nSpatial Merge 2×2\n❶ Merge FIRST',    52, 'merge',  10, False),
        ('<b>nn.LayerNorm(4608)</b>\nself.norm (postshuffle)\n❷ Normalize AFTER merge', 52, 'norm', 10, False),
        ('<b>nn.Linear(4608, 4608)</b>\nself.linear_fc1',                       44, 'linear', 10, False),
        ('<b>nn.GELU()</b>\nself.act_fn',                                       36, 'act',    10, False),
        ('<b>nn.Linear(4608, 3584)</b>\nself.linear_fc2',                       44, 'linear', 10, False),
        ('Output: <b>(N/4, 3584)</b>\n→ LLM Layer 0 / 1 / 2 (DeepStack)',     48, 'tap',    10, False),
    ]

    y_bottom = oy + 68 + sum(h + 10 for _, h, *_ in items)
    ids = vstack(items, ox + 20, y_bottom, w=BW, gap=10)

    # Right-side annotations
    rx = ox + 20 + BW + 8
    txt('<b>(N/4, 4608)</b>', rx, ids[1][1] + 15, 110, 16, fs=9, color='#9673a6', align='left')
    txt('(N/4, 4608)', rx, ids[2][1] + 15, 100, 16, fs=9, color='#82b366', align='left')
    txt('<b>(N/4, 3584)</b>', rx, ids[5][1] + 12, 110, 16, fs=9, color='#d6b656', align='left')

    # Highlight bracket: merge → norm order  (OPPOSITE of panel b)
    txt('❶ → ❷', rx + 5, ids[1][1] + 45, 60, 20, fs=10, bold=True, color='#c62828')
    txt('Merge first,\nthen norm', rx, ids[1][1] + 60, 110, 30, fs=8, color='#c62828', align='left')

    # Key difference callout
    kx = ox + 20
    ky = ids[-1][1] - 45
    box('⚠ Key Difference: Norm placement is SWAPPED\n'
        'Final: LN(1152) → merge → MLP\n'
        'DeepStack: merge → LN(4608) → MLP',
        kx, ky, BW, 48, col='key', fs=9)


panel_c()


# ════════════════════════════════════════════════════════════
# (d) ViT Pipeline — Merger Extraction Points
# ════════════════════════════════════════════════════════════

def panel_d():
    ox, oy = 860, 400

    txt('<b>(d) 27-Block ViT Pipeline — Merger Tap Points</b>',
        ox, oy, 420, 25, fs=13)
    txt('deepstack_visual_indexes = [8, 16, 24]',
        ox, oy + 24, 420, 16, fs=9, color='#c62828')

    BW, gap = 150, 5
    sx = ox + 30
    y = oy + 490  # start from bottom

    # PatchEmbed
    pe = box('<b>PatchEmbed</b>\nConv3d(3,1152,\nkernel=(2,16,16))', sx, y, BW, 48, col='input', fs=9)
    y -= (48 + gap + 4)

    # + Pos Embed
    pos = box('+ AbsPosEmbed\n+ 2D RoPE', sx, y, BW, 34, col='norm', fs=9)
    arr_up(pe, pos)
    y -= (34 + gap)

    # Block groups with taps
    groups = [
        ('Blocks 0–7',  8, False, None),
        ('Block 8',      1, True,  'DeepStack\nMerger₀\n→ LLM L0'),
        ('Blocks 9–15', 7, False, None),
        ('Block 16',     1, True,  'DeepStack\nMerger₁\n→ LLM L1'),
        ('Blocks 17–23', 7, False, None),
        ('Block 24',     1, True,  'DeepStack\nMerger₂\n→ LLM L2'),
        ('Blocks 25–26', 2, False, None),
    ]

    prev = pos
    for name, count, is_tap, tap_label in groups:
        h = max(26, min(count * 7, 50))
        if is_tap:
            b = box(f'<b>{name}</b>', sx, y, BW, h, col='tap', fs=9)
            tb = box(f'<b>{tap_label}</b>', sx + BW + 40, y - 4, 120, h + 8, col='tap', fs=8)
            arr_tap(b, tb)
        else:
            b = box(f'{name}\n({count}× VisionBlock)', sx, y, BW, h, col='block', fs=9)
        arr_up(prev, b)
        prev = b
        y -= (h + gap)

    # Final Merger
    fm = box('<b>Final Merger</b>\n(panel b)', sx, y, BW, 34, col='merge', fs=9)
    arr_up(prev, fm)
    y -= (34 + gap)

    # Output to LLM
    llm = box('<b>→ LLM Decoder</b>\n(N/4, 3584)', sx, y, BW, 34, col='output', fs=10)
    arr_up(fm, llm)


panel_d()


# ════════════════════════════════════════════════════════════
# (e) Comparison Table
# ════════════════════════════════════════════════════════════

def panel_e():
    ox, oy = 20, 870

    txt('<b>(e) PatchMerger Comparison: Qwen2.5-VL vs Qwen3-VL</b>',
        ox, oy, 750, 25, fs=13)

    tx, ty = ox + 10, oy + 32
    cw = [155, 175, 195, 210]  # column widths
    rh = 28

    headers = ['Property', 'Qwen2.5-VL', 'Qwen3-VL Final', 'Qwen3-VL DeepStack']
    rows = [
        ['Class',        'Qwen2_5_VLPatchMerger',  'Qwen3VLVisionPatchMerger',  'Qwen3VLVisionPatchMerger'],
        ['Count',        '1 merger only',           '1 (self.merger)',            '3 (deepstack_merger_list)'],
        ['Norm Type',    'RMSNorm',                 'LayerNorm',                  'LayerNorm'],
        ['Norm Dim',     '1280 (pre-merge)',        '1152 (pre-merge)',           '4608 (post-merge)'],
        ['Norm Order',   'Norm → Merge → MLP',     'Norm → Merge → MLP',        'Merge → Norm → MLP'],
        ['ViT Width',    '1280',                    '1152',                       '1152'],
        ['Merged Dim',   '4×1280 = 5120',          '4×1152 = 4608',             '4×1152 = 4608'],
        ['Output Dim',   '3584 (LLM hidden)',       '3584 (LLM hidden)',          '3584 (LLM hidden)'],
        ['Activation',   'nn.GELU()',               'nn.GELU()',                  'nn.GELU()'],
        ['Destination',  'LLM input tokens',        'LLM input tokens',           'LLM layers 0,1,2'],
        ['Source Block',  'Block 31 (last)',         'Block 26 (last)',            'Blocks 8, 16, 24'],
    ]

    # Draw headers
    cx = tx
    for j, h in enumerate(headers):
        box(f'<b>{h}</b>', cx, ty, cw[j], rh, col='hdr', fs=9)
        cx += cw[j] + 2

    # Draw rows
    for i, row in enumerate(rows):
        cx = tx
        for j, cell in enumerate(row):
            if j == 0:
                col = 'hdr'
            elif i in [3, 4]:  # Highlight key difference rows (Norm Dim, Norm Order)
                col = 'key'
            elif i % 2 == 0:
                col = 'grid'
            else:
                col = 'white'
            box(cell, cx, ty + (i + 1) * (rh + 2), cw[j], rh, col=col, fs=8,
                bold=(j == 0))
            cx += cw[j] + 2

    # Key difference annotation
    ky = ty + (len(rows) + 1) * (rh + 2) + 5
    box('★ Key Differences: (1) RMSNorm → LayerNorm  '
        '(2) DeepStack mergers normalize AFTER spatial merge (dim=4608 vs 1152)  '
        '(3) 3 additional mergers for multi-scale feature injection',
        tx, ky, sum(cw) + 6, 38, col='key', fs=9)


panel_e()


# ════════════════════════════════════════════════════════════
# Assemble XML
# ════════════════════════════════════════════════════════════

header = """\
<?xml version="1.0" encoding="UTF-8"?>
<mxfile>
<diagram name="PatchMerger Detail" id="patch_merger">
<mxGraphModel dx="1400" dy="1600" grid="1" gridSize="10" guides="1">
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
outpath = os.path.join(outdir, 'patch_merger_detail.drawio')
with open(outpath, 'w', encoding='utf-8') as f:
    f.write(output)

print(f'Generated: {outpath}')
print(f'Total mxCell elements: {_id}')
