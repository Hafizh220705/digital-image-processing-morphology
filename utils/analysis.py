def format_fence_analysis(results):
    lines = [
        f"**Total Lubang Terdeteksi:** {results['num_holes']}",
        f"**Komponen raw (sebelum filter):** {results['num_raw']}",
        "---",
    ]
    for h in results.get('holes', []):
        r, c = h['centroid']
        lines += [
            f"**Lubang #{h['id']}**",
            f"- Ukuran: `{h['size_pixels']} px`",
            f"- Centroid: `(row={r}, col={c})`",
            f"- Bounding box: `{h['bbox']}`",
        ]
    return "\n".join(lines)

def format_tetris_analysis(results):
    lines = [
        f"**Total Piece Terdeteksi:** {results['num_pieces']}",
        "---",
    ]
    for p in results.get('pieces', []):
        lines += [
            f"**Piece #{p['id']} — {p['type']}**",
            f"- Area: `{p['area']} px`",
            f"- Dimensi: `{p['height']}h × {p['width']}w`",
            f"- Aspect Ratio: `{p['aspect_ratio']}`",
            f"- Centroid: `{p['centroid']}`",
        ]
    return "\n".join(lines)

def format_diamond_analysis(results):
    lines = [
        f"**Total Shape:** {len(results.get('shapes', []))}",
        f"**Diamond (♦) Terdeteksi:** {results['num_diamonds']}",
        "---",
    ]
    for d in results.get('shapes', []):
        lines += [
            f"**Shape #{d['id']} — {d['type']}**",
            f"- Area: `{d['area']} px`",
            f"- Solidity: `{d['solidity']}` *(diamond ≈ 0.5)*",
            f"- Aspect Ratio: `{d['aspect_ratio']}`",
            f"- Centroid: `{d['centroid']}`",
        ]
    return "\n".join(lines)