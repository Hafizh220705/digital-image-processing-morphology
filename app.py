import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import io
import os

from utils.image_gen   import generate_chainlink_fence, generate_tetris_board, generate_card_diamonds
from utils.morphology  import (
    to_binary,
    apply_erosion, apply_dilation, apply_opening,
    apply_closing, apply_morph_gradient, apply_skeletonize,
    detect_fence_holes, detect_tetris_pieces, detect_card_diamonds,
)
from utils.analysis    import format_fence_analysis, format_tetris_analysis, format_diamond_analysis

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Morphological Image Processing",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────
def plot_image(ax, img, title, cmap='gray'):
    ax.imshow(img, cmap=cmap, interpolation='nearest')
    ax.set_title(title, fontsize=9, fontweight='bold', pad=5)
    ax.axis('off')

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    return buf

def load_default_photo():
    """Load foto default dari assets/, fallback ke gambar generated."""
    path = "assets/default_photo.jpg"
    if os.path.exists(path):
        img_color = np.array(Image.open(path).convert("RGB").resize((300, 300)))
        img_gray  = np.array(Image.open(path).convert("L").resize((300, 300)))
    else:
        # Fallback: buat gambar test pattern kalau file tidak ada
        img_gray = np.zeros((300, 300), dtype=np.uint8)
        for i in range(0, 300, 30):
            img_gray[i:i+15, :] = 128 + (i % 60)
        for j in range(0, 300, 40):
            img_gray[:, j:j+20] = np.clip(img_gray[:, j:j+20] + 60, 0, 255)
        img_color = np.stack([img_gray, img_gray, img_gray], axis=-1)
    return img_color, img_gray

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("Morphological Dashboard")
st.sidebar.caption("Hafizh Fadhl Muhammad — 140810230070")
st.sidebar.divider()

section = st.sidebar.radio(
    "Pilih Section:",
    ["Section 1: Morphological Operations", "Section 2: Challenge Solver"],
)

st.sidebar.divider()
with st.sidebar.expander("Teori Singkat"):
    st.markdown("""
**Erosion** — Susutkan foreground, hapus piksel di batas.  
**Dilation** — Perluas foreground, tambah piksel di batas.  
**Opening** = Erosion -> Dilation *(hilangkan noise)*  
**Closing** = Dilation -> Erosion *(tutup lubang)*  
**Morph. Gradient** = Dilation - Erosion *(deteksi tepi)*  
**Skeletonize** = Tipiskan objek jadi 1px *(kerangka)*
    """)


# ══════════════════════════════════════════════
# SECTION 1: MORPHOLOGICAL OPERATIONS EXPLORER
# ══════════════════════════════════════════════
if section == "Section 1: Morphological Operations":

    st.title("Section 1: Morphological Operations Explorer")
    st.divider()

    # ── INPUT IMAGE ──────────────────────────────
    st.subheader("Input Image")
    col_src, col_preview = st.columns([1, 2])

    with col_src:
        input_mode = st.radio(
            "Sumber Gambar:",
            ["Gunakan Foto Default", "Upload Gambar"],
            horizontal=True,
        )

        if input_mode == "Upload Gambar":
            uploaded = st.file_uploader("Upload gambar (PNG/JPG):", type=["png", "jpg", "jpeg"])
            if uploaded:
                img_color = np.array(Image.open(uploaded).convert("RGB").resize((300, 300)))
                img_gray  = np.array(Image.open(uploaded).convert("L").resize((300, 300)))
                st.success("Gambar berhasil diupload!")
            else:
                st.info("Belum ada file — menggunakan foto default.")
                img_color, img_gray = load_default_photo()
        else:
            img_color, img_gray = load_default_photo()

        raw_img = img_gray
        binary_img, otsu_thresh = to_binary(raw_img)
        st.caption(f"Ukuran: `{raw_img.shape}` | Otsu threshold: `{otsu_thresh:.3f}`")

    with col_preview:
        # Tampilkan 3 gambar: Default (Color), Grayscale, Binary
        fig_prev, axes_prev = plt.subplots(1, 3, figsize=(10, 3.5))
        axes_prev[0].imshow(img_color, interpolation='nearest')
        axes_prev[0].set_title("Default (Color)", fontsize=9, fontweight='bold', pad=5)
        axes_prev[0].axis('off')
        plot_image(axes_prev[1], raw_img,    "Input (Grayscale)")
        plot_image(axes_prev[2], binary_img, "Binary (Otsu Threshold)")
        fig_prev.tight_layout()
        st.pyplot(fig_prev)
        plt.close(fig_prev)

    st.divider()

    # ── PARAMETER SLIDERS ────────────────────────
    st.subheader("Parameter per Operasi")

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        r_erosion   = st.slider("Erosion — Disk Radius",   1, 15, 3)
        r_dilation  = st.slider("Dilation — Disk Radius",  1, 15, 3)
    with col_p2:
        r_opening   = st.slider("Opening — Disk Radius",   1, 15, 3)
        r_closing   = st.slider("Closing — Disk Radius",   1, 15, 3)
    with col_p3:
        r_gradient  = st.slider("Morph. Gradient — Disk Radius", 1, 10, 2)
        st.markdown("**Skeletonize** — tidak ada parameter (morphological thinning)")

    st.divider()

    # ── TOMBOL 6 OPERASI ─────────────────────────
    st.subheader("Jalankan Operasi")

    col_b1, col_b2, col_b3, col_b4, col_b5, col_b6 = st.columns(6)
    run_erosion   = col_b1.button("Erosion",          use_container_width=True)
    run_dilation  = col_b2.button("Dilation",         use_container_width=True)
    run_opening   = col_b3.button("Opening",          use_container_width=True)
    run_closing   = col_b4.button("Closing",          use_container_width=True)
    run_gradient  = col_b5.button("Morph. Gradient",  use_container_width=True)
    run_skeleton  = col_b6.button("Skeletonize",      use_container_width=True)

    # Simpan state operasi yang sudah dijalankan
    if 'ops_results' not in st.session_state:
        st.session_state.ops_results = {}

    if run_erosion:
        st.session_state.ops_results['erosion'] = {
            'img':   apply_erosion(binary_img, r_erosion),
            'label': f"Erosion (r={r_erosion})",
            'color': '#FF4B4B',
            'desc':  "Foreground **menyusut** — piksel di batas dihapus. Noise kecil hilang, objek mengecil.",
        }
    if run_dilation:
        st.session_state.ops_results['dilation'] = {
            'img':   apply_dilation(binary_img, r_dilation),
            'label': f"Dilation (r={r_dilation})",
            'color': '#21C354',
            'desc':  "Foreground **melebar** — piksel ditambah di batas. Celah kecil tertutup, objek membesar.",
        }
    if run_opening:
        st.session_state.ops_results['opening'] = {
            'img':   apply_opening(binary_img, r_opening),
            'label': f"Opening (r={r_opening})",
            'color': '#FFD700',
            'desc':  "**Erosion -> Dilation.** Noise kecil hilang, koneksi tipis antar objek terputus.",
        }
    if run_closing:
        st.session_state.ops_results['closing'] = {
            'img':   apply_closing(binary_img, r_closing),
            'label': f"Closing (r={r_closing})",
            'color': '#1C83E1',
            'desc':  "**Dilation -> Erosion.** Lubang kecil tertutup, celah kecil antar objek tersambung.",
        }
    if run_gradient:
        st.session_state.ops_results['gradient'] = {
            'img':   apply_morph_gradient(binary_img, r_gradient),
            'label': f"Morph. Gradient (r={r_gradient})",
            'color': '#FF8C00',
            'desc':  "**Dilation - Erosion.** Menghasilkan tepi/edge dari objek foreground.",
        }
    if run_skeleton:
        st.session_state.ops_results['skeleton'] = {
            'img':   apply_skeletonize(binary_img),
            'label': "Skeletonize",
            'color': '#A855F7',
            'desc':  "**Morphological thinning.** Menipiskan objek jadi 1 piksel lebar (kerangka/tulang).",
        }

    # ── TAMPILKAN HASIL ───────────────────────────
    if st.session_state.ops_results:
        st.divider()
        st.subheader("Hasil Operasi")

        ops = list(st.session_state.ops_results.values())
        n   = len(ops)
        ncols = min(n + 1, 4)
        nrows = (n + 1 + ncols - 1) // ncols

        fig_res, axes_res = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
        if nrows == 1:
            axes_res = [axes_res] if ncols == 1 else list(axes_res)
        else:
            axes_res = [ax for row in axes_res for ax in row]

        plot_image(axes_res[0], binary_img, "Original Binary", cmap='gray')

        for i, op in enumerate(ops, 1):
            if i < len(axes_res):
                plot_image(axes_res[i], op['img'], op['label'], cmap='gray')

        for ax in axes_res[n + 1:]:
            ax.set_visible(False)

        fig_res.suptitle("Hasil Morphological Operations", fontsize=13, fontweight='bold')
        fig_res.tight_layout()
        st.pyplot(fig_res)
        plt.close(fig_res)

        # Penjelasan tiap operasi
        st.subheader("Penjelasan Hasil")
        for key, op in st.session_state.ops_results.items():
            with st.expander(f"{op['label']}", expanded=True):
                col_img, col_txt = st.columns([1, 2])
                with col_img:
                    fig_s, ax_s = plt.subplots(1, 2, figsize=(5, 2.5))
                    plot_image(ax_s[0], binary_img, "Original")
                    plot_image(ax_s[1], op['img'],  op['label'])
                    fig_s.tight_layout()
                    st.pyplot(fig_s)
                    plt.close(fig_s)
                with col_txt:
                    st.markdown(op['desc'])
                    diff = op['img'].astype(int) - binary_img.astype(int)
                    gained = int((diff > 0).sum())
                    lost   = int((diff < 0).sum())
                    st.markdown(f"- Piksel **bertambah**: `{gained}`")
                    st.markdown(f"- Piksel **berkurang**: `{lost}`")
                    st.markdown(f"- Foreground sebelum: `{int(binary_img.sum())} px`")
                    st.markdown(f"- Foreground sesudah: `{int(op['img'].sum())} px`")

        if st.button("Reset Semua Hasil"):
            st.session_state.ops_results = {}
            st.rerun()


# ══════════════════════════════════════════════
# SECTION 2: CHALLENGE SOLVER
# ══════════════════════════════════════════════
else:
    st.title("Section 2: Challenge Solver")
    st.divider()

    challenge = st.tabs([
        "Challenge 1: Chain-link Fence",
        "Challenge 2: Tetris Pieces",
        "Challenge 3: Card Diamonds",
    ])

    # ──────────────────────────────────────────
    # CHALLENGE 1: CHAIN-LINK FENCE
    # ──────────────────────────────────────────
    with challenge[0]:
        st.header("Deteksi Lubang pada Chain-Link Fence")
        st.markdown("""
        **Tujuan:** Deteksi dan lokalisasi **lubang** pada gambar chain-link fence menggunakan morphological operators.  
        **Pipeline:** Erosion -> Fill Holes -> XOR -> Closing -> Labeling -> Filter Size
        """)
        st.divider()

        col_in, col_par = st.columns([1, 1])

        with col_in:
            st.subheader("Input")
            mode = st.radio("Sumber:", ["Default (Generated)", "Upload Gambar"], horizontal=True, key="c1_mode")
            if mode == "Upload Sendiri":
                up = st.file_uploader("Upload gambar fence:", type=["png","jpg","jpeg"], key="c1_up")
                raw = np.array(Image.open(up).convert("L").resize((300,300))) if up else generate_chainlink_fence(300)
                if not up: st.info("Menggunakan gambar default.")
            else:
                raw = generate_chainlink_fence(300)

            binary, thresh = to_binary(raw)
            st.caption(f"Otsu threshold: `{thresh:.3f}`")

            fig0, ax0 = plt.subplots(1, 2, figsize=(6, 3))
            plot_image(ax0[0], raw,    "Input (Grayscale)")
            plot_image(ax0[1], binary, "Binary (Otsu)")
            fig0.tight_layout()
            st.pyplot(fig0)
            plt.close(fig0)

        with col_par:
            st.subheader("Parameter")
            ero_iter      = st.slider("Iterasi Erosion",      1, 15,  5, key="c1_ero")
            min_hole_size = st.slider("Min. Hole Size (px)", 20, 500, 100, key="c1_min")
            st.markdown("### Pipeline Steps")
            st.markdown("""
| Step | Operasi | Tujuan |
|------|---------|--------|
| 1 | **Erosion** | Hilangkan noise fence |
| 2 | **Fill Holes** | Isi semua lubang |
| 3 | **XOR** | Isolasi region lubang |
| 4 | **Closing** | Satukan lubang berdekatan |
| 5 | **Labeling** | Beri ID tiap komponen |
| 6 | **Filter** | Buang region terlalu kecil |
            """)

        st.divider()
        if st.button("Jalankan Deteksi Lubang", key="c1_run", use_container_width=True):
            with st.spinner("Memproses..."):
                res = detect_fence_holes(binary, erosion_iter=ero_iter, min_hole_size=min_hole_size)

            st.subheader("Step-by-Step Visualisasi")
            fig1, axes1 = plt.subplots(2, 4, figsize=(16, 8))
            axes1 = axes1.flatten()

            steps = [
                (res['original'],        'gray',          'Step 0: Binary Original'),
                (res['eroded'],          'gray',          f'Step 1: Eroded ({ero_iter}x)'),
                (res['filled'],          'gray',          'Step 2: Filled Holes'),
                (res['holes_mask'],      'hot',           'Step 3: Holes Mask (XOR)'),
                (res['closed'],          'gray',          'Step 4: Closing'),
                (res['labeled'],         'nipy_spectral', 'Step 5: Labeled (raw)'),
                (res['filtered_labeled'],'nipy_spectral', f'Step 6: Filtered (>={min_hole_size}px)'),
            ]
            for ax, (img, cmap, title) in zip(axes1[:7], steps):
                plot_image(ax, img, title, cmap=cmap)

            # Overlay final
            overlay = np.stack([res['original']] * 3, axis=-1).astype(float)
            for h in res['holes']:
                rmin, cmin, rmax, cmax = h['bbox']
                overlay[rmin:rmax, cmin:cmax, 0] = 1.0
                overlay[rmin:rmax, cmin:cmax, 1] = 0.0
                overlay[rmin:rmax, cmin:cmax, 2] = 0.0
            axes1[7].imshow(overlay)
            axes1[7].set_title(f'Final: {res["num_holes"]} Lubang', fontsize=9, fontweight='bold')
            axes1[7].axis('off')
            for h in res['holes']:
                r, c = h['centroid']
                axes1[7].annotate(f"#{h['id']}", xy=(c, r), color='yellow',
                                  fontsize=11, fontweight='bold', ha='center')

            fig1.suptitle("Chain-link Fence — Morphological Pipeline", fontsize=12, fontweight='bold')
            fig1.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)

            st.divider()
            st.subheader("Analisis Hasil")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Lubang Terdeteksi", res['num_holes'])
                st.markdown(format_fence_analysis(res))
            with col_b:
                if res['holes']:
                    sizes  = [h['size_pixels'] for h in res['holes']]
                    labels = [f"Lubang #{h['id']}" for h in res['holes']]
                    fig_b, ax_b = plt.subplots(figsize=(5, 3))
                    ax_b.bar(labels, sizes, color='steelblue')
                    ax_b.set_ylabel("Ukuran (piksel)")
                    ax_b.set_title("Ukuran Tiap Lubang")
                    st.pyplot(fig_b)
                    plt.close(fig_b)

    # ──────────────────────────────────────────
    # CHALLENGE 2: TETRIS
    # ──────────────────────────────────────────
    with challenge[1]:
        st.header("Deteksi Piece pada Gambar Tetris")
        st.markdown("""
        **Tujuan:** Deteksi, pisahkan, dan klasifikasikan **piece Tetris** menggunakan morphological operators.  
        **Pipeline:** Erosion (disk) -> Labeling -> Dilation -> Morph. Gradient -> Skeletonize -> Klasifikasi
        """)
        st.divider()

        col_in2, col_par2 = st.columns([1, 1])

        with col_in2:
            st.subheader("Input")
            mode2 = st.radio("Sumber:", ["Default (Generated)", "Upload Sendiri"], horizontal=True, key="c2_mode")
            if mode2 == "Upload Sendiri":
                up2  = st.file_uploader("Upload gambar Tetris:", type=["png","jpg","jpeg"], key="c2_up")
                raw2 = np.array(Image.open(up2).convert("L").resize((300,300))) if up2 else generate_tetris_board(300)
                if not up2: st.info("Menggunakan gambar default.")
            else:
                raw2 = generate_tetris_board(300)

            binary2, thresh2 = to_binary(raw2)
            st.caption(f"Otsu threshold: `{thresh2:.3f}`")

            fig0b, ax0b = plt.subplots(1, 2, figsize=(6, 3))
            plot_image(ax0b[0], raw2,    "Input (Grayscale)")
            plot_image(ax0b[1], binary2, "Binary (Otsu)")
            fig0b.tight_layout()
            st.pyplot(fig0b)
            plt.close(fig0b)

        with col_par2:
            st.subheader("Parameter")
            ero_r2 = st.slider("Erosion Radius (disk)",  1, 10, 3, key="c2_ero")
            dil_r2 = st.slider("Dilation Radius (disk)", 1, 10, 4, key="c2_dil")
            st.markdown("### Pipeline Steps")
            st.markdown("""
| Step | Operasi | Tujuan |
|------|---------|--------|
| 1 | **Erosion (disk)** | Pisahkan piece |
| 2 | **Labeling** | Beri ID tiap piece |
| 3 | **Dilation** | Restore ukuran |
| 4 | **Morph. Gradient** | Tepi tiap piece |
| 5 | **Skeletonize** | Kerangka piece |
| 6 | **Klasifikasi** | Tipe piece dari aspect ratio |
            """)

        st.divider()
        if st.button("Jalankan Deteksi Piece", key="c2_run", use_container_width=True):
            with st.spinner("Memproses..."):
                res2 = detect_tetris_pieces(binary2, erosion_radius=ero_r2, dilation_radius=dil_r2)

            st.subheader("Step-by-Step Visualisasi")
            fig2, axes2 = plt.subplots(2, 3, figsize=(15, 9))
            axes2 = axes2.flatten()
            steps2 = [
                (res2['original'],    'gray',          'Step 0: Binary Original'),
                (res2['eroded'],      'gray',          f'Step 1: Eroded (r={ero_r2})'),
                (res2['labeled_raw'], 'nipy_spectral', 'Step 2: Labeled'),
                (res2['dilated'],     'gray',          f'Step 3: Dilated (r={dil_r2})'),
                (res2['edges'],       'hot',           'Step 4: Morph. Gradient'),
                (res2['skeleton'],    'gray',          'Step 5: Skeleton'),
            ]
            for ax, (img, cmap, title) in zip(axes2, steps2):
                plot_image(ax, img, title, cmap=cmap)
            for p in res2['pieces']:
                r, c = p['centroid']
                axes2[2].annotate(f"#{p['id']}", xy=(c, r), color='white',
                                  fontsize=8, fontweight='bold', ha='center')
            fig2.suptitle("Tetris Pieces — Morphological Pipeline", fontsize=12, fontweight='bold')
            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

            st.subheader("Hasil Deteksi")
            fig3, ax3 = plt.subplots(1, 2, figsize=(12, 5))
            plot_image(ax3[0], res2['original'], "Binary Original")
            ax3[1].imshow(res2['filtered_labeled'], cmap='nipy_spectral', interpolation='nearest')
            ax3[1].set_title(f"Detected: {res2['num_pieces']} Pieces", fontsize=11, fontweight='bold')
            ax3[1].axis('off')
            for p in res2['pieces']:
                r, c = p['centroid']
                ax3[1].annotate(
                    p['type'].split()[0], xy=(c, r), color='white', fontsize=8,
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5)
                )
            fig3.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)

            st.divider()
            st.subheader("Analisis Hasil")
            col_a2, col_b2 = st.columns(2)
            with col_a2:
                st.metric("Piece Terdeteksi", res2['num_pieces'])
                st.markdown(format_tetris_analysis(res2))
            with col_b2:
                if res2['pieces']:
                    types  = [p['type'] for p in res2['pieces']]
                    counts = {t: types.count(t) for t in set(types)}
                    fig_pie, ax_pie = plt.subplots(figsize=(5, 4))
                    ax_pie.pie(counts.values(), labels=counts.keys(), autopct='%1.0f%%', startangle=90)
                    ax_pie.set_title("Distribusi Tipe Piece")
                    st.pyplot(fig_pie)
                    plt.close(fig_pie)

    # ──────────────────────────────────────────
    # CHALLENGE 3: CARD DIAMONDS
    # ──────────────────────────────────────────
    with challenge[2]:
        st.header("Deteksi Diamond pada Gambar Kartu")
        st.markdown("""
        **Tujuan:** Deteksi bentuk **diamond** pada gambar kartu menggunakan morphological operators.  
        **Pipeline:** Erosion -> Dilation -> Morph. Gradient -> Distance Transform -> Labeling -> Shape Analysis (Solidity)
        """)
        st.divider()

        col_in3, col_par3 = st.columns([1, 1])

        with col_in3:
            st.subheader("Input")
            mode3 = st.radio("Sumber:", ["Default (Generated)", "Upload Sendiri"], horizontal=True, key="c3_mode")
            if mode3 == "Upload Sendiri":
                up3  = st.file_uploader("Upload gambar kartu:", type=["png","jpg","jpeg"], key="c3_up")
                raw3 = np.array(Image.open(up3).convert("L").resize((300,300))) if up3 else generate_card_diamonds(300)
                if not up3: st.info("Menggunakan gambar default.")
            else:
                raw3 = generate_card_diamonds(300)

            binary3, thresh3 = to_binary(raw3)
            st.caption(f"Otsu threshold: `{thresh3:.3f}`")

            fig0c, ax0c = plt.subplots(1, 2, figsize=(6, 3))
            plot_image(ax0c[0], raw3,    "Input (Grayscale)")
            plot_image(ax0c[1], binary3, "Binary (Otsu)")
            fig0c.tight_layout()
            st.pyplot(fig0c)
            plt.close(fig0c)

        with col_par3:
            st.subheader("Parameter")
            ero_r3   = st.slider("Erosion Radius (disk)", 1, 8,   2, key="c3_ero")
            min_area = st.slider("Min. Area (px)",        10, 300, 50, key="c3_area")
            st.markdown("### Pipeline Steps")
            st.markdown("""
| Step | Operasi | Tujuan |
|------|---------|--------|
| 1 | **Erosion (disk)** | Pisahkan shape |
| 2 | **Dilation** | Referensi gradient |
| 3 | **Morph. Gradient** | Tepi tiap shape |
| 4 | **Distance Transform** | Jarak ke batas |
| 5 | **Labeling** | ID tiap shape |
| 6 | **Shape Analysis** | Solidity -> Diamond? |

**Kriteria Diamond:** Solidity `0.35 - 0.65`, Aspect ratio `0.7 - 1.4`
            """)

        st.divider()
        if st.button("Jalankan Deteksi Diamond", key="c3_run", use_container_width=True):
            with st.spinner("Memproses..."):
                res3 = detect_card_diamonds(binary3, erosion_radius=ero_r3, min_area=min_area)

            st.subheader("Step-by-Step Visualisasi")
            fig4, axes4 = plt.subplots(2, 3, figsize=(15, 9))
            axes4 = axes4.flatten()
            steps3 = [
                (res3['original'],           'gray',          'Step 0: Binary Original'),
                (res3['eroded'],             'gray',          f'Step 1: Eroded (r={ero_r3})'),
                (res3['dilated'],            'gray',          'Step 2: Dilated'),
                (res3['edges'],              'hot',           'Step 3: Morph. Gradient'),
                (res3['distance_transform'], 'jet',           'Step 4: Distance Transform'),
                (res3['labeled'],            'nipy_spectral', 'Step 5: Labeled'),
            ]
            for ax, (img, cmap, title) in zip(axes4, steps3):
                plot_image(ax, img, title, cmap=cmap)
            fig4.suptitle("Card Diamonds — Morphological Pipeline", fontsize=12, fontweight='bold')
            fig4.tight_layout()
            st.pyplot(fig4)
            plt.close(fig4)

            st.subheader("Hasil Deteksi")
            fig5, axes5 = plt.subplots(1, 2, figsize=(12, 5))
            plot_image(axes5[0], res3['original'], "Binary Original")
            overlay3 = np.zeros((*binary3.shape, 3), dtype=float)
            for shape in res3['shapes']:
                region = (res3['labeled'] == shape['id'])
                if shape['type'] == "Diamond":
                    overlay3[region] = [0.0, 0.85, 0.3]
                else:
                    overlay3[region] = [0.85, 0.2, 0.0]
            axes5[1].imshow(overlay3)
            axes5[1].set_title(
                f"Diamond: {res3['num_diamonds']}  |  Lainnya: {len(res3['shapes']) - res3['num_diamonds']}",
                fontsize=11, fontweight='bold'
            )
            axes5[1].axis('off')
            for shape in res3['shapes']:
                r, c = shape['centroid']
                axes5[1].annotate(
                    "D" if shape['type'] == "Diamond" else "?",
                    xy=(c, r), color='white', fontsize=12,
                    ha='center', va='center', fontweight='bold'
                )
            green_patch = mpatches.Patch(color=[0.0, 0.85, 0.3], label='Diamond')
            red_patch   = mpatches.Patch(color=[0.85, 0.2, 0.0], label='Other Shape')
            axes5[1].legend(handles=[green_patch, red_patch], loc='lower right', fontsize=9)
            fig5.tight_layout()
            st.pyplot(fig5)
            plt.close(fig5)

            st.divider()
            st.subheader("Analisis Hasil")
            col_a3, col_b3 = st.columns(2)
            with col_a3:
                st.metric("Diamond Terdeteksi", res3['num_diamonds'])
                st.metric("Total Shape",        len(res3['shapes']))
                st.markdown(format_diamond_analysis(res3))
            with col_b3:
                if res3['shapes']:
                    solidities = [s['solidity'] for s in res3['shapes']]
                    labels_s   = [f"#{s['id']}" for s in res3['shapes']]
                    colors_s   = ['#21C354' if s['type'] == "Diamond" else '#FF4B4B' for s in res3['shapes']]
                    fig_b3, ax_b3 = plt.subplots(figsize=(5, 3))
                    ax_b3.bar(labels_s, solidities, color=colors_s)
                    ax_b3.axhline(0.35, color='gray', linestyle='--', linewidth=1, label='Min (0.35)')
                    ax_b3.axhline(0.65, color='gray', linestyle=':',  linewidth=1, label='Max (0.65)')
                    ax_b3.set_ylabel("Solidity")
                    ax_b3.set_title("Solidity Tiap Shape\n(Hijau = Diamond)")
                    ax_b3.legend(fontsize=7)
                    ax_b3.set_ylim(0, 1)
                    st.pyplot(fig_b3)
                    plt.close(fig_b3)