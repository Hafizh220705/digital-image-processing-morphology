import numpy as np
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage import morphology as morph
from scipy.ndimage import binary_fill_holes, label as scipy_label, distance_transform_edt

def to_binary(image, threshold=None):
    """Konversi image ke binary. Jika threshold=None pakai Otsu."""
    if image.ndim == 3:
        image = rgb2gray(image)
    if image.max() > 1.0:
        image = image.astype(np.float64) / 255.0
    if threshold is None:
        threshold = threshold_otsu(image)
    binary = (image > threshold).astype(np.uint8)
    return binary, float(threshold)

def apply_erosion(binary_img, radius=3):
    """
    Erosion: susutkan foreground (hapus piksel di batas objek).
    Efek: objek mengecil, noise kecil hilang.
    """
    fp = morph.disk(radius)
    result = morph.erosion(binary_img, footprint=fp).astype(np.uint8)
    return result

def apply_dilation(binary_img, radius=3):
    """
    Dilation: perluas foreground (tambah piksel di batas objek).
    Efek: objek membesar, celah kecil tertutup.
    """
    fp = morph.disk(radius)
    result = morph.dilation(binary_img, footprint=fp).astype(np.uint8)
    return result

def apply_opening(binary_img, radius=3):
    """
    Opening = Erosion → Dilation.
    Efek: hilangkan noise kecil, pisahkan objek yang terhubung tipis.
    """
    fp = morph.disk(radius)
    result = morph.opening(binary_img, footprint=fp).astype(np.uint8)
    return result

def apply_closing(binary_img, radius=3):
    """
    Closing = Dilation → Erosion.
    Efek: tutup lubang kecil di dalam objek, sambungkan celah kecil.
    """
    fp = morph.disk(radius)
    result = morph.closing(binary_img, footprint=fp).astype(np.uint8)
    return result

def apply_morph_gradient(binary_img, radius=2):
    """
    Morphological Gradient = Dilation − Erosion.
    Efek: deteksi tepi / edge dari objek.
    """
    fp = morph.disk(radius)
    dilated = morph.dilation(binary_img, footprint=fp).astype(int)
    eroded  = morph.erosion(binary_img,  footprint=fp).astype(int)
    result  = (dilated - eroded).clip(0, 1).astype(np.uint8)
    return result

def apply_skeletonize(binary_img):
    """
    Skeletonize: tipiskan objek jadi 1 piksel lebar (kerangka/tulang).
    Menggunakan morphological thinning.
    """
    result = morph.skeletonize(binary_img.astype(bool)).astype(np.uint8)
    return result

def detect_fence_holes(binary_img, erosion_iter=5, min_hole_size=100):
    """Challenge 1: Deteksi lubang pada chain-link fence."""
    results = {'original': binary_img.copy()}

    se = np.ones((3, 3))
    from scipy.ndimage import binary_erosion as scipy_erosion
    eroded = scipy_erosion(binary_img, structure=se, iterations=erosion_iter).astype(np.uint8)
    results['eroded'] = eroded

    filled = binary_fill_holes(eroded).astype(np.uint8)
    results['filled'] = filled

    holes_mask = (filled - eroded).clip(0, 1).astype(np.uint8)
    results['holes_mask'] = holes_mask

    closed = morph.closing(holes_mask, footprint=morph.disk(5)).astype(np.uint8)
    results['closed'] = closed

    labeled, num_features = scipy_label(closed)
    results['labeled']  = labeled
    results['num_raw']  = num_features

    filtered_labeled = np.zeros_like(labeled)
    hole_data        = []
    valid_count      = 0

    for region_id in range(1, num_features + 1):
        region_mask = (labeled == region_id)
        region_size = int(region_mask.sum())
        if region_size < min_hole_size:
            continue
        valid_count += 1
        filtered_labeled[region_mask] = valid_count
        rows, cols = np.where(region_mask)
        hole_data.append({
            'id':          valid_count,
            'size_pixels': region_size,
            'centroid':    (int(rows.mean()), int(cols.mean())),
            'bbox':        (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())),
        })

    results['filtered_labeled'] = filtered_labeled
    results['holes']            = hole_data
    results['num_holes']        = valid_count
    return results

def detect_tetris_pieces(binary_img, erosion_radius=3, dilation_radius=4):
    """Challenge 2: Deteksi & klasifikasi piece pada gambar Tetris."""
    results = {'original': binary_img.copy()}

    eroded = morph.erosion(binary_img, footprint=morph.disk(erosion_radius)).astype(np.uint8)
    results['eroded'] = eroded

    labeled_raw = morph.label(eroded, connectivity=2)
    results['labeled_raw'] = labeled_raw

    dilated = morph.dilation(binary_img, footprint=morph.disk(dilation_radius)).astype(np.uint8)
    results['dilated'] = dilated

    eroded2 = morph.erosion(binary_img, footprint=morph.disk(2)).astype(np.uint8)
    edges   = (dilated.astype(int) - eroded2.astype(int)).clip(0, 1).astype(np.uint8)
    results['edges'] = edges

    skeleton = morph.skeletonize(binary_img.astype(bool)).astype(np.uint8)
    results['skeleton'] = skeleton

    pieces    = []
    num_labels = labeled_raw.max()

    for lbl in range(1, num_labels + 1):
        region = (labeled_raw == lbl)
        area   = int(region.sum())
        if area < 20:
            continue
        rows, cols   = np.where(region)
        height       = int(rows.max() - rows.min() + 1)
        width        = int(cols.max() - cols.min() + 1)
        aspect_ratio = width / height if height > 0 else 0

        if   aspect_ratio > 3.0:            piece_type = "I-piece (Horizontal)"
        elif aspect_ratio < 0.4:            piece_type = "I-piece (Vertical)"
        elif 0.8 <= aspect_ratio <= 1.2:    piece_type = "O-piece (Square)"
        elif aspect_ratio > 1.5:            piece_type = "L/J-piece"
        else:                               piece_type = "T/S/Z-piece"

        pieces.append({
            'id':           lbl,
            'area':         area,
            'height':       height,
            'width':        width,
            'aspect_ratio': round(aspect_ratio, 2),
            'centroid':     (int(rows.mean()), int(cols.mean())),
            'type':         piece_type,
        })

    filtered_labeled = np.zeros_like(labeled_raw)
    for i, p in enumerate(pieces, 1):
        filtered_labeled[labeled_raw == p['id']] = i

    results['pieces']           = pieces
    results['num_pieces']       = len(pieces)
    results['filtered_labeled'] = filtered_labeled
    return results

def detect_card_diamonds(binary_img, erosion_radius=2, min_area=50):
    """Challenge 3: Deteksi bentuk diamond pada gambar kartu."""
    results = {'original': binary_img.copy()}

    eroded  = morph.erosion(binary_img,  footprint=morph.disk(erosion_radius)).astype(np.uint8)
    dilated = morph.dilation(binary_img, footprint=morph.disk(2)).astype(np.uint8)
    results['eroded']  = eroded
    results['dilated'] = dilated

    edges = (dilated.astype(int) - eroded.astype(int)).clip(0, 1).astype(np.uint8)
    results['edges'] = edges

    dt = distance_transform_edt(binary_img.astype(bool))
    results['distance_transform'] = dt

    labeled, num_features = scipy_label(binary_img)
    results['labeled']  = labeled
    results['num_raw']  = num_features

    shapes = []
    for region_id in range(1, num_features + 1):
        region   = (labeled == region_id)
        area     = int(region.sum())
        if area < min_area:
            continue
        rows, cols   = np.where(region)
        height       = int(rows.max() - rows.min() + 1)
        width        = int(cols.max() - cols.min() + 1)
        bbox_area    = height * width
        solidity     = area / bbox_area if bbox_area > 0 else 0
        aspect_ratio = width / height   if height  > 0 else 0
        is_diamond   = (0.35 <= solidity <= 0.65) and (0.7 <= aspect_ratio <= 1.4)

        shapes.append({
            'id':           region_id,
            'area':         area,
            'height':       height,
            'width':        width,
            'bbox_area':    bbox_area,
            'solidity':     round(solidity, 3),
            'aspect_ratio': round(aspect_ratio, 2),
            'centroid':     (int(rows.mean()), int(cols.mean())),
            'type':         "Diamond ♦" if is_diamond else "Other Shape",
        })

    results['shapes']        = shapes
    results['diamonds']      = [d for d in shapes if d['type'] == "Diamond ♦"]
    results['num_diamonds']  = len(results['diamonds'])
    return results