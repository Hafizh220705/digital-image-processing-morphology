import numpy as np
from skimage import morphology as morph
from skimage.draw import rectangle

def generate_chainlink_fence(size=300):
    """Gambar simulasi chain-link fence dengan 2 lubang."""
    img = np.zeros((size, size), dtype=np.uint8)

    for i in range(0, size, 12):
        for j in range(size):
            x = i + (j % 12)
            if 0 <= x < size:
                img[j, x] = 1
            x2 = i - (j % 12)
            if 0 <= x2 < size:
                img[j, x2] = 1

    img = morph.dilation(img, footprint=np.ones((2, 2))).astype(np.uint8)

    rr, cc = rectangle(start=(60, 50), end=(110, 100), shape=img.shape)
    img[rr, cc] = 0

    rr, cc = rectangle(start=(180, 190), end=(230, 240), shape=img.shape)
    img[rr, cc] = 0

    return img


def generate_tetris_board(size=300):
    """Gambar simulasi papan Tetris dengan beberapa piece."""
    img = np.zeros((size, size), dtype=np.uint8)
    cell = 30

    for col in range(1, 5):
        rr, cc = rectangle(
            start=(20, col * cell + 5),
            end=(20 + cell - 2, (col + 1) * cell + 3),
            shape=img.shape
        )
        img[rr, cc] = 1

    for dr in range(2):
        for dc in range(2):
            rr, cc = rectangle(
                start=(80 + dr * cell, 40 + dc * cell),
                end=(80 + dr * cell + cell - 3, 40 + dc * cell + cell - 3),
                shape=img.shape
            )
            img[rr, cc] = 1

    for col in range(3):
        rr, cc = rectangle(
            start=(160, 100 + col * cell),
            end=(160 + cell - 3, 100 + col * cell + cell - 3),
            shape=img.shape
        )
        img[rr, cc] = 1
    rr, cc = rectangle(
        start=(160 + cell, 100 + cell),
        end=(160 + 2 * cell - 3, 100 + 2 * cell - 3),
        shape=img.shape
    )
    img[rr, cc] = 1

    for col in range(2):
        rr, cc = rectangle(
            start=(230, 150 + col * cell),
            end=(230 + cell - 3, 150 + (col + 1) * cell - 3),
            shape=img.shape
        )
        img[rr, cc] = 1
    for col in range(2):
        rr, cc = rectangle(
            start=(230 + cell, 150 - cell + col * cell),
            end=(230 + 2 * cell - 3, 150 - cell + (col + 1) * cell - 3),
            shape=img.shape
        )
        img[rr, cc] = 1

    return img

def generate_card_diamonds(size=300):
    """Gambar simulasi kartu Diamond."""
    img = np.zeros((size, size), dtype=np.uint8)

    def draw_diamond(center_r, center_c, radius, image):
        for r in range(image.shape[0]):
            for c in range(image.shape[1]):
                if abs(r - center_r) + abs(c - center_c) <= radius:
                    image[r, c] = 1
        return image

    positions = [
        (70,  70,  25),
        (70,  230, 25),
        (230, 70,  25),
        (230, 230, 25),
    ]
    for (r, c, rad) in positions:
        img = draw_diamond(r, c, rad, img)
    img = draw_diamond(150, 150, 18, img)

    return img