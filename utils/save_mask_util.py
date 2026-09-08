import numpy as np


def get_v2_pallete(num_cls=71):
    pallete = [0] * (num_cls * 3)
    for j in range(0, num_cls):
        lab = j
        pallete[j * 3 + 0] = 0
        pallete[j * 3 + 1] = 0
        pallete[j * 3 + 2] = 0
        i = 0
        while lab > 0:
            pallete[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            pallete[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            pallete[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i = i + 1
            lab >>= 3
    v2_pallete = np.array(pallete).reshape(-1, 3)
    return v2_pallete


def colored_masks(mask):
    pallete = get_v2_pallete()
    valid_num, H, W = mask.shape
    rgb_mask = np.zeros((valid_num, H, W, 3), dtype=np.uint8)

    for cls_idx in range(len(pallete)):
        rgb = pallete[cls_idx]
        rgb_mask[mask == cls_idx] = rgb
    return rgb_mask
