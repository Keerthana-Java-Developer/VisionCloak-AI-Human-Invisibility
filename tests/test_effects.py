import numpy as np

from visioncloak.effects import apply_invisibility_effect


def test_mask_replaces_only_human_region_with_background():
    live = np.zeros((4, 4, 3), dtype=np.uint8)
    background = np.full((4, 4, 3), 200, dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.float32)
    mask[1:3, 1:3] = 1.0

    result = apply_invisibility_effect(live, background, mask)

    assert np.all(result[1:3, 1:3] == 200)
    assert np.all(result[0, 0] == 0)
