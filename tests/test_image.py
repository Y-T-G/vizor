import numpy as np
import pytest

from vizor.utils.image import crop, fit, montage


def block(w, h, value=200):
    return np.full((h, w, 3), value, np.uint8)


def test_fit_keeps_the_shape_and_pads_the_rest():
    out = fit(block(50, 200), 100, 100, color=(0, 0, 0))
    assert out.shape == (100, 100, 3)
    # a 1:4 crop in a square cell fills a quarter of the width, the rest is padding
    filled = (out != 0).any(axis=2)
    assert 20 <= filled.sum(axis=1).max() <= 30
    assert filled.sum(axis=0).max() == 100


def test_fit_grows_a_small_crop():
    assert fit(block(4, 4), 64, 64).shape == (64, 64, 3)


def test_fit_takes_a_grayscale_crop():
    assert fit(np.full((10, 10), 90, np.uint8), 32, 32).shape == (32, 32, 3)


def test_fit_rejects_an_empty_crop():
    with pytest.raises(ValueError):
        fit(np.zeros((0, 10, 3), np.uint8), 32, 32)


def test_montage_lays_out_a_square_grid():
    # four cells of 128 in a 2 x 2, with a 4px gutter around and between them
    assert montage([block(40, 80)] * 4).shape == (2 * 128 + 3 * 4, 2 * 128 + 3 * 4, 3)


def test_montage_takes_a_rectangular_cell_and_a_column_count():
    out = montage([block(40, 80)] * 3, cols=3, cell=(60, 120), pad=2)
    assert out.shape == (1 * 120 + 2 * 2, 3 * 60 + 4 * 2, 3)


def test_montage_pads_the_last_row_when_the_grid_is_not_full():
    # five images in a 3 wide grid means two rows, the second one short
    assert montage([block(40, 80)] * 5, cell=32, pad=0).shape == (2 * 32, 3 * 32, 3)


def test_montage_drops_empty_crops():
    imgs = [block(40, 80), np.zeros((0, 0, 3), np.uint8), block(40, 80)]
    assert montage(imgs, cols=2, cell=32, pad=0).shape == (32, 64, 3)


def test_montage_needs_something_to_tile():
    with pytest.raises(ValueError):
        montage([np.zeros((0, 0, 3), np.uint8)])


def test_montage_keeps_the_crops_apart():
    # the gutter colour has to survive, or the model sees one image not four
    out = montage([block(40, 80, 255)] * 4, cell=32, pad=6, color=(0, 0, 0))
    assert out[0, 0].tolist() == [0, 0, 0]


def test_montage_of_real_crops():
    frame = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    tiles = [crop(frame, [x, 20, x + 40, 120]) for x in (0, 60, 120, 180)]
    assert montage(tiles, cell=64, pad=0).shape == (128, 128, 3)
