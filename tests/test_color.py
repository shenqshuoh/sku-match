import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.color import color_descriptor_dim, extract_color_descriptor
from src.color_store import ColorStore


def test_descriptor_shape_and_norm():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    desc = extract_color_descriptor(image)
    assert desc.shape == (49,)
    assert desc.dtype == np.float32
    assert abs(float(np.linalg.norm(desc)) - 1.0) < 1e-4


def test_descriptor_custom_bins():
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    desc = extract_color_descriptor(image, n_bins=8)
    assert desc.shape == (25,)


def test_descriptor_deterministic():
    rng = np.random.default_rng(2)
    image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    a = extract_color_descriptor(image)
    b = extract_color_descriptor(image)
    np.testing.assert_array_equal(a, b)


def test_descriptor_rejects_bad_input():
    grayscale = np.zeros((64, 64), dtype=np.uint8)
    try:
        extract_color_descriptor(grayscale)
    except ValueError:
        pass
    else:
        raise AssertionError("grayscale image should raise ValueError")

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    try:
        extract_color_descriptor(image, n_bins=0)
    except ValueError:
        pass
    else:
        raise AssertionError("n_bins=0 should raise ValueError")


def test_descriptor_small_image():
    rng = np.random.default_rng(3)
    image = rng.integers(0, 256, size=(4, 4, 3), dtype=np.uint8)
    desc = extract_color_descriptor(image)
    assert desc.shape == (49,)


def test_color_store_roundtrip(tmp_path):
    store = ColorStore(tmp_path)
    rng = np.random.default_rng(4)
    descriptor = rng.standard_normal(49).astype(np.float32)
    descriptor /= np.linalg.norm(descriptor) + 1e-8

    path = store.save("sku1__m1", descriptor)
    assert path.exists()
    loaded = store.load("sku1__m1")
    np.testing.assert_allclose(loaded, descriptor.astype(np.float32))

    store.delete("sku1__m1")
    assert not path.exists()


def test_color_store_load_batch_skip_missing(tmp_path):
    store = ColorStore(tmp_path)
    rng = np.random.default_rng(5)
    d1 = rng.standard_normal(49).astype(np.float32)
    d2 = rng.standard_normal(49).astype(np.float32)

    store.save("sku1__a", d1)
    store.save("sku1__b", d2)

    result = store.load_batch(["sku1__a", "sku1__b", "sku1__missing"])
    assert set(result.keys()) == {"sku1__a", "sku1__b"}


def test_color_store_delete_sku(tmp_path):
    store = ColorStore(tmp_path)
    rng = np.random.default_rng(6)
    d = rng.standard_normal(49).astype(np.float32)

    p1 = store.save("sku1__a", d)
    p2 = store.save("sku2__b", d)

    deleted = store.delete_sku("sku1")
    assert deleted == 1
    assert not p1.exists()
    assert p2.exists()


def test_color_descriptor_dim_matches_extractor():
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    for bins in (1, 8, 16, 32):
        desc = extract_color_descriptor(image, n_bins=bins)
        assert color_descriptor_dim(bins) == desc.shape[0]
        assert color_descriptor_dim(bins) == 3 * bins + 1


def test_color_descriptor_dim_rejects_zero():
    try:
        color_descriptor_dim(0)
    except ValueError:
        pass
    else:
        raise AssertionError("n_bins=0 should raise ValueError")


if __name__ == "__main__":
    # Allow running without pytest for smoke checks.
    import tempfile

    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            params = fn.__code__.co_varnames[: fn.__code__.co_argcount]
            if "tmp_path" in params:
                with tempfile.TemporaryDirectory() as td:
                    fn(Path(td))
            else:
                fn()
    print("All color tests passed!")
