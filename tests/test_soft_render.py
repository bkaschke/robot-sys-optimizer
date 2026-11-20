"""core.soft_render: the offline z-buffer renderer behind the batch review pictures."""

import io
import time

import numpy as np
import pytest
import trimesh
from PIL import Image

from core import soft_render as sr
from core.golden_parts import flat_plate


def _img(png: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), int)


def _plate():
    return flat_plate()  # 1 x 0.6 x 0.02 m, z up


def test_png_size_and_background():
    png = sr.render(_plate(), size=(200, 150))
    im = Image.open(io.BytesIO(png))
    assert im.format == "PNG" and im.size == (200, 150)
    a = _img(png)
    assert (a[0, 0] == 255).all()  # background corner
    assert (a != 255).any()


def test_highlighted_half_is_accent_from_top():
    m = _plate()
    top = m.triangles_center[:, 2] > 0
    left = m.triangles_center[:, 0] < 0
    png = sr.render(m, highlight=top & left, view="top", size=(300, 200))
    a = _img(png)
    h, w = a.shape[:2]
    lft, rgt = a[h // 2, w // 4], a[h // 2, 3 * w // 4]
    assert lft[0] > 150 and lft[1] < 120 and lft[2] > 120  # magenta-ish
    assert abs(rgt[0] - rgt[2]) < 40 and rgt[1] > 100  # neutral grey
    # ids work like a mask, and the second colour differs from the accent
    same = sr.render(m, highlight=np.flatnonzero(top & left), view="top", size=(300, 200))
    assert same == png
    sec = _img(sr.render(m, secondary=top & left, view="top", size=(300, 200)))
    assert sec[h // 2, w // 4][2] < 100  # orange has little blue


def test_deterministic_and_views_differ():
    m = trimesh.creation.icosphere(subdivisions=3)
    a = sr.render(m, highlight=m.triangles_center[:, 2] > 0.3)
    assert a == sr.render(m, highlight=m.triangles_center[:, 2] > 0.3)
    box = trimesh.creation.box((1, 0.4, 0.2))
    assert sr.render(box, view="front") != sr.render(box, view="top")


def test_back_faces_are_drawn_darker():
    m = _plate()
    flipped = trimesh.Trimesh(m.vertices, m.faces[:, ::-1], process=False)
    a, b = _img(sr.render(m, view="top")), _img(sr.render(flipped, view="top"))
    assert (b != 255).any()  # double sided: still drawn
    inside = (a != 255).any(2)
    assert b[inside].mean() < 0.8 * a[inside].mean()  # the turned-away side is darker


def test_bad_input_says_what_is_wrong():
    with pytest.raises(ValueError, match="view must be"):
        sr.render(_plate(), view="back")
    with pytest.raises(ValueError, match="highlight mask has"):
        sr.render(_plate(), highlight=np.zeros(3, bool))


def test_100k_faces_in_a_few_seconds():
    m = trimesh.creation.icosphere(subdivisions=6)  # 81920 faces
    t = time.perf_counter()
    png = sr.render(m, highlight=m.triangles_center[:, 2] > 0)
    assert time.perf_counter() - t < 8.0
    assert _img(png).shape == (360, 480, 3)
