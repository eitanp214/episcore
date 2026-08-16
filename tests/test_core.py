"""Tests for the measurement core, against inputs with known answers.

The point of these is not coverage. It is that the redundancy metric makes a
quantitative claim about someone's data, so its properties should be pinned
down rather than assumed: identical content must register, unrelated content
must not, the first frame can never be redundant, and the curve must fall
monotonically with the threshold.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from dedup import measure, summarise  # noqa: E402
from frames import laplacian_variance  # noqa: E402
from local_source import discover  # noqa: E402
from score import SATURATION_CEILING, saturation  # noqa: E402


def unit(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise rows, matching what Embedder emits."""
    v = np.asarray(vectors, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# --- measure() -----------------------------------------------------------

def test_first_frame_is_never_redundant():
    """Nothing precedes frame zero, so it cannot duplicate earlier content."""
    result = measure(unit(np.eye(4)))
    assert result.match_idx[0] == -1
    assert result.max_sim[0] == pytest.approx(-1.0)


def test_exact_duplicate_is_detected():
    base = unit(np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]]))
    result = measure(base)
    assert result.max_sim[2] == pytest.approx(1.0, abs=1e-4)
    assert result.match_idx[2] == 0


def test_orthogonal_frames_are_not_redundant():
    result = measure(unit(np.eye(6)))
    assert np.all(result.max_sim[1:] < 0.5)
    assert result.curve[0.80] == pytest.approx(0.0)


def test_only_earlier_frames_count():
    """A frame must not match something that comes after it."""
    vectors = unit(np.array([[1, 0], [0, 1], [1, 0]]))
    result = measure(vectors)
    # frame 0 duplicates frame 2, but 2 is later, so 0 stays unmatched
    assert result.match_idx[0] == -1
    assert result.match_idx[2] == 0


def test_curve_is_monotonically_non_increasing():
    rng = np.random.default_rng(0)
    result = measure(unit(rng.normal(size=(200, 16))))
    values = [result.curve[t] for t in sorted(result.curve)]
    assert all(a >= b - 1e-12 for a, b in zip(values, values[1:]))


def test_block_boundary_is_handled():
    """Matches must be found across the internal 4096-frame block edge."""
    from dedup import BLOCK

    n = BLOCK + 10
    rng = np.random.default_rng(1)
    vectors = unit(rng.normal(size=(n, 8)))
    vectors[-1] = vectors[0]  # duplicate of a frame in the previous block
    result = measure(vectors)
    assert result.max_sim[-1] == pytest.approx(1.0, abs=1e-4)
    assert result.match_idx[-1] == 0


def test_summarise_weights_by_frame_count():
    small = measure(unit(np.array([[1, 0], [1, 0]])))      # 2 frames, redundant
    large = measure(unit(np.eye(2).repeat(20, axis=0)))    # 40 frames, mostly redundant
    blended = summarise({"small": small, "large": large})
    assert 0.0 <= blended[0.92] <= 1.0


# --- sharpness -----------------------------------------------------------

def test_blur_lowers_sharpness_score():
    rng = np.random.default_rng(2)
    noise = (rng.random((160, 160, 3)) * 255).astype("uint8")
    sharp = Image.fromarray(noise)
    blurred = sharp.filter(ImageFilter.GaussianBlur(radius=4))
    assert laplacian_variance(sharp) > laplacian_variance(blurred) * 3


def test_flat_image_has_near_zero_sharpness():
    flat = Image.new("RGB", (160, 160), (128, 128, 128))
    assert laplacian_variance(flat) < 1.0


# --- saturation gate -----------------------------------------------------
#
# This gate exists because of a real failure. An end-to-end run on synthetic
# footage reported 97.4% corpus redundancy and ranked a deliberately varied
# operator as the MOST repetitive one. The separability check passed it at
# 2.5x chance, because operators were still coarsely distinguishable — but
# unrelated frames were already scoring 0.904 against a 0.92 threshold, so
# "duplicate" had stopped meaning anything. Measured real footage sits at a
# 0.647 median with 0.03% of random pairs above threshold.

def test_saturated_embedding_is_caught():
    """Nearly-parallel vectors: everything looks like everything."""
    rng = np.random.default_rng(3)
    base = np.array([1.0, 0.0, 0.0, 0.0])
    vectors = unit(base + rng.normal(scale=0.05, size=(400, 4)))
    result = saturation(vectors, 0.92)
    assert result["share"] > SATURATION_CEILING
    assert result["median"] > 0.9


def test_well_spread_embedding_passes():
    """Isotropic vectors in high dimension are near-orthogonal by default."""
    rng = np.random.default_rng(4)
    vectors = unit(rng.normal(size=(400, 128)))
    result = saturation(vectors, 0.92)
    assert result["share"] <= SATURATION_CEILING
    assert result["median"] < 0.5


def test_saturation_handles_tiny_input():
    """Too few frames to sample pairs must not raise."""
    result = saturation(unit(np.eye(3)), 0.92)
    assert result["pairs"] == 0


# --- local_source.discover() --------------------------------------------

def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00")


def test_subdirectories_become_operators(tmp_path):
    _touch(tmp_path / "alice" / "a1.mp4")
    _touch(tmp_path / "alice" / "a2.mp4")
    _touch(tmp_path / "bob" / "b1.mp4")
    units = discover(tmp_path)
    assert [u.worker for u in units] == ["alice", "bob"]
    assert len(units[0].paths) == 2


def test_flat_directory_falls_back_to_one_bucket(tmp_path):
    _touch(tmp_path / "x.mp4")
    _touch(tmp_path / "y.mov")
    units = discover(tmp_path)
    assert len(units) == 1
    assert units[0].worker == "all"
    assert len(units[0].paths) == 2


def test_pattern_grouping_overrides_directories(tmp_path):
    _touch(tmp_path / "d1" / "op7_clip1.mp4")
    _touch(tmp_path / "d2" / "op7_clip2.mp4")
    _touch(tmp_path / "d3" / "op9_clip1.mp4")
    units = discover(tmp_path, pattern=r"(?P<operator>op\d+)")
    assert {u.worker for u in units} == {"op7", "op9"}
    assert len(next(u for u in units if u.worker == "op7").paths) == 2


def test_non_video_files_are_ignored(tmp_path):
    _touch(tmp_path / "a" / "clip.mp4")
    _touch(tmp_path / "a" / "notes.txt")
    _touch(tmp_path / "b" / "clip.mp4")
    units = discover(tmp_path)
    assert sum(len(u.paths) for u in units) == 2


def test_missing_directory_raises(tmp_path):
    with pytest.raises(NotADirectoryError):
        discover(tmp_path / "nope")


def test_clip_cap_is_applied(tmp_path):
    for i in range(5):
        _touch(tmp_path / "op" / f"c{i}.mp4")
    _touch(tmp_path / "other" / "c.mp4")
    units = discover(tmp_path, clips_per_operator=2)
    assert max(len(u.paths) for u in units) == 2
