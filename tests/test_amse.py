"""Unit checks for the V2-E4 AMSE training loss (spec V2-E4):  python -m pytest tests -q  (or python tests/test_amse.py)"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.train_nn import AMSE, amse_loss_fn, dct_matrix, loss_fn  # noqa: E402
from twin.config import LAT, NLAT, NLON, TP                            # noqa: E402

LAT_W = torch.tensor(np.cos(np.deg2rad(LAT)) / np.cos(np.deg2rad(LAT)).mean(), dtype=torch.float32)[:, None]
CH_W = torch.ones(6)


def test_dct_orthonormal():
    for n in (NLAT, NLON):
        c = dct_matrix(n)
        assert torch.allclose(c @ c.T, torch.eye(n), atol=1e-5)


def test_bands_cover_every_coefficient_once():
    a = AMSE(LAT_W)
    assert torch.equal(a.bands.sum(1), torch.ones(NLAT * NLON))


def test_zero_for_identical_maps():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 6, NLAT, NLON)
    assert amse_loss_fn(x, x.clone(), AMSE(LAT_W), LAT_W, CH_W).item() < 1e-4


def test_mse_scale_and_upper_bound():
    torch.manual_seed(0)
    a = AMSE(LAT_W)
    t = torch.randn(2, 3, 6, NLAT, NLON)
    p = t + 0.5 * torch.randn_like(t)
    mse = loss_fn(p, t, LAT_W, CH_W).item()
    am = amse_loss_fn(p, t, a, LAT_W, CH_W).item()
    assert mse <= am + 1e-4 and am < 2.0 * mse                       # AMSE >= MSE, same order of magnitude
    # pure amplitude error (same pattern, half amplitude): AMSE = MSE exactly (coherence 1 in every band)
    assert abs(amse_loss_fn(0.5 * t, t, a, LAT_W, CH_W).item() - loss_fn(0.5 * t, t, LAT_W, CH_W).item()) < 1e-3
    # a blurred (damped) forecast with partial coherence is penalised more than by MSE
    blur = 0.0 * t
    assert amse_loss_fn(blur, t, a, LAT_W, CH_W).item() >= loss_fn(blur, t, LAT_W, CH_W).item() - 1e-4


def test_heavy_rain_term_and_gradients():
    torch.manual_seed(0)
    a = AMSE(LAT_W)
    t = torch.randn(2, 2, 6, NLAT, NLON)
    p = (t + 0.3 * torch.randn_like(t)).requires_grad_(True)
    heavy = torch.rand(2, 2, NLAT, NLON) > 0.8
    l1 = amse_loss_fn(p, t, a, LAT_W, CH_W)
    l3 = amse_loss_fn(p, t, a, LAT_W, CH_W, heavy, 3.0)
    extra = 2.0 * ((p[:, :, TP] - t[:, :, TP]) ** 2 * LAT_W * heavy).mean() / 6
    assert abs((l3 - l1).item() - extra.item()) < 1e-5
    l3.backward()
    assert torch.isfinite(p.grad).all()


if __name__ == "__main__":
    for _name, _f in list(globals().items()):
        if _name.startswith("test_"):
            _f()
    print("ok")
