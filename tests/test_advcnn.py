"""Unit tests for the v2 E7 advect-and-correct family (spec V2-E7):  python tests/test_advcnn.py

1. A Gaussian blob in a constant velocity field moves by the expected number of cells (bilinear and bicubic,
   2 substeps per day, the same `advect` the network uses).
2. The displacement cap: total displacement per day < 3 cells even for huge raw velocities.
3. With zero velocity the net equals the v1 CNN with the same weights (warm start starts at r3_cnn exactly).
4. Reach budget (spec V2-E7b, V2-E7c): with random weights and large velocities, a 1-cell input change moves nothing
   beyond reach_bound() (Euclidean cells); E7b's bound is <= 10, E7c's <= 13.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from twin.nets import build_net                       # noqa: E402
from twin.nets.advcnn import advect, squash, warm_start  # noqa: E402

H, W = 32, 36


def _blob(cy, cx, sig=2.0):
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    return torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sig ** 2))


def _centroid(f):
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    return float((f * yy).sum() / f.sum()), float((f * xx).sum() / f.sum())


def test_blob_moves():
    for mode in ("bilinear", "bicubic"):
        f = _blob(12.0, 14.0)[None, None, None]                       # [B=1,G=1,C=1,H,W]
        d = torch.zeros(1, 1, 2, H, W); d[:, :, 0] = 1.5; d[:, :, 1] = -1.0   # cells per substep (dy, dx)
        for _ in range(2):                                            # 2 substeps = 1 day
            f = advect(f, d, mode)
        cy, cx = _centroid(f[0, 0, 0])
        assert abs(cy - 15.0) < 0.05 and abs(cx - 12.0) < 0.05, (mode, cy, cx)
        for _ in range(6):                                            # 3 more days: 12 + 4*3 = 24, 14 - 4*2 = 6
            f = advect(f, d, mode)
        cy, cx = _centroid(f[0, 0, 0])
        assert abs(cy - 24.0) < 0.1 and abs(cx - 6.0) < 0.1, (mode, cy, cx)


def test_cap():
    r = torch.randn(4, 8, 2, H, W) * 100
    d = squash(r, 1.5)
    assert float(d.norm(dim=2).max()) < 1.5 + 1e-5
    assert float(squash(torch.zeros(1, 1, 2, 2, 2), 1.5).abs().max()) == 0.0


def test_zero_velocity_is_cnn():
    torch.manual_seed(0)
    cnn = build_net("cnn", 29, 7).eval()
    torch.nn.init.normal_(cnn.out.weight, std=0.1)
    adv = build_net("advcnn", 29, 7).eval()
    reused, _ = warm_start(adv, cnn.state_dict())
    assert len(reused) == len(cnn.state_dict())
    x = torch.randn(2, 29, H, W)
    with torch.no_grad():
        a, c = adv(x), cnn(x)
    assert float((a - c).abs().max()) < 1e-5
    assert sum(p.numel() for p in adv.parameters()) < 500_000


def test_reach_budget():
    for block_k, budget in (([3, 1, 1], 10.0), ([3, 3, 1], 13.0)):          # E7b (V2-E7b), E7c (V2-E7c)
        _check_reach({"max_disp": 2.5, "interp": "bilinear", "block_k": block_k}, budget)


def _check_reach(cfg, budget):
    torch.manual_seed(1)
    net = build_net("advcnn", 29, 7, **cfg).eval()
    assert net.reach_bound() <= budget
    for f in net.flow:                                                # big random velocities (hit the cap)
        torch.nn.init.normal_(f[-1].weight, std=2.0); torch.nn.init.normal_(f[-1].bias, std=2.0)
    torch.nn.init.normal_(net.out.weight, std=0.1)
    ii, jj = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    dist = torch.sqrt((ii - 16.0) ** 2 + (jj - 18.0) ** 2)
    with torch.no_grad():
        for seed in range(3):
            torch.manual_seed(seed)
            x = torch.randn(1, 29, H, W, dtype=torch.float64)
            net64 = net.double()
            x2 = x.clone(); x2[0, :, 16, 18] += 1.0
            d = (net64(x2) - net64(x)).abs().amax(dim=(0, 1))
            far = float(d[dist > net.reach_bound()].max())
            near = float(d.max())
            assert far < 1e-12 and near > 1e-6, (far, near)
            assert float(dist[d > 1e-12].max()) <= net.reach_bound()


if __name__ == "__main__":
    test_blob_moves(); test_cap(); test_zero_velocity_is_cnn(); test_reach_budget()
    print("advcnn tests passed")
