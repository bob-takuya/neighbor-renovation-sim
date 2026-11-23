"""3×3 の全配置（23,887,872 通り）の総当たり。満足の合計 W と最小の満足 minS を配列で返す。
配列の形は (64 開口, 512 張り出し, 729 収納)、添字は s0.encode と一致する。"""
import itertools
import numpy as np
from . import s0

E = np.array(list(itertools.product([0, 1], repeat=9)), dtype=np.int8)
Kc = np.array(list(itertools.product([-1, 0, 1], repeat=6)), dtype=np.int8)
O = np.array(list(itertools.product([0, 1], repeat=6)), dtype=np.int8)


def enumerate_all(W, form="linear"):
    Wsum = np.empty((64, 512, 729), dtype=np.float32)
    Smin = np.empty((64, 512, 729), dtype=np.float32)
    for oi in range(64):
        o = O[oi]
        tot = np.zeros((512, 729)); mn = np.full((512, 729), np.inf)
        for i in range(9):
            area = s0.A0 + s0.EXT_A * E[:, i][:, None] + np.zeros((1, 729))
            r, l = s0.rb(i), s0.lb(i)
            op = 0.0
            if r is not None:
                area = area + s0.CLO_A * Kc[:, r][None, :]; op += o[r]
            if l is not None:
                area = area - s0.CLO_A * Kc[:, l][None, :]; op += o[l]
            sun = np.full((512, 1), s0.SUN0[i])
            for d, sh in zip((1, 2), s0.SHADE):
                a = s0.above(i, d)
                if a is not None:
                    sun = sun - sh * E[:, a][:, None]
            za = np.clip((area - s0.A_LO) / (s0.A_HI - s0.A_LO), 0, 1)
            zs = np.clip(sun / s0.SUN_HI, 0, 1)
            if form == "concave":
                za, zs = np.sqrt(za), np.sqrt(zs)
            if form == "floor":
                zs = np.where(sun < 2.0, zs - 0.6 * (2.0 - sun) / 2.0, zs)
            si = W[i, 0] * za + W[i, 1] * zs + W[i, 2] * (1 - op / 2) + W[i, 3] * (op / 2)
            tot += si; mn = np.minimum(mn, si)
        Wsum[oi] = tot; Smin[oi] = mn
    return Wsum, Smin


def decode(flat):
    io, rest = divmod(int(flat), 512 * 729)
    ie, ik = divmod(rest, 729)
    return E[ie].astype(int), Kc[ik].astype(int), O[io].astype(int)
