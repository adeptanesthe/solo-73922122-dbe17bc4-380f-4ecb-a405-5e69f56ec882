"""核心 GP 选点逻辑的单元测试。"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from app.gp import (
    KernelNotPSDError,
    NoFeasibleCombinationError,
    NonFiniteInputError,
    select_pair,
    squared_exponential_kernel,
)

RTOL_SAME = 1e-9
ATOL_SAME = 1e-12


def _brute_force_best(t_obs, y_obs, t_cand, *, length_scale, noise_std,
                      signal_variance, min_interval):
    """独立参照实现：对每个合法点对显式重建 GP 后验并求总方差。

    与 app.gp.select_pair 的向量化公式相互独立，用于交叉验证。
    返回 (best_reduction, best_pair, var_after_by_pair)。
    """
    del y_obs  # 方差不依赖观测值
    t_obs = np.asarray(t_obs, float)
    t_cand = np.asarray(t_cand, float)
    noise_var = noise_std ** 2

    def posterior_var(obs_t):
        k_cc = squared_exponential_kernel(t_cand, t_cand, length_scale, signal_variance)
        if len(obs_t) == 0:
            return np.diag(k_cc).copy()
        k_oo = (squared_exponential_kernel(obs_t, obs_t, length_scale, signal_variance)
                + noise_var * np.eye(len(obs_t)))
        k_co = squared_exponential_kernel(t_cand, obs_t, length_scale, signal_variance)
        sol = np.linalg.solve(k_oo, k_co.T)
        return np.clip(np.diag(k_cc - k_co @ sol), 0.0, None)

    base = posterior_var(t_obs)
    best = None
    after = {}
    for i, j in combinations(range(len(t_cand)), 2):
        if np.any(np.isclose(t_cand[i], t_obs, rtol=RTOL_SAME, atol=ATOL_SAME)):
            continue
        if np.any(np.isclose(t_cand[j], t_obs, rtol=RTOL_SAME, atol=ATOL_SAME)):
            continue
        if np.isclose(t_cand[i], t_cand[j], rtol=RTOL_SAME, atol=ATOL_SAME):
            continue
        if abs(t_cand[i] - t_cand[j]) < min_interval:
            continue
        aug = np.concatenate([t_obs, [t_cand[i], t_cand[j]]])
        var = posterior_var(aug)
        red = float(base.sum() - var.sum())
        after[(i, j)] = var
        if best is None or red > best[0] + 1e-12 * max(1.0, abs(best[0])):
            best = (red, (i, j))
    return best, after


# ---------------- 核函数 ----------------

def test_kernel_properties():
    k = squared_exponential_kernel([0.0, 1.0], [0.0, 1.0], length_scale=2.0, signal_variance=4.0)
    assert k.shape == (2, 2)
    assert np.allclose(np.diag(k), 4.0)                 # k(x, x) = 信号方差
    assert np.allclose(k, k.T)                          # 对称
    assert k[0, 1] < k[0, 0]                            # 随距离衰减
    assert np.all(np.linalg.eigvalsh(k) > 0)            # 不同点上严格正定


# ---------------- 后验方差 ----------------

def test_posterior_variance_shrinks_at_observed():
    res = select_pair([0.0, 1.0, 2.0], [10.0, 20.0, 15.0], [0.0, 1.0, 2.0, 5.0, 9.0],
                      length_scale=1.0, noise_std=0.1, signal_variance=4.0)
    # 已测点上的后验方差远小于先验方差 4.0
    assert np.all(res.variance_before[:3] < 0.05)
    # 远离已测点的候选点方差接近先验
    assert res.variance_before[4] > 3.0


def test_posterior_mean_interpolates_with_tiny_noise():
    res = select_pair([0.0, 1.0], [100.0, 200.0], [0.0, 1.0, 0.5, 2.5, 3.5],
                      length_scale=1.0, noise_std=1e-4, signal_variance=1e4)
    assert res.posterior_mean[0] == pytest.approx(100.0, abs=0.5)
    assert res.posterior_mean[1] == pytest.approx(200.0, abs=0.5)


# ---------------- 与独立暴力枚举交叉验证 ----------------

def test_selection_matches_independent_bruteforce():
    rng = np.random.default_rng(42)
    t_obs = np.sort(rng.uniform(0, 10, 7))
    y_obs = np.sin(t_obs) * 300 + 400
    t_cand = np.sort(rng.uniform(0, 10, 9))
    params = dict(length_scale=1.7, noise_std=0.15, signal_variance=2.5e4, min_interval=0.4)

    res = select_pair(t_obs, y_obs, t_cand, **params)
    (best_red, best_pair), after = _brute_force_best(t_obs, y_obs, t_cand, **params)

    assert best_pair is not None
    # 选中的时刻集合一致
    assert set(res.selected_indices) == set(best_pair)
    # 总方差下降一致
    assert res.total_reduction == pytest.approx(best_red, rel=1e-6)
    # 每个候选点的选后方差与参照实现一致
    np.testing.assert_allclose(res.variance_after, after[best_pair], rtol=1e-6, atol=1e-8)


# ---------------- 排除规则 ----------------

def test_measured_points_excluded():
    res = select_pair([0.0], [500.0], [0.0, 10.0, 20.0],
                      length_scale=1.0, noise_std=0.1, signal_variance=1e4)
    assert not res.eligible[0]
    assert res.exclusion_reasons[0] is not None
    assert set(res.selected_times) == {10.0, 20.0}


def test_duplicate_candidate_pair_excluded():
    # 候选 [5, 5, 9]：点对 (5, 5) 是重复点，必须排除
    res = select_pair([0.0], [500.0], [5.0, 5.0, 9.0],
                      length_scale=1.0, noise_std=0.1, signal_variance=1e4,
                      min_interval=0.0)
    assert set(res.selected_times) == {5.0, 9.0}
    assert res.n_pairs_feasible == 2  # (5a,9) 与 (5b,9)


def test_min_interval_enforced():
    res = select_pair([-5.0], [100.0], [0.0, 1.0, 2.0],
                      length_scale=1.0, noise_std=0.1, signal_variance=1e4,
                      min_interval=1.5)
    assert res.n_pairs_feasible == 1  # 只有 (0, 2) 满足间隔
    assert set(res.selected_times) == {0.0, 2.0}


# ---------------- 整批拒绝 ----------------

def test_no_feasible_combination_raises():
    with pytest.raises(NoFeasibleCombinationError):
        select_pair([], [], [0.0, 1.0], length_scale=1.0, noise_std=0.1,
                    min_interval=5.0)
    with pytest.raises(NoFeasibleCombinationError):
        select_pair([1.0, 2.0], [10.0, 20.0], [1.0, 2.0],
                    length_scale=1.0, noise_std=0.1)  # 候选全是已测点
    with pytest.raises(NoFeasibleCombinationError):
        select_pair([], [], [1.0], length_scale=1.0, noise_std=0.1)  # 候选不足


def test_non_finite_input_rejected():
    base = dict(length_scale=1.0, noise_std=0.1)
    with pytest.raises(NonFiniteInputError):
        select_pair([0.0], [1.0], [1.0, float("nan"), 2.0], **base)
    with pytest.raises(NonFiniteInputError):
        select_pair([0.0], [float("inf")], [1.0, 2.0], **base)
    with pytest.raises(NonFiniteInputError):
        select_pair([0.0], [1.0], [1.0, 2.0], length_scale=float("nan"), noise_std=0.1)
    with pytest.raises(NonFiniteInputError):
        select_pair([0.0], [1.0], [1.0, 2.0], length_scale=1.0, noise_std=float("inf"))


def test_non_psd_kernel_rejected():
    # 重复已测时刻 + 零观测噪声 -> 观测核矩阵奇异
    with pytest.raises(KernelNotPSDError):
        select_pair([1.0, 1.0], [100.0, 100.0], [2.0, 3.0],
                    length_scale=1.0, noise_std=0.0)
    # 同样的数据，加一点噪声即可通过（对照）
    res = select_pair([1.0, 1.0], [100.0, 100.0], [2.0, 3.0],
                      length_scale=1.0, noise_std=0.1)
    assert len(res.selected_times) == 2


# ---------------- 结果一致性 ----------------

def test_variance_monotonic_and_totals_consistent():
    rng = np.random.default_rng(7)
    t_obs = np.sort(rng.uniform(0, 20, 8))
    y_obs = rng.normal(500, 50, 8)
    t_cand = np.sort(rng.uniform(0, 20, 15))
    res = select_pair(t_obs, y_obs, t_cand, length_scale=2.5, noise_std=1.0,
                      signal_variance=3e3, min_interval=0.5)
    # 选后方差处处不超过选前（允许数值误差）
    assert np.all(res.variance_after <= res.variance_before + 1e-8)
    # 总量自洽
    assert res.total_variance_before == pytest.approx(res.variance_before.sum())
    assert res.total_variance_after == pytest.approx(res.variance_after.sum())
    assert res.total_reduction == pytest.approx(
        res.total_variance_before - res.total_variance_after)
    assert res.total_reduction > 0
    # 选中点自身方差显著下降
    for idx in res.selected_indices:
        assert res.variance_after[idx] < res.variance_before[idx]


def test_empty_observations_allowed():
    res = select_pair([], [], [0.0, 1.0, 5.0], length_scale=1.0, noise_std=0.1,
                      signal_variance=2.0, min_interval=0.5)
    # 无观测时退化为先验：各候选点方差 = 信号方差
    assert np.allclose(res.variance_before, 2.0)
    assert len(res.selected_times) == 2
