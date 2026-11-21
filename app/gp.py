"""高斯过程（GP）后验协方差与新增采样点对选择（仅依赖 NumPy）。

数学模型
--------
平方指数核:  k(x, x') = signal_variance * exp(-(x - x')^2 / (2 * length_scale^2))
观测模型:    y = f(x) + ε,  ε ~ N(0, noise_std^2)

给定已测数据后，候选时刻上的后验协方差为

    Σ = K_cc - K_co (K_oo + σ_n² I)^{-1} K_oc

若在某个候选点对 p = {i, j} 上补充观测（同样含噪声 σ_n²），则

    Σ' = Σ - A (B + σ_n² I)^{-1} Aᵀ,   其中 A = Σ[:, p],  B = Σ[p, p]

全部候选时刻的总方差下降为

    Δ(p) = tr(Σ - Σ') = tr((B + σ_n² I)^{-1} Aᵀ A)

枚举所有合法点对，取 Δ 最大者。温度观测值只影响后验均值，不影响后验方差，
因此选点阶段无需知道未来将测得的温度。

整批拒绝（不给出任何部分建议）的情形
------------------------------------
* 输入含 NaN / Inf            -> NonFiniteInputError
* 观测核矩阵非正定            -> KernelNotPSDError
* 不存在合法候选点对          -> NoFeasibleCombinationError
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "GPError",
    "NonFiniteInputError",
    "KernelNotPSDError",
    "NoFeasibleCombinationError",
    "SelectionResult",
    "squared_exponential_kernel",
    "select_pair",
]

# “同一时刻”判定的相对/绝对容差（用于已测点排除与点对去重）
_SAME_TIME_RTOL = 1e-9
_SAME_TIME_ATOL = 1e-12

# 点对协方差矩阵行列式的最小相对值，低于此值视为数值退化（该点对不可选）
_DET_REL_TOL = 1e-12


class GPError(Exception):
    """领域错误基类：出现即整批拒绝。"""

    code = "GP_ERROR"


class NonFiniteInputError(GPError):
    """输入包含 NaN 或无穷值。"""

    code = "NON_FINITE_INPUT"


class KernelNotPSDError(GPError):
    """核矩阵（含观测噪声）非正定。"""

    code = "KERNEL_NOT_PSD"


class NoFeasibleCombinationError(GPError):
    """不存在满足约束的候选点对。"""

    code = "NO_FEASIBLE_COMBINATION"


@dataclass
class SelectionResult:
    """一次选点计算的全部结果。"""

    selected_indices: tuple[int, int]
    selected_times: tuple[float, float]
    total_variance_before: float
    total_variance_after: float
    total_reduction: float
    variance_before: np.ndarray      # 每个候选时刻的选前后验方差
    variance_after: np.ndarray
    posterior_mean: np.ndarray       # 每个候选时刻的选前后验均值
    eligible: np.ndarray             # 每个候选时刻是否可参与选点
    exclusion_reasons: list          # 被排除原因（未排除为 None）
    n_pairs_total: int               # 候选点对总数 C(m, 2)
    n_pairs_feasible: int            # 合法点对数


def _as_finite_1d(values, name: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.ndim != 1:
        raise NonFiniteInputError(f"{name} 必须是一维数值序列")
    if not np.all(np.isfinite(arr)):
        raise NonFiniteInputError(f"{name} 含有 NaN 或无穷值，整批拒绝")
    return arr


def _finite_scalar(value: float, name: str) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise NonFiniteInputError(f"参数 {name} 不是有限值，整批拒绝")
    return v


def squared_exponential_kernel(x, y, length_scale: float, signal_variance: float) -> np.ndarray:
    """平方指数（RBF）核矩阵。"""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    d2 = (x[:, None] - y[None, :]) ** 2
    return signal_variance * np.exp(-0.5 * d2 / (length_scale * length_scale))


def _posterior_mean_cov(
    t_obs: np.ndarray,
    y_obs: np.ndarray,
    t_cand: np.ndarray,
    length_scale: float,
    noise_var: float,
    signal_variance: float,
):
    """候选时刻上的后验均值与后验协方差矩阵。"""
    k_cc = squared_exponential_kernel(t_cand, t_cand, length_scale, signal_variance)
    n_obs = t_obs.size
    if n_obs == 0:
        return np.zeros(t_cand.size), k_cc

    k_oo = squared_exponential_kernel(t_obs, t_obs, length_scale, signal_variance)
    k_oo = k_oo + noise_var * np.eye(n_obs)
    # 零噪声 + 重复已测时刻 => 核矩阵必然奇异（数学上只是半正定），直接判定非正定。
    if noise_var == 0.0 and n_obs > 1:
        iu0, iu1 = np.triu_indices(n_obs, k=1)
        if np.any(np.isclose(t_obs[iu0], t_obs[iu1], rtol=_SAME_TIME_RTOL, atol=_SAME_TIME_ATOL)):
            raise KernelNotPSDError(
                "观测噪声为 0 且存在重复已测时刻，观测核矩阵奇异（非正定），整批拒绝。"
                "请去重已测数据或设置正的观测噪声。"
            )
    try:
        chol = np.linalg.cholesky(k_oo)
    except np.linalg.LinAlgError as exc:
        raise KernelNotPSDError(
            "观测核矩阵（含观测噪声）非正定：可能存在重复/近重复的已测时刻且观测噪声为 0，"
            "或长度尺度相对数据范围过大导致矩阵数值奇异。请修正数据或增大观测噪声后重试。"
        ) from exc
    # Cholesky 成功还不够：主元接近 0 说明矩阵只是“数值上勉强正定”（例如近重复
    # 的已测时刻在浮点下偶然通过），按非正定处理，整批拒绝。
    pivot_scale = math.sqrt(max(float(np.max(np.diag(k_oo))), np.finfo(float).tiny))
    if np.any(np.abs(np.diag(chol)) <= 1e-7 * pivot_scale):
        raise KernelNotPSDError(
            "观测核矩阵（含观测噪声）数值上非正定（Cholesky 主元近零）：可能存在近重复的"
            "已测时刻且观测噪声过小。请修正数据或增大观测噪声后重试。"
        )

    k_co = squared_exponential_kernel(t_cand, t_obs, length_scale, signal_variance)
    # 用 Cholesky 因子求解，避免显式求逆
    w = np.linalg.solve(chol, k_co.T)                 # (n_obs, n_cand)
    cov = k_cc - w.T @ w
    cov = 0.5 * (cov + cov.T)                         # 消除非对称数值误差
    z = np.linalg.solve(chol, y_obs)
    mean = k_co @ np.linalg.solve(chol.T, z)
    return mean, cov


def select_pair(
    measured_times,
    measured_values,
    candidate_times,
    *,
    length_scale: float,
    noise_std: float,
    signal_variance: float = 1.0,
    min_interval: float = 0.0,
) -> SelectionResult:
    """枚举全部合法候选点对，返回总方差下降最大的组合。

    参数
    ----
    measured_times / measured_values : 已测时刻与温度（长度需一致，可为空）
    candidate_times                  : 候选采样时刻（至少 2 个）
    length_scale                     : 平方指数核长度尺度 ℓ (> 0)
    noise_std                        : 观测噪声标准差 σ_n (>= 0)
    signal_variance                  : 核信号方差 (> 0)
    min_interval                     : 两个新增采样点之间的最小时间间隔 (>= 0)

    合法点对需满足：两点均为非已测时刻、两点不重复、时间间隔 >= min_interval，
    且点对协方差矩阵非数值退化。任何输入异常都会抛出 GPError 子类，整批拒绝。
    """
    length_scale = _finite_scalar(length_scale, "length_scale")
    noise_std = _finite_scalar(noise_std, "noise_std")
    signal_variance = _finite_scalar(signal_variance, "signal_variance")
    min_interval = _finite_scalar(min_interval, "min_interval")
    if length_scale <= 0:
        raise ValueError("length_scale 必须 > 0")
    if noise_std < 0:
        raise ValueError("noise_std 必须 >= 0")
    if signal_variance <= 0:
        raise ValueError("signal_variance 必须 > 0")
    if min_interval < 0:
        raise ValueError("min_interval 必须 >= 0")

    t_obs = _as_finite_1d(measured_times, "已测时刻")
    y_obs = _as_finite_1d(measured_values, "已测温度")
    if t_obs.size != y_obs.size:
        raise ValueError("已测时刻与已测温度数量不一致")
    t_cand = _as_finite_1d(candidate_times, "候选时刻")
    n_cand = t_cand.size
    if n_cand < 2:
        raise NoFeasibleCombinationError("候选时刻少于 2 个，无法组成采样点对")

    noise_var = noise_std * noise_std
    mean, cov = _posterior_mean_cov(t_obs, y_obs, t_cand, length_scale, noise_var, signal_variance)

    var_before = np.diag(cov).copy()
    neg_tol = 1e-8 * max(signal_variance, noise_var, 1.0)
    if np.any(var_before < -neg_tol):
        raise KernelNotPSDError("后验协方差对角线出现显著负值，核矩阵数值异常，整批拒绝")
    var_before = np.clip(var_before, 0.0, None)

    # ---- 合法性规则 1：与已测时刻重复的候选点不可选 ----
    eligible = np.ones(n_cand, dtype=bool)
    reasons: list = [None] * n_cand
    for i, t in enumerate(t_cand):
        if t_obs.size and np.any(
            np.isclose(t, t_obs, rtol=_SAME_TIME_RTOL, atol=_SAME_TIME_ATOL)
        ):
            eligible[i] = False
            reasons[i] = "与已测时刻重复"

    # ---- 枚举全部点对（向量化）----
    ii, jj = np.triu_indices(n_cand, k=1)
    dt = np.abs(t_cand[ii] - t_cand[jj])
    # 规则 2：两点不得重复；规则 3：两点间隔不得小于 min_interval
    same = np.isclose(t_cand[ii], t_cand[jj], rtol=_SAME_TIME_RTOL, atol=_SAME_TIME_ATOL)
    feasible = eligible[ii] & eligible[jj] & (~same) & (dt >= min_interval)

    # 总方差下降 Δ(i,j) = tr((B + σ_n²I)^{-1} S)，其中
    # B = Σ[[i,j],[i,j]]，S = (AᵀA)，A = Σ[:, [i,j]]，S[p,q] = (ΣᵀΣ)[p,q]
    h_mat = cov @ cov  # cov 已对称化：h_mat[i, j] = <cov[:, i], cov[:, j]>
    b00 = var_before[ii] + noise_var
    b11 = var_before[jj] + noise_var
    b01 = cov[ii, jj]
    det = b00 * b11 - b01 * b01
    # 点对矩阵数值退化（近似奇异）的组合不可选
    det_ok = det > _DET_REL_TOL * np.maximum(b00 * b11, np.finfo(float).tiny)
    feasible = feasible & det_ok

    s00 = h_mat[ii, ii]
    s11 = h_mat[jj, jj]
    s01 = h_mat[ii, jj]
    with np.errstate(divide="ignore", invalid="ignore"):
        reductions = (b11 * s00 + b00 * s11 - 2.0 * b01 * s01) / det
    reductions = np.where(feasible & np.isfinite(reductions), reductions, -np.inf)

    n_pairs_total = int(ii.size)
    n_pairs_feasible = int(np.count_nonzero(feasible))
    if n_pairs_feasible == 0:
        raise NoFeasibleCombinationError(
            "没有满足约束的候选点对（已测点/重复点被排除，或点对不满足最小时间间隔），整批拒绝"
        )

    best = int(np.argmax(reductions))
    sel_i, sel_j = int(ii[best]), int(jj[best])

    # ---- 选中方案下，每个候选时刻的选后后验方差 ----
    a_mat = cov[:, [sel_i, sel_j]]                                    # (n_cand, 2)
    b_mat = cov[np.ix_([sel_i, sel_j], [sel_i, sel_j])] + noise_var * np.eye(2)
    x_mat = np.linalg.solve(b_mat, a_mat.T)                           # (2, n_cand)
    var_after = var_before - np.sum(a_mat * x_mat.T, axis=1)
    var_after = np.clip(var_after, 0.0, None)

    total_before = float(np.sum(var_before))
    total_after = float(np.sum(var_after))

    return SelectionResult(
        selected_indices=(sel_i, sel_j),
        selected_times=(float(t_cand[sel_i]), float(t_cand[sel_j])),
        total_variance_before=total_before,
        total_variance_after=total_after,
        total_reduction=total_before - total_after,
        variance_before=var_before,
        variance_after=var_after,
        posterior_mean=mean,
        eligible=eligible,
        exclusion_reasons=reasons,
        n_pairs_total=n_pairs_total,
        n_pairs_feasible=n_pairs_feasible,
    )
