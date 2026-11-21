"""高斯过程（GP）后验协方差与新增采样点对选择（仅依赖 NumPy）。

数学模型
--------
平方指数核:  k(x, x') = signal_variance * exp(-(x - x')^2 / (2 * length_scale^2))
观测模型:    y = f(x) + ε,  ε ~ N(0, noise_std^2)

给定已测数据后，候选时刻上的后验协方差为

    Σ = K_cc - K_co (K_oo + σ_n² I)^{-1} K_oc

若在某个候选点对 p = {i, j} 上补充观测（同样含噪声 σ_n²），则

    Σ' = Σ - A (B + σ_n² I)^{-1} Aᵀ,   其中 A = Σ[:, p],  B = Σ[p, p]

全部候选时刻的选后总方差为 tr(Σ')，等价地总方差下降为

    Δ(p) = tr(Σ - Σ') = tr((B + σ_n² I)^{-1} Aᵀ A)

枚举所有合法点对，取选后总方差最小（即 Δ 最大）者。温度观测值只影响后验均值，
不影响后验方差，因此选点阶段无需知道未来将测得的温度。

数值策略
--------
后验协方差 Σ = K_cc − WᵀW 存在灾难性抵消：当核长度尺度远大于补测窗口且观测
噪声很小时，后验方差（~σ_n² 量级）是 O(signal_variance) 的两个量相减的结果，
float64（约 16 位有效数字）不足以分辨不同点对选后总方差的差异，可能推荐
实际更差（选后总方差更高）的点对。为此：

* 核矩阵、Cholesky 分解、三角求解、后验协方差与点对评分全部在
  np.longdouble（x86-64 Linux 上为 80 位扩展精度，约 19 位有效数字）中计算；
* 点对评分使用数值稳定的残差（Schur 补）形式

      Δ(i, j) = ||A_i||² / b00 + ||A_j − c·A_i||² / l11²,
      c = b01 / b00,  l11² = b11 − b01² / b00,

  避免闭式分子 b11·s00 + b00·s11 − 2·b01·s01 的抵消；评分按行分块进行，
  点对比较的额外内存峰值随候选数平方增长（O(m²)）；
* 选中方案用“把选中点对并入观测集、直接重建后验”的方式独立重算交叉验证。
  两种计算不一致超过相对容差 _GUARD_RTOL，或增广核矩阵在扩展精度下仍非正定，
  即判定数值异常，整批拒绝（NumericalInstabilityError）。

整批拒绝（不给出任何部分建议）的情形
------------------------------------
* 输入含 NaN / Inf            -> NonFiniteInputError
* 观测核矩阵非正定            -> KernelNotPSDError
* 不存在合法候选点对          -> NoFeasibleCombinationError
* 数值精度不足无法可靠选点    -> NumericalInstabilityError
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
    "NumericalInstabilityError",
    "SelectionResult",
    "squared_exponential_kernel",
    "select_pair",
]

# “同一时刻”判定的相对/绝对容差（用于已测点排除与点对去重）
_SAME_TIME_RTOL = 1e-9
_SAME_TIME_ATOL = 1e-12

# 点对协方差矩阵行列式的最小相对值，低于此值视为数值退化（该点对不可选）
_DET_REL_TOL = 1e-12

# 扩展精度工作类型：x86-64 Linux 上为 80 位扩展精度（约 19 位有效数字）；
# 在 longdouble 退化为 float64 的平台（如 Windows）上，下方的可靠性核验
# 会在病态情形下整批拒绝，而不是给出不可靠的建议。
_LD = np.longdouble

# 流水线结果与独立重算结果允许的最大相对偏差，超过即判数值不可靠。
# 标定依据（详见 tests 中的病态回归用例）：条件数 ≲1e16 时两者差异 ≲2e-5，
# 条件数 ≳4e17（选点开始出错的区域）差异 ≳6e-4，阈值 1e-4 位于其间。
_GUARD_RTOL = 1e-4


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


class NumericalInstabilityError(GPError):
    """数值精度不足，无法可靠选点。"""

    code = "NUMERICAL_INSTABILITY"


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
    """平方指数（RBF）核矩阵（float64，供外部/测试使用）。"""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    d2 = (x[:, None] - y[None, :]) ** 2
    return signal_variance * np.exp(-0.5 * d2 / (length_scale * length_scale))


def _kernel_ld(x, y, length_scale, signal_variance) -> np.ndarray:
    """longdouble 平方指数核矩阵；标量参数需已为 longdouble。"""
    x = np.asarray(x, dtype=_LD).ravel()
    y = np.asarray(y, dtype=_LD).ravel()
    d2 = (x[:, None] - y[None, :]) ** 2
    return signal_variance * np.exp(-0.5 * d2 / (length_scale * length_scale))


def _cholesky_ld(a: np.ndarray) -> np.ndarray:
    """longdouble 的 Cholesky 分解（np.linalg 不支持扩展精度，手写向量化版本）。

    矩阵非正定（含 NaN 主元）时抛出 np.linalg.LinAlgError。
    """
    n = a.shape[0]
    L = np.zeros_like(a)
    for i in range(n):
        s = a[i, i] - np.dot(L[i, :i], L[i, :i])
        if not s > 0:  # 同时捕获 s <= 0 与 NaN
            raise np.linalg.LinAlgError("matrix is not positive definite")
        L[i, i] = np.sqrt(s)
        if i + 1 < n:
            L[i + 1:, i] = (a[i + 1:, i] - L[i + 1:, :i] @ L[i, :i]) / L[i, i]
    return L


def _solve_lower_ld(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """前代法解 L x = b（L 下三角，b 形状 (n, k)）。"""
    n = L.shape[0]
    x = np.empty_like(b)
    for i in range(n):
        x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]
    return x


def _solve_upper_ld(u: np.ndarray, b: np.ndarray) -> np.ndarray:
    """回代法解 U x = b（U 上三角，b 形状 (n, k)）。"""
    n = u.shape[0]
    x = np.empty_like(b)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - u[i, i + 1:] @ x[i + 1:]) / u[i, i]
    return x


def _posterior_mean_cov(
    t_obs: np.ndarray,
    y_obs: np.ndarray,
    t_cand: np.ndarray,
    length_scale: float,
    noise_var: float,
    signal_variance: float,
):
    """候选时刻上的后验均值与后验协方差矩阵（longdouble 计算）。"""
    ls = _LD(length_scale)
    sv = _LD(signal_variance)
    nv = _LD(noise_var)
    t_obs = np.asarray(t_obs, dtype=_LD)
    y_obs = np.asarray(y_obs, dtype=_LD)
    t_cand = np.asarray(t_cand, dtype=_LD)

    k_cc = _kernel_ld(t_cand, t_cand, ls, sv)
    n_obs = t_obs.size
    if n_obs == 0:
        return np.zeros(t_cand.size, dtype=_LD), k_cc

    k_oo = _kernel_ld(t_obs, t_obs, ls, sv)
    k_oo = k_oo + nv * np.eye(n_obs, dtype=_LD)
    # 零噪声 + 重复已测时刻 => 核矩阵必然奇异（数学上只是半正定），直接判定非正定。
    if noise_var == 0.0 and n_obs > 1:
        iu0, iu1 = np.triu_indices(n_obs, k=1)
        if np.any(np.isclose(t_obs[iu0], t_obs[iu1], rtol=_SAME_TIME_RTOL, atol=_SAME_TIME_ATOL)):
            raise KernelNotPSDError(
                "观测噪声为 0 且存在重复已测时刻，观测核矩阵奇异（非正定），整批拒绝。"
                "请去重已测数据或设置正的观测噪声。"
            )
    try:
        chol = _cholesky_ld(k_oo)
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

    k_co = _kernel_ld(t_cand, t_obs, ls, sv)
    # 用 Cholesky 因子求解，避免显式求逆
    w = _solve_lower_ld(chol, k_co.T)                   # (n_obs, n_cand)
    cov = k_cc - w.T @ w
    cov = 0.5 * (cov + cov.T)                           # 消除非对称数值误差
    z = _solve_lower_ld(chol, y_obs[:, None])
    mean = (k_co @ _solve_upper_ld(chol.T, z)).ravel()
    return mean, cov


def _verify_selected_pair(
    t_obs: np.ndarray,
    t_cand: np.ndarray,
    sel_i: int,
    sel_j: int,
    *,
    length_scale: float,
    noise_var: float,
    signal_variance: float,
    var_after: np.ndarray,
    total_after,
) -> None:
    """独立重算核验：把选中点对并入观测集直接重建后验，与流水线结果交叉验证。

    两种计算（基于后验协方差 Σ 的 2x2 更新 vs. 增广观测集的整体 Cholesky 重建）
    的舍入误差来源不同，若它们对选中方案的选后方差/总方差不一致超过
    _GUARD_RTOL，说明当前精度不足以可靠选点，抛出 NumericalInstabilityError
    整批拒绝；增广核矩阵在扩展精度下仍非正定同样整批拒绝。
    """
    ls = _LD(length_scale)
    sv = _LD(signal_variance)
    nv = _LD(noise_var)
    aug = np.concatenate([t_obs, [t_cand[sel_i], t_cand[sel_j]]])
    k_oo = _kernel_ld(aug, aug, ls, sv) + nv * np.eye(aug.size, dtype=_LD)
    try:
        chol = _cholesky_ld(k_oo)
    except np.linalg.LinAlgError as exc:
        raise NumericalInstabilityError(
            "增广观测核矩阵在扩展精度下仍非正定（长度尺度相对数据范围过大且观测噪声过小），"
            "无法可靠核验选点结果，整批拒绝。请增大观测噪声或缩短长度尺度后重试。"
        ) from exc
    k_co = _kernel_ld(t_cand, aug, ls, sv)
    w = _solve_lower_ld(chol, k_co.T)
    var_direct = sv - np.sum(w * w, axis=0)             # k(x, x) = signal_variance
    total_direct = float(np.sum(var_direct))
    if not np.all(np.isfinite(var_direct)):
        raise NumericalInstabilityError("独立重算出现非有限值，数值异常，整批拒绝")

    # 每个候选点的选后方差理论上不低于观测噪声与先验的调和量级，
    # 用它给相对容差一个下界，防止总方差接近 0 时容差失效。
    n_cand = t_cand.size
    floor = 0.5 * n_cand * min(float(nv), float(sv))
    scale = max(total_direct, floor, np.finfo(float).tiny)
    tol = _GUARD_RTOL * scale
    if abs(float(total_after) - total_direct) > tol:
        raise NumericalInstabilityError(
            "选中方案的选后总方差与独立重算结果不一致（数值精度不足），"
            "无法可靠选点，整批拒绝。请增大观测噪声或缩短长度尺度后重试。"
        )
    if float(np.max(np.abs(var_after - var_direct))) > tol:
        raise NumericalInstabilityError(
            "选中方案的逐点选后方差与独立重算结果不一致（数值精度不足），"
            "无法可靠选点，整批拒绝。请增大观测噪声或缩短长度尺度后重试。"
        )


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
    """枚举全部合法候选点对，返回全部候选时刻选后总方差最小的组合。

    参数
    ----
    measured_times / measured_values : 已测时刻与温度（长度需一致，可为空）
    candidate_times                  : 候选采样时刻（至少 2 个）
    length_scale                     : 平方指数核长度尺度 ℓ (> 0)
    noise_std                        : 观测噪声标准差 σ_n (>= 0)
    signal_variance                  : 核信号方差 (> 0)
    min_interval                     : 两个新增采样点之间的最小时间间隔 (>= 0)

    合法点对需满足：两点均为非已测时刻、两点不重复、时间间隔 >= min_interval，
    且点对协方差矩阵非数值退化。任何输入异常或数值异常都会抛出 GPError 子类，
    整批拒绝。
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
    t_obs_ld = np.asarray(t_obs, dtype=_LD)
    t_cand_ld = np.asarray(t_cand, dtype=_LD)
    nv = _LD(noise_var)

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

    # 规则 4：点对矩阵数值退化（近似奇异）的组合不可选。
    # B = Σ[[i,j],[i,j]] + σ_n²I，det 为其行列式；与评分共用量按对预计算。
    b00 = var_before[ii] + nv
    b11 = var_before[jj] + nv
    b01 = cov[ii, jj]
    det = b00 * b11 - b01 * b01
    det_ok = det > _DET_REL_TOL * np.maximum(b00 * b11, np.finfo(float).tiny)
    feasible = feasible & det_ok

    n_pairs_total = int(ii.size)
    n_pairs_feasible = int(np.count_nonzero(feasible))
    if n_pairs_feasible == 0:
        raise NoFeasibleCombinationError(
            "没有满足约束的候选点对（已测点/重复点被排除，或点对不满足最小时间间隔），整批拒绝"
        )

    # ---- 稳定评分：总方差下降 Δ(i,j) = tr((B+σ_n²I)^{-1} AᵀA) 的残差形式 ----
    # Δ(i,j) = ||A_i||²/b00 + ||A_j − c·A_i||²/l11²，c = b01/b00，
    # l11² = b11 − b01²/b00（Schur 补）。按 i 行分块计算，块内对 j 向量化，
    # 额外内存峰值 O(m²)。选后总方差 = 选前总方差 − Δ，取 Δ 最大者即选后总方差最小。
    colnorm2 = np.sum(cov * cov, axis=0)                # ||A_i||² = (ΣᵀΣ)[i, i]
    reductions = np.full(n_pairs_total, -np.inf, dtype=_LD)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for i in range(n_cand - 1):
            start = i * (2 * n_cand - i - 1) // 2       # 第 i 行点对在 ii/jj 中的偏移
            sl = slice(start, start + (n_cand - 1 - i))
            b00_i = var_before[i] + nv
            c_row = cov[i, i + 1:] / b00_i
            schur_row = (var_before[i + 1:] + nv) - c_row * cov[i, i + 1:]
            resid = cov[:, i + 1:] - cov[:, i:i + 1] * c_row
            r2 = np.sum(resid * resid, axis=0)
            reductions[sl] = colnorm2[i] / b00_i + r2 / schur_row
    reductions = np.where(feasible & np.isfinite(reductions), reductions, -np.inf)

    best = int(np.argmax(reductions))
    if reductions[best] == -np.inf:
        raise NumericalInstabilityError(
            "所有合法点对的方差下降均无法有限计算（数值上溢/下溢），整批拒绝"
        )
    sel_i, sel_j = int(ii[best]), int(jj[best])

    # ---- 选中方案下，每个候选时刻的选后后验方差（同一稳定残差形式）----
    a0 = cov[:, sel_i]
    a1 = cov[:, sel_j]
    b00_sel = var_before[sel_i] + nv
    c_sel = cov[sel_i, sel_j] / b00_sel
    schur_sel = var_before[sel_j] + nv - c_sel * cov[sel_i, sel_j]
    resid_sel = a1 - c_sel * a0
    var_after = var_before - (a0 * a0 / b00_sel + resid_sel * resid_sel / schur_sel)
    var_after = np.clip(var_after, 0.0, None)

    total_before = np.sum(var_before)
    total_after = np.sum(var_after)

    # ---- 数值可靠性核验：独立重算选中方案，不一致则整批拒绝 ----
    _verify_selected_pair(
        t_obs_ld, t_cand_ld, sel_i, sel_j,
        length_scale=length_scale, noise_var=noise_var, signal_variance=signal_variance,
        var_after=var_after, total_after=total_after,
    )

    # ---- 输出（转回 float64，API 字段不变）----
    var_before_f = np.asarray(var_before, dtype=float)
    var_after_f = np.asarray(var_after, dtype=float)
    mean_f = np.asarray(mean, dtype=float)
    if not (np.all(np.isfinite(var_before_f)) and np.all(np.isfinite(var_after_f))
            and np.all(np.isfinite(mean_f))):
        raise NumericalInstabilityError("计算结果含非有限值，数值异常，整批拒绝")
    total_before_f = float(total_before)
    total_after_f = float(total_after)
    if not (math.isfinite(total_before_f) and math.isfinite(total_after_f)):
        raise NumericalInstabilityError("总方差含非有限值，数值异常，整批拒绝")

    return SelectionResult(
        selected_indices=(sel_i, sel_j),
        selected_times=(float(t_cand[sel_i]), float(t_cand[sel_j])),
        total_variance_before=total_before_f,
        total_variance_after=total_after_f,
        total_reduction=total_before_f - total_after_f,
        variance_before=var_before_f,
        variance_after=var_after_f,
        posterior_mean=mean_f,
        eligible=eligible,
        exclusion_reasons=reasons,
        n_pairs_total=n_pairs_total,
        n_pairs_feasible=n_pairs_feasible,
    )
