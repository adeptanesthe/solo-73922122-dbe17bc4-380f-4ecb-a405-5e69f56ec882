# 高温炉温度采样选点（GP 后验不确定性下降）

为高温炉温度实验选择 **2 个新增采样时刻** 的网页工具。实验员输入已测时刻/温度、
候选采样时刻、平方指数核长度尺度与观测噪声，系统基于高斯过程（GP）后验协方差，
枚举全部合法候选点对，选择 **全部候选时刻总方差下降最大** 的组合，并展示选点前后
每个候选时刻的方差。

## 数学原理

- 核函数（平方指数）：`k(x, x') = signal_variance · exp(-(x-x')² / (2ℓ²))`
- 观测模型：`y = f(x) + ε`，`ε ~ N(0, σ_n²)`
- 已测数据下的后验协方差：`Σ = K_cc − K_co (K_oo + σ_n²I)⁻¹ K_oc`
- 若在点对 `p = {i, j}` 补充观测：`Σ' = Σ − A (B + σ_n²I)⁻¹ Aᵀ`，
  其中 `A = Σ[:, p]`，`B = Σ[p, p]`
- 目标：最大化总方差下降 `Δ(p) = tr(Σ − Σ') = tr((B + σ_n²I)⁻¹ AᵀA)`，
  枚举所有合法点对取最大（`app/gp.py` 中做了 O(m²) 向量化）

温度观测值只影响后验均值，不影响后验方差，因此选点无需预知未来的测量值；
页面同时展示选点前的后验均值曲线 ±2σ 供参考。

## 合法性与整批拒绝

合法点对需同时满足：

1. 两个点都**不是已测时刻**（容差：相对 1e-9 / 绝对 1e-12）；
2. 两个点**不重复**；
3. 两点时间间隔 **≥ 最小时间间隔**；
4. 点对协方差矩阵非数值退化（行列式相对阈值 1e-12）。

以下情况**整批拒绝**（HTTP 422，不返回任何部分建议）：

| 情形 | detail.code |
|---|---|
| 输入含 NaN / Inf、参数越界、字段缺失 | `INVALID_INPUT` / `NON_FINITE_INPUT` |
| 观测核矩阵非正定（含数值近奇异，如零噪声下重复已测时刻） | `KERNEL_NOT_PSD` |
| 不存在合法候选点对 | `NO_FEASIBLE_COMBINATION` |

## 运行

### Docker Compose（推荐）

```bash
docker compose up --build
# 打开 http://localhost:8000
```

### 本地运行

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
# 打开 http://localhost:8000
```

## 网页使用

1. 粘贴或导入 CSV（已测数据每行 `时刻,温度`；候选时刻每行一个）；
2. 填写核长度尺度 ℓ、观测噪声 σₙ（标准差）、核信号方差（温度² 量级）、最小时间间隔；
3. 点击「计算选点」：页面显示选中的两个时刻、总方差前后对比、
   后验均值 ±2σ 曲线、逐候选点方差（选前/选后）明细表；
4. 「载入示例数据」可一键填充一组炉温升温曲线示例。

## API

`POST /api/select`，请求体：

```json
{
  "measured_times": [0, 1, 2, 3, 5, 8, 13],
  "measured_values": [25.3, 196.1, 330.8, 433.5, 578.2, 695.1, 769.9],
  "candidate_times": [0.5, 1.5, 2.5, 10.5, 15.0],
  "length_scale": 2.0,
  "noise_std": 5.0,
  "signal_variance": 100000.0,
  "min_interval": 1.0
}
```

响应：`selected_times`、`selected_indices`、`total_variance_before/after`、
`total_reduction`、`n_pairs_feasible/n_pairs_total`，以及每个候选点的
`eligible / exclusion_reason / variance_before / variance_after / posterior_mean / selected`。
另见交互式文档 `/docs`（FastAPI 自动生成）。规模上限：已测点 ≤ 1000，候选点 ≤ 500。

## 自测

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
```

覆盖：核函数性质、后验方差/均值正确性、与独立暴力枚举实现交叉验证选点结果、
三类排除规则、三类整批拒绝、方差单调性与总量自洽、API 正常流与错误流。

## 设计决策与边界

- **新增点数固定为 2**（按需求），核心枚举对所有 C(m, 2) 点对精确评分，非贪心近似。
- 最小时间间隔只约束两个新采样点之间；与已测时刻的间隔通过「已测点排除 + 容差」处理。
- 核矩阵判定严格：零噪声下重复已测时刻直接判非正定；Cholesky 主元低于相对阈值
  （1e-7）也判非正定，不做 jitter 修补——宁可整批拒绝也不给出数值上不可靠的建议。
- 已测时刻允许为空（退化为先验），此时选点仍有效。
- 观测噪声参数 `noise_std` 是标准差（矩阵中以其平方进入）。
