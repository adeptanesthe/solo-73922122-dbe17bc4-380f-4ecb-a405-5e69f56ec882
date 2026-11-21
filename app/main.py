"""FastAPI 入口：高温炉温度采样选点服务。"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from .gp import GPError, select_pair
from .schemas import SelectRequest

app = FastAPI(
    title="高温炉温度采样选点",
    version="1.0.0",
    summary="基于高斯过程后验协方差下降，为炉温实验选择两个新增采样点",
)

_STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.exception_handler(RequestValidationError)
async def _validation_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """把 schema 校验失败统一成结构化 422（整批拒绝，不给部分建议）。"""
    errors = [
        {
            "loc": [str(p) for p in err.get("loc", [])],
            "msg": err.get("msg", ""),
            "type": err.get("type", ""),
        }
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "code": "INVALID_INPUT",
                "message": "输入校验失败（字段缺失、非有限值或参数越界），整批拒绝",
                "errors": errors,
            }
        },
    )


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/select")
def select(req: SelectRequest) -> dict:
    """计算最优新增采样点对，返回选点前后每个候选时刻的方差。

    失败时返回 422，detail.code 取值：
    NON_FINITE_INPUT / KERNEL_NOT_PSD / NO_FEASIBLE_COMBINATION / INVALID_INPUT。
    """
    try:
        res = select_pair(
            req.measured_times,
            req.measured_values,
            req.candidate_times,
            length_scale=req.length_scale,
            noise_std=req.noise_std,
            signal_variance=req.signal_variance,
            min_interval=req.min_interval,
        )
    except GPError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc

    candidates = [
        {
            "index": i,
            "time": float(t),
            "eligible": bool(res.eligible[i]),
            "exclusion_reason": res.exclusion_reasons[i],
            "variance_before": float(res.variance_before[i]),
            "variance_after": float(res.variance_after[i]),
            "posterior_mean": float(res.posterior_mean[i]),
            "selected": i in res.selected_indices,
        }
        for i, t in enumerate(req.candidate_times)
    ]
    return {
        "selected_indices": list(res.selected_indices),
        "selected_times": list(res.selected_times),
        "total_variance_before": res.total_variance_before,
        "total_variance_after": res.total_variance_after,
        "total_reduction": res.total_reduction,
        "n_pairs_total": res.n_pairs_total,
        "n_pairs_feasible": res.n_pairs_feasible,
        "candidates": candidates,
    }


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")
