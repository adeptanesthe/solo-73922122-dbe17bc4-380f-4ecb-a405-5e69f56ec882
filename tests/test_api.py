"""API 层测试：正常流程、页面服务与整批拒绝行为。"""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "measured_times": [0, 1, 2, 3, 5, 8, 13],
    "measured_values": [25.3, 196.1, 330.8, 433.5, 578.2, 695.1, 769.9],
    "candidate_times": [0.5, 1.5, 2.5, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5,
                        8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 14, 14.5, 15, 15.5],
    "length_scale": 2.0,
    "noise_std": 5.0,
    "signal_variance": 100000.0,
    "min_interval": 1.0,
}


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_index_page_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "高温炉温度采样选点" in r.text


def test_select_ok():
    r = client.post("/api/select", json=VALID_PAYLOAD)
    assert r.status_code == 200, r.text
    data = r.json()
    # 选中两个不同的候选时刻
    assert len(data["selected_times"]) == 2
    assert data["selected_times"][0] != data["selected_times"][1]
    assert len(data["selected_indices"]) == 2
    # 总量自洽
    assert data["total_variance_before"] > data["total_variance_after"]
    assert abs(data["total_variance_before"] - data["total_variance_after"]
               - data["total_reduction"]) < 1e-6 * data["total_variance_before"]
    # 候选明细完整且选后方差不超过选前
    cands = data["candidates"]
    assert len(cands) == len(VALID_PAYLOAD["candidate_times"])
    for c in cands:
        assert c["variance_after"] <= c["variance_before"] + 1e-6
    # 已测时刻 5.0 对应的候选点被排除
    c5 = next(c for c in cands if c["time"] == 5.0)
    assert not c5["eligible"] and c5["exclusion_reason"]
    # 选中的点必须合法
    for idx in data["selected_indices"]:
        assert cands[idx]["eligible"] and cands[idx]["selected"]
    assert 0 < data["n_pairs_feasible"] <= data["n_pairs_total"]


def test_nan_in_candidates_rejected():
    payload = dict(VALID_PAYLOAD)
    raw = json.dumps(payload).replace("0.5", "NaN", 1)  # 构造含 NaN 的非法 JSON 字面量
    r = client.post("/api/select", content=raw,
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert r.json()["detail"]["code"] in ("INVALID_INPUT", "NON_FINITE_INPUT")


def test_non_psd_kernel_rejected():
    payload = dict(VALID_PAYLOAD, measured_times=[1.0, 1.0],
                   measured_values=[100.0, 100.0], noise_std=0.0)
    r = client.post("/api/select", json=payload)
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "KERNEL_NOT_PSD"


def test_no_feasible_combination_rejected():
    payload = dict(VALID_PAYLOAD, min_interval=1e6)
    r = client.post("/api/select", json=payload)
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "NO_FEASIBLE_COMBINATION"


def test_numerical_instability_rejected():
    """长度尺度远大于补测窗口且噪声极小、扩展精度也无法可靠选点时，
    整批拒绝（422），不返回任何部分建议。"""
    payload = dict(
        VALID_PAYLOAD,
        measured_times=[3.0, 7.0],
        measured_values=[400.0, 600.0],
        candidate_times=[i * 10.0 / 11 for i in range(12)],
        length_scale=1e6,
        noise_std=1e-6,
        signal_variance=1e5,
        min_interval=0.0,
    )
    r = client.post("/api/select", json=payload)
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "NUMERICAL_INSTABILITY"
    body = r.json()
    assert "candidates" not in body and "selected_times" not in body


def test_ill_conditioned_regime_selects_optimal_pair():
    """回归：长度尺度 >> 补测窗口 + 极小噪声时，选中的点对必须是
    全部合法点对中选后总方差最小者（旧实现会推荐更差的一对）。"""
    payload = dict(
        VALID_PAYLOAD,
        measured_times=[4.936198339561148, 4.9786759878537445],
        measured_values=[451.2470472198759, 451.7622688746877],
        candidate_times=[
            0.8083602389560218, 1.3339575545169313, 1.7452781614402846,
            1.776925857619811, 3.697257926524288, 3.764865843772726,
            4.304962777294131, 4.771535238392063, 5.113275528143616,
            5.439414007634982, 6.073558319950296, 6.088516168445476,
            6.656574169657534, 7.0486474556474255, 7.889467175443294,
            8.019012069858073, 8.716352741876564, 9.022150797159883,
            9.290263877654194, 9.428036791291671, 9.430561055723675,
            9.689328693162349, 9.762437057077042, 9.841529999311213,
        ],
        length_scale=1e5,
        noise_std=1e-5,
        signal_variance=1e5,
        min_interval=0.0,
    )
    r = client.post("/api/select", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    # 该用例经 50 位任意精度参考核验：最优点对为索引 (0, 23)；
    # 旧 float64 实现会错误地推荐 (0, 22)（其真实选后总方差约为最优的 5 倍）
    assert sorted(data["selected_indices"]) == [0, 23]
    assert data["total_variance_after"] < data["total_variance_before"]
    for c in data["candidates"]:
        assert c["variance_after"] <= c["variance_before"] + 1e-9


def test_mismatched_measured_lengths_rejected():
    payload = dict(VALID_PAYLOAD, measured_values=[1.0, 2.0])
    r = client.post("/api/select", json=payload)
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "INVALID_INPUT"


def test_invalid_parameters_rejected():
    for field, bad in [("length_scale", 0.0), ("noise_std", -1.0),
                       ("signal_variance", -2.0), ("min_interval", -0.5)]:
        r = client.post("/api/select", json=dict(VALID_PAYLOAD, **{field: bad}))
        assert r.status_code == 422, field
        assert r.json()["detail"]["code"] == "INVALID_INPUT"


def test_error_response_has_no_partial_result():
    payload = dict(VALID_PAYLOAD, min_interval=1e6)
    r = client.post("/api/select", json=payload)
    body = r.json()
    assert "candidates" not in body and "selected_times" not in body
