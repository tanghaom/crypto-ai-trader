"""
web_data.py

面向前端的数据组装/聚合层：
- 提供模型元数据、状态、快照与总览曲线等只读接口
- 读取 deepseekok2 中的共享状态（MODEL_CONTEXTS、MODEL_ORDER、history_store 等）
- 线程安全：保留对 ctx.lock 的使用

注意：避免循环依赖，仅在此模块 import deepseekok2，deepseekok2 不应 import 本模块。
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import copy

# 仅用于读取共享状态与历史存储，不做初始化/副作用
import deepseekok2 as core

# 时间范围预设
RANGE_PRESETS = {
    "1d": timedelta(days=1),
    "7d": timedelta(days=7),
    "15d": timedelta(days=15),
    "1m": timedelta(days=30),
    "1y": timedelta(days=365),
}


def list_model_keys() -> List[str]:
    """返回模型键列表（顺序与 core.MODEL_ORDER 一致）。"""
    return core.MODEL_ORDER


def get_model_metadata() -> List[Dict[str, str]]:
    """返回供前端展示的模型元数据。"""
    return [
        {
            "key": key,
            "display": ctx.display,
            "model_name": ctx.model_name,
            "provider": ctx.provider,
            "sub_account": getattr(ctx, "sub_account", None),
        }
        for key, ctx in core.MODEL_CONTEXTS.items()
    ]


def get_models_status() -> List[Dict[str, Dict]]:
    """返回前端需要的模型连接状态与账户概要。"""
    statuses: List[Dict[str, Dict]] = []
    for key in core.MODEL_ORDER:
        ctx = core.MODEL_CONTEXTS[key]
        with ctx.lock:
            statuses.append(
                {
                    "key": key,
                    "display": ctx.display,
                    "model_name": ctx.model_name,
                    "provider": ctx.provider,
                    "sub_account": getattr(ctx, "sub_account", None),
                    "ai_model_info": copy.deepcopy(ctx.web_data["ai_model_info"]),
                    "account_summary": copy.deepcopy(ctx.web_data["account_summary"]),
                }
            )
    return statuses


def get_model_snapshot(model_key: str) -> Dict:
    """返回单个模型的完整快照（供前端各页面使用）。"""
    ctx = core.MODEL_CONTEXTS.get(model_key)
    if not ctx:
        raise KeyError(f"未知模型: {model_key}")

    with ctx.lock:
        snapshot = copy.deepcopy(ctx.web_data)
        snapshot["model"] = ctx.key
        snapshot["display"] = ctx.display
        # 将 signal_history 的 deque 等不可序列化结构转为 list
        snapshot["signal_history"] = {symbol: list(records) for symbol, records in ctx.signal_history.items()}
    return snapshot


def resolve_time_range(range_key: str, now: Optional[datetime] = None):
    """将前端传入的范围标识转为起止时间字符串。"""
    now = now or datetime.now()
    if range_key == "all":
        start = datetime(1970, 1, 1)
    else:
        delta = RANGE_PRESETS.get(range_key, RANGE_PRESETS["7d"])
        start = now - delta
    return start.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d %H:%M:%S")


def get_overview_payload(range_key: str = "1d") -> Dict:
    """首页总览：多模型资金曲线、占比等。

    返回结构：
    {
        "range": str,
        "series": Dict[model_key, List[point]],
        "aggregate_series": List[point_by_model],
        "models": Dict[model_key, summary],
        "aggregate": {"timestamp": str, "total_equity": float, "ratios": Dict[model_key, float]},
    }
    """
    start_ts, end_ts = resolve_time_range(range_key)
    series_by_model: Dict[str, List[Dict[str, float]]] = {}
    aggregate_series_map: Dict[str, Dict[str, float]] = {}

    for key in core.MODEL_ORDER:
        data = core.history_store.fetch_balance_range(key, start_ts, end_ts)
        if not data:
            # 如果该范围内无数据，则使用内存中的最后一段
            data = core.MODEL_CONTEXTS[key].balance_history[-200:]
        formatted = [
            {
                "timestamp": item["timestamp"],
                "total_equity": item["total_equity"],
                "available_balance": item["available_balance"],
                "unrealized_pnl": item.get("unrealized_pnl"),
            }
            for item in data
        ]
        series_by_model[key] = formatted

        for point in formatted:
            ts = point["timestamp"]
            bucket = aggregate_series_map.setdefault(ts, {})
            bucket[key] = point["total_equity"]

    aggregate_series: List[Dict[str, float]] = []
    for ts in sorted(aggregate_series_map.keys()):
        entry: Dict[str, float] = {"timestamp": ts}  # type: ignore[assignment]
        for key in core.MODEL_ORDER:
            entry[key] = aggregate_series_map[ts].get(key)  # type: ignore[index]
        aggregate_series.append(entry)

    models_summary: Dict[str, Dict] = {}
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for key in core.MODEL_ORDER:
        ctx = core.MODEL_CONTEXTS[key]
        latest = core.history_store.get_latest_before(key, now_str) or {
            "total_equity": ctx.web_data["account_summary"].get("total_equity", 0),
            "available_balance": ctx.web_data["account_summary"].get("available_balance", 0),
            "unrealized_pnl": ctx.web_data["account_summary"].get("total_unrealized_pnl", 0),
            "timestamp": now_str,
        }

        base = core.history_store.get_latest_before(key, start_ts)
        change_abs = None
        change_pct = None
        if base and base.get("total_equity"):
            change_abs = latest["total_equity"] - base["total_equity"]
            change_pct = change_abs / base["total_equity"] if base["total_equity"] else None

        models_summary[key] = {
            "display": ctx.display,
            "model_name": ctx.model_name,
            "provider": ctx.provider,
            "sub_account": getattr(ctx, "sub_account", None),
            "latest_equity": latest["total_equity"],
            "available_balance": latest.get("available_balance", 0),
            "unrealized_pnl": latest.get("unrealized_pnl", 0),
            "change_abs": change_abs,
            "change_pct": change_pct,
        }

    total_equity = sum(models_summary[key]["latest_equity"] for key in core.MODEL_ORDER)
    model_ratios: Dict[str, float] = {}
    if total_equity:
        for key in core.MODEL_ORDER:
            model_ratios[key] = models_summary[key]["latest_equity"] / total_equity

    return {
        "range": range_key,
        "series": series_by_model,
        "aggregate_series": aggregate_series,
        "models": models_summary,
        "aggregate": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_equity": total_equity,
            "ratios": model_ratios,
        },
    }
