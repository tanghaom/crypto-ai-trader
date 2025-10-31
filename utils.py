# -*- coding: utf-8 -*-
"""
通用工具函数集合：
- 数值夹取、步长取整
- 安全类型转换
- 等待到下一个执行周期
- 可中断睡眠（通过可选的 threading.Event 实现）

注意：函数尽量保持纯粹，避免对业务全局变量的硬依赖。
"""
from __future__ import annotations

import os
import time
from datetime import datetime
from threading import Event
from typing import Optional


def clamp_value(value, min_val, max_val):
    """将 value 限制在 [min_val, max_val] 区间内。
    示例：clamp_value(12, 0, 10) -> 10
    """
    return max(min_val, min(value, max_val))


def round_to_step(value: float, step: float) -> float:
    """按给定步长进行四舍五入。
    例如：value=1.234, step=0.01 -> 1.23 或 1.24（标准四舍五入）
    """
    if not step:
        return value
    return round(value / step) * step


def safe_float(value, default: float = 0.0) -> float:
    """安全地将值转换为浮点数，失败时返回默认值。
    - None、空字符串、非法字符串等都会回退到 default。
    """
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def wait_for_next_period() -> int:
    """计算并打印距离下一个整周期（默认 5 分钟为单位）的等待秒数，并返回该秒数。
    读取环境变量 TRADE_INTERVAL_MINUTES，非法值回退为 5。
    """
    now = datetime.now()
    current_minute = now.minute
    current_second = now.second

    # 计算下一个整点时间（每N分钟：00, 05, 10, ...）
    try:
        interval = int(os.getenv("TRADE_INTERVAL_MINUTES", "5"))
    except Exception:
        interval = 5
    if interval <= 0:
        interval = 5

    next_period_minute = ((current_minute // interval) + 1) * interval
    if next_period_minute == 60:
        next_period_minute = 0

    # 需要等待的总秒数
    if next_period_minute > current_minute:
        minutes_to_wait = next_period_minute - current_minute
    else:
        minutes_to_wait = 60 - current_minute + next_period_minute

    seconds_to_wait = minutes_to_wait * 60 - current_second

    # 友好显示
    display_minutes = minutes_to_wait - 1 if current_second > 0 else minutes_to_wait
    display_seconds = 60 - current_second if current_second > 0 else 0

    if display_minutes > 0:
        print(f"🕒 等待 {display_minutes} 分 {display_seconds} 秒到整点...")
    else:
        print(f"🕒 等待 {display_seconds} 秒到整点...")

    return max(0, seconds_to_wait)


def sleep_interruptible(total_seconds: int, stop_event: Optional[Event] = None) -> None:
    """按秒睡眠，并在每秒检查一次 stop_event（若提供）。
    收到停止信号时提前返回。

    参数：
        total_seconds: 计划睡眠的秒数
        stop_event: 可选的 threading.Event；若为 None，则不检查中断
    """
    try:
        total_seconds = int(total_seconds)
    except Exception:
        total_seconds = 0

    for _ in range(max(0, total_seconds)):
        if stop_event is not None and stop_event.is_set():
            break
        time.sleep(1)
