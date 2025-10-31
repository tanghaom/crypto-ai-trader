# -*- coding: utf-8 -*-
"""
集中化配置：交易参数、风险参数、模型与路径等。
备注：请通过环境变量覆盖敏感项（API Key 等）。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

# =============== 路径相关（使用 pathlib） ===============
BASE_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = BASE_DIR / "data"
ARCHIVE_DIR: Path = BASE_DIR / "archives"
DB_PATH: Path = DATA_DIR / "history.db"

# 确保目录存在（在导入时即可安全使用）
DATA_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

# =============== 风险与仓位参数 ===============
HOLD_TOLERANCE: float = 0.5  # HOLD 信号允许的价差百分比

# 根据信心等级分配可用保证金的百分比
CONFIDENCE_RATIOS: Dict[str, float] = {
    "HIGH": 0.30,
    "MEDIUM": 0.20,
    "LOW": 0.05,
}

# 保证金管理（风险管理配置，非交易所限制）
MAX_TOTAL_MARGIN_RATIO: float = 0.85  # 总保证金不超过权益的比例
MARGIN_SAFETY_BUFFER: float = 0.90  # 安全缓冲比例

# =============== 模型配置 ===============
MODEL_METADATA: Dict[str, Dict[str, str]] = {
    "deepseek": {
        "display": "DeepSeek 策略",
        "provider": "deepseek",
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    },
    "qwen": {
        "display": "Qwen 策略",
        "provider": "qwen",
        "model": os.getenv("QWEN_MODEL", "qwen-max"),
        "base_url": os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    },
}

ENABLED_MODELS = [m.strip().lower() for m in os.getenv("ENABLED_MODELS", "deepseek,qwen").split(",") if m.strip()]

# =============== 交易对配置 ===============
TRADE_CONFIGS: Dict[str, Dict] = {
    # "BTC/USDT:USDT": {
    #     "display": "BTC-USDT",
    #     "amount": 0.0001,
    #     "leverage": 2,
    #     "leverage_min": 1,
    #     "leverage_max": 3,
    #     "leverage_default": 2,
    #     "leverage_step": 1,
    #     "timeframe": "5m",
    #     "test_mode": False,
    #     "data_points": 96,
    #     "analysis_periods": {"short_term": 20, "medium_term": 50, "long_term": 96},
    # },
    "ETH/USDT:USDT": {
        "display": "ETH-USDT",
        "amount": 0.001,
        "leverage": 2,
        "leverage_min": 1,
        "leverage_max": 3,
        "leverage_default": 2,
        "leverage_step": 1,
        "timeframe": "5m",
        "test_mode": False,
        "enable_add_position": False,  # 同方向信号不加仓
        "data_points": 96,
        "analysis_periods": {"short_term": 20, "medium_term": 50, "long_term": 96},
    },
    # 'OKB/USDT:USDT': {
    #     'display': 'OKB-USDT',
    #     'amount': 1,
    #     'leverage': 3,
    #     'leverage_min': 2,
    #     'leverage_max': 5,
    #     'leverage_default': 3,
    #     'leverage_step': 1,
    #     'timeframe': '5m',
    #     'test_mode': False,
    #     'data_points': 96,
    #     'analysis_periods': {
    #         'short_term': 20,
    #         'medium_term': 50,
    #         'long_term': 96
    #     }
    # },
    # 'SOL/USDT:USDT': {
    #     'display': 'SOL-USDT',
    #     'amount': 0.1,
    #     'leverage': 3,
    #     'leverage_min': 2,
    #     'leverage_max': 5,
    #     'leverage_default': 3,
    #     'leverage_step': 1,
    #     'timeframe': '5m',
    #     'test_mode': False,
    #     'data_points': 96,
    #     'analysis_periods': {
    #         'short_term': 20,
    #         'medium_term': 50,
    #         'long_term': 96
    #     }
    # },
    # 'DOGE/USDT:USDT': {
    #     'display': 'DOGE-USDT',
    #     'amount': 10,
    #     'leverage': 3,
    #     'leverage_min': 2,
    #     'leverage_max': 5,
    #     'leverage_default': 3,
    #     'leverage_step': 1,
    #     'timeframe': '5m',
    #     'test_mode': False,
    #     'data_points': 96,
    #     'analysis_periods': {
    #         'short_term': 20,
    #         'medium_term': 50,
    #         'long_term': 96
    #     }
    # },
    # 'XRP/USDT:USDT': {
    #     'display': 'XRP-USDT',
    #     'amount': 10,
    #     'leverage': 3,
    #     'leverage_min': 2,
    #     'leverage_max': 5,
    #     'leverage_default': 3,
    #     'leverage_step': 1,
    #     'timeframe': '5m',
    #     'test_mode': False,
    #     'data_points': 96,
    #     'analysis_periods': {
    #         'short_term': 20,
    #         'medium_term': 50,
    #         'long_term': 96
    #     }
    # }
}

# 单交易对兼容（保留旧接口需求）
DEFAULT_TRADE_SYMBOL: str = "ETH/USDT:USDT"
