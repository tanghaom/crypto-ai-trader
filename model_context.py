# -*- coding: utf-8 -*-
"""模型运行上下文封装

说明：
- 仅负责初始化并持有：AI 客户端、OKX 交易所客户端、与前端交互所需的 web 状态容器。
- 不进行 .env 加载，不产生对 deepseekok2.py 的反向依赖，避免循环导入。
- 需要的配置通过 config.settings 导入（如 TRADE_CONFIGS）。
"""
import os
import threading
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import ccxt
from openai import OpenAI

from config.settings import TRADE_CONFIGS

__all__ = ["ModelContext"]


class ModelContext:
    """封装单个大模型的运行上下文（AI客户端 + 交易所 + 状态容器）"""

    def __init__(self, key: str, meta: Dict[str, str]):
        self.key = key
        self.display = meta.get("display", key.title())
        self.provider = meta.get("provider", key)
        self.model_name = meta.get("model")
        self.base_url = meta.get("base_url")
        self.ai_client = self._create_ai_client()
        self.exchange = self._create_exchange()
        self.markets: Dict[str, dict] = {}
        try:
            markets = self.exchange.load_markets()
            self.markets = {symbol: markets.get(symbol) for symbol in TRADE_CONFIGS if symbol in markets}
        except Exception as e:
            print(f"⚠️ {self.display} 加载市场信息失败")
            print(f"   错误类型: {type(e).__name__}")
            print(f"   错误信息: {str(e)}")
            if hasattr(e, "response") and e.response:
                print(f'   HTTP状态码: {getattr(e.response, "status_code", "未知")}')
                print(f'   响应内容: {getattr(e.response, "text", "无")[:500]}')
        # 历史与状态容器
        self.signal_history = defaultdict(list)
        self.price_history = defaultdict(list)
        self.position_state = defaultdict(dict)
        self.initial_balance = defaultdict(lambda: None)
        self.initial_total_equity: Optional[float] = None
        self.lock = threading.Lock()
        self.web_data = self._create_web_state()
        self.balance_history: List[Dict[str, float]] = []
        self.start_time = datetime.now()
        self.metrics = {"ai_calls": 0, "signals_generated": 0, "trades_opened": 0, "trades_closed": 0, "ai_errors": 0}

    # ---------- 初始化辅助 ----------
    def _create_ai_client(self) -> OpenAI:
        """根据 provider 创建对应的 OpenAI 兼容客户端"""
        if self.provider == "qwen":
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                raise RuntimeError("缺少 DASHSCOPE_API_KEY，用于初始化 Qwen 模型。")
            return OpenAI(api_key=api_key, base_url=self.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1")

        # 默认 DeepSeek
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("缺少 DEEPSEEK_API_KEY，用于初始化 DeepSeek 模型。")
        return OpenAI(api_key=api_key, base_url=self.base_url or "https://api.deepseek.com/v1")

    def _create_exchange(self) -> ccxt.okx:
        """初始化 OKX 交易所客户端（支持子账户与代理）"""
        suffix = self.key.upper()
        api_key = os.getenv(f"OKX_API_KEY_{suffix}", os.getenv("OKX_API_KEY"))
        secret = os.getenv(f"OKX_SECRET_{suffix}", os.getenv("OKX_SECRET"))
        password = os.getenv(f"OKX_PASSWORD_{suffix}", os.getenv("OKX_PASSWORD"))
        sub_account = os.getenv(f"OKX_SUBACCOUNT_{suffix}")

        # 打印配置加载状态（隐藏敏感信息）
        print(f"[{self.display}] OKX API 配置检查:")
        print(f'   API Key: {"已配置" if api_key else "❌ 未配置"} (前6位: {api_key[:6] if api_key else "无"}...)')
        print(f'   Secret: {"已配置" if secret else "❌ 未配置"} (前6位: {secret[:6] if secret else "无"}...)')
        print(f'   Password: {"已配置" if password else "❌ 未配置"}')
        print(f'   子账户: {sub_account if sub_account else "使用主账户"}')

        if not all([api_key, secret, password]):
            raise RuntimeError(
                f"缺少 {self.display} 的 OKX API 配置，请设置 OKX_API_KEY_{suffix}/OKX_SECRET_{suffix}/OKX_PASSWORD_{suffix}"
            )

        self.sub_account = sub_account

        # 代理配置（如果需要）
        proxy = os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")
        proxies = None
        if proxy:
            proxies = {"http": proxy, "https": proxy}
            print(f"   使用代理: {proxy}")

        client = ccxt.okx(
            {
                "options": {"defaultType": "swap", "defaultSettle": "usdt"},  # USDⓈ 永续
                "apiKey": api_key,
                "secret": secret,
                "password": password,
                "enableRateLimit": True,  # 启用请求限频
                "timeout": 30000,  # 30秒超时
                "proxies": proxies,  # 代理配置
            }
        )

        # 子账户支持
        if sub_account:
            client.headers = client.headers or {}
            client.headers.update({"OK-ACCESS-SUBACCOUNT": sub_account})

        return client

    def _create_web_state(self) -> Dict:
        """初始化前端展示所需的 web 数据结构（按 symbol 分组）"""
        symbol_states = {
            symbol: {
                "account_info": {},
                "current_position": None,
                "current_price": 0,
                "trade_history": [],
                "ai_decisions": [],
                "performance": {
                    "total_profit": 0,
                    "win_rate": 0,
                    "total_trades": 0,
                    "current_leverage": config["leverage_default"],
                    "suggested_leverage": config["leverage_default"],
                    "leverage_history": [],
                    "last_order_value": 0,
                    "last_order_quantity": 0,
                },
                "kline_data": [],
                "profit_curve": [],
                "analysis_records": [],
                "last_update": None,
            }
            for symbol, config in TRADE_CONFIGS.items()
        }

        return {
            "model": self.key,
            "display": self.display,
            "symbols": symbol_states,
            "ai_model_info": {
                "provider": self.provider,
                "model": self.model_name,
                "status": "unknown",
                "last_check": None,
                "error_message": None,
            },
            "account_summary": {
                "total_balance": 0,
                "available_balance": 0,
                "total_equity": 0,
                "total_unrealized_pnl": 0,
            },
            "account_info": {},
            "balance_history": [],
        }
