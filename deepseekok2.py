# -*- coding: utf-8 -*-
from dotenv import load_dotenv

load_dotenv()

import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Optional

import ccxt

from ai_analysis import analyze_with_llm
from config.settings import (
    ARCHIVE_DIR,
    CONFIDENCE_RATIOS,
    DB_PATH,
    DEFAULT_TRADE_SYMBOL,
    ENABLED_MODELS,
    HOLD_TOLERANCE,
    MARGIN_SAFETY_BUFFER,
    MAX_TOTAL_MARGIN_RATIO,
    MODEL_METADATA,
    TRADE_CONFIGS,
)
from history_store import HistoryStore
from market_utils import (
    adjust_contract_quantity,
    base_to_contracts,
    contracts_to_base,
    get_current_position,
    get_symbol_contract_specs,
    get_symbol_min_amount,
    get_symbol_ohlcv_enhanced,
)
from model_context import ModelContext
from prompt_builder import format_currency
from utils import (
    safe_float,
    sleep_interruptible,
    wait_for_next_period,
)

# ==================== 多模型上下文管理 ====================

AI_PROVIDER = "deepseek"
AI_MODEL = "deepseek-chat"
ai_client = None
deepseek_client = None
exchange = None
ACTIVE_CONTEXT: Optional["ModelContext"] = None


@contextmanager
def activate_context(ctx: ModelContext):
    """切换全局变量到指定模型上下文，确保旧函数兼容"""
    global exchange, ai_client, deepseek_client, AI_PROVIDER, AI_MODEL, ACTIVE_CONTEXT
    global signal_history, price_history, position_state, web_data, initial_balance

    prev_exchange = exchange
    prev_ai_client = ai_client
    prev_deepseek_client = deepseek_client
    prev_ai_provider = AI_PROVIDER
    prev_ai_model = AI_MODEL
    prev_signal_history = signal_history
    prev_price_history = price_history
    prev_position_state = position_state
    prev_web_data = web_data
    prev_initial_balance = initial_balance
    prev_active_context = ACTIVE_CONTEXT

    try:
        exchange = ctx.exchange
        ai_client = ctx.ai_client
        deepseek_client = ctx.ai_client
        AI_PROVIDER = ctx.provider
        AI_MODEL = ctx.model_name
        signal_history = ctx.signal_history
        price_history = ctx.price_history
        position_state = ctx.position_state
        web_data = ctx.web_data
        initial_balance = ctx.initial_balance
        ACTIVE_CONTEXT = ctx
        yield
    finally:
        exchange = prev_exchange
        ai_client = prev_ai_client
        deepseek_client = prev_deepseek_client
        AI_PROVIDER = prev_ai_provider
        AI_MODEL = prev_ai_model
        signal_history = prev_signal_history
        price_history = prev_price_history
        position_state = prev_position_state
        web_data = prev_web_data
        initial_balance = prev_initial_balance
        ACTIVE_CONTEXT = prev_active_context


# 多交易对配置 - 移至 config.settings
TRADE_CONFIG = TRADE_CONFIGS[DEFAULT_TRADE_SYMBOL]

# 预置占位容器；实际数据由每个模型上下文维护
price_history = defaultdict(list)
signal_history = defaultdict(list)
position_state = defaultdict(dict)
initial_balance = defaultdict(lambda: None)
web_data: Dict = {}

# 概览状态（首页使用），后续在运行时维护
overview_state = {"series": [], "models": {}, "aggregate": {}}

# 线程锁保护共享数据（跨模型共享）
data_lock = threading.Lock()
order_execution_lock = threading.Lock()

# 交易机器人启停信号（线程安全）
STOP_EVENT = threading.Event()


def request_stop_trading_bot() -> None:
    """
    请求停止交易机器人（置位停止信号）。
    """
    STOP_EVENT.set()


def clear_stop_signal() -> None:
    """
    清除停止信号，便于后续重新启动交易机器人。
    """
    STOP_EVENT.clear()


def is_stop_requested() -> bool:
    """
    返回是否已请求停止。
    """
    return STOP_EVENT.is_set()


# ==================== 模型上下文初始化 ====================

MODEL_CONTEXTS: Dict[str, ModelContext] = {}
for model_key in ENABLED_MODELS:
    if model_key in MODEL_METADATA:
        MODEL_CONTEXTS[model_key] = ModelContext(model_key, MODEL_METADATA[model_key])
    else:
        print(f"⚠️ 未识别的模型标识: {model_key}，已跳过。")

if not MODEL_CONTEXTS:
    raise RuntimeError("未启用任何可用模型，请检查 ENABLED_MODELS 配置。")

MODEL_ORDER = list(MODEL_CONTEXTS.keys())
DEFAULT_MODEL_KEY = MODEL_ORDER[0]
DEFAULT_CONTEXT = MODEL_CONTEXTS[DEFAULT_MODEL_KEY]

# 初始化全局引用，使旧逻辑默认指向第一个模型
ai_client = DEFAULT_CONTEXT.ai_client
deepseek_client = ai_client
exchange = DEFAULT_CONTEXT.exchange
AI_PROVIDER = DEFAULT_CONTEXT.provider
AI_MODEL = DEFAULT_CONTEXT.model_name
price_history = DEFAULT_CONTEXT.price_history
signal_history = DEFAULT_CONTEXT.signal_history
position_state = DEFAULT_CONTEXT.position_state
initial_balance = DEFAULT_CONTEXT.initial_balance
web_data = DEFAULT_CONTEXT.web_data
ACTIVE_CONTEXT = DEFAULT_CONTEXT

# 概览初始状态
overview_state["models"] = {
    key: {
        "display": ctx.display,
        "ai_model_info": ctx.web_data["ai_model_info"],
        "account_summary": ctx.web_data["account_summary"],
        "sub_account": getattr(ctx, "sub_account", None),
    }
    for key, ctx in MODEL_CONTEXTS.items()
}


# ==================== 辅助函数 ====================


def get_symbol_config(symbol: str) -> dict:
    """返回指定交易对的配置字典"""
    return TRADE_CONFIGS.get(symbol, TRADE_CONFIG)


def ensure_symbol_state(symbol: str) -> None:
    """初始化缺失的 web_data / position_state / history 容器"""
    with data_lock:
        if symbol not in web_data["symbols"]:
            config = get_symbol_config(symbol)
            web_data["symbols"][symbol] = {
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
                    "last_order_contracts": 0,
                },
                "kline_data": [],
                "profit_curve": [],
                "last_update": None,
            }


def evaluate_signal_result(signal: str, price_change_pct: float) -> bool:
    signal = (signal or "").upper()
    if signal == "BUY":
        return price_change_pct >= 0
    if signal == "SELL":
        return price_change_pct <= 0
    if signal == "HOLD":
        return abs(price_change_pct) <= HOLD_TOLERANCE
    return False


def update_signal_validation(symbol: str, current_price: float, timestamp: str) -> None:
    ctx = get_active_context()
    history = ctx.signal_history[symbol]
    updated = False
    for record in history:
        if record.get("validation_price") is None and record.get("entry_price"):
            entry_price = record["entry_price"]
            if entry_price:
                change_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                change_pct = 0.0
            record["validation_price"] = current_price
            record["validation_timestamp"] = timestamp
            record["price_change_pct"] = change_pct
            result = evaluate_signal_result(record.get("signal"), change_pct)
            record["result"] = "success" if result else "fail"
            updated = True
    if updated:
        ctx.web_data["symbols"][symbol]["analysis_records"] = history[-100:]


def append_signal_record(symbol: str, signal_data: Dict, entry_price: float, timestamp: Optional[str] = None) -> Dict:
    ctx = get_active_context()
    history = ctx.signal_history[symbol]
    record = {
        "timestamp": timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "signal": (signal_data.get("signal") or "").upper(),
        "confidence": (signal_data.get("confidence") or "MEDIUM").upper(),
        "leverage": int(signal_data.get("leverage", 0)) if signal_data.get("leverage") is not None else None,
        "entry_price": entry_price,
        "validation_price": None,
        "validation_timestamp": None,
        "price_change_pct": None,
        "result": None,
        "reason": signal_data.get("reason"),
        "stop_loss": signal_data.get("stop_loss"),
        "take_profit": signal_data.get("take_profit"),
    }
    history.append(record)
    if len(history) > 200:
        history.pop(0)
    ctx.web_data["symbols"][symbol]["analysis_records"] = list(history[-100:])
    return record


def setup_exchange():
    """设置交易所参数 - 多交易对版本"""
    try:
        # 为所有交易对设置杠杆
        for symbol, config in TRADE_CONFIGS.items():
            try:
                exchange.set_leverage(config["leverage_default"], symbol, {"mgnMode": "cross"})  # 全仓模式
                print(f"✓ {config['display']}: 杠杆 {config['leverage_default']}x")
            except Exception as e:
                print(f"✗ {config['display']}: 杠杆设置失败 - {e}")

        # 获取余额
        balance = exchange.fetch_balance()

        # 解析 OKX 余额结构
        usdt_balance = 0
        total_equity = 0

        # 方法1: 标准格式
        if "USDT" in balance and balance["USDT"]:
            usdt_balance = float(balance["USDT"].get("free", 0) or 0)
            total_equity = float(balance["USDT"].get("total", 0) or 0)

        # 方法2: 从 info.data[0].details 中解析
        elif "info" in balance and "data" in balance["info"]:
            for data_item in balance["info"]["data"]:
                details = data_item.get("details", [])
                for detail in details:
                    if detail.get("ccy") == "USDT":
                        usdt_balance = float(detail.get("availBal", "0") or 0)
                        total_equity = float(detail.get("eq", "0") or 0)
                        break
                if usdt_balance > 0:
                    break

        if usdt_balance <= 0:
            print("⚠️ 警告: 交易账户USDT余额为0")
            print("💡 提示：请从【资金账户】划转USDT到【交易账户】")
            print("💡 OKX网页 → 资产 → 资金划转 → 从资金账户转到交易账户")

        # 更新账户摘要
        with data_lock:
            web_data["account_summary"].update({"total_balance": usdt_balance, "available_balance": usdt_balance, "total_equity": total_equity})

        print(f"\n💰 当前USDT余额: {usdt_balance:.2f}")
        print(f"💰 总权益: {total_equity:.2f}\n")

        return True
    except Exception as e:
        print("❌ 交易所设置失败")
        print(f"   错误类型: {type(e).__name__}")
        print(f"   错误信息: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def capture_balance_snapshot(ctx: ModelContext, timestamp: Optional[str] = None) -> Optional[Dict[str, float]]:
    """抓取并缓存当前账户余额信息"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        balance = exchange.fetch_balance()
        usdt_info = balance.get("USDT") or {}
        available = float(usdt_info.get("free") or usdt_info.get("available", 0) or 0)
        total_equity = float(usdt_info.get("total") or usdt_info.get("equity", 0) or 0)
        unrealized = float(usdt_info.get("unrealizedPnl", 0) or 0)
    except Exception as e:
        print(f"[{ctx.display}] ⚠️ 获取余额失败")
        print(f"   错误类型: {type(e).__name__}")
        print(f"   错误信息: {str(e)}")
        if hasattr(e, "response") and e.response:
            print(f"   HTTP状态码: {getattr(e.response, 'status_code', '未知')}")
            print(f"   响应内容: {getattr(e.response, 'text', '无')[:500]}")
        import traceback

        traceback.print_exc()
        return None

    snapshot = {
        "timestamp": timestamp,
        "available_balance": available,
        "total_equity": total_equity,
        "unrealized_pnl": unrealized,
        "currency": "USDT",
    }

    with data_lock:
        ctx.web_data["account_summary"].update(
            {"total_balance": available, "available_balance": available, "total_equity": total_equity, "total_unrealized_pnl": unrealized}
        )

        ctx.web_data.setdefault("balance_history", []).append(snapshot)
        if len(ctx.web_data["balance_history"]) > 1000:
            ctx.web_data["balance_history"].pop(0)

        ctx.balance_history.append(snapshot)
        if len(ctx.balance_history) > 5000:
            ctx.balance_history.pop(0)

    history_store.append_balance(ctx.key, snapshot)

    return snapshot


def refresh_overview_from_context(ctx: ModelContext):
    """同步单个模型的账户摘要与AI状态到概览数据"""
    overview_state["models"][ctx.key] = {
        "display": ctx.display,
        "ai_model_info": ctx.web_data["ai_model_info"],
        "account_summary": ctx.web_data["account_summary"],
        "sub_account": getattr(ctx, "sub_account", None),
    }


def record_overview_point(timestamp: Optional[str] = None):
    """记录所有模型的总金额，用于首页曲线"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    point = {"timestamp": timestamp}
    total_equity = 0.0

    for key, ctx in MODEL_CONTEXTS.items():
        equity = ctx.web_data["account_summary"].get("total_equity", 0) or 0
        point[key] = float(equity)
        total_equity += equity

    overview_state["series"].append(point)
    if len(overview_state["series"]) > 500:
        overview_state["series"].pop(0)

    ratios = {}
    if total_equity > 0:
        for key in MODEL_CONTEXTS.keys():
            ratios[key] = point[key] / total_equity

    overview_state["aggregate"] = {"timestamp": timestamp, "total_equity": total_equity, "ratios": ratios}


# ==================== 历史数据存储 ====================


# 历史数据存储
history_store = HistoryStore(DB_PATH, ARCHIVE_DIR)

for key in MODEL_ORDER:
    ctx = MODEL_CONTEXTS[key]
    loaded_history = history_store.load_recent_balance(ctx.key, limit=1000)
    if loaded_history:
        ctx.balance_history = loaded_history
        ctx.web_data["balance_history"] = list(loaded_history)
        last_point = loaded_history[-1]
        ctx.web_data["account_summary"].update(
            {
                "total_balance": last_point.get("available_balance", 0),
                "available_balance": last_point.get("available_balance", 0),
                "total_equity": last_point.get("total_equity", 0),
                "total_unrealized_pnl": last_point.get("unrealized_pnl", 0),
            }
        )


def execute_trade(symbol, signal_data, price_data, config):
    """执行交易 - OKX版本（多交易对+动态杠杆+动态资金）"""
    global web_data

    current_position = get_current_position(symbol)
    trade_history = web_data["symbols"][symbol].get("trade_history", [])

    # 🔴 统一的交易保护机制：防止频繁交易和频繁反转
    if signal_data["signal"] not in ["HOLD"] and len(trade_history) >= 1:
        last_trade = trade_history[-1]
        last_trade_time = datetime.strptime(last_trade["timestamp"], "%Y-%m-%d %H:%M:%S")
        time_diff = (datetime.now() - last_trade_time).total_seconds() / 60  # 转为分钟

        # 1. 基础时间间隔保护（适用于所有交易，包括CLOSE）
        if time_diff < 10:  # 10分钟内无条件拒绝
            print(f"[{config['display']}] 🔒 距离上次交易仅{time_diff:.1f}分钟，避免过度频繁交易")
            return
        elif time_diff < 20 and signal_data["confidence"] != "HIGH":  # 10-20分钟内只允许HIGH信心
            print(f"[{config['display']}] 🔒 距离上次交易{time_diff:.1f}分钟，非HIGH信心不执行")
            return

        # 2. 来回反转保护（防止：多→空→多 或 空→多→空）
        # 适用于BUY/SELL信号（CLOSE不会开新仓，不需要此保护）
        if signal_data["signal"] in ["BUY", "SELL"] and current_position:
            current_side = current_position["side"]
            new_side = "long" if signal_data["signal"] == "BUY" else "short"

            # 如果要反转到另一个方向
            if new_side != current_side:
                last_trade_side = last_trade.get("side")
                # 如果上次交易就是这个方向，说明是来回反转（如：多→空→多）
                if last_trade_side == new_side and time_diff < 30:
                    print(f"[{config['display']}] 🔒 {time_diff:.1f}分钟前刚从{new_side}反转出来，避免来回反转")
                    return

    print(f"[{config['display']}] 交易信号: {signal_data.get('signal')}")
    print(f"[{config['display']}] 信心程度: {signal_data.get('confidence')}")
    print(f"[{config['display']}] 理由: {signal_data.get('reason')}")
    print(f"[{config['display']}] 止损: {format_currency(signal_data.get('stop_loss'))}")
    print(f"[{config['display']}] 止盈: {format_currency(signal_data.get('take_profit'))}")
    print(f"[{config['display']}] 当前持仓: {current_position}")

    # 处理CLOSE平仓信号
    if signal_data.get("signal", "").upper() == "CLOSE":
        if not current_position:
            print(f"[{config['display']}] ⚠️ CLOSE信号但无持仓，忽略")
            return

        # CLOSE信号也需要HIGH信心才能执行，避免频繁交易
        if signal_data["confidence"] != "HIGH":
            print(f"[{config['display']}] 🔒 CLOSE信号信心度为{signal_data['confidence']}（需要HIGH），不执行平仓")
            return

        print(f"[{config['display']}] 🔴 执行CLOSE平仓信号 (信心度: HIGH)")

        if config["test_mode"]:
            print(f"[{config['display']}] 测试模式 - 仅模拟平仓")
            return

        # 执行平仓
        try:
            ctx = get_active_context()
            size_contracts = float(current_position.get("size", 0) or 0)
            if size_contracts <= 0:
                print(f"[{config['display']}] ⚠️ 持仓数量为0，无需平仓")
                return

            side = current_position.get("side")
            # 平仓订单方向与持仓方向相反
            order_side = "buy" if side == "short" else "sell"

            print(f"[{config['display']}] 平仓 {side} 仓位: {size_contracts:.6f} 张，订单方向: {order_side.upper()}")

            # 使用市价单平仓，设置reduceOnly确保只平仓不开新仓
            order = ctx.exchange.create_market_order(symbol, order_side, size_contracts, params={"reduceOnly": True})

            print(f"[{config['display']}] ✅ 平仓成功: 订单ID {order.get('id', 'N/A')}")
            ctx.metrics["trades_closed"] += 1

            # 记录平仓信号到历史
            ctx.signal_history[symbol].append(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "signal": "CLOSE",
                    "confidence": signal_data.get("confidence", "MEDIUM"),
                    "reason": signal_data.get("reason", "平仓"),
                    "price": price_data["price"],
                }
            )

            # 记录平仓交易到前端交易历史
            close_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            trade_record = {
                "timestamp": close_timestamp,
                "trade_type": "close_position",  # 平仓类型标识
                "trade_type_display": "平仓",  # 平仓类型中文显示
                "signal": "CLOSE",  # 原始信号
                "side": side,  # 保留原持仓方向
                "price": price_data["price"],
                "amount": 0,  # 平仓不涉及金额
                "contracts": size_contracts,
                "leverage": current_position.get("leverage", 0),
                "confidence": signal_data.get("confidence", "MEDIUM"),
                "reason": signal_data.get("reason", "平仓"),
                "pnl": current_position.get("unrealized_pnl", 0),  # 记录平仓时的盈亏
            }

            with data_lock:
                ctx.web_data["symbols"][symbol]["trade_history"].append(trade_record)
                if len(ctx.web_data["symbols"][symbol]["trade_history"]) > 100:  # 只保留最近100条
                    ctx.web_data["symbols"][symbol]["trade_history"].pop(0)

                # 更新持仓信息为空
                ctx.web_data["symbols"][symbol]["current_position"] = None

        except Exception as e:
            print(f"[{config['display']}] ❌ 平仓失败: {e}")
            import traceback

            traceback.print_exc()

        return

    if signal_data.get("signal", "").upper() == "HOLD":
        print(f"[{config['display']}] ℹ️ HOLD 信号，不执行下单流程")
        return

    # 风险管理：低信心信号不执行
    if signal_data["confidence"] == "LOW" and not config["test_mode"]:
        print(f"[{config['display']}] ⚠️ 低信心信号，跳过执行")
        return

    if config["test_mode"]:
        print(f"[{config['display']}] 测试模式 - 仅模拟交易")
        return

    try:
        # 🔒 获取全局执行锁，防止多个交易对并发下单导致保证金竞争
        with order_execution_lock:
            print(f"[{config['display']}] 🔒 已获取交易执行锁，开始处理...")

            # 📊 获取账户余额
            balance = exchange.fetch_balance()

            # 解析 OKX 特殊的余额结构
            usdt_balance = 0

            # 方法1: 标准格式
            if "USDT" in balance and balance["USDT"]:
                usdt_balance = float(balance["USDT"].get("free", 0) or 0)

            # 方法2: 从 info.data[0].details 中解析
            elif "info" in balance and "data" in balance["info"]:
                for data_item in balance["info"]["data"]:
                    details = data_item.get("details", [])
                    for detail in details:
                        if detail.get("ccy") == "USDT":
                            avail_bal = detail.get("availBal", "0")
                            usdt_balance = float(avail_bal) if avail_bal else 0
                            break
                    if usdt_balance > 0:
                        break

            if usdt_balance <= 0:
                print(f"[{config['display']}] ⚠️ 交易账户USDT余额为0")
                print(f"[{config['display']}] 💡 提示：请先从【资金账户】划转USDT到【交易账户】")
                print(f"[{config['display']}] 💡 操作路径：OKX网页 → 资产 → 资金划转")
                return

            # 获取AI建议的杠杆和数量（确保类型转换）
            suggested_leverage = safe_float(signal_data.get("leverage"), config["leverage_default"])
            order_value = safe_float(signal_data.get("order_value"), 0)
            order_quantity = safe_float(signal_data.get("order_quantity"), 0)

            # 🆕 双重验证机制：智能计算实际可用保证金
            current_price = price_data["price"]

            contract_specs = get_symbol_contract_specs(symbol)
            contract_size = contract_specs["contract_size"]
            min_contracts = contract_specs.get("min_contracts") or 0
            if min_contracts and min_contracts > 0:
                min_contracts = adjust_contract_quantity(symbol, min_contracts, round_up=True)
            min_quantity = contracts_to_base(symbol, min_contracts) if min_contracts else get_symbol_min_amount(symbol)

            # 🔴 关键修复：从OKX balance结构中提取更准确的数据
            try:
                # 尝试从info.details中获取USDT的详细信息
                usdt_details = None
                if "info" in balance and "data" in balance["info"]:
                    for data_item in balance["info"]["data"]:
                        if "details" in data_item:
                            for detail in data_item["details"]:
                                if detail.get("ccy") == "USDT":
                                    usdt_details = detail
                                    break

                if usdt_details:
                    # 使用OKX的实际可用余额和保证金率计算
                    avail_bal = float(usdt_details.get("availBal", usdt_balance))
                    total_eq = float(usdt_details.get("eq", usdt_balance))
                    frozen_bal = float(usdt_details.get("frozenBal", 0))
                    current_imr = float(usdt_details.get("imr", 0))

                    print(f"[{config['display']}] 📊 OKX账户详情:")
                    print(f"[{config['display']}]    - 可用余额: {avail_bal:.2f} USDT")
                    print(f"[{config['display']}]    - 总权益: {total_eq:.2f} USDT")
                    print(f"[{config['display']}]    - 已冻结: {frozen_bal:.2f} USDT")
                    print(f"[{config['display']}]    - 已占用保证金: {current_imr:.2f} USDT")

                    # 🔴 方案B++：智能计算保证金（使用可配置的阈值和缓冲）
                    # 说明：考虑OKX隐藏buffer、手续费、价格波动等因素，使用更保守的参数
                    max_total_imr = total_eq * MAX_TOTAL_MARGIN_RATIO  # 总保证金不超过权益的配置比例（应对OKX梯度保证金制度）
                    max_new_margin = max_total_imr - current_imr  # 可用于新仓位的保证金

                    # 取两者的较小值，并应用安全缓冲（应对价格波动、手续费、OKX buffer）
                    max_usable_margin = min(avail_bal, max_new_margin) * MARGIN_SAFETY_BUFFER

                    print(f"[{config['display']}] 💡 智能计算:")
                    print(f"[{config['display']}]    - 最大允许总保证金: {max_total_imr:.2f} USDT (权益的{MAX_TOTAL_MARGIN_RATIO*100:.0f}%)")
                    print(f"[{config['display']}]    - 可用于新仓位: {max_new_margin:.2f} USDT")
                    print(f"[{config['display']}]    - 最终可用保证金: {max_usable_margin:.2f} USDT (含{MARGIN_SAFETY_BUFFER*100:.0f}%安全缓冲)")
                else:
                    # 降级方案：简单计算
                    max_usable_margin = usdt_balance * 0.35
                    print(f"[{config['display']}] ⚠️ 未找到详细信息，使用简单计算: {max_usable_margin:.2f} USDT")
            except Exception as e:
                # 异常时使用保守策略
                max_usable_margin = usdt_balance * 0.35
                print(f"[{config['display']}] ⚠️ 解析balance失败: {e}，使用保守值: {max_usable_margin:.2f} USDT")

            # 为当前信心等级和杠杆计算有效仓位
            confidence = signal_data.get("confidence", "MEDIUM")
            ratio = CONFIDENCE_RATIOS.get(confidence, 0.10)

            margin_pool = max_usable_margin * ratio
            expected_position_value = margin_pool * suggested_leverage
            expected_quantity = expected_position_value / current_price if current_price else 0
            expected_contracts = base_to_contracts(symbol, expected_quantity)
            expected_contracts = (
                adjust_contract_quantity(symbol, max(expected_contracts, min_contracts), round_up=True) if current_price else min_contracts
            )
            expected_quantity = contracts_to_base(symbol, expected_contracts)

            # 确定交易张数
            if order_quantity > 0:
                trade_contracts = base_to_contracts(symbol, order_quantity)
                trade_amount = contracts_to_base(symbol, trade_contracts)
                lower_bound = expected_quantity * 0.8
                upper_bound = expected_quantity * 1.2
                if expected_quantity > 0 and (trade_amount < lower_bound or trade_amount > upper_bound):
                    print(f"[{config['display']}] ⚠️ AI返回的数量 {trade_amount:.6f} 超出预期范围 [{lower_bound:.6f}, {upper_bound:.6f}]")
                    print(f"[{config['display']}] 🔧 自动调整为标准仓位: {expected_quantity:.6f}")
                    trade_contracts = expected_contracts
            elif order_value > 0:
                raw_quantity = order_value / current_price if current_price else 0
                trade_contracts = base_to_contracts(symbol, raw_quantity)
            else:
                trade_contracts = expected_contracts
                print(f"[{config['display']}] 💡 AI未指定数量，使用标准仓位: {contracts_to_base(symbol, trade_contracts):.6f}")

            if min_contracts and trade_contracts < min_contracts:
                print(f"[{config['display']}] ⚠️ 交易张数 {trade_contracts:.6f} 低于最小张数 {min_contracts:.6f}")
                test_margin = current_price * contracts_to_base(symbol, min_contracts) / suggested_leverage if current_price else 0
                if test_margin <= max_usable_margin:
                    print(f"[{config['display']}] 🔧 调整为最小交易量: {contracts_to_base(symbol, min_contracts):.6f}")
                    trade_contracts = min_contracts
                else:
                    print(f"[{config['display']}] ❌ 即使最小交易量也需要 {test_margin:.2f} USDT保证金，超出可用 {max_usable_margin:.2f} USDT")
                    print(
                        f"[{config['display']}] 💡 建议充值至少: {(contracts_to_base(symbol, min_contracts) * current_price / suggested_leverage):.2f} USDT"
                    )
                    return

            trade_contracts = adjust_contract_quantity(symbol, max(trade_contracts, min_contracts), round_up=True)
            trade_amount = contracts_to_base(symbol, trade_contracts)

            if min_contracts and trade_contracts < min_contracts:
                print(f"[{config['display']}] ❌ 调整到交易精度后张数仍低于最小要求 {min_contracts}")
                return

            # 计算所需保证金（第1次验证）
            required_margin = current_price * trade_amount / suggested_leverage

            if required_margin > max_usable_margin:
                print(f"[{config['display']}] ⚠️ 初步验证：保证金不足")
                print(f"[{config['display']}] 需要: {required_margin:.2f} USDT")
                print(f"[{config['display']}] 可用: {max_usable_margin:.2f} USDT")

                # 🆕 尝试动态调整数量
                adjusted_contracts = base_to_contracts(
                    symbol, (max_usable_margin * 0.95) * suggested_leverage / current_price if current_price else 0
                )
                adjusted_contracts = adjust_contract_quantity(symbol, max(adjusted_contracts, min_contracts), round_up=True)
                adjusted_amount = contracts_to_base(symbol, adjusted_contracts)
                if adjusted_contracts >= min_contracts and adjusted_amount >= min_quantity:
                    print(
                        f"[{config['display']}] 💡 动态调整数量: {trade_amount:.6f} ({trade_contracts:.6f}张) → {adjusted_amount:.6f} ({adjusted_contracts:.6f}张)"
                    )
                    trade_contracts = adjusted_contracts
                    trade_amount = adjusted_amount
                    required_margin = current_price * trade_amount / suggested_leverage
                else:
                    print(f"[{config['display']}] ❌ 即使调整也无法满足最小交易量，跳过")
                    return

            # 显示初步计算结果
            print(f"[{config['display']}] 📊 初步计算参数:")
            print(f"[{config['display']}]    - 数量: {trade_amount:.6f} ({trade_contracts:.6f} 张, 合约面值 {contract_size:g})")
            print(f"[{config['display']}]    - 杠杆: {suggested_leverage}x")
            print(f"[{config['display']}]    - 所需保证金: {required_margin:.2f} USDT")
            print(f"[{config['display']}]    - 仓位价值: ${(current_price * trade_amount):.2f}")
            print(f"[{config['display']}]    - 保证金占用率: {(required_margin / max_usable_margin * 100):.1f}%")

            # ============ 🆕 关键改进：下单前实时验证 ============
            print(f"\n[{config['display']}] 🔄 下单前重新验证余额...")
            time.sleep(0.5)  # 短暂延迟，让其他线程订单生效

            # 📊 第2次余额获取（实时）+ 智能计算
            fresh_balance = exchange.fetch_balance()
            fresh_usdt = fresh_balance["USDT"]["free"]

            # 🔴 关键修复：应用同样的智能保证金计算
            try:
                # 解析OKX详细余额信息
                fresh_usdt_details = None
                if "info" in fresh_balance and "data" in fresh_balance["info"]:
                    for data_item in fresh_balance["info"]["data"]:
                        if "details" in data_item:
                            for detail in data_item["details"]:
                                if detail.get("ccy") == "USDT":
                                    fresh_usdt_details = detail
                                    break

                if fresh_usdt_details:
                    # 使用OKX的实际可用余额和保证金率计算
                    fresh_avail_bal = float(fresh_usdt_details.get("availBal", fresh_usdt))
                    fresh_total_eq = float(fresh_usdt_details.get("eq", fresh_usdt))
                    fresh_current_imr = float(fresh_usdt_details.get("imr", 0))

                    # 🔴 方案B++：智能计算保证金（使用可配置的阈值和缓冲）- 与第一阶段完全一致
                    # 说明：考虑OKX隐藏buffer、手续费、价格波动等因素，使用更保守的参数
                    fresh_max_total_imr = fresh_total_eq * MAX_TOTAL_MARGIN_RATIO  # 总保证金不超过权益的配置比例（应对OKX梯度保证金制度）
                    fresh_max_new_margin = fresh_max_total_imr - fresh_current_imr

                    # 取两者的较小值，并应用安全缓冲（应对价格波动、手续费、OKX buffer）
                    fresh_max_margin = min(fresh_avail_bal, fresh_max_new_margin) * MARGIN_SAFETY_BUFFER

                    print(f"[{config['display']}] 💰 实时余额: {fresh_usdt:.2f} USDT")
                    print(f"[{config['display']}] 💡 实时智能计算:")
                    print(f"[{config['display']}]    - 总权益: {fresh_total_eq:.2f} USDT")
                    print(f"[{config['display']}]    - 已占用保证金: {fresh_current_imr:.2f} USDT")
                    print(f"[{config['display']}]    - 可用于新仓位: {fresh_max_new_margin:.2f} USDT")
                    print(f"[{config['display']}]    - 最终可用保证金: {fresh_max_margin:.2f} USDT (含{MARGIN_SAFETY_BUFFER*100:.0f}%安全缓冲)")
                else:
                    # 降级方案：简单计算
                    fresh_max_margin = fresh_usdt * 0.35
                    print(f"[{config['display']}] 💰 实时余额: {fresh_usdt:.2f} USDT")
                    print(f"[{config['display']}] ⚠️ 未找到详细信息，使用简单计算: {fresh_max_margin:.2f} USDT")
            except Exception as e:
                # 异常时使用保守策略
                fresh_max_margin = fresh_usdt * 0.35
                print(f"[{config['display']}] 💰 实时余额: {fresh_usdt:.2f} USDT")
                print(f"[{config['display']}] ⚠️ 实时解析失败: {e}，使用保守值: {fresh_max_margin:.2f} USDT")

            # 🆕 第2次验证
            if required_margin > fresh_max_margin:
                print(f"[{config['display']}] ❌ 实时验证失败：保证金不足")
                print(f"[{config['display']}] 需要: {required_margin:.2f} USDT")
                print(f"[{config['display']}] 实时: {fresh_max_margin:.2f} USDT")
                print(f"[{config['display']}] 💡 可能其他交易对已占用保证金")

                # 🆕 再次尝试动态调整
                final_adjusted_contracts = base_to_contracts(
                    symbol, (fresh_max_margin * 0.95) * suggested_leverage / current_price if current_price else 0
                )
                final_adjusted_contracts = adjust_contract_quantity(symbol, max(final_adjusted_contracts, min_contracts), round_up=True)
                final_adjusted_amount = contracts_to_base(symbol, final_adjusted_contracts)
                if final_adjusted_contracts >= min_contracts and final_adjusted_amount >= min_quantity:
                    print(
                        f"[{config['display']}] 💡 最终调整数量: {trade_amount:.6f} ({trade_contracts:.6f}张) → {final_adjusted_amount:.6f} ({final_adjusted_contracts:.6f}张)"
                    )
                    trade_contracts = final_adjusted_contracts
                    trade_amount = final_adjusted_amount
                    required_margin = current_price * trade_amount / suggested_leverage
                else:
                    print(f"[{config['display']}] ❌ 无法调整，彻底放弃")
                    return

            print(f"[{config['display']}] ✅ 实时验证通过")
            print(f"[{config['display']}] 📊 最终交易参数:")
            print(f"[{config['display']}]    - 数量: {trade_amount:.6f} ({trade_contracts:.6f} 张)")
            print(f"[{config['display']}]    - 杠杆: {suggested_leverage}x")
            print(f"[{config['display']}]    - 所需保证金: {required_margin:.2f} USDT")

            # 🆕 在验证通过后才设置杠杆（避免验证失败导致的杠杆副作用）
            current_leverage = current_position["leverage"] if current_position else config["leverage_default"]
            if suggested_leverage != current_leverage:
                try:
                    exchange.set_leverage(suggested_leverage, symbol, {"mgnMode": "cross"})
                    print(f"[{config['display']}] ✓ 杠杆已设置为 {suggested_leverage}x")
                except Exception as e:
                    print(f"[{config['display']}] ⚠️ 杠杆设置失败: {e}")
                    # 如果杠杆设置失败，使用当前杠杆重新计算
                    suggested_leverage = current_leverage
                    required_margin = current_price * trade_amount / suggested_leverage
                    print(f"[{config['display']}] 使用当前杠杆 {suggested_leverage}x")

            # ============ 🆕 执行交易（带重试机制） ============
            max_retries = 2
            trade_type = None  # 交易类型：open_long, open_short, add_long, add_short, reverse_long_to_short, reverse_short_to_long
            for attempt in range(max_retries):
                try:
                    print(f"\n[{config['display']}] 📤 执行交易（尝试 {attempt + 1}/{max_retries}）...")

                    # 执行交易逻辑 - tag是经纪商api
                    if signal_data["signal"] == "BUY":
                        if current_position and current_position["side"] == "short":
                            # 平空仓并开多仓（反转）
                            trade_type = "reverse_short_to_long"
                            close_contracts = float(current_position.get("size", 0) or 0)
                            base_token = symbol.split("/")[0]
                            close_amount = contracts_to_base(symbol, close_contracts)
                            print(f"[{config['display']}] 平空仓并开多仓... 平空 {close_contracts:.6f} 张 (~{close_amount:.6f} {base_token})")
                            # 平空仓
                            exchange.create_market_order(symbol, "buy", close_contracts, params={"reduceOnly": True, "tag": "60bb4a8d3416BCDE"})
                            time.sleep(1)
                            # 开多仓
                            exchange.create_market_order(symbol, "buy", trade_contracts, params={"tag": "60bb4a8d3416BCDE"})
                        elif current_position and current_position["side"] == "long":
                            # 🆕 支持加仓：HIGH信心时可以在同方向加仓（需启用开关）
                            if config.get("enable_add_position", False) and signal_data.get("confidence") == "HIGH":
                                current_size = float(current_position.get("size", 0) or 0)
                                # 计算当前仓位价值：合约数量转为基础资产数量，再乘以当前价格
                                current_base_qty = contracts_to_base(symbol, current_size)
                                current_value = current_base_qty * current_price
                                add_value = trade_amount * current_price

                                # 检查仓位上限：总仓位不超过可用保证金的合理范围内（max_usable_margin已包含安全缓冲）
                                # 仓位价值 = 保证金 * 杠杆，所以理论上最大仓位 = max_usable_margin * leverage
                                max_position_value = max_usable_margin * suggested_leverage
                                new_total_value = current_value + add_value

                                if new_total_value <= max_position_value:
                                    trade_type = "add_long"
                                    print(f"[{config['display']}] 📈 HIGH信心加仓机会：当前 {current_size:.6f}张 → 追加 {trade_contracts:.6f}张")
                                    print(f"[{config['display']}]    当前仓位价值: {current_value:.2f} USDT")
                                    print(f"[{config['display']}]    追加仓位价值: {add_value:.2f} USDT")
                                    print(f"[{config['display']}]    总仓位价值: {new_total_value:.2f} USDT")
                                    # 直接加仓（同方向开仓会自动追加）
                                    exchange.create_market_order(symbol, "buy", trade_contracts, params={"tag": "60bb4a8d3416BCDE"})
                                else:
                                    print(f"[{config['display']}] ⚠️ 加仓后超出仓位上限（{new_total_value:.2f} > {max_position_value:.2f}），保持现状")
                            else:
                                if not config.get("enable_add_position", False):
                                    print(f"[{config['display']}] 已有多头持仓，保持现状（加仓功能已禁用）")
                                else:
                                    print(f"[{config['display']}] 已有多头持仓，保持现状（非HIGH信心不加仓）")
                        else:
                            # 无持仓时开多仓
                            trade_type = "open_long"
                            print(f"[{config['display']}] 开多仓...")
                            exchange.create_market_order(symbol, "buy", trade_contracts, params={"tag": "60bb4a8d3416BCDE"})

                    elif signal_data["signal"] == "SELL":
                        if current_position and current_position["side"] == "long":
                            # 平多仓并开空仓（反转）
                            trade_type = "reverse_long_to_short"
                            close_contracts = float(current_position.get("size", 0) or 0)
                            base_token = symbol.split("/")[0]
                            close_amount = contracts_to_base(symbol, close_contracts)
                            print(f"[{config['display']}] 平多仓并开空仓... 平多 {close_contracts:.6f} 张 (~{close_amount:.6f} {base_token})")
                            # 平多仓
                            exchange.create_market_order(symbol, "sell", close_contracts, params={"reduceOnly": True, "tag": "60bb4a8d3416BCDE"})
                            time.sleep(1)
                            # 开空仓
                            exchange.create_market_order(symbol, "sell", trade_contracts, params={"tag": "60bb4a8d3416BCDE"})
                        elif current_position and current_position["side"] == "short":
                            # 🆕 支持加仓：HIGH信心时可以在同方向加仓（需启用开关）
                            if config.get("enable_add_position", False) and signal_data.get("confidence") == "HIGH":
                                current_size = float(current_position.get("size", 0) or 0)
                                # 计算当前仓位价值：合约数量转为基础资产数量，再乘以当前价格
                                current_base_qty = contracts_to_base(symbol, current_size)
                                current_value = current_base_qty * current_price
                                add_value = trade_amount * current_price

                                # 检查仓位上限：总仓位不超过可用保证金的合理范围内（max_usable_margin已包含安全缓冲）
                                # 仓位价值 = 保证金 * 杠杆，所以理论上最大仓位 = max_usable_margin * leverage
                                max_position_value = max_usable_margin * suggested_leverage
                                new_total_value = current_value + add_value

                                if new_total_value <= max_position_value:
                                    trade_type = "add_short"
                                    print(f"[{config['display']}] 📈 HIGH信心加仓机会：当前 {current_size:.6f}张 → 追加 {trade_contracts:.6f}张")
                                    print(f"[{config['display']}]    当前仓位价值: {current_value:.2f} USDT")
                                    print(f"[{config['display']}]    追加仓位价值: {add_value:.2f} USDT")
                                    print(f"[{config['display']}]    总仓位价值: {new_total_value:.2f} USDT")
                                    # 直接加仓（同方向开仓会自动追加）
                                    exchange.create_market_order(symbol, "sell", trade_contracts, params={"tag": "60bb4a8d3416BCDE"})
                                else:
                                    print(f"[{config['display']}] ⚠️ 加仓后超出仓位上限（{new_total_value:.2f} > {max_position_value:.2f}），保持现状")
                            else:
                                if not config.get("enable_add_position", False):
                                    print(f"[{config['display']}] 已有空头持仓，保持现状（加仓功能已禁用）")
                                else:
                                    print(f"[{config['display']}] 已有空头持仓，保持现状（非HIGH信心不加仓）")
                        else:
                            # 无持仓时开空仓
                            trade_type = "open_short"
                            print(f"[{config['display']}] 开空仓...")
                            exchange.create_market_order(symbol, "sell", trade_contracts, params={"tag": "60bb4a8d3416BCDE"})

                    print(f"[{config['display']}] ✓ 订单执行成功")
                    break  # 成功则跳出重试循环

                except ccxt.InsufficientFunds as e:
                    # 🆕 捕获51008保证金不足错误
                    print(f"[{config['display']}] ❌ 保证金不足错误: {e}")

                    if attempt < max_retries - 1:
                        # 还有重试机会，尝试减少50%数量
                        print(f"[{config['display']}] 💡 尝试减少50%数量重试...")
                        trade_contracts = adjust_contract_quantity(symbol, trade_contracts * 0.5, round_up=True)
                        trade_amount = contracts_to_base(symbol, trade_contracts)
                        if min_contracts and trade_contracts < min_contracts:
                            print(f"[{config['display']}] ❌ 减少后仍低于最小张数{min_contracts}，放弃")
                            return
                        required_margin = current_price * trade_amount / suggested_leverage
                        print(f"[{config['display']}] 新数量: {trade_amount:.6f} ({trade_contracts:.6f}张), 新保证金: {required_margin:.2f} USDT")
                        time.sleep(1)  # 等待1秒后重试
                    else:
                        print(f"[{config['display']}] ❌ 重试次数已用完，彻底放弃")
                        return

                except Exception as e:
                    print(f"[{config['display']}] ❌ 订单执行失败: {e}")
                    if attempt < max_retries - 1:
                        print(f"[{config['display']}] 等待2秒后重试...")
                        time.sleep(2)
                    else:
                        import traceback

                        traceback.print_exc()
                        return

            # 等待订单完全生效
            time.sleep(2)

            # 更新持仓信息
            updated_position = get_current_position(symbol)
            print(f"[{config['display']}] 更新后持仓: {updated_position}")
            ctx = get_active_context()
            if current_position and not updated_position:
                ctx.metrics["trades_closed"] += 1
            elif not current_position and updated_position:
                ctx.metrics["trades_opened"] += 1

            # 记录交易历史（仅在实际执行交易时记录，使用线程锁保护）
            if trade_type is not None:  # 只有实际执行了交易才记录
                # 交易类型的中文描述
                trade_type_display = {
                    "open_long": "开多仓",
                    "open_short": "开空仓",
                    "add_long": "加多仓",
                    "add_short": "加空仓",
                    "reverse_long_to_short": "反转（平多→开空）",
                    "reverse_short_to_long": "反转（平空→开多）",
                }.get(trade_type, trade_type)

                trade_record = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "trade_type": trade_type,  # 交易类型标识
                    "trade_type_display": trade_type_display,  # 交易类型中文显示
                    "signal": signal_data["signal"],  # 原始信号（保留用于兼容）
                    "side": "long"
                    if trade_type in ["open_long", "add_long", "reverse_short_to_long"]
                    else "short"
                    if trade_type in ["open_short", "add_short", "reverse_long_to_short"]
                    else "neutral",
                    "price": price_data["price"],
                    "amount": trade_amount,
                    "contracts": trade_contracts,
                    "leverage": suggested_leverage,
                    "confidence": signal_data["confidence"],
                    "reason": signal_data.get("reason", ""),
                }

                with data_lock:
                    web_data["symbols"][symbol]["trade_history"].append(trade_record)
                    if len(web_data["symbols"][symbol]["trade_history"]) > 100:  # 只保留最近100条
                        web_data["symbols"][symbol]["trade_history"].pop(0)

                    # 更新持仓信息
                    web_data["symbols"][symbol]["current_position"] = updated_position

                    # 更新杠杆记录
                    web_data["symbols"][symbol]["performance"]["current_leverage"] = suggested_leverage
                    web_data["symbols"][symbol]["performance"]["suggested_leverage"] = suggested_leverage
                    web_data["symbols"][symbol]["performance"]["last_order_value"] = price_data["price"] * trade_amount
                    web_data["symbols"][symbol]["performance"]["last_order_quantity"] = trade_amount
                    web_data["symbols"][symbol]["performance"]["last_order_contracts"] = trade_contracts

            print(f"[{config['display']}] 🔓 释放交易执行锁")
            # with块结束，自动释放order_execution_lock

    except Exception as e:
        print(f"[{config['display']}] ❌ 订单执行失败: {e}")
        import traceback

        traceback.print_exc()


def check_stop_loss_take_profit(symbol, current_price, config):
    """
    检查当前持仓是否触发止盈止损

    Args:
        symbol: 交易对符号
        current_price: 当前价格
        config: 交易配置

    Returns:
        dict: 包含是否需要平仓及原因的字典
              {'should_close': bool, 'reason': str, 'trigger_type': str}
    """
    ctx = get_active_context()
    current_position = get_current_position(symbol)

    if not current_position:
        return {"should_close": False, "reason": "无持仓", "trigger_type": None}

    # 获取持仓信息
    entry_price = safe_float(current_position.get("entry_price"), 0)
    side = current_position.get("side")  # 'long' or 'short'

    if not entry_price or not side:
        return {"should_close": False, "reason": "持仓信息不完整", "trigger_type": None}

    # 尝试从signal_history获取止盈止损价格
    stop_loss = None
    take_profit = None

    if symbol in ctx.signal_history and len(ctx.signal_history[symbol]) > 0:
        # 获取最近一次与当前持仓方向匹配的开仓信号的止盈止损
        for sig in reversed(ctx.signal_history[symbol]):
            sig_signal = sig.get("signal")
            # 只匹配与当前持仓方向一致的信号
            # 多头持仓 -> 只看BUY信号，空头持仓 -> 只看SELL信号
            if (side == "long" and sig_signal == "BUY") or (side == "short" and sig_signal == "SELL"):
                stop_loss = safe_float(sig.get("stop_loss"), 0)
                take_profit = safe_float(sig.get("take_profit"), 0)
                if stop_loss or take_profit:
                    break

    # 如果没有找到，使用默认止盈止损比例
    if not stop_loss or not take_profit:
        if side == "long":
            stop_loss = entry_price * 0.95  # 默认5%止损
            take_profit = entry_price * 1.05  # 默认5%止盈
        else:  # short
            stop_loss = entry_price * 1.05  # 默认5%止损
            take_profit = entry_price * 0.95  # 默认5%止盈

    # 计算当前盈亏百分比
    if side == "long":
        pnl_percent = ((current_price - entry_price) / entry_price) * 100
        # 多头：价格跌破止损或突破止盈
        if current_price <= stop_loss:
            return {
                "should_close": True,
                "reason": f"触发止损 (入场: ${entry_price:.2f}, 当前: ${current_price:.2f}, 止损: ${stop_loss:.2f}, 亏损: {pnl_percent:.2f}%)",
                "trigger_type": "stop_loss",
                "pnl_percent": pnl_percent,
            }
        elif current_price >= take_profit:
            return {
                "should_close": True,
                "reason": f"触发止盈 (入场: ${entry_price:.2f}, 当前: ${current_price:.2f}, 止盈: ${take_profit:.2f}, 盈利: {pnl_percent:.2f}%)",
                "trigger_type": "take_profit",
                "pnl_percent": pnl_percent,
            }
    else:  # short
        pnl_percent = ((entry_price - current_price) / entry_price) * 100
        # 空头：价格突破止损或跌破止盈
        if current_price >= stop_loss:
            return {
                "should_close": True,
                "reason": f"触发止损 (入场: ${entry_price:.2f}, 当前: ${current_price:.2f}, 止损: ${stop_loss:.2f}, 亏损: {pnl_percent:.2f}%)",
                "trigger_type": "stop_loss",
                "pnl_percent": pnl_percent,
            }
        elif current_price <= take_profit:
            return {
                "should_close": True,
                "reason": f"触发止盈 (入场: ${entry_price:.2f}, 当前: ${current_price:.2f}, 止盈: ${take_profit:.2f}, 盈利: {pnl_percent:.2f}%)",
                "trigger_type": "take_profit",
                "pnl_percent": pnl_percent,
            }

    # 未触发止盈止损
    return {
        "should_close": False,
        "reason": f"持仓中 (入场: ${entry_price:.2f}, 当前: ${current_price:.2f}, 止损: ${stop_loss:.2f}, 止盈: ${take_profit:.2f})",
        "trigger_type": None,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
    }


def run_symbol_cycle(symbol, config):
    """单个交易对的完整执行周期"""
    get_active_context()
    try:
        ensure_symbol_state(symbol)

        print(f"\n[{config['display']}] {'='*50}")
        print(f"[{config['display']}] 执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 1. 获取K线数据
        price_data = get_symbol_ohlcv_enhanced(symbol, config)
        if not price_data:
            print(f"[{config['display']}] ❌ 获取数据失败，跳过")
            return

        print(f"[{config['display']}] 当前价格: ${price_data['price']:,.2f} ({price_data['price_change']:+.2f}%)")

        # 1.5. 检查止盈止损（优先级最高）
        stop_check = check_stop_loss_take_profit(symbol, price_data["price"], config)
        if stop_check["should_close"]:
            print(f"[{config['display']}] 🚨 {stop_check['reason']}")

            # 创建强制平仓信号
            forced_close_signal = {
                "signal": "CLOSE",
                "confidence": "HIGH",
                "reason": stop_check["reason"],
                "stop_loss": 0,
                "take_profit": 0,
                "leverage": config["leverage_default"],
                "order_quantity": 0,
                "is_forced_close": True,
                "trigger_type": stop_check.get("trigger_type", "unknown"),
            }

            # 直接执行平仓，跳过AI分析
            execute_trade(symbol, forced_close_signal, price_data, config)
            print(f"[{config['display']}] ✓ 止盈止损处理完成")
            return
        else:
            # 输出当前持仓状态
            if stop_check.get("stop_loss") and stop_check.get("take_profit"):
                print(f"[{config['display']}] 💡 {stop_check['reason']}")

        # 2. AI分析
        signal_data = analyze_with_llm(symbol, price_data, config)

        # 3. 更新Web数据
        with data_lock:
            # 更新持仓信息
            current_position = get_current_position(symbol)
            web_data["symbols"][symbol].update(
                {
                    "current_price": price_data["price"],
                    "current_position": current_position,
                    "kline_data": price_data["kline_data"],
                    "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            # 更新价格变化百分比到performance
            if "performance" in web_data["symbols"][symbol]:
                web_data["symbols"][symbol]["performance"]["price_change"] = price_data.get("price_change", 0)

            # 保存AI决策
            ai_decision = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "signal": signal_data["signal"],
                "confidence": signal_data["confidence"],
                "reason": signal_data["reason"],
                "stop_loss": safe_float(signal_data.get("stop_loss"), 0),
                "take_profit": safe_float(signal_data.get("take_profit"), 0),
                "leverage": safe_float(signal_data.get("leverage"), config["leverage_default"]),
                "order_value": safe_float(signal_data.get("order_value"), 0),
                "order_quantity": safe_float(signal_data.get("order_quantity"), 0),
                "price": price_data["price"],
            }
            web_data["symbols"][symbol]["ai_decisions"].append(ai_decision)
            if len(web_data["symbols"][symbol]["ai_decisions"]) > 50:
                web_data["symbols"][symbol]["ai_decisions"].pop(0)

        # 🛑 调试断点：分析完成后直接返回，避免进入实际下单
        # input("即将进入下单流程，按回车继续")

        # 4. 执行交易
        execute_trade(symbol, signal_data, price_data, config)

        print(f"[{config['display']}] ✓ 周期完成")

    except Exception as e:
        print(f"[{config.get('display', symbol)}] ❌ 执行失败: {e}")
        import traceback

        traceback.print_exc()


def run_all_symbols_parallel(model_display: str):
    """并行执行所有交易对（针对单个模型上下文）"""
    print("\n" + "=" * 70)
    print(f"🚀 [{model_display}] 开始新一轮分析 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 使用线程池并行执行
    with ThreadPoolExecutor(max_workers=len(TRADE_CONFIGS)) as executor:
        futures = []
        for symbol, config in TRADE_CONFIGS.items():
            # 在提交任务阶段检查停止信号
            if STOP_EVENT.is_set():
                print(f"🛑 [{model_display}] 停止信号触发，终止任务提交。")
                break
            future = executor.submit(run_symbol_cycle, symbol, config)
            futures.append((symbol, future))

            # 添加延迟避免API限频
            time.sleep(2)

        # 等待所有任务完成（或停止）
        for symbol, future in futures:
            if STOP_EVENT.is_set():
                print(f"🛑 [{model_display}] 停止信号触发，跳过剩余任务等待。")
                break
            try:
                future.result(timeout=60)  # 60秒超时
            except Exception as e:
                print(f"[{model_display} | {TRADE_CONFIGS[symbol]['display']}] ⚠️ 任务异常: {e}")

    print("\n" + "=" * 70)
    print(f"✓ [{model_display}] 本轮分析完成")
    print("=" * 70 + "\n")


def main():
    """主入口：同时调度多模型、多交易对"""
    print("\n" + "=" * 70)
    print("🧠 多交易对自动交易机器人启动")
    print("=" * 70)
    print(f"启用模型: {', '.join([MODEL_CONTEXTS[key].display for key in MODEL_ORDER])}")
    print(f"交易对数量: {len(TRADE_CONFIGS)}")
    print(f"交易对列表: {', '.join([c['display'] for c in TRADE_CONFIGS.values()])}")
    print("=" * 70 + "\n")

    test_mode_count = sum(1 for c in TRADE_CONFIGS.values() if c.get("test_mode", True))
    if test_mode_count > 0:
        print(f"⚠️  {test_mode_count}/{len(TRADE_CONFIGS)} 个交易对处于测试模式")
    else:
        print("🔴 实盘交易模式 - 请谨慎操作！")

    print("\n初始化各模型的 OKX 账户...")
    for model_key in MODEL_ORDER:
        ctx = MODEL_CONTEXTS[model_key]
        sub_account = getattr(ctx, "sub_account", None) or "主账户"
        print(f"\n[{ctx.display}] 绑定子账户: {sub_account}")
        with activate_context(ctx):
            if not setup_exchange():
                print(f"❌ {ctx.display} 交易所初始化失败，程序退出")
                return
            capture_balance_snapshot(ctx)
            refresh_overview_from_context(ctx)
        print(f"✓ {ctx.display} 交易所配置完成")

    print("\n系统参数：")
    print("- 执行模式: 每模型并行交易对")
    print("- 执行频率: 每5分钟整点 (00,05,10,15,20,25,30,35,40,45,50,55)")
    print("- API防限频延迟: 2秒/交易对\n")

    record_overview_point(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    while True:
        # 在循环开头检查停止信号
        if STOP_EVENT.is_set():
            print("🛑 收到停止信号，退出交易循环。")
            break

        wait_seconds = wait_for_next_period()
        if wait_seconds > 0:
            # 可中断等待到整点
            sleep_interruptible(wait_seconds, STOP_EVENT)
            if STOP_EVENT.is_set():
                print("🛑 停止信号触发于等待阶段，退出交易循环。")
                break

        cycle_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for model_key in MODEL_ORDER:
            if STOP_EVENT.is_set():
                print("🛑 停止信号触发于模型处理阶段，退出交易循环。")
                break
            ctx = MODEL_CONTEXTS[model_key]
            with activate_context(ctx):
                run_all_symbols_parallel(ctx.display)
                capture_balance_snapshot(ctx, cycle_timestamp)
                refresh_overview_from_context(ctx)

        if STOP_EVENT.is_set():
            break

        record_overview_point(cycle_timestamp)
        history_store.compress_if_needed(datetime.now())
        # 末尾休眠可被停止信号打断
        sleep_interruptible(60, STOP_EVENT)


def get_active_context() -> ModelContext:
    if ACTIVE_CONTEXT is None:
        raise RuntimeError("当前没有激活的模型上下文。")
    return ACTIVE_CONTEXT


if __name__ == "__main__":
    main()
