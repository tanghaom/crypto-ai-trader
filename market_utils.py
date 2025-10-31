# -*- coding: utf-8 -*-
"""
市场数据与技术指标工具函数集合。
注意：为避免循环依赖，本模块在函数内部按需懒加载 deepseekok2.exchange 与 deepseekok2.get_symbol_config。
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd
import requests


def _get_exchange():
    """获取全局交易所对象（懒加载 deepseekok2.exchange）。"""
    try:
        from deepseekok2 import exchange  # 延迟导入避免循环依赖

        return exchange
    except Exception:
        return None


def _get_symbol_config(symbol: str) -> Dict:
    """获取交易对配置（懒加载 deepseekok2.get_symbol_config）。"""
    try:
        from deepseekok2 import get_symbol_config  # 延迟导入避免循环依赖

        return get_symbol_config(symbol)
    except Exception:
        return {}


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标 - 增强版（包含 ATR、EMA、多周期 RSI、布林带、成交量比等）。"""
    try:
        # 移动平均线 (SMA)
        df["sma_5"] = df["close"].rolling(window=5, min_periods=1).mean()
        df["sma_20"] = df["close"].rolling(window=20, min_periods=1).mean()
        df["sma_50"] = df["close"].rolling(window=50, min_periods=1).mean()

        # 指数移动平均线 (EMA)
        df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
        df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

        # MACD
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # 相对强弱指数 (RSI) - 14 周期
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # 相对强弱指数 (RSI) - 7 周期
        gain_7 = (delta.where(delta > 0, 0)).rolling(7).mean()
        loss_7 = (-delta.where(delta < 0, 0)).rolling(7).mean()
        rs_7 = gain_7 / loss_7
        df["rsi_7"] = 100 - (100 / (1 + rs_7))

        # 布林带
        df["bb_middle"] = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
        df["bb_lower"] = df["bb_middle"] - (bb_std * 2)
        # 价格在布林带中的相对位置（0-1）
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # ATR (Average True Range)
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = true_range.rolling(14).mean()
        df["atr_3"] = true_range.rolling(3).mean()  # 短期波动率

        # 成交量指标
        df["volume_ma"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"]

        # 静态支撑/阻力（近 20 根）
        df["resistance"] = df["high"].rolling(20).max()
        df["support"] = df["low"].rolling(20).min()

        # 填充 NaN 值
        df = df.bfill().ffill()
        return df
    except Exception as e:
        print(f"技术指标计算失败: {e}")
        return df


def get_support_resistance_levels(df: pd.DataFrame, lookback: int = 20) -> Dict[str, float]:
    """计算支撑阻力位（含静态与布林带动态）。"""
    try:
        recent_high = df["high"].tail(lookback).max()
        recent_low = df["low"].tail(lookback).min()
        current_price = df["close"].iloc[-1]

        resistance_level = recent_high
        support_level = recent_low

        # 动态支撑阻力（基于布林带）
        bb_upper = df["bb_upper"].iloc[-1]
        bb_lower = df["bb_lower"].iloc[-1]

        return {
            "static_resistance": float(resistance_level),
            "static_support": float(support_level),
            "dynamic_resistance": float(bb_upper),
            "dynamic_support": float(bb_lower),
            "price_vs_resistance": float(((resistance_level - current_price) / current_price) * 100),
            "price_vs_support": float(((current_price - support_level) / support_level) * 100),
        }
    except Exception as e:
        print(f"支撑阻力计算失败: {e}")
        return {}


def get_sentiment_indicators(token: str = "BTC") -> Optional[Dict[str, float]]:
    """获取情绪指标 - 支持多币种版本。

    参数:
        token: 币种代码，如 "BTC", "ETH", "SOL" 等。
    """
    try:
        API_URL = "https://service.cryptoracle.network/openapi/v2/endpoint"
        API_KEY = "b54bcf4d-1bca-4e8e-9a24-22ff2c3d76d5"

        # 获取最近 4 小时数据
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=4)

        request_body = {
            "apiKey": API_KEY,
            "endpoints": ["CO-A-02-01", "CO-A-02-02"],  # 核心指标
            "startTime": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "endTime": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "timeType": "15m",
            "token": [token],
        }
        headers = {"Content-Type": "application/json", "X-API-KEY": API_KEY}
        response = requests.post(API_URL, json=request_body, headers=headers)

        if response.status_code == 200:
            data = response.json()
            if data.get("code") == 200 and data.get("data"):
                time_periods = data["data"][0]["timePeriods"]

                for period in time_periods:  # 取第一个有效时间段
                    period_data = period.get("data", [])
                    sentiment: Dict[str, float] = {}
                    valid_data_found = False

                    for item in period_data:
                        endpoint = item.get("endpoint")
                        value = (item.get("value", "") or "").strip()
                        if not value:
                            continue
                        try:
                            if endpoint in ["CO-A-02-01", "CO-A-02-02"]:
                                sentiment[endpoint] = float(value)
                                valid_data_found = True
                        except (ValueError, TypeError):
                            continue

                    if valid_data_found and "CO-A-02-01" in sentiment and "CO-A-02-02" in sentiment:
                        positive = sentiment["CO-A-02-01"]
                        negative = sentiment["CO-A-02-02"]
                        net_sentiment = positive - negative
                        data_delay = int((datetime.now() - datetime.strptime(period["startTime"], "%Y-%m-%d %H:%M:%S")).total_seconds() // 60)
                        print(f"✅ 使用情绪数据时间: {period['startTime']} (延迟: {data_delay}分钟)")
                        return {
                            "positive_ratio": positive,
                            "negative_ratio": negative,
                            "net_sentiment": net_sentiment,
                            "data_time": period["startTime"],
                            "data_delay_minutes": data_delay,
                        }

                print("❌ 所有时间段数据都为空")
                return None
        return None
    except Exception as e:
        print(f"情绪指标获取失败: {e}")
        return None


def get_market_trend(df: pd.DataFrame) -> Dict[str, object]:
    """基于多时间框架与 MACD 判断市场趋势。"""
    try:
        current_price = float(df["close"].iloc[-1])
        trend_short = "上涨" if current_price > float(df["sma_20"].iloc[-1]) else "下跌"
        trend_medium = "上涨" if current_price > float(df["sma_50"].iloc[-1]) else "下跌"
        macd_trend = "bullish" if float(df["macd"].iloc[-1]) > float(df["macd_signal"].iloc[-1]) else "bearish"

        if trend_short == "上涨" and trend_medium == "上涨":
            overall_trend = "强势上涨"
        elif trend_short == "下跌" and trend_medium == "下跌":
            overall_trend = "强势下跌"
        else:
            overall_trend = "震荡整理"

        return {
            "short_term": trend_short,
            "medium_term": trend_medium,
            "macd": macd_trend,
            "overall": overall_trend,
            "rsi_level": float(df["rsi"].iloc[-1]),
        }
    except Exception as e:
        print(f"趋势分析失败: {e}")
        return {}


def get_symbol_ohlcv_enhanced(symbol: str, config: Dict) -> Optional[Dict]:
    """增强版：获取交易对 K 线数据并计算技术指标（多交易对版本）。"""
    ex = _get_exchange()
    if ex is None:
        print("[get_symbol_ohlcv_enhanced] 未能获取交易所实例 exchange")
        return None
    try:
        # 获取 K 线数据
        ohlcv = ex.fetch_ohlcv(symbol, config["timeframe"], limit=config["data_points"])  # type: ignore[index]
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        # 计算技术指标
        df = calculate_technical_indicators(df)

        current_data = df.iloc[-1]
        previous_data = df.iloc[-2]

        # 获取技术分析数据
        trend_analysis = get_market_trend(df)
        levels_analysis = get_support_resistance_levels(df)

        return {
            "symbol": symbol,
            "display": config.get("display", symbol),
            "price": float(current_data["close"]),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "high": float(current_data["high"]),
            "low": float(current_data["low"]),
            "volume": float(current_data["volume"]),
            "timeframe": config.get("timeframe"),
            "price_change": float(((current_data["close"] - previous_data["close"]) / previous_data["close"]) * 100),
            "kline_data": df[["timestamp", "open", "high", "low", "close", "volume"]].tail(10).to_dict("records"),
            "technical_data": {
                "sma_5": float(current_data.get("sma_5", 0)),
                "sma_20": float(current_data.get("sma_20", 0)),
                "sma_50": float(current_data.get("sma_50", 0)),
                "ema_20": float(current_data.get("ema_20", 0)),
                "ema_50": float(current_data.get("ema_50", 0)),
                "rsi": float(current_data.get("rsi", 0)),
                "rsi_7": float(current_data.get("rsi_7", 0)),
                "macd": float(current_data.get("macd", 0)),
                "macd_signal": float(current_data.get("macd_signal", 0)),
                "macd_histogram": float(current_data.get("macd_histogram", 0)),
                "bb_upper": float(current_data.get("bb_upper", 0)),
                "bb_lower": float(current_data.get("bb_lower", 0)),
                "bb_position": float(current_data.get("bb_position", 0)),
                "atr": float(current_data.get("atr", 0)),
                "atr_3": float(current_data.get("atr_3", 0)),
                "volume_ratio": float(current_data.get("volume_ratio", 0)),
                "volume_ma": float(current_data.get("volume_ma", 0)),
            },
            "trend_analysis": trend_analysis,
            "levels_analysis": levels_analysis,
            "full_data": df,
        }
    except Exception as e:
        print(f"[{config.get('display', symbol)}] 获取K线数据失败")
        print(f"   错误类型: {type(e).__name__}")
        print(f"   错误信息: {str(e)}")
        if hasattr(e, "response") and getattr(e, "response"):
            print(f"   HTTP状态码: {getattr(e.response, 'status_code', '未知')}")
            print(f"   响应内容: {getattr(e.response, 'text', '无')[:500]}")
        import traceback

        traceback.print_exc()
        return None


def get_funding_rate(symbol: str) -> Optional[Dict[str, float]]:
    """获取资金费率（永续合约）。"""
    ex = _get_exchange()
    if ex is None:
        print("[get_funding_rate] 未能获取交易所实例 exchange")
        return None
    try:
        info = ex.fetch_funding_rate(symbol)
        rate = info.get("fundingRate", 0)
        ts = info.get("fundingTimestamp", 0)
        next_time = info.get("fundingDatetime", "")
        return {
            "funding_rate": float(rate) if rate else 0.0,
            "funding_rate_percentage": float(rate) * 100 if rate else 0.0,
            "next_funding_time": next_time,
            "funding_timestamp": ts,
        }
    except Exception as e:
        print(f"[{symbol}] 获取资金费率失败: {e}")
        return None


def get_open_interest(symbol: str) -> Optional[Dict[str, float]]:
    """获取持仓量（Open Interest）。"""
    ex = _get_exchange()
    if ex is None:
        print("[get_open_interest] 未能获取交易所实例 exchange")
        return None
    try:
        oi_info = ex.fetch_open_interest(symbol)
        open_interest = oi_info.get("openInterestAmount", 0) or oi_info.get("openInterest", 0)
        return {
            "open_interest": float(open_interest) if open_interest else 0.0,
            "timestamp": oi_info.get("timestamp", 0),
        }
    except Exception as e:
        print(f"[{symbol}] 获取持仓量失败: {e}")
        return None


def get_current_position(symbol: Optional[str] = None) -> Optional[Dict]:
    """获取当前持仓情况 - OKX 版本（多交易对）。

    说明：默认 symbol 为 "BTC/USDT:USDT"，杠杆缺省按配置回退为 10。
    """
    ex = _get_exchange()
    if ex is None:
        print("[get_current_position] 未能获取交易所实例 exchange")
        return None
    try:
        symbol = symbol or "BTC/USDT:USDT"
        positions = ex.fetch_positions([symbol])
        for pos in positions:
            if pos.get("symbol") == symbol:
                contracts = float(pos.get("contracts") or 0)
                if contracts > 0:
                    config = _get_symbol_config(symbol)
                    return {
                        "side": pos.get("side"),  # 'long' or 'short'
                        "size": contracts,
                        "entry_price": float(pos.get("entryPrice") or 0),
                        "unrealized_pnl": float(pos.get("unrealizedPnl") or 0),
                        "leverage": float(pos.get("leverage") or config.get("leverage_default", 10)),
                        "symbol": pos.get("symbol"),
                    }
        return None
    except Exception as e:
        print(f"[{symbol}] 获取持仓失败: {e}")
        import traceback

        traceback.print_exc()
        return None
