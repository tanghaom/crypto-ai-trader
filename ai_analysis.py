# -*- coding: utf-8 -*-
"""
AI 分析模块：将 analyze_with_deepseek 抽离为独立模块，避免主模块臃肿。
说明：
- 为避免循环依赖，deepseekok2 的依赖通过函数内部局部导入。
- 外部只需 from ai_analysis import analyze_with_llm 使用即可。
"""
import json
import re
from datetime import datetime
from typing import Dict

from config.settings import CONFIDENCE_RATIOS
from prompt_builder import build_professional_prompt, build_system_prompt


def safe_json_parse(json_str):
    """安全解析JSON，处理格式不规范的情况"""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            # 尝试提取JSON代码块（如果AI包在```json```中）
            if "```json" in json_str:
                start = json_str.find("```json") + 7
                end = json_str.find("```", start)
                if end != -1:
                    json_str = json_str[start:end].strip()
            elif "```" in json_str:
                start = json_str.find("```") + 3
                end = json_str.find("```", start)
                if end != -1:
                    json_str = json_str[start:end].strip()

            # 尝试直接解析
            try:
                return json.loads(json_str)
            except:
                pass

            # 修复常见的JSON格式问题
            json_str = json_str.replace("'", '"')
            json_str = re.sub(r"(\w+):", r'"\1":', json_str)
            json_str = re.sub(r",\s*}", "}", json_str)
            json_str = re.sub(r",\s*]", "]", json_str)
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON解析失败，原始内容: {json_str[:200]}")
            print(f"错误详情: {e}")
            return None


def validate_and_correct_leverage(signal_data: Dict, config: Dict) -> Dict:
    """
    验证并修正AI返回的杠杆值，确保在配置范围内

    Args:
        signal_data: AI返回的信号数据
        config: 交易对配置

    Returns:
        修正后的信号数据
    """
    leverage = signal_data.get("leverage")
    leverage_min = config["leverage_min"]
    leverage_max = config["leverage_max"]
    leverage_default = config["leverage_default"]

    # 如果没有提供杠杆值，使用默认值
    if leverage is None:
        print(f"[{config['display']}] ⚠️ AI未返回杠杆值，使用默认值 {leverage_default}x")
        signal_data["leverage"] = leverage_default
        return signal_data

    # 转换为整数
    try:
        leverage = int(leverage)
    except (ValueError, TypeError):
        print(f"[{config['display']}] ⚠️ 杠杆值格式错误: {leverage}，使用默认值 {leverage_default}x")
        signal_data["leverage"] = leverage_default
        return signal_data

    # 检查是否超出范围
    if leverage < leverage_min or leverage > leverage_max:
        original_leverage = leverage
        # 限制在配置范围内
        leverage = max(leverage_min, min(leverage, leverage_max))
        print(f"[{config['display']}] ⚠️ 杠杆值 {original_leverage}x 超出配置范围 [{leverage_min}-{leverage_max}]，已修正为 {leverage}x")
        signal_data["leverage"] = leverage
    else:
        print(f"[{config['display']}] ✓ 杠杆值 {leverage}x 在有效范围内")
        signal_data["leverage"] = leverage

    return signal_data


def analyze_with_llm(symbol: str, price_data: Dict, config: Dict) -> Dict:
    """
    使用LLM分析市场并生成交易信号（多交易对+动态杠杆+智能资金管理版本）

    参数：
        symbol: 交易对，如 "ETH/USDT:USDT"
        price_data: 价格与技术指标数据
        config: 该交易对的配置字典

    返回：
        标准化的交易信号字典
    """
    # 延迟导入，避免循环依赖
    from deepseekok2 import (
        AI_MODEL,
        AI_PROVIDER,
        adjust_contract_quantity,
        ai_client,
        append_signal_record,
        base_to_contracts,
        contracts_to_base,
        exchange,
        get_active_context,
        get_symbol_contract_specs,
        get_symbol_min_amount,
        signal_history,
        update_signal_validation,
        web_data,
    )
    from market_utils import get_funding_rate, get_open_interest, get_sentiment_indicators, get_current_position

    # 1) 获取账户余额并做容错
    try:
        balance = exchange.fetch_balance()
        available_balance = 0.0
        # 标准结构
        if "USDT" in balance and balance["USDT"]:
            available_balance = float(balance["USDT"].get("free", 0) or 0)
            float(balance["USDT"].get("total", 0) or 0)
        # OKX info.data.details 结构
        elif "info" in balance and "data" in balance["info"]:
            for data_item in balance["info"]["data"]:
                details = data_item.get("details", [])
                for detail in details:
                    if detail.get("ccy") == "USDT":
                        available_balance = float(detail.get("availBal", "0") or 0)
                        float(detail.get("eq", "0") or 0)
                        break
                if available_balance > 0:
                    break
        if available_balance <= 0:
            available_balance = 1000.0
    except Exception as e:
        print(f"⚠️ 获取余额失败: {e}")
        available_balance = 1000.0

    print(f"[{config['display']}] 🔍 AI分析-获取余额: {available_balance:.2f} USDT")

    # 2) 资金管理：预计算仓位组合
    current_price = price_data["price"]
    max_usable_margin = available_balance * 0.8
    print(f"[{config['display']}] 🔍 最大可用保证金: {max_usable_margin:.2f} USDT (80%)")

    position_suggestions: Dict[str, Dict] = {}
    specs = get_symbol_contract_specs(symbol)
    contract_size = specs["contract_size"]
    min_contracts = specs["min_contracts"]
    min_quantity = get_symbol_min_amount(symbol)
    leverage_list = [config["leverage_min"], config["leverage_default"], config["leverage_max"]]

    for confidence in ["HIGH", "MEDIUM", "LOW"]:
        ratio = CONFIDENCE_RATIOS[confidence]
        for lev in leverage_list:
            target_margin = max_usable_margin * ratio
            raw_quantity = (target_margin * lev / current_price) if current_price else 0
            base_quantity = max(raw_quantity, min_quantity)
            contracts = base_to_contracts(symbol, base_quantity)
            if min_contracts:
                contracts = max(contracts, min_contracts)
            adjusted_contracts = adjust_contract_quantity(symbol, contracts, round_up=True)
            adjusted_quantity = contracts_to_base(symbol, adjusted_contracts)
            adjusted_margin = adjusted_quantity * current_price / lev if lev else 0
            meets_min = adjusted_contracts >= (min_contracts if min_contracts else 0)
            meets_margin = adjusted_margin <= max_usable_margin if max_usable_margin else True

            # 调试一组示例
            if confidence == "LOW" and lev == config["leverage_max"]:
                print(f"[{config['display']}] 🔍 检查组合: {confidence}信心 + {lev}倍杠杆")
                print(f"[{config['display']}]    需要数量: {adjusted_quantity:.6f} ETH ({adjusted_contracts:.3f}张)")
                print(f"[{config['display']}]    需要保证金: {adjusted_margin:.2f} USDT")
                print(f"[{config['display']}]    最小合约: {min_contracts:.3f}张, 满足: {meets_min}")
                print(f"[{config['display']}]    保证金充足: {meets_margin} (需要{adjusted_margin:.2f} <= 可用{max_usable_margin:.2f})")
                print(f"[{config['display']}]    最终判断: {meets_min and meets_margin}")

            key = f"{confidence}_{lev}"
            position_suggestions[key] = {
                "quantity": adjusted_quantity,
                "contracts": adjusted_contracts,
                "contract_size": contract_size,
                "value": adjusted_quantity * current_price,
                "margin": adjusted_margin,
                "meets_min": meets_min,
                "meets_margin": meets_margin,
                "meets": meets_min and meets_margin,
            }

    can_trade = any(pos.get("meets") for pos in position_suggestions.values())
    position_suggestions["available_balance"] = available_balance
    position_suggestions["current_price"] = current_price
    position_suggestions["usable_margin"] = max_usable_margin
    position_suggestions["min_quantity"] = min_quantity
    position_suggestions["min_contracts"] = min_contracts
    position_suggestions["contract_size"] = contract_size

    ctx = get_active_context()

    if not can_trade:
        min_contracts_display = min_contracts if min_contracts else base_to_contracts(symbol, min_quantity)
        print(f"[{config['display']}] ⚠️ 余额不足：即使最大杠杆也无法满足最小交易量 {min_quantity} ({min_contracts_display:.3f} 张)")
        print(f"[{config['display']}] 💡 当前余额: {available_balance:.2f} USDT")
        print(f"[{config['display']}] 💡 建议充值至少: {(min_quantity * current_price / config['leverage_max']):.2f} USDT")

        fallback_signal = {
            "signal": "HOLD",
            "reason": f"账户余额不足({available_balance:.2f} USDT)，无法满足最小交易量要求({min_quantity}，约{min_contracts_display:.3f}张)，建议充值至少{(min_quantity * current_price / config['leverage_max']):.2f} USDT",
            "stop_loss": current_price * 0.98,
            "take_profit": current_price * 1.02,
            "confidence": "LOW",
            "leverage": config["leverage_default"],
            "order_quantity": 0,
            "is_insufficient_balance": True,
        }
        fallback_signal["timestamp"] = price_data["timestamp"]
        append_signal_record(symbol, fallback_signal, current_price, fallback_signal["timestamp"])
        ctx.metrics["signals_generated"] += 1

        print(f"[{config['display']}] 💡 跳过AI分析（余额不足），直接返回HOLD信号")
        return fallback_signal

    # 3) 更新历史记录验证信息
    update_signal_validation(symbol, price_data["price"], price_data["timestamp"])

    # 4) 情绪数据
    token = symbol.split("/")[0] if "/" in symbol else symbol
    sentiment_text = ""
    sentiment_data = get_sentiment_indicators(token)

    if sentiment_data:
        sign = "+" if sentiment_data["net_sentiment"] >= 0 else ""
        sentiment_text = f"{token}市场情绪 乐观{sentiment_data['positive_ratio']:.1%} 悲观{sentiment_data['negative_ratio']:.1%} 净值{sign}{sentiment_data['net_sentiment']:.3f}"
        print(f"[{config['display']}] {sentiment_text}")
    else:
        if token != "BTC":
            print(f"[{config['display']}] ⚠️ {token}情绪数据不可用，尝试使用BTC市场情绪...")
            btc_sentiment = get_sentiment_indicators("BTC")
            if btc_sentiment:
                sign = "+" if btc_sentiment["net_sentiment"] >= 0 else ""
                sentiment_text = f"BTC市场情绪(参考) 乐观{btc_sentiment['positive_ratio']:.1%} 悲观{btc_sentiment['negative_ratio']:.1%} 净值{sign}{btc_sentiment['net_sentiment']:.3f}"
                print(f"[{config['display']}] {sentiment_text}")
            else:
                sentiment_text = "市场情绪暂无有效数据"
        else:
            sentiment_text = "市场情绪暂无有效数据"

    # 5) 最小交易量约束与上下文指标
    current_position = get_current_position(symbol)
    specs = get_symbol_contract_specs(symbol)
    contract_size = specs["contract_size"]
    min_contracts = max(specs["min_contracts"], base_to_contracts(symbol, get_symbol_min_amount(symbol)))
    min_contracts = adjust_contract_quantity(symbol, min_contracts, round_up=True) if min_contracts else 0
    min_quantity = contracts_to_base(symbol, min_contracts) if min_contracts else get_symbol_min_amount(symbol)
    ctx.metrics["ai_calls"] += 1

    # 6) 资金费率与持仓量文本
    try:
        funding_info = get_funding_rate(symbol)
        price_data["funding_rate_text"] = f"{funding_info['funding_rate_percentage']:.4f}%" if funding_info else "暂无数据"
    except:
        price_data["funding_rate_text"] = "暂无数据"

    try:
        oi_info = get_open_interest(symbol)
        price_data["open_interest_text"] = f"{oi_info['open_interest']:,.0f}" if oi_info else "暂无数据"
    except:
        price_data["open_interest_text"] = "暂无数据"

    # 7) 构建提示词并调用大模型
    prompt = build_professional_prompt(ctx, symbol, price_data, config, position_suggestions, sentiment_text, current_position)
    try:
        print(f"⏳ 正在调用{AI_PROVIDER.upper()} API ({AI_MODEL})...")
        system_prompt = build_system_prompt(config)

        response = ai_client.chat.completions.create(
            model=AI_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            stream=False,
            temperature=0.1,
            timeout=30.0,
        )
        print("✓ API调用成功")

        # 更新AI连接状态
        web_data["ai_model_info"]["status"] = "connected"
        web_data["ai_model_info"]["last_check"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        web_data["ai_model_info"]["error_message"] = None

        if not response:
            print(f"❌ {AI_PROVIDER.upper()}返回空响应")
            web_data["ai_model_info"]["status"] = "error"
            web_data["ai_model_info"]["error_message"] = "响应为空"
            return create_fallback_signal(price_data)

        if isinstance(response, str):
            result = response
        elif hasattr(response, "choices") and response.choices:
            result = response.choices[0].message.content
        else:
            print(f"❌ {AI_PROVIDER.upper()}返回格式异常: {type(response)}")
            print(f"   响应内容: {str(response)[:200]}")
            web_data["ai_model_info"]["status"] = "error"
            web_data["ai_model_info"]["error_message"] = "响应格式异常"
            return create_fallback_signal(price_data)

        if not result:
            print(f"❌ {AI_PROVIDER.upper()}返回空内容")
            return create_fallback_signal(price_data)

        print(f"\n{'='*60}")
        print(f"{AI_PROVIDER.upper()}原始回复:")
        print(result)
        print(f"{'='*60}\n")

        # 提取 JSON
        start_idx = result.find("{")
        end_idx = result.rfind("}") + 1

        if start_idx != -1 and end_idx != 0:
            json_str = result[start_idx:end_idx]
            signal_data = safe_json_parse(json_str)
            if signal_data is None:
                print("⚠️ JSON解析失败，使用备用信号")
                signal_data = create_fallback_signal(price_data)
            else:
                print(f"✓ 成功解析AI决策: {signal_data.get('signal')} - {signal_data.get('confidence')}")
        else:
            print("⚠️ 未找到JSON格式，使用备用信号")
            signal_data = create_fallback_signal(price_data)

        # 字段校验与杠杆修正
        required_fields = ["signal", "reason", "stop_loss", "take_profit", "confidence"]
        if not all(field in signal_data for field in required_fields):
            missing = [f for f in required_fields if f not in signal_data]
            print(f"⚠️ 缺少必需字段: {missing}，使用备用信号")
            signal_data = create_fallback_signal(price_data)

        signal_data = validate_and_correct_leverage(signal_data, config)

        # 写入历史记录与统计
        signal_data["timestamp"] = price_data["timestamp"]
        record = append_signal_record(symbol, signal_data, price_data["price"], signal_data["timestamp"])
        history = signal_history[symbol]
        ctx.metrics["signals_generated"] += 1

        signal_count = len([s for s in history if s.get("signal") == record.get("signal")])
        total_signals = len(history)
        print(f"[{config['display']}] 信号统计: {signal_data['signal']} (最近{total_signals}次中出现{signal_count}次)")

        if len(history) >= 3:
            last_three = [s["signal"] for s in history[-3:]]
            if len(set(last_three)) == 1:
                print(f"[{config['display']}] ⚠️ 注意：连续3次{signal_data['signal']}信号")

        if len(history) >= 20:
            recent_20 = history[-20:]
            conf_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
            for r in recent_20:
                conf = r.get("confidence", "MEDIUM")
                conf_counts[conf] = conf_counts.get(conf, 0) + 1

            low_ratio = conf_counts["LOW"] / len(recent_20)
            high_ratio = conf_counts["HIGH"] / len(recent_20)

            if low_ratio > 0.5:
                print(f"[{config['display']}] ⚠️ 信心度警告：最近20次中{low_ratio*100:.0f}%是LOW，模型可能过于保守")
                print(f"[{config['display']}]    分布: HIGH={conf_counts['HIGH']} MED={conf_counts['MEDIUM']} LOW={conf_counts['LOW']}")
            elif high_ratio < 0.2:
                print(f"[{config['display']}] 💡 提示：最近20次中HIGH仅{high_ratio*100:.0f}%，可能错过高确定性机会")

        return signal_data

    except Exception as e:
        print(f"[{config['display']}] ❌ {AI_PROVIDER.upper()}分析失败: {e}")
        import traceback

        traceback.print_exc()
        ctx.metrics["ai_errors"] += 1
        web_data["ai_model_info"]["status"] = "error"
        web_data["ai_model_info"]["last_check"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        web_data["ai_model_info"]["error_message"] = str(e)
        fallback = create_fallback_signal(price_data)
        fallback["timestamp"] = price_data["timestamp"]
        append_signal_record(symbol, fallback, price_data["price"], fallback["timestamp"])
        return fallback


def create_fallback_signal(price_data):
    """创建备用交易信号"""
    return {
        "signal": "HOLD",
        "reason": "因技术分析暂时不可用，采取保守策略",
        "stop_loss": price_data["price"] * 0.98,  # -2%
        "take_profit": price_data["price"] * 1.02,  # +2%
        "confidence": "LOW",
        "is_fallback": True,
    }
