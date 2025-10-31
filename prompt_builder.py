"""
提示词构建模块
包含所有与AI提示词生成相关的格式化和构建函数
"""

from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

# ==================== 基础格式化函数 ====================


def format_number(value, decimals: int = 2) -> str:
    """格式化数字，自动处理整数和小数"""
    if value is None:
        return "--"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(val - round(val)) < 1e-6:
        return str(int(round(val)))
    formatted = f"{val:.{decimals}f}"
    return formatted.rstrip("0").rstrip(".") if "." in formatted else formatted


def format_percentage(value: Optional[float]) -> str:
    """格式化百分比"""
    if value is None:
        return "--"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f}%"


def format_currency(value: Optional[float], decimals: int = 2) -> str:
    """格式化货币数值，值为空时返回 --"""
    if value is None:
        return "--"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"${val:,.{decimals}f}"


def format_sequence(values: List[float], indent: int = 2, per_line: int = 10, decimals: int = 2) -> str:
    """格式化数字序列为多行显示"""
    if not values:
        return " " * indent + "[]"
    parts = [format_number(v, decimals) for v in values]
    lines = []
    for i in range(0, len(parts), per_line):
        chunk = ", ".join(parts[i : i + per_line])
        lines.append(chunk)
    if not lines:
        return " " * indent + "[]"
    result_lines = []
    result_lines.append(" " * indent + "[" + lines[0] + ("," if len(lines) > 1 else "]"))
    for idx in range(1, len(lines)):
        suffix = "," if idx < len(lines) - 1 else "]"
        result_lines.append(" " * (indent + 1) + lines[idx] + suffix)
    return "\n".join(result_lines)


# ==================== 历史数据分析函数 ====================


def compute_accuracy_metrics(history: List[Dict]) -> Dict:
    """计算历史信号准确率指标"""
    evaluated = [rec for rec in history if rec.get("result") in ("success", "fail")]

    def summarize(records: List[Dict]) -> Dict:
        total = len(records)
        success = sum(1 for r in records if r.get("result") == "success")
        ratio = success / total if total else None
        return {"total": total, "success": success, "ratio": ratio}

    metrics = {
        "windows": {"10": summarize(evaluated[-10:]), "30": summarize(evaluated[-30:]), "50": summarize(evaluated[-50:])},
        "by_signal": {},
        "by_confidence": {},
        "by_leverage": {},
    }

    for signal_label in ["BUY", "SELL", "HOLD"]:
        metrics["by_signal"][signal_label] = summarize([r for r in evaluated if r.get("signal") == signal_label])

    for confidence in ["HIGH", "MEDIUM", "LOW"]:
        metrics["by_confidence"][confidence] = summarize([r for r in evaluated if r.get("confidence") == confidence])

    leverage_buckets = {"3-8x": lambda lev: 3 <= lev <= 8, "9-12x": lambda lev: 9 <= lev <= 12, "13-20x": lambda lev: 13 <= lev <= 20}
    for label, predicate in leverage_buckets.items():
        metrics["by_leverage"][label] = summarize(
            [r for r in evaluated if isinstance(r.get("leverage"), (int, float)) and predicate(int(r["leverage"]))]
        )
    return metrics


def format_ratio(summary: Dict) -> str:
    """格式化准确率比例"""
    total = summary.get("total", 0)
    success = summary.get("success", 0)
    ratio = summary.get("ratio")
    if not total:
        return "-- (--/0)"
    percent = f"{ratio * 100:.0f}%"
    return f"{percent} ({success}✓/{total})"


def format_history_table(history: List[Dict]) -> str:
    """格式化历史判断验证表格"""
    if not history:
        return "  无历史信号记录\n"
    last_records = history[-50:]
    total = len(last_records)
    lines = ["  序号 信号  信心 杠杆  入场价  验证价  涨跌    结果"]
    for idx, record in enumerate(last_records):
        seq_no = idx - total
        signal = (record.get("signal") or "--").upper().ljust(4)
        confidence = (record.get("confidence") or "--").upper().ljust(3)
        leverage = f"{int(record.get('leverage', 0)):>2}x"
        entry = format_number(record.get("entry_price"))
        validation = format_number(record.get("validation_price"))
        change_pct = format_percentage(record.get("price_change_pct"))
        result_symbol = {"success": "✓", "fail": "✗"}.get(record.get("result"), "·")
        lines.append(f"  {seq_no:>3}  {signal} {confidence} {leverage:>4}  {entry:>7}  {validation:>7}  {change_pct:>6}   {result_symbol}")
    return "\n".join(lines)


def format_accuracy_summary(metrics: Dict) -> str:
    """格式化准确率统计摘要"""
    lines = ["  【准确率统计分析】", "", "  时间窗口:"]
    lines.append(f"  - 最近10次: {format_ratio(metrics['windows']['10'])}")
    lines.append(f"  - 最近30次: {format_ratio(metrics['windows']['30'])}")
    lines.append(f"  - 最近50次: {format_ratio(metrics['windows']['50'])}")
    lines.append("")
    lines.append("  按信号类型:")
    for signal_label in ["BUY", "SELL", "HOLD"]:
        lines.append(f"  - {signal_label:<4}: {format_ratio(metrics['by_signal'][signal_label])}")
    lines.append("")
    lines.append("  按信心等级:")
    for confidence in ["HIGH", "MEDIUM", "LOW"]:
        lines.append(f"  - {confidence:<6}: {format_ratio(metrics['by_confidence'][confidence])}")
    lines.append("")
    lines.append("  按杠杆范围:")
    for bucket in ["3-8x", "9-12x", "13-20x"]:
        lines.append(f"  - {bucket:<6}: {format_ratio(metrics['by_leverage'][bucket])}")
    lines.append("")
    lines.append("  关键观察:")
    lines.append("  - 高信心信号准确率显著优于低信心，应积极寻找HIGH机会")
    lines.append("  - 理想信心度分布: HIGH 25% | MEDIUM 50% | LOW 25%")
    lines.append("  - ⚠️ 不要过度保守！只在真正不确定时才用LOW")
    return "\n".join(lines)


# ==================== 仓位建议表格构建 ====================


def build_position_suggestion_table(position_suggestions: Dict[str, Dict], config: Dict, asset_name: str) -> str:
    """构建智能仓位建议表格"""
    lines = []
    leverage_min = config["leverage_min"]
    leverage_default = config["leverage_default"]
    leverage_max = config["leverage_max"]
    min_quantity = position_suggestions.get("min_quantity", config["amount"])
    min_contracts = position_suggestions.get("min_contracts", 0)

    def row(confidence_label: str, leverage: int) -> str:
        key = f"{confidence_label}_{leverage}"
        suggestion = position_suggestions.get(key, {})
        quantity = suggestion.get("quantity", 0)
        contracts = suggestion.get("contracts")
        value = suggestion.get("value", 0)
        margin = suggestion.get("margin", 0)
        meets_min = suggestion.get("meets_min", True)
        meets_margin = suggestion.get("meets_margin", True)
        status_parts = []
        status_parts.append("满足最小交易量" if meets_min else "低于最小交易量")
        status_parts.append("保证金充足" if meets_margin else "保证金不足")
        flag = "✅" if suggestion.get("meets", True) else "❌"
        status = " & ".join(status_parts)
        contracts_info = f"{contracts:.3f}张, " if contracts is not None else ""
        return f"  • {leverage}x: {quantity:.6f} {asset_name} ({contracts_info}价值 ${value:,.2f}), 需 {margin:.2f} USDT {flag} {status}"

    lines.append("  【智能仓位建议表】- 已为你精确计算")
    lines.append("")
    usable_margin = position_suggestions.get("usable_margin", position_suggestions.get("available_balance", 0) * 0.8)
    lines.append(
        f"  账户状态: 可用 {position_suggestions.get('available_balance', 0):.2f} USDT | 可用保证金 {usable_margin:.2f} USDT | 价格 ${position_suggestions.get('current_price', 0):,.2f} | 最小量 {min_quantity} {asset_name} ({min_contracts:.3f} 张)"
    )
    lines.append("")
    sections = [("HIGH", "高信心(HIGH) - 70%保证金"), ("MEDIUM", "中信心(MEDIUM) - 50%保证金"), ("LOW", "低信心(LOW) - 30%保证金")]
    for confidence_key, title in sections:
        lines.append(f"  {title}:")
        for lev in [leverage_min, leverage_default, leverage_max]:
            lines.append(row(confidence_key, lev))
        lines.append("")
    return "\n".join(lines)


# ==================== 交易历史表格构建 ====================


def format_trade_history_table(trade_history: List[Dict], max_rows: int = 20) -> str:
    """格式化实际交易历史表格"""
    if not trade_history:
        return "  暂无实际交易记录\n"

    lines = []
    lines.append("  序号 时间          操作类型      方向   价格      数量(ETH)  杠杆 信心  盈亏(USDT)")
    lines.append("  " + "-" * 85)

    recent_trades = trade_history[-max_rows:] if len(trade_history) > max_rows else trade_history

    for idx, trade in enumerate(recent_trades, start=1):
        seq = idx - len(recent_trades)  # 负数序号，-1表示最近一次
        timestamp = trade.get("timestamp", "")[:16]  # 只取日期和时分
        trade_type_display = trade.get("trade_type_display", "")[:12]  # 限制长度
        side = trade.get("side", "")
        side_display = "多" if side == "long" else "空" if side == "short" else "--"
        price = trade.get("price", 0)
        amount = trade.get("amount", 0)
        leverage = trade.get("leverage", 0)
        confidence = trade.get("confidence", "MED")[:3]
        pnl = trade.get("pnl", 0)

        lines.append(
            f"  {seq:>3}  {timestamp}  {trade_type_display:<12}  {side_display:<4}  {price:>8.2f}  {amount:>9.6f}  {leverage:>2}x  {confidence:<4}  {pnl:>+8.2f}"
        )

    return "\n".join(lines) + "\n"


def build_trade_frequency_warning(trade_history: List[Dict]) -> str:
    """分析交易频率并生成警告信息"""
    if not trade_history or len(trade_history) < 2:
        return ""

    warnings = []
    now = datetime.now()

    # 分析最近的交易
    recent_10 = trade_history[-10:] if len(trade_history) >= 10 else trade_history

    # 1. 检查最近一次交易的时间间隔
    last_trade = trade_history[-1]
    last_trade_time = datetime.strptime(last_trade["timestamp"], "%Y-%m-%d %H:%M:%S")
    minutes_since_last = (now - last_trade_time).total_seconds() / 60

    if minutes_since_last < 15:
        warnings.append(f"  🔴 警告：距离上次交易仅{minutes_since_last:.1f}分钟，请谨慎操作避免频繁交易！")

    # 2. 检查最近交易的频率
    if len(recent_10) >= 5:
        first_time = datetime.strptime(recent_10[0]["timestamp"], "%Y-%m-%d %H:%M:%S")
        last_time = datetime.strptime(recent_10[-1]["timestamp"], "%Y-%m-%d %H:%M:%S")
        time_span_hours = (last_time - first_time).total_seconds() / 3600

        if time_span_hours > 0:
            trades_per_hour = len(recent_10) / time_span_hours
            if trades_per_hour > 2:  # 每小时超过2次交易
                warnings.append(f"  ⚠️ 提示：最近交易频率较高（{trades_per_hour:.1f}次/小时），建议降低交易频率提高质量")

    # 3. 检查来回反转模式（多->空->多 或 空->多->空）
    if len(recent_10) >= 4:
        # 检测是否存在短时间内的来回反转
        flip_flop_count = 0
        for i in range(2, len(recent_10)):
            side_a = recent_10[i - 2].get("side")
            side_b = recent_10[i - 1].get("side")
            side_c = recent_10[i].get("side")
            # 如果A和C相同，但B不同，说明来回反转了
            if side_a and side_b and side_c and side_a == side_c and side_b != side_a:
                flip_flop_count += 1

        if flip_flop_count >= 2:
            warnings.append(f"  ⚠️ 提示：检测到{flip_flop_count}次来回反转（如：多→空→多），这种模式通常导致亏损")

    # 4. 计算最近交易的盈亏情况
    total_pnl = sum(t.get("pnl", 0) for t in recent_10 if "pnl" in t)
    profitable_trades = len([t for t in recent_10 if t.get("pnl", 0) > 0])

    if len(recent_10) >= 5:
        win_rate = profitable_trades / len(recent_10) * 100
        if total_pnl < 0:
            warnings.append(f"  💡 分析：最近{len(recent_10)}笔交易累计亏损{abs(total_pnl):.2f} USDT（胜率{win_rate:.0f}%），建议提高信号质量")
        elif win_rate < 50:
            warnings.append(f"  💡 分析：最近{len(recent_10)}笔交易胜率{win_rate:.0f}%，建议更谨慎选择交易时机")

    # 5. 添加交易建议
    if warnings:
        warnings.append("\n  💡 建议策略：")
        warnings.append("     • 只在HIGH信心且多个指标共振时才交易")
        warnings.append("     • 避免在15-30分钟内重复开平仓")
        warnings.append("     • 使用HOLD信号耐心等待更好的机会")
        warnings.append("     • 减少低质量交易，宁可少赚也不多亏\n")

    return "\n".join(warnings) if warnings else ""


# ==================== 主提示词构建函数 ====================


def build_professional_prompt(
    ctx, symbol: str, price_data: Dict, config: Dict, position_suggestions: Dict[str, Dict], sentiment_text: str, current_position: Optional[Dict]
) -> str:
    """构建专业的交易分析提示词"""
    df: pd.DataFrame = price_data.get("full_data")  # type: ignore
    short_df = df.tail(20) if df is not None else None

    # 提取价格和技术指标序列
    prices = short_df["close"].tolist() if short_df is not None else []
    sma5 = short_df["sma_5"].tolist() if short_df is not None else []
    sma20 = short_df["sma_20"].tolist() if short_df is not None else []
    ema20 = short_df["ema_20"].tolist() if short_df is not None else []
    rsi = short_df["rsi"].tolist() if short_df is not None else []
    rsi_7 = short_df["rsi_7"].tolist() if short_df is not None else []
    macd = short_df["macd"].tolist() if short_df is not None else []
    volume = short_df["volume"].tolist() if short_df is not None else []

    # 获取历史记录和统计指标
    history = ctx.signal_history[symbol]
    metrics = compute_accuracy_metrics(history)
    history_table = format_history_table(history)
    accuracy_summary = format_accuracy_summary(metrics)

    # 获取实际交易历史
    trade_history = ctx.web_data["symbols"][symbol].get("trade_history", [])
    trade_history_table = format_trade_history_table(trade_history)

    # 系统运行状态
    runtime_minutes = int((datetime.now() - ctx.start_time).total_seconds() / 60)
    runtime_hours = runtime_minutes / 60
    ai_calls = ctx.metrics["ai_calls"]
    open_positions = sum(1 for pos in ctx.position_state.values() if pos)
    closed_trades = ctx.metrics["trades_closed"]

    asset_name = config["display"].split("-")[0]
    position_table = build_position_suggestion_table(position_suggestions, config, asset_name)

    # 当前持仓状态
    if current_position:
        position_status = f"{current_position.get('side', '--')} {current_position.get('size', 0)} {asset_name} @{format_number(current_position.get('entry_price'))}, 未实现盈亏: {format_number(current_position.get('unrealized_pnl'))} USDT"
    else:
        position_status = "无持仓"

    # 获取技术指标数据
    tech = price_data["technical_data"]
    levels = price_data.get("levels_analysis", {})

    # 获取资金费率和持仓量（如果有）
    # 注意：为避免循环导入，这些数据应该在price_data中提供或作为参数传入
    funding_rate_text = price_data.get("funding_rate_text", "暂无数据")
    open_interest_text = price_data.get("open_interest_text", "暂无数据")

    # 构建提示词各部分
    prompt_sections = [
        f"\n  你是专业的加密货币交易分析师 | {config['display']} {config['timeframe']}周期\n",
        f"\n  【系统运行状态】\n  运行时长: {runtime_minutes}分钟 ({runtime_hours:.1f}小时) | AI分析: {ai_calls}次 | 开仓: {ctx.metrics['trades_opened']}次 | 平仓: {closed_trades}次 | 当前持仓: {open_positions}个\n",
        "  ⚠️ 重要: 以下所有时间序列数据按 最旧→最新 排列\n",
        "  【短期序列】最近20周期 = 100分钟 (最旧→最新)\n",
        "  价格 (USDT):\n" + format_sequence(prices, decimals=2),
        "\n  SMA5周期均线:\n" + format_sequence(sma5, decimals=2),
        "\n  SMA20周期均线:\n" + format_sequence(sma20, decimals=2),
        "\n  EMA20周期均线:\n" + format_sequence(ema20, decimals=2),
        "\n  RSI (14周期):\n" + format_sequence(rsi, decimals=2),
        "\n  RSI (7周期,更敏感):\n" + format_sequence(rsi_7, decimals=2),
        "\n  MACD线:\n" + format_sequence(macd, decimals=2),
        "\n  成交量 (" + asset_name + "):\n" + format_sequence(volume, decimals=2),
        "\n  【你的历史判断验证】最近50次 (最旧→最新)\n" + history_table + "\n",
        accuracy_summary + "\n",
        "\n  【实际交易历史】最近20次真实执行的交易 (最旧→最新)\n",
        "  ⚠️ 这是实际下单记录，与上面的判断验证不同。上面是所有信号，这里是真正执行的交易。\n",
        trade_history_table,
        build_trade_frequency_warning(trade_history),
        "\n  【当前市场状况】\n",
        f"  当前价格: ${price_data['price']:,} (相比上周期: {price_data.get('price_change', 0):+.2f}%)\n"
        f"  当前持仓: {position_status}\n"
        f"  市场情绪: {sentiment_text or '暂无数据'}\n",
        f"  资金费率: {funding_rate_text}\n" f"  持仓量: {open_interest_text}\n",
        "  \n  技术指标详情:\n"
        f"  - 短期趋势: {price_data['trend_analysis'].get('short_term', 'N/A')}\n"
        f"  - 中期趋势: {price_data['trend_analysis'].get('medium_term', 'N/A')}\n"
        f"  - SMA50: ${tech.get('sma_50', 0):.2f} (价格偏离: {((price_data['price'] - tech.get('sma_50', 0)) / tech.get('sma_50', 1) * 100):+.2f}%)\n"
        f"  - EMA20: ${tech.get('ema_20', 0):.2f} (价格偏离: {((price_data['price'] - tech.get('ema_20', 0)) / tech.get('ema_20', 1) * 100):+.2f}%)\n"
        f"  - EMA50: ${tech.get('ema_50', 0):.2f} (价格偏离: {((price_data['price'] - tech.get('ema_50', 0)) / tech.get('ema_50', 1) * 100):+.2f}%)\n"
        f"  - RSI(14): {tech.get('rsi', 0):.2f} | RSI(7): {tech.get('rsi_7', 0):.2f}\n"
        f"  - MACD线: {tech.get('macd', 0):.4f} | MACD信号线: {tech.get('macd_signal', 0):.4f}\n"
        f"  - MACD柱状图: {tech.get('macd_histogram', 0):.4f} ({'金叉看涨' if tech.get('macd_histogram', 0) > 0 else '死叉看跌'})\n"
        f"  - 布林带上轨: ${tech.get('bb_upper', 0):.2f} | 下轨: ${tech.get('bb_lower', 0):.2f}\n"
        f"  - 布林带位置: {tech.get('bb_position', 0):.2%} ({'超买区' if tech.get('bb_position', 0) > 0.8 else '超卖区' if tech.get('bb_position', 0) < 0.2 else '正常区'})\n"
        f"  - ATR(14): ${tech.get('atr', 0):.2f} | ATR(3): ${tech.get('atr_3', 0):.2f} (波动率参考)\n"
        f"  - 成交量: {price_data.get('volume', 0):.2f} {asset_name} | 20周期均量: {tech.get('volume_ma', 0):.2f}\n"
        f"  - 成交量比率: {tech.get('volume_ratio', 0):.2f}倍 ({'放量' if tech.get('volume_ratio', 0) > 1.2 else '缩量' if tech.get('volume_ratio', 0) < 0.8 else '正常'})\n"
        f"  - 支撑位: ${levels.get('static_support', 0):.2f} | 阻力位: ${levels.get('static_resistance', 0):.2f}\n",
        position_table,
        "  【信心度判断标准】⭐ 重要\n"
        "  HIGH (高信心) - 同时满足以下条件时使用:\n"
        "  ✓ 多个技术指标强烈共振（EMA/SMA均线、RSI双周期、MACD金叉/死叉、成交量、ATR波动率）\n"
        "  ✓ 价格突破关键支撑/阻力位，且有明显成交量配合（成交量比率>1.2）\n"
        "  ✓ 形态清晰（如金叉/死叉、突破/跌破均线、布林带突破等）\n"
        "  ✓ 资金费率和持仓量支持该方向判断\n"
        "  ✓ 历史数据显示HIGH准确率最高，应果断使用\n"
        "  MEDIUM (中信心) - 以下情况使用:\n"
        "  • 技术指标有2-3个支持该方向，但存在1个分歧\n"
        "  • 趋势方向明确但动能不强（成交量一般，ATR未放大）\n"
        "  • 突破但未完全确认（如价格在EMA20和EMA50之间）\n"
        "  • 应作为主要选择，占比约50%\n"
        "  LOW (低信心) - 仅在以下情况使用:\n"
        "  • 技术指标严重分歧（多空信号各半）\n"
        "  • 盘整震荡，完全无方向（布林带收窄，ATR萎缩）\n"
        "  • 成交量极度萎缩（成交量比率<0.6）\n"
        "  • 注意：LOW准确率最低，应尽量避免，占比应<30%\n"
        "  【信号选择指南】⭐ 重要\n"
        "  根据技术指标综合分析，选择BUY/SELL/CLOSE/HOLD：\n"
        "  • BUY: 技术指标显示上涨趋势（均线排列、RSI双周期、MACD金叉、放量、资金费率正向等）\n"
        "  • SELL: 技术指标显示下跌趋势时，应选择SELL做空（⚠️ SELL不是平仓，而是做空机会）\n"
        "  • CLOSE: ⭐ 新增平仓信号 - 当有持仓且满足以下条件时使用：\n"
        "     ✓ 趋势反转信号明确（如多头持仓时出现死叉、跌破关键支撑）\n"
        "     ✓ 接近或触及止盈止损位（盈利>3%或亏损>2%）\n"
        "     ✓ 技术指标显示趋势衰竭（RSI背离、成交量萎缩、布林带收窄）\n"
        "     ✓ 市场情绪恶化（资金费率异常、持仓量骤降）\n"
        "     ⚠️ 使用CLOSE时无需填写order_quantity和leverage（平仓会全平当前持仓）\n"
        "  • HOLD: 技术指标分歧或方向不明确时选择持有\n"
        "  ⚠️ 重要：不要只关注上涨机会，当下跌趋势明确时也应果断选择SELL\n"
        "  【决策要求】\n"
        "  1️⃣ 综合分析所有技术指标 + 50次历史验证 + 统计规律 + 资金费率/持仓量\n"
        "  2️⃣ 积极寻找HIGH机会: 当多个指标共振时应果断给HIGH（无论是上涨还是下跌趋势）\n"
        "  3️⃣ 避免过度保守: MEDIUM和HIGH应是主流(共75%)，LOW应是少数(25%)\n"
        "  4️⃣ 平衡多空机会: 综合分析技术指标，不要只关注上涨，当下跌趋势明确时也应选择SELL\n"
        "  5️⃣ ⭐ 持仓管理优化：评估当前持仓是否需要CLOSE平仓、加仓或反向开仓\n"
        "  6️⃣ 注意ATR波动率：高波动时需更宽的止损，低波动时可能预示突破\n"
        "  7️⃣ 🔴 防止频繁交易：参考【实际交易历史】和警告提示，避免在短时间内重复交易\n"
        "     • 如果距离上次交易<15分钟，除非有极强的反转信号，否则应选择HOLD\n"
        "     • 如果最近交易频率过高或胜率低，提高交易标准，只选择HIGH信心的机会\n"
        "     • 宁可错过机会，也不要频繁交易增加手续费成本\n"
        "  8️⃣ ⚠️ 重要：数量选择规则\n"
        f"     - 先确定 confidence (HIGH/MEDIUM/LOW)\n"
        f"     - 再确定 leverage ({config['leverage_min']}x/{config['leverage_default']}x/{config['leverage_max']}x)\n"
        f"     - 在建议表中找到对应【信心等级】的【杠杆倍数】那一行的数量\n"
        f"     - 必须完全复制该数量值（6位小数），禁止自行计算或四舍五入\n"
        f"     - 例如：confidence=MEDIUM, leverage=3x → 找到「中信心(MEDIUM)」栏下的「3x:」那一行的数量\n"
        f"     - 注意：signal=CLOSE时，无需填写order_quantity和leverage\n"
        "  9️⃣ ⭐ 止盈止损设置原则\n"
        "     • 根据ATR波动率、支撑/阻力位、信心等级综合确定\n"
        "     • 建议范围：止损2-8%，止盈4-15%（根据信心和波动率调整）\n"
        "     • 风险收益比：止盈距离应≥止损距离×1.5（推荐1:2或更高）\n"
        "     • CLOSE信号：填0即可；HOLD信号：填当前观望区间\n"
        "     • ⚠️ 避免过窄（<1%）或风险收益比倒挂\n",
        "  请用JSON格式返回:\n"
        "  {\n"
        '    "signal": "BUY|SELL|CLOSE|HOLD",\n'
        '    "reason": "结合20周期趋势+历史准确率的分析(50字内)",\n'
        '    "stop_loss": 具体价格（根据ATR、支撑/阻力位、风险收益比综合确定，详见规则9️⃣）,\n'
        '    "take_profit": 具体价格（根据ATR、支撑/阻力位、风险收益比综合确定，详见规则9️⃣）,\n'
        '    "confidence": "HIGH|MEDIUM|LOW",\n'
        f"    \"leverage\": {config['leverage_min']}-{config['leverage_max']}范围整数（CLOSE信号时可省略）,\n"
        '    "order_quantity": 从建议表中对应【信心等级+杠杆倍数】行的数量（完全复制，6位小数）（CLOSE信号时可省略）\n'
        "  }\n"
        "  ---",
    ]

    return "\n".join(prompt_sections)


# ==================== 系统提示词构建 ====================


def build_system_prompt(config: Dict) -> str:
    """构建系统提示词"""
    return f"""你是专业的加密货币量化交易分析师，擅长多维度技术分析和风险控制。

【你的专长】
- 精通多时间周期趋势分析（SMA/EMA均线系统）
- 擅长多指标共振分析（RSI双周期、MACD完整系统、布林带、ATR波动率）
- 理解市场微观结构（成交量分析、资金费率、持仓量）
- 具备风险管理意识（ATR动态止损、仓位管理）

【分析原则】
1. 多指标验证：不依赖单一指标，寻找多个指标共振
2. 趋势为王：顺势交易，在明确趋势中寻找高概率机会
3. 风险优先：考虑ATR波动率，动态调整止损位置
4. 数据驱动：基于历史准确率统计，优化决策质量
5. 市场情绪：结合资金费率和持仓量判断市场情绪

【当前任务】
分析 {config['display']} 的 {config['timeframe']} 周期数据，给出交易决策。
严格按照JSON格式返回，包含所有必需字段。

【决策要求】
- HIGH信心：多个指标强烈共振（均线、RSI双周期、MACD、成交量、ATR）
- MEDIUM信心：2-3个指标支持，存在分歧但方向明确
- LOW信心：指标分歧严重或盘整震荡
- 注意：根据历史统计，HIGH准确率最高，应积极寻找高确定性机会

【⚠️ 重要：多空平衡与平仓管理】
这是永续合约双向交易系统，必须平衡做多、做空和平仓：
- BUY：当技术指标显示上涨趋势时（价格>均线、RSI上升、MACD金叉、放量等）
- SELL：当技术指标显示下跌趋势时（价格<均线、RSI下降、MACD死叉、资金费率负值等）
- CLOSE：⭐ 当有持仓且应该平仓时使用（趋势反转、触及止盈止损、技术指标衰竭）
- HOLD：只在技术指标严重分歧或震荡时使用

⚠️ 不要只关注做多机会！下跌趋势同样是交易机会！
当看到明确的下跌信号时（如：价格跌破EMA20/50、MACD死叉、RSI<40、成交量放大），应果断选择SELL做空。
SELL不是平仓，而是开空仓获利的机会！

⭐ 平仓时机管理：
- 持有多头时，如出现明确下跌信号（死叉、跌破支撑），应选择CLOSE平仓，而不是等待止损
- 持有空头时，如出现明确上涨信号（金叉、突破阻力），应选择CLOSE平仓，而不是等待止损
- 当盈利达到止盈目标附近（如>3%），也应考虑CLOSE落袋为安
- CLOSE信号可以避免被动止损，实现主动风险控制"""
