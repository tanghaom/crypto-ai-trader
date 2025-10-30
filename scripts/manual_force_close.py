#!/usr/bin/env python3
# 临时强制平仓脚本：用于在主程序运行期间手动关闭指定交易对的持仓

import argparse
from datetime import datetime
from pathlib import Path
import sys

# 确保可以从项目根目录导入 deepseekok2
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deepseekok2 import (
    MODEL_CONTEXTS,
    DEFAULT_MODEL_KEY,
    activate_context,
    get_current_position,
    contracts_to_base,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="强制使用 reduceOnly 平仓指定交易对的现有持仓。"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_KEY,
        help=f"目标模型标识，可选值：{', '.join(MODEL_CONTEXTS.keys())}（默认：{DEFAULT_MODEL_KEY}）",
    )
    parser.add_argument(
        "--symbol",
        required=True,
        help="需要平仓的交易对，例如 BTC/USDT:USDT",
    )
    parser.add_argument(
        "--tag",
        default="60bb4a8d3416BCDE",
        help="订单标识 tag；默认为主程序使用的 60bb4a8d3416BCDE，可传空字符串以取消。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将要执行的操作，不真正提交订单。",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="减少输出，仅在成功或失败时打印关键信息。",
    )
    return parser.parse_args()


def log(message: str, quiet: bool = False) -> None:
    if not quiet:
        print(message)


def main() -> int:
    args = parse_args()

    if args.model not in MODEL_CONTEXTS:
        print(f"❌ 未找到模型 '{args.model}'，可用模型：{', '.join(MODEL_CONTEXTS.keys())}")
        return 1

    ctx = MODEL_CONTEXTS[args.model]
    log(f"🧠 使用模型：{ctx.display}（标识：{ctx.key}）", args.quiet)
    log(f"🎯 目标交易对：{args.symbol}", args.quiet)

    with activate_context(ctx):
        position = get_current_position(args.symbol)
        if not position:
            print(f"ℹ️ {args.symbol} 当前无持仓，无需平仓。")
            return 0

        size_contracts = float(position.get("size") or 0)
        if size_contracts <= 0:
            print(f"ℹ️ {args.symbol} 持仓合约数为 0，跳过。")
            return 0

        base_qty = contracts_to_base(args.symbol, size_contracts)
        side = position.get("side")
        order_side = "buy" if side == "short" else "sell"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log(
            f"[{timestamp}] ⚙️ 准备平{side or '未知'}仓 {size_contracts:.6f} 张"
            f"（≈ {base_qty:.6f} 基础资产），订单方向：{order_side.upper()}，使用 reduceOnly。",
            args.quiet,
        )

        if args.dry_run:
            print("✅ dry-run 模式，仅展示计划，不提交订单。")
            return 0

        params = {"reduceOnly": True}
        if args.tag:
            params["tag"] = args.tag

        try:
            ctx.exchange.create_market_order(
                args.symbol,
                order_side,
                size_contracts,
                params=params,
            )
            print("✅ 已提交 reduceOnly 平仓订单。")
            return 0
        except Exception as exc:  # 捕获所有异常，便于快速反馈
            print(f"❌ 平仓失败：{exc}")
            return 2


if __name__ == "__main__":
    sys.exit(main())
