from datetime import datetime
from flask import Flask, jsonify, render_template, request, abort
from flask_cors import CORS
import threading
import sys
import os
import logging

# 获取当前文件所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 导入主程序
import deepseekok2

# 明确指定模板和静态文件路径
app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))
DEFAULT_MODEL = deepseekok2.DEFAULT_MODEL_KEY
CORS(app)


def get_snapshot(model_key: str):
    try:
        return deepseekok2.get_model_snapshot(model_key)
    except KeyError:
        abort(404, description=f"模型 {model_key} 未配置")


def resolve_model_key():
    model_key = request.args.get('model', DEFAULT_MODEL)
    if model_key not in deepseekok2.MODEL_CONTEXTS:
        abort(404, description=f"模型 {model_key} 未配置")
    return model_key

@app.route('/')
def index():
    """主页"""
    try:
        return render_template('index.html')
    except Exception as e:
        return f"<h1>模板加载错误</h1><p>{str(e)}</p><p>模板路径: {app.template_folder}</p>"

@app.route('/api/dashboard')
def get_dashboard_data():
    """获取所有交易对的仪表板数据"""
    model_key = resolve_model_key()
    snapshot = get_snapshot(model_key)
    try:
        symbols_data = []
        for symbol, config in deepseekok2.TRADE_CONFIGS.items():
            symbol_data = snapshot['symbols'].get(symbol, {})
            symbols_data.append({
                'symbol': symbol,
                'display': config['display'],
                'current_price': symbol_data.get('current_price', 0),
                'current_position': symbol_data.get('current_position'),
                'performance': symbol_data.get('performance', {}),
                'analysis_records': symbol_data.get('analysis_records', []),
                'last_update': symbol_data.get('last_update'),
                'config': {
                    'timeframe': config['timeframe'],
                    'test_mode': config.get('test_mode', True),
                    'leverage_range': f"{config['leverage_min']}-{config['leverage_max']}"
                }
            })

        data = {
            'model': model_key,
            'display': snapshot['display'],
            'symbols': symbols_data,
            'ai_model_info': snapshot['ai_model_info'],
            'account_summary': snapshot['account_summary'],
            'balance_history': snapshot.get('balance_history', [])
        }
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/kline')
def get_kline_data():
    """获取K线数据 - 支持symbol参数"""
    model_key = resolve_model_key()
    snapshot = get_snapshot(model_key)
    try:
        symbol = request.args.get('symbol', 'BTC/USDT:USDT')
        if symbol in snapshot['symbols']:
            return jsonify(snapshot['symbols'][symbol].get('kline_data', []))
        return jsonify([])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/trades')
def get_trade_history():
    """获取交易历史 - 支持symbol参数"""
    model_key = resolve_model_key()
    snapshot = get_snapshot(model_key)
    try:
        symbol = request.args.get('symbol')
        if symbol and symbol in snapshot['symbols']:
            return jsonify(snapshot['symbols'][symbol].get('trade_history', []))

        # 返回所有交易对的交易历史
        all_trades = {}
        for sym in deepseekok2.TRADE_CONFIGS.keys():
            all_trades[sym] = snapshot['symbols'][sym].get('trade_history', [])
        return jsonify(all_trades)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai_decisions')
def get_ai_decisions():
    """获取AI决策历史 - 支持symbol参数"""
    model_key = resolve_model_key()
    snapshot = get_snapshot(model_key)
    try:
        symbol = request.args.get('symbol')
        if symbol and symbol in snapshot['symbols']:
            return jsonify(snapshot['symbols'][symbol].get('ai_decisions', []))

        # 返回所有交易对的AI决策
        all_decisions = {}
        for sym in deepseekok2.TRADE_CONFIGS.keys():
            all_decisions[sym] = snapshot['symbols'][sym].get('ai_decisions', [])
        return jsonify(all_decisions)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/signals')
def get_signal_history():
    """获取信号历史统计 - 支持symbol参数"""
    model_key = resolve_model_key()
    snapshot = get_snapshot(model_key)
    try:
        symbol = request.args.get('symbol')

        signal_map = snapshot.get('signal_history', {})

        if symbol and symbol in signal_map:
            signals = signal_map[symbol]
        else:
            # 合并所有交易对的信号
            signals = []
            for sym_signals in signal_map.values():
                signals.extend(sym_signals)

        # 统计信号分布
        signal_stats = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        confidence_stats = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}

        for signal in signals:
            signal_type = signal.get('signal', 'HOLD')
            confidence = signal.get('confidence', 'LOW')
            signal_stats[signal_type] = signal_stats.get(signal_type, 0) + 1
            confidence_stats[confidence] = confidence_stats.get(confidence, 0) + 1

        return jsonify({
            'signal_stats': signal_stats,
            'confidence_stats': confidence_stats,
            'total_signals': len(signals),
            'recent_signals': signals[-10:] if signals else []
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/profit_curve')
def get_profit_curve():
    """获取模型的总金额曲线，支持按范围筛选"""
    model_key = resolve_model_key()
    range_key = request.args.get('range', '7d')
    try:
        start_ts, end_ts = deepseekok2.resolve_time_range(range_key)
        data = deepseekok2.history_store.fetch_balance_range(model_key, start_ts, end_ts)
        if not data:
            snapshot = get_snapshot(model_key)
            data = snapshot.get('balance_history', [])
        return jsonify({
            'model': model_key,
            'range': range_key,
            'series': data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai_model_info')
def get_ai_model_info():
    """获取AI模型信息和连接状态"""
    try:
        return jsonify(deepseekok2.get_models_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/test_ai')
def test_ai_connection():
    """手动测试AI连接"""
    model_key = request.args.get('model')
    try:
        result = deepseekok2.test_ai_connection(model_key) if model_key else deepseekok2.test_ai_connection()
        statuses = deepseekok2.get_models_status()
        if isinstance(result, dict):
            success = all(result.values())
        else:
            success = bool(result)
        return jsonify({
            'success': success,
            'detail': result,
            'statuses': statuses
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/overview')
def get_overview_data():
    """首页总览数据（含多模型资金曲线）"""
    range_key = request.args.get('range', '1d')
    try:
        payload = deepseekok2.get_overview_payload(range_key)
        payload['models_metadata'] = deepseekok2.get_model_metadata()
        return jsonify(payload)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models')
def list_models():
    """返回模型列表与基础信息"""
    try:
        return jsonify({
            'default': deepseekok2.DEFAULT_MODEL_KEY,
            'models': deepseekok2.get_model_metadata()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def initialize_data():
    """启动时立即初始化所有交易对数据"""
    try:
        print("\n正在初始化多模型数据...")

        # 逐模型进行一次完整轮询
        for model_key in deepseekok2.MODEL_ORDER:
            ctx = deepseekok2.MODEL_CONTEXTS[model_key]
            print(f"→ {ctx.display} 初始化")
            with deepseekok2.activate_context(ctx):
                deepseekok2.run_all_symbols_parallel(ctx.display)
                deepseekok2.capture_balance_snapshot(ctx)
                deepseekok2.refresh_overview_from_context(ctx)

        deepseekok2.record_overview_point(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print("✅ 初始化完成\n")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()

def run_trading_bot():
    """在独立线程中运行交易机器人"""
    deepseekok2.main()

if __name__ == '__main__':
    # 立即初始化数据
    print("\n" + "="*60)
    print("🚀 启动多交易对交易机器人Web监控...")
    print("="*60 + "\n")

    initialize_data()

    # 启动交易机器人线程
    bot_thread = threading.Thread(target=run_trading_bot, daemon=True)
    bot_thread.start()

    # 禁用Flask/Werkzeug的HTTP请求日志输出
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    # 启动Web服务器
    PORT = 8080
    print("\n" + "="*60)
    print("🌐 Web管理界面启动成功！")
    print(f"📊 访问地址: http://localhost:{PORT}")
    print(f"📁 模板目录: {app.template_folder}")
    print(f"📁 静态目录: {app.static_folder}")
    print("="*60 + "\n")

    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)
