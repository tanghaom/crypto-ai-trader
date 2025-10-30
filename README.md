# OKX AI 量化交易机器人

基于 DeepSeek/Qwen 大语言模型的加密货币量化交易系统，支持多交易对、多策略的自动化交易。

## 📋 项目简介

本项目是一个集成了大语言模型（LLM）智能分析的加密货币量化交易机器人，通过 OKX 交易所 API 进行实盘交易。系统采用 AI 驱动的技术分析方法，结合传统技术指标（MA、EMA、RSI、MACD、布林带等），实现自动化的交易决策和风险管理。

### 核心特性

- 🤖 **AI 驱动**: 支持 DeepSeek 和 Qwen 大模型进行市场分析
- 📊 **多交易对**: 支持同时监控和交易多个加密货币交易对
- 🔄 **自动化交易**: 自动执行开仓、平仓、止盈止损操作
- 💰 **智能仓位管理**: 基于信心等级的动态仓位分配（高/中/低）
- 📈 **技术指标分析**: 集成多种技术指标（SMA、EMA、RSI、MACD、布林带、ATR）
- 🌐 **Web 监控面板**: 实时查看交易状态、持仓情况和盈亏统计
- 🗄️ **历史记录**: SQLite 数据库记录所有交易历史
- 🛡️ **风险控制**: 多层风险管理机制，包括最大保证金比例、安全缓冲等

## 🚀 快速开始

### 环境要求

- Python 3.8+
- OKX 交易所账户（API Key）
- DeepSeek 或 阿里云百炼（Qwen）API Key

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置环境变量

创建 `.env` 文件并配置以下变量：

```env
# OKX API 配置
# DeepSeek 模型使用的 OKX 账户
OKX_API_KEY_DEEPSEEK=your-okx-api-key-for-deepseek
OKX_SECRET_DEEPSEEK=your-okx-secret-for-deepseek
OKX_PASSWORD_DEEPSEEK=your-okx-password-for-deepseek
# 如果使用子账户
# OKX_SUBACCOUNT_DEEPSEEK=DeepSeek策略账户

# Qwen 模型使用的 OKX 账户
OKX_API_KEY_QWEN=your-okx-api-key-for-qwen
OKX_SECRET_QWEN=your-okx-secret-for-qwen
OKX_PASSWORD_QWEN=your-okx-password-for-qwen
# 如果使用子账户
# OKX_SUBACCOUNT_QWEN=Qwen策略账户

# AI 模型配置
# DeepSeek API 配置
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# Qwen (通义千问) API 配置
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
QWEN_MODEL=qwen-max
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 启用模型
ENABLED_MODELS=deepseek,qwen
```

### 启动系统

运行以下命令即可同时启动交易机器人和 Web 监控面板：

```bash
python web_server.py
```

然后在浏览器访问 `http://localhost:8080` 查看实时监控面板。

> 💡 **提示**: `web_server.py` 会自动启动交易机器人，无需单独运行 `deepseekok2.py`

## 📁 项目结构

```
alpha-okx-deepseek-qwen/
├── deepseekok2.py              # 主交易程序
├── web_server.py               # Web 监控服务器
├── requirements.txt            # 项目依赖
├── .env                        # 环境变量配置（需创建）
├── .gitignore                  # Git 忽略配置
│
├── scripts/                    # 工具脚本
│   ├── export_history.py      # 导出交易历史到 Excel
│   └── manual_force_close.py  # 手动强制平仓工具
│
├── templates/                  # Web 界面模板
│   └── index.html             # 监控面板 HTML
│
├── static/                     # 静态资源
│   ├── css/
│   │   └── style.css          # 样式文件
│   └── js/
│       └── app.js             # 前端交互逻辑
│
├── data/                       # 数据目录（自动生成）
│   └── history.db             # 交易历史数据库
│
└── archives/                   # 归档目录（自动生成）
    └── balances-*.xlsx        # 历史余额导出文件
```

## ⚙️ 配置说明

### 交易对配置

在 `deepseekok2.py` 中修改 `TRADE_CONFIGS` 字典来配置交易对及其参数：

```python
TRADE_CONFIGS = {
    'BTC-USDT-SWAP': {
        'leverage': 10,          # 杠杆倍数
        'interval': 5,           # 分析间隔（分钟）
        'short_window': 20,      # 短期窗口
        'long_window': 50        # 长期窗口
    },
    # 添加更多交易对...
}
```

### 仓位配置

根据风险偏好调整仓位比例（在 `deepseekok2.py` 开头）：

```python
CONFIDENCE_RATIOS = {
    'HIGH': 0.3,    # 高信心：30% 可用保证金
    'MEDIUM': 0.2,  # 中信心：20% 可用保证金
    'LOW': 0.05     # 低信心：5% 可用保证金
}
```

### 风险控制参数

```python
MAX_TOTAL_MARGIN_RATIO = 0.85  # 总保证金不超过权益的 85%
MARGIN_SAFETY_BUFFER = 0.90    # 安全缓冲 90%
```

## 🔧 工具脚本

### 导出交易历史

```bash
python scripts/export_history.py
```

将数据库中的交易历史导出为 Excel 文件，保存到 `archives/` 目录。

### 手动强制平仓

```bash
python scripts/manual_force_close.py
```

紧急情况下手动平掉所有持仓。

## 📊 AI 分析机制

系统会定期（默认 5 分钟）对每个交易对进行 AI 分析：

1. **数据采集**: 获取最新的价格、技术指标数据
2. **AI 分析**: 将市场数据发送给大模型进行分析
3. **信号生成**: AI 返回 LONG/SHORT/CLOSE/HOLD 信号及信心等级
4. **风控检查**: 验证保证金、仓位限制等风险参数
5. **执行交易**: 根据信号执行相应的交易操作
6. **记录存储**: 将交易记录保存到数据库

### AI 输出格式

AI 模型需要返回 JSON 格式的分析结果：

```json
{
  "action": "LONG",
  "confidence": "HIGH",
  "analysis": "市场分析说明...",
  "entry_price": 42000.00,
  "tp_price": 43000.00,
  "sl_price": 41000.00
}
```

## 🛡️ 风险提示

⚠️ **重要警告**：

- 加密货币交易具有高风险，可能导致本金损失
- 本项目仅供学习和研究使用
- 实盘交易前请充分测试和理解代码逻辑
- 请根据自身风险承受能力调整仓位和杠杆
- 建议从小资金开始测试
- 使用前请务必理解所有配置参数的含义

## 📝 开发建议

### 模型选择

- **DeepSeek**: 推理能力强，适合复杂市场分析
- **Qwen**: 响应速度快，适合高频交易场景

### 性能优化

- 使用多线程并发处理多个交易对
- 缓存市场数据减少 API 调用
- 数据库连接池管理
- AI 调用频率控制

### 监控建议

- 定期查看 Web 面板了解运行状态
- 关注资金使用率和盈亏情况
- 监控 AI 分析错误率
- 定期导出交易历史进行复盘

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目仅供学习交流使用，使用者需自行承担交易风险。

## 🔗 相关资源

- [OKX API 文档](https://www.okx.com/docs-v5/zh/)
- [DeepSeek 文档](https://platform.deepseek.com/docs)
- [阿里云百炼文档](https://help.aliyun.com/zh/model-studio/)
- [CCXT 库文档](https://docs.ccxt.com/)

---

**最后更新**: 2025-10-30

