// ==================== 全局状态 ====================
let homeBalanceChart = null;
let modelBalanceChart = null;
let signalChart = null;
let confidenceChart = null;

let currentView = 'home';
let currentModel = null;
let modelsMetadata = [];
let symbolsData = {};
let currentSymbol = 'BTC/USDT:USDT';

let homeRange = '1d';
let modelRange = '7d';

// ==================== 初始化 ====================
document.addEventListener('DOMContentLoaded', async () => {
    await loadModelsMetadata();
    initCharts();
    initRangeSelectors();
    initEventHandlers();
    switchView('home');
    setInterval(pollData, 10000);
});

async function loadModelsMetadata() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        modelsMetadata = data.models || [];
        currentModel = data.default || (modelsMetadata[0] ? modelsMetadata[0].key : null);

        buildViewTabs();
        buildHomeCards();
    } catch (error) {
        console.error('加载模型信息失败:', error);
    }
}

function initCharts() {
    modelBalanceChart = echarts.init(document.getElementById('modelBalanceChart'));
    homeBalanceChart = echarts.init(document.getElementById('homeBalanceChart'));
    signalChart = echarts.init(document.getElementById('signalChart'));
    confidenceChart = echarts.init(document.getElementById('confidenceChart'));

    window.addEventListener('resize', () => {
        modelBalanceChart.resize();
        homeBalanceChart.resize();
        signalChart.resize();
        confidenceChart.resize();
    });
}

function initRangeSelectors() {
    const homeSelector = document.getElementById('homeRangeSelector');
    const modelSelector = document.getElementById('modelRangeSelector');

    if (homeSelector) {
        homeSelector.addEventListener('click', (event) => {
            const btn = event.target.closest('[data-range]');
            if (!btn) return;
            homeRange = btn.getAttribute('data-range');
            setActiveRangeButton(homeSelector, homeRange);
            updateHomeOverview();
        });
        setActiveRangeButton(homeSelector, homeRange);
    }

    if (modelSelector) {
        modelSelector.addEventListener('click', (event) => {
            const btn = event.target.closest('[data-range]');
            if (!btn) return;
            modelRange = btn.getAttribute('data-range');
            setActiveRangeButton(modelSelector, modelRange);
            updateModelBalanceChart();
        });
        setActiveRangeButton(modelSelector, modelRange);
    }
}

function initEventHandlers() {
    const homeTable = document.getElementById('homeModelTableBody');
    if (homeTable) {
        homeTable.addEventListener('click', (event) => {
            const row = event.target.closest('[data-model]');
            if (!row) return;
            const modelKey = row.getAttribute('data-model');
            switchView('model', modelKey);
        });
    }
}

// ==================== 视图切换 ====================
function buildViewTabs() {
    const container = document.getElementById('viewTabs');
    if (!container) return;

    container.innerHTML = '';

    const homeBtn = document.createElement('button');
    homeBtn.className = 'view-tab active';
    homeBtn.textContent = '首页';
    homeBtn.dataset.view = 'home';
    container.appendChild(homeBtn);

    modelsMetadata.forEach((meta) => {
        const btn = document.createElement('button');
        btn.className = 'view-tab';
        btn.textContent = meta.display;
        btn.dataset.view = 'model';
        btn.dataset.model = meta.key;
        container.appendChild(btn);
    });

    container.addEventListener('click', (event) => {
        const btn = event.target.closest('.view-tab');
        if (!btn) return;
        const view = btn.dataset.view;
        if (view === 'home') {
            switchView('home');
        } else {
            switchView('model', btn.dataset.model);
        }
    });
}

function setActiveViewTab(view, modelKey = null) {
    document.querySelectorAll('.view-tab').forEach((tab) => {
        tab.classList.remove('active');
        if (tab.dataset.view === view) {
            if (view === 'home' || tab.dataset.model === modelKey) {
                tab.classList.add('active');
            }
        }
    });
}

function switchView(view, modelKey = null) {
    currentView = view;
    const homeView = document.getElementById('homeView');
    const modelView = document.getElementById('modelView');

    if (view === 'home') {
        if (homeView) homeView.classList.remove('hidden');
        if (modelView) modelView.classList.add('hidden');
        updateHomeOverview();
    } else {
        currentModel = modelKey || currentModel;
        if (homeView) homeView.classList.add('hidden');
        if (modelView) modelView.classList.remove('hidden');
        resetModelState();
        updateModelView(true);
    }

    setActiveViewTab(view, currentModel);
}

function resetModelState() {
    symbolsData = {};
    currentSymbol = 'BTC/USDT:USDT';
    document.getElementById('symbolTabs').innerHTML = '';
}

// ==================== home 视图 ====================
function buildHomeCards() {
    const container = document.getElementById('homeCards');
    if (!container) return;

    container.innerHTML = `
        <div class="card home-card">
            <div class="card-header">
                <i class="fas fa-coins"></i>
                <h3>总资金</h3>
            </div>
            <div class="card-body">
                <div class="info-item">
                    <span class="label">总权益</span>
                    <span class="value" id="homeTotalEquity">--</span>
                </div>
                <div class="info-item">
                    <span class="label">模型数量</span>
                    <span class="value" id="homeModelCount">--</span>
                </div>
            </div>
        </div>
    `;

    modelsMetadata.forEach((meta) => {
        const card = document.createElement('div');
        card.className = 'card home-card';
        card.dataset.modelCard = meta.key;
        card.innerHTML = `
            <div class="card-header">
                <i class="fas fa-robot"></i>
                <h3>${meta.display}</h3>
            </div>
            <div class="card-body">
                <div class="info-item">
                    <span class="label">总权益</span>
                    <span class="value" data-field="equity">--</span>
                </div>
                <div class="info-item">
                    <span class="label">可用余额</span>
                    <span class="value" data-field="available">--</span>
                </div>
                <div class="info-item">
                    <span class="label">24h变化</span>
                    <span class="value change" data-field="change">--</span>
                </div>
            </div>
        `;
        container.appendChild(card);
    });
}

async function updateHomeOverview() {
    try {
        const response = await fetch(`/api/overview?range=${homeRange}`);
        const data = await response.json();

        updateHomeCards(data);
        renderHomeChart(data);
        renderHomeTable(data);
    } catch (error) {
        console.error('更新首页概览失败:', error);
    }
}

function updateHomeCards(data) {
    const totalEquityEl = document.getElementById('homeTotalEquity');
    const modelCountEl = document.getElementById('homeModelCount');
    if (totalEquityEl) totalEquityEl.textContent = formatCurrency(data.aggregate.total_equity || 0);
    if (modelCountEl) modelCountEl.textContent = modelsMetadata.length;

    modelsMetadata.forEach((meta) => {
        const summary = data.models ? data.models[meta.key] : null;
        const card = document.querySelector(`[data-model-card="${meta.key}"]`);
        if (!card || !summary) return;

        card.querySelector('[data-field="equity"]').textContent = formatCurrency(summary.latest_equity || 0);
        card.querySelector('[data-field="available"]').textContent = formatCurrency(summary.available_balance || 0);
        const changeEl = card.querySelector('[data-field="change"]');
        changeEl.textContent = formatChange(summary.change_abs, summary.change_pct);
        setChangeClass(changeEl, summary.change_abs);
    });
}

function renderHomeChart(data) {
    if (!homeBalanceChart) return;
    const series = [];
    const legends = [];

    modelsMetadata.forEach((meta) => {
        const points = (data.series && data.series[meta.key]) || [];
        legends.push(meta.display);
        series.push({
            name: meta.display,
            type: 'line',
            smooth: true,
            showSymbol: false,
            data: points.map((item) => [item.timestamp, item.total_equity])
        });
    });

    const option = {
        tooltip: {
            trigger: 'axis',
            formatter: (params) => {
                // 格式化时间显示
                const timestamp = params[0].axisValue;
                const formattedTime = formatTimestamp(timestamp);
                const lines = [formattedTime];
                params.forEach((item) => {
                    lines.push(`${item.marker}${item.seriesName}: ${formatCurrency(item.data[1])}`);
                });
                return lines.join('<br/>');
            }
        },
        legend: {
            data: legends,
            textStyle: { color: '#fff' }
        },
        grid: { left: 40, right: 20, top: 40, bottom: 40 },
        xAxis: {
            type: 'time',
            axisLabel: { 
                color: '#9ca3af',
                formatter: (value) => formatTimestamp(value)
            }
        },
        yAxis: {
            type: 'value',
            axisLabel: { color: '#9ca3af', formatter: (value) => `$${value.toFixed(0)}` }
        },
        series
    };

    homeBalanceChart.setOption(option, true);
}

function renderHomeTable(data) {
    const tbody = document.getElementById('homeModelTableBody');
    if (!tbody) return;

    tbody.innerHTML = '';
    modelsMetadata.forEach((meta) => {
        const summary = data.models ? data.models[meta.key] : null;
        if (!summary) return;

        const tr = document.createElement('tr');
        tr.dataset.model = meta.key;
        tr.innerHTML = `
            <td>${meta.display}</td>
            <td>${formatCurrency(summary.latest_equity || 0)}</td>
            <td>${formatCurrency(summary.available_balance || 0)}</td>
            <td class="${summary.change_abs > 0 ? 'positive' : summary.change_abs < 0 ? 'negative' : ''}">
                ${formatChange(summary.change_abs, summary.change_pct)}
            </td>
            <td>${summary.sub_account || '主账户'}</td>
        `;
        tbody.appendChild(tr);
    });
}

function setActiveRangeButton(container, activeRange) {
    container.querySelectorAll('[data-range]').forEach((btn) => {
        btn.classList.toggle('active', btn.getAttribute('data-range') === activeRange);
    });
}

// ==================== 模型视图 ====================
async function updateModelView(resetSymbol = false) {
    if (!currentModel) return;
    if (resetSymbol) currentSymbol = 'BTC/USDT:USDT';

    await Promise.all([
        updateAIModelInfo(),
        updateDashboard(),
        updateModelBalanceChart(),
        updateAIDecisions(),
        updateTrades(),
        updateSignalStats()
    ]).catch((error) => console.error('更新模型视图失败:', error));
}

async function updateAIModelInfo() {
    try {
        const response = await fetch('/api/ai_model_info');
        const data = await response.json();
        const current = data.find((item) => item.key === currentModel);
        if (!current) return;

        const modelNameMap = {
            deepseek: 'DeepSeek',
            qwen: '阿里百炼 Qwen'
        };

        document.getElementById('aiModelName').textContent =
            `${modelNameMap[current.provider] || current.display} (${current.model_name})`;

        const statusDot = document.getElementById('aiStatusDot');
        const statusText = document.getElementById('aiStatusText');
        statusDot.className = 'status-dot';

        if (current.ai_model_info.status === 'connected') {
            statusDot.classList.add('connected');
            statusText.textContent = '已连接';
            statusText.style.color = 'var(--success-color)';
        } else if (current.ai_model_info.status === 'error') {
            statusDot.classList.add('error');
            statusText.textContent = '连接失败';
            statusText.style.color = 'var(--danger-color)';
        } else {
            statusDot.classList.add('unknown');
            statusText.textContent = '检测中';
            statusText.style.color = 'var(--warning-color)';
        }
    } catch (error) {
        console.error('AI模型信息更新失败:', error);
    }
}

async function updateDashboard() {
    try {
        const response = await fetch(`/api/dashboard?model=${currentModel}`);
        const data = await response.json();

        symbolsData = {};
        data.symbols.forEach((symbolInfo, index) => {
            symbolsData[symbolInfo.symbol] = symbolInfo;
            if (index === 0) currentSymbol = symbolInfo.symbol;
        });

        renderSymbolTabs(data.symbols);
        updateDisplayForCurrentSymbol();

        document.getElementById('usdtBalance').textContent =
            formatCurrency(data.account_summary.available_balance || 0);
        document.getElementById('totalEquity').textContent =
            formatCurrency(data.account_summary.total_equity || 0);
        document.getElementById('lastUpdate').textContent = data.account_summary.last_update || '--';
    } catch (error) {
        console.error('仪表板更新失败:', error);
    }
}

async function updateModelBalanceChart() {
    try {
        const response = await fetch(`/api/profit_curve?model=${currentModel}&range=${modelRange}`);
        const data = await response.json();
        const seriesData = (data.series || []).map((item) => [item.timestamp, item.total_equity]);

        const option = {
            tooltip: {
                trigger: 'axis',
                formatter: (params) => {
                    if (!params[0]) return '';
                    const timestamp = params[0].axisValue;
                    const formattedTime = formatTimestamp(timestamp);
                    return `${formattedTime}<br/>${formatCurrency(params[0].data[1])}`;
                }
            },
            grid: { left: 40, right: 20, top: 30, bottom: 40 },
            xAxis: {
                type: 'time',
                axisLabel: { 
                    color: '#9ca3af',
                    formatter: (value) => formatTimestamp(value)
                }
            },
            yAxis: {
                type: 'value',
                axisLabel: { color: '#9ca3af', formatter: (value) => `$${value.toFixed(0)}` }
            },
            series: [
                {
                    name: '总金额',
                    type: 'line',
                    smooth: true,
                    showSymbol: false,
                    areaStyle: {
                        color: 'rgba(26, 115, 232, 0.15)'
                    },
                    data: seriesData
                }
            ]
        };

        modelBalanceChart.setOption(option, true);
    } catch (error) {
        console.error('模型金额曲线更新失败:', error);
    }
}

async function updateAIDecisions() {
    try {
        const response = await fetch(`/api/ai_decisions?model=${currentModel}&symbol=${encodeURIComponent(currentSymbol)}`);
        const decisions = await response.json();
        const container = document.getElementById('aiDecisionList');
        if (!container) return;

        container.innerHTML = decisions.slice(-10).reverse().map((decision) => `
            <div class="decision-item">
                <div class="decision-header">
                    <span class="decision-signal ${decision.signal.toLowerCase()}">${decision.signal}</span>
                    <span class="decision-confidence">${decision.confidence || '--'}</span>
                </div>
                <div class="decision-body">
                    <p>${decision.reason || '无理由说明'}</p>
                    <div class="decision-meta">
                        <span>价格: ${formatCurrency(decision.price || 0)}</span>
                        <span>时间: ${decision.timestamp || '--'}</span>
                    </div>
                </div>
            </div>
        `).join('');
    } catch (error) {
        console.error('AI决策更新失败:', error);
    }
}

async function updateTrades() {
    try {
        const response = await fetch(`/api/trades?model=${currentModel}&symbol=${encodeURIComponent(currentSymbol)}`);
        const trades = await response.json();
        const container = document.getElementById('tradeHistory');
        if (!container) return;

        container.innerHTML = trades.slice(-15).reverse().map((trade) => {
            // 显示交易类型（如果有）或信号（向后兼容）
            const tradeTypeDisplay = trade.trade_type_display || 
                                    (trade.trade_type === 'open_long' ? '开多仓' :
                                     trade.trade_type === 'open_short' ? '开空仓' :
                                     trade.trade_type === 'add_long' ? '加多仓' :
                                     trade.trade_type === 'add_short' ? '加空仓' :
                                     trade.trade_type === 'reverse_long_to_short' ? '反转（平多→开空）' :
                                     trade.trade_type === 'reverse_short_to_long' ? '反转（平空→开多）' :
                                     trade.signal || '--');
            
            const side = trade.side || (trade.signal === 'BUY' ? 'long' : trade.signal === 'SELL' ? 'short' : 'neutral');
            
            return `
            <div class="trade-item">
                <div class="trade-header">
                    <span class="trade-side ${side}">${tradeTypeDisplay}</span>
                    <span class="trade-price">${formatCurrency(trade.price || 0)}</span>
                </div>
                <div class="trade-body">
                    <span>数量: ${trade.amount || '--'}</span>
                    <span>杠杆: ${trade.leverage || '--'}x</span>
                    <span>时间: ${trade.timestamp || '--'}</span>
                </div>
            </div>
        `;
        }).join('');
    } catch (error) {
        console.error('交易记录更新失败:', error);
    }
}

async function updateSignalStats() {
    try {
        const response = await fetch(`/api/signals?model=${currentModel}&symbol=${encodeURIComponent(currentSymbol)}`);
        const data = await response.json();

        const signalOption = {
            tooltip: { trigger: 'item' },
            legend: { show: false },
            series: [
                {
                    name: '信号分布',
                    type: 'pie',
                    radius: ['45%', '70%'],
                    itemStyle: { borderRadius: 5, borderColor: '#0f0f23', borderWidth: 2 },
                    label: { color: '#fff' },
                    data: [
                        { value: data.signal_stats.BUY, name: 'BUY' },
                        { value: data.signal_stats.SELL, name: 'SELL' },
                        { value: data.signal_stats.HOLD, name: 'HOLD' }
                    ]
                }
            ]
        };
        signalChart.setOption(signalOption, true);

        const confidenceOption = {
            xAxis: {
                type: 'category',
                data: ['HIGH', 'MEDIUM', 'LOW'],
                axisLabel: { color: '#9ca3af' }
            },
            yAxis: {
                type: 'value',
                axisLabel: { color: '#9ca3af' }
            },
            grid: { left: 40, right: 20, top: 20, bottom: 30 },
            series: [
                {
                    data: [
                        data.confidence_stats.HIGH,
                        data.confidence_stats.MEDIUM,
                        data.confidence_stats.LOW
                    ],
                    type: 'bar',
                    itemStyle: {
                        color: (params) => ['#34a853', '#1a73e8', '#fbbc04'][params.dataIndex]
                    }
                }
            ]
        };
        confidenceChart.setOption(confidenceOption, true);
    } catch (error) {
        console.error('信号统计更新失败:', error);
    }
}

function renderSymbolTabs(symbols) {
    const tabsContainer = document.getElementById('symbolTabs');
    if (!tabsContainer) return;

    const tabsHTML = symbols.map((symbolData, index) => {
        const isActive = (index === 0 && !symbolsData[currentSymbol]) || symbolData.symbol === currentSymbol ? 'active' : '';
        if (index === 0 && (!currentSymbol || !symbolsData[currentSymbol])) {
            currentSymbol = symbolData.symbol;
        }

        const priceChange = symbolData.performance?.price_change || 0;
        const changeClass = priceChange >= 0 ? 'positive' : 'negative';
        const changeSign = priceChange >= 0 ? '+' : '';

        return `
            <div class="symbol-tab ${isActive}" data-symbol="${symbolData.symbol}" onclick="switchSymbol('${symbolData.symbol}')">
                <span class="symbol-name">${symbolData.display}</span>
                <span class="symbol-price">$${(symbolData.current_price || 0).toFixed(2)}</span>
                <span class="symbol-change ${changeClass}">${changeSign}${priceChange.toFixed(2)}%</span>
            </div>
        `;
    }).join('');

    tabsContainer.innerHTML = tabsHTML;
}

function switchSymbol(symbol) {
    currentSymbol = symbol;
    document.querySelectorAll('.symbol-tab').forEach((tab) => {
        tab.classList.toggle('active', tab.dataset.symbol === symbol);
    });
    updateDisplayForCurrentSymbol();
    updateAIDecisions();
    updateTrades();
    updateSignalStats();
}

function updateDisplayForCurrentSymbol() {
    const data = symbolsData[currentSymbol];
    if (!data) return;

    document.getElementById('currentSymbol').textContent = data.display;

    const leverageText = data.config?.leverage_range ||
        (data.performance?.current_leverage ? `${data.performance.current_leverage}x` : '--');
    document.getElementById('leverage').textContent = leverageText;
    document.getElementById('timeframe').textContent = data.config?.timeframe || '--';
    document.getElementById('tradeMode').textContent = data.config?.test_mode ? '模拟模式' : '实盘模式';

    // 更新价格显示，包含价格变化百分比
    if (data.current_price) {
        const priceElement = document.getElementById('currentPrice');
        const priceChangeElement = document.getElementById('priceChange');
        const priceChange = data.performance?.price_change || 0;
        const priceChangeSign = priceChange >= 0 ? '+' : '';
        const priceChangeClass = priceChange >= 0 ? 'positive' : priceChange < 0 ? 'negative' : '';
        
        priceElement.textContent =
            `$${Number(data.current_price).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
        
        if (priceChangeElement) {
            priceChangeElement.textContent = `${priceChangeSign}${priceChange.toFixed(2)}%`;
            priceChangeElement.className = `price-change ${priceChangeClass}`;
        }
    }

    if (data.current_position) {
        const pos = data.current_position;
        const posType = document.getElementById('positionType');
        posType.textContent = pos.side === 'long' ? '多头持仓' : '空头持仓';
        posType.className = `position-type ${pos.side}`;

        const assetName = data.display.split('-')[0];
        document.getElementById('positionSize').textContent = `${pos.size} ${assetName}`;
        document.getElementById('entryPrice').textContent = `$${pos.entry_price.toFixed(2)}`;

        // 计算持仓盈亏百分比
        const currentPrice = data.current_price || 0;
        let pnlPercent = 0;
        if (pos.entry_price && pos.entry_price > 0) {
            if (pos.side === 'long') {
                // 多头持仓：盈亏百分比 = (当前价格 - 入场价格) / 入场价格 * 100
                pnlPercent = ((currentPrice - pos.entry_price) / pos.entry_price) * 100;
            } else {
                // 空头持仓：盈亏百分比 = (入场价格 - 当前价格) / 入场价格 * 100
                pnlPercent = ((pos.entry_price - currentPrice) / pos.entry_price) * 100;
            }
        }

        const pnlElement = document.getElementById('unrealizedPnl');
        const pnlSign = pos.unrealized_pnl >= 0 ? '+' : '';
        const pnlPercentSign = pnlPercent >= 0 ? '+' : '';
        pnlElement.textContent = `${pnlSign}${pos.unrealized_pnl.toFixed(2)} USDT (${pnlPercentSign}${pnlPercent.toFixed(2)}%)`;
        pnlElement.className = `value pnl ${pos.unrealized_pnl >= 0 ? 'positive' : 'negative'}`;
    } else {
        document.getElementById('positionType').textContent = '无持仓';
        document.getElementById('positionType').className = 'position-type neutral';
        document.getElementById('positionSize').textContent = '--';
        document.getElementById('entryPrice').textContent = '--';
        document.getElementById('unrealizedPnl').textContent = '--';
        document.getElementById('unrealizedPnl').className = 'value pnl';
    }

    document.getElementById('totalProfit').textContent =
        formatCurrency(data.performance?.total_profit || 0);
    document.getElementById('winRate').textContent =
        data.performance?.win_rate ? `${(data.performance.win_rate * 100).toFixed(1)}%` : '--';
    document.getElementById('totalTrades').textContent =
        data.performance?.total_trades ?? '--';
}

// ==================== 公共函数 ====================
async function pollData() {
    if (currentView === 'home') {
        await updateHomeOverview();
    } else {
        await updateModelView();
    }
}

function formatCurrency(value) {
    return `$${Number(value || 0).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function formatChange(changeAbs, changePct) {
    if (changeAbs === null || changeAbs === undefined) return '--';
    const pctText = changePct !== null && changePct !== undefined
        ? ` (${(changePct * 100).toFixed(2)}%)`
        : '';
    const sign = changeAbs > 0 ? '+' : '';
    return `${sign}${changeAbs.toFixed(2)}${pctText}`;
}

function setChangeClass(element, change) {
    if (!element) return;
    element.classList.remove('positive', 'negative');
    if (change > 0) element.classList.add('positive');
    if (change < 0) element.classList.add('negative');
}

// 格式化时间戳为可读时间
function formatTimestamp(timestamp) {
    if (!timestamp) return '';
    
    // 如果是日期对象，直接使用
    let date;
    if (timestamp instanceof Date) {
        date = timestamp;
    } else if (typeof timestamp === 'number') {
        // 如果是数字，判断是秒还是毫秒时间戳
        // 如果小于10000000000，认为是秒级时间戳，需要乘以1000
        date = new Date(timestamp < 10000000000 ? timestamp * 1000 : timestamp);
    } else if (typeof timestamp === 'string') {
        // 如果是字符串，尝试解析
        date = new Date(timestamp);
    } else {
        return String(timestamp);
    }
    
    if (isNaN(date.getTime())) return String(timestamp); // 如果无法解析，返回原值
    
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    const seconds = String(date.getSeconds()).padStart(2, '0');
    
    return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
}

// 兼容旧有全局函数调用
window.switchSymbol = switchSymbol;
