from dotenv import load_dotenv
from openrouter_config import generate_content


if __name__ == "__main__":
    load_dotenv()
    system_content = '''
        你是一名投资组合经理，正在做出最终的交易决策。你的工作是在考虑风险管理准则的同时，基于团队的分析来做出交易决策。

        ### 风险管理准则：
        - 尽可能在风险管理规定的最大持仓规模范围内操作。
        - 将风险管理部门给出的交易行动建议视为强有力的推荐。
        - 风险信号应与潜在机会进行权衡。

        ### 在权衡有关交易方向和时机的不同信号时：
        1. **技术分析（权重35%）**
            - 短期交易决策的主要驱动因素。
            - 入场/出场时机的关键。
            - 对于更积极的交易给予更高权重。
        2. **基本面分析（权重30%）**
            - 对企业质量和增长的评估。
            - 决定持仓的信心程度。
        3. **估值分析（权重25%）**
            - 长期价值评估。
            - 对入场/出场点位的二次确认。
        4. **情绪分析（权重10%）**
            - 作为最后的考量因素。
            - 可能影响持仓规模和交易时机。

        ### 决策过程应如下：
        1. 评估技术信号以确定交易时机。
        2. 检查基本面的支撑情况。
        3. 通过估值进行确认。
        4. 考虑市场情绪。
        5. 以风险管理准则作为指导。

        ### 在你的输出中提供以下内容：
        - **“行动”**：“买入” | “卖出” | “持有”
        - **“数量”**：<正整数>
        - **“信心程度”**：<0到1之间的浮点数>
        - **“代理信号”**：<包含代理名称、信号（看涨 | 看跌 | 中性）以及他们的信心程度的代理信号列表>
        - **“推理过程”**：<对决策的简要解释，包括你如何权衡各种信号>

        ### 交易规则：
        - 尽可能遵守风险管理的持仓限制。
        - 只有在有可用现金时才买入。
        - 只有在持有股票时才卖出。
        - 卖出时的数量必须≤当前持仓量。
        - 将风险管理规定的最大持仓规模视为指导准则。
    '''
    
    user_content = '''
        基于以下团队分析，做出你的交易决策。

        ### 技术分析交易信号：
        {"signal": "看涨", "confidence": "26%", "strategy_signals": {"trend_following": {"signal": "看跌", "confidence": "27%", "metrics": {"adx": 27.083765560212584, "trend_strength": 0.27083765560212586}},"mean_reversion": {"signal": "中性", "confidence": "50%", "metrics": {"z_score": -1.8484088231337192, "price_vs_bb": 0.1609669202985599, "rsi_14": 27.974149823875962, "rsi_28": 39.86400080487812}},"momentum": {"signal": "看涨", "confidence": "47%", "metrics": {"momentum_1m": -0.1961300886929781, "momentum_3m": 0.03850951047047524, "momentum_6m": 0.5352375957764027, "volume_momentum": 1.0209602249883676}},"volatility": {"signal": "看涨", "confidence": "40%", "metrics": {"historical_volatility": 0.4736186104257821, "volatility_regime": 0.730117991560607, "volatility_z_score": -1.2010466090072818, "atr_ratio": 0.05174871433436999}},"statistical_arbitrage": {"signal": "中性", "confidence": "50%", "metrics": {"hurst_exponent": 0.0, "skewness": 0.23588970243274135, "kurtosis": -0.3122693894261317}}}}

        ### 基本面分析交易信号：
        {"signal": "看跌", "confidence": "50%", "reasoning": {"profitability_signal": {"signal": "看跌", "details": "净资产收益率(ROE): 10.42%, 净利润率: 7.26%, 营业利润率: 6.16%"}, "growth_signal": {"signal": "看跌", "details": "营收增长率: 0.95%, 盈利增长率: -52.46%"}, "financial_health_signal": {"signal": "中性", "details": "流动比率: 2.02, 资产负债率: 18.49"}, "price_ratios_signal": {"signal": "看涨", "details": "市盈率(P/E): 104.26, 市净率(P/B): 14.90, 市销率(P/S): 11.12"}}}

        ### 情绪分析交易信号：
        {"signal": "中性", "confidence": "100%", "reasoning": "基于截至2025年2月24日的0篇近期新闻文章，情绪得分: 0.00"}

        ### 估值分析交易信号：
        {"signal": "看跌", "confidence": "99%", "reasoning": {"dcf_analysis": {"signal": "看跌", "details": "内在价值: $7,317,204,386.62, 市值: $1,086,540,414,976.00, 差距: -99.3%"}, "owner_earnings_analysis": {"signal": "看跌", "details": "所有者收益价值: $12,522,252,517.85, 市值: $1,086,540,414,976.00, 差距: -98.8%"}}}

        ### 风险管理交易信号：
        {"max_position_size": 12500.0, "risk_score": 8, "trading_action": "减持", "risk_metrics": {"volatility": 0.6455352560303443, "value_at_risk_95": -0.05579199989296868, "max_drawdown": -0.31542531149095665, "market_risk_score": 4}, "reasoning": "风险评分8/10: 市场风险=4, 波动率=64.55%, 风险价值(VaR)=-5.58%, 最大回撤=-31.54%"}

        ### 以下是当前投资组合：
        投资组合：
        现金：100000.00
        当前持仓：0股

        在你的输出中仅以JSON格式包含行动、数量、推理过程、信心程度和代理信号。不要包含任何JSON格式的markdown。请记住，行动必须是买入、卖出或持有。只有在有可用现金时才能买入。只有在投资组合中有股票可卖时才能卖出。 
    '''
    
    messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
    content = generate_content(model="gemini-1.5-flash", messages=messages)
    print("推理过程：\n", content["reasoning"])
    print("输出结果：\n", content["content"])