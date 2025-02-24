from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from tools.openrouter_config import get_chat_completion

from agents.state import AgentState, show_agent_reasoning


##### Portfolio Management Agent #####
def portfolio_management_agent(state: AgentState):
    """Makes final trading decisions and generates orders"""
    show_reasoning = state["metadata"]["show_reasoning"]
    portfolio = state["data"]["portfolio"]

    # Get the technical analyst, fundamentals agent, and risk management agent messages
    technical_message = next(
        msg for msg in state["messages"] if msg.name == "technical_analyst_agent")
    fundamentals_message = next(
        msg for msg in state["messages"] if msg.name == "fundamentals_agent")
    sentiment_message = next(
        msg for msg in state["messages"] if msg.name == "sentiment_agent")
    valuation_message = next(
        msg for msg in state["messages"] if msg.name == "valuation_agent")
    risk_message = next(
        msg for msg in state["messages"] if msg.name == "risk_management_agent")

    # Create the system message
    system_message = {
        "role": "system",
        "content": """You are a portfolio manager making final trading decisions.
            Your job is to make a trading decision based on the team's analysis while considering
            risk management guidelines.

            RISK MANAGEMENT GUIDELINES:
            - Try to stay within max_position_size from risk management when possible
            - Consider risk management's trading_action as a strong recommendation
            - Risk signals should be weighed against potential opportunities

            When weighing the different signals for direction and timing:
            1. Technical Analysis (35% weight)
               - Primary driver for short-term trading decisions
               - Key for entry/exit timing
               - Higher weight for more active trading
            
            2. Fundamental Analysis (30% weight)
               - Business quality and growth assessment
               - Determines conviction in position
            
            3. Valuation Analysis (25% weight)
               - Long-term value assessment
               - Secondary confirmation of entry/exit points
            
            4. Sentiment Analysis (10% weight)
               - Final consideration
               - Can influence sizing and timing
            
            The decision process should be:
            1. Evaluate technical signals for timing
            2. Check fundamental support
            3. Confirm with valuation
            4. Consider sentiment
            5. Apply risk management as guidelines
            
            Provide the following in your output:
            - "action": "buy" | "sell" | "hold",
            - "quantity": <positive integer>
            - "confidence": <float between 0 and 1>
            - "agent_signals": <list of agent signals including agent name, signal (bullish | bearish | neutral), and their confidence>
            - "reasoning": <concise explanation of the decision including how you weighted the signals>

            Trading Rules:
            - Try to stay within risk management position limits when possible
            - Only buy if you have available cash
            - Only sell if you have shares to sell
            - Quantity must be ≤ current position for sells
            - Consider max_position_size from risk management as a guideline"""
    }

    chinese_system_message = {
        "role": "system",
        "content": """你是一名投资组合经理，正在做出最终的交易决策。你的工作是在考虑风险管理准则的同时，基于团队的分析来做出交易决策。
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
            - 将风险管理规定的最大持仓规模视为指导准则。"""
    }

    # Create the user message
    user_message = {
        "role": "user",
        "content": f"""Based on the team's analysis below, make your trading decision.

            Technical Analysis Trading Signal: {technical_message.content}
            Fundamental Analysis Trading Signal: {fundamentals_message.content}
            Sentiment Analysis Trading Signal: {sentiment_message.content}
            Valuation Analysis Trading Signal: {valuation_message.content}
            Risk Management Trading Signal: {risk_message.content}

            Here is the current portfolio:
            Portfolio:
            Cash: {portfolio['cash']:.2f}
            Current Position: {portfolio['stock']} shares

            Only include the action, quantity, reasoning, confidence, and agent_signals in your output as JSON.  Do not include any JSON markdown.

            Remember, the action must be either buy, sell, or hold.
            You can only buy if you have available cash.
            You can only sell if you have shares in the portfolio to sell."""
    }

    chinese_user_message = {
        "role": "user",
        "content": f"""基于以下团队分析，做出你的交易决策。
        - 技术分析交易信号：{technical_message.content}
        - 基本面分析交易信号：{fundamentals_message.content}
        - 情绪分析交易信号：{sentiment_message.content}
        - 估值分析交易信号：{valuation_message.content}
        - 风险管理交易信号：{risk_message.content}
        - 当前投资组合：现金：{portfolio['cash']:.2f},当前持仓：{portfolio['stock']}股
        在输出中仅以JSON格式包含行动、数量、推理过程、信心程度和代理信号。不要包含任何JSON格式的markdown。
        行动必须是买入、卖出或持有。只有在有可用现金时才能买入。只有在投资组合中有股票可卖时才能卖出。"""
    }

    # Get the completion from OpenRouter
    result = get_chat_completion([chinese_system_message, chinese_user_message])

    # 如果 API 调用失败,返回默认的 hold 决策
    if result is None:
        result = '''
        {
            "action": "hold",
            "quantity": 0,
            "confidence": 0.5,
            "agent_signals": [
                {"name": "technical_analyst", "signal": "neutral", "confidence": 0.5},
                {"name": "fundamentals", "signal": "neutral", "confidence": 0.5},
                {"name": "sentiment", "signal": "neutral", "confidence": 0.5},
                {"name": "valuation", "signal": "neutral", "confidence": 0.5},
                {"name": "risk_management", "signal": "hold", "confidence": 1.0}
            ],
            "reasoning": "API call failed, defaulting to hold position for safety"
        }
        '''

    # Create the portfolio management message
    message = HumanMessage(
        content=result,
        name="portfolio_management",
    )

    # Print the decision if the flag is set
    if show_reasoning:
        show_agent_reasoning(message.content, "Portfolio Management Agent")

    return {"messages": state["messages"] + [message]}
