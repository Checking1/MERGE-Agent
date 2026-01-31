"""
AAT-Agent: Attribution Analysis Toolkit Agent

通用的归因分析智能体框架，支持多领域知识图谱问答和归因分析。

主要模块：
- strategy_agent: 策略智能体（生成查询计划）
- search_agent: 搜索智能体（执行图谱查询）
- tools: 工具集（数据摘要、RAG检索、Schema查询等）
- utils: 工具函数（数据库连接、序列化等）
"""

__version__ = '1.0.0'
__author__ = 'AAT-Agent Team'

from .config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    DEEPSEEK_TEMPERATURE,
    DEEPSEEK_MAX_TOKENS,
    NEO4J_PARAMS,
)

from .strategy_agent import (
    StrategyAgent,
    StrategyConfig,
    StrategyPlan,
    QueryBlock,
    SearchFilter,
    TaskContext,
)

from .search_agent import (
    SearchAgent,
    SearchConfig,
    EvidenceNormalizer,
    SearchMemory,
    CypherExecutor,
)

__all__ = [
    # Config
    'DEEPSEEK_API_KEY',
    'DEEPSEEK_BASE_URL',
    'DEEPSEEK_MODEL',
    'DEEPSEEK_TEMPERATURE',
    'DEEPSEEK_MAX_TOKENS',
    'NEO4J_PARAMS',
    
    # Strategy Agent
    'StrategyAgent',
    'StrategyConfig',
    'StrategyPlan',
    'QueryBlock',
    'SearchFilter',
    'TaskContext',
    
    # Search Agent
    'SearchAgent',
    'SearchConfig',
    'EvidenceNormalizer',
    'SearchMemory',
    'CypherExecutor',
]
