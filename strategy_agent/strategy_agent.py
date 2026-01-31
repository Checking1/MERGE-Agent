"""
Strategy Agent V2: 基于LangChain create_agent的真正智能体

改进：
1. 使用create_agent替代流水线LLM调用
2. 集成工具：summarize_events, retrieve_domain_rules
3. Agent自主决策何时调用工具
4. 支持调试和中间步骤追踪
"""

import json
import os
import uuid
import logging
from typing import Dict, Any, List, Optional

from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import SystemMessage

from tools import (
    create_data_summary_tool,
    create_rag_retrieval_tool,
)
from .data_summary import DataSummary
from .plan import StrategyPlan, QueryBlock, SearchFilter, TaskContext, QueryStep
from .rag import Rag
from utils import serialize

# 配置日志
logger = logging.getLogger(__name__)


class StrategyConfig:
    """Strategy Agent配置"""
    def __init__(
        self,
        embedding_model_path: str,
        rag_data_path: str,
        rag_db_dir: str,
        llm_base_url: str,
        llm_api_key: str,
        llm_model: str = "deepseek-chat",
        llm_temperature: float = 0.2,
        llm_enabled: bool = True,
    ):
        self.embedding_model_path = embedding_model_path
        self.rag_data_path = rag_data_path
        self.rag_db_dir = rag_db_dir
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.llm_enabled = llm_enabled


class StrategyAgent:
    """
    Strategy Agent V2: 具备工具调用能力的策略智能体

    工具集：
    - summarize_events: 统计事件数量
    - retrieve_domain_rules: 检索领域规则

    工作流（Agent自主决策）：
    1. 调用summarize_events获取事件统计
    2. 分析统计结果，形成假设
    3. 调用retrieve_domain_rules获取相关规则
    4. 生成QueryBlocks计划
    """

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.data_summary = DataSummary()
        self.rag = Rag(embedding_model_path=config.embedding_model_path)
        self._prepare_rag_stores()

        # 创建LLM
        self.llm = None
        if config.llm_enabled:
            self.llm = ChatDeepSeek(
                api_key=config.llm_api_key,
                base_url=config.llm_base_url,
                model=config.llm_model,
                temperature=config.llm_temperature,
            )

        # 创建工具集
        self.tools = self._create_tools()

        # 创建Agent
        self.agent_graph = self._create_agent()

    def _prepare_rag_stores(self):
        """准备RAG向量存储"""
        os.makedirs(self.config.rag_db_dir, exist_ok=True)
        title_db = os.path.join(self.config.rag_db_dir, "title")
        content_db = os.path.join(self.config.rag_db_dir, "content")

        # 始终加载title_docs和content_docs（用于精确匹配）
        self.rag.data_process_4_build_vector_store(self.config.rag_data_path)

        if not (
            os.path.exists(os.path.join(title_db, "chroma-collections.parquet"))
            and os.path.exists(os.path.join(content_db, "chroma-collections.parquet"))
        ):
            self.rag.build_title_vector_store(title_db)
            self.rag.build_content_vector_store(content_db)
        else:
            self.rag.load_title_vector_store(title_db)
            self.rag.load_content_vector_store(content_db)

    def _create_tools(self) -> List:
        """创建工具集合"""
        return [
            create_data_summary_tool(self.data_summary),
            create_rag_retrieval_tool(self.rag),
        ]

    def _create_agent(self):
        """创建基于create_agent的智能体"""
        if not self.llm:
            raise RuntimeError("LLM未配置，无法创建Agent")

        # 系统提示词
        system_prompt = """You are a PRRS risk attribution strategy planning agent.

**Your Mission:**
Analyze the alert and generate a query plan (QueryBlocks) for the search agent to find risk evidence in the knowledge graph.

**Your Tools:**
1. summarize_events: Get event statistics (intro, group, breed, delivery, normal_immu, abort events)
2. retrieve_domain_rules: Get domain knowledge about risk patterns

**Workflow:**
1. ALWAYS start by calling summarize_events to understand the situation
2. Analyze the summary and formulate a hypothesis
3. For each production stage with risks, retrieve rules and plan the matching query block, eg xx_risk_trace, please try to make follow queries to trace to AbortEvent and ConceptNode(流产率异常)
4. Call retrieve_domain_rules to get relevant patterns
5. Generate QueryBlocks in JSON format

**Event Type to Risk Link Mapping (IMPORTANT):**
- intro events → risk link "引种"
- group events → risk link "入群"
- breed events → risk link "配种"
- delivery events → risk link "分娩" (产房管理风险)
- normal_immu events → risk link "免疫" (NormalImmuEvent)

**QueryBlock Structure:**
Each QueryBlock must have:
- name: Query block identifier (e.g., "intro_risk_trace", "group_risk_trace")
- description: What this query block does
- expected_relations: List of relationship types (e.g., ["OCCUR", "EXIST", "LEAD_TO"])
- filters: {window_days, org_inv_dk, reference_date, ...}
- steps: Array of query steps

**Query Step Structure:**
Each step must have:
- step_id: Unique step identifier
- description: What this step does
- action: "seed" (find starting events) or "follow" (trace from previous step)
- target_event: Node type (IntroEvent, GroupEvent, AbortEvent, etc.)
- expected_relations: Relationships to traverse
- filters: Parameters for this step
- limit: Max results (default 100 for seed, 200 for follow)

**Critical Rules:**
1. ONLY generate intro_risk_trace if intro events have risks (risk_events > 0)
2. ONLY generate group_risk_trace if group events have risks (risk_events > 0)
3. ONLY generate breed_risk_trace if breed events have risks (risk_events > 0)
4. ONLY generate delivery_risk_trace if delivery events have risks (risk_events > 0)
5. ONLY generate normal_immu_risk_trace if normal_immu events have risks (risk_events > 0)
6. Each QueryBlock should have 3 steps: seed event -> follow to AbortEvent -> follow to ConceptNode(流产率异常)
7. The final step must use target_concept_type='流产率异常' to connect AbortEvent to 流产率异常 concept

**Output Format:**
Return ONLY a valid JSON array of QueryBlocks, no extra text.

Example:
[
  {
    "name": "intro_risk_trace",
    "description": "围绕引种风险进行分步追踪",
    "expected_relations": ["OCCUR", "EXIST", "LEAD_TO", "BY", "AFFECTS", "EXHIBIT", "HARBOR"],
    "filters": {
      "window_days": 60,
      "org_inv_dk": "xxx",
      "reference_date": "2025-08-28"
    },
    "steps": [
      {
        "step_id": "seed_intro_risk",
        "description": "筛选窗口内存在引种风险的引种事件",
        "action": "seed",
        "target_event": "IntroEvent",
        "expected_relations": ["OCCUR", "EXIST", "LEAD_TO"],
        "filters": {},
        "limit": 100
      },
      {
        "step_id": "follow_intro_to_group_abort",
        "description": "追踪对应批次的入群事件并追踪到流产事件",
        "action": "follow",
        "target_event": "AbortEvent",
        "expected_relations": ["BY", "EXIST", "LEAD_TO", "AFFECTS"],
        "filters": {},
        "limit": 200
      },
      {
        "step_id": "follow_abort_to_abnormal",
        "description": "从流产事件连接到流产率异常概念",
        "action": "follow",
        "target_concept_type": "流产率异常",
        "expected_relations": ["EXHIBIT", "HARBOR"],
        "filters": {},
        "limit": 200
      }
    ]
  }
]
"""

        # 创建Agent
        agent_graph = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt,
            debug=True  # 开启调试模式
        )

        return agent_graph

    def generate_strategy(self, context: TaskContext) -> StrategyPlan:
        """使用Agent生成策略计划"""
        logger.info("="*80)
        logger.info("策略智能体V2: 开始生成归因计划（Agent模式）")
        logger.info("="*80)

        # 构建用户消息
        user_message = f"""Generate query plan for PRRS risk attribution:

**Alert Context:**
- Trigger Type: {context.trigger_type}
- Farm ID: {context.org_inv_dk}
- Reference Date: {context.reference_date}
- Time Window: {context.window_days} days

**Instructions:**
1. Use summarize_events tool with the above parameters
2. Based on the summary, decide which QueryBlocks to generate
3. Use retrieve_domain_rules to understand risk patterns if needed
4. Output QueryBlocks JSON array

Remember:
- Only include intro_risk_trace if intro events have risks
- Only include group_risk_trace if group events have risks
- Only include normal_immu_risk_trace if normal_immu events have risks
- Each QueryBlock needs proper filters and steps
"""

        try:
            # 调用Agent
            messages = [{"role": "user", "content": user_message}]
            result = self.agent_graph.invoke({"messages": messages})

            # 提取最后一条消息（Agent的最终回答）
            final_message = result["messages"][-1]
            final_answer = final_message.content

            logger.info(f"\nAgent执行完成，共 {len(result['messages'])} 条消息")

            # 解析QueryBlocks
            plan = self._parse_plan_from_answer(final_answer, context)

            logger.info(f"\n策略计划生成完成 (plan_id={plan.plan_id})")
            logger.info(f"  查询块数量: {len(plan.query_blocks)}")
            for qb in plan.query_blocks:
                logger.info(f"    - {qb.name}: {len(qb.steps)} 步骤")
            logger.info("="*80)

            return plan

        except Exception as e:
            logger.error(f"Agent执行失败: {str(e)}", exc_info=True)
            raise

    def _parse_plan_from_answer(self, answer: str, context: TaskContext) -> StrategyPlan:
        """从Agent的回答中解析StrategyPlan"""
        try:
            # 清理可能的markdown代码块和说明文字
            answer = answer.strip()

            # 查找JSON数组的起始位置（以'['开头）
            json_start = answer.find('[')
            if json_start == -1:
                raise ValueError("未找到JSON数组起始标记'['")

            # 从起始位置截取
            answer = answer[json_start:]

            # 移除末尾的markdown代码块标记
            if answer.endswith("```"):
                answer = answer[:-3]
            answer = answer.strip()

            # 解析JSON
            query_blocks_data = json.loads(answer)

            if not isinstance(query_blocks_data, list):
                raise ValueError("QueryBlocks必须是JSON数组")

            # 转换为QueryBlock对象
            query_blocks = []
            for qb_data in query_blocks_data:
                # QueryBlock级别的filters
                qb_filters_data = qb_data.get("filters", {})
                qb_filters = SearchFilter(
                    window_days=qb_filters_data.get("window_days", context.window_days),
                    org_inv_dk=qb_filters_data.get("org_inv_dk", context.org_inv_dk),
                    reference_date=qb_filters_data.get("reference_date", context.reference_date)
                )

                steps = []
                for step_data in qb_data.get("steps", []):
                    # Step级别的filters继承QueryBlock的filters
                    step_filters_data = step_data.get("filters", {})
                    step_filters = SearchFilter(
                        window_days=step_filters_data.get("window_days", qb_filters.window_days),
                        org_inv_dk=step_filters_data.get("org_inv_dk", qb_filters.org_inv_dk),
                        reference_date=step_filters_data.get("reference_date", qb_filters.reference_date),
                        pig_batch=step_filters_data.get("pig_batch"),
                        source_org_dk=step_filters_data.get("source_org_dk")
                    )

                    step = QueryStep(
                        step_id=step_data["step_id"],
                        description=step_data["description"],
                        action=step_data["action"],
                        target_event=step_data.get("target_event"),
                        target_concept_type=step_data.get("target_concept_type"),
                        expected_relations=step_data.get("expected_relations", []),
                        filters=step_filters,
                        limit=step_data.get("limit", 100)
                    )
                    steps.append(step)

                query_block = QueryBlock(
                    name=qb_data["name"],
                    description=qb_data.get("description", ""),
                    expected_relations=qb_data.get("expected_relations", []),
                    filters=qb_filters,
                    steps=steps
                )
                query_blocks.append(query_block)

            # 创建StrategyPlan
            plan = StrategyPlan(
                plan_id=str(uuid.uuid4()),
                trigger_type=context.trigger_type,
                hypothesis=f"基于{len(query_blocks)}个查询块的PRRS风险归因假设",
                query_blocks=query_blocks,
                evidence_summary=f"待执行{sum(len(qb.steps) for qb in query_blocks)}个查询步骤"
            )

            return plan

        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {str(e)}")
            logger.error(f"原始回答: {answer}")
            raise ValueError(f"Agent返回的不是有效的JSON: {str(e)}")
        except Exception as e:
            logger.error(f"解析失败: {str(e)}")
            raise
