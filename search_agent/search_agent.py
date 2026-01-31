"""
Search Agent: 完整的Schema感知搜索智能体

功能：
1. 使用create_agent + 工具调用生成Cypher
2. 执行Cypher查询并保存结果
3. 记忆库管理避免重复查询
4. 分步查询（seed → follow → follow）
5. 证据规范化和序列化
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek

from tools import (
    create_node_schema_tool,
    create_kg_relationship_tool,
)
from strategy_agent.plan import (
    QueryStep,
    QueryBlock,
    StrategyPlan,
    SearchFilter,
    SearchResult,
    SearchEvidence,
)
from utils.graph_db import Neo4JDB
import config as agent_config
from .execution import EvidenceNormalizer, SearchMemory, CypherExecutor

logger = logging.getLogger(__name__)


def _default_dataset_base_path() -> str:
    env = os.getenv("PRRS_DATASET_BASE_PATH")
    if env:
        return env

    # Walk upwards to find repo root (contains `data/`).
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "data" / "interim" / "PRRS_Risk_Attribution"
        if candidate.is_dir():
            return str(candidate)
    return str(Path.cwd() / "data" / "interim" / "PRRS_Risk_Attribution")


class SearchConfig:
    """Search Agent配置"""
    def __init__(
        self,
        llm_base_url: str,
        llm_api_key: str,
        llm_model: str = "deepseek-chat",
        llm_temperature: float = 0.2,
        llm_enabled: bool = True,
        neo4j_params: Optional[Dict[str, Any]] = None,
        default_limit: int = 200,
        query_timeout_ms: int = 120000,
        dataset_base_path: str = None,
        cypher_log_path: Optional[str] = None,
        core_log_path: Optional[str] = None,
    ):
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.llm_enabled = llm_enabled
        self.neo4j_params = neo4j_params or dict(agent_config.NEO4J_PARAMS)
        self.default_limit = default_limit
        self.query_timeout_ms = query_timeout_ms
        self.dataset_base_path = dataset_base_path or _default_dataset_base_path()
        self.cypher_log_path = cypher_log_path
        self.core_log_path = core_log_path


class SearchAgent:
    """
    Search Agent: 完整功能的Schema感知搜索智能体

    工具集：
    - get_node_schema: 查询节点字段信息
    - query_kg_relationships: 查询关系定义

    核心功能：
    1. 生成Cypher（Agent + 工具调用）
    2. 执行Cypher（Neo4j查询）
    3. 记忆库管理（避免重复）
    4. 证据规范化（序列化保存）
    """

    def __init__(self, config: SearchConfig):
        self.config = config

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
        self.node_schema_tool = None
        self.kg_relationship_tool = None
        self.tools = self._create_tools()

        # 创建Agent
        self.agent_graph = self._create_cypher_agent()

        # 创建执行组件
        self.normalizer = EvidenceNormalizer()
        self.memory = SearchMemory()
        self.db = None
        self.executor = None

    def _create_tools(self) -> List:
        """创建工具集合"""
        self.node_schema_tool = create_node_schema_tool(self.config.dataset_base_path)
        self.kg_relationship_tool = create_kg_relationship_tool(
            self.config.neo4j_params,
            max_db_calls=1,
        )
        return [self.node_schema_tool, self.kg_relationship_tool]

    def _create_cypher_agent(self):
        """创建Cypher生成Agent"""
        if not self.llm:
            raise RuntimeError("LLM未配置，无法创建Agent")

        system_prompt = """You are a Neo4j Cypher query generation expert for PRRS risk attribution.

**Mission**
Generate Cypher that returns attribution subgraphs. For **seed** steps, return a minimal subgraph that starts from PigFarm and includes the target event plus its RiskEvent -> ConceptNode bridge. For **follow** steps, return a complete subgraph that starts from PigFarm and ends at ConceptNode(type='流产率异常'), passing through events, risks, and concept bridges. RiskEvent nodes must NOT be left hanging with only EXIST; they must connect via ConceptNodes using the schema relations below.

**Tools**
1. get_node_schema: Query node field information (CRITICAL - call before generating Cypher)
2. query_kg_relationships: Learn KG schema relationships/types on demand (cached; avoid repeated calls)

**Mandatory Graph Shape (generalized, not tied to a single pattern)**
1) Start: PigFarm -[:OCCUR]-> target event
2) RiskEvent attachment: Event -[:EXIST]-> RiskEvent -[:LEAD_TO|INHIBITS]-> ConceptNode (risk meaning)
3) Concept bridge to downstream events: ConceptNode -[:BY|LEAD_TO]-> next event (when applicable)
4) Downstream to outcome (follow only): ConceptNode (e.g., 基础母猪风险) -[:AFFECTS]-> AbortEvent -[:EXHIBIT|HARBOR]-> ConceptNode(type='流产率异常')
5) Always return path_nodes and path_relationships from MATCH path patterns; no partial edges.
6) Avoid returning a standalone PigFarm-[:OCCUR]->AbortEvent path. Attribution subgraphs must include at least one production event between PigFarm and AbortEvent; for follow_abort_to_abnormal, only return AbortEvent->ConceptNode and let upstream steps provide the PigFarm start.

**Workflow**
1. ALWAYS call get_node_schema for target_event
2. If schema relations or type constraints are unclear, call query_kg_relationships (prefer query_type="all")
3. When query_kg_relationships provides concept_outgoing/concept_incoming, use them to bridge ConceptNodes to downstream events (avoid hardcoding a single ConceptNode type that blocks the chain)
4. Use MATCH path = ... to capture required paths (seed: PigFarm -> target event -> RiskEvent -> ConceptNode; follow: PigFarm -> ... -> ConceptNode(流产率异常))
5. Use collect(distinct ...) to keep one row per seed; avoid UNWIND unless necessary
6. Return ONLY the Cypher query, no explanations
7. **When generating follow queries to AbortEvent**: Even though the query step only targets AbortEvent (not 流产率异常), you MUST still build the COMPLETE path through ALL intermediate ConceptNodes. The task decomposition is for workflow organization, NOT for path simplification.
   - **ALWAYS call query_kg_relationships** to discover the full schema from your starting ConceptNode
   - **Parse concept_outgoing/concept_incoming carefully**: When the tool returns these fields for a ConceptNode, examine the target_labels field:
     * If target_labels = ["ConceptNode"], the ConceptNode connects to ANOTHER ConceptNode (NOT to AbortEvent)
     * If target_labels = ["AbortEvent"], the ConceptNode connects directly to AbortEvent
     * The key insight: you may need to recursively check the concept_outgoing of intermediate ConceptNodes until you find one that connects to AbortEvent
   - **Build complete multi-hop paths**: Chain together ALL intermediate ConceptNodes before reaching AbortEvent (e.g., ConceptNode1 -[:LEAD_TO]-> ConceptNode2 -[:AFFECTS]-> AbortEvent)
   - **Never skip intermediate nodes**: Even if your task is "follow to AbortEvent", you cannot skip ConceptNodes that lie on the schema path
   - Include ALL discovered relationships and intermediate ConceptNodes in the Cypher query to ensure complete path coverage

**Critical Rules**
1. Every node filter must include inference_date = $reference_date
2. Time window filtering (ROBUST, schema-driven):
   - For event-like nodes that represent occurrences within the observation window (typically seed targets), choose ONE *occurrence-time* field from get_node_schema (or its time_filter_guidance) and filter it between $start_date and $end_date.
   - Do NOT hardcode field names. Infer semantics from field descriptions and sample values; treat inference_date as required snapshot time (NOT a window field) and treat stats_dt/stat_dt-like fields as *soft-avoid* unless schema/semantics indicate they represent business/occurrence time (or they are the only date-like option).
   - Do NOT output placeholders like "<EVENT_TIME_FIELD>" in the final Cypher; replace them with the actual field name you selected.
   - Snapshot/aggregate nodes MUST NOT use $start_date/$end_date filtering:
     * AbortEvent: filter ONLY by org_inv_dk + inference_date (no window)
     * NormalImmuEvent: filter ONLY by org_inv_dk + inference_date (no window; do NOT filter by stats_dt)
3. Seed queries must include PigFarm start and the target event
4. Follow queries with target_event: return full PigFarm -> ... -> target event chain
5. Follow queries with target_concept_type: return full PigFarm -> ... -> ConceptNode(type=target_concept_type) chain
6. RiskEvent must connect to ConceptNode via LEAD_TO/INHIBITS; do NOT stop at EXIST
7. For follow queries anchored by a seed event_id, ALWAYS include the edge (farm:PigFarm {org_inv_dk:$org_inv_dk})-[:OCCUR]->(anchored seed event) in returned paths
8. If you use multiple WITH clauses, carry forward any path variables (or their collected lists). Never reference path0/path1 after a WITH unless they are explicitly carried; use collected lists like base_paths.
9. **AbortEvent -> ConceptNode(流产率异常) relationship semantics (CRITICAL)**:
   - Both EXHIBIT and HARBOR relationships exist in graph for ALL AbortEvents
   - HARBOR means: abort_rate_7days ≤ 0.0025 (≤ 0.25%, potential risk)
   - EXHIBIT means: abort_rate_7days > 0.0025 (> 0.25%, confirmed abnormality)
10. AbortEvent has no event_id; match ONLY by org_inv_dk + inference_date (no seed filtering)
11. Do NOT include PigFarm-[:OCCUR]->AbortEvent in follow_abort_to_abnormal returns; this step only supplies AbortEvent -> ConceptNode.
   - When target_concept_type='流产率异常', use two OPTIONAL MATCH with different variables and coalesce:
     ```cypher
     OPTIONAL MATCH path1 = (abortion)-[:EXHIBIT]->(abort_exhibit:ConceptNode {type: '流产率异常', ...})
     WHERE abortion.abort_rate_7days > 0.0025
     OPTIONAL MATCH path2 = (abortion)-[:HARBOR]->(abort_harbor:ConceptNode {type: '流产率异常', ...})
     WHERE abortion.abort_rate_7days <= 0.0025
     WITH ..., coalesce(abort_exhibit, abort_harbor) AS abort_concept
     ```
   - Both use same variable name abort_concept; only one will match per abortion based on abort_rate_7days

**Few-shot Examples (structure for reference; adapt fields by schema)**

**Example 1 (seed - IntroEvent):**
```
Description: "筛选窗口内存在引种风险的引种事件"
Target: IntroEvent
Action: seed

Generated Cypher:
MATCH path = (farm:PigFarm)-[:OCCUR]->(intro:IntroEvent)
-[:EXIST]->(intro_risk:RiskEvent {link: '引种', inference_date: $reference_date})
-[:LEAD_TO]->(reserve_risk:ConceptNode {type: '后备猪风险', inference_date: $reference_date})
WHERE farm.org_inv_dk = $org_inv_dk
  AND intro.inference_date = $reference_date
  AND intro.<EVENT_TIME_FIELD> >= $start_date  // choose <EVENT_TIME_FIELD> via get_node_schema (occurrence time)
  AND intro.<EVENT_TIME_FIELD> <= $end_date
WITH intro, collect(path) as paths
RETURN intro,
       [p in paths | nodes(p)] as path_nodes,
       [p in paths | relationships(p)] as path_relationships
LIMIT $limit
```

**Example 2 (seed - GroupEvent, minimal chain):**
```
Description: "筛选窗口内存在入群风险的入群事件"
Target: GroupEvent
Action: seed

Generated Cypher:
MATCH path1 = (farm:PigFarm)-[:OCCUR]->(group:GroupEvent)
-[:EXIST]->(group_risk:RiskEvent {link: '入群', inference_date: $reference_date})
-[:LEAD_TO]->(base_risk:ConceptNode {type: '基础母猪风险', inference_date: $reference_date})
WHERE farm.org_inv_dk = $org_inv_dk
  AND group.inference_date = $reference_date
  AND group.<EVENT_TIME_FIELD> >= $start_date  // choose <EVENT_TIME_FIELD> via get_node_schema (occurrence time)
  AND group.<EVENT_TIME_FIELD> <= $end_date
WITH group, collect(path1) as base_paths
RETURN group,
       [p in base_paths | nodes(p)] as path_nodes,
       [p in base_paths | relationships(p)] as path_relationships
LIMIT $limit
```

**Example 3 (follow - IntroEvent to GroupEvent):**
```
Description: "从引种事件追踪对应的入群事件"
Target: GroupEvent
Action: follow
Seed params: seed_intro_event_id, seed_pig_batch, seed_end_date

Generated Cypher:
MATCH path0 = (intro:IntroEvent {event_id: $seed_intro_event_id, inference_date: $reference_date})
-[:EXIST]->(intro_risk:RiskEvent {link: '引种', inference_date: $reference_date})
-[:LEAD_TO]->(reserve_risk:ConceptNode {type: '后备猪风险', inference_date: $reference_date})
-[:BY]->(group:GroupEvent {inference_date: $reference_date})
WHERE group.org_inv_dk = $org_inv_dk
  AND (
    ($seed_pig_batch IS NOT NULL AND group.pig_batch = $seed_pig_batch)
    OR ($seed_pig_batch IS NULL AND $seed_end_date IS NOT NULL AND group.begin_date = $seed_end_date)
  )
OPTIONAL MATCH path1 = (group)-[:EXIST]->(group_risk:RiskEvent {link: '入群', inference_date: $reference_date})
-[:LEAD_TO]->(base_risk:ConceptNode {type: '基础母猪风险', inference_date: $reference_date})
-[:AFFECTS]->(abortion:AbortEvent {inference_date: $reference_date})
WHERE abortion.org_inv_dk = $org_inv_dk
WITH intro, intro_risk, reserve_risk, group, abortion,
     collect(distinct path0) as intro_paths,
     collect(distinct path1) as group_paths
RETURN intro, intro_risk, reserve_risk, group, abortion,
       [p in intro_paths | nodes(p)] +
       [p in group_paths WHERE p IS NOT NULL | nodes(p)] as path_nodes,
       [p in intro_paths | relationships(p)] +
       [p in group_paths WHERE p IS NOT NULL | relationships(p)] as path_relationships
LIMIT $limit
```

**Example 4 (follow - AbortEvent to ConceptNode(流产率异常)):**
```
Description: "从流产事件连接到流产率异常概念（根据流产率选择EXHIBIT或HARBOR关系；AbortEvent无event_id，每个猪场每个reference_date仅一条）"
Target Concept: ConceptNode(type='流产率异常')
Action: follow
Seed params: None

Generated Cypher:
MATCH (abortion:AbortEvent {org_inv_dk: $org_inv_dk, inference_date: $reference_date})
OPTIONAL MATCH path1 = (abortion)-[:EXHIBIT]->(abort_exhibit:ConceptNode {type: '流产率异常', inference_date: $reference_date})
WHERE abortion.abort_rate_7days > 0.0025
OPTIONAL MATCH path2 = (abortion)-[:HARBOR]->(abort_harbor:ConceptNode {type: '流产率异常', inference_date: $reference_date})
WHERE abortion.abort_rate_7days <= 0.0025
WITH abortion,
     coalesce(abort_exhibit, abort_harbor) AS abort_concept,
     collect(distinct path1) as exhibit_paths,
     collect(distinct path2) as harbor_paths
RETURN abortion, abort_concept,
       [p in exhibit_paths WHERE p IS NOT NULL | nodes(p)] +
       [p in harbor_paths WHERE p IS NOT NULL | nodes(p)] as path_nodes,
       [p in exhibit_paths WHERE p IS NOT NULL | relationships(p)] +
       [p in harbor_paths WHERE p IS NOT NULL | relationships(p)] as path_relationships
LIMIT $limit
```

**Output Format:**
Return ONLY the Cypher query, no markdown code blocks, no explanations.
"""

        agent_graph = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt,
            debug=True
        )

        return agent_graph

    def generate_cypher(
        self,
        step: QueryStep,
        base_params: Dict[str, Any],
        seed_param_keys: Optional[List[str]] = None,
    ) -> str:
        """使用Agent生成Cypher查询"""
        logger.info(f"    使用Agent生成Cypher for step: {step.step_id}")

        seed_param_text = "None"
        if seed_param_keys:
            seed_param_text = ", ".join(seed_param_keys)

        # 构建 target 描述
        if step.target_event:
            target_desc = f"- Target Event: {step.target_event}"
            schema_instruction = f'1. Call get_node_schema with node_type="{step.target_event}" to understand fields'
        elif step.target_concept_type:
            target_desc = f"- Target Concept: ConceptNode(type='{step.target_concept_type}')"
            schema_instruction = "1. You may call query_kg_relationships to understand the graph structure if needed"
        else:
            target_desc = "- Target: Not specified"
            schema_instruction = "1. Call get_node_schema or query_kg_relationships as needed"

        user_message = f"""Generate Cypher query for this step:

**Step Information:**
- Step ID: {step.step_id}
- Description: {step.description}
- Action: {step.action}
{target_desc}
 - Expected Relations: {step.expected_relations}

**Available Parameters:**
{json.dumps(base_params, indent=2, ensure_ascii=False)}
**Seed Parameters (must anchor follow queries if provided):**
{seed_param_text}

**Instructions:**
{schema_instruction}
2. If action is "seed", generate seed query (find events) and RETURN a minimal subgraph: PigFarm -> target event -> RiskEvent -> ConceptNode (risk meaning)
3. If action is "follow" with target_event, generate follow query and RETURN the full PigFarm -> ... -> target event path
4. If action is "follow" with target_concept_type, generate follow query and RETURN the full PigFarm -> ... -> ConceptNode(type=target_concept_type) path
5. RiskEvent must connect via ConceptNode using LEAD_TO/INHIBITS (no dangling EXIST-only risk edges)
6. When seed params are present, FILTER by the seed event_id. Do NOT broaden with new time windows.
7. Follow(xxEvent to AbortEvent) MUST also RETURN the path (farm:PigFarm)-[:OCCUR]->(anchored seed event) so the subgraph contains PigFarm -> seed event, not only PigFarm -> AbortEvent.

Output ONLY the Cypher query.
"""

        try:
            messages = [{"role": "user", "content": user_message}]
            result = self.agent_graph.invoke({"messages": messages})

            final_message = result["messages"][-1]
            cypher = final_message.content.strip()

            # 清理markdown代码块
            if cypher.startswith("```cypher"):
                cypher = cypher[9:]
            elif cypher.startswith("```"):
                cypher = cypher[3:]
            if cypher.endswith("```"):
                cypher = cypher[:-3]
            cypher = cypher.strip()

            logger.info(f"    ✅ Cypher生成成功 ({len(cypher)} 字符)")
            return cypher

        except Exception as e:
            logger.error(f"    ❌ Cypher生成失败: {str(e)}", exc_info=True)
            raise

    def _append_cypher_log(self, record: Dict[str, Any]) -> None:
        path = self.config.cypher_log_path
        if not path:
            return
        try:
            log_path = Path(path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        except Exception as e:
            logger.error(f"    Cypher log write failed: {str(e)}")

    def _append_core_log(self, text: str) -> None:
        path = self.config.core_log_path
        if not path:
            return
        try:
            log_path = Path(path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(text.rstrip() + "\n")
        except Exception as e:
            logger.error(f"    Core log write failed: {str(e)}")

    def connect_db(self):
        """连接Neo4j数据库"""
        if self.db is None:
            logger.info("正在连接Neo4j数据库...")
            self.db = Neo4JDB()
            self.db.connect()
            self.executor = CypherExecutor(self.db, self.normalizer, self.memory)
            logger.info("数据库连接成功")

    def close_db(self):
        """关闭数据库连接"""
        if self.db:
            self.db = None
            self.executor = None

    def run(self, plan: StrategyPlan) -> List[SearchResult]:
        """
        执行搜索计划（完整流程）

        流程：
        1. 连接数据库
        2. 遍历每个QueryBlock
        3. 为每个Step生成Cypher
        4. 执行Cypher查询
        5. 提取seed并级联follow查询
        6. 规范化证据并保存
        """
        logger.info("="*80)
        logger.info("搜索智能体V2: 开始执行搜索计划")
        logger.info("="*80)
        logger.info(f"计划ID: {plan.plan_id}")
        logger.info(f"查询块数量: {len(plan.query_blocks)}")
        self._append_core_log("=" * 100)
        self._append_core_log(f"[SEARCH] plan_id={plan.plan_id} blocks={len(plan.query_blocks)}")
        self._append_core_log("=" * 100)

        # 连接数据库
        self.connect_db()

        results: List[SearchResult] = []
        for idx, block in enumerate(plan.query_blocks):
            logger.info(f"\n{'='*60}")
            logger.info(f"执行查询块 {idx+1}/{len(plan.query_blocks)}: {block.name}")
            logger.info(f"描述: {block.description}")
            logger.info(f"步骤数: {len(block.steps)}")

            block_result = self._execute_block_steps(plan.plan_id, block)
            results.append(block_result)

            logger.info(f"查询块执行完成: status={block_result.status}, "
                       f"evidence_count={len(block_result.evidence)}")

        # 输出记忆库统计
        memory_stats = self.memory.get_stats()
        logger.info(f"\n记忆库统计: {json.dumps(memory_stats, ensure_ascii=False)}")
        logger.info("="*80)

        return results

    def _execute_block_steps(
        self,
        plan_id: str,
        block: QueryBlock
    ) -> SearchResult:
        """
        执行QueryBlock的分步查询

        流程：
        1. seed查询 → 提取seed_inputs
        2. follow查询 → 使用seed_inputs逐个查询
        3. follow查询 → 使用新的seed_inputs继续
        """
        issues: List[str] = []
        all_evidence: List[SearchEvidence] = []
        total_count = 0
        start_ts = time.time()
        seed_inputs: List[Dict[str, Any]] = []

        self._append_core_log("")
        self._append_core_log(f"[BLOCK] {block.name} - {block.description}")
        self._append_core_log(f"[BLOCK] filters={block.filters.to_dict() if block.filters else None} limit={block.limit}")

        for step_idx, step in enumerate(block.steps, 1):
            logger.info(f"\n  步骤 {step_idx}/{len(block.steps)}: {step.step_id}")
            logger.info(f"    描述: {step.description}")
            logger.info(f"    动作: {step.action}")
            logger.info(f"    目标: {step.target_event}")

            # 构建参数
            step_filters = step.filters or block.filters
            base_params = self._build_base_params(step_filters)
            base_params["limit"] = step.limit or block.limit or self.config.default_limit

            # 准备seed参数提示（仅follow需要；target_concept_type不传seed）
            seed_param_keys = None
            if step.action == "follow" and seed_inputs and not step.target_concept_type:
                seed_param_keys = list(seed_inputs[0].keys())

            # 生成Cypher
            try:
                cypher = self.generate_cypher(step, base_params, seed_param_keys)
            except Exception as e:
                logger.error(f"    Cypher生成失败: {str(e)}")
                issues.append(f"cypher_gen_error:{step.step_id}")
                continue

            self._append_cypher_log({
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "plan_id": plan_id,
                "query_block": block.name,
                "step_id": step.step_id,
                "action": step.action,
                "target_event": step.target_event,
                "expected_relations": step.expected_relations,
                "base_params": base_params,
                "seed_param_keys": seed_param_keys,
                "cypher": cypher,
            })
            self._append_core_log(f"[STEP] {block.name}/{step.step_id} action={step.action} target_event={step.target_event} target_concept={step.target_concept_type}")
            self._append_core_log(f"[STEP] description={step.description}")
            self._append_core_log(f"[STEP] base_params={json.dumps(base_params, ensure_ascii=False)} seed_param_keys={seed_param_keys}")
            self._append_core_log("[CYPHER]")
            self._append_core_log(cypher)
            self._append_core_log("-" * 60)


            # 执行查询
            if step.action == "seed":
                logger.info(f"    执行seed查询...")
                records, evidence = self._execute_seed(cypher, base_params, step.step_id, block.name)
                logger.info(f"    返回 {len(records)} 条记录")
                self._append_core_log(f"[RESULT] records={len(records)} evidence={len(evidence)} (seed)")
                total_count += len(records)
                all_evidence.extend(evidence)

                # 提取seed_inputs
                seed_inputs = self.executor.extract_seed_inputs(records, step.target_event)
                logger.info(f"    提取 {len(seed_inputs)} 个seed")

                # 记忆库过滤
                filtered_before = len(seed_inputs)
                seed_inputs = self.executor.filter_seed_by_memory(seed_inputs, step.target_event)
                logger.info(f"    记忆库过滤: {filtered_before} → {len(seed_inputs)}")
                self._append_core_log(f"[SEED] extracted={filtered_before} after_memory={len(seed_inputs)}")

                if not seed_inputs:
                    logger.warning(f"    ⚠️ 没有有效的seed输入")
                    issues.append("no_seed")

            else:  # follow
                # 特殊处理：如果 target_concept_type='流产率异常'，直接执行查询（不需要 seed）
                if step.target_concept_type == "流产率异常":
                    logger.info(f"    执行follow查询（连接到流产率异常概念，无需seed）...")
                    # 使用 seed 查询逻辑（因为不需要循环）
                    records, evidence = self._execute_seed(cypher, base_params, step.step_id, block.name)
                else:
                    if not seed_inputs:
                        logger.warning(f"    ⚠️ 没有seed输入，跳过follow")
                        issues.append("no_seed_for_follow")
                        self._append_core_log("[RESULT] skipped follow (no seeds)")
                        continue

                    logger.info(f"    执行follow查询（{len(seed_inputs)} 个seed）...")
                    records, evidence = self._execute_follow(cypher, base_params, seed_inputs, step.step_id, block.name)
                logger.info(f"    返回 {len(records)} 条记录")
                self._append_core_log(f"[RESULT] records={len(records)} evidence={len(evidence)} (follow)")
                total_count += len(records)
                all_evidence.extend(evidence)

                # 提取新的seed_inputs供下一步使用
                # 只有 target_event 才需要提取 seed，target_concept_type 是终点，不需要提取
                if records and step.target_event:
                    seed_inputs = self.executor.extract_seed_inputs(records, step.target_event)
                    logger.info(f"    提取 {len(seed_inputs)} 个新seed供下一步使用")

        elapsed_ms = int((time.time() - start_ts) * 1000)
        status = "success" if all_evidence else "no_data"

        return SearchResult(
            plan_id=plan_id,
            query_block=block.name,
            status=status,
            evidence=all_evidence,
            issues=issues,
            metrics={"records": total_count, "latency_ms": elapsed_ms},
        )

    def _execute_seed(
        self,
        cypher: str,
        base_params: Dict[str, Any],
        step_id: str,
        block_name: str
    ) -> Tuple[List[Dict[str, Any]], List[SearchEvidence]]:
        """执行seed查询"""
        metadata = {
            "query_name": step_id,
            "query_block": block_name,
            "params": base_params,
        }
        return self.executor.execute_query(cypher, base_params, metadata)

    def _execute_follow(
        self,
        cypher: str,
        base_params: Dict[str, Any],
        seed_inputs: List[Dict[str, Any]],
        step_id: str,
        block_name: str
    ) -> Tuple[List[Dict[str, Any]], List[SearchEvidence]]:
        """执行follow查询（遍历所有seed）"""
        all_records = []
        all_evidence = []

        total_seeds = len(seed_inputs)
        for idx, seed in enumerate(seed_inputs, 1):
            # 添加进度日志
            if idx % 5 == 1 or idx == total_seeds:  # 每5个或最后一个输出进度
                logger.info(f"      执行进度: {idx}/{total_seeds}")

            params = dict(base_params)
            params.update(seed)

            metadata = {
                "query_name": step_id,
                "query_block": block_name,
                "params": params,
            }

            try:
                records, evidence = self.executor.execute_query(cypher, params, metadata)
                all_records.extend(records)
                all_evidence.extend(evidence)

                # 更新记忆库
                if records:
                    self.executor.update_memory_from_records(records, seed)
            except Exception as e:
                logger.error(f"      follow查询失败 (seed #{idx}): {str(e)}")
                continue

        logger.info(f"      follow查询完成: {len(all_records)} 条记录")
        return all_records, all_evidence

    def _build_base_params(self, filters: SearchFilter) -> Dict[str, Any]:
        """构建基础参数"""
        reference_date = filters.reference_date
        window_days = filters.window_days
        start_date = self._shift_date(reference_date, -window_days)
        return {
            "org_inv_dk": filters.org_inv_dk,
            "reference_date": reference_date,
            "window_days": window_days,
            "start_date": start_date,
            "end_date": reference_date,
            "limit": self.config.default_limit,
        }

    def _shift_date(self, date_str: str, delta_days: int) -> str:
        """日期偏移"""
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        return (date_obj + timedelta(days=delta_days)).strftime("%Y-%m-%d")
