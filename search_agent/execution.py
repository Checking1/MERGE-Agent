"""
Search Agent V2 执行模块

包含：
1. 证据规范化（EvidenceNormalizer）
2. 搜索记忆库（SearchMemory）
3. Cypher执行器（CypherExecutor）
"""

import json
import time
import logging
from typing import Dict, Any, List, Optional, Tuple

from py2neo.data import Node as NeoNode
from py2neo.data import Relationship as NeoRelationship
from py2neo.data import Path as NeoPath

from utils.graph_db import Neo4JDB
from strategy_agent.plan import SearchEvidence

logger = logging.getLogger(__name__)


class EvidenceNormalizer:
    """
    证据规范化器

    功能：将Neo4j返回的原始数据（Node, Relationship, Path）
    转换为统一的字典格式，便于序列化和存储
    """

    def normalize_records(
        self, records: List[Dict[str, Any]], metadata: Dict[str, Any]
    ) -> List[SearchEvidence]:
        """
        将Neo4j查询结果规范化为SearchEvidence列表

        Args:
            records: Neo4j查询返回的记录列表
            metadata: 元数据（query_name, query_block, params等）

        Returns:
            List[SearchEvidence]: 规范化后的证据列表
        """
        evidence_list: List[SearchEvidence] = []
        for record in records:
            nodes: List[Dict[str, Any]] = []
            relationships: List[Dict[str, Any]] = []
            self._collect_entities(record, nodes, relationships)
            evidence_list.append(
                SearchEvidence(
                    nodes=self._dedupe(nodes),
                    relationships=self._dedupe(relationships),
                    metadata=dict(metadata),
                )
            )
        return evidence_list

    def _collect_entities(
        self,
        obj: Any,
        nodes: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
    ) -> None:
        """递归收集所有节点和关系"""
        if obj is None:
            return
        if isinstance(obj, NeoNode):
            nodes.append(self._node_to_dict(obj))
            return
        if isinstance(obj, NeoRelationship):
            relationships.append(self._rel_to_dict(obj))
            return
        if isinstance(obj, NeoPath):
            for node in obj.nodes:
                nodes.append(self._node_to_dict(node))
            for rel in obj.relationships:
                relationships.append(self._rel_to_dict(rel))
            return
        if isinstance(obj, dict):
            for value in obj.values():
                self._collect_entities(value, nodes, relationships)
            return
        if isinstance(obj, list):
            for item in obj:
                self._collect_entities(item, nodes, relationships)
            return

    def _node_to_dict(self, node: NeoNode) -> Dict[str, Any]:
        """将Neo4j节点转换为字典"""
        return {
            "id": node.identity,
            "labels": list(node.labels),
            "properties": dict(node),
        }

    def _rel_to_dict(self, rel: NeoRelationship) -> Dict[str, Any]:
        """将Neo4j关系转换为字典"""
        return {
            "id": rel.identity,
            "type": rel.__class__.__name__ if hasattr(rel, "__class__") else "REL",
            "start": rel.start_node.identity if rel.start_node else None,
            "end": rel.end_node.identity if rel.end_node else None,
            "properties": dict(rel),
        }

    def _dedupe(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """去重"""
        seen = set()
        result = []
        for item in items:
            key = json.dumps(item, sort_keys=True, ensure_ascii=False)
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result


class SearchMemory:
    """
    搜索智能体记忆库

    核心职责: 记录已匹配的引种-入群事件对，避免重复查询

    设计说明:
    - 记录“引种事件 ↔ 入群事件”的匹配对（无论入群是否有风险）
    - 在后续查询中，跳过已匹配的事件，避免重复分析
    """

    def __init__(self):
        # 记录已匹配的事件ID与事件对
        self.completed_intro_events = set()  # 已匹配入群的引种事件
        self.completed_group_events = set()  # 已匹配引种的入群事件
        self.completed_pairs = set()  # (intro_event_id, group_event_id) 匹配事件对

    def record_complete_risk_path(
        self, intro_event_id: Optional[str], group_event_id: Optional[str]
    ) -> None:
        """
        记录匹配的引种-入群事件对

        只要满足以下条件就记录:
        1. intro_event_id 和 group_event_id 都存在

        Args:
            intro_event_id: 引种事件ID
            group_event_id: 入群事件ID
        """
        if intro_event_id and group_event_id:
            self.completed_intro_events.add(intro_event_id)
            self.completed_group_events.add(group_event_id)
            self.completed_pairs.add((intro_event_id, group_event_id))

    def is_intro_completed(self, intro_event_id: Optional[str]) -> bool:
        """检查引种事件是否已匹配入群事件"""
        return bool(intro_event_id) and intro_event_id in self.completed_intro_events

    def is_group_completed(self, group_event_id: Optional[str]) -> bool:
        """检查入群事件是否已匹配引种事件"""
        return bool(group_event_id) and group_event_id in self.completed_group_events

    def get_stats(self) -> Dict[str, int]:
        """获取记忆库统计信息"""
        return {
            "completed_intro_events": len(self.completed_intro_events),
            "completed_group_events": len(self.completed_group_events),
            "completed_pairs": len(self.completed_pairs),
        }


class CypherExecutor:
    """
    Cypher执行器

    功能：
    1. 执行Cypher查询
    2. 提取seed输入
    3. 更新记忆库
    4. 规范化证据
    """

    def __init__(self, db: Neo4JDB, normalizer: EvidenceNormalizer, memory: SearchMemory):
        self.db = db
        self.normalizer = normalizer
        self.memory = memory

    def execute_query(
        self,
        cypher: str,
        params: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[SearchEvidence]]:
        """
        执行单个Cypher查询

        Returns:
            (records, evidence): 原始记录和规范化后的证据
        """
        try:
            records = self.db.query(cypher, **params)
            evidence = []
            if records:
                evidence = self.normalizer.normalize_records(records, metadata)
            return records or [], evidence
        except Exception as e:
            logger.error(f"Cypher执行失败: {str(e)}")
            logger.error(f"Cypher: {cypher[:200]}...")
            logger.error(f"Params: {params}")
            raise

    def extract_seed_inputs(
        self,
        records: List[Dict[str, Any]],
        target_event: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        从查询结果中提取seed输入

        Args:
            records: 查询返回的记录
            target_event: 目标事件类型（IntroEvent, GroupEvent等）

        Returns:
            List[Dict]: seed参数列表，供follow查询使用
        """
        if not target_event:
            return []

        seeds: List[Dict[str, Any]] = []
        for record in records:
            node = self._find_first_node(record, target_event)
            if not node:
                continue

            seed = {}
            event_id = node.get("event_id") if isinstance(node, dict) else None

            if target_event == "IntroEvent":
                seed = {
                    "seed_intro_event_id": event_id,
                    "seed_pig_batch": node.get("pig_batch") if isinstance(node, dict) else None,
                    "seed_end_date": node.get("end_date") if isinstance(node, dict) else None,
                }
            elif target_event == "GroupEvent":
                seed = {
                    "seed_group_event_id": event_id,
                    "seed_pig_batch": node.get("pig_batch") if isinstance(node, dict) else None,
                    "seed_begin_date": node.get("begin_date") if isinstance(node, dict) else None,
                }
            elif target_event == "BreedEvent":
                seed = {"seed_breed_event_id": event_id}
            elif target_event == "DeliveryEvent":
                seed = {"seed_delivery_event_id": event_id}
            elif target_event == "NormalImmuEvent":
                seed = {"seed_normal_immu_event_id": event_id}
            elif target_event == "WeatherEvent":
                seed = {"seed_weather_event_id": event_id}
            # 注意：AbortEvent 没有 event_id，不提取 seed
            # Step 3 (follow到流产率异常) 直接使用 org_inv_dk + inference_date 查询
            elif event_id:
                seed = {"seed_event_id": event_id, "seed_event_label": target_event}

            if seed:
                seeds.append(seed)

        return seeds

    def filter_seed_by_memory(
        self,
        seeds: List[Dict[str, Any]],
        target_event: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        使用记忆库过滤已处理的seed

        根据目标事件类型过滤:
        - IntroEvent: 过滤已匹配入群的引种事件
        - GroupEvent: 过滤已匹配引种的入群事件
        """
        if target_event == "IntroEvent":
            filtered = []
            for seed in seeds:
                intro_event_id = seed.get("seed_intro_event_id")
                if not self.memory.is_intro_completed(intro_event_id):
                    filtered.append(seed)
            return filtered
        elif target_event == "GroupEvent":
            filtered = []
            for seed in seeds:
                group_event_id = seed.get("seed_group_event_id")
                if not self.memory.is_group_completed(group_event_id):
                    filtered.append(seed)
            return filtered
        else:
            return seeds

    def update_memory_from_records(
        self,
        records: List[Dict[str, Any]],
        seed: Dict[str, Any]
    ) -> None:
        """
        更新记忆库，记录匹配到的引种-入群事件对
        """
        for record in records:
            # 只要同一条记录中匹配到引种与入群事件就记录
            intro_event_id = seed.get("seed_intro_event_id") or self._find_event_id(
                record, "IntroEvent"
            )
            group_event_id = seed.get("seed_group_event_id") or self._find_event_id(
                record, "GroupEvent"
            )
            if intro_event_id and group_event_id:
                self.memory.record_complete_risk_path(intro_event_id, group_event_id)

    def _find_first_node(
        self,
        record: Dict[str, Any],
        label: str
    ) -> Optional[Dict[str, Any]]:
        """查找第一个匹配标签的节点（递归遍历 record / list / dict / path_nodes 等结构）"""

        def walk(obj: Any) -> Optional[Dict[str, Any]]:
            if obj is None:
                return None
            if isinstance(obj, NeoNode):
                if label in obj.labels:
                    return dict(obj)
                return None
            if isinstance(obj, NeoPath):
                for n in obj.nodes:
                    if label in n.labels:
                        return dict(n)
                return None
            if isinstance(obj, dict):
                if obj.get("label") == label:
                    return obj
                for v in obj.values():
                    hit = walk(v)
                    if hit:
                        return hit
                return None
            if isinstance(obj, list):
                for item in obj:
                    hit = walk(item)
                    if hit:
                        return hit
                return None
            return None

        return walk(record)

    def _find_event_id(
        self,
        record: Dict[str, Any],
        label: str
    ) -> Optional[str]:
        """查找指定类型节点的event_id"""
        node = self._find_first_node(record, label)
        return node.get("event_id") if node else None

    def _has_intro_risk(self, record: Dict[str, Any]) -> bool:
        """检查记录中是否存在引种风险"""
        intro_risks = record.get("intro_risks")
        if isinstance(intro_risks, list) and intro_risks:
            return True
        for risk in self._iter_risk_nodes(record):
            link = risk.get("link")
            if link == "引种":
                return True
        return False

    def _has_group_risk(self, record: Dict[str, Any]) -> bool:
        """检查记录中是否存在入群风险"""
        group_risks = record.get("group_risks")
        if isinstance(group_risks, list) and group_risks:
            return True
        for risk in self._iter_risk_nodes(record):
            link = risk.get("link")
            if link == "入群":
                return True
        return False

    def _iter_risk_nodes(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """迭代记录中的所有RiskEvent节点"""
        risks: List[Dict[str, Any]] = []
        for value in record.values():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, NeoNode) and "RiskEvent" in item.labels:
                        risks.append(dict(item))
                    elif isinstance(item, dict) and item.get("risk_event"):
                        risks.append(item)
            elif isinstance(value, NeoNode) and "RiskEvent" in value.labels:
                risks.append(dict(value))
            elif isinstance(value, dict) and value.get("risk_event"):
                risks.append(value)
        return risks
