"""
知识图谱关系发现工具（基于实例学习）

目标：让搜索智能体从 Neo4j 实例中“学习 schema”，而不是依赖静态模板。

核心设计：
- ConceptNode 通过 `type` 作为“子标签”学习关系
- RiskEvent 通过 `link` 作为“子标签”学习关系
- 不输出白名单、不做 support 排序，只输出去重后的结构信息
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import tool
from py2neo import Graph


def _build_graph(neo4j_params: Dict[str, Any]) -> Graph:
    uri = neo4j_params.get("uri")
    if not uri:
        host = neo4j_params.get("host", "localhost")
        port = neo4j_params.get("port", 7687)
        uri = f"bolt://{host}:{port}"

    user = neo4j_params.get("user", "neo4j")
    password = neo4j_params.get("password", os.getenv("NEO4J_PASSWORD", ""))
    name = neo4j_params.get("name")
    if name:
        return Graph(uri, auth=(user, password), name=name)
    return Graph(uri, auth=(user, password))


def create_kg_relationship_tool(
    neo4j_params: Optional[Dict[str, Any]] = None,
    cache_ttl_s: int = 300,
    max_db_calls: int = 1,
):
    """
    创建 KG 关系学习工具。

    Args:
        neo4j_params: Neo4j 连接参数（host/port/user/password/name 或 uri/name）。
        cache_ttl_s: 缓存 TTL（秒），避免重复扫描实例图。
        max_db_calls: Max allowed Neo4j calls; after limit, return cached payload only.
    """

    params: Dict[str, Any] = dict(neo4j_params or {})
    graph: Optional[Graph] = None
    cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    db_call_count = 0
    last_cache_key: Optional[Tuple[Any, ...]] = None

    def get_graph() -> Graph:
        nonlocal graph
        if graph is None:
            graph = _build_graph(params)
        return graph

    def cached(key: Tuple[Any, ...]) -> Optional[Dict[str, Any]]:
        item = cache.get(key)
        if not item:
            return None
        if cache_ttl_s <= 0:
            return item
        if (time.time() - item.get("_ts", 0)) <= cache_ttl_s:
            return item
        return None

    def cached_fallback(key: Tuple[Any, ...]) -> Optional[Dict[str, Any]]:
        item = cache.get(key)
        if item:
            return item
        if last_cache_key is not None:
            return cache.get(last_cache_key)
        return None

    def set_cache(key: Tuple[Any, ...], payload: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal last_cache_key
        payload["_ts"] = time.time()
        cache[key] = payload
        last_cache_key = key
        return payload

    def run(query: str, **kwargs) -> List[Dict[str, Any]]:
        g = get_graph()
        return g.run(query, **kwargs).data()

    @tool
    def query_kg_relationships(
        query_type: str = "all",
        org_inv_dk: Optional[str] = None,
        reference_date: Optional[str] = None,
        use_consistency_constraints: bool = True,
        max_rows: int = 2000,
    ) -> str:
        """从Neo4j实例中动态学习知识图谱schema。

        Args:
            query_type: 查询类型
                - "topology": 学习图拓扑结构 (from_labels)-[rel]->(to_labels)
                - "risk_links": 学习所有RiskEvent.link值
                - "concept_types": 学习所有ConceptNode.type值
                - "typed_signatures": 学习RiskEvent->ConceptNode连接模式（risk_link + rel_type + concept_type）
                - "concept_relationships": 学习ConceptNode的入边和出边关系（按concept_type分组）
                - "all": 返回以上所有信息
            org_inv_dk: 猪场ID（当前版本暂未使用，预留）
            reference_date: 推理日期（当前版本暂未使用，预留）
            use_consistency_constraints: 是否使用一致性约束（当前版本暂未使用，预留）
            max_rows: 最大返回行数限制

        Returns:
            JSON格式的schema信息，包含:
            - topology: 所有节点类型间的关系
            - risk_links: 所有RiskEvent.link值
            - concept_types: 所有ConceptNode.type值
            - risk_to_concept: RiskEvent->ConceptNode的具体连接
            - concept_outgoing: ConceptNode的出边关系（按type分组）
            - concept_incoming: ConceptNode的入边关系（按type分组）
        """
        nonlocal db_call_count, last_cache_key

        query_type_norm = (query_type or "all").strip().lower()

        key = (query_type_norm, org_inv_dk, reference_date, bool(use_consistency_constraints), int(max_rows))
        hit = cached(key)
        if hit is not None:
            payload = dict(hit)
            payload.pop("_ts", None)
            payload["cached"] = True
            return json.dumps(payload, ensure_ascii=False, indent=2)
        if max_db_calls >= 0 and db_call_count >= max_db_calls:
            fallback = cached_fallback(key)
            if fallback is not None:
                payload = dict(fallback)
                payload.pop("_ts", None)
                payload["cached"] = True
                payload["cache_fallback"] = True
                return json.dumps(payload, ensure_ascii=False, indent=2)
            return json.dumps(
                {
                    "success": False,
                    "query_type": query_type_norm,
                    "error": "query_kg_relationships reached max call limit and no cache is available",
                },
                ensure_ascii=False,
                indent=2,
            )

        try:
            db_call_count += 1
            payload: Dict[str, Any] = {
                "success": True,
                "query_type": query_type_norm,
                "cached": False,
                "filters": {
                    "org_inv_dk": org_inv_dk,
                    "reference_date": reference_date,
                    "use_consistency_constraints": bool(use_consistency_constraints),
                    "max_rows": int(max_rows),
                },
            }

            if query_type_norm in ("all", "topology"):
                rows = run(
                    "MATCH (a)-[r]->(b)\n"
                    "RETURN DISTINCT labels(a) AS from_labels, type(r) AS rel_type, labels(b) AS to_labels"
                )
                payload["topology"] = rows

            if query_type_norm in ("all", "risk_links"):
                rows = run(
                    "MATCH (r:RiskEvent)\n"
                    "WHERE r.link IS NOT NULL\n"
                    "RETURN DISTINCT r.link AS risk_link"
                )
                payload["risk_links"] = [r["risk_link"] for r in rows if r.get("risk_link") is not None]

            if query_type_norm in ("all", "concept_types"):
                rows = run(
                    "MATCH (c:ConceptNode)\n"
                    "WHERE c.type IS NOT NULL\n"
                    "RETURN DISTINCT c.type AS concept_type"
                )
                payload["concept_types"] = [r["concept_type"] for r in rows if r.get("concept_type") is not None]

            if query_type_norm in ("all", "typed_signatures"):
                # RiskEvent -> ConceptNode 签名
                rows = run(
                    "MATCH (r:RiskEvent)-[rel]->(c:ConceptNode)\n"
                    "WHERE r.link IS NOT NULL AND c.type IS NOT NULL\n"
                    "RETURN DISTINCT r.link AS risk_link, type(rel) AS rel_type, c.type AS concept_type"
                )
                payload["risk_to_concept"] = rows

            if query_type_norm in ("all", "concept_relationships"):
                # ConceptNode 出边关系 (按concept_type分组)
                rows = run(
                    "MATCH (c:ConceptNode)-[rel]->(target)\n"
                    "WHERE c.type IS NOT NULL\n"
                    "RETURN DISTINCT c.type AS concept_type, type(rel) AS rel_type, labels(target) AS target_labels"
                )
                payload["concept_outgoing"] = rows

                # ConceptNode 入边关系 (按concept_type分组)
                rows = run(
                    "MATCH (source)-[rel]->(c:ConceptNode)\n"
                    "WHERE c.type IS NOT NULL\n"
                    "RETURN DISTINCT labels(source) AS source_labels, type(rel) AS rel_type, c.type AS concept_type"
                )
                payload["concept_incoming"] = rows

            payload = set_cache(key, payload)
            payload.pop("_ts", None)
            return json.dumps(payload, ensure_ascii=False, indent=2)

        except Exception as e:
            return json.dumps(
                {
                    "success": False,
                    "query_type": query_type_norm,
                    "error": f"关系查询失败: {str(e)}",
                },
                ensure_ascii=False,
                indent=2,
            )

    return query_kg_relationships
