from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class TaskContext:
    trigger_type: str
    reference_date: str
    window_days: int
    org_inv_dk: Optional[str] = None
    farm_name: Optional[str] = None
    abortion_rate: Optional[float] = None
    recent_events: Optional[List[Dict[str, Any]]] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trigger_type": self.trigger_type,
            "reference_date": self.reference_date,
            "window_days": self.window_days,
            "org_inv_dk": self.org_inv_dk,
            "farm_name": self.farm_name,
            "abortion_rate": self.abortion_rate,
            "recent_events": self.recent_events,
            "tags": self.tags,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskContext":
        return cls(
            trigger_type=data.get("trigger_type", "abortion_rate_spike"),
            reference_date=data.get("reference_date"),
            window_days=int(data.get("window_days", 90)),
            org_inv_dk=data.get("org_inv_dk"),
            farm_name=data.get("farm_name"),
            abortion_rate=data.get("abortion_rate"),
            recent_events=data.get("recent_events"),
            tags=data.get("tags", {}),
            extra=data.get("extra", {}),
        )


@dataclass
class SearchFilter:
    window_days: int
    org_inv_dk: Optional[str] = None
    reference_date: Optional[str] = None
    pig_batch: Optional[str] = None
    source_org_dk: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_days": self.window_days,
            "org_inv_dk": self.org_inv_dk,
            "reference_date": self.reference_date,
            "pig_batch": self.pig_batch,
            "source_org_dk": self.source_org_dk,
            "extra": self.extra,
        }


@dataclass
class QueryStep:
    step_id: str
    description: str
    action: str
    target_event: Optional[str] = None
    target_concept_type: Optional[str] = None  # 新增：支持 target 为 ConceptNode
    expected_relations: List[str] = field(default_factory=list)
    filters: Optional[SearchFilter] = None
    limit: int = 200

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "description": self.description,
            "action": self.action,
            "target_event": self.target_event,
            "target_concept_type": self.target_concept_type,
            "expected_relations": self.expected_relations,
            "filters": self.filters.to_dict() if self.filters else None,
            "limit": self.limit,
        }


@dataclass
class QueryBlock:
    name: str
    description: str
    expected_relations: List[str]
    filters: SearchFilter
    priority: int = 1
    max_depth: int = 6
    limit: int = 200
    steps: List[QueryStep] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "expected_relations": self.expected_relations,
            "filters": self.filters.to_dict(),
            "priority": self.priority,
            "max_depth": self.max_depth,
            "limit": self.limit,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass
class StrategyPlan:
    plan_id: str
    trigger_type: str
    hypothesis: str
    query_blocks: List[QueryBlock]
    evidence_summary: str
    retrieved_rules: List[Dict[str, Any]] = field(default_factory=list)
    priority: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "trigger_type": self.trigger_type,
            "hypothesis": self.hypothesis,
            "priority": self.priority,
            "evidence_summary": self.evidence_summary,
            "retrieved_rules": self.retrieved_rules,
            "query_blocks": [
                qb.to_dict()
                for qb in self.query_blocks
            ],
        }


@dataclass
class CypherQuery:
    name: str
    query_block: str
    cypher: str
    params: Dict[str, Any]
    expected_relations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchEvidence:
    nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_subgraph_json(self) -> Dict[str, Any]:
        """
        将证据转换为子图JSON格式，便于直接保存或导出到归因表
        """
        def _normalize_node(node: Dict[str, Any]) -> Dict[str, Any]:
            labels = node.get("labels") or []
            label = labels[0] if labels else "Unknown"
            return {
                "id": str(node.get("id")),
                "label": label,
                "labels": labels,
                "properties": node.get("properties", {}),
            }

        def _normalize_relationship(rel: Dict[str, Any]) -> Dict[str, Any]:
            rel_type = rel.get("type")
            return {
                "source": str(rel.get("start")),
                "target": str(rel.get("end")),
                "type": rel_type,
                "relationship": rel_type,
                "properties": rel.get("properties", {}),
            }

        return {
            "nodes": [_normalize_node(node) for node in self.nodes],
            "relationships": [_normalize_relationship(rel) for rel in self.relationships],
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": self.nodes,
            "relationships": self.relationships,
            "metadata": self.metadata,
            "nodes_count": len(self.nodes),
            "relationships_count": len(self.relationships),
            "sub_graph_json": self.to_subgraph_json(),
        }


@dataclass
class SearchResult:
    plan_id: str
    query_block: str
    status: str
    evidence: List[SearchEvidence] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "query_block": self.query_block,
            "status": self.status,
            "evidence_count": len(self.evidence),
            "evidence": [item.to_dict() for item in self.evidence],
            "issues": self.issues,
            "metrics": self.metrics,
        }


@dataclass
class AuditResult:
    plan_id: str
    query_block: str
    status: str
    score: float
    issues: List[str] = field(default_factory=list)
    accepted_indices: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "query_block": self.query_block,
            "status": self.status,
            "score": self.score,
            "issues": self.issues,
            "accepted_indices": self.accepted_indices,
        }


@dataclass
class OrchestrationResult:
    plan: StrategyPlan
    search_results: List[SearchResult]
    audit_results: List[AuditResult]
    selected_evidence: List[SearchEvidence] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "search_results": [item.to_dict() for item in self.search_results],
            "audit_results": [item.to_dict() for item in self.audit_results],
            "selected_evidence": [item.to_dict() for item in self.selected_evidence],
        }
