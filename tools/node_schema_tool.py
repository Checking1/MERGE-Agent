"""
节点Schema查询工具

功能：查询知识图谱节点的字段信息
用于：Search Agent生成正确的Cypher查询
关键价值：让Agent了解节点有哪些字段，避免使用不存在的字段
"""

from datetime import datetime
import json
import os
import re
from typing import Any, Dict, List

import pandas as pd
from langchain_core.tools import tool


def create_node_schema_tool(
    dataset_base_path: str,
    include_example_values: bool = False,
    example_rows: int = 0,
):
    """
    创建节点Schema查询工具

    Args:
        dataset_base_path: 数据集根目录路径

    Returns:
        Tool对象
    """

    # 映射节点类型到CSV文件路径
    NODE_TYPE_TO_CSV = {
        "IntroEvent": "intro_event_dataset.csv",
        "GroupEvent": "group_event_dataset.csv",
        "BreedEvent": "breed_event_dataset.csv",
        "DeliveryEvent": "delivery_event_dataset.csv",
        "NormalImmuEvent": "normal_immu_event_dataset.csv",
        "AbortEvent": "abort_event_dataset.csv",
        "RiskEvent": "risk_event_dataset.csv",
    }
    SUPPORTED_TYPES = set(NODE_TYPE_TO_CSV.keys()) | {"ConceptNode"}
    SNAPSHOT_NODE_TYPES = {"AbortEvent", "NormalImmuEvent"}
    AVOID_TIME_FIELDS = {
        "inference_date",
        "program_running_dt",
        "program_run_dt",
        "running_dt",
        "run_dt",
    }
    SOFT_AVOID_TIME_FIELDS = {
        "stats_dt",
        "stat_dt",
        "stat_date",
        "stats_date",
    }

    # 关键字段说明（帮助Agent理解字段用途）
    FIELD_DESCRIPTIONS = {
        "IntroEvent": {
            "event_id": "事件唯一标识",
            "pig_batch": "批次号（用于关联GroupEvent，重要！）",
            "allot_dt": "引种日期（用于时间窗口过滤，WHERE条件）",
            "end_date": "结束日期（可用于关联group.begin_date）",
            "begin_date": "开始日期",
            "org_inv_dk": "猪场ID（WHERE条件必须）",
            "inference_date": "推理日期（必须 = $reference_date）",
            "link": "固定值'引种'"
        },
        "GroupEvent": {
            "event_id": "事件唯一标识",
            "pig_batch": "批次号（用于关联IntroEvent，重要！）",
            "min_boar_inpop_dt": "入群日期（用于时间窗口过滤，WHERE条件）",
            "begin_date": "开始日期（可用于关联intro.end_date）",
            "end_date": "结束日期",
            "org_inv_dk": "猪场ID（WHERE条件必须）",
            "inference_date": "推理日期（必须 = $reference_date）",
            "link": "固定值'入群'"
        },
        "RiskEvent": {
            "event_id": "关联的生产事件ID",
            "link": "关联类型（'引种'或'入群'等，WHERE条件）",
            "inference_date": "推理日期（必须 = $reference_date）",
            "risk_event_occur_dt": "风险发生日期"
        },
        "AbortEvent": {
            "inference_date": "推理日期（必须 = $reference_date）",
            "stats_dt": "统计日期（通常为inference_date - 1，不要用于WHERE过滤）",
            "org_inv_dk": "猪场ID（WHERE条件必须）"
        },
        "ConceptNode": {
            "type": "概念类型（如'后备猪风险'、'基础母猪风险'、'流产率异常'）",
            "inference_date": "推理日期（必须 = $reference_date）"
        },
        "BreedEvent": {
            "event_id": "事件唯一标识",
            "inference_date": "推理日期（必须 = $reference_date）",
            "org_inv_dk": "猪场ID"
        },
        "DeliveryEvent": {
            "event_id": "事件唯一标识",
            "stats_dt": "统计日期（分娩事件的业务发生/统计时间；当缺少明确发生日期时可用于时间窗口过滤）",
            "begin_date": "统计窗口起始日期（汇总区间边界，非单点业务时间）",
            "end_date": "统计窗口结束日期（汇总区间边界，非单点业务时间）",
            "inference_date": "推理日期（必须 = $reference_date）",
            "org_inv_dk": "猪场ID",
            "link": "固定值“分娩”"
        },
        "NormalImmuEvent": {
            "event_id": "事件唯一标识",
            "inference_date": "推理日期（必须 = $reference_date）",
            "org_inv_dk": "猪场ID",
            "stats_dt": "统计日期（常规免疫是推理日期的快照节点；stats_dt 通常是推理日前固定一天，不要用于 start/end 时间窗口过滤）",
            "link": "固定为“免疫”"
        }
    }

    def _is_date_like(value: object) -> bool:
        if value is None:
            return False
        text = str(value).strip()
        if not text:
            return False
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
            try:
                datetime.strptime(text[:10], fmt)
                return True
            except Exception:
                pass
        return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", text[:10]))

    def _infer_time_fields(node_type: str, df: pd.DataFrame) -> Dict[str, Any]:
        candidates: List[Dict[str, Any]] = []
        for col in df.columns:
            col_name = str(col)
            col_l = col_name.lower()
            desc = (FIELD_DESCRIPTIONS.get(node_type, {}) or {}).get(col_name, "")
            samples = []
            if include_example_values and len(df) > 0:
                samples = df[col].dropna().head(5).tolist()

            score = 0
            reasons: List[str] = []

            if any(k in col_l for k in ("dt", "date", "time")):
                score += 2
                reasons.append("name_pattern")

            if any(_is_date_like(v) for v in samples):
                score += 3
                reasons.append("date_like_samples")

            if desc:
                if any(k in desc for k in ("事件发生", "发生日期", "业务发生", "业务日期", "用于时间窗口", "用于窗口过滤")):
                    score += 3
                    reasons.append("description_occurrence_time")
                elif any(k in desc for k in ("日期", "时间", "发生", "分配", "入群", "配种", "分娩", "免疫")):
                    score += 1
                    reasons.append("description_mentions_time")

                if any(k in desc for k in ("快照", "推理日期", "统计窗口", "汇总")):
                    score -= 2
                    reasons.append("description_snapshot_or_stats")
                elif "统计" in desc and not any(k in desc for k in ("事件发生", "发生日期", "业务发生", "业务日期", "用于时间窗口", "用于窗口过滤")):
                    score -= 2
                    reasons.append("description_stats_not_occurrence_time")

            if col_l in AVOID_TIME_FIELDS:
                score -= 5
                reasons.append("avoid_field")
            elif col_l in SOFT_AVOID_TIME_FIELDS:
                score -= 1
                reasons.append("soft_avoid_stats_field")

            if score > 0 and ("name_pattern" in reasons or "date_like_samples" in reasons):
                candidates.append(
                    {
                        "field_name": col_name,
                        "score": score,
                        "reasons": reasons,
                        "sample_values": ([str(v) for v in samples[:2]] if include_example_values else []),
                        "description": desc,
                    }
                )

        candidates.sort(key=lambda x: x["score"], reverse=True)
        recommended = candidates[0]["field_name"] if candidates else None
        window_filter_recommended = (
            node_type not in SNAPSHOT_NODE_TYPES and node_type not in {"RiskEvent", "ConceptNode"}
        )

        if node_type in SNAPSHOT_NODE_TYPES:
            notes = (
                f"{node_type} is treated as a snapshot/aggregate node in this KG; "
                "do NOT apply $start_date/$end_date window filtering; "
                "filter by inference_date (and org_inv_dk if available) only."
            )
        else:
            notes = (
                "Prefer an event-occurrence time field for $start_date/$end_date filtering; "
                "avoid inference_date and pure stats_dt/stat_dt snapshot-like fields unless the schema/field semantics indicate they represent business time."
            )

        return {
            "snapshot_node": node_type in SNAPSHOT_NODE_TYPES,
            "window_filter_recommended": bool(window_filter_recommended),
            "avoid_fields": sorted(AVOID_TIME_FIELDS),
            "soft_avoid_fields": sorted(SOFT_AVOID_TIME_FIELDS),
            "candidate_time_fields": candidates[:8],
            "recommended_time_field": recommended,
            "example_window_predicate": (
                f"{node_type}.{recommended} >= $start_date AND {node_type}.{recommended} <= $end_date"
                if window_filter_recommended and recommended
                else None
            ),
            "notes": notes
            + " If the only date-like field is stats_dt/stat_dt on a non-snapshot node, you may use it as the occurrence-time field.",
        }

    @tool
    def get_node_schema(node_type: str) -> str:
        """查询知识图谱中指定节点类型的Schema信息（字段列表和说明）。

        **关键用途**：
        1. 生成Cypher查询前，了解节点有哪些可用字段
        2. 确定时间过滤字段（基于字段名/样例值/字段描述推断，避免硬编码）
        3. 确定关联字段（如pig_batch, event_id）
        4. 避免使用不存在的字段导致查询失败

        **重要规则**：
        - 所有节点都必须匹配 inference_date = $reference_date
        - AbortEvent不需要时间窗口过滤，只匹配inference_date（图谱快照节点）
        - NormalImmuEvent通常也不需要时间窗口过滤，只匹配inference_date（图谱快照节点）
        - 其他事件节点的时间窗口字段应基于字段含义推断（优先选择“事件发生/业务发生日期”；若schema提示stats_dt是唯一/最佳业务时间字段，也可用它进行窗口过滤）
        - IntroEvent和GroupEvent可通过pig_batch关联
        - 当关键字段发生变更（例如日期字段重命名）时，应通过schema信息与字段语义重新选择过滤字段，而不是依赖固定模板

        Args:
            node_type: 节点类型名称，支持的类型：
                - IntroEvent: 引种事件
                - GroupEvent: 入群事件
                - BreedEvent: 配种事件
                - DeliveryEvent: 分娩事件
                - NormalImmuEvent: 常规免疫事件
                - AbortEvent: 流产事件
                - RiskEvent: 风险事件
                - ConceptNode: 概念节点

        Returns:
            JSON格式的Schema信息，包含：
            - node_type: 节点类型
            - total_fields: 字段总数
            - fields: 字段列表（每个字段包含name, type, description, importance）
            - cypher_tips: Cypher查询建议

        使用场景：
        - 不确定节点有哪些字段时
        - 需要写WHERE条件但不知道字段名时
        - 需要关联两个节点但不知道用哪个字段时
        """
        try:
            # 检查节点类型
            if node_type not in SUPPORTED_TYPES:
                return json.dumps({
                    "success": False,
                    "error": f"不支持的节点类型 '{node_type}'",
                    "supported_types": sorted(SUPPORTED_TYPES)
                }, ensure_ascii=False, indent=2)

            if node_type == "ConceptNode":
                fields = [
                    {
                        "field_name": "type",
                        "data_type": "object",
                        "sample_values": [],
                        "description": "概念类型",
                        "importance": "high"
                    },
                    {
                        "field_name": "inference_date",
                        "data_type": "object",
                        "sample_values": [],
                        "description": "推理日期（必须= $reference_date）",
                        "importance": "high"
                    }
                ]
                cypher_tips = {
                    "必须条件": "ConceptNode.inference_date = $reference_date",
                    "概念类型": "ConceptNode.type 用于限定概念"
                }
                schema_info = {
                    "success": True,
                    "node_type": node_type,
                    "label": node_type,
                    "total_fields": len(fields),
                    "fields": fields,
                    "cypher_tips": cypher_tips
                }
                return json.dumps(schema_info, ensure_ascii=False, indent=2)

            # 读取CSV获取schema
            csv_path = os.path.join(dataset_base_path, NODE_TYPE_TO_CSV[node_type])

            if not os.path.exists(csv_path):
                return json.dumps({
                    "success": False,
                    "error": f"数据集文件不存在: {csv_path}"
                }, ensure_ascii=False, indent=2)

            # Privacy-safe default: read header only (nrows=0) and avoid returning example values.
            # If you really need example values for internal debugging, set include_example_values=True.
            read_rows = int(example_rows) if include_example_values and int(example_rows) > 0 else 0
            df = pd.read_csv(csv_path, nrows=read_rows)

            # 提取字段信息
            fields = []
            for col in df.columns:
                field_info = {
                    "field_name": col,
                    "data_type": str(df[col].dtype),
                    "sample_values": (
                        df[col].dropna().head(2).tolist()
                        if include_example_values and len(df) > 0
                        else []
                    ),
                }

                # 添加字段说明（如果有）
                if node_type in FIELD_DESCRIPTIONS and col in FIELD_DESCRIPTIONS[node_type]:
                    field_info["description"] = FIELD_DESCRIPTIONS[node_type][col]
                    field_info["importance"] = "high"
                else:
                    field_info["importance"] = "low"

                fields.append(field_info)

            time_filter_guidance = _infer_time_fields(node_type, df)

            # 构建Cypher提示
            cypher_tips = {
                "必须条件": f"{node_type}.inference_date = $reference_date",
                "猪场过滤": f"{node_type}.org_inv_dk = $org_inv_dk" if node_type != "ConceptNode" else "ConceptNode无org_inv_dk字段"
            }

            # 添加特定节点的提示
            if node_type == "IntroEvent":
                cypher_tips["关联字段"] = "pig_batch（关联GroupEvent）或 end_date（关联group.begin_date）"
            elif node_type == "GroupEvent":
                cypher_tips["关联字段"] = "pig_batch（关联IntroEvent）或 begin_date（关联intro.end_date）"
            elif node_type == "RiskEvent":
                cypher_tips["关联类型"] = "link字段区分风险类型（如'引种'、'入群'等）"
            elif node_type == "ConceptNode":
                cypher_tips["概念类型"] = "type字段：'后备猪风险'、'基础母猪风险'、'流产率异常'等"

            # 这些规则属于图谱构建的“自然背景”（非工程兜底），保留以保证语义一致性
            if node_type == "AbortEvent":
                cypher_tips["重要提示"] = "不要使用stats_dt做时间窗口过滤，只匹配inference_date"
            elif node_type == "NormalImmuEvent":
                cypher_tips["重要提示"] = (
                    "NormalImmuEvent 是按 inference_date 聚合的快照节点（一个猪场在一个 inference_date 通常只有 1 个节点）；"
                    "不要用 stats_dt 做 start/end 时间窗口过滤，只匹配 inference_date + org_inv_dk。"
                )

            # ------------------------------------------------------------------
            # 鲁棒模式：基于字段含义推断时间窗口字段（覆盖上面可能存在的硬编码提示）
            # ------------------------------------------------------------------
            if time_filter_guidance.get("window_filter_recommended") and time_filter_guidance.get("recommended_time_field"):
                tf = time_filter_guidance["recommended_time_field"]
                cypher_tips["时间窗口"] = f"{node_type}.{tf} >= $start_date AND {node_type}.{tf} <= $end_date"
            else:
                cypher_tips["时间窗口"] = "不使用 $start_date/$end_date；仅使用 inference_date（以及可用时的 org_inv_dk）过滤"

            cypher_tips["时间字段推断原则"] = (
                "优先选择事件发生/业务发生日期字段；避免使用 inference_date 作为时间窗口字段；"
                "stats_dt/stat_dt 等字段通常为统计/快照时间，但当 schema/字段语义表明其代表业务时间（或为唯一候选）时也可用于窗口过滤。"
            )
            cypher_tips["候选时间字段"] = [
                {"field_name": c["field_name"], "score": c["score"], "reasons": c["reasons"]}
                for c in (time_filter_guidance.get("candidate_time_fields") or [])
            ]

            # 构建返回结果
            schema_info = {
                "success": True,
                "node_type": node_type,
                "label": node_type,
                "total_fields": len(fields),
                "fields": fields,
                "time_filter_guidance": time_filter_guidance,
                "cypher_tips": cypher_tips
            }

            return json.dumps(schema_info, ensure_ascii=False, indent=2)

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Schema查询失败: {str(e)}"
            }, ensure_ascii=False, indent=2)

    return get_node_schema
