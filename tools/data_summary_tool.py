"""
数据摘要工具

功能：统计时间窗口内的各类事件数量
用于：Strategy Agent了解当前风险场景
"""

from langchain_core.tools import tool
from typing import Optional
import json


def create_data_summary_tool(data_summary_instance):
    """
    创建数据摘要工具

    Args:
        data_summary_instance: DataSummary实例

    Returns:
        Tool对象
    """

    @tool
    def summarize_events(
        org_inv_dk: str,
        reference_date: str,
        window_days: int
    ) -> str:
        """统计时间窗口内的猪场生产事件和风险事件数量。

        这个工具会查询CSV数据集，统计以下事件：
        - IntroEvent (引种事件): 总数 + 伴随风险数
        - GroupEvent (入群事件): 总数 + 伴随风险数
        - BreedEvent (配种事件): 总数 + 伴随风险数
        - DeliveryEvent (分娩事件): 总数 + 伴随风险数
        - NormalImmuEvent (常规免疫事件): 总数 + 伴随风险数
        - AbortEvent (流产事件): 总数

        Args:
            org_inv_dk: 猪场ID，例如 'bDoAAJUqXrnM567U'
            reference_date: 参考日期，格式 'YYYY-MM-DD'，例如 '2025-08-28'
            window_days: 时间窗口天数，例如 60 或 90

        Returns:
            JSON格式的事件统计，包含：
            - summary_text: 格式化的摘要文本
            - intro: {"events": 总数, "risk_events": 风险数}
            - group: {"events": 总数, "risk_events": 风险数}
            - breed: {"events": 总数, "risk_events": 风险数}
            - delivery: {"events": 总数, "risk_events": 风险数}
            - normal_immu: {"events": 总数, "risk_events": 风险数}
            - abort: {"events": 总数}

        使用场景：
        - 生成风险归因计划前，需要了解窗口内有多少引种、入群、配种、分娩、常规免疫、流产事件
        - 判断是否需要关注引种风险、入群风险、配种风险、分娩风险、常规免疫风险
        - 决定生成哪些QueryBlock（intro_risk_trace, group_risk_trace, breed_risk_trace, delivery_risk_trace, normal_immu_risk_trace等）
        """
        try:
            # 调用DataSummary实例
            summary = data_summary_instance.summarize(
                org_inv_dk=org_inv_dk,
                reference_date=reference_date,
                window_days=window_days
            )

            # 格式化输出
            text = data_summary_instance.format_summary_text(summary)

            result = {
                "success": True,
                "summary_text": text,
                "intro": summary.get("intro", {}),
                "group": summary.get("group", {}),
                "breed": summary.get("breed", {}),
                "delivery": summary.get("delivery", {}),
                "normal_immu": summary.get("normal_immu", {}),
                "abort": summary.get("abort", {}),
                "window_days": window_days,
                "reference_date": reference_date
            }

            return json.dumps(result, ensure_ascii=False, indent=2)

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"数据摘要生成失败: {str(e)}"
            }, ensure_ascii=False)

    return summarize_events
