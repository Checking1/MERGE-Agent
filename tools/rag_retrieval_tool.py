"""
RAG检索工具

功能：从专家知识库检索相关领域规则
用于：Strategy Agent获取风险传播模式和事件关联规则
"""

from langchain_core.tools import tool
import json


def create_rag_retrieval_tool(rag_instance):
    """
    创建RAG检索工具

    Args:
        rag_instance: Rag实例

    Returns:
        Tool对象
    """

    @tool
    def retrieve_domain_rules(query: str, top_k: int = 10) -> str:
        """从专家知识库检索与当前场景相关的领域规则。

        知识库包含：
        - 引种风险规则：引种事件如何通过批次关联到入群事件
        - 入群风险规则：入群风险的传播机制
        - 配种风险规则：配种风险的传播机制和影响
        - 风险传播模式：后备猪风险 → 基础母猪风险的链路
        - 事件关联规则：IntroEvent和GroupEvent的关联方式

        Args:
            query: 查询文本，描述当前的风险场景或需要的规则。
                   例如: "引种事件3次有风险，入群28次有风险，如何关联"
            top_k: 返回最相关的规则数量，默认10条

        Returns:
            JSON格式的规则列表，每条规则包含：
            - rule_id: 规则编号
            - title: 规则标题
            - content: 规则内容（截取前300字符）
            - relevance: 相关度（high/medium/low）

        使用场景：
        - 了解引种和入群如何通过批次号或时间关联
        - 了解风险如何从引种传播到入群再到流产
        - 获取查询计划设计的参考规则
        - 确定需要追踪哪些风险路径
        """
        try:
            # 混合检索策略：精确匹配 + 向量检索 + 去重

            # 1. 提取query中的关键事件类型（精确匹配）
            event_keywords = ["引种", "入群", "配种", "分娩", "免疫",'产房管理']
            exact_results = []
            for keyword in event_keywords:
                if keyword in query:
                    keyword_docs = rag_instance.exact_title_search(keyword)
                    exact_results.extend(keyword_docs)

            # 2. 向量检索（语义相似度）
            vector_results = rag_instance.similarity_content_search(query, k=top_k * 2)

            # 3. 合并结果（精确匹配优先）
            all_results = exact_results + vector_results

            # 4. 去重（基于title）
            seen_titles = set()
            unique_results = []
            for doc in all_results:
                title = doc.metadata.get("title", "")
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    unique_results.append(doc)

            # 5. 截取top_k
            unique_results = unique_results[:top_k]

            # 6. 格式化输出
            formatted_rules = []
            for i, doc in enumerate(unique_results, 1):
                # 确定相关度：精确匹配的规则相关度更高
                is_exact_match = doc in exact_results
                relevance = "high" if is_exact_match else ("medium" if i <= 2 else "low")

                formatted_rules.append({
                    "rule_id": i,
                    "title": doc.metadata.get("title", "未知规则"),
                    "content": doc.metadata.get("content", "")[:300],  # 限制长度
                    "relevance": relevance
                })

            result = {
                "success": True,
                "total_rules": len(formatted_rules),
                "rules": formatted_rules
            }

            return json.dumps(result, ensure_ascii=False, indent=2)

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"规则检索失败: {str(e)}"
            }, ensure_ascii=False)

    return retrieve_domain_rules
