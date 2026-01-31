"""
AAT-Agent: Attribution Analysis Tool - Single Task Runner

功能：
1. 执行单个PRRS风险归因任务
2. 使用Strategy Agent生成归因策略
3. 使用Search Agent执行知识图谱查询
4. 输出归因结果和证据子图

使用方法：
python run_attribution.py --farm-id FARM001 --reference-date 2025-01-01 --output results/output.json
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from search_agent import SearchAgent, SearchConfig
from strategy_agent import StrategyAgent, StrategyConfig, TaskContext
import config as agent_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('attribution.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def _append_core_log(path: Path, text: str) -> None:
    """追加核心日志"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def _log_plan_core(path: Path, plan) -> None:
    """记录策略计划到核心日志"""
    _append_core_log(path, "=" * 100)
    _append_core_log(path, f"[PLAN] plan_id={plan.plan_id} trigger_type={plan.trigger_type}")
    _append_core_log(path, f"[PLAN] hypothesis={plan.hypothesis}")
    _append_core_log(path, f"[PLAN] blocks={len(plan.query_blocks)}")
    for idx, qb in enumerate(plan.query_blocks, 1):
        _append_core_log(path, f"[BLOCK#{idx}] {qb.name} - {qb.description}")
        _append_core_log(path, f"[BLOCK#{idx}] filters={qb.filters.to_dict() if qb.filters else None} limit={qb.limit}")
        for sidx, step in enumerate(qb.steps, 1):
            target = step.target_event or (
                f"ConceptNode(type='{step.target_concept_type}')"
                if step.target_concept_type
                else None
            )
            _append_core_log(
                path,
                f"  [STEP#{sidx}] {step.step_id} action={step.action} target={target} limit={step.limit}",
            )
            _append_core_log(path, f"  [STEP#{sidx}] description={step.description}")
    _append_core_log(path, "=" * 100)


def build_task(args: argparse.Namespace) -> Dict[str, Any]:
    """构建任务上下文"""
    return {
        "org_inv_dk": args.farm_id,
        "reference_date": args.reference_date,
        "window_days": args.window_days,
        "trigger_type": args.trigger_type,
    }


def main() -> int:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Run PRRS attribution analysis with AAT-Agent"
    )
    parser.add_argument(
        "--farm-id",
        default=os.getenv("AAT_FARM_ID", "bDoAAD6k2aHM567U"),
        help="Pig farm ID (set AAT_FARM_ID for convenience)",
    )
    parser.add_argument(
        "--reference-date",
        default=os.getenv("AAT_REFERENCE_DATE", "2025-06-28"),
        help="Reference date (YYYY-MM-DD)",
    )
    parser.add_argument("--window-days", type=int, default=60, help="Window days for attribution")
    parser.add_argument("--trigger-type", default="abortion_rate_spike", help="Trigger type")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM calls")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "results" / "output.json"),
        help="JSON output path"
    )

    args = parser.parse_args()

    logger.info("=" * 100)
    logger.info("AAT-Agent: PRRS风险归因分析工具启动")
    logger.info("=" * 100)
    logger.info(f"参数: farm_id={args.farm_id}, reference_date={args.reference_date}, "
               f"window_days={args.window_days}, trigger_type={args.trigger_type}")
    logger.info(f"LLM状态: {'启用' if not args.no_llm else '禁用'}")

    llm_enabled = not args.no_llm

    output_path = Path(args.output)
    plan_path = output_path.with_suffix(".plan.json")
    cypher_log_path = output_path.with_suffix(".cypher.jsonl")
    core_log_path = output_path.with_suffix(".core.log")

    # 创建Strategy Agent配置
    strategy_cfg = StrategyConfig(
        embedding_model_path="sentence-transformers/all-MiniLM-L6-v2",
        rag_data_path="data/rules.csv",
        rag_db_dir="data/rag_db",
        llm_base_url=agent_config.DEEPSEEK_BASE_URL,
        llm_api_key=agent_config.DEEPSEEK_API_KEY,
        llm_model=agent_config.DEEPSEEK_MODEL,
        llm_temperature=0.2,
        llm_enabled=llm_enabled,
    )

    # 创建Search Agent配置
    search_cfg = SearchConfig(
        llm_base_url=agent_config.DEEPSEEK_BASE_URL,
        llm_api_key=agent_config.DEEPSEEK_API_KEY,
        llm_model=agent_config.DEEPSEEK_MODEL,
        llm_temperature=0.2,
        llm_enabled=llm_enabled,
        cypher_log_path=str(cypher_log_path),
        core_log_path=str(core_log_path),
    )

    # 初始化Agent实例
    logger.info("\n初始化AAT-Agent...")
    strategy_agent = StrategyAgent(strategy_cfg)
    search_agent = SearchAgent(search_cfg)

    # 构建任务上下文
    task = build_task(args)
    context = TaskContext.from_dict(task)

    # Step 1: Strategy Agent生成计划
    logger.info("\n" + "="*100)
    logger.info("Step 1: Strategy Agent 生成归因策略")
    logger.info("="*100)
    try:
        plan = strategy_agent.generate_strategy(context)
        logger.info(f"✅ 策略生成成功: {len(plan.query_blocks)} 个QueryBlocks")
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(
            json.dumps(plan.to_dict(), ensure_ascii=True, indent=2),
            encoding="utf-8"
        )
        cypher_log_path.write_text("", encoding="utf-8")
        core_log_path.write_text("", encoding="utf-8")
        _append_core_log(core_log_path, f"[RUN] timestamp={datetime.utcnow().isoformat()}Z")
        _append_core_log(
            core_log_path,
            f"[RUN] farm_id={args.farm_id} reference_date={args.reference_date} window_days={args.window_days} trigger_type={args.trigger_type}",
        )
        _log_plan_core(core_log_path, plan)
        logger.info(f"Plan saved to: {plan_path}")
        logger.info(f"Cypher log: {cypher_log_path}")
        logger.info(f"Core log: {core_log_path}")
    except Exception as e:
        logger.error(f"❌ 策略生成失败: {str(e)}", exc_info=True)
        return 1

    # Step 2: Search Agent执行搜索
    logger.info("\n" + "="*100)
    logger.info("Step 2: Search Agent 执行知识图谱搜索")
    logger.info("="*100)
    try:
        search_results = search_agent.run(plan)
        logger.info(f"✅ 搜索执行成功: {len(search_results)} 个查询块")
    except Exception as e:
        logger.error(f"❌ 搜索执行失败: {str(e)}", exc_info=True)
        return 1
    finally:
        search_agent.close_db()

    # 构建输出payload
    payload = {
        "version": "aat-agent-1.0",
        "plan": plan.to_dict() if plan else None,
        "search_results": [sr.to_dict() for sr in search_results] if search_results else None,
        "memory_stats": search_agent.memory.get_stats(),
    }

    # 输出结果摘要
    logger.info("\n" + "=" * 100)
    logger.info("执行完成")
    logger.info("=" * 100)

    if plan:
        logger.info(f"策略计划: {len(plan.query_blocks)} 个QueryBlocks")
        for idx, qb in enumerate(plan.query_blocks, 1):
            logger.info(f"  QueryBlock {idx}: {qb.name} - {len(qb.steps)} 步骤")

    if search_results:
        total_evidence = sum(len(sr.evidence) for sr in search_results)
        logger.info(f"搜索结果: {len(search_results)} 个查询块, 总证据数={total_evidence}")
        for idx, sr in enumerate(search_results, 1):
            logger.info(f"  QueryBlock {idx}: {sr.query_block} - {sr.status} - "
                       f"证据数={len(sr.evidence)}, 记录数={sr.metrics.get('records', 0)}, "
                       f"耗时={sr.metrics.get('latency_ms', 0)}ms")

    memory_stats = search_agent.memory.get_stats()
    logger.info(f"\n记忆库统计:")
    logger.info(f"  已匹配入群的引种事件: {memory_stats['completed_intro_events']}")
    logger.info(f"  已匹配引种的入群事件: {memory_stats['completed_group_events']}")
    logger.info(f"  匹配事件对数: {memory_stats['completed_pairs']}")

    # 保存到文件
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info(f"\n结果已保存到: {args.output}")

    logger.info("\n" + "=" * 100)
    logger.info("AAT-Agent 执行结束")
    logger.info("=" * 100)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
