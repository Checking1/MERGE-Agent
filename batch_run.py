"""
Batch attribution execution script for AAT-Agent

Features:
1. Load dataset from CSV
2. Batch execute attribution tasks via run_attribution.py
3. Convert agent output to subgraph CSV format
4. Track execution status and timing
5. Generate execution summary report

Usage:
python batch_run.py --sample-data dataset.csv --output-dir outputs
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd


# ========== Subgraph conversion functions ==========

def normalize_evidence(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize evidence to ensure nodes and relationships fields exist"""
    normalized = dict(evidence or {})
    nodes = normalized.get("nodes")
    relationships = normalized.get("relationships")

    sub_graph = normalized.get("sub_graph_json") or {}
    if not nodes:
        nodes = sub_graph.get("nodes", [])
    if not relationships:
        rels = sub_graph.get("relationships", [])
        relationships = []
        for rel in rels:
            if "start" in rel or "end" in rel:
                relationships.append(rel)
            else:
                relationships.append({
                    "id": rel.get("id"),
                    "type": rel.get("type") or rel.get("relationship"),
                    "start": rel.get("source"),
                    "end": rel.get("target"),
                    "properties": rel.get("properties", {}),
                })

    normalized["nodes"] = nodes or []
    normalized["relationships"] = relationships or []
    normalized["metadata"] = normalized.get("metadata", {})
    return normalized


def normalize_evidences(evidences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Batch normalize evidences"""
    return [normalize_evidence(ev) for ev in evidences]


def extract_context_from_nodes(nodes: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    """Extract farm_id and inference_date from nodes"""
    farm_id = None
    inference_date = None
    for node in nodes or []:
        props = node.get("properties", {})
        if farm_id is None:
            farm_id = props.get("org_inv_dk")
        if inference_date is None:
            inference_date = props.get("inference_date")
        if farm_id and inference_date:
            break
    return farm_id, inference_date


SEED_QUERY_TO_LABEL = {
    "seed_intro_risk": "IntroEvent",
    "seed_group_risk": "GroupEvent",
    "seed_breed_risk": "BreedEvent",
    "seed_delivery_risk": "DeliveryEvent",
    "seed_normal_immu_risk": "NormalImmuEvent",
    "seed_immu_risk": "NormalImmuEvent",
    "seed_weather_risk": "WeatherEvent",
}

SEED_PARAM_KEY_TO_LABEL = {
    "seed_intro_event_id": "IntroEvent",
    "seed_group_event_id": "GroupEvent",
    "seed_breed_event_id": "BreedEvent",
    "seed_delivery_event_id": "DeliveryEvent",
    "seed_normal_immu_event_id": "NormalImmuEvent",
    "seed_weather_event_id": "WeatherEvent",
}


def extract_anchor_from_evidence(evidence: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """Extract anchor (event_label, event_id) from evidence"""
    metadata = evidence.get("metadata", {})
    params = metadata.get("params", {}) if isinstance(metadata, dict) else {}

    for seed_key, label in SEED_PARAM_KEY_TO_LABEL.items():
        event_id = params.get(seed_key)
        if event_id:
            return label, str(event_id)

    query_name = metadata.get("query_name")
    label = SEED_QUERY_TO_LABEL.get(query_name)
    if not label:
        return None

    for node in evidence.get("nodes", []):
        labels = set(node.get("labels", []))
        if label in labels:
            event_id = (node.get("properties") or {}).get("event_id")
            if event_id:
                return label, str(event_id)
    return None


def merge_evidence_group(evidences: List[Dict[str, Any]], metadata_extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Merge nodes and relationships from multiple evidences"""
    nodes_dict: Dict[Any, Dict[str, Any]] = {}
    relationships_dict: Dict[Tuple[Any, Any, Any], Dict[str, Any]] = {}
    metadata: Dict[str, Any] = {}

    for evidence in evidences:
        nodes = evidence.get("nodes", [])
        relationships = evidence.get("relationships", [])
        if not metadata:
            metadata = dict(evidence.get("metadata", {}))

        for node in nodes:
            node_id = node.get("id")
            if node_id not in nodes_dict:
                nodes_dict[node_id] = node

        for rel in relationships:
            rel_key = (rel.get("start"), rel.get("end"), rel.get("type"))
            if rel_key not in relationships_dict:
                relationships_dict[rel_key] = rel

    if metadata_extra:
        metadata.update(metadata_extra)

    return {
        "nodes": list(nodes_dict.values()),
        "relationships": list(relationships_dict.values()),
        "metadata": metadata,
    }


def merge_evidences_by_seed(evidences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge evidences by seed event to form complete attribution paths"""
    if not evidences:
        return []

    evidences = normalize_evidences(evidences)
    anchor_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    tail_groups: Dict[Tuple[Optional[str], Optional[str]], List[Dict[str, Any]]] = defaultdict(list)
    unanchored: List[Dict[str, Any]] = []

    for evidence in evidences:
        metadata = evidence.get("metadata", {})
        if metadata.get("query_name") == "follow_abort_to_abnormal":
            context = extract_context_from_nodes(evidence.get("nodes", []))
            tail_groups[context].append(evidence)
            continue

        anchor = extract_anchor_from_evidence(evidence)
        if anchor:
            anchor_groups[anchor].append(evidence)
        else:
            unanchored.append(evidence)

    tail_by_context: Dict[Tuple[Optional[str], Optional[str]], Dict[str, Any]] = {}
    for context, tail_evs in tail_groups.items():
        tail_by_context[context] = merge_evidence_group(
            tail_evs,
            metadata_extra={"query_name": "follow_abort_to_abnormal"},
        )

    merged_paths: List[Dict[str, Any]] = []
    for (label, event_id), group in anchor_groups.items():
        merged = merge_evidence_group(
            group,
            metadata_extra={
                "merged": True,
                "seed_event_type": label,
                "seed_event_id": event_id,
                "original_evidence_count": len(group),
            },
        )
        context = extract_context_from_nodes(merged.get("nodes", []))
        tail = tail_by_context.get(context)
        if tail:
            merged = merge_evidence_group([merged, tail], metadata_extra=merged.get("metadata", {}))
        merged_paths.append(merged)

    return merged_paths


def is_complete_attribution_chain(evidence: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Check if subgraph is a complete attribution chain"""
    nodes = evidence.get("nodes", []) or []
    rels = evidence.get("relationships", []) or []

    node_labels: Set[str] = set()
    for node in nodes:
        node_labels.update(node.get("labels", []) or [])

    rel_types = {rel.get("type") for rel in rels if rel.get("type")}

    reasons: List[str] = []
    if "PigFarm" not in node_labels:
        reasons.append("missing_pigfarm")
    if "AbortEvent" not in node_labels:
        reasons.append("missing_abort_event")
    if "AFFECTS" not in rel_types:
        reasons.append("missing_affects_to_abort")
    if not ({"EXHIBIT", "HARBOR"} & rel_types):
        reasons.append("missing_abort_to_abnormal")

    return len(reasons) == 0, reasons


def convert_evidence_to_subgraph_row(evidence: Dict[str, Any], path_id: int, inference_date: str, org_inv_nm: str) -> Dict[str, Any]:
    """Convert single evidence to CSV row format"""
    nodes_raw = evidence.get('nodes', [])
    relationships_raw = evidence.get('relationships', [])
    metadata = evidence.get('metadata', {})

    # Convert node format
    nodes = []
    for node in nodes_raw:
        node_id = str(node['id'])
        labels = node.get('labels', [])
        label = labels[0] if labels else 'Unknown'
        properties = node.get('properties', {})

        nodes.append({
            'id': node_id,
            'label': label,
            'labels': labels,
            'properties': properties
        })

    # Convert relationship format
    relationships = []
    for rel in relationships_raw:
        source_id = str(rel['start'])
        target_id = str(rel['end'])
        rel_type = rel['type']

        relationships.append({
            'source': source_id,
            'target': target_id,
            'type': rel_type,
            'relationship': rel_type
        })

    # Build sub_graph_json
    sub_graph = {
        'nodes': nodes,
        'relationships': relationships
    }

    # Extract batch information
    batch = 'N/A'
    for node in nodes_raw:
        props = node.get('properties', {})
        if 'pig_batch' in props and props['pig_batch']:
            batch = props['pig_batch']
            break

    # Build CSV row
    row = {
        'path_id': f'path_{path_id}',
        'sub_graph_json': json.dumps(sub_graph, ensure_ascii=False),
        'org_inv_nm': org_inv_nm,
        'batch': batch,
        'inference_date': inference_date
    }

    return row


def convert_agent_output_to_subgraph_csv(json_output_path: Path, csv_output_path: Path, filter_incomplete: bool = False) -> Dict[str, Any]:
    """
    Convert agent output JSON to subgraph CSV format

    Args:
        json_output_path: Agent output JSON file path
        csv_output_path: Output CSV file path
        filter_incomplete: Whether to filter out incomplete attribution chains (default: False)

    Returns:
        Conversion statistics dictionary
    """
    try:
        # Load JSON output
        with open(json_output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract basic information
        plan = data.get("plan", {})
        farm_id = "Unknown"
        inference_date = "Unknown"

        for qb in plan.get("query_blocks", []):
            filters = qb.get("filters", {})
            if not farm_id or farm_id == "Unknown":
                farm_id = filters.get("org_inv_dk", "Unknown")
            if not inference_date or inference_date == "Unknown":
                inference_date = filters.get("reference_date", "Unknown")

        # Extract farm name
        org_inv_nm = 'Unknown'
        search_results = data.get('search_results', [])

        for result in search_results:
            evidences = normalize_evidences(result.get('evidence', []))
            if evidences:
                first_evidence = evidences[0]
                nodes = first_evidence.get('nodes', [])
                for node in nodes:
                    if 'PigFarm' in node.get('labels', []):
                        org_inv_nm = node.get('properties', {}).get('org_inv_nm', 'Unknown')
                        break
                if org_inv_nm != 'Unknown':
                    break

        # Collect all evidences
        all_evidences = []
        for result in search_results:
            evidences = result.get('evidence', [])
            all_evidences.extend(evidences)

        all_evidences = normalize_evidences(all_evidences)

        # Merge by seed
        processed_evidences = merge_evidences_by_seed(all_evidences)

        # Filter complete chains (optional)
        dropped_count = 0
        if filter_incomplete:
            kept: List[Dict[str, Any]] = []
            for ev in processed_evidences:
                ok, reasons = is_complete_attribution_chain(ev)
                if ok:
                    kept.append(ev)
                else:
                    dropped_count += 1
            processed_evidences = kept

        # Convert to CSV rows
        rows = []
        for idx, evidence in enumerate(processed_evidences, start=1):
            row = convert_evidence_to_subgraph_row(
                evidence=evidence,
                path_id=idx,
                inference_date=inference_date,
                org_inv_nm=org_inv_nm
            )
            rows.append(row)

        # Write CSV
        if rows:
            fieldnames = ['path_id', 'sub_graph_json', 'org_inv_nm', 'batch', 'inference_date']
            csv_output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(csv_output_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)

        return {
            "success": True,
            "total_evidences": len(all_evidences),
            "merged_evidences": len(processed_evidences),
            "dropped_incomplete": dropped_count,
            "subgraph_count": len(rows),
            "org_inv_nm": org_inv_nm,
            "inference_date": inference_date
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# ========== Batch attribution execution ==========

def run_attribution_task(
    farm_id: str,
    reference_date: str,
    window_days: int,
    output_path: Path,
    run_script_path: Path,
    no_llm: bool = False
) -> dict:
    """
    Execute single attribution task

    Args:
        farm_id: Farm ID
        reference_date: Attribution date
        window_days: Lookback window in days
        output_path: Output JSON path
        run_script_path: run_attribution.py script path
        no_llm: Whether to disable LLM

    Returns:
        Execution result dict with status, elapsed_time, error etc.
    """
    cmd = [
        sys.executable,
        str(run_script_path),
        "--farm-id", farm_id,
        "--reference-date", reference_date,
        "--window-days", str(window_days),
        "--output", str(output_path)
    ]

    if no_llm:
        cmd.append("--no-llm")

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=600  # 10 minutes timeout
        )
        elapsed_time = time.time() - start_time

        stdout = result.stdout or ""
        stderr = result.stderr or ""

        if result.returncode == 0:
            return {
                "status": "success",
                "elapsed_time": elapsed_time,
                "returncode": result.returncode,
                "stdout": stdout[-1000:] if len(stdout) > 1000 else stdout,
                "stderr": stderr[-500:] if len(stderr) > 500 else stderr
            }
        else:
            error_msg = f"Process returned non-zero exit code: {result.returncode}"

            stderr_lines = stderr.strip().split('\n') if stderr else []
            if stderr_lines:
                key_errors = [line for line in stderr_lines if any(keyword in line for keyword in ['Error', 'Exception', 'Failed', 'Traceback'])]
                if key_errors:
                    error_msg += f"\nKey error: {key_errors[-1]}"

            return {
                "status": "failed",
                "elapsed_time": elapsed_time,
                "returncode": result.returncode,
                "stdout": stdout[-1000:] if len(stdout) > 1000 else stdout,
                "stderr": stderr[-1500:] if len(stderr) > 1500 else stderr,
                "error": error_msg
            }

    except subprocess.TimeoutExpired as e:
        elapsed_time = time.time() - start_time
        stdout = getattr(e, 'stdout', None) or ""
        stderr = getattr(e, 'stderr', None) or ""
        return {
            "status": "timeout",
            "elapsed_time": elapsed_time,
            "stdout": stdout,
            "stderr": stderr,
            "error": "Process timed out after 600 seconds"
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        import traceback
        error_traceback = traceback.format_exc()
        return {
            "status": "error",
            "elapsed_time": elapsed_time,
            "stdout": "",
            "stderr": error_traceback,
            "error": f"{type(e).__name__}: {str(e)}"
        }


def load_output_summary(output_path: Path) -> dict:
    """
    Load and parse attribution task output JSON, extract key statistics

    Returns:
        Dictionary with evidence_count, query_blocks and other metrics
    """
    try:
        if not output_path.exists():
            return {"error": "Output file not found"}

        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        summary = {
            "version": data.get("version"),
            "query_blocks": len(data.get("search_results", [])),
            "total_evidence": 0,
            "total_records": 0,
            "total_latency_ms": 0
        }

        search_results = data.get("search_results", [])
        for sr in search_results:
            summary["total_evidence"] += len(sr.get("evidence", []))
            metrics = sr.get("metrics", {})
            summary["total_records"] += metrics.get("records", 0)
            summary["total_latency_ms"] += metrics.get("latency_ms", 0)

        memory_stats = data.get("memory_stats", {})
        summary["completed_pairs"] = memory_stats.get("completed_pairs", 0)

        return summary

    except Exception as e:
        return {"error": f"Failed to parse output: {str(e)}"}


def convert_task_output(json_output_path: Path, csv_output_path: Path, filter_incomplete: bool = False) -> dict:
    """
    Convert attribution task output to subgraph CSV format

    Args:
        json_output_path: JSON output file path
        csv_output_path: CSV output file path
        filter_incomplete: Whether to filter out incomplete attribution chains (default: False)

    Returns:
        Conversion result statistics
    """
    try:
        conversion_result = convert_agent_output_to_subgraph_csv(json_output_path, csv_output_path, filter_incomplete)
        return conversion_result
    except Exception as e:
        return {
            "success": False,
            "error": f"Conversion failed: {str(e)}"
        }


def main():
    parser = argparse.ArgumentParser(description="Batch run attribution tasks")
    parser.add_argument(
        "--sample-data",
        default=r"d:\PycharmData\PRRS\PRRS-Alert-Diagnoser-Breeding-Farm\src\PRRS_Attribution_agent\test\dataset\result1_sample.csv",
        help="Path to sampled dataset CSV"
    )
    parser.add_argument(
        "--output-dir",
        default=r"d:\PycharmData\PRRS\PRRS-Alert-Diagnoser-Breeding-Farm\data\result",
        help="Directory to save attribution outputs"
    )
    parser.add_argument(
        "--run-script",
        default=r"d:\PycharmData\PRRS\PRRS-Alert-Diagnoser-Breeding-Farm\AAT-Agent\run_attribution.py",
        help="Path to run_attribution.py script"
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=60,
        help="Window days for attribution"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM calls for all tasks"
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Maximum number of tasks to run (for testing)"
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Start from task index (0-based, for resuming)"
    )
    parser.add_argument(
        "--skip-conversion",
        action="store_true",
        help="Skip subgraph CSV conversion (only run attribution)"
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        default=True,
        help="Skip tasks that already have CSV output (resume from incomplete, default: True)"
    )
    parser.add_argument(
        "--filter-incomplete",
        action="store_true",
        default=False,
        help="Filter out incomplete attribution chains (default: False, keep all chains)"
    )

    args = parser.parse_args()

    sys.stdout.flush()
    sys.stderr.flush()

    print("=" * 100, flush=True)
    print("AAT-Agent Batch Attribution Execution", flush=True)
    print("=" * 100, flush=True)
    print(f"Sample data: {args.sample_data}", flush=True)
    print(f"Output directory: {args.output_dir}", flush=True)
    print(f"Attribution script: {args.run_script}", flush=True)
    print(f"Lookback window: {args.window_days} days", flush=True)
    print(f"LLM status: {'Disabled' if args.no_llm else 'Enabled'}", flush=True)
    print(f"Filter incomplete chains: {'Yes' if args.filter_incomplete else 'No (keep all)'}", flush=True)
    print(flush=True)

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subgraph_csv_dir = output_dir / "subgraph_csvs"
    if not args.skip_conversion:
        subgraph_csv_dir.mkdir(parents=True, exist_ok=True)

    # Load sample data
    print("Loading sample data...", flush=True)
    sample_df = pd.read_csv(args.sample_data)
    print(f"Total tasks: {len(sample_df)}", flush=True)
    print(f"Farms: {sample_df['org_inv_dk'].nunique()}", flush=True)
    print(f"Date range: {sample_df['reference_date'].min()} to {sample_df['reference_date'].max()}", flush=True)
    print(flush=True)

    # Skip completed tasks if enabled
    if args.skip_completed:
        original_count = len(sample_df)
        completed_tasks = set()

        for csv_file in subgraph_csv_dir.rglob("*.csv"):
            csv_name = csv_file.stem
            parts = csv_name.rsplit('_', 1)
            if len(parts) == 2:
                parent_dir = csv_file.parent.name
                if parent_dir != "subgraph_csvs":
                    full_farm_id = f"{parent_dir}/{parts[0]}"
                    task_key = f"{full_farm_id}_{parts[1]}"
                else:
                    task_key = csv_name
                completed_tasks.add(task_key)

        def is_not_completed(row):
            task_key = f"{row['org_inv_dk']}_{row['reference_date']}"
            return task_key not in completed_tasks

        sample_df = sample_df[sample_df.apply(is_not_completed, axis=1)].reset_index(drop=True)
        skipped_count = original_count - len(sample_df)

        print(f"✅ Skipped completed tasks: {skipped_count}", flush=True)
        print(f"📋 Remaining tasks: {len(sample_df)}", flush=True)
        print(flush=True)

    # Apply start index and max tasks limit
    if args.start_from > 0:
        sample_df = sample_df.iloc[args.start_from:].reset_index(drop=True)
        print(f"Starting from task #{args.start_from}", flush=True)

    if args.max_tasks is not None:
        sample_df = sample_df.head(args.max_tasks)
        print(f"Limited to {len(sample_df)} tasks", flush=True)

    # Execute batch attribution
    results = []
    run_script_path = Path(args.run_script)

    batch_start_time = time.time()
    print("\n" + "=" * 100, flush=True)
    print("Starting batch attribution execution", flush=True)
    print("=" * 100, flush=True)

    for idx, row in sample_df.iterrows():
        task_idx = args.start_from + idx
        farm_id = row['org_inv_dk']
        org_inv_nm = row['org_inv_nm']
        reference_date = row['reference_date']
        abort_rate = row['abort_rate']
        quartile = row.get('quartile', 'N/A')

        output_filename = f"{farm_id}_{reference_date}.json"
        output_path = output_dir / output_filename

        print(f"\n[{task_idx + 1}/{args.start_from + len(sample_df)}] Executing attribution task")
        print(f"  Farm: {org_inv_nm} ({farm_id})")
        print(f"  Attribution date: {reference_date}")
        print(f"  Abort rate: {abort_rate:.4f}")
        print(f"  Quartile: {quartile}")
        print(f"  Output file: {output_filename}")

        # Execute attribution task
        task_result = run_attribution_task(
            farm_id=farm_id,
            reference_date=reference_date,
            window_days=args.window_days,
            output_path=output_path,
            run_script_path=run_script_path,
            no_llm=args.no_llm
        )

        attribution_summary = {}
        conversion_summary = {}

        if task_result["status"] == "success":
            attribution_summary = load_output_summary(output_path)

            # Convert to subgraph CSV
            if not args.skip_conversion:
                csv_filename = f"{farm_id}_{reference_date}.csv"
                csv_output_path = subgraph_csv_dir / csv_filename

                print(f"  🔄 Converting to subgraph CSV: {csv_filename}")
                conversion_summary = convert_task_output(output_path, csv_output_path, args.filter_incomplete)

                if conversion_summary.get("success"):
                    print(f"  ✅ Conversion successful: {conversion_summary.get('subgraph_count', 0)} subgraphs")
                else:
                    print(f"  ⚠️  Conversion failed: {conversion_summary.get('error', 'Unknown error')}")

        # Record result
        result_record = {
            "task_idx": task_idx,
            "farm_id": farm_id,
            "org_inv_nm": org_inv_nm,
            "reference_date": reference_date,
            "abort_rate": abort_rate,
            "quartile": quartile,
            "status": task_result["status"],
            "elapsed_time": task_result["elapsed_time"],
            "output_file": output_filename,
            **attribution_summary
        }

        if conversion_summary:
            result_record["subgraph_count"] = conversion_summary.get("subgraph_count", 0)
            result_record["conversion_success"] = conversion_summary.get("success", False)

        if task_result["status"] != "success":
            result_record["error"] = task_result.get("error", "Unknown error")

            error_log_path = output_dir / f"error_{farm_id}_{reference_date}.log"
            try:
                with open(error_log_path, 'w', encoding='utf-8') as f:
                    f.write(f"Task: {org_inv_nm} ({farm_id})\n")
                    f.write(f"Reference Date: {reference_date}\n")
                    f.write(f"Abort Rate: {abort_rate}\n")
                    f.write(f"Status: {task_result['status']}\n")
                    f.write(f"Return Code: {task_result.get('returncode', 'N/A')}\n")
                    f.write(f"Error: {task_result.get('error', 'Unknown')}\n")
                    f.write("\n" + "="*80 + "\n")
                    f.write("STDOUT:\n")
                    f.write(task_result.get('stdout', '(empty)'))
                    f.write("\n" + "="*80 + "\n")
                    f.write("STDERR:\n")
                    f.write(task_result.get('stderr', '(empty)'))
                result_record["error_log_file"] = error_log_path.name
            except Exception as e:
                print(f"  ⚠️  Failed to save error log: {str(e)}")

        results.append(result_record)

        # Print execution result
        status_emoji = "✅" if task_result["status"] == "success" else "❌"
        print(f"  {status_emoji} Status: {task_result['status']} (time: {task_result['elapsed_time']:.2f}s)")

        if task_result["status"] == "success" and attribution_summary:
            print(f"  📊 Evidence: {attribution_summary.get('total_evidence', 0)}, "
                  f"Query blocks: {attribution_summary.get('query_blocks', 0)}, "
                  f"Records: {attribution_summary.get('total_records', 0)}")

        if task_result["status"] != "success":
            error_msg = task_result.get('error', 'Unknown error')
            print(f"  ⚠️  Error: {error_msg}")

            stderr = task_result.get('stderr', '')
            if stderr and len(stderr) > 50:
                stderr_lines = stderr.strip().split('\n')
                last_lines = stderr_lines[-3:]
                print(f"  💬 Error details:")
                for line in last_lines:
                    if line.strip():
                        print(f"     {line.strip()[:120]}")

    batch_elapsed_time = time.time() - batch_start_time

    # Save execution summary
    results_df = pd.DataFrame(results)
    summary_path = output_dir / "batch_execution_summary.csv"
    results_df.to_csv(summary_path, index=False, encoding='utf-8-sig')

    # Generate statistics report
    print("\n" + "=" * 100)
    print("Batch execution completed")
    print("=" * 100)
    print(f"Total tasks: {len(results_df)}")
    print(f"Total time: {batch_elapsed_time:.2f}s ({batch_elapsed_time/60:.2f}min)")
    print(f"Average time: {batch_elapsed_time/len(results_df):.2f}s/task")
    print()

    status_counts = results_df['status'].value_counts()
    print("Execution status:")
    for status, count in status_counts.items():
        percentage = (count / len(results_df)) * 100
        print(f"  {status}: {count} ({percentage:.1f}%)")
    print()

    # Statistics for successful tasks
    success_df = results_df[results_df['status'] == 'success']
    if not success_df.empty:
        print("Successful task statistics:")
        print(f"  Total evidence: {success_df['total_evidence'].sum()}")
        print(f"  Average evidence: {success_df['total_evidence'].mean():.2f}")
        print(f"  Average query blocks: {success_df['query_blocks'].mean():.2f}")
        print(f"  Average records: {success_df['total_records'].mean():.2f}")
        print()

        if not args.skip_conversion and 'subgraph_count' in success_df.columns:
            print("Subgraph conversion statistics:")
            print(f"  Total subgraphs: {success_df['subgraph_count'].sum()}")
            print(f"  Average subgraphs/task: {success_df['subgraph_count'].mean():.2f}")
            conversion_success = success_df['conversion_success'].sum() if 'conversion_success' in success_df.columns else 0
            print(f"  Conversion success rate: {conversion_success / len(success_df) * 100:.1f}%")
            print()

    # Statistics by quartile
    if 'quartile' in results_df.columns:
        print("Statistics by quartile:")
        for quartile in results_df['quartile'].unique():
            quartile_df = results_df[results_df['quartile'] == quartile]
            success_count = len(quartile_df[quartile_df['status'] == 'success'])
            success_rate = (success_count / len(quartile_df)) * 100
            print(f"  {quartile}: {len(quartile_df)} tasks, success rate {success_rate:.1f}%")
        print()

    print(f"Execution summary saved to: {summary_path}")

    if not args.skip_conversion:
        print(f"Subgraph CSV files saved to: {subgraph_csv_dir}")
        print(f"\nVisualization usage:")
        print(f"  1. Open visual.html")
        print(f"  2. Select CSV file from {subgraph_csv_dir}")
        print(f"  3. Select subgraph from dropdown for visualization")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
