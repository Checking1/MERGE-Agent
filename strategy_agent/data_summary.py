import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from utils import serialize


class DataSummary:
    """
    Summarize event/risk counts within a time window for a given farm.

    Data paths are configured via environment variables:
    - AAT_INTRO_EVENT_PATH: IntroEvent dataset path
    - AAT_GROUP_EVENT_PATH: GroupEvent dataset path
    - AAT_BREED_EVENT_PATH: BreedEvent dataset path
    - AAT_DELIVERY_EVENT_PATH: DeliveryEvent dataset path
    - AAT_NORMAL_IMMU_EVENT_PATH: NormalImmuEvent dataset path
    - AAT_ABORT_EVENT_PATH: AbortEvent dataset path
    - AAT_RISK_EVENT_PATH: RiskEvent dataset path
    - AAT_SECURITY_EVENT_PATH: SecurityEvent dataset path (optional)
    """

    def __init__(self):
        # 获取项目根目录（AAT-Agent的父目录）
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        project_root = os.path.dirname(current_dir)
        data_base = os.path.join(project_root, "data", "interim", "PRRS_Risk_Attribution")
        
        self.paths = {
            "intro": os.getenv("AAT_INTRO_EVENT_PATH", os.path.join(data_base, "intro_event_dataset.csv")),
            "group": os.getenv("AAT_GROUP_EVENT_PATH", os.path.join(data_base, "group_event_dataset.csv")),
            "breed": os.getenv("AAT_BREED_EVENT_PATH", os.path.join(data_base, "breed_event_dataset.csv")),
            "delivery": os.getenv("AAT_DELIVERY_EVENT_PATH", os.path.join(data_base, "delivery_event_dataset.csv")),
            "normal_immu": os.getenv("AAT_NORMAL_IMMU_EVENT_PATH", os.path.join(data_base, "normal_immu_event_dataset.csv")),
            "abort": os.getenv("AAT_ABORT_EVENT_PATH", os.path.join(data_base, "abort_event_dataset.csv")),
            "risk": os.getenv("AAT_RISK_EVENT_PATH", os.path.join(data_base, "risk_event_dataset.csv")),
            "security": os.getenv("AAT_SECURITY_EVENT_PATH", os.path.join(data_base, "security_event_dataset.csv")),
        }
        # dataset -> preferred date columns (ordered)
        self.date_cols: Dict[str, List[str]] = {
            "intro": ["allot_dt", "begin_date", "end_date"],
            "group": ["min_boar_inpop_dt", "stats_dt", "begin_date", "end_date"],
            "breed": ["sow_dt", "min_boar_inpop_dt", "stats_dt", "begin_date", "end_date"],
            "delivery": ["stats_dt", "begin_date", "end_date"],
            "normal_immu": ["stats_dt"],
            "abort": ["stats_dt", "begin_date", "end_date"],
            "risk": ["risk_event_occur_dt", "inference_date", "stats_dt"],
            "security": ["security_event_occur_dt", "inference_date", "stats_dt"],
        }

    def _load_df(self, path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            return pd.DataFrame()
        df = serialize.dataframe_read(path, file_type="csv")
        if df is None:
            return pd.DataFrame()
        return df.reset_index(drop=True)

    def _filter_window(
        self,
        df: pd.DataFrame,
        org_inv_dk: Optional[str],
        reference_date: str,
        window_days: int,
        date_col_candidates: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if df.empty:
            return df
        ref = datetime.strptime(reference_date, "%Y-%m-%d").date()
        start = ref - timedelta(days=window_days)
        date_col = None
        for c in date_col_candidates or []:
            if c in df.columns:
                date_col = c
                break
        if date_col is None:
            return pd.DataFrame()
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date
        mask = (df[date_col] >= start) & (df[date_col] <= ref)
        if org_inv_dk and "org_inv_dk" in df.columns:
            mask &= df["org_inv_dk"] == org_inv_dk
        return df.loc[mask]

    def summarize(
        self,
        org_inv_dk: Optional[str],
        reference_date: str,
        window_days: int,
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {"window_days": window_days, "reference_date": reference_date}

        def count_with_risk(event_key: str, risk_link: Optional[str] = None):
            df = self._load_df(self.paths[event_key])
            df = self._filter_window(df, org_inv_dk, reference_date, window_days, self.date_cols.get(event_key, []))
            count = len(df)
            risk_count = 0
            if risk_link:
                risk_df = self._load_df(self.paths["risk"])
                risk_df = self._filter_window(risk_df, org_inv_dk, reference_date, window_days, self.date_cols.get("risk", []))
                if not risk_df.empty:
                    risk_df = risk_df[risk_df.get("link") == risk_link] if "link" in risk_df.columns else risk_df
                    if "event_id" in risk_df.columns:
                        risk_count = risk_df["event_id"].nunique()
            return count, risk_count

        def count_with_risk_by_event_id(event_key: str):
            df = self._load_df(self.paths[event_key])
            df = self._filter_window(df, org_inv_dk, reference_date, window_days, self.date_cols.get(event_key, []))
            count = len(df)
            risk_count = 0
            if count and "event_id" in df.columns:
                event_ids = df["event_id"].dropna().unique().tolist()
                if event_ids:
                    risk_df = self._load_df(self.paths["risk"])
                    risk_df = self._filter_window(risk_df, org_inv_dk, reference_date, window_days, self.date_cols.get("risk", []))
                    if not risk_df.empty and "event_id" in risk_df.columns:
                        risk_df = risk_df[risk_df["event_id"].isin(event_ids)]
                        risk_count = risk_df["event_id"].nunique()
            return count, risk_count

        intro_c, intro_r = count_with_risk("intro", risk_link="引种")
        group_c, group_r = count_with_risk("group", risk_link="入群")
        breed_c, breed_r = count_with_risk("breed", risk_link="配种")
        delivery_c, delivery_r = count_with_risk("delivery", risk_link="分娩")
        immu_c, immu_r = count_with_risk_by_event_id("normal_immu")

        abort_df = self._load_df(self.paths["abort"])
        abort_df = self._filter_window(abort_df, org_inv_dk, reference_date, window_days, self.date_cols.get("abort", []))
        abort_c = len(abort_df)

        summary["intro"] = {"events": intro_c, "risk_events": intro_r}
        summary["group"] = {"events": group_c, "risk_events": group_r}
        summary["breed"] = {"events": breed_c, "risk_events": breed_r}
        summary["delivery"] = {"events": delivery_c, "risk_events": delivery_r}
        summary["normal_immu"] = {"events": immu_c, "risk_events": immu_r}
        summary["abort"] = {"events": abort_c}

        return summary

    def format_summary_text(self, summary: Dict[str, Any]) -> str:
        intro = summary.get("intro", {})
        group = summary.get("group", {})
        breed = summary.get("breed", {})
        delivery = summary.get("delivery", {})
        normal_immu = summary.get("normal_immu", {})
        abort_c = summary.get("abort", {}).get("events", 0)
        return (
            f"窗口内{summary.get('window_days')}天，参照日{summary.get('reference_date')}："
            f"引种事件{intro.get('events',0)}次，其中伴随风险{intro.get('risk_events',0)}次；"
            f"入群事件{group.get('events',0)}次，其中伴随风险{group.get('risk_events',0)}次；"
            f"配种事件{breed.get('events',0)}次，其中伴随风险{breed.get('risk_events',0)}次；"
            f"产房管理事件{delivery.get('events',0)}次，其中伴随风险{delivery.get('risk_events',0)}次；"
            f"常规免疫事件{normal_immu.get('events',0)}次，其中伴随风险{normal_immu.get('risk_events',0)}次；"
            f"检测到流产事件{abort_c}次。"
        )
