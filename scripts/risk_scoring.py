"""
risk_scoring.py — Rule-Based Risk Scoring Engine for Aviation Insider Threats

Loads synthetic SOC logs, computes per-event risk weights, aggregates scores
per employee per day, applies compound-event bonuses and decay, then classifies
alert levels and produces output files.

Usage:
    python risk_scoring.py
"""

from __future__ import annotations

import json
import os
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("synthetic_logs")
OUT_DIR = Path("risk_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore", category=UserWarning)

# ── Risk weights (from PDF specification) ────────────────────────────────────
WEIGHTS = {
    "off_hours_badge": 3,
    "off_hours_sensitive_door": 5,  # cumulative with above (total 8)
    "unusual_country": 4,
    "brute_force_vpn": 3,
    "bulk_download": 10,
    "unauth_security_access": 6,
    "maintenance_tamper": 12,
    "cargo_weight_mismatch": 6,
    "off_hours_cargo_door": 3,
    "compound_bonus": 6,
}

HIGH_RISK_COUNTRIES = {"RU", "CN", "IR", "KP", "SY"}
SENSITIVE_DOORS = {"CargoArea", "DataCenter", "MaintenanceHangar"}
SECURITY_ROLES = {"SecurityOfficer", "ITAdmin", "Pilot"}
CRITICAL_COMPONENTS = {"Engine", "FuelSystem", "FlightControls"}
TAMPER_ACTIONS = {"override_sensor", "disable_alarm", "edit_log"}
OFF_HOURS = {22, 23, 0, 1, 2, 3, 4, 5}

# ══════════════════════════════════════════════════════════════════════════════
#  1. Load & preprocess
# ══════════════════════════════════════════════════════════════════════════════

def load_and_preprocess() -> tuple[
    pd.DataFrame,  # employee_table
    pd.DataFrame,  # badge_logs enriched
    pd.DataFrame,  # vpn_logs enriched
    pd.DataFrame,  # crew_portal_logs enriched
    pd.DataFrame,  # maintenance_logs enriched
    pd.DataFrame,  # cargo_logs enriched
]:
    """Load all CSVs, parse timestamps, merge with employee_table for context."""
    print("=" * 60)
    print("  Loading and preprocessing data …")
    print("=" * 60)

    emp = pd.read_csv(DATA_DIR / "employee_table.csv")
    emp["termination_date"] = pd.to_datetime(emp["termination_date"], errors="coerce")

    # ── Badge ────────────────────────────────────────────────────────────────
    badge = pd.read_csv(DATA_DIR / "badge_logs.csv")
    badge["timestamp"] = pd.to_datetime(badge["timestamp"])
    badge["hour_of_day"] = badge["timestamp"].dt.hour
    badge = badge.merge(emp[["employee_id", "shift_start", "shift_end", "typical_access_zones"]],
                        on="employee_id", how="left")
    # Derive shift_scheduled
    badge["shift_scheduled"] = badge.apply(_shift_scheduled, axis=1)
    print(f"  badge_logs: {len(badge):,} rows loaded")

    # ── VPN ──────────────────────────────────────────────────────────────────
    vpn = pd.read_csv(DATA_DIR / "vpn_logs.csv")
    vpn["timestamp"] = pd.to_datetime(vpn["timestamp"])
    vpn["hour_of_day"] = vpn["timestamp"].dt.hour
    vpn = vpn.merge(emp[["employee_id", "typical_country", "shift_start", "shift_end"]],
                    left_on="username", right_on="employee_id", how="left")
    vpn["shift_scheduled"] = vpn.apply(_shift_scheduled, axis=1)
    print(f"  vpn_logs: {len(vpn):,} rows loaded")

    # ── Crew portal ──────────────────────────────────────────────────────────
    cp = pd.read_csv(DATA_DIR / "crew_portal_logs.csv")
    cp["timestamp"] = pd.to_datetime(cp["timestamp"])
    cp["hour_of_day"] = cp["timestamp"].dt.hour
    cp = cp.merge(emp[["employee_id", "shift_start", "shift_end"]],
                  left_on="user", right_on="employee_id", how="left")
    cp["shift_scheduled"] = cp.apply(_shift_scheduled, axis=1)
    print(f"  crew_portal_logs: {len(cp):,} rows loaded")

    # ── Maintenance ──────────────────────────────────────────────────────────
    maint = pd.read_csv(DATA_DIR / "maintenance_logs.csv")
    maint["timestamp"] = pd.to_datetime(maint["timestamp"])
    maint["hour_of_day"] = maint["timestamp"].dt.hour
    print(f"  maintenance_logs: {len(maint):,} rows loaded")

    # ── Cargo ────────────────────────────────────────────────────────────────
    cargo = pd.read_csv(DATA_DIR / "cargo_logs.csv")
    cargo["timestamp"] = pd.to_datetime(cargo["timestamp"])
    cargo["hour_of_day"] = cargo["timestamp"].dt.hour
    cargo = cargo.merge(emp[["employee_id", "shift_start", "shift_end"]],
                        left_on="handler_id", right_on="employee_id", how="left")
    cargo["shift_scheduled"] = cargo.apply(_shift_scheduled, axis=1)
    # Sort for weight-delta computation
    cargo = cargo.sort_values(["cargo_id", "timestamp"]).reset_index(drop=True)
    print(f"  cargo_logs: {len(cargo):,} rows loaded")

    print(f"  employee_table: {len(emp):,} employees\n")
    return emp, badge, vpn, cp, maint, cargo


def _shift_scheduled(row: pd.Series) -> bool:
    """Determine if the event hour falls within the employee's shift window."""
    if pd.isna(row.get("shift_start")) or pd.isna(row.get("shift_end")):
        return True  # default safe
    h = int(row["hour_of_day"])
    sh = int(row["shift_start"])
    se = int(row["shift_end"])
    if sh <= se:
        return not (h < sh or h >= se)
    else:  # overnight shift e.g. 22:00-06:00
        return not (se <= h < sh)


# ══════════════════════════════════════════════════════════════════════════════
#  2. Event-level weight computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_badge_weights(badge: pd.DataFrame) -> pd.DataFrame:
    """Badge rules: off-hours access, off-hours + sensitive door."""
    events: list[dict] = []
    for _, row in badge.iterrows():
        weight = 0
        types = []
        hr = row["hour_of_day"]
        off = hr in OFF_HOURS
        scheduled = row["shift_scheduled"]

        if off and not scheduled:
            weight += WEIGHTS["off_hours_badge"]
            types.append("off_hours_badge")

        if off and not scheduled and row["door_location"] in SENSITIVE_DOORS:
            weight += WEIGHTS["off_hours_sensitive_door"]
            types.append("off_hours_sensitive_door")

        if weight > 0:
            events.append({
                "timestamp": row["timestamp"],
                "employee_id": row["employee_id"],
                "role": row["role"],
                "weight": weight,
                "event_type": "+".join(types),
                "details": f"badge|{row['door_location']}|hour={hr}|scheduled={scheduled}",
            })
    return pd.DataFrame(events)


def compute_vpn_weights(vpn: pd.DataFrame) -> pd.DataFrame:
    """VPN rules: unusual/high-risk country, brute-force clusters."""
    events: list[dict] = []

    # ── Unusual country ──────────────────────────────────────────────────
    for _, row in vpn.iterrows():
        weight = 0
        types = []
        country = row["geo_country"]
        typical = row.get("typical_country")

        if country in HIGH_RISK_COUNTRIES or (typical is not None and country != typical):
            weight += WEIGHTS["unusual_country"]
            types.append("unusual_country")

        if weight > 0:
            events.append({
                "timestamp": row["timestamp"],
                "employee_id": row["username"],
                "role": row["role"],
                "weight": weight,
                "event_type": "+".join(types),
                "details": f"vpn|country={country}|typical={typical}",
            })

    # ── Brute-force: >3 failures in 10 min per user-IP ───────────────────
    vpn_sorted = vpn.sort_values(["username", "src_ip", "timestamp"]).reset_index(drop=True)
    vpn_sorted["next_ts"] = vpn_sorted.groupby(["username", "src_ip"])["timestamp"].shift(-1)
    vpn_sorted["fail_window"] = (
        vpn_sorted["next_ts"] - vpn_sorted["timestamp"]
    ).dt.total_seconds().le(600) & (vpn_sorted["auth_result"] == "failure")

    # Group consecutive failures within 10 minutes
    group_key = (vpn_sorted["fail_window"] != vpn_sorted.groupby(["username", "src_ip"])["fail_window"].shift()).cumsum()
    for (uname, src_ip), grp in vpn_sorted.groupby(["username", "src_ip"]):
        fails = grp[grp["auth_result"] == "failure"]
        if len(fails) >= 4:
            # Check they fall within a 10-minute span
            span = (fails["timestamp"].max() - fails["timestamp"].min()).total_seconds() / 60
            if span <= 10:
                events.append({
                    "timestamp": fails["timestamp"].iloc[0],
                    "employee_id": uname,
                    "role": fails["role"].iloc[0],
                    "weight": WEIGHTS["brute_force_vpn"],
                    "event_type": "brute_force_vpn",
                    "details": f"vpn_brute|{len(fails)} failures|ip={src_ip}|span={span:.1f}min",
                })
    return pd.DataFrame(events)


def compute_crew_portal_weights(cp: pd.DataFrame) -> pd.DataFrame:
    """Crew portal rules: bulk download, unauthorised SecurityProtocol access."""
    events: list[dict] = []
    for _, row in cp.iterrows():
        weight = 0
        types = []
        if row["action"] == "download" and (
            row["record_count"] > 100 or row["bytes_transferred"] > 500_000_000
        ):
            weight += WEIGHTS["bulk_download"]
            types.append("bulk_download")

        if row["record_type"] == "SecurityProtocol" and row["role"] not in SECURITY_ROLES:
            weight += WEIGHTS["unauth_security_access"]
            types.append("unauth_security_access")

        if weight > 0:
            events.append({
                "timestamp": row["timestamp"],
                "employee_id": row["user"],
                "role": row["role"],
                "weight": weight,
                "event_type": "+".join(types),
                "details": (
                    f"crew|action={row['action']}|type={row['record_type']}|"
                    f"count={row['record_count']}|bytes={row['bytes_transferred']}"
                ),
            })
    return pd.DataFrame(events)


def compute_maintenance_weights(maint: pd.DataFrame) -> pd.DataFrame:
    """Maintenance rule: tampering with critical components (only without valid work order)."""
    events: list[dict] = []
    for _, row in maint.iterrows():
        work_valid = bool(row.get("work_order_valid", True))
        if (
            row["action"] in TAMPER_ACTIONS
            and row["component"] in CRITICAL_COMPONENTS
            and not work_valid
        ):
            events.append({
                "timestamp": row["timestamp"],
                "employee_id": row["technician_id"],
                "role": row["role"],
                "weight": WEIGHTS["maintenance_tamper"],
                "event_type": "maintenance_tamper",
                "details": (
                    f"maint|action={row['action']}|component={row['component']}|"
                    f"work_order={row.get('work_order_valid', 'N/A')}|"
                    f"source={row.get('source', 'N/A')}"
                ),
            })
    return pd.DataFrame(events)


def compute_cargo_weights(cargo: pd.DataFrame) -> pd.DataFrame:
    """Cargo rules: weight mismatch, off-hours door open."""
    events: list[dict] = []

    # ── Weight mismatch (delta > 20 kg on same cargo_id within 1 hour) ───
    cargo["prev_weight"] = cargo.groupby("cargo_id")["weight_kg"].shift(1)
    cargo["prev_ts"] = cargo.groupby("cargo_id")["timestamp"].shift(1)
    cargo["delta"] = cargo["prev_weight"] - cargo["weight_kg"]
    cargo["time_since_prev"] = (
        (cargo["timestamp"] - cargo["prev_ts"]).dt.total_seconds() / 3600
    )

    mismatch = (
        (cargo["action"] == "edit_manifest")
        & (cargo["delta"] > 20)
        & (cargo["time_since_prev"] <= 1)
    )
    for _, row in cargo[mismatch].iterrows():
        events.append({
            "timestamp": row["timestamp"],
            "employee_id": row["handler_id"],
            "role": row["role"],
            "weight": WEIGHTS["cargo_weight_mismatch"],
            "event_type": "cargo_weight_mismatch",
            "details": (
                f"cargo|id={row['cargo_id']}|prev_weight={row['prev_weight']}|"
                f"new_weight={row['weight_kg']}|delta={row['delta']:.1f}"
            ),
        })

    # ── Off-hours cargo door open by unscheduled handler ─────────────────
    off_door = (
        (cargo["action"] == "open_door")
        & (cargo["hour_of_day"].isin(OFF_HOURS))
        & (~cargo["shift_scheduled"])
    )
    for _, row in cargo[off_door].iterrows():
        events.append({
            "timestamp": row["timestamp"],
            "employee_id": row["handler_id"],
            "role": row["role"],
            "weight": WEIGHTS["off_hours_cargo_door"],
            "event_type": "off_hours_cargo_door",
            "details": (
                f"cargo|door_open|location={row['location']}|"
                f"hour={row['hour_of_day']}|scheduled={row['shift_scheduled']}"
            ),
        })

    return pd.DataFrame(events)


# ══════════════════════════════════════════════════════════════════════════════
#  3. Aggregate & compound bonus
# ══════════════════════════════════════════════════════════════════════════════

def apply_compound_bonus(events: pd.DataFrame) -> pd.DataFrame:
    """Add +20 bonus when an employee triggers ≥2 distinct event types within 1 h."""
    if events.empty:
        return events

    events = events.sort_values(["employee_id", "timestamp"]).reset_index(drop=True)
    new_rows: list[dict] = []

    for eid, grp in events.groupby("employee_id"):
        grp = grp.sort_values("timestamp")
        windows: list[list] = []
        current: list = []

        for _, row in grp.iterrows():
            if not current:
                current = [dict(row)]
                continue
            # If within 1 hour of current window start
            if row["timestamp"] - current[0]["timestamp"] <= timedelta(hours=1):
                current.append(dict(row))
            else:
                windows.append(current)
                current = [dict(row)]
        if current:
            windows.append(current)

        for win in windows:
            if len(win) < 2:
                continue
            types = set(e["event_type"] for e in win)
            if len(types) >= 2:
                new_rows.append({
                    "timestamp": win[0]["timestamp"],
                    "employee_id": eid,
                    "role": win[0]["role"],
                    "weight": WEIGHTS["compound_bonus"],
                    "event_type": "compound_bonus",
                    "details": f"compound|types={'+'.join(sorted(types))}|events_in_window={len(win)}",
                })

    if new_rows:
        bonus_df = pd.DataFrame(new_rows)
        events = pd.concat([events, bonus_df], ignore_index=True)
    return events.sort_values(["employee_id", "timestamp"]).reset_index(drop=True)


def apply_decay(
    daily: pd.DataFrame,  # columns: employee_id, date, daily_raw_score
    half_life_days: int = 3,
) -> pd.DataFrame:
    """Apply exponential decay to scores during gaps with no events.
    
    Reads the raw daily score from `daily_raw_score` and writes the decayed
    cumulative score into `total_score`.  `daily_raw_score` is preserved
    for alert classification.
    """
    if daily.empty:
        return daily
    if "daily_raw_score" not in daily.columns:
        daily["daily_raw_score"] = daily["total_score"]
    daily = daily.sort_values(["employee_id", "date"]).reset_index(drop=True)
    daily["total_score"] = 0.0

    for eid, grp in daily.groupby("employee_id"):
        grp = grp.sort_values("date").copy()
        decayed = 0.0
        prev_date = None
        for idx, row in grp.iterrows():
            if prev_date is not None:
                gap_days = (row["date"] - prev_date).days
                decay_factor = 0.5 ** (gap_days / half_life_days)
                decayed *= decay_factor
            decayed += row["daily_raw_score"]
            daily.at[idx, "total_score"] = decayed
            prev_date = row["date"]

    return daily


def aggregate_scores(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per employee per day, apply compound bonus & decay.
    
    Returns daily DataFrame with both daily_raw_score (pre-decay) and
    total_score (post-decay cumulative).
    """
    if events.empty:
        return pd.DataFrame(columns=["employee_id", "role", "date", "total_score", "alert_level"])

    events["date"] = events["timestamp"].dt.date
    daily = (
        events.groupby(["employee_id", "role", "date"])["weight"]
        .sum()
        .reset_index()
        .rename(columns={"weight": "daily_raw_score"})
    )

    daily["total_score"] = daily["daily_raw_score"]
    daily = apply_decay(daily)
    # daily_raw_score is the pre-decay raw score for the day (used for alert classification)
    # total_score is the decayed cumulative score (used for trending / visualisation)
    return daily


# ══════════════════════════════════════════════════════════════════════════════
#  4. Alert classification
# ══════════════════════════════════════════════════════════════════════════════

def classify_alerts(
    daily: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Add alert_level based on the raw daily score (pre-decay) and build
    contributing_events summary.  total_score (post-decay cumulative) is
    kept for trending."""
    if daily.empty:
        return daily

    def level(score):
        if score < 30:
            return "low"
        elif score < 60:
            return "medium"
        return "high"

    # Classify based on raw daily score so thresholds (30/60) are meaningful
    score_col = "daily_raw_score" if "daily_raw_score" in daily.columns else "total_score"
    daily["alert_level"] = daily[score_col].apply(level)

    # Attach contributing event descriptions
    contrib = (
        events.groupby(["employee_id", events["timestamp"].dt.date])
        .agg({
            "event_type": lambda x: "; ".join(sorted(set(x))),
            "details": lambda x: " | ".join(x),
        })
        .reset_index()
        .rename(columns={"timestamp": "date"})
    )
    daily = daily.merge(contrib, on=["employee_id", "date"], how="left")

    return daily.sort_values("total_score", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
#  5. Output
# ══════════════════════════════════════════════════════════════════════════════

def generate_outputs(
    daily: pd.DataFrame,
    events: pd.DataFrame,
    log_counts: dict[str, int],
) -> None:
    """Write CSV/JSON outputs and print console summary."""
    # ── risk_summary.csv ─────────────────────────────────────────────────────
    summary_path = DATA_DIR / "risk_summary.csv"
    daily.to_csv(summary_path, index=False)
    print(f"  -> {summary_path}")

    # ── alerts_medium_high.csv ───────────────────────────────────────────────
    med_high = daily[daily["alert_level"].isin(["medium", "high"])].copy()
    alerts_path = DATA_DIR / "alerts_medium_high.csv"
    med_high.to_csv(alerts_path, index=False)
    print(f"  -> {alerts_path}")

    # ── high_risk_employees.json ─────────────────────────────────────────────
    high = daily[daily["alert_level"] == "high"]
    high_json: list[dict] = []
    for _, row in high.iterrows():
        emp_events = events[
            (events["employee_id"] == row["employee_id"])
            & (events["timestamp"].dt.date == row["date"])
        ]
        high_json.append({
            "employee_id": row["employee_id"],
            "role": row["role"],
            "date": str(row["date"]),
            "total_score": int(row["total_score"]),
            "alert_level": row["alert_level"],
            "event_types": list(emp_events["event_type"].unique()),
            "event_details": emp_events["details"].tolist(),
        })

    json_path = DATA_DIR / "high_risk_employees.json"
    with open(json_path, "w") as fh:
        json.dump(high_json, fh, indent=2)
    print(f"  -> {json_path}")

    # ── Console summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RISK SCORING SUMMARY")
    print("=" * 60)
    for name, cnt in log_counts.items():
        print(f"  {name:30s} {cnt:>8,}")

    n_med = len(daily[daily["alert_level"] == "medium"]["employee_id"].unique())
    n_high = len(daily[daily["alert_level"] == "high"]["employee_id"].unique())
    print(f"\n  Employees with MEDIUM risk: {n_med}")
    print(f"  Employees with HIGH risk:  {n_high}")

    latest = daily.sort_values("date").groupby("employee_id").last().reset_index()
    top3 = latest.nlargest(3, "total_score")
    print(f"\n  Top 3 highest cumulative risk scores (final):")
    for _, row in top3.iterrows():
        print(f"    {row['employee_id']} ({row['role']})  —  {row['total_score']:.0f} pts  [{row['alert_level']}]")

    total_events = len(events)
    total_weight = events["weight"].sum()
    print(f"\n  Total weighted events: {total_events:,}")
    print(f"  Cumulative risk weight: {total_weight:,}")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
#  6. Visualizations
# ══════════════════════════════════════════════════════════════════════════════

def visualize(daily: pd.DataFrame, events: pd.DataFrame) -> None:
    """Generate and save three plots to risk_output/."""
    print("\n  Generating visualizations …")

    # ── Bar: Top 10 employees by final cumulative risk score ─────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    latest = daily.sort_values("date").groupby("employee_id").last()
    top10 = (
        latest["total_score"]
        .sort_values(ascending=False)
        .head(10)
    )
    colors = plt.cm.Reds(0.3 + 0.7 * (top10.values / top10.values.max()))
    ax.barh(top10.index, top10.values, color=colors)
    ax.set_xlabel("Cumulative Risk Score")
    ax.set_title("Top 10 Employees by Total Risk Score")
    ax.invert_yaxis()
    for bar, v in zip(ax.containers[0], top10.values):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                f"{v:.0f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "top10_risk_scores.png")
    plt.close(fig)
    print("  -> top10_risk_scores.png")

    # ── Timeline: highest-risk employee ─────────────────────────────────────
    if not daily.empty:
        top_emp = latest["total_score"].idxmax()
        emp_daily = daily[daily["employee_id"] == top_emp].sort_values("date")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(
            [str(d) for d in emp_daily["date"]],
            emp_daily["total_score"],
            marker="o", color="#e74c3c", linewidth=1.5,
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Daily Risk Score")
        ax.set_title(f"Risk Score Timeline – {top_emp}")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"timeline_{top_emp}.png")
        plt.close(fig)
        print(f"  -> timeline_{top_emp}.png")

    # ── Pie: Alert level distribution ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    counts = daily["alert_level"].value_counts()
    colors_pie = {"low": "#2ecc71", "medium": "#f39c12", "high": "#e74c3c"}
    ax.pie(
        counts.values,
        labels=[f"{k}\n({v})" for k, v in counts.items()],
        colors=[colors_pie.get(k, "#95a5a6") for k in counts.index],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title("Alert Level Distribution (employee-days)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "alert_level_pie.png")
    plt.close(fig)
    print("  -> alert_level_pie.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # 1. Load
    emp, badge, vpn, cp, maint, cargo = load_and_preprocess()

    log_counts = {
        "badge_logs": len(badge),
        "vpn_logs": len(vpn),
        "crew_portal_logs": len(cp),
        "maintenance_logs": len(maint),
        "cargo_logs": len(cargo),
    }

    # 2. Compute event-level weights
    print("=" * 60)
    print("  Computing risk weights per event …")
    print("=" * 60)

    badge_events = compute_badge_weights(badge)
    print(f"  badge weighted events: {len(badge_events)}")

    vpn_events = compute_vpn_weights(vpn)
    print(f"  vpn weighted events:   {len(vpn_events)}")

    cp_events = compute_crew_portal_weights(cp)
    print(f"  crew portal events:    {len(cp_events)}")

    maint_events = compute_maintenance_weights(maint)
    print(f"  maintenance events:    {len(maint_events)}")

    cargo_events = compute_cargo_weights(cargo)
    print(f"  cargo events:          {len(cargo_events)}")

    # 3. Combine
    all_events = pd.concat(
        [badge_events, vpn_events, cp_events, maint_events, cargo_events],
        ignore_index=True,
    )
    print(f"\n  Total weighted events: {len(all_events):,}")

    # 4. Compound bonus
    print("\n  Applying compound event bonus …")
    all_events = apply_compound_bonus(all_events)
    print(f"  After compound bonus:  {len(all_events):,}")

    # 5. Aggregate daily scores
    print("\n  Aggregating daily scores …")
    daily = aggregate_scores(all_events)

    # 6. Classify alerts
    print("  Classifying alert levels …")
    daily = classify_alerts(daily, all_events)

    # 7. Output
    print("\n" + "=" * 60)
    print("  Writing output files …")
    print("=" * 60)
    generate_outputs(daily, all_events, log_counts)

    # 8. Visualize
    visualize(daily, all_events)

    print("\n  All outputs saved to risk_output/ and data/ directory.\n")


if __name__ == "__main__":
    main()
