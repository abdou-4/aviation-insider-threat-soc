"""
validate_logs.py — Validation & quality assurance for synthetic aviation SOC logs.

Loads all CSV files from synthetic_logs/, runs structural, statistical, and
semantic checks, then produces a markdown report and plots in validation_output/.

Usage:
    python validate_logs.py
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("synthetic_logs")
OUT_DIR = Path("validation_output")
REPORT_PATH = "data_validation_report.md"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Matplotlib style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 120,
    "figure.figsize": (10, 5),
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})
sns.set_style("whitegrid")

# ── File manifest ────────────────────────────────────────────────────────────
LOG_FILES = {
    "badge_logs.csv": {
        "label": "Badge Access Logs",
        "expected_cols": [
            "timestamp", "employee_id", "role", "door_location",
            "granted", "is_anomaly",
        ],
    },
    "vpn_logs.csv": {
        "label": "VPN Authentication Logs",
        "expected_cols": [
            "timestamp", "username", "role", "src_ip",
            "auth_result", "geo_country", "is_anomaly",
        ],
    },
    "crew_portal_logs.csv": {
        "label": "Crew Portal Activity Logs",
        "expected_cols": [
            "timestamp", "user", "role", "action",
            "record_type", "record_count", "bytes_transferred", "is_anomaly",
        ],
    },
    "maintenance_logs.csv": {
        "label": "Maintenance System Logs",
        "expected_cols": [
            "timestamp", "technician_id", "role", "aircraft_id",
            "action", "component", "status", "is_anomaly",
        ],
    },
    "cargo_logs.csv": {
        "label": "Cargo Handling Logs",
        "expected_cols": [
            "timestamp", "handler_id", "role", "cargo_id",
            "action", "weight_kg", "location", "is_anomaly",
        ],
    },
    "employee_table.csv": {
        "label": "Employee Table",
        "expected_cols": [
            "employee_id", "name", "role", "shift_start",
            "shift_end", "termination_date",
            "typical_access_zones", "typical_country",
        ],
    },
}

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


# ══════════════════════════════════════════════════════════════════════════════
#  1. Load and basic info
# ══════════════════════════════════════════════════════════════════════════════

def load_all() -> dict[str, pd.DataFrame]:
    """Load every CSV from DATA_DIR and return a dict of DataFrames."""
    data: dict[str, pd.DataFrame] = {}
    for fname, meta in LOG_FILES.items():
        path = DATA_DIR / fname
        print(f"[1/8] Loading {fname} …")
        if not path.exists():
            print(f"  ⚠  FILE NOT FOUND – {path}. Skipping.")
            continue
        df = pd.read_csv(path)
        data[fname] = df

        # basic info
        print(f"  Shape: {df.shape}")
        print(f"  Columns ({len(df.columns)}): {list(df.columns)}")
        # find missing / extra columns
        expected = set(meta["expected_cols"])
        actual = set(df.columns)
        missing = expected - actual
        extra = actual - expected
        if missing:
            print(f"  ⚠  Missing expected columns: {missing}")
        if extra:
            print(f"  ℹ  Extra columns present: {extra}")
        print(f"  Dtypes:\n{df.dtypes.to_string()}")
        print(f"  First 2 rows:\n{df.head(2).to_string()}")
        nulls = df.isnull().sum()
        nulls = nulls[nulls > 0]
        if nulls.empty:
            print("  Missing values: None")
        else:
            print(f"  Missing values:\n{nulls.to_string()}")
        print()
    return data


# ══════════════════════════════════════════════════════════════════════════════
#  2. Anomaly percentage
# ══════════════════════════════════════════════════════════════════════════════

def anomaly_percentage(data: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Print anomaly percentages and warn if out of expected band (1–10 %)."""
    print("[2/8] Anomaly percentage per file …")
    results: dict[str, float] = {}
    for fname in LOG_FILES:
        if fname == "employee_table.csv":
            continue
        df = data.get(fname)
        if df is None or "is_anomaly" not in df.columns:
            print(f"  ⚠  {fname}: is_anomaly column missing – skipping.")
            continue
        total = len(df)
        n_anom = df["is_anomaly"].sum()
        pct = (n_anom / total * 100) if total else 0.0
        results[fname] = pct
        label = LOG_FILES[fname]["label"]
        status = "✓" if 0 < pct <= 10 else "⚠  OUT OF BAND"
        print(f"  {status}  {label:35s}  {n_anom:>6,} / {total:>8,}  ({pct:.2f}%)")
        if pct == 0:
            print(f"         ⚠  Zero anomalies detected – injection may have failed.")
        elif pct > 10:
            print(f"         ⚠  Anomaly rate >10% – verify injection logic.")
    print()
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  3. Column-specific checks
# ══════════════════════════════════════════════════════════════════════════════

def column_specific_checks(data: dict[str, pd.DataFrame]) -> list[str]:
    """Run per-file semantic checks and return a list of finding strings."""
    print("[3/8] Column-specific semantic checks …")
    findings: list[str] = []

    # ── Badge ────────────────────────────────────────────────────────────────
    print("  Badge: off-hours vs shift correlation …")
    df = data.get("badge_logs.csv")
    if df is not None and "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        df["_hour"] = ts.dt.hour
        # off-hours defined as 22:00-06:00
        df["_off_hours"] = ((df["_hour"] >= 22) | (df["_hour"] < 6))

        # If shift_scheduled column exists, use it; otherwise print a note.
        if "shift_scheduled" in df.columns:
            anom_off = df[df["_off_hours"] & df["is_anomaly"]]
            scheduled_in_anom = anom_off["shift_scheduled"].sum()
            total_anom_off = len(anom_off)
            corr_pct = (
                (1 - scheduled_in_anom / total_anom_off) * 100
                if total_anom_off else 0
            )
            print(f"    Off-hours anomalous rows: {total_anom_off}")
            print(f"    … of which shift_scheduled==True: {scheduled_in_anom}")
            print(f"    … so {corr_pct:.1f}% are genuinely unscheduled (desired).")
            findings.append(
                f"Badge: {total_anom_off} off-hours anomalous accesses; "
                f"{corr_pct:.1f}% are shift-unscheduled."
            )
        else:
            print("    ℹ  'shift_scheduled' column not present – skipping correlation.")
            anom_off = df[df["_off_hours"] & df["is_anomaly"]]
            findings.append(
                f"Badge: {len(anom_off)} off-hours anomalous accesses "
                f"(shift_scheduled column unavailable for correlation)."
            )
    else:
        print("  ⚠  badge_logs.csv unavailable – skipping.")

    # ── VPN ──────────────────────────────────────────────────────────────────
    print("  VPN: top countries by anomaly status …")
    df = data.get("vpn_logs.csv")
    if df is not None and "geo_country" in df.columns and "is_anomaly" in df.columns:
        # ensure is_anomaly is boolean
        df["is_anomaly"] = df["is_anomaly"].astype(bool)
        normal = df[~df["is_anomaly"]]["geo_country"].value_counts().head(3)
        anom = df[df["is_anomaly"]]["geo_country"].value_counts().head(3)
        print("    Normal top-3 countries:")
        for c, n in normal.items():
            print(f"      {c}: {n}")
        print("    Anomalous top-3 countries:")
        for c, n in anom.items():
            print(f"      {c}: {n}")
        # Check for suspicious countries in anomalous set
        suspicious = {"RU", "CN", "IR", "KP", "SY"}
        anom_countries = set(df[df["is_anomaly"]]["geo_country"].unique())
        found_susp = anom_countries & suspicious
        if found_susp:
            print(f"    ✓  Suspicious countries found in anomalies: {found_susp}")
            findings.append(f"VPN: suspicious countries {found_susp} appear in anomalous logins.")
        else:
            print("    ⚠  No suspicious countries (RU, CN, IR…) in anomalies.")
            findings.append("VPN: no suspicious countries found in anomalies.")
    else:
        print("  ⚠  vpn_logs.csv unavailable – skipping.")

    # ── Crew portal ──────────────────────────────────────────────────────────
    print("  Crew portal: bulk download / large transfer check …")
    df = data.get("crew_portal_logs.csv")
    if df is not None and "is_anomaly" in df.columns:
        anom = df[df["is_anomaly"]]
        # rows where record_count > 100 or bytes_transferred > 500 MB
        mask_bulk = (anom.get("record_count", pd.Series(0)) > 100) | (
            anom.get("bytes_transferred", pd.Series(0)) > 500_000_000
        )
        n_bulk = mask_bulk.sum()
        print(f"    Anomalous rows with bulk indicator: {n_bulk} / {len(anom)}")
        findings.append(
            f"Crew portal: {n_bulk} anomalous rows satisfy "
            f"record_count>100 or bytes_transferred>500MB."
        )

        # Check SecurityProtocol access by non-security roles
        if "record_type" in df.columns and "role" in df.columns:
            sec_roles = {"SecurityOfficer", "ITAdmin", "Pilot"}
            non_sec = df[
                df["record_type"] == "SecurityProtocol"
            ]
            non_sec = non_sec[~non_sec["role"].isin(sec_roles)]
            print(f"    SecurityProtocol accessed by non-security roles: {len(non_sec)} rows")
            if len(non_sec) > 0:
                findings.append(
                    f"Crew portal: {len(non_sec)} SecurityProtocol accesses "
                    f"by non-security personnel."
                )
    else:
        print("  ⚠  crew_portal_logs.csv unavailable – skipping.")

    # ── Maintenance ──────────────────────────────────────────────────────────
    print("  Maintenance: distinct actions in anomalous rows …")
    df = data.get("maintenance_logs.csv")
    if df is not None and "is_anomaly" in df.columns and "action" in df.columns:
        anom_actions = df[df["is_anomaly"]]["action"].value_counts()
        print(f"    Anomalous action distribution:\n{anom_actions.to_string()}")

        # If work_order_valid column exists, cross-check with override_sensor
        if "work_order_valid" in df.columns:
            invalid_overrides = df[
                (df["is_anomaly"])
                & (df["action"].isin(["override_sensor", "disable_alarm"]))
                & (~df["work_order_valid"].astype(bool))
            ]
            print(f"    Override/disable with invalid work order: {len(invalid_overrides)}")
            findings.append(
                f"Maintenance: {len(invalid_overrides)} override_sensor/disable_alarm "
                f"events with invalid work order."
            )
        else:
            print("    ℹ  work_order_valid column absent – skipping work-order check.")

        # Check for USB source in install_software
        if "source" in df.columns:
            usb_installs = df[
                (df["is_anomaly"])
                & (df["action"] == "install_software")
                & (df["source"] == "USB")
            ]
            print(f"    USB software installations (anomalous): {len(usb_installs)}")
            if len(usb_installs) > 0:
                findings.append(
                    f"Maintenance: {len(usb_installs)} untrusted USB installs detected."
                )
    else:
        print("  ⚠  maintenance_logs.csv unavailable – skipping.")

    # ── Cargo ────────────────────────────────────────────────────────────────
    print("  Cargo: weight stats for anomalous edit_manifest rows …")
    df = data.get("cargo_logs.csv")
    if df is not None and "is_anomaly" in df.columns:
        anom = df[df["is_anomaly"]]
        edit_anom = anom[anom["action"] == "edit_manifest"] if "action" in anom.columns else anom
        if "weight_kg" in edit_anom.columns and len(edit_anom):
            print(f"    Anomalous edit_manifest rows: {len(edit_anom)}")
            print(f"    Weight (kg) – min: {edit_anom['weight_kg'].min():.1f}, "
                  f"max: {edit_anom['weight_kg'].max():.1f}, "
                  f"mean: {edit_anom['weight_kg'].mean():.1f}")
            light = (edit_anom["weight_kg"] < 5).sum()
            print(f"    Rows with weight < 5 kg (suspiciously light): {light}")
            findings.append(
                f"Cargo: {len(edit_anom)} anomalous edit_manifest events; "
                f"{light} have weight <5 kg suggesting large reductions."
            )
        else:
            print("    No anomalous edit_manifest rows with weight data.")
    else:
        print("  ⚠  cargo_logs.csv unavailable – skipping.")

    print()
    return findings


# ══════════════════════════════════════════════════════════════════════════════
#  4. Time-based analysis — plots
# ══════════════════════════════════════════════════════════════════════════════

def time_based_analysis(data: dict[str, pd.DataFrame]) -> None:
    """Plot histograms of event hour for normal vs anomalous (badge + VPN)."""
    print("[4/8] Time-based analysis – generating histograms …")

    for fname, label_prefix in [("badge_logs.csv", "Badge"), ("vpn_logs.csv", "VPN")]:
        df = data.get(fname)
        if df is None or "timestamp" not in df.columns or "is_anomaly" not in df.columns:
            print(f"  ⚠  {fname} unavailable – skipping plot.")
            continue

        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        df["_hour"] = ts.dt.hour
        df["is_anomaly"] = df["is_anomaly"].astype(bool)

        fig, ax = plt.subplots()
        for label, mask in [("Normal", False), ("Anomalous", True)]:
            subset = df[df["is_anomaly"] == mask]["_hour"]
            ax.hist(
                subset,
                bins=24,
                range=(-0.5, 23.5),
                alpha=0.6,
                label=label,
                color="#2ecc71" if label == "Normal" else "#e74c3c",
            )

        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Event Count")
        ax.set_title(f"{label_prefix} Access – Hour Distribution: Normal vs Anomalous")
        ax.legend()
        ax.set_xticks(range(0, 24, 2))

        out_path = OUT_DIR / f"{fname.replace('.csv', '')}_hours_normal_vs_anomaly.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        print(f"  -> Saved {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  5. Role-based anomaly distribution
# ══════════════════════════════════════════════════════════════════════════════

def role_anomaly_distribution(data: dict[str, pd.DataFrame]) -> str | None:
    """Print anomaly count per role for badge_logs; return path to saved CSV."""
    print("[5/8] Role-based anomaly distribution (badge) …")
    df = data.get("badge_logs.csv")
    if df is None or "role" not in df.columns or "is_anomaly" not in df.columns:
        print("  ⚠  badge_logs.csv unavailable – skipping.")
        return None

    df["is_anomaly"] = df["is_anomaly"].astype(bool)
    summary = (
        df.groupby("role")["is_anomaly"]
        .agg(["count", "sum"])
        .rename(columns={"count": "total", "sum": "anomalies"})
    )
    summary["anomaly_pct"] = (summary["anomalies"] / summary["total"] * 100).round(2)
    summary = summary.sort_values("anomaly_pct", ascending=False)

    print(f"  {'Role':25s} {'Total':>8s} {'Anomalies':>10s} {'%':>7s}")
    print("  " + "-" * 52)
    for _, row in summary.iterrows():
        print(
            f"  {row.name:25s} {int(row['total']):>8,} "
            f"{int(row['anomalies']):>10,} {row['anomaly_pct']:>6.2f}%"
        )

    csv_path = OUT_DIR / "badge_anomaly_by_role.csv"
    summary.to_csv(csv_path)
    print(f"  -> Saved {csv_path}")
    return csv_path


# ══════════════════════════════════════════════════════════════════════════════
#  6. Merge with employee table – off-hours validation
# ══════════════════════════════════════════════════════════════════════════════

def merge_off_hours_check(data: dict[str, pd.DataFrame]) -> None:
    """Merge badge logs with employee table and validate off-hours anomalies."""
    print("[6/8] Merge badge_logs with employee_table – off-hours check …")
    badge = data.get("badge_logs.csv")
    emp = data.get("employee_table.csv")

    if badge is None or emp is None:
        print("  ⚠  Missing badge_logs.csv or employee_table.csv – skipping.")
        return

    if "employee_id" not in badge.columns or "employee_id" not in emp.columns:
        print("  ⚠  employee_id column missing in one of the tables – skipping.")
        return

    merged = badge.merge(emp, on="employee_id", how="left", suffixes=("", "_emp"))
    ts = pd.to_datetime(merged["timestamp"], errors="coerce")
    merged["_hour"] = ts.dt.hour

    # Map off-hours based on employee shift
    def is_off_hours(hour: int, sh: int, se: int) -> bool:
        if pd.isna(hour) or pd.isna(sh) or pd.isna(se):
            return False
        hour = int(hour)
        sh = int(sh)
        se = int(se)
        if sh <= se:
            return hour < sh or hour >= se
        else:  # overnight shift
            return se <= hour < sh

    merged["_off_hours_expected"] = merged.apply(
        lambda r: is_off_hours(r["_hour"], r["shift_start"], r["shift_end"]),
        axis=1,
    )

    anom = merged[merged["is_anomaly"].astype(bool)]
    n_anom_off = anom["_off_hours_expected"].sum()
    total_anom = len(anom)
    pct = (n_anom_off / total_anom * 100) if total_anom else 0.0
    print(f"  Anomalous rows: {total_anom}")
    print(f"  … where hour is outside employee's shift: {n_anom_off} ({pct:.1f}%)")
    print(f"  → Off-hours anomaly detection is {pct:.1f}% aligned with employee shift data.")

    findings.append(
        f"Badge-employee merge: {pct:.1f}% of anomalous badge events fall "
        f"outside the employee's scheduled shift."
    )


# ══════════════════════════════════════════════════════════════════════════════
#  7. Generate summary report
# ══════════════════════════════════════════════════════════════════════════════

def write_report(
    data: dict[str, pd.DataFrame],
    anom_pcts: dict[str, float],
    findings: list[str],
) -> None:
    """Write a markdown report summarizing all validation results."""
    print("[7/8] Writing summary report …")

    lines = [
        "# Data Validation Report – Aviation SOC Synthetic Logs",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## Overview",
        "",
        "| File | Rows | Columns | Anomaly % |",
        "|------|-----:|--------:|----------:|",
    ]

    for fname in LOG_FILES:
        df = data.get(fname)
        if df is None:
            lines.append(f"| {fname} | — | — | — |")
            continue
        n_cols = df.shape[1]
        n_rows = df.shape[0]
        pct = anom_pcts.get(fname, None)
        pct_str = f"{pct:.2f}%" if pct is not None else "N/A"
        lines.append(f"| {fname} | {n_rows:,} | {n_cols} | {pct_str} |")

    lines += [
        "",
        "---",
        "",
        "## Key Findings",
        "",
    ]

    for i, f in enumerate(findings, 1):
        lines.append(f"{i}. {f}")

    lines += [
        "",
        "---",
        "",
        "## Data Quality Warnings",
        "",
    ]

    warnings_list: list[str] = []
    for fname, df in data.items():
        if df is None:
            continue
        nulls = df.isnull().sum()
        nulls = nulls[nulls > 0]
        if not nulls.empty:
            for col, n in nulls.items():
                warnings_list.append(
                    f"- **{fname}**: {n} missing values in column `{col}`."
                )
        # check for unexpected column types
        for col in df.columns:
            if df[col].dtype == object:
                nunique = df[col].nunique()
                if nunique == 1 and col != "role":
                    warnings_list.append(
                        f"- **{fname}**: Column `{col}` has only one unique value "
                        f"('{df[col].iloc[0]}') – possible generation issue."
                    )

    # Check if anomaly percentages are out of band (already flagged earlier)
    for fname, pct in anom_pcts.items():
        if pct == 0:
            warnings_list.append(
                f"- **{fname}**: Anomaly rate is 0% – injection may not have worked."
            )
        elif pct > 10:
            warnings_list.append(
                f"- **{fname}**: Anomaly rate is {pct:.1f}% (>10%) – "
                f"higher than expected 3-5%."
            )

    if not warnings_list:
        warnings_list.append("- No data quality issues detected.")

    lines += warnings_list

    lines += [
        "",
        "---",
        "",
        "## Generated Plots",
        "",
        f"- **Badge hour distribution:** `validation_output/badge_logs_hours_normal_vs_anomaly.png`",
        f"- **VPN hour distribution:** `validation_output/vpn_logs_hours_normal_vs_anomaly.png`",
        f"- **Validation dashboard:** `validation_output/validation_dashboard.png`",
        "",
        "---",
        "",
        "_Report generated automatically by validate_logs.py_",
        "",
    ]

    report = "\n".join(lines)
    with open(REPORT_PATH, "w") as fh:
        fh.write(report)
    print(f"  -> Saved {REPORT_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
#  8. Dashboard (4 subplots)
# ══════════════════════════════════════════════════════════════════════════════

def validation_dashboard(data: dict[str, pd.DataFrame]) -> None:
    """Create a 4-panel dashboard figure and save it."""
    print("[8/8] Generating validation dashboard …")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Synthetic Aviation SOC Logs – Validation Dashboard", fontsize=14, y=1.02)

    # ── Panel A: Anomaly count by role (badge) ───────────────────────────────
    ax = axes[0, 0]
    df = data.get("badge_logs.csv")
    if df is not None and "role" in df.columns and "is_anomaly" in df.columns:
        df["is_anomaly"] = df["is_anomaly"].astype(bool)
        role_anom = df[df["is_anomaly"]]["role"].value_counts()
        colors = sns.color_palette("Reds_r", n_colors=len(role_anom))
        bars = ax.barh(role_anom.index, role_anom.values, color=colors)
        ax.set_xlabel("Anomaly Count")
        ax.set_title("Badge: Anomaly Count by Role")
        for bar, v in zip(bars, role_anom.values):
            ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
                    str(v), va="center", fontsize=7)
    else:
        ax.text(0.5, 0.5, "Data unavailable", ha="center", va="center")

    # ── Panel B: Heatmap — access frequency by hour × role (badge) ──────────
    ax = axes[0, 1]
    if df is not None and "_hour" not in df.columns and "timestamp" in df.columns:
        df["_hour"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.hour
    if df is not None and "_hour" in df.columns and "role" in df.columns:
        pivot = df.pivot_table(
            index="_hour", columns="role", aggfunc="size", fill_value=0
        )
        sns.heatmap(pivot, ax=ax, cmap="YlOrRd", cbar_kws={"label": "Access count"})
        ax.set_title("Badge: Access Frequency (Hour × Role)")
        ax.set_xlabel("Role")
        ax.set_ylabel("Hour of Day")
    else:
        ax.text(0.5, 0.5, "Data unavailable", ha="center", va="center")

    # ── Panel C: Top 3 countries for normal vs anomalous VPN ────────────────
    ax = axes[1, 0]
    vpn = data.get("vpn_logs.csv")
    if vpn is not None and "geo_country" in vpn.columns and "is_anomaly" in vpn.columns:
        vpn["is_anomaly"] = vpn["is_anomaly"].astype(bool)
        top_n = vpn[~vpn["is_anomaly"]]["geo_country"].value_counts().head(3).index
        top_a = vpn[vpn["is_anomaly"]]["geo_country"].value_counts().head(3).index
        # build a small grouped bar
        normal_counts = (
            vpn[~vpn["is_anomaly"]]["geo_country"]
            .value_counts()
            .reindex(top_n)
            .fillna(0)
        )
        anom_counts = (
            vpn[vpn["is_anomaly"]]["geo_country"]
            .value_counts()
            .reindex(top_a)
            .fillna(0)
        )

        x = np.arange(max(len(top_n), len(top_a)))
        w = 0.35
        ax.bar(x[: len(top_n)] - w / 2, normal_counts.values, w,
               label="Normal", color="#2ecc71")
        ax.bar(x[: len(top_a)] + w / 2, anom_counts.values, w,
               label="Anomalous", color="#e74c3c")
        ax.set_xticks(x[: max(len(top_n), len(top_a))])
        # combine labels
        all_labels = list(set(list(top_n) + list(top_a)))
        ax.set_xticklabels(all_labels, rotation=45, ha="right")
        ax.legend()
        ax.set_ylabel("Count")
        ax.set_title("VPN: Top Countries (Normal vs Anomalous)")
    else:
        ax.text(0.5, 0.5, "Data unavailable", ha="center", va="center")

    # ── Panel D: File transfer sizes — crew portal ──────────────────────────
    ax = axes[1, 1]
    cp = data.get("crew_portal_logs.csv")
    if cp is not None and "bytes_transferred" in cp.columns and "is_anomaly" in cp.columns:
        cp["is_anomaly"] = cp["is_anomaly"].astype(bool)
        mb = cp["bytes_transferred"] / 1_000_000  # convert to MB
        ax.hist(
            mb[~cp["is_anomaly"]],
            bins=50,
            alpha=0.6,
            label="Normal",
            color="#2ecc71",
        )
        ax.hist(
            mb[cp["is_anomaly"]],
            bins=50,
            alpha=0.6,
            label="Anomalous",
            color="#e74c3c",
        )
        ax.set_xlabel("Transfer Size (MB)")
        ax.set_ylabel("Frequency")
        ax.set_title("Crew Portal: Transfer Sizes (Normal vs Anomalous)")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Data unavailable", ha="center", va="center")

    fig.tight_layout()
    out_path = OUT_DIR / "validation_dashboard.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("  Aviation SOC Log Validation")
    print("=" * 60)
    print()

    # 1
    data = load_all()

    # 2
    anom_pcts = anomaly_percentage(data)

    # 3 (findings list shared across steps 3 & 6)
    global findings
    findings = column_specific_checks(data)

    # 4
    time_based_analysis(data)

    # 5
    role_anomaly_distribution(data)

    # 6
    merge_off_hours_check(data)

    # 7
    write_report(data, anom_pcts, findings)

    # 8
    validation_dashboard(data)

    print()
    print("=" * 60)
    print("  Validation complete.")
    print(f"  Reports saved to {OUT_DIR}/ and {REPORT_PATH}")
    print("=" * 60)


findings: list[str] = []

if __name__ == "__main__":
    main()
