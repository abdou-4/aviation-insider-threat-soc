"""
soc_dashboard.py — SOC Visual Dashboard for Aviation Insider Threat Detection

Generates a comprehensive set of static PNG visualizations from the synthetic
aviation SOC logs and risk-scoring outputs.  Produces 8+ plots in
dashboard_output/ plus an executive summary composite.

Usage:
    python soc_dashboard.py
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("synthetic_logs")
RISK_DIR = Path("synthetic_logs")  # risk_summary.csv lives here
OUT_DIR = Path("dashboard_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-darkgrid")
COLOR_HIGH = "#e74c3c"
COLOR_MED = "#f39c12"
COLOR_LOW = "#3498db"
COLOR_NORMAL = "#2ecc71"
COLOR_ANOM = "#e74c3c"
ALERT_COLORS = {"high": COLOR_HIGH, "medium": COLOR_MED, "low": COLOR_LOW}

sns.set_palette("deep")
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
})

HIGH_RISK_COUNTRIES = {"RU", "CN", "IR", "KP", "SY"}
TAMPER_ACTIONS = ("override_sensor", "disable_alarm", "edit_log")
CRITICAL_COMPONENTS = ("Engine", "FuelSystem", "FlightControls")


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load(path: Path, name: str) -> pd.DataFrame | None:
    """Load a CSV, warn if missing, return None on failure."""
    if not path.exists():
        print(f"  ⚠  File not found: {path}")
        return None
    try:
        df = pd.read_csv(path)
        print(f"  ✓  Loaded {name}: {len(df):,} rows, {len(df.columns)} cols")
        return df
    except Exception as exc:
        print(f"  ⚠  Could not load {path}: {exc}")
        return None


def _save(fig: plt.Figure, name: str) -> None:
    """Save figure and close."""
    path = OUT_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


def _warn_missing(name: str) -> None:
    print(f"  ⚠  Skipping plot {name} — required data missing.\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Load all data
# ══════════════════════════════════════════════════════════════════════════════

def load_data() -> dict[str, pd.DataFrame]:
    """Return a dict of all available DataFrames."""
    print("=" * 60)
    print("  Loading data …")
    print("=" * 60)
    d: dict[str, pd.DataFrame] = {}

    for fname, key in [
        ("badge_logs.csv", "badge"),
        ("vpn_logs.csv", "vpn"),
        ("crew_portal_logs.csv", "crew"),
        ("maintenance_logs.csv", "maint"),
        ("cargo_logs.csv", "cargo"),
        ("employee_table.csv", "emp"),
    ]:
        d[key] = _load(DATA_DIR / fname, fname)

    # risk summary may be in DATA_DIR or RISK_DIR
    for folder, label in [(DATA_DIR, "synthetic_logs"), (RISK_DIR, "risk_output")]:
        p = folder / "risk_summary.csv"
        if p.exists():
            d["risk"] = _load(p, f"risk_summary.csv [{label}]")
            break
    else:
        print("  ⚠  risk_summary.csv not found anywhere.")

    for folder, label in [(DATA_DIR, "synthetic_logs"), (RISK_DIR, "risk_output")]:
        p = folder / "alerts_medium_high.csv"
        if p.exists():
            d["alerts_mh"] = _load(p, f"alerts_medium_high.csv [{label}]")
            break
    else:
        print("  ⚠  alerts_medium_high.csv not found anywhere.")

    print()
    return d


# ══════════════════════════════════════════════════════════════════════════════
#  1a. Top 10 employees by total risk score
# ══════════════════════════════════════════════════════════════════════════════

def plot_top10_risk(data: dict) -> bool:
    """Bar chart: top 10 employees by cumulative risk score."""
    print("[1a] Top 10 employees by total risk score …")
    risk = data.get("risk")
    if risk is None:
        _warn_missing("top10_risk")
        return False

    agg = risk.groupby(["employee_id", "role"], as_index=False)["total_score"].sum()
    top10 = agg.sort_values("total_score", ascending=False).head(10)
    top10["label"] = top10["employee_id"] + " (" + top10["role"] + ")"

    # Determine alert level for each bar
    def _level(s):
        return "high" if s >= 60 else ("medium" if s >= 30 else "low")

    top10["level"] = top10["total_score"].apply(_level)
    bar_colors = top10["level"].map(ALERT_COLORS)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(
        range(len(top10)), top10["total_score"],
        color=bar_colors, edgecolor="white", height=0.7,
    )
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(top10["label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Cumulative Risk Score")
    ax.set_title("Top 10 Employees by Total Risk Score")

    for bar, val in zip(bars, top10["total_score"]):
        ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}", va="center", fontsize=7)

    # Legend
    patches = [mpatches.Patch(color=c, label=lvl.title())
               for lvl, c in ALERT_COLORS.items()]
    ax.legend(handles=patches, title="Alert Level", fontsize=7, title_fontsize=8)

    _save(fig, "top10_risk.png")
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  1b. Risk timeline for highest-risk employee
# ══════════════════════════════════════════════════════════════════════════════

def plot_risk_timeline(data: dict) -> bool:
    """Line chart: daily risk score with threshold lines."""
    print("[1b] Risk timeline for highest-risk employee …")
    risk = data.get("risk")
    if risk is None:
        _warn_missing("risk_timeline")
        return False

    agg = risk.groupby("employee_id")["total_score"].sum()
    top_emp = agg.idxmax()
    emp_data = risk[risk["employee_id"] == top_emp].sort_values("date")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(
        pd.to_datetime(emp_data["date"]),
        emp_data["total_score"],
        marker="o", color=COLOR_HIGH, linewidth=1.2, markersize=4,
        label=f"{top_emp}",
    )
    ax.axhline(60, ls="--", color=COLOR_HIGH, alpha=0.7, label="High threshold (60)")
    ax.axhline(30, ls="--", color=COLOR_MED, alpha=0.7, label="Medium threshold (30)")
    ax.fill_between(
        pd.to_datetime(emp_data["date"]), 0, emp_data["total_score"],
        alpha=0.1, color=COLOR_HIGH,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Risk Score")
    ax.set_title(f"Risk Score Timeline — Highest-Risk Employee: {top_emp}")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()

    # Annotate max point
    max_row = emp_data.loc[emp_data["total_score"].idxmax()]
    ax.annotate(
        f"Peak: {max_row['total_score']:.0f}",
        xy=(pd.to_datetime(max_row["date"]), max_row["total_score"]),
        xytext=(10, 20), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=8,
    )

    _save(fig, "risk_timeline_highest_employee.png")
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  2. Alert level distribution (donut)
# ══════════════════════════════════════════════════════════════════════════════

def plot_alert_donut(data: dict) -> bool:
    """Donut chart: proportion of low / medium / high alert-days."""
    print("[2] Alert level distribution …")
    risk = data.get("risk")
    if risk is None:
        _warn_missing("alert_donut")
        return False

    counts = risk["alert_level"].value_counts()
    colors = [ALERT_COLORS.get(lvl, "#95a5a6") for lvl in counts.index]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=None,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.78,
        colors=colors,
        wedgeprops={"width": 0.4, "edgecolor": "white", "linewidth": 1.5},
    )
    for t in autotexts:
        t.set_fontsize(9)

    # Legend
    labels = [f"{k.title()}  ({v})" for k, v in counts.items()]
    ax.legend(wedges, labels, title="Alert Level", loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1), fontsize=8)
    ax.set_title("Alert Level Distribution (Employee-Days)", pad=20)

    _save(fig, "alert_level_donut.png")
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  3. Stacked bar: alerts per role by alert level
# ══════════════════════════════════════════════════════════════════════════════

def plot_alerts_by_role(data: dict) -> bool:
    """Stacked horizontal bar: number of alerts per role, broken by level."""
    print("[3] Alerts per role by alert level …")
    risk = data.get("risk")
    if risk is None:
        _warn_missing("alerts_by_role")
        return False

    ctab = pd.crosstab(risk["role"], risk["alert_level"])
    for lvl in ("low", "medium", "high"):
        if lvl not in ctab.columns:
            ctab[lvl] = 0
    ctab = ctab[["low", "medium", "high"]].sort_values("high", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ctab.plot(
        kind="barh", stacked=True, ax=ax,
        color=[COLOR_LOW, COLOR_MED, COLOR_HIGH],
        width=0.7,
    )
    ax.set_xlabel("Number of Alert-Days")
    ax.set_ylabel("Role")
    ax.set_title("Alerts per Role by Alert Level")
    ax.legend(title="Level", fontsize=8, title_fontsize=9)

    # Annotate bar segments
    for container in ax.containers:
        for bar in container:
            w = bar.get_width()
            if w > 0:
                ax.annotate(
                    f"{int(w)}",
                    xy=(bar.get_x() + w, bar.get_y() + bar.get_height() / 2),
                    ha="left", va="center", fontsize=6, color="#333",
                )

    _save(fig, "alerts_by_role_stacked.png")
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  4. Heatmap: badge access by hour and role
# ══════════════════════════════════════════════════════════════════════════════

def plot_badge_heatmap(data: dict) -> bool:
    """Heatmap of badge access frequency by hour × role."""
    print("[4] Badge access heatmap (hour × role) …")
    badge = data.get("badge")
    if badge is None:
        _warn_missing("badge_heatmap")
        return False

    ts = pd.to_datetime(badge["timestamp"], errors="coerce")
    badge["hour"] = ts.dt.hour

    pivot = badge.pivot_table(
        index="hour", columns="role", aggfunc="size", fill_value=0,
    )
    # Normalise per role (column)
    pivot_norm = pivot.divide(pivot.sum(axis=0), axis=1) * 100

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(
        pivot_norm, ax=ax, cmap="YlOrRd", cbar_kws={"label": "% of role's accesses"},
        linewidths=0.3, linecolor="#f0f0f0",
    )
    # Highlight off-hours band
    for hr in range(22, 24):
        ax.add_patch(plt.Rectangle((0, hr), pivot_norm.shape[1], 1,
                                   fill=False, edgecolor=COLOR_HIGH,
                                   linewidth=2.5, linestyle="--"))
    for hr in range(0, 6):
        ax.add_patch(plt.Rectangle((0, hr), pivot_norm.shape[1], 1,
                                   fill=False, edgecolor=COLOR_HIGH,
                                   linewidth=2.5, linestyle="--"))

    ax.set_xlabel("Role")
    ax.set_ylabel("Hour of Day")
    ax.set_title("Badge Access Frequency by Hour and Role\n(dashed border = off-hours 22:00–06:00)")

    _save(fig, "badge_heatmap_hour_role.png")
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  5. Histogram: file transfer sizes (normal vs anomalous)
# ══════════════════════════════════════════════════════════════════════════════

def plot_transfer_histogram(data: dict) -> bool:
    """Overlaid histogram of bytes_transferred for normal vs anomalous."""
    print("[5] Transfer size histogram (normal vs anomalous) …")
    crew = data.get("crew")
    if crew is None or "bytes_transferred" not in crew.columns:
        _warn_missing("transfer_histogram")
        return False

    # Use is_anomaly if available, else rule-based
    if "is_anomaly" in crew.columns:
        crew["anom"] = crew["is_anomaly"].astype(bool)
    else:
        crew["anom"] = (crew["bytes_transferred"] > 500_000_000)

    mb = crew["bytes_transferred"] / 1_000_000

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Linear scale
    ax = axes[0]
    ax.hist(mb[~crew["anom"]], bins=40, alpha=0.6, color=COLOR_NORMAL,
            label=f"Normal (n={(~crew['anom']).sum():,})")
    ax.hist(mb[crew["anom"]], bins=40, alpha=0.6, color=COLOR_ANOM,
            label=f"Anomalous ({crew['anom'].sum():,})")
    ax.set_xlabel("Transfer Size (MB)")
    ax.set_ylabel("Frequency")
    ax.set_title("Linear Scale")
    ax.legend(fontsize=7)

    # Log scale
    ax = axes[1]
    ax.hist(mb[~crew["anom"]], bins=40, alpha=0.6, color=COLOR_NORMAL,
            label="Normal")
    ax.hist(mb[crew["anom"]], bins=40, alpha=0.6, color=COLOR_ANOM,
            label="Anomalous")
    ax.set_xscale("log")
    ax.set_xlabel("Transfer Size (MB, log scale)")
    ax.set_ylabel("Frequency")
    ax.set_title("Log Scale")
    ax.legend(fontsize=7)

    fig.suptitle("Crew Portal — File Transfer Sizes: Normal vs Anomalous", fontsize=12, y=1.02)

    _save(fig, "transfer_size_histogram.png")
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  6. Bar chart: top 3 countries for normal vs anomalous VPN
# ══════════════════════════════════════════════════════════════════════════════

def plot_vpn_countries(data: dict) -> bool:
    """Side-by-side bars: top 3 geo_country for normal and anomalous VPN."""
    print("[6] VPN countries — normal vs anomalous …")
    vpn = data.get("vpn")
    if vpn is None or "geo_country" not in vpn.columns:
        _warn_missing("vpn_countries")
        return False

    if "is_anomaly" in vpn.columns:
        vpn["anom"] = vpn["is_anomaly"].astype(bool)
    else:
        typical = data.get("emp")
        if typical is not None:
            tc = typical.set_index("employee_id")["typical_country"].to_dict()
            vpn["_typical"] = vpn["username"].map(tc)
            vpn["anom"] = vpn["geo_country"].apply(
                lambda c: c in HIGH_RISK_COUNTRIES
            )
        else:
            vpn["anom"] = False

    normal_top = vpn[~vpn["anom"]]["geo_country"].value_counts().head(3)
    anom_top = vpn[vpn["anom"]]["geo_country"].value_counts().head(3)

    all_countries = list(dict.fromkeys(list(normal_top.index) + list(anom_top.index)))
    x = np.arange(len(all_countries))
    w = 0.35

    n_vals = [normal_top.get(c, 0) for c in all_countries]
    a_vals = [anom_top.get(c, 0) for c in all_countries]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, n_vals, w, label="Normal", color=COLOR_NORMAL, edgecolor="white")
    ax.bar(x + w / 2, a_vals, w, label="Anomalous", color=COLOR_ANOM, edgecolor="white")

    # Annotate bars
    for xi, nv, av in zip(x, n_vals, a_vals):
        if nv:
            ax.text(xi - w / 2, nv + max(n_vals + a_vals) * 0.01,
                    str(nv), ha="center", va="bottom", fontsize=7)
        if av:
            ax.text(xi + w / 2, av + max(n_vals + a_vals) * 0.01,
                    str(av), ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(all_countries, fontsize=9)
    ax.set_ylabel("Login Count")
    ax.set_title("Top 3 Geo-Countries: Normal vs Anomalous VPN Logins")
    ax.legend(fontsize=8)
    ax.set_yscale("log")  # handle large disparities

    _save(fig, "vpn_countries_normal_vs_anomalous.png")
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  7. Maintenance tampering per component
# ══════════════════════════════════════════════════════════════════════════════

def plot_maint_tamper(data: dict) -> bool:
    """Horizontal bar: tampering actions per critical component."""
    print("[7] Maintenance tampering per component …")
    maint = data.get("maint")
    if maint is None:
        _warn_missing("maint_tamper")
        return False

    tamper = maint[
        (maint["action"].isin(TAMPER_ACTIONS))
        & (maint["component"].isin(CRITICAL_COMPONENTS))
    ]
    if tamper.empty:
        print("  ⚠  No maintenance tampering events found.")
        return False

    ctab = pd.crosstab(tamper["component"], tamper["action"])
    for a in TAMPER_ACTIONS:
        if a not in ctab.columns:
            ctab[a] = 0
    ctab = ctab[list(TAMPER_ACTIONS)]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ctab.plot(
        kind="barh", ax=ax, stacked=True,
        color=["#e74c3c", "#c0392b", "#e67e22"],
        width=0.6,
    )
    ax.set_xlabel("Count")
    ax.set_ylabel("Critical Component")
    ax.set_title("Maintenance Tampering Actions by Critical Component")
    ax.legend(title="Action", fontsize=8, title_fontsize=9)

    for container in ax.containers:
        for bar in container:
            w = bar.get_width()
            if w:
                ax.text(w + 0.3, bar.get_y() + bar.get_height() / 2,
                        str(int(w)), ha="left", va="center", fontsize=7)

    _save(fig, "maint_tamper_by_component.png")
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  Executive summary dashboard (single composite)
# ══════════════════════════════════════════════════════════════════════════════

def plot_executive_dashboard(data: dict) -> bool:
    """4-panel composite for executive review."""
    print("[Executive] Composite dashboard …")
    risk = data.get("risk")
    badge = data.get("badge")
    if risk is None:
        _warn_missing("executive_dashboard")
        return False

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("SOC Executive Dashboard — Aviation Insider Threat Detection",
                 fontsize=14, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.28)

    # ── Top-Left: Top 3 high-risk employees ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    agg = risk.groupby(["employee_id", "role"], as_index=False)["total_score"].sum()
    top3 = agg.sort_values("total_score", ascending=False).head(3)
    top3["lbl"] = top3["employee_id"] + "\n(" + top3["role"] + ")"
    bars = ax1.bar(
        range(len(top3)), top3["total_score"],
        color=[COLOR_HIGH, COLOR_MED, COLOR_MED], edgecolor="white", width=0.5,
    )
    ax1.set_xticks(range(len(top3)))
    ax1.set_xticklabels(top3["lbl"], fontsize=8)
    ax1.set_ylabel("Cumulative Risk Score")
    ax1.set_title("Top 3 High-Risk Employees")
    for bar, val in zip(bars, top3["total_score"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                 f"{val:.0f}", ha="center", va="bottom", fontsize=8,
                 fontweight="bold")

    # ── Top-Right: Alerts over time ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    risk["date_dt"] = pd.to_datetime(risk["date"])
    daily_alerts = (
        risk[risk["alert_level"].isin(["medium", "high"])]
        .groupby("date_dt")
        .size()
    )
    if not daily_alerts.empty:
        ax2.plot(daily_alerts.index, daily_alerts.values,
                 color=COLOR_HIGH, marker=".", linewidth=1, markersize=3)
        ax2.fill_between(daily_alerts.index, 0, daily_alerts.values,
                         alpha=0.12, color=COLOR_HIGH)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Medium + High Alerts")
    ax2.set_title("Daily Alert Volume (Medium & High)")
    fig.autofmt_xdate()

    # ── Bottom-Left: Alert level pie ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    counts = risk["alert_level"].value_counts()
    colors_pie = [ALERT_COLORS.get(lvl, "#95a5a6") for lvl in counts.index]
    wedges, texts, autotexts = ax3.pie(
        counts.values, labels=None, autopct="%1.1f%%",
        startangle=90, colors=colors_pie,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    ax3.legend(
        wedges,
        [f"{k.title()}  ({v})" for k, v in counts.items()],
        title="Alert Level", loc="center left",
        bbox_to_anchor=(1.1, 0, 0.5, 1), fontsize=7,
    )
    ax3.set_title("Alert Level Breakdown")

    # ── Bottom-Right: Badge heatmap (small) ─────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    if badge is not None:
        ts = pd.to_datetime(badge["timestamp"], errors="coerce")
        badge["hour"] = ts.dt.hour
        pivot = badge.pivot_table(
            index="hour", columns="role", aggfunc="size", fill_value=0,
        )
        pivot_norm = pivot.divide(pivot.sum(axis=0), axis=1) * 100
        sns.heatmap(pivot_norm, ax=ax4, cmap="YlOrRd",
                    cbar_kws={"label": "%", "shrink": 0.6},
                    linewidths=0.1, linecolor="#f5f5f5")
    ax4.set_xlabel("Role")
    ax4.set_ylabel("Hour")
    ax4.set_title("Badge Access Heatmap")

    _save(fig, "executive_dashboard.png")
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print()
    data = load_data()

    print("=" * 60)
    print("  Generating dashboard plots …")
    print("=" * 60)

    results = {}
    results["1a"] = plot_top10_risk(data)
    results["1b"] = plot_risk_timeline(data)
    results["2"] = plot_alert_donut(data)
    results["3"] = plot_alerts_by_role(data)
    results["4"] = plot_badge_heatmap(data)
    results["5"] = plot_transfer_histogram(data)
    results["6"] = plot_vpn_countries(data)
    results["7"] = plot_maint_tamper(data)
    results["exec"] = plot_executive_dashboard(data)

    n_saved = sum(1 for v in results.values() if v)
    print("\n" + "=" * 60)
    print(f"  Dashboard complete: {n_saved} plots saved to {OUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
