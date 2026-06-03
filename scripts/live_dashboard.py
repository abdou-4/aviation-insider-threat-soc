"""
live_dashboard.py — Interactive SOC Dashboard for Aviation Insider Threats

Loads synthetic log CSVs from synthetic_logs/ and presents a multi-tab
Streamlit dashboard with Plotly charts.

Usage:
    pip install streamlit pandas numpy plotly
    streamlit run live_dashboard.py
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Aviation SOC Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ───────────────────────────────────────────────────────────────────
DATA_DIR = Path("synthetic_logs")

# ── Constants ───────────────────────────────────────────────────────────────
HIGH_RISK_COUNTRIES = {"RU", "CN", "IR", "KP", "SY"}
SENSITIVE_DOORS = {"CargoArea", "DataCenter", "MaintenanceHangar"}
SECURITY_ROLES = {"SecurityOfficer", "ITAdmin", "Pilot"}
CRITICAL_COMPONENTS = {"Engine", "FuelSystem", "FlightControls"}
TAMPER_ACTIONS = {"override_sensor", "disable_alarm", "edit_log"}
OFF_HOURS_SET = {22, 23, 0, 1, 2, 3, 4, 5}
DOOR_LOCATIONS = [
    "CockpitDoor", "CargoArea", "MaintenanceHangar", "FuelFarm",
    "OpsCenter", "SecureGate", "BaggageCarousel", "DataCenter",
]
ALERT_COLORS = {"low": "#3498db", "medium": "#f39c12", "high": "#e74c3c"}


# ══════════════════════════════════════════════════════════════════════════════
#  Dynamic CSS theme injection
# ══════════════════════════════════════════════════════════════════════════════

def inject_theme_css(dark_mode: bool) -> None:
    if dark_mode:
        st.markdown("""
        <style>
        .stApp, .stTabs, .stMarkdown, .stDataFrame, .stSelectbox {
            background-color: #1e1e1e !important;
            color: #e0e0e0 !important;
        }
        .stSidebar {
            background-color: #252526 !important;
        }
        .st-bb, .st-at, .st-ae {
            background-color: #2d2d2d !important;
        }
        h1, h2, h3, h4, h5, h6, p, li, label, span {
            color: #e0e0e0 !important;
        }
        .stDataFrame {
            color: #e0e0e0 !important;
        }
        .stMetric {
            background-color: #2d2d2d !important;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp { background-color: #ffffff; }
        </style>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Cached data loading
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def load_employee_table() -> pd.DataFrame:
    return _load_csv("employee_table.csv")


@st.cache_data(ttl=300)
def load_badge_logs() -> pd.DataFrame:
    return _load_csv("badge_logs.csv")


@st.cache_data(ttl=300)
def load_vpn_logs() -> pd.DataFrame:
    return _load_csv("vpn_logs.csv")


@st.cache_data(ttl=300)
def load_crew_portal_logs() -> pd.DataFrame:
    return _load_csv("crew_portal_logs.csv")


@st.cache_data(ttl=300)
def load_maintenance_logs() -> pd.DataFrame:
    return _load_csv("maintenance_logs.csv")


@st.cache_data(ttl=300)
def load_cargo_logs() -> pd.DataFrame:
    return _load_csv("cargo_logs.csv")


@st.cache_data(ttl=300)
def load_risk_summary() -> pd.DataFrame:
    return _load_csv("risk_summary.csv")


def _load_csv(fname: str) -> pd.DataFrame:
    path = DATA_DIR / fname
    if not path.exists():
        st.warning(f"⚠️ File not found: `{path}` — skipping.")
        return pd.DataFrame()
    return pd.read_csv(path)


# ── Enrichment helpers ──────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def enrich_badge(badge: pd.DataFrame, emp: pd.DataFrame) -> pd.DataFrame:
    if badge.empty or emp.empty:
        return badge
    df = badge.merge(
        emp[["employee_id", "shift_start", "shift_end"]],
        on="employee_id", how="left",
    )
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour"] = ts.dt.hour
    df["date"] = ts.dt.date

    def _scheduled(r):
        if pd.isna(r.get("shift_start")) or pd.isna(r.get("shift_end")):
            return True
        h, sh, se = int(r["hour"]), int(r["shift_start"]), int(r["shift_end"])
        return not (h < sh or h >= se) if sh <= se else not (se <= h < sh)

    df["shift_scheduled"] = df.apply(_scheduled, axis=1)
    df["off_hours"] = df["hour"].isin(OFF_HOURS_SET)
    df["sensitive_door"] = df["door_location"].isin(SENSITIVE_DOORS)
    return df


@st.cache_data(ttl=300)
def enrich_vpn(vpn: pd.DataFrame, emp: pd.DataFrame) -> pd.DataFrame:
    if vpn.empty or emp.empty:
        return vpn
    df = vpn.merge(
        emp[["employee_id", "typical_country"]],
        left_on="username", right_on="employee_id", how="left",
    )
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour"] = ts.dt.hour
    df["date"] = ts.dt.date
    df["high_risk_country"] = df["geo_country"].isin(HIGH_RISK_COUNTRIES)
    df["country_mismatch"] = (
        (df["geo_country"] != df["typical_country"]) & df["typical_country"].notna()
    )
    df["anom_country"] = df["high_risk_country"] | df["country_mismatch"]
    return df


@st.cache_data(ttl=300)
def enrich_crew(cp: pd.DataFrame) -> pd.DataFrame:
    if cp.empty:
        return cp
    cp = cp.copy()
    ts = pd.to_datetime(cp["timestamp"], errors="coerce")
    cp["hour"] = ts.dt.hour
    cp["date"] = ts.dt.date
    if "is_anomaly" not in cp.columns:
        cp["is_anomaly"] = cp["bytes_transferred"] > 500_000_000
    cp["bulk"] = (cp["action"] == "download") & (
        (cp["record_count"] > 100) | (cp["bytes_transferred"] > 500_000_000)
    )
    return cp


@st.cache_data(ttl=300)
def enrich_maint(maint: pd.DataFrame) -> pd.DataFrame:
    if maint.empty:
        return maint
    maint = maint.copy()
    ts = pd.to_datetime(maint["timestamp"], errors="coerce")
    maint["hour"] = ts.dt.hour
    maint["date"] = ts.dt.date
    maint["is_tamper"] = maint["action"].isin(TAMPER_ACTIONS) & maint["component"].isin(CRITICAL_COMPONENTS)
    return maint


@st.cache_data(ttl=300)
def enrich_cargo(cargo: pd.DataFrame, emp: pd.DataFrame) -> pd.DataFrame:
    if cargo.empty:
        return cargo
    df = cargo.merge(
        emp[["employee_id", "shift_start", "shift_end"]],
        left_on="handler_id", right_on="employee_id", how="left",
    )
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour"] = ts.dt.hour
    df["date"] = ts.dt.date

    def _scheduled(r):
        if pd.isna(r.get("shift_start")) or pd.isna(r.get("shift_end")):
            return True
        h, sh, se = int(r["hour"]), int(r["shift_start"]), int(r["shift_end"])
        return not (h < sh or h >= se) if sh <= se else not (se <= h < sh)

    df["shift_scheduled"] = df.apply(_scheduled, axis=1)
    df = df.sort_values(["cargo_id", "timestamp"])
    df["prev_weight"] = df.groupby("cargo_id")["weight_kg"].shift(1)
    df["delta"] = df["prev_weight"] - df["weight_kg"]
    df["weight_anom"] = (df["action"] == "edit_manifest") & (df["delta"] > 20)
    df["unauth_role"] = ~df["role"].isin(["RampAgent"])
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Utility
# ══════════════════════════════════════════════════════════════════════════════

def _to_csv_download(df: pd.DataFrame) -> str:
    return df.to_csv(index=False)


def detect_brute_force(vpn: pd.DataFrame) -> pd.DataFrame:
    """Return grouped brute-force attempts (>=4 failures in 10 min per user+ip)."""
    if vpn.empty or "auth_result" not in vpn.columns:
        return pd.DataFrame()
    fails = (
        vpn[vpn["auth_result"] == "failure"]
        .sort_values(["username", "src_ip", "timestamp"])
        .copy()
    )
    fails["ts"] = pd.to_datetime(fails["timestamp"])
    fails["next_ts"] = fails.groupby(["username", "src_ip"])["ts"].shift(-1)
    fails["window"] = (
        (fails["next_ts"] - fails["ts"]).dt.total_seconds().le(600)
    )
    results = []
    for (uname, ip), grp in fails.groupby(["username", "src_ip"]):
        grp = grp.sort_values("ts")
        span = (grp["ts"].max() - grp["ts"].min()).total_seconds() / 60
        if len(grp) >= 4 and span <= 10:
            results.append({
                "username": uname,
                "src_ip": ip,
                "fail_count": len(grp),
                "first_ts": grp["ts"].iloc[0],
                "last_ts": grp["ts"].iloc[-1],
            })
    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ══════════════════════════════════════════════════════════════════════════════

def build_sidebar(risk: pd.DataFrame, emp: pd.DataFrame) -> dict:
    st.sidebar.title("✈️ Aviation SOC")
    st.sidebar.caption("Insider Threat Detection Dashboard")

    # ── Dark mode toggle ──────────────────────────────────────────────────
    dark_mode = st.sidebar.checkbox("🌙 Dark mode", value=False)
    inject_theme_css(dark_mode)

    st.sidebar.markdown("---")

    # ── Date range ────────────────────────────────────────────────────────
    all_dates: list = []
    for src in [risk]:
        if not src.empty and "date" in src.columns:
            dts = pd.to_datetime(src["date"], errors="coerce")
            all_dates.extend(dts.dropna())
    min_date = min(all_dates).date() if all_dates else datetime(2025, 5, 1).date()
    max_date = max(all_dates).date() if all_dates else datetime(2025, 5, 30).date()

    date_range = st.sidebar.date_input(
        "📅 Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    # ── Role multi-select ─────────────────────────────────────────────────
    all_roles = sorted(emp["role"].unique()) if not emp.empty else []
    if not all_roles and not risk.empty:
        all_roles = sorted(risk["role"].unique())
    selected_roles = st.sidebar.multiselect(
        "👤 Role", all_roles, default=all_roles,
    )

    # ── Employee select ───────────────────────────────────────────────────
    if not emp.empty:
        emp_choices = ["All"] + sorted(
            emp[emp["role"].isin(selected_roles)]["employee_id"].tolist()
            if selected_roles else emp["employee_id"].tolist()
        )
    else:
        emp_choices = ["All"]
    selected_emp = st.sidebar.selectbox("🆔 Employee ID", emp_choices, index=0)

    # ── Alert level ───────────────────────────────────────────────────────
    alert_levels = st.sidebar.multiselect(
        "🚦 Alert level", ["low", "medium", "high"],
        default=["low", "medium", "high"],
    )

    # ── Refresh ───────────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()

    return {
        "start_date": start_date,
        "end_date": end_date,
        "roles": selected_roles,
        "employee": selected_emp,
        "alert_levels": alert_levels,
        "dark_mode": dark_mode,
    }


def filter_df(
    df: pd.DataFrame,
    filters: dict,
    date_col: str = "date",
    role_col: str = "role",
    emp_col: str | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df
    mask = pd.Series(True, index=df.index)
    if date_col in df.columns:
        dts = pd.to_datetime(df[date_col], errors="coerce")
        mask &= (dts.dt.date >= filters["start_date"]) & (
            dts.dt.date <= filters["end_date"]
        )
    if role_col in df.columns and filters["roles"]:
        mask &= df[role_col].isin(filters["roles"])
    if emp_col and emp_col in df.columns and filters["employee"] != "All":
        mask &= df[emp_col] == filters["employee"]
    return df[mask].copy()


def get_plotly_template(filters: dict) -> str:
    return "plotly_dark" if filters.get("dark_mode") else "plotly_white"


# ══════════════════════════════════════════════════════════════════════════════
#  Tab 1: Risk Overview
# ══════════════════════════════════════════════════════════════════════════════

def render_tab1_risk(risk: pd.DataFrame, filters: dict) -> None:
    st.markdown("### 📊 Risk Score Overview")
    risk_f = filter_df(risk, filters)
    if risk_f.empty:
        st.info("No data matches filters.")
        return

    # Since risk_summary stores cumulative decayed scores per day,
    # pull the latest (final cumulative) score per employee.
    latest = risk_f.sort_values("date").groupby(
        ["employee_id", "role"], as_index=False
    ).last()
    top10 = latest.nlargest(10, "total_score")
    top10["label"] = top10["employee_id"] + " (" + top10["role"] + ")"
    tpl = get_plotly_template(filters)

    col1, col2 = st.columns([1.6, 1])

    with col1:
        fig = px.bar(
            top10,
            x="total_score",
            y="label",
            orientation="h",
            title="Top 10 Employees by Cumulative Risk Score",
            labels={"total_score": "Cumulative Risk Score", "label": ""},
            color="total_score",
            color_continuous_scale="Reds",
            text_auto=".0f",
            template=tpl,
        )
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=40, b=0))
        fig.update_traces(textposition="outside")
        # Store clicked point in session state
        fig.update_layout(clickmode="event+select")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        counts = risk_f["alert_level"].value_counts()
        fig = go.Figure(
            go.Pie(
                labels=[k.title() for k in counts.index],
                values=counts.values,
                marker_colors=[ALERT_COLORS.get(lvl, "#95a5a6") for lvl in counts.index],
                hole=0.4,
                textinfo="label+percent",
            )
        )
        fig.update_layout(title="Alert Level Distribution", height=420, template=tpl)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns([1, 1])

    with col3:
        top_emp_id = top10.iloc[0]["employee_id"]
        emp_ts = risk_f[risk_f["employee_id"] == top_emp_id].sort_values("date")
        if not emp_ts.empty:
            fig = px.line(
                emp_ts,
                x="date",
                y="total_score",
                title=f"Daily Risk Score – {top_emp_id} ({top10.iloc[0]['role']})",
                markers=True,
                template=tpl,
            )
            fig.add_hline(y=60, line_dash="dash", line_color="red", annotation_text="High (60)")
            fig.add_hline(y=30, line_dash="dash", line_color="orange", annotation_text="Medium (30)")
            fig.update_layout(height=360)
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        ctab = pd.crosstab(risk_f["role"], risk_f["alert_level"])
        for lvl in ["low", "medium", "high"]:
            if lvl not in ctab.columns:
                ctab[lvl] = 0
        ctab = ctab[["low", "medium", "high"]].sort_values("high", ascending=True)
        fig = px.bar(
            ctab,
            orientation="h",
            barmode="stack",
            title="Alerts per Role by Level",
            labels={"value": "Alert-Days", "index": "Role"},
            color_discrete_map=ALERT_COLORS,
            template=tpl,
        )
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Tab 2: Badge Access
# ══════════════════════════════════════════════════════════════════════════════

def render_tab2_badge(badge: pd.DataFrame, emp: pd.DataFrame, filters: dict) -> None:
    st.markdown("### 🪪 Badge Access Analysis")

    badge = enrich_badge(badge, emp)
    badge_f = filter_df(badge, filters, date_col="date", emp_col="employee_id")
    if badge_f.empty:
        st.info("No data matches filters.")
        return

    tpl = get_plotly_template(filters)

    col1, col2 = st.columns([1.4, 1])

    with col1:
        pivot = badge_f.pivot_table(
            index="hour", columns="role", aggfunc="size", fill_value=0,
        )
        fig = px.imshow(
            pivot,
            aspect="auto",
            color_continuous_scale="YlOrRd",
            title="Badge Access Frequency (Hour × Role)",
            labels={"x": "Role", "y": "Hour of Day", "color": "Accesses"},
            template=tpl,
        )
        fig.add_hline(y=21.5, line_dash="dash", line_color="red", line_width=2)
        fig.add_hline(y=5.5, line_dash="dash", line_color="red", line_width=2)
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        off = badge_f[badge_f["off_hours"]]
        st.metric("Total Off‑Hours Events", f"{len(off):,}")
        st.metric("Off‑Hours & Unscheduled", f"{int((~off['shift_scheduled']).sum()):,}")
        st.metric("Off‑Hours & Sensitive Door", f"{int((off['off_hours'] & off['sensitive_door']).sum()):,}")

        st.markdown("#### Filter Off‑Hours Table")
        door_filter = st.multiselect("Door location", DOOR_LOCATIONS, default=[])
        role_filter = st.multiselect(
            "Role", sorted(badge_f["role"].unique()), default=[],
            key="badge_role_filter",
        )

    st.markdown("---")
    st.markdown("#### Off‑Hours Access Events (22:00–06:00, Unscheduled)")

    off_sus = badge_f[badge_f["off_hours"] & ~badge_f["shift_scheduled"]]
    if door_filter:
        off_sus = off_sus[off_sus["door_location"].isin(door_filter)]
    if role_filter:
        off_sus = off_sus[off_sus["role"].isin(role_filter)]

    if not off_sus.empty:
        show = off_sus[
            ["timestamp", "employee_id", "role", "door_location", "granted"]
        ].copy()
        show["timestamp"] = pd.to_datetime(show["timestamp"]).dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        st.dataframe(show, use_container_width=True, height=350)
        st.download_button(
            "📥 Download off‑hours badge events",
            _to_csv_download(show),
            "badge_off_hours.csv",
            "text/csv",
        )
    else:
        st.info("No matching events.")


# ══════════════════════════════════════════════════════════════════════════════
#  Tab 3: VPN & Network
# ══════════════════════════════════════════════════════════════════════════════

def render_tab3_vpn(vpn: pd.DataFrame, emp: pd.DataFrame, filters: dict) -> None:
    st.markdown("### 🔐 VPN & Network Anomalies")

    vpn = enrich_vpn(vpn, emp)
    vpn_f = filter_df(vpn, filters, date_col="date", emp_col="username")
    if vpn_f.empty:
        st.info("No data matches filters.")
        return

    tpl = get_plotly_template(filters)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        normal_top = vpn_f[~vpn_f["anom_country"]]["geo_country"].value_counts().head(3)
        anom_top = vpn_f[vpn_f["anom_country"]]["geo_country"].value_counts().head(3)
        all_ctry = list(dict.fromkeys(list(normal_top.index) + list(anom_top.index)))
        n_vals = [normal_top.get(c, 0) for c in all_ctry]
        a_vals = [anom_top.get(c, 0) for c in all_ctry]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(name="Normal", x=all_ctry, y=n_vals, marker_color="#2ecc71")
        )
        fig.add_trace(
            go.Bar(name="Anomalous", x=all_ctry, y=a_vals, marker_color="#e74c3c")
        )
        fig.update_layout(
            barmode="group",
            title="VPN Logins by Country (Normal vs Anomalous)",
            yaxis_type="log",
            height=360,
            template=tpl,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("Total VPN Events", f"{len(vpn_f):,}")
        st.metric("Anomalous Country Logins", f"{int(vpn_f['anom_country'].sum()):,}")
        st.metric("High‑Risk Country Hits", f"{int(vpn_f['high_risk_country'].sum()):,}")

    st.markdown("---")
    st.markdown("#### Suspicious VPN Logins")

    sus_vpn = vpn_f[vpn_f["anom_country"]]
    if not sus_vpn.empty:
        show = sus_vpn[
            ["timestamp", "username", "role", "src_ip", "geo_country", "auth_result", "typical_country"]
        ].copy()
        show["timestamp"] = pd.to_datetime(show["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(show, use_container_width=True, height=250)
        st.download_button(
            "📥 Download suspicious VPN logins",
            _to_csv_download(show),
            "vpn_suspicious.csv",
            "text/csv",
        )

    st.markdown("---")
    st.markdown("#### Brute‑Force Attempts (≥4 failures in 10 min)")

    bf = detect_brute_force(vpn_f)
    if not bf.empty:
        bf["first_ts"] = pd.to_datetime(bf["first_ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(
            bf[["first_ts", "username", "src_ip", "fail_count"]],
            use_container_width=True,
            height=200,
        )
    else:
        st.info("No brute‑force patterns detected in filtered data.")


# ══════════════════════════════════════════════════════════════════════════════
#  Tab 4: Data Exfiltration
# ══════════════════════════════════════════════════════════════════════════════

def render_tab4_exfil(cp: pd.DataFrame, filters: dict) -> None:
    st.markdown("### 📁 Data Exfiltration (Crew Portal)")

    cp = enrich_crew(cp)
    cp_f = filter_df(cp, filters, date_col="date", emp_col="user")
    if cp_f.empty:
        st.info("No data matches filters.")
        return

    tpl = get_plotly_template(filters)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        mb = cp_f["bytes_transferred"] / 1_000_000
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=mb[~cp_f["is_anomaly"]],
                name="Normal",
                marker_color="#2ecc71",
                opacity=0.7,
                nbinsx=50,
            )
        )
        fig.add_trace(
            go.Histogram(
                x=mb[cp_f["is_anomaly"]],
                name="Anomalous",
                marker_color="#e74c3c",
                opacity=0.7,
                nbinsx=50,
            )
        )
        fig.update_layout(
            barmode="overlay",
            title="File Transfer Sizes (Normal vs Anomalous)",
            xaxis_title="Transfer Size (MB)",
            yaxis_title="Frequency",
            height=360,
            template=tpl,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("Total Crew Portal Events", f"{len(cp_f):,}")
        st.metric("Bulk Downloads", f"{int(cp_f['bulk'].sum()):,}")
        st.metric("Anomalous Events", f"{int(cp_f['is_anomaly'].sum()):,}")

    st.markdown("---")
    st.markdown("#### Bulk Downloads (record_count > 100 OR bytes > 500 MB)")

    bulk = cp_f[cp_f["bulk"]]
    if not bulk.empty:
        show = bulk[
            ["timestamp", "user", "role", "record_type", "record_count", "bytes_transferred"]
        ].copy()
        show["timestamp"] = pd.to_datetime(show["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        show["bytes_transferred"] = show["bytes_transferred"].apply(
            lambda b: f"{b/1_000_000:.1f} MB"
        )
        st.dataframe(show, use_container_width=True, height=250)
        st.download_button(
            "📥 Download bulk downloads",
            _to_csv_download(show),
            "crew_bulk_downloads.csv",
            "text/csv",
        )
    else:
        st.info("No bulk downloads found.")

    st.markdown("---")
    st.markdown("#### SecurityProtocol Access by Non‑Security Roles")

    sus = cp_f[
        (cp_f["record_type"] == "SecurityProtocol")
        & ~cp_f["role"].isin(SECURITY_ROLES)
    ]
    if not sus.empty:
        show = sus[["timestamp", "user", "role", "action", "record_count"]].copy()
        show["timestamp"] = pd.to_datetime(show["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(show, use_container_width=True, height=250)
    else:
        st.info("No unauthorised SecurityProtocol access.")


# ══════════════════════════════════════════════════════════════════════════════
#  Tab 5: Maintenance & Cargo
# ══════════════════════════════════════════════════════════════════════════════

def render_tab5_maintcargo(
    maint: pd.DataFrame, cargo: pd.DataFrame, emp: pd.DataFrame, filters: dict
) -> None:
    st.markdown("### 🔧 Maintenance & Cargo Tampering")

    tpl = get_plotly_template(filters)
    col1, col2 = st.columns([1, 1])

    # ── Column 1: Maintenance ─────────────────────────────────────────────
    with col1:
        st.subheader("Maintenance Tampering")
        maint = enrich_maint(maint)
        maint_f = filter_df(maint, filters, date_col="date", emp_col="technician_id")
        if maint_f.empty:
            st.info("No maintenance data.")
        else:
            tamper = maint_f[maint_f["is_tamper"]]
            if not tamper.empty:
                ctab = pd.crosstab(tamper["component"], tamper["action"])
                for a in TAMPER_ACTIONS:
                    if a not in ctab.columns:
                        ctab[a] = 0
                ctab = ctab[list(TAMPER_ACTIONS)]

                fig = px.bar(
                    ctab,
                    orientation="h",
                    barmode="stack",
                    title="Tampering Actions by Critical Component",
                    labels={"value": "Count", "index": "Component"},
                    color_discrete_sequence=["#e74c3c", "#c0392b", "#e67e22"],
                    template=tpl,
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

                show = tamper[
                    ["timestamp", "technician_id", "role", "action", "component"]
                ].copy()
                show["timestamp"] = pd.to_datetime(show["timestamp"]).dt.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                st.markdown("**Tampering Events**")
                st.dataframe(show, use_container_width=True, height=200)
            else:
                st.info("No tampering on critical components.")

    # ── Column 2: Cargo ───────────────────────────────────────────────────
    with col2:
        st.subheader("Cargo Weight Anomalies")
        cargo = enrich_cargo(cargo, emp)
        cargo_f = filter_df(cargo, filters, date_col="date", emp_col="handler_id")
        if cargo_f.empty:
            st.info("No cargo data.")
        else:
            st.metric("Total Cargo Events", f"{len(cargo_f):,}")
            st.metric("Weight Anomalies (delta > 20 kg)", f"{int(cargo_f['weight_anom'].sum()):,}")

            off_door = cargo_f[
                (cargo_f["action"] == "open_door")
                & (cargo_f["hour"].isin(OFF_HOURS_SET))
                & (~cargo_f["shift_scheduled"])
            ]
            st.metric("Off‑Hours Door Open (Unscheduled)", len(off_door))

            wm = cargo_f[cargo_f["weight_anom"]]
            if not wm.empty:
                show = wm[
                    ["timestamp", "handler_id", "role", "cargo_id", "prev_weight", "weight_kg", "delta"]
                ].copy()
                show["timestamp"] = pd.to_datetime(show["timestamp"]).dt.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                st.markdown("**Weight Mismatch Events**")
                st.dataframe(show, use_container_width=True, height=200)
                st.download_button(
                    "📥 Download cargo anomalies",
                    _to_csv_download(show),
                    "cargo_anomalies.csv",
                    "text/csv",
                )
            else:
                st.info("No weight mismatch events.")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # Load all data
    with st.spinner("Loading data …"):
        emp = load_employee_table()
        badge = load_badge_logs()
        vpn = load_vpn_logs()
        cp = load_crew_portal_logs()
        maint = load_maintenance_logs()
        cargo = load_cargo_logs()
        risk = load_risk_summary()

    # Sidebar
    filters = build_sidebar(risk, emp)

    # Auto-refresh happens because filters trigger a re-run natively in Streamlit

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Risk Overview",
        "🪪 Badge Access",
        "🔐 VPN & Network",
        "📁 Data Exfiltration",
        "🔧 Maint & Cargo",
    ])

    with tab1:
        render_tab1_risk(risk, filters)

    with tab2:
        render_tab2_badge(badge, emp, filters)

    with tab3:
        render_tab3_vpn(vpn, emp, filters)

    with tab4:
        render_tab4_exfil(cp, filters)

    with tab5:
        render_tab5_maintcargo(maint, cargo, emp, filters)


if __name__ == "__main__":
    main()
