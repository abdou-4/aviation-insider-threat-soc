"""
generate_logs.py — Aviation Insider Threat Detection SOC Synthetic Log Generator

Produces five CSV log files (plus an optional employee table) simulating 30 days
of airport operations.  Anomalies are injected at ~3-5 % for SOC detection
exercises.

Usage:
    python generate_logs.py
"""

from __future__ import annotations

import os
import random
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Configuration ───────────────────────────────────────────────────────────
START_DATE = "2025-05-01"
END_DATE = "2025-05-30"
OUTPUT_DIR = "synthetic_logs"
INCLUDE_ANOMALY_LABEL = True  # keep is_anomaly column in output CSVs

# Daily row targets (approximate) per log type
ROWS_PER_DAY = {
    "badge": 2500,
    "vpn": 800,
    "crew_portal": 1200,
    "maintenance": 600,
    "cargo": 900,
}

# ── Roles ───────────────────────────────────────────────────────────────────
ROLES = [
    "Pilot",
    "FlightAttendant",
    "RampAgent",
    "BaggageHandler",
    "Mechanic",
    "Fueler",
    "ITAdmin",
    "Caterer",
    "SecurityOfficer",
]

# ── Door locations and which roles can normally access them ────────────────
DOOR_LOCATIONS = [
    "CockpitDoor",
    "CargoArea",
    "MaintenanceHangar",
    "FuelFarm",
    "OpsCenter",
    "SecureGate",
    "BaggageCarousel",
    "DataCenter",
]

ROLE_DOOR_MAP: dict[str, list[str]] = {
    "Pilot":             ["CockpitDoor", "OpsCenter", "SecureGate"],
    "FlightAttendant":   ["CockpitDoor", "SecureGate", "BaggageCarousel"],
    "RampAgent":         ["CargoArea", "SecureGate", "BaggageCarousel"],
    "BaggageHandler":    ["CargoArea", "BaggageCarousel"],
    "Mechanic":          ["MaintenanceHangar", "CargoArea", "SecureGate", "OpsCenter"],
    "Fueler":            ["FuelFarm", "SecureGate"],
    "ITAdmin":           ["DataCenter", "OpsCenter", "SecureGate"],
    "Caterer":           ["SecureGate", "BaggageCarousel", "CargoArea"],
    "SecurityOfficer":   ["SecureGate", "OpsCenter", "CockpitDoor", "DataCenter",
                          "FuelFarm", "MaintenanceHangar", "CargoArea", "BaggageCarousel"],
}

# ── Countries ───────────────────────────────────────────────────────────────
ROLE_COUNTRY_MAP: dict[str, str] = {
    "Pilot":             "US",
    "FlightAttendant":   "US",
    "RampAgent":         "US",
    "BaggageHandler":    "US",
    "Mechanic":          "US",
    "Fueler":            "US",
    "ITAdmin":           "US",
    "Caterer":           "US",
    "SecurityOfficer":   "US",
}

ALL_COUNTRIES = [
    "US", "GB", "FR", "DE", "CA", "AU", "JP", "SG", "BR", "MX", "IT", "ES",
]
SUSPICIOUS_COUNTRIES = ["RU", "CN", "IR", "KP", "SY"]

# ── Suspicious IP prefixes (simulated Tor / VPN exit nodes) ────────────────
SUSPICIOUS_IPS = [
    "185.220.101.",
    "185.220.102.",
    "185.220.103.",
    "91.239.100.",
    "176.10.99.",
    "198.98.48.",
    "162.247.74.",
    "103.28.52.",
]

# ── Crew portal ─────────────────────────────────────────────────────────────
ACTIONS = ["view", "download", "upload", "search", "print"]
RECORD_TYPES = [
    "PassengerManifest", "FlightPlan", "CrewSchedule",
    "MaintenanceLog", "SecurityProtocol",
]

# Roles that are allowed to access SecurityProtocol
SECURITY_ROLES = {"SecurityOfficer", "ITAdmin", "Pilot"}

# ── Maintenance ─────────────────────────────────────────────────────────────
MAINT_ACTIONS = ["edit_log", "override_sensor", "disable_alarm", "run_test", "install_software"]
COMPONENTS = ["Navigation", "Engine", "FuelSystem", "Brakes", "Avionics", "FlightControls"]
CRITICAL_COMPONENTS = {"Engine", "FuelSystem", "FlightControls"}

# ── Cargo ───────────────────────────────────────────────────────────────────
CARGO_ACTIONS = ["open_door", "close_door", "edit_manifest", "scan", "load", "unload"]
CARGO_LOCATIONS = ["Gate_A1", "Gate_B2", "Gate_C3", "CargoHold_1", "CargoHold_2", "SortingFacility"]

# ── Aircraft IDs ────────────────────────────────────────────────────────────
AIRCRAFT_IDS = [f"N{random.randint(100, 999)}{random.choice('ABCDEFGH')}" for _ in range(30)]


# ══════════════════════════════════════════════════════════════════════════════
#  Helper utilities
# ══════════════════════════════════════════════════════════════════════════════

def random_ip() -> str:
    return f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"


def random_timestamp(start: datetime, end: datetime) -> datetime:
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def shift_biased_timestamp(
    emp_info: dict, start: datetime, end: datetime
) -> datetime:
    """Pick a random day, then pick an hour biased toward the employee's shift
    using a triangular distribution that peaks mid-shift and tails off toward
    off-hours.  This prevents ~33% of badge events from being falsely flagged
    as off-hours."""
    days_range = (end - start).days or 1
    random_day = start + timedelta(days=random.randint(0, days_range))
    sh = emp_info.get("shift_start", 8)
    eh = emp_info.get("shift_end", 18)
    # triangular: mode at mid-shift, lower bound 2 h before shift, upper at shift end
    low = max(0, sh - 2)
    high = min(23, eh + 1)
    mode = (sh + eh) / 2
    hour = int(np.random.triangular(low, mode, high))
    hour = max(0, min(23, hour))
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    try:
        return random_day.replace(hour=hour, minute=minute, second=second)
    except ValueError:
        return random_day.replace(hour=12, minute=minute, second=second)


def biased_hour(role: str, employees: dict[str, dict[str, Any]]) -> int:
    """Return an hour (0-23) weighted toward the employee's shift window."""
    if role in employees:
        sh = employees[role].get("shift_start", 8)
        eh = employees[role].get("shift_end", 18)
    else:
        sh, eh = 8, 20
    # triangular distribution peaks around mid-shift
    hour = int(np.random.triangular(sh, (sh + eh) / 2, eh))
    return max(0, min(23, hour))


def is_off_hours(hour: int, shift_start: int, shift_end: int) -> bool:
    """True if *hour* falls completely outside the shift window."""
    if shift_start <= shift_end:
        return hour < shift_start or hour >= shift_end
    else:  # overnight shift, e.g. 22:00-06:00
        return shift_end <= hour < shift_start


def build_employee_dict(emp_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Convert employee DataFrame to a dict keyed by employee_id for fast lookup."""
    d: dict[str, dict[str, Any]] = {}
    for _, row in emp_df.iterrows():
        d[row["employee_id"]] = {
            "role": row["role"],
            "shift_start": row["shift_start"],
            "shift_end": row["shift_end"],
            "termination_date": row.get("termination_date"),
            "typical_access_zones": row.get("typical_access_zones", []),
            "typical_country": row.get("typical_country", "US"),
        }
    return d


# ══════════════════════════════════════════════════════════════════════════════
#  Employee table
# ══════════════════════════════════════════════════════════════════════════════

def generate_employee_table() -> pd.DataFrame:
    """Create a synthetic employee roster used by all log generators."""
    names_pool = {
        "Pilot":             ["J. Smith", "A. Patel", "M. Chen", "K. Brown", "L. Garcia"],
        "FlightAttendant":   ["S. Lee", "R. Williams", "T. Davis", "E. Wilson", "N. Taylor"],
        "RampAgent":         ["D. Miller", "C. Anderson", "P. Thomas", "B. Jackson"],
        "BaggageHandler":    ["M. White", "J. Harris", "L. Martin", "F. Lewis"],
        "Mechanic":          ["R. Clark", "W. Hall", "T. Young", "H. King", "S. Wright"],
        "Fueler":            ["G. Lopez", "V. Adams", "J. Scott"],
        "ITAdmin":           ["P. Moore", "D. Turner", "Z. Collins"],
        "Caterer":           ["K. Stewart", "N. Sanchez", "O. Morris"],
        "SecurityOfficer":   ["X. Reed", "Y. Cook", "U. Bailey", "I. Rivera"],
    }

    records = []
    employee_id = 1
    for role in ROLES:
        for name in names_pool.get(role, ["Unknown"]):
            shift_start = random.choice([6, 6, 7, 8, 8, 8, 9, 14, 22])
            shift_end = shift_start + random.choice([8, 10, 12])
            if shift_end > 24:
                shift_end = 23

            # Some terminated employees (about 5%)
            termination: str | None = None
            if random.random() < 0.05:
                term_day = random.randint(5, 25)
                termination = f"2025-05-{term_day:02d}"

            eid = f"EMP_{employee_id:04d}"
            employee_id += 1
            records.append({
                "employee_id": eid,
                "name": name,
                "role": role,
                "shift_start": shift_start,
                "shift_end": shift_end,
                "termination_date": termination,
                "typical_access_zones": ",".join(ROLE_DOOR_MAP.get(role, [])),
                "typical_country": ROLE_COUNTRY_MAP.get(role, "US"),
            })
    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
#  Badge logs
# ══════════════════════════════════════════════════════════════════════════════

def generate_badge_logs(
    employees: pd.DataFrame,
    start: datetime,
    end: datetime,
    rows_per_day: int,
) -> pd.DataFrame:
    """Generate physical access control logs with injected anomalies."""
    emp_dict = build_employee_dict(employees)
    num_days = (end - start).days or 1
    total_rows = num_days * rows_per_day

    chosen = employees.sample(n=total_rows, replace=True, random_state=SEED).reset_index(drop=True)
    records: list[dict[str, Any]] = []

    for _, emp in chosen.iterrows():
        role = emp["role"]
        info = emp_dict.get(emp["employee_id"], {})
        ts = shift_biased_timestamp(info, start, end)
        sh = info.get("shift_start", 8)
        eh = info.get("shift_end", 20)
        term_date = info.get("termination_date")

        # Determine door — mostly normal, sometimes anomalous
        normal_doors = ROLE_DOOR_MAP.get(role, ["SecureGate"])
        door = random.choice(normal_doors)

        granted = True
        anomaly = False
        anomaly_type: str | None = None

        # 1) Off-hours access
        hr = ts.hour
        if is_off_hours(hr, sh, eh) and random.random() < 0.15:
            anomaly = True
            anomaly_type = "off_hours_access"
            # still grant access (tailgating / badge sharing scenario)

        # 2) Wrong zone
        wrong_doors = [d for d in DOOR_LOCATIONS if d not in normal_doors]
        if not anomaly and wrong_doors and random.random() < 0.04:
            door = random.choice(wrong_doors)
            anomaly = True
            anomaly_type = "wrong_zone"

        # 3) Terminated employee accessing after termination
        if not anomaly and term_date is not None and not (isinstance(term_date, float) and np.isnan(term_date)):
            term_dt = datetime.strptime(str(term_date), "%Y-%m-%d")
            if ts > term_dt:
                anomaly = True
                anomaly_type = "terminated_access"
                # often denied
                if random.random() < 0.7:
                    granted = False

        # 4) Rapid tailgating simulation — handled by creating an extra row
        # (We'll inject clusters below)

        # Cap anomaly rate ~5%
        if not anomaly and random.random() < 0.01:
            anomaly = True
            anomaly_type = "random_anomaly"

        records.append({
            "timestamp": ts.isoformat(),
            "employee_id": emp["employee_id"],
            "role": role,
            "door_location": door,
            "granted": granted,
            "is_anomaly": anomaly,
            "anomaly_type": anomaly_type or "",
        })

    df = pd.DataFrame(records)

    # Inject rapid tailgating clusters (multiple high-security doors in <5 min)
    tailgate_group: list[dict[str, Any]] = []
    for i in range(30):
        base_idx = random.randint(0, len(df) - 1)
        base_row = df.iloc[base_idx]
        base_ts = datetime.fromisoformat(base_row["timestamp"])
        emp_id = base_row["employee_id"]
        # Find a terminated or random employee for the tailgate
        if random.random() < 0.5:
            tg_emp = employees[employees["employee_id"] != emp_id].sample(1).iloc[0]
        else:
            tg_emp = employees.sample(1).iloc[0]

        for j in range(random.randint(2, 4)):
            door = random.choice(["CockpitDoor", "DataCenter", "SecureGate", "OpsCenter"])
            ts_tail = base_ts + timedelta(seconds=random.randint(10, 290))
            if ts_tail > end:
                continue
            tailgate_group.append({
                "timestamp": ts_tail.isoformat(),
                "employee_id": tg_emp["employee_id"],
                "role": tg_emp["role"],
                "door_location": door,
                "granted": True,
                "is_anomaly": True,
                "anomaly_type": "tailgating",
            })

    if tailgate_group:
        df = pd.concat([df, pd.DataFrame(tailgate_group)], ignore_index=True)

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  VPN logs
# ══════════════════════════════════════════════════════════════════════════════

def generate_vpn_logs(
    employees: pd.DataFrame,
    start: datetime,
    end: datetime,
    rows_per_day: int,
) -> pd.DataFrame:
    """Generate VPN authentication logs with injected anomalies."""
    emp_dict = build_employee_dict(employees)
    num_days = (end - start).days or 1
    total_rows = num_days * rows_per_day

    chosen = employees.sample(n=total_rows, replace=True, random_state=SEED + 1).reset_index(drop=True)
    records: list[dict[str, Any]] = []

    for _, emp in chosen.iterrows():
        ts = random_timestamp(start, end)
        role = emp["role"]
        info = emp_dict.get(emp["employee_id"], {})
        sh = info.get("shift_start", 8)
        eh = info.get("shift_end", 20)
        typical_country = info.get("typical_country", "US")

        country = typical_country
        ip = random_ip()
        auth_result = "success"
        anomaly = False
        anomaly_type: str | None = None

        # 1) Suspicious country
        if random.random() < 0.035:
            country = random.choice(SUSPICIOUS_COUNTRIES)
            ip = random_ip()
            anomaly = True
            anomaly_type = "foreign_country"
            auth_result = "failure" if random.random() < 0.4 else "success"

        # 2) Off-hours login
        hr = ts.hour
        if not anomaly and is_off_hours(hr, sh, eh) and random.random() < 0.08:
            anomaly = True
            anomaly_type = "off_hours_login"

        # 3) Suspicious IP (Tor / VPN)
        if not anomaly and random.random() < 0.02:
            prefix = random.choice(SUSPICIOUS_IPS)
            ip = f"{prefix}{random.randint(1, 254)}"
            country = random.choice(["US", "DE", "NL", "RU"])
            anomaly = True
            anomaly_type = "suspicious_ip"
            auth_result = random.choice(["success", "success", "failure"])

        records.append({
            "timestamp": ts.isoformat(),
            "username": emp["employee_id"],
            "role": role,
            "src_ip": ip,
            "auth_result": auth_result,
            "geo_country": country,
            "is_anomaly": anomaly,
            "anomaly_type": anomaly_type or "",
        })

    df = pd.DataFrame(records)

    # Inject brute-force clusters: >3 failures in 10 minutes
    bf_records: list[dict[str, Any]] = []
    for i in range(25):
        base_idx = random.randint(0, len(df) - 1)
        base_row = df.iloc[base_idx]
        base_ts = datetime.fromisoformat(base_row["timestamp"])
        emp = employees.sample(1).iloc[0]
        ip = random_ip()
        n_attempts = random.randint(4, 8)
        for j in range(n_attempts):
            ts_bf = base_ts + timedelta(seconds=random.randint(0, 599))
            if ts_bf > end:
                continue
            bf_records.append({
                "timestamp": ts_bf.isoformat(),
                "username": emp["employee_id"],
                "role": emp["role"],
                "src_ip": ip,
                "auth_result": "failure",
                "geo_country": "US",
                "is_anomaly": True,
                "anomaly_type": "brute_force",
            })
    if bf_records:
        df = pd.concat([df, pd.DataFrame(bf_records)], ignore_index=True)

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Crew portal logs
# ══════════════════════════════════════════════════════════════════════════════

def generate_crew_portal_logs(
    employees: pd.DataFrame,
    start: datetime,
    end: datetime,
    rows_per_day: int,
) -> pd.DataFrame:
    """Generate crew portal / flight ops activity logs with anomalies."""
    emp_dict = build_employee_dict(employees)
    num_days = (end - start).days or 1
    total_rows = num_days * rows_per_day

    chosen = employees.sample(n=total_rows, replace=True, random_state=SEED + 2).reset_index(drop=True)
    records: list[dict[str, Any]] = []

    for _, emp in chosen.iterrows():
        ts = random_timestamp(start, end)
        role = emp["role"]
        info = emp_dict.get(emp["employee_id"], {})
        sh = info.get("shift_start", 8)
        eh = info.get("shift_end", 20)

        action = random.choices(
            ACTIONS,
            weights=[50, 15, 5, 25, 5],
            k=1,
        )[0]

        record_type = random.choice(RECORD_TYPES)
        record_count = int(np.random.exponential(10)) + 1
        bytes_xfer = record_count * random.randint(5000, 500000)

        anomaly = False
        anomaly_type: str | None = None

        # 1) Bulk download
        if action == "download" and record_type == "PassengerManifest" and record_count > 100:
            anomaly = True
            anomaly_type = "bulk_download"
        if not anomaly and action == "download" and bytes_xfer > 500_000_000:
            anomaly = True
            anomaly_type = "bulk_download"

        # 2) Unauthorized SecurityProtocol access
        if not anomaly and record_type == "SecurityProtocol" and role not in SECURITY_ROLES:
            if random.random() < 0.15:
                anomaly = True
                anomaly_type = "unauthorized_security_access"
                record_count = random.randint(1, 5)
                bytes_xfer = record_count * random.randint(10000, 100000)

        # 3) Off-hours access
        hr = ts.hour
        if not anomaly and is_off_hours(hr, sh, eh) and random.random() < 0.07:
            anomaly = True
            anomaly_type = "off_hours_access"

        # 4) Trigger bulk anomaly deterministically sometimes
        if not anomaly and action == "download" and random.random() < 0.01:
            record_count = random.randint(150, 300)
            bytes_xfer = record_count * random.randint(300000, 600000)
            anomaly = True
            anomaly_type = "bulk_download"

        records.append({
            "timestamp": ts.isoformat(),
            "user": emp["employee_id"],
            "role": role,
            "action": action,
            "record_type": record_type,
            "record_count": record_count,
            "bytes_transferred": bytes_xfer,
            "is_anomaly": anomaly,
            "anomaly_type": anomaly_type or "",
        })

    df = pd.DataFrame(records)

    # Inject rapid sensitive-data download clusters
    cluster_records: list[dict[str, Any]] = []
    for i in range(20):
        base_idx = random.randint(0, len(df) - 1)
        base_ts = datetime.fromisoformat(df.iloc[base_idx]["timestamp"])
        emp = employees.sample(1).iloc[0]
        for j in range(random.randint(3, 6)):
            ts_c = base_ts + timedelta(minutes=random.randint(1, 25))
            if ts_c > end:
                continue
            cluster_records.append({
                "timestamp": ts_c.isoformat(),
                "user": emp["employee_id"],
                "role": emp["role"],
                "action": "download",
                "record_type": random.choice(["PassengerManifest", "SecurityProtocol"]),
                "record_count": random.randint(50, 200),
                "bytes_transferred": random.randint(50_000_000, 300_000_000),
                "is_anomaly": True,
                "anomaly_type": "rapid_sensitive_downloads",
            })

    if cluster_records:
        df = pd.concat([df, pd.DataFrame(cluster_records)], ignore_index=True)

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Maintenance logs
# ══════════════════════════════════════════════════════════════════════════════

def generate_maintenance_logs(
    employees: pd.DataFrame,
    start: datetime,
    end: datetime,
    rows_per_day: int,
) -> pd.DataFrame:
    """Generate aircraft maintenance system logs with anomalies."""
    emp_dict = build_employee_dict(employees)
    # Only Mechanics and ITAdmin handle maintenance
    maint_roles = employees[employees["role"].isin(["Mechanic", "ITAdmin"])].copy()
    if maint_roles.empty:
        maint_roles = employees.copy()

    num_days = (end - start).days or 1
    total_rows = num_days * rows_per_day

    chosen = maint_roles.sample(n=total_rows, replace=True, random_state=SEED + 3).reset_index(drop=True)
    records: list[dict[str, Any]] = []

    for _, emp in chosen.iterrows():
        ts = random_timestamp(start, end)
        role = emp["role"]
        info = emp_dict.get(emp["employee_id"], {})
        sh = info.get("shift_start", 8)
        eh = info.get("shift_end", 20)

        action = random.choices(
            MAINT_ACTIONS,
            weights=[10, 5, 3, 60, 5],  # mostly run_test
            k=1,
        )[0]
        component = random.choice(COMPONENTS)
        status = random.choices(
            ["passed", "passed", "passed", "failed", "warning"],
            weights=[40, 30, 20, 8, 2],
            k=1,
        )[0]
        work_order_valid = True
        source = "system"

        anomaly = False
        anomaly_type: str | None = None

        # 1) Override sensor / disable alarm without valid work order
        if action in ("override_sensor", "disable_alarm") and random.random() < 0.15:
            work_order_valid = False
            anomaly = True
            anomaly_type = "invalid_work_order"

        # 2) Tampering with critical components outside maintenance window
        hr = ts.hour
        if not anomaly and component in CRITICAL_COMPONENTS and is_off_hours(hr, sh, eh):
            if random.random() < 0.12:
                anomaly = True
                anomaly_type = "off_hours_critical_tamper"

        # 3) Edit log to remove critical finding (change failed → passed)
        if not anomaly and action == "edit_log" and status == "passed":
            if random.random() < 0.08:
                anomaly = True
                anomaly_type = "log_tamper_coverup"
                # mark that the original was failed
                status = "passed (was failed)"

        # 4) Software install from untrusted USB
        if not anomaly and action == "install_software" and random.random() < 0.10:
            anomaly = True
            anomaly_type = "untrusted_usb_install"
            source = "USB"

        # 5) Random anomaly boost
        if not anomaly and random.random() < 0.005:
            anomaly = True
            anomaly_type = "random_anomaly"

        records.append({
            "timestamp": ts.isoformat(),
            "technician_id": emp["employee_id"],
            "role": role,
            "aircraft_id": random.choice(AIRCRAFT_IDS),
            "action": action,
            "component": component,
            "status": status,
            "work_order_valid": work_order_valid,
            "source": source,
            "is_anomaly": anomaly,
            "anomaly_type": anomaly_type or "",
        })

    df = pd.DataFrame(records)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Cargo logs
# ══════════════════════════════════════════════════════════════════════════════

def generate_cargo_logs(
    employees: pd.DataFrame,
    start: datetime,
    end: datetime,
    rows_per_day: int,
) -> pd.DataFrame:
    """Generate baggage and cargo handling logs with anomalies."""
    emp_dict = build_employee_dict(employees)
    cargo_roles = employees[employees["role"].isin(["RampAgent", "BaggageHandler", "Fueler", "Caterer"])].copy()
    if cargo_roles.empty:
        cargo_roles = employees.copy()

    num_days = (end - start).days or 1
    total_rows = num_days * rows_per_day

    chosen = cargo_roles.sample(n=total_rows, replace=True, random_state=SEED + 4).reset_index(drop=True)
    records: list[dict[str, Any]] = []

    for _, emp in chosen.iterrows():
        ts = random_timestamp(start, end)
        role = emp["role"]
        info = emp_dict.get(emp["employee_id"], {})
        sh = info.get("shift_start", 8)
        eh = info.get("shift_end", 20)

        action = random.choices(
            CARGO_ACTIONS,
            weights=[5, 5, 8, 40, 30, 12],
            k=1,
        )[0]
        weight = round(random.uniform(5, 50), 1)
        location = random.choice(CARGO_LOCATIONS)
        cargo_id = f"CARGO_{random.randint(10000, 99999)}"

        anomaly = False
        anomaly_type: str | None = None

        # 1) Weight mismatch on edit_manifest
        if action == "edit_manifest":
            if random.random() < 0.08:
                weight = round(max(0, weight - random.uniform(20, 60)), 1)
                anomaly = True
                anomaly_type = "weight_mismatch"

        # 2) Cargo door opened off-hours by handler not on shift
        hr = ts.hour
        if not anomaly and action == "open_door" and is_off_hours(hr, sh, eh):
            if random.random() < 0.12:
                anomaly = True
                anomaly_type = "off_hours_door_open"

        # 3) Random anomaly
        if not anomaly and random.random() < 0.01:
            anomaly = True
            anomaly_type = "random_anomaly"

        records.append({
            "timestamp": ts.isoformat(),
            "handler_id": emp["employee_id"],
            "role": role,
            "cargo_id": cargo_id,
            "action": action,
            "weight_kg": weight,
            "location": location,
            "is_anomaly": anomaly,
            "anomaly_type": anomaly_type or "",
        })

    df = pd.DataFrame(records)

    # Inject rapid cargo edit clusters (possible theft cover-up)
    cluster: list[dict[str, Any]] = []
    for i in range(15):
        base_idx = random.randint(0, len(df) - 1)
        base_ts = datetime.fromisoformat(df.iloc[base_idx]["timestamp"])
        emp = employees.sample(1).iloc[0]
        for j in range(random.randint(3, 6)):
            ts_c = base_ts + timedelta(minutes=random.randint(1, 10))
            if ts_c > end:
                continue
            cluster.append({
                "timestamp": ts_c.isoformat(),
                "handler_id": emp["employee_id"],
                "role": emp["role"],
                "cargo_id": f"CARGO_{random.randint(10000, 99999)}",
                "action": "edit_manifest",
                "weight_kg": round(random.uniform(0, 15), 1),
                "location": random.choice(CARGO_LOCATIONS),
                "is_anomaly": True,
                "anomaly_type": "rapid_cargo_edits",
            })

    if cluster:
        df = pd.concat([df, pd.DataFrame(cluster)], ignore_index=True)

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Summary
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(logs: dict[str, pd.DataFrame]) -> None:
    """Print a summary table of generated logs to stdout."""
    print("=" * 72)
    print("  SYNTHETIC LOG GENERATION — SUMMARY")
    print("=" * 72)

    for name, df in logs.items():
        n_anom = df["is_anomaly"].sum() if "is_anomaly" in df else 0
        pct = (n_anom / max(len(df), 1)) * 100
        print(f"\n  {name}:")
        print(f"    Total rows:      {len(df):,}")
        print(f"    Anomalies:       {n_anom:,}  ({pct:.1f}%)")
        if "anomaly_type" in df.columns:
            top = (
                df[df["anomaly_type"] != ""]["anomaly_type"]
                .value_counts()
                .head(5)
            )
            if not top.empty:
                print("    Top anomaly types:")
                for atype, cnt in top.items():
                    print(f"      - {atype}: {cnt}")

    print("\n" + "=" * 72)
    print("  Files written to:", OUTPUT_DIR)
    print("=" * 72)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_dt = datetime.strptime(END_DATE, "%Y-%m-%d") + timedelta(days=1) - timedelta(seconds=1)

    print("Generating employee table...")
    employees = generate_employee_table()
    emp_path = os.path.join(OUTPUT_DIR, "employee_table.csv")
    employees.to_csv(emp_path, index=False)
    print(f"  -> {emp_path}  ({len(employees)} employees)")

    generators = {
        "badge_logs.csv": lambda: generate_badge_logs(
            employees, start_dt, end_dt, ROWS_PER_DAY["badge"],
        ),
        "vpn_logs.csv": lambda: generate_vpn_logs(
            employees, start_dt, end_dt, ROWS_PER_DAY["vpn"],
        ),
        "crew_portal_logs.csv": lambda: generate_crew_portal_logs(
            employees, start_dt, end_dt, ROWS_PER_DAY["crew_portal"],
        ),
        "maintenance_logs.csv": lambda: generate_maintenance_logs(
            employees, start_dt, end_dt, ROWS_PER_DAY["maintenance"],
        ),
        "cargo_logs.csv": lambda: generate_cargo_logs(
            employees, start_dt, end_dt, ROWS_PER_DAY["cargo"],
        ),
    }

    results: dict[str, pd.DataFrame] = {"employee_table.csv": employees}

    for fname, gen_func in generators.items():
        print(f"Generating {fname} ...")
        df = gen_func()
        path = os.path.join(OUTPUT_DIR, fname)
        cols_to_write = [c for c in df.columns if c != "anomaly_type"]
        if not INCLUDE_ANOMALY_LABEL:
            cols_to_write = [c for c in cols_to_write if c != "is_anomaly"]
        df[cols_to_write].to_csv(path, index=False)
        print(f"  -> {path}  ({len(df):,} rows)")
        results[fname] = df

    print_summary(results)


if __name__ == "__main__":
    main()
