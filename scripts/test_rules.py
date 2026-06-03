"""
test_rules.py — Emulate Sigma rule matches against synthetic CSV logs.

Usage:
    python test_rules.py
"""
import pandas as pd

DATA_DIR = "synthetic_logs"


def test_off_hours_badge():
    """Rule 1: Off-hours sensitive area access."""
    badge = pd.read_csv(f"{DATA_DIR}/badge_logs.csv")
    emp = pd.read_csv(f"{DATA_DIR}/employee_table.csv")
    merged = badge.merge(emp, on="employee_id", how="left")
    merged["hour_of_day"] = pd.to_datetime(merged["timestamp"]).dt.hour

    def shift_scheduled(row):
        if pd.isna(row["shift_start"]) or pd.isna(row["shift_end"]):
            return True
        sh, se = int(row["shift_start"]), int(row["shift_end"])
        h = int(row["hour_of_day"])
        if sh <= se:
            return not (h < sh or h >= se)
        else:
            return not (se <= h < sh)

    merged["shift_scheduled"] = merged.apply(shift_scheduled, axis=1)
    off_hours = merged["hour_of_day"].isin([22, 23, 0, 1, 2, 3, 4, 5])
    sensitive = merged["door_location"].isin(["CargoArea", "DataCenter", "MaintenanceHangar"])
    unscheduled = ~merged["shift_scheduled"]
    alerts = merged[off_hours & sensitive & unscheduled]
    print(f"[BADGE Rule 1] Off-hours sensitive access: {len(alerts)} alerts")
    return alerts


def test_unusual_country_vpn():
    """Rule 2: VPN from unusual / high-risk country."""
    vpn = pd.read_csv(f"{DATA_DIR}/vpn_logs.csv")
    emp = pd.read_csv(f"{DATA_DIR}/employee_table.csv")
    merged = vpn.merge(emp, left_on="username", right_on="employee_id", how="left")
    high_risk = merged["geo_country"].isin(["RU", "CN", "IR", "KP", "SY"])
    mismatch = merged["geo_country"] != merged["typical_country"]
    alerts = merged[high_risk | mismatch]
    print(f"[VPN Rule 2] Unusual country login: {len(alerts)} alerts")
    return alerts


def test_bulk_exfil():
    """Rule 3: Bulk download of sensitive data."""
    cp = pd.read_csv(f"{DATA_DIR}/crew_portal_logs.csv")
    download = cp["action"] == "download"
    bulk = (cp["record_count"] > 100) | (cp["bytes_transferred"] >= 500_000_000)
    sensitive = cp["record_type"].isin(["PassengerManifest", "SecurityProtocol"])
    alerts = cp[download & bulk & sensitive]
    print(f"[CREW PORTAL Rule 3] Bulk exfil: {len(alerts)} alerts")
    return alerts


def test_maintenance_tamper():
    """Rule 4: Maintenance tampering on critical components."""
    maint = pd.read_csv(f"{DATA_DIR}/maintenance_logs.csv")
    tamper_actions = maint["action"].isin(["override_sensor", "disable_alarm", "edit_log"])
    critical = maint["component"].isin(["Engine", "FuelSystem", "FlightControls"])
    no_wr = ~maint["work_order_valid"].astype(bool)
    alerts = maint[tamper_actions & critical & no_wr]
    print(f"[MAINTENANCE Rule 4] Critical tamper (no work order): {len(alerts)} alerts")
    return alerts


def test_cargo_weight():
    """Rule 5: Cargo weight anomaly."""
    cargo = pd.read_csv(f"{DATA_DIR}/cargo_logs.csv")
    emp = pd.read_csv(f"{DATA_DIR}/employee_table.csv")
    merged = cargo.merge(emp, left_on="handler_id", right_on="employee_id", how="left")
    merged["hour_of_day"] = pd.to_datetime(merged["timestamp"]).dt.hour
    edit = merged["action"] == "edit_manifest"
    light = merged["weight_kg"] <= 5
    off_hours = merged["hour_of_day"].isin([22, 23, 0, 1, 2, 3, 4, 5])
    role_col = "role" if "role" in merged.columns else "role_x"
    unauth_role = merged[role_col].isin(["BaggageHandler", "Fueler", "Caterer"])
    alerts = merged[edit & (light | (off_hours & unauth_role))]
    print(f"[CARGO Rule 5] Weight anomaly: {len(alerts)} alerts")
    return alerts


def main():
    print("=" * 60)
    print("  Sigma Rule Emulation Tests")
    print("=" * 60)
    results = {}
    results["badge"] = test_off_hours_badge()
    results["vpn"] = test_unusual_country_vpn()
    results["crew"] = test_bulk_exfil()
    results["maint"] = test_maintenance_tamper()
    results["cargo"] = test_cargo_weight()
    total = sum(len(v) for v in results.values())
    print(f"\nTotal alerts across all 5 rules: {total}")
    print("=" * 60)


if __name__ == "__main__":
    main()
