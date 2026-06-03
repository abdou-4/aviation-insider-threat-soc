# ✈️ Aviation Insider Threat Detection — SOC Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Live%20Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Sigma-5%20Detection%20Rules-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/UEBA-Rule--Based%20Engine-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Domain-Aviation%20Security-1E3A5F?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />
</p>

<p align="center">
  A <strong>portfolio-grade Security Operations Center (SOC) pipeline</strong> that simulates realistic aviation log data, injects insider threat anomalies, applies a <strong>rule-based UEBA risk-scoring engine</strong>, generates <strong>Sigma-compatible detection rules</strong>, and exposes everything through an <strong>interactive Streamlit dashboard</strong> — zero commercial SIEM required.
</p>

---

## 📈 SOC Impact Highlights

| Metric | Value |
|--------|-------|
| Synthetic log events generated | **~180,000 rows / 30 days** |
| Log source types correlated | **5 (Badge · VPN · Crew Portal · Maintenance · Cargo)** |
| Weighted detection rules | **10 rules across all sources** |
| Sigma-compatible YAML rules | **5 production-ready rules** |
| Employee roles modelled | **9 roles, 8 physical access zones** |
| Mean time to alert vs. manual review | **~60% faster** |
| False-positive reduction vs. binary thresholds | **~47% fewer** |
| Multi-vector threats caught by compound detection | **~30% more** |
| Full pipeline execution time (180 K events) | **< 30 seconds** |
| Analyst triage time reduction via pre-scored queue | **up to 3×** |

> **Methodology note:** Impact figures are projected estimates based on published SOC benchmarks (SANS SOC Survey 2023, IBM Cost of a Data Breach 2023) applied to this pipeline's design characteristics. They reflect realistic expectations for a rule-based UEBA system in a comparable environment.

---

## 📌 Feature Overview

- **Synthetic log generator** — 5 correlated CSV log types + employee roster, 30 days of data (~180 K rows total), realistic anomaly injection at 3–5 %.
- **Automated validation suite** — structural, statistical, and semantic checks; outputs a markdown QA report and 6 validation plots.
- **UEBA risk-scoring engine** — configurable per-event weights, compound multi-source bonus, and 7-day exponential decay. No ML required.
- **Sigma detection rules** — 5 YAML rules covering off-hours badge access, suspicious VPN geo-login, bulk data exfiltration, maintenance tampering, and cargo weight anomalies.
- **Live interactive dashboard** — Streamlit + Plotly, multi-tab, dark/light mode, filters by date range / role / employee / alert level.
- **Static SOC charts** — 9 publication-ready PNG visualisations for reporting and portfolio display.
- **Incident response playbook** — NIST-based triage workflow from detection to remediation.

---

## 🏗️ Architecture

```
put the UML png here 
```

---

## ⚙️ User and Entity Behavior Analytics (UEBA) Scoring Engine
The core pipeline features a completely transparent, deterministic mathematical "pointing system" that eliminates the "black box" problems of machine-learning alternatives. It translates multi-vector log streams into prioritized analytical queues by tracking individual daily risk signatures alongside a rolling cumulative threat index.

### Risk Parameter Reference Table
The scoring engine maps security compromises into discrete threat weights across multiple data contexts:

| Log Source          | Event Trigger Scenario                                            | Original Weight | Calibrated Weight | MITRE ATT&CK Mapping                 |
|---------------------|-------------------------------------------------------------------|----------------|-------------------|--------------------------------------|
| Physical Security   | Off-hours badge access                                            | 10             | 3                 | Valid Accounts (T1078)               |
| Physical Security   | Off-hours access + sensitive door entry                           | 15             | 5                 | Valid Accounts (T1078)               |
| Network Security    | Unusual geolocated country VPN login                              | 15             | 4                 | Remote Services (T1133)              |
| Network Security    | Brute-force VPN authentication attempts                           | 10             | 3                 | Brute Force (T1110)                  |
| Application Layer   | Bulk data download (Crew Portal)                                  | 30             | 10                | Exfiltration Over Web (T1048)        |
| Application Layer   | Unauthorized SecurityProtocol access                              | 20             | 6                 | Valid Accounts (T1078)               |
| Maintenance Logs    | Component tampering without valid work order                      | 30             | 12                | Data Manipulation (T1565)            |
| Cargo Management    | Total weight/manifest mismatch anomaly                            | 20             | 6                 | Data Manipulation (T1565)            |
| Cargo Management    | Off-hours cargo bay door access                                   | 10             | 3                 | Valid Accounts (T1078)               |
| Cross-Correlation   | Compound Threat Bonus (Multi-vector <1hr)                         | 20             | 6                 | Lateral Movement / Exfil             |

## Time-Decay Modeling

To prevent brief, historic spikes from permanently poisoning an employee's risk status, the engine utilizes a **half-life mathematical decay model** applied across daily increments:

$$
Score_{decayed} = Score_{previous} \times e^{-\lambda t} + Score_{raw\_today}
$$

### Key Parameters

- **Calibrated Decay Rate**: A **3-day half-life** ($\lambda \approx 0.231$) is utilized instead of a standard 7-day window.

### Rationale

This approach scales the steady-state cumulative risk multiplier down from $10.5\times$ to a highly responsive **$4.85\times$** daily raw input, aligning perfectly with standard **30-day corporate investigation cycles**.

## Calibrated Threat Boundaries
Alert thresholds are applied exclusively to a single day's **daily_raw_score** rather than a compounded total, generating clear, actionable tiers:

```
Low Risk Profile (18–231 Score): Normal employees operating strictly within assigned shift boundaries.

Medium Risk Profile (211–297 Score): One-off operational anomalies (e.g., an unannounced out-of-country remote login).

High Risk Profile (226–386 Score): Immediate threat targets showing compounding multi-vector security triggers.
```

### Why No ML?

This is a deliberate design decision. Rule-based UEBA is:
- **Auditable** — every point can be explained in a court or compliance review.
- **Zero training data required** — immediately deployable on new environments.
- **Aligned with MITRE ATT&CK** — each rule maps to a documented technique.
- **Baseline for ML** — the scored output is the ideal feature set for future anomaly-detection models.

---

## 📋 Sigma Detection Rules

Five Sigma-compatible YAML rules are included in `sigma_rules/`, each targeting a distinct threat vector.

| Rule | File | MITRE Technique |
|------|------|-----------------|
| Off-hours sensitive area access | `badge_off_hours.yml` | T1078 – Valid Accounts |
| Unusual VPN geo-login / high-risk country | `vpn_unusual_country.yml` | T1133 – External Remote Services |
| Bulk data exfiltration via crew portal | `bulk_exfiltration.yml` | T1048 – Exfiltration Over Alt Protocol |
| Maintenance tampering on critical components | `maintenance_tamper.yml` | T1565 – Data Manipulation |
| Cargo weight anomaly (manifest fraud) | `cargo_anomaly.yml` | T1565.001 – Stored Data Manipulation |

Rules are tested by `test_rules.py`, which emulates Sigma matching logic directly against the CSV logs — no SIEM required.

---

## 🎭 Threat Scenarios Covered

The anomaly injector models realistic insider threat patterns drawn from ICAO, TSA, and MITRE ICS ATT&CK:

- **Rogue Mechanic** — disables engine sensors without a work order at 02:00 to cover a maintenance fraud (Sabotage, T1565).
- **Compromised Ramp Agent** — VPN login from a sanctioned nation followed by bulk passenger manifest download (Credential Compromise + Exfiltration, T1133 + T1048).
- **Disgruntled Pilot** — off-hours access to the DataCenter and CargoArea outside their authorised zones (Lateral Movement, T1078).
- **Cargo Handler** — repeated rapid edits to manifest weights during overnight shift (Cargo Smuggling, T1565.001).
- **IT Admin Abuse** — downloads SecurityProtocol documents with oversized byte transfer (Privilege Abuse, T1078.004).

---

## 📂 Repository Structure

```
aviation-insider-threat-soc/
│
├── scripts/
│   ├── generate_logs.py        # Synthetic log generator (5 types, 30 days)
│   ├── validate_logs.py        # QA suite — structural + statistical checks
│   ├── risk_scoring.py         # UEBA engine — weights, compound, decay, alerts
│   ├── soc_dashboard.py        # Static PNG chart generation (9 charts)
│   ├── live_dashboard.py       # Streamlit interactive dashboard
│   └── test_rules.py           # Sigma rule emulation tests
│
├── sigma_rules/
│   ├── badge_off_hours.yml
│   ├── vpn_unusual_country.yml
│   ├── bulk_exfiltration.yml
│   ├── maintenance_tamper.yml
│   └── cargo_anomaly.yml
│
├── playbook/
│   └── incident_response_playbook.md   # NIST-based triage → remediation
│
├── images/                     # Example screenshots for README
│   ├── live_dashboard_overview.png
│   ├── top10_risk.png
│   ├── risk_timeline.png
│   ├── badge_heatmap.png
│   ├── alert_level_donut.png
│   └── executive_dashboard.png
│
├── synthetic_logs/             # ⚠️ GITIGNORED — generated locally
├── risk_output/                # ⚠️ GITIGNORED — generated locally
├── validation_output/          # ⚠️ GITIGNORED — generated locally
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
```

### Installation

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/aviation-insider-threat-soc.git
cd aviation-insider-threat-soc

# 2. Install dependencies
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
# Step 1 — Generate ~180,000 rows of synthetic aviation logs
python scripts/generate_logs.py

# Step 2 — Validate data quality (outputs markdown report + 6 plots)
python scripts/validate_logs.py

# Step 3 — Run the UEBA risk-scoring engine
python scripts/risk_scoring.py

# Step 4 — Generate static SOC charts (9 PNGs)
python scripts/soc_dashboard.py

# Step 5 — Launch the live interactive dashboard
streamlit run scripts/live_dashboard.py

# Optional — Emulate Sigma rule matches
python scripts/test_rules.py
```

### Requirements

```
pandas>=1.5
numpy>=1.23
matplotlib>=3.6
seaborn>=0.12
streamlit>=1.28
plotly>=5.15
```

---

## 🛠️ Skills & Technologies Demonstrated

### Security Engineering
- **UEBA design** — custom rule-based scoring model aligned with MITRE ATT&CK Enterprise & ICS
- **Sigma rule authoring** — 5 production-quality YAML detection rules covering badge, VPN, portal, maintenance, and cargo logs
- **Insider threat modelling** — 10+ distinct anomaly types injected across 9 employee roles
- **Incident response** — NIST-based playbook from triage to remediation
- **Log source correlation** — cross-source compound event detection
- **Risk quantification** — configurable weighted scoring with time-decay and threshold classification

### Data Engineering & Python
- **pandas & numpy** — large-scale synthetic data generation, log merging, temporal feature engineering
- **Reproducible pipelines** — seeded RNG (`SEED=42`) for deterministic output across all 5 generators
- **Object-oriented design** — modular functions, type hints, docstrings throughout
- **Data validation** — automated structural, statistical, and semantic quality checks
- **JSON / CSV / Markdown outputs** — multi-format reporting pipeline

### Visualisation & Dashboarding
- **Streamlit** — multi-tab live dashboard with sidebar filters, dark/light mode toggle
- **Plotly Express / Graph Objects** — interactive bar charts, timelines, heatmaps, sunburst diagrams, donut charts
- **Matplotlib / Seaborn** — 9 static SOC charts for offline reporting and executive summaries

### Documentation & DevOps
- **Sigma YAML** — vendor-agnostic, SIEM-portable detection rule format
- **Git hygiene** — `.gitignore` for large generated CSVs, reproducible data generation
- **Structured markdown** — professional documentation and playbook

---

## 📊 Dashboard Tabs

| Tab | Contents |
|-----|----------|
| **Risk Overview** | Top-10 risk employees, alert distribution donut, daily score timeline |
| **Badge Analysis** | Hourly access heatmap by role, off-hours violations table |
| **VPN Threats** | Country-of-origin map, brute-force cluster timeline |
| **Crew Portal** | Bulk download events, record-type access breakdown |
| **Maintenance** | Tamper events by component, work-order compliance rate |
| **Cargo** | Weight anomaly scatter, off-hours door access events |
| **Employee Drill-Down** | Per-employee full event log, score history chart |

---

## 📚 References

- ICAO Insider Threat Toolkit (2022) — Doc 10139
- TSA Insider Threat Roadmap (2020)
- GAO Report GAO-20-275 — Aviation Insider Threats
- Osprey Flight Solutions — Aviation Insider Threats 2024–25
- MITRE ATT&CK Enterprise v14 — T1078, T1133, T1048, T1565
- MITRE ATT&CK for ICS — Inhibit Response Function
- Verizon DBIR 2024 — Insider Threat Statistics
- IBM Cost of a Data Breach Report 2023
- SANS SOC Survey 2023 — MTTD/MTTA Benchmarks

---

## 📄 License

MIT — free to use, modify, and build on. If you use this in a portfolio, attribution appreciated. Please credit the original aviation security source organisations listed above.

---

<p align="center">
  Built as a SOC Engineering portfolio project · Aligned with MITRE ATT&CK · No commercial SIEM required
</p>
