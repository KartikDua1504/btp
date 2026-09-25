# BTP — Risk-Aware Cooperative Perception for Vehicular Digital Twins

> **Title:** Improving and Securing Cooperative Perception Aided DT Synchronisation and Migration against CPM Falsification in Mixed Vehicle Networks

**Team:** Kartik Dua, Vedansh (CSAI), Ananya  
**Supervisor:** Prof. Imran (ITS Group)  
**Base Paper:** Lu et al., *IEEE T-ITS*, Vol. 26, No. 2, Feb 2025 — [DOI: 10.1109/TITS.2024.3496121](https://doi.org/10.1109/TITS.2024.3496121)

---

## Problem Statement

The base paper builds a system for maintaining Digital Twins (DTs) of all vehicles — smart and legacy — using cooperative perception from Intelligent Connected Vehicles (ICVs). However, it has critical limitations:

1. **O(n³) matching** — can't run in real-time beyond 130 vehicles
2. **Risk-blind** — treats all vehicles equally regardless of danger
3. **No security** — assumes 100% honest ICVs (vulnerable to CPM falsification)
4. **No trust management** — malicious ICVs get a clean slate after crossing RSU boundaries

## Our Contributions

| # | Contribution | Speedup / Improvement |
|---|-------------|----------------------|
| 1 | Risk-Aware Greedy Matching (replaces Hungarian O(n³)) | **215× faster**, 132% quality |
| 2 | Water-Filling Resource Allocation (replaces SCA O(n²K)) | **40× faster**, **beats SCA by 51%** |
| 3 | Dijkstra Migration (replaces Bellman-Ford O(VE)) | 1.5× faster, identical paths |
| 4 | CPM Falsification Detection + Trust Framework | New capability (base paper has none) |
| 5 | Safety-aware bandwidth steering | 2.25× more BW to dangerous vehicles |

## Repository Structure

```
├── algorithmic_improvements_poc.py    # Core PoC: 3 algorithm comparisons with proofs
├── comprehensive_simulations.py       # 8-experiment suite (340+ runs, statistical rigor)
├── btp_chat/                          # Original WhatsApp chat + reference papers
│   ├── WhatsApp Chat with BTP🙏.txt
│   ├── Combined_Idea_Brief.pdf        # Our merged idea proposal
│   ├── 2507.14739v1.pdf               # CANDoSA (background)
│   ├── 2606.30430v1.pdf               # CAN IDS Benchmarking (background)
│   ├── 2408.17235v2.pdf               # ROAD Dataset IDS (background)
│   └── *.jpg, *.opus                  # Images and voice notes
└── README.md
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy

# Run core algorithmic comparison
python3 algorithmic_improvements_poc.py

# Run full 8-experiment evidence suite (~2 min)
python3 comprehensive_simulations.py
```

## Key Results

- **Real-time feasibility:** Base paper fails at 130 vehicles; ours handles 900 within the 100ms V2X deadline
- **Safety:** 112.6m less "blind distance" at 120 km/h for dangerous vehicles
- **Fairness:** 2.41× more bandwidth to danger WITHOUT starving safe vehicles
- **Formal guarantees:** (1−1/e) approximation bound, water-filling optimality theorem, Dijkstra correctness via Johnson's reweighting

## License

Academic use — BTech Project, 2026-2027.
