# Energy Systems Testing: Decision Quality Under Uncertainty

This repository contains executable test suites that model how complex energy systems fail **without anything being "broken"**.

The focus is not on protocol correctness or happy paths, but on **decision-making under uncertainty**:

- Delayed or partial data
- Out-of-order events
- Degraded visibility
- Automated actions taken with insufficient confidence

> The accompanying article explains *why* these failures occur.  
> This repository shows *how* they are tested.

---

## What This Repository Is (and Is Not)

### This repository is:

- A **deterministic testing harness** for distributed decision systems
- A way to **validate decision invariants** under degraded conditions
- A place to exercise **time, ordering, visibility, and human-impact guardrails**

### This repository is not:

- A smart-meter implementation
- A full grid simulator intended for production
- A tutorial or framework

Mocks here are stand-ins for real endpoints. In production testing, they are replaced with real services, devices, or APIs.

---

## Core Testing Philosophy

Across all suites, the same principles apply:

| Principle | Implication |
|-----------|-------------|
| Correct data can still be unsafe | Freshness and confidence matter |
| Partial visibility is not total failure | Degraded modes must be explicit |
| Event ordering matters more than event correctness | Late facts can invalidate early decisions |
| Customer-facing actions require higher confidence | Internal vs external thresholds differ |
| Automation must be reversible until certainty is achieved | Hold windows, cancellation logic |
| Automation is a decision-maker | Send-time rechecks and reversibility are required |

Each test enforces at least one of these invariants.

---

## Project Structure

```
energy-testing/
├── src/energy_testing/
│   ├── harness/                    # Test harness infrastructure
│   │   ├── time_control.py         # Deterministic time manipulation
│   │   ├── message_bus.py          # Message ordering, delay, drop, duplicate
│   │   ├── stubs.py                # Service stubs (ingestion, payments, notifications)
│   │   ├── decision_trace.py       # Decision records with reason codes
│   │   ├── invariants.py           # Core invariant definitions (INV-1..4)
│   │   └── arrears_engine.py       # Arrears decision logic with confidence gating
│   ├── simulators/
│   │   ├── grid.py                 # Energy grid simulator
│   │   └── mocks.py                # Mock producers, consumers, operators, markets
│   ├── models/
│   │   └── energy.py               # Data models (EnergySource, EnergyReading, etc.)
│   └── test_constants.py           # Shared test constants and helpers
│
├── tests/
│   ├── test_arrears_harness.py     # Decision logic, confidence gating, notifications
│   ├── test_clib_network_fault.py  # LD_PRELOAD fault injection (partial visibility)
│   ├── test_grid.py                # System-level behavior & stability
│   ├── test_mocks.py               # Component behavior (stand-ins for real APIs)
│   └── conftest.py                 # Fixtures: time control, ledgers, stubs
│
└── test_resources/
    └── c_helpers/
        └── network_fault.c         # C library for syscall-level fault injection
```

---

## Representative Failure Classes & Tests

The article highlights several failure patterns. Below are representative test cases that demonstrate how those patterns are exercised in code.

These are examples — the full suite goes deeper.

### 1️⃣ Stale-but-Valid Telemetry

**Failure pattern:** Data arrives successfully but is no longer relevant. Systems treat "latest received" as "current".

**Representative test:** `SM-AR-001` — stale reading misinterpreted as current  
**Located in:** `test_arrears_harness.py`

**What the test enforces:**
- Freshness boundaries are explicit
- Stale data downgrades system confidence
- Customer-facing arrears notifications are blocked

### 2️⃣ Out-of-Order Events (Eventual Consistency)

**Failure pattern:** Payment succeeds after arrears evaluation but before notification delivery.

**Representative test:** `SM-AR-003` — payment arrives after arrears evaluation  
**Located in:** `test_arrears_harness.py`

**What the test enforces:**
- Notifications are held, not immediately sent
- Late-arriving facts can cancel pending actions
- Final customer state reflects the most complete information available

### 3️⃣ Partial Visibility (Not Binary Failure)

**Failure pattern:** Some nodes are reachable, others are silent. The system still "sees something" and overreacts.

**Representative test:** `NF-001` — 30% nodes silent (partial visibility)  
**Located in:** `test_clib_network_fault.py`

**What the test enforces:**
- Partial loss does not trigger full protection logic
- Degraded modes are entered explicitly
- Faults are injected at the syscall level (not mocked)

### 4️⃣ Human-Impact Guardrails (Calendar & Confidence)

**Failure pattern:** Technically correct arrears logic triggers emotionally harmful communication.

**Representative test:** `SM-AR-009` — holiday / calendar policy  
**Located in:** `test_arrears_harness.py`

**What the test enforces:**
- Calendar rules are first-class inputs
- Confidence thresholds are higher for customer-facing actions
- Automation respects human context

### 5️⃣ Automation Acting on Provisional State

**Failure pattern:**  
Automation behaves correctly according to rules, but acts before uncertainty has resolved. No component fails; the action itself becomes the incident.

**Representative tests:**  
- `SM-AR-011` — provisional state holds until reconciliation  
- `SM-AR-012` — hard notification rechecked before release  
- `SM-AR-013` — idempotency across duplicate evaluations  
- `SM-AR-014` — acknowledgement without materialized state  

**Located in:** `test_arrears_harness.py`

**What the tests enforce:**
- Decisions are reversible until confidence thresholds are met
- Held actions are revalidated at release time
- Duplicate or replayed evaluations do not amplify harm
- Acknowledgements are not treated as final truth

These tests focus on the layers *around* the product — orchestration, timing, and confidence — where many modern incidents originate.

---

## Fault Injection vs Mocks

Two different techniques are used intentionally:

### 🔹 Mocks

Used to:
- Control inputs deterministically
- Model external systems (payments, ingestion, notifications)
- Validate decision logic

### 🔹 C-Library Fault Injection (LD_PRELOAD)

Used to:
- Simulate partial connectivity
- Inject asymmetric latency or packet loss
- Reproduce failures that mocks cannot model

This distinction mirrors real production testing: **mocks for control**, **fault injection for realism**.

---

## Determinism & Reproducibility

All tests are designed to be:
- **Deterministic** — same inputs produce same outputs
- **Time-controlled** — no reliance on wall-clock time
- **Repeatable in CI** — no flaky assertions

Key mechanisms:
- Explicit `TimeController` for freezing/advancing time
- Deterministic message bus with configurable ordering
- Bounded assertions (not "exact timing" expectations)

---

## Installation

```bash
pip install -e .
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with markers
pytest -m "case" tests/                    # Tests with case IDs
pytest -m "not linux_only" tests/          # Skip Linux-only tests

# Run specific test cases
pytest -k "SM-AR-001" tests/               # Single case
pytest -k "stale or holiday" tests/        # Multiple patterns
```

## Quick Start Example

```python
from energy_testing.simulators.grid import EnergyGridSimulator
from energy_testing.models.energy import EnergySource

# Create a grid with mixed producers and consumers
simulator = EnergyGridSimulator()
simulator.add_producer("solar_farm", EnergySource.SOLAR, capacity_kw=1000)
simulator.add_consumer("residential_area", max_demand_kw=500, consumer_type="residential")

# Run simulation
results = simulator.simulate(hours=24)
print(f"Total energy produced: {results.total_energy_produced} kWh")
print(f"Total energy consumed: {results.total_energy_consumed} kWh")
```

---

## How This Repo Relates to the Article

| Layer | Purpose |
|-------|---------|
| **Article** | Explains *what* fails and *why* teams miss it |
| **This repo** | Demonstrates *how* those failures are exercised and prevented |

The article intentionally does not include code.  
This repository intentionally does.

Together, they form:
- A **reasoning layer** (article)
- An **execution layer** (tests)

---

## Using This Repository

This repository is meant to be:
- **Read selectively** — start with representative cases
- **Executed in parts** — not all tests need to run
- **Adapted to other domains** — GovTech, payments, control systems

You do not need to run every test to understand the approach.  
Start with the representative cases listed above.

---

## Final Note

> Most incidents are not caused by broken components.  
> They are caused by systems making **confident decisions with incomplete information**.

Testing that reality requires:
- **Time as an input**
- **Uncertainty as a state**
- **Reversibility as a requirement**

That is what this repository is built to test.

---

## License

MIT
