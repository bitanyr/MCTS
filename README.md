# Physics-Informed AlphaZero for Active Distribution Network Planning





# Abstract

The integration of Distributed Energy Resources (DERs) has transformed passive distribution grids into Active Distribution Networks (ADNs). While this transition offers significant economic and environmental benefits, it introduces severe operational challenges, including bidirectional power flows, voltage limit violations, and thermal congestion.

This repository provides an advanced, open-source framework for the optimal siting and sizing of dispatchable and non-dispatchable DERs. To overcome the NP-hard nature of the resulting Mixed-Integer Non-Linear Programming (MINLP) problem, we propose a novel architecture that couples Deep Reinforcement Learning—specifically, the AlphaZero algorithm utilizing Monte Carlo Tree Search (MCTS)—with an exact mathematical physics engine (Pyomo/Ipopt).

This ensures that all AI-generated expansion strategies are strictly bound by the physical laws of alternating current (AC) power systems.

---

# 1. Mathematical Framework & Power System Innovations

Traditional planning algorithms rely heavily on meta-heuristics (e.g., PSO, Genetic Algorithms) which suffer from dimensionality curses, or brute-force MINLP solvers that fail to scale for multi-period dynamics.

This project introduces several domain-specific structural innovations:

## 1.1 Tractable MINLP Decomposition

The framework decouples the planning problem into two hierarchical stages:

* **The Master Problem (MDP Framework):**
  
  The AlphaZero agent handles the discrete integer decisions, navigating the immense combinatorial space of component siting (bus locations) and sizing (capacity steps).

* **The Sub-Problem (NLP Solver):**

  A deterministic Multi-Period Optimal Power Flow (MP-OPF) evaluates the operational viability of the agent's proposed topology across continuous temporal horizons.

---

## 1.2 Convexification of AC Power Flow (SOCP Relaxation)

To provide reliable, deterministic reward signals to the neural network and guarantee sub-problem convergence, the non-convex branch flow model (DistFlow) is relaxed using Second-Order Cone Programming (SOCP).

The non-linear apparent power constraint for a branch connecting node $i$ to node $j$ at time $t$ is originally:

$$
P_{ij,t}^2 + Q_{ij,t}^2 \le V_{i,t}^2 I_{ij,t}^2
$$

By introducing squared variable substitutions where:

$$
v_i = V_i^2
$$

and

$$
l_{ij}=I_{ij}^2
$$

the constraint is reformulated into a strictly convex cone:

$$
P_{ij,t}^2 + Q_{ij,t}^2 \le v_{i,t}\cdot l_{ij,t}
$$

This relaxation guarantees the extraction of global optima during the sub-problem evaluation phase.

---

## 1.3 Advanced Grid Dynamics Modeling

### Spatio-Temporal Co-Optimization

Evaluates configurations across a 96-hour continuous horizon encompassing four distinct seasonal profiles.

This enables the accurate modeling of Battery Energy Storage System (BESS) state-of-charge (SOC) continuity and energy arbitrage under Real-Time Pricing (RTP).

### Strictly Interior Warm Start

Implemented a robust initialization methodology to bypass algorithmic failures in the Ipopt solver.

By initializing slack variables and current magnitudes with small positive scalars:

$$
\epsilon > 0
$$

we prevent logarithmic barrier singularities during the stochastic exploration phase of MCTS.

### Constraint Softening

Slack variables are integrated into thermal and voltage constraints to guarantee mathematical feasibility, preventing solver crashes and returning heavy scalar penalties to guide the AI away from unstable topologies.

---
# 2. Markov Decision Process (MDP) Formulation

The physical environment is mathematically encapsulated into an MDP tailored for electrical networks:

## State Space (S)

A multidimensional tensor representation of the grid topology, current nodal capacities, time-series load profiles, PV generation curves, and localized marginal prices.

## Action Space (A)

Discrete control variables dictating the installation or capacity augmentation of specific devices (BESS, Gas Generators, SVCs, PV arrays) at specific bus indices.

## Reward Function (R)

A composite scalar signal defined as:

$$
R = - \Big( \text{CAPEX} \cdot \text{CRF} + \text{OPEX} + \lambda_v \sum \Delta v + \lambda_c \sum \Delta l \Big)
$$

Where CRF is the Capital Recovery Factor, and $\lambda$ represents penalty multipliers.

Rewards are linearly normalized to a $[-2.0, 2.0]$ domain to stabilize the deep neural network backpropagation and prevent exploding gradients.

---

# 3. Performance Metrics & Analytics

The framework was benchmarked on the standard IEEE 33-bus radial distribution system.

Training converged after 200 autonomous self-play episodes.

---

## 3.1 Economic Optimization

| Economic Metric | Base Case Network | AlphaZero Topology | System Impact |
| :--- | :--- | :--- | :--- |
| **Grid Import Cost (RTP)** | $562,620 | $273,725 | 51.3% Reduction |
| **Cost of Active Power Losses** | $63,225 | $45,691 | 27.7% Reduction |
| **PV Curtailment Penalty** | $14,500 | $1,200 | 91.7% Reduction |
| **Net Annualized Savings** | N/A | **$98,080** | Post-CAPEX deduction |

---

## 3.2 Technical Grid Enhancement

| Technical Metric | Base Case Network | AlphaZero Topology | System Impact |
| :--- | :--- | :--- | :--- |
| **Voltage Deviation Index (VDI)** | 0.048 p.u. | 0.012 p.u. | 75.0% Improvement |
| **Minimum Node Voltage** | 0.89 p.u. (Bus 18) | 0.95 p.u. | Restored to ANSI limits |
| **Peak-to-Average Ratio (PAR)** | 1.85 | 1.22 | Load profile flattened |
| **Max Feeder Congestion** | 112% (Lines 1-3) | 84% | Thermal overloads cleared |

---

## 3.3 Engineering Insights

Extraction of the final policy revealed sophisticated autonomous grid planning logic:

### 1. Duck Curve Mitigation

The algorithm autonomously placed a BESS at Bus 4 and dispatchable gas units deep in lateral branches (Buses 23, 29, 32).

It successfully learned to charge the BESS during solar peaks (mitigating reverse power flow overvoltages) and discharge during evening peaks to relieve the main substation transformer.

### 2. Coordinated VAR Compensation

Static Var Compensators (SVCs) were sited in precise electrical proximity to active power injection nodes, minimizing reactive power transit across long distribution lines, thus minimizing real power network losses.

---

# 4. Software Architecture

The repository is highly modular, rigorously separating the physics simulator from the reinforcement learning logic.

```text
ADN-AlphaZero-Planning/
│
├── data/
│   ├── ieee33.py              # Topology, line parameters (R, X), and thermal limits
│   ├── devices.py             # DER technical constraints and economic parameters
│   └── scenarios.py           # 96-hour deterministic profiles (Load, PV, LMP)
│
├── optimization/
│   ├── variables.py           # Initialization of spatio-temporal and slack variables
│   ├── constraints.py         # SOCP DistFlow equations and BESS continuity bounds
│   ├── objective.py           # Multi-objective cost function definition
│   └── model_builder.py       # Pyomo model synthesis and solver interfacing
│
├── env/
│   ├── state.py               # MDP state extraction and serialization logic
│   └── aps_env.py             # Environment step transitions and reward generation
│
├── core/
│   ├── network.py             # ResNet-based Deep Neural Network (Policy & Value heads)
│   ├── node.py                # MCTS node data structures (Edges, Priors, Visits)
│   ├── mcts.py                # PUCT algorithm implementation for Tree Search
│   └── replay_buffer.py       # Experience replay memory
│
├── self_play.py               # Main autonomous training loop execution
└── evaluate_model.py          # Post-training analysis and metric plotting
```
# 5. Installation & Configuration

## Prerequisites

Deployment within an isolated virtual environment (e.g., `venv` or `conda`) is mandatory due to strict dependency requirements for the customized Keras backend.

### Required Software

- Python 3.6 - 3.8
- Solvers:
  - Ipopt (Recommended for NLP optimization)
  - Gurobi/Mosek (Optional)

---

# 6. Execution

## Phase 1: Training the Agent

To initialize the Deep Neural Network and trigger the self-play reinforcement learning loop:

```bash
python self_play.py
```

> **Note**
>
> High MCTS simulation counts require significant computational resources.
> Network weights and transition data are checkpointed automatically.

---

## Phase 2: Evaluation & Benchmarking

To load the converged policy weights, simulate the final optimized topology, and generate analytical outputs:

```bash
python evaluate_model.py
```

---

# 7. Future Work

The following extensions are planned for future development:

- **Phase Unbalance**

  Extending the DistFlow constraints to model unbalanced three-phase distribution networks.

- **Stochastic Environments**

  Integrating auto-regressive uncertainty forecasting modules for load and PV generation directly into environment transitions.

- **EV Integration**

  Implementing temporal constraints for Electric Vehicle fleet charging and Vehicle-to-Grid (V2G) bidirectional discharging.

---
