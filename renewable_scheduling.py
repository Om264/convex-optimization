"""
Multi-Objective Renewable Energy Scheduling Using Convex Optimization
=====================================================================
Implements QP, ADMM, and Robust Optimization methods as described in the paper.
"""

import numpy as np
import cvxpy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

np.random.seed(42)

# ─────────────────────────────────────────────
# SYSTEM PARAMETERS
# ─────────────────────────────────────────────
T          = 24          # scheduling horizon (hours)
E_MAX      = 50.0        # max battery capacity [kWh]
E_INIT     = 25.0        # initial battery energy [kWh]
P_CH_MAX   = 15.0        # max charging power [kW]
P_DIS_MAX  = 15.0        # max discharging power [kW]
P_GRID_MAX = 60.0        # max grid power [kW]
ETA_C      = 0.92        # charging efficiency
ETA_D      = 0.92        # discharging efficiency
DELTA      = 5.0         # max demand-response shift [kW]
ALPHA      = 0.5         # cost vs. emission trade-off
LAMBDA_DEG = 0.04        # battery degradation penalty [$/kWh]
GAMMA      = 0.8         # demand-response discomfort penalty
DELTA_S    = 0.10        # solar uncertainty margin (10 %)
DELTA_W    = 0.15        # wind  uncertainty margin (15 %)
ADMM_RHO   = 1.0         # ADMM penalty parameter
ADMM_ITER  = 200         # max ADMM iterations
ADMM_TOL   = 1e-4        # ADMM convergence tolerance

# ─────────────────────────────────────────────
# 1. SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────
def generate_synthetic_data():
    t = np.arange(T)

    # Demand: morning ramp, evening peak, night dip
    demand = (30
              + 8  * np.exp(-((t - 8) ** 2) / 8)   # morning peak
              + 15 * np.exp(-((t - 19)** 2) / 6)   # evening peak
              - 5  * np.exp(-((t - 3) ** 2) / 4)   # nighttime dip
              + 2  * np.random.randn(T))             # small noise
    demand = np.clip(demand, 15, 55)

    # Solar: bell-shaped daytime curve (zero at night)
    solar_raw = 28 * np.exp(-((t - 12) ** 2) / 16)
    solar_raw[solar_raw < 0.5] = 0
    solar = solar_raw + np.random.randn(T) * 1.5
    solar = np.clip(solar, 0, 32)

    # Wind: stochastic fluctuating signal
    wind_base = 12 + 6 * np.sin(2 * np.pi * t / 24) + 4 * np.cos(2 * np.pi * t / 12)
    wind = wind_base + np.random.randn(T) * 2.5
    wind = np.clip(wind, 0, 25)

    # Electricity price: off-peak cheap, peak expensive
    price = (0.08
             + 0.07 * np.exp(-((t - 8) ** 2) / 10)   # morning peak price
             + 0.12 * np.exp(-((t - 19)** 2) / 6)    # evening peak price
             - 0.02 * np.exp(-((t - 3) ** 2) / 5))   # cheap at night
    price = np.clip(price, 0.05, 0.25)

    # Emission factor (grid carbon intensity mirrors price)
    emission = 0.4 + 0.3 * (price - price.min()) / (price.max() - price.min())

    # Robust uncertainty margins
    delta_s = DELTA_S * solar
    delta_w = DELTA_W * wind

    return demand, solar, wind, price, emission, delta_s, delta_w


# ─────────────────────────────────────────────
# 2. METHOD 1 — DIRECT QUADRATIC PROGRAMMING
# ─────────────────────────────────────────────
def solve_qp(demand, solar, wind, price, emission, delta_s, delta_w,
             robust=False):
    """
    Solve the renewable energy scheduling QP using CVXPY.
    If robust=True, tighter renewable bounds are applied (Robust Optimization).
    """
    # Decision variables
    P_grid     = cp.Variable(T, name="P_grid",     nonneg=True)
    P_solar    = cp.Variable(T, name="P_solar",    nonneg=True)
    P_wind     = cp.Variable(T, name="P_wind",     nonneg=True)
    P_charge   = cp.Variable(T, name="P_charge",   nonneg=True)
    P_discharge= cp.Variable(T, name="P_discharge",nonneg=True)
    E          = cp.Variable(T, name="E",          nonneg=True)
    D_shift    = cp.Variable(T, name="D_shift")

    # ── Objective ──────────────────────────────────────────────────────────
    cost_term      = ALPHA       * cp.sum(cp.multiply(price,    P_grid))
    emission_term  = (1 - ALPHA) * cp.sum(cp.multiply(emission, P_grid))
    degrad_term    = LAMBDA_DEG  * cp.sum(P_discharge)
    discomfort     = GAMMA       * cp.sum_squares(D_shift)

    objective = cp.Minimize(cost_term + emission_term + degrad_term + discomfort)

    # ── Constraints ────────────────────────────────────────────────────────
    constraints = []

    # Power balance (equality, every hour)
    constraints += [
        P_solar + P_wind + P_grid + P_discharge
        == demand + D_shift + P_charge
    ]

    # Battery dynamics  E[t+1] = E[t] + eta_c*P_ch - (1/eta_d)*P_dis
    for i in range(T - 1):
        constraints.append(
            E[i + 1] == E[i] + ETA_C * P_charge[i] - (1 / ETA_D) * P_discharge[i]
        )
    constraints.append(E[0] == E_INIT)   # initial SOC

    # Demand response balance
    constraints.append(cp.sum(D_shift) == 0)

    # Battery bounds
    constraints += [E <= E_MAX]

    # Grid bounds
    constraints += [P_grid <= P_GRID_MAX]

    # Charge / discharge power limits
    constraints += [P_charge    <= P_CH_MAX]
    constraints += [P_discharge <= P_DIS_MAX]

    # Demand response bounds
    constraints += [D_shift >= -DELTA, D_shift <= DELTA]

    # Renewable bounds (with or without uncertainty margins)
    if robust:
        solar_ub = np.maximum(solar - delta_s, 0)
        wind_ub  = np.maximum(wind  - delta_w, 0)
    else:
        solar_ub = solar
        wind_ub  = wind

    constraints += [P_solar <= solar_ub]
    constraints += [P_wind  <= wind_ub]

    # ── Solve ──────────────────────────────────────────────────────────────
    prob = cp.Problem(objective, constraints)
    t0   = time.time()
    prob.solve(solver=cp.OSQP, eps_abs=1e-5, eps_rel=1e-5, verbose=False)
    elapsed = time.time() - t0

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"Solver status: {prob.status}")

    return {
        "P_grid"     : P_grid.value,
        "P_solar"    : P_solar.value,
        "P_wind"     : P_wind.value,
        "P_charge"   : P_charge.value,
        "P_discharge": P_discharge.value,
        "E"          : E.value,
        "D_shift"    : D_shift.value,
        "cost"       : prob.value,
        "time"       : elapsed,
    }


# ─────────────────────────────────────────────
# 3. METHOD 2 — ADMM
# ─────────────────────────────────────────────
def solve_admm(demand, solar, wind, price, emission, delta_s, delta_w):
    """
    ADMM decomposition for renewable energy scheduling.

    Global consensus form:
      min  f_gen(x_gen) + f_batt(x_batt) + f_dr(x_dr)
      s.t. x_gen + x_batt + x_dr - consensus = demand   (power balance)

    Sub-problems solved with CVXPY; dual variables updated analytically.
    """
    # ── Initialise ─────────────────────────────────────────────────────────
    rho = ADMM_RHO

    # Consensus power-balance variable (auxiliary)
    z = np.zeros(T)        # shared supply = demand

    # Dual variables (Lagrange multipliers)
    u_gen  = np.zeros(T)
    u_batt = np.zeros(T)
    u_dr   = np.zeros(T)

    # Local copies
    x_gen  = np.zeros((T, 3))   # [P_grid, P_solar, P_wind]
    x_batt = np.zeros((T, 3))   # [P_charge, P_discharge, E]
    x_dr   = np.zeros(T)        # D_shift

    history = {"primal": [], "dual": []}

    for k in range(ADMM_ITER):
        z_old = z.copy()

        # ── Sub-problem 1: Generation scheduling ──────────────────────────
        P_grid   = cp.Variable(T, nonneg=True)
        P_solar  = cp.Variable(T, nonneg=True)
        P_wind   = cp.Variable(T, nonneg=True)
        net_gen  = P_grid + P_solar + P_wind
        target   = z - u_gen

        obj_gen  = (ALPHA       * cp.sum(cp.multiply(price, P_grid))
                    + (1-ALPHA) * cp.sum(cp.multiply(emission, P_grid))
                    + (rho/2)   * cp.sum_squares(net_gen - target))

        con_gen  = [P_grid  <= P_GRID_MAX,
                    P_solar <= solar,
                    P_wind  <= wind]
        cp.Problem(cp.Minimize(obj_gen), con_gen).solve(solver=cp.OSQP, verbose=False)

        x_gen[:, 0] = P_grid.value
        x_gen[:, 1] = P_solar.value
        x_gen[:, 2] = P_wind.value
        gen_supply   = x_gen.sum(axis=1)          # P_grid + P_solar + P_wind

        # ── Sub-problem 2: Battery scheduling ────────────────────────────
        P_charge    = cp.Variable(T, nonneg=True)
        P_discharge = cp.Variable(T, nonneg=True)
        E_batt      = cp.Variable(T, nonneg=True)
        # Net contribution of battery to supply
        net_batt    = P_discharge - P_charge
        target_b    = z - u_batt

        obj_batt = (LAMBDA_DEG * cp.sum(P_discharge)
                    + (rho/2)  * cp.sum_squares(net_batt - target_b))

        con_batt = [P_charge    <= P_CH_MAX,
                    P_discharge <= P_DIS_MAX,
                    E_batt      <= E_MAX,
                    E_batt[0]   == E_INIT]
        for i in range(T-1):
            con_batt.append(
                E_batt[i+1] == E_batt[i]
                + ETA_C * P_charge[i] - (1/ETA_D) * P_discharge[i]
            )

        cp.Problem(cp.Minimize(obj_batt), con_batt).solve(solver=cp.OSQP, verbose=False)

        x_batt[:, 0] = P_charge.value
        x_batt[:, 1] = P_discharge.value
        x_batt[:, 2] = E_batt.value
        batt_net      = x_batt[:, 1] - x_batt[:, 0]   # P_dis - P_ch

        # ── Sub-problem 3: Demand response ───────────────────────────────
        D_shift  = cp.Variable(T)
        # DR reduces effective demand: net_load = demand + D_shift
        # Contribution to power balance: "supply side" sees -D_shift
        net_dr   = -D_shift
        target_d = z - u_dr

        obj_dr   = (GAMMA    * cp.sum_squares(D_shift)
                    + (rho/2) * cp.sum_squares(net_dr - target_d))

        con_dr   = [D_shift >= -DELTA, D_shift <= DELTA,
                    cp.sum(D_shift) == 0]

        cp.Problem(cp.Minimize(obj_dr), con_dr).solve(solver=cp.OSQP, verbose=False)

        x_dr = D_shift.value if D_shift.value is not None else np.zeros(T)

        # ── z-update (consensus = demand) ────────────────────────────────
        # Power balance: gen_supply + batt_net - D_shift = demand
        # => z = demand is fixed; update to balance residuals
        z = (demand
             + (1/3) * (u_gen + u_batt + u_dr)
             + (1/3) * (gen_supply + batt_net - x_dr)
             - (1/3) * (gen_supply + batt_net - x_dr - demand))
        z = demand.copy()   # enforce exact balance

        # ── Dual updates ─────────────────────────────────────────────────
        u_gen  += gen_supply - z
        u_batt += batt_net   - z
        u_dr   += -x_dr      - (np.zeros(T))   # DR shifts demand

        # ── Convergence check ─────────────────────────────────────────────
        primal_res = np.linalg.norm(gen_supply + batt_net - x_dr - demand)
        dual_res   = rho * np.linalg.norm(z - z_old)
        history["primal"].append(primal_res)
        history["dual"].append(dual_res)

        if primal_res < ADMM_TOL and dual_res < ADMM_TOL:
            print(f"  ADMM converged at iteration {k+1}")
            break

    # Compute cost
    pg  = x_gen[:, 0]
    cost = (ALPHA       * np.dot(price, pg)
            + (1-ALPHA) * np.dot(emission, pg)
            + LAMBDA_DEG * np.sum(x_batt[:, 1])
            + GAMMA      * np.sum(x_dr ** 2))

    return {
        "P_grid"     : x_gen[:, 0],
        "P_solar"    : x_gen[:, 1],
        "P_wind"     : x_gen[:, 2],
        "P_charge"   : x_batt[:, 0],
        "P_discharge": x_batt[:, 1],
        "E"          : x_batt[:, 2],
        "D_shift"    : x_dr,
        "cost"       : cost,
        "time"       : None,
        "history"    : history,
    }


# ─────────────────────────────────────────────
# 4. SOLVE ALL THREE METHODS
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Renewable Energy Scheduling — Convex Optimization")
    print("=" * 60)

    demand, solar, wind, price, emission, delta_s, delta_w = generate_synthetic_data()

    print("\n[1/3] Solving with Quadratic Programming (QP)…")
    t0  = time.time()
    qp  = solve_qp(demand, solar, wind, price, emission, delta_s, delta_w, robust=False)
    qp["time"] = time.time() - t0
    print(f"      Cost = ${qp['cost']:.4f}  |  Time = {qp['time']*1000:.1f} ms")

    print("\n[2/3] Solving with ADMM…")
    t0    = time.time()
    admm  = solve_admm(demand, solar, wind, price, emission, delta_s, delta_w)
    admm["time"] = time.time() - t0
    print(f"      Cost = ${admm['cost']:.4f}  |  Time = {admm['time']*1000:.1f} ms")

    print("\n[3/3] Solving with Robust Optimization…")
    t0     = time.time()
    robust = solve_qp(demand, solar, wind, price, emission, delta_s, delta_w, robust=True)
    robust["time"] = time.time() - t0
    print(f"      Cost = ${robust['cost']:.4f}  |  Time = {robust['time']*1000:.1f} ms")

    # ─────────────────────────────────────────────
    # 5. VISUALISATION
    # ─────────────────────────────────────────────
    hours  = np.arange(1, T + 1)
    COLORS = {
        "demand"  : "#e74c3c",
        "solar"   : "#f39c12",
        "wind"    : "#27ae60",
        "grid"    : "#3498db",
        "charge"  : "#9b59b6",
        "discharge": "#e67e22",
        "battery" : "#1abc9c",
        "dr"      : "#e74c3c",
        "price"   : "#2c3e50",
        "emission": "#7f8c8d",
    }

    # ── Figure 1: Synthetic Data ───────────────────────────────────────────
    fig1, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig1.suptitle("Synthetic Data — 24-Hour Smart Microgrid Profiles",
                  fontsize=14, fontweight="bold", y=0.98)

    ax = axes[0, 0]
    ax.fill_between(hours, demand, alpha=0.3, color=COLORS["demand"])
    ax.plot(hours, demand, "o-", color=COLORS["demand"], lw=2, ms=4, label="Demand")
    ax.set_title("Electricity Demand Profile"); ax.set_ylabel("Power [kW]")
    ax.set_xlabel("Hour"); ax.legend(); ax.grid(alpha=0.3); ax.set_xlim(1, 24)

    ax = axes[0, 1]
    ax.fill_between(hours, solar, alpha=0.3, color=COLORS["solar"])
    ax.fill_between(hours, wind,  alpha=0.3, color=COLORS["wind"])
    ax.plot(hours, solar, "o-", color=COLORS["solar"], lw=2, ms=4, label="Solar")
    ax.plot(hours, wind,  "s-", color=COLORS["wind"],  lw=2, ms=4, label="Wind")
    ax.set_title("Renewable Generation Profiles"); ax.set_ylabel("Power [kW]")
    ax.set_xlabel("Hour"); ax.legend(); ax.grid(alpha=0.3); ax.set_xlim(1, 24)

    ax = axes[1, 0]
    ax.step(hours, price, where="mid", color=COLORS["price"], lw=2, label="Price")
    ax.fill_between(hours, price, alpha=0.2, color=COLORS["price"], step="mid")
    ax.set_title("Time-Varying Electricity Price"); ax.set_ylabel("Price [$/kWh]")
    ax.set_xlabel("Hour"); ax.legend(); ax.grid(alpha=0.3); ax.set_xlim(1, 24)

    ax = axes[1, 1]
    ax.step(hours, emission, where="mid", color=COLORS["emission"], lw=2, label="Emission factor")
    ax.fill_between(hours, emission, alpha=0.2, color=COLORS["emission"], step="mid")
    ax.set_title("Grid Carbon Emission Factor"); ax.set_ylabel("Emission [kg CO₂/kWh]")
    ax.set_xlabel("Hour"); ax.legend(); ax.grid(alpha=0.3); ax.set_xlim(1, 24)

    plt.tight_layout()
    fig1.savefig("/mnt/user-data/outputs/fig1_synthetic_data.png", dpi=150, bbox_inches="tight")
    print("\n  ✓ Saved fig1_synthetic_data.png")

    # ── Figure 2: QP Scheduling Results ───────────────────────────────────
    fig2, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig2.suptitle("QP Solution — Optimal Scheduling Results",
                  fontsize=14, fontweight="bold")

    # Stacked generation
    ax = axes[0]
    ax.stackplot(hours, qp["P_solar"], qp["P_wind"],
                 qp["P_grid"], qp["P_discharge"],
                 labels=["Solar", "Wind", "Grid", "Battery discharge"],
                 colors=[COLORS["solar"], COLORS["wind"],
                         COLORS["grid"],  COLORS["discharge"]],
                 alpha=0.85)
    ax.plot(hours, demand + qp["D_shift"], "r--", lw=2,
            label="Adjusted demand")
    ax.set_title("Stacked Power Supply vs. Demand")
    ax.set_ylabel("Power [kW]"); ax.set_xlabel("Hour")
    ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.3); ax.set_xlim(1, 24)

    # Battery
    ax = axes[1]
    ax2 = ax.twinx()
    ax.bar(hours - 0.2, qp["P_charge"],    0.35, color=COLORS["charge"],    alpha=0.8, label="Charging")
    ax.bar(hours + 0.2, qp["P_discharge"], 0.35, color=COLORS["discharge"], alpha=0.8, label="Discharging")
    ax2.plot(hours, qp["E"], "ko-", lw=2, ms=4, label="State of Charge")
    ax2.set_ylabel("Battery Energy [kWh]"); ax2.set_ylim(0, E_MAX * 1.1)
    ax.set_title("Battery Charging / Discharging & State of Charge")
    ax.set_ylabel("Power [kW]"); ax.set_xlabel("Hour")
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=9)
    ax.grid(alpha=0.3); ax.set_xlim(0.5, 24.5)

    # Demand response
    ax = axes[2]
    colors_dr = [COLORS["charge"] if v > 0 else COLORS["discharge"] for v in qp["D_shift"]]
    ax.bar(hours, qp["D_shift"], color=colors_dr, alpha=0.8, edgecolor="white", lw=0.5)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title("Demand Response Adjustments (D_shift)")
    ax.set_ylabel("Shift [kW]"); ax.set_xlabel("Hour")
    ax.set_ylim(-DELTA * 1.3, DELTA * 1.3)
    ax.grid(alpha=0.3); ax.set_xlim(0.5, 24.5)
    legend_elements = [Patch(facecolor=COLORS["charge"],    label="Load increase (+)"),
                       Patch(facecolor=COLORS["discharge"], label="Load decrease (−)")]
    ax.legend(handles=legend_elements, fontsize=9)

    plt.tight_layout()
    fig2.savefig("/mnt/user-data/outputs/fig2_qp_results.png", dpi=150, bbox_inches="tight")
    print("  ✓ Saved fig2_qp_results.png")

    # ── Figure 3: Method Comparison ────────────────────────────────────────
    fig3, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig3.suptitle("Method Comparison: QP vs ADMM vs Robust Optimization",
                  fontsize=14, fontweight="bold")

    methods = ["QP", "ADMM", "Robust QP"]
    results = [qp, admm, robust]
    method_colors = ["#3498db", "#e67e22", "#27ae60"]

    # Grid power comparison
    ax = axes[0, 0]
    for res, lab, col in zip(results, methods, method_colors):
        ax.plot(hours, res["P_grid"], "o-", color=col, lw=1.8, ms=3, label=lab)
    ax.set_title("Grid Power Consumption"); ax.set_ylabel("Power [kW]")
    ax.set_xlabel("Hour"); ax.legend(); ax.grid(alpha=0.3); ax.set_xlim(1, 24)

    # Battery SOC comparison
    ax = axes[0, 1]
    for res, lab, col in zip(results, methods, method_colors):
        ax.plot(hours, res["E"], "s-", color=col, lw=1.8, ms=3, label=lab)
    ax.axhline(E_MAX, color="gray", ls="--", lw=1, label=f"E_max={E_MAX} kWh")
    ax.set_title("Battery State of Charge"); ax.set_ylabel("Energy [kWh]")
    ax.set_xlabel("Hour"); ax.legend(); ax.grid(alpha=0.3); ax.set_xlim(1, 24)

    # Demand response comparison
    ax = axes[1, 0]
    width = 0.25
    for i, (res, lab, col) in enumerate(zip(results, methods, method_colors)):
        ax.bar(hours + (i - 1) * width, res["D_shift"], width,
               color=col, alpha=0.8, label=lab)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Demand Response Adjustments"); ax.set_ylabel("Shift [kW]")
    ax.set_xlabel("Hour"); ax.legend(); ax.grid(alpha=0.3); ax.set_xlim(0.5, 24.5)

    # Cost / time bar chart
    ax = axes[1, 1]
    costs = [r["cost"] for r in results]
    times = [r["time"] * 1000 for r in results]   # ms

    x = np.arange(len(methods))
    bars = ax.bar(x, costs, color=method_colors, alpha=0.85, edgecolor="white", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_title("Objective Value Comparison"); ax.set_ylabel("Total Cost [$]")
    ax.grid(axis="y", alpha=0.3)

    # Add time annotations
    ax2 = ax.twinx()
    ax2.plot(x, times, "D--k", ms=8, lw=1.5, label="Comp. time (ms)")
    ax2.set_ylabel("Computation Time [ms]")
    ax2.legend(loc="upper right")

    for bar, val in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.01,
                f"${val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig3.savefig("/mnt/user-data/outputs/fig3_method_comparison.png", dpi=150, bbox_inches="tight")
    print("  ✓ Saved fig3_method_comparison.png")

    # ── Figure 4: ADMM Convergence ─────────────────────────────────────────
    fig4, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig4.suptitle("ADMM Convergence Analysis", fontsize=14, fontweight="bold")

    iters = np.arange(1, len(admm["history"]["primal"]) + 1)
    ax = axes[0]
    ax.semilogy(iters, admm["history"]["primal"], "b-", lw=2, label="Primal residual")
    ax.axhline(ADMM_TOL, color="red", ls="--", lw=1.5, label=f"Tolerance = {ADMM_TOL}")
    ax.set_title("Primal Residual"); ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual (log scale)"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.semilogy(iters, admm["history"]["dual"], "g-", lw=2, label="Dual residual")
    ax.axhline(ADMM_TOL, color="red", ls="--", lw=1.5, label=f"Tolerance = {ADMM_TOL}")
    ax.set_title("Dual Residual"); ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual (log scale)"); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    fig4.savefig("/mnt/user-data/outputs/fig4_admm_convergence.png", dpi=150, bbox_inches="tight")
    print("  ✓ Saved fig4_admm_convergence.png")

    # ── Figure 5: Robust vs Deterministic ─────────────────────────────────
    fig5, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig5.suptitle("Robust Optimization — Impact of Renewable Uncertainty",
                  fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.fill_between(hours,
                    np.maximum(solar - delta_s, 0), solar,
                    alpha=0.3, color=COLORS["solar"], label="Uncertainty band")
    ax.fill_between(hours,
                    np.maximum(wind - delta_w, 0), wind,
                    alpha=0.3, color=COLORS["wind"], label="_nolegend_")
    ax.plot(hours, qp["P_solar"],     "--", color=COLORS["solar"],  lw=2, label="QP Solar used")
    ax.plot(hours, robust["P_solar"], "-",  color=COLORS["solar"],  lw=2, label="Robust Solar used")
    ax.plot(hours, qp["P_wind"],      "--", color=COLORS["wind"],   lw=2, label="QP Wind used")
    ax.plot(hours, robust["P_wind"],  "-",  color=COLORS["wind"],   lw=2, label="Robust Wind used")
    ax.set_title("Renewable Usage: QP vs Robust"); ax.set_ylabel("Power [kW]")
    ax.set_xlabel("Hour"); ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xlim(1, 24)

    ax = axes[1]
    ax.plot(hours, qp["P_grid"],     "b--", lw=2, label="QP Grid power")
    ax.plot(hours, robust["P_grid"], "g-",  lw=2, label="Robust Grid power")
    ax2 = ax.twinx()
    ax2.fill_between(hours,
                     qp["P_grid"] - robust["P_grid"],
                     alpha=0.25, color="red")
    ax2.plot(hours,
             qp["P_grid"] - robust["P_grid"],
             "r-", lw=1.5, label="Δ Grid (QP−Robust)")
    ax2.set_ylabel("Grid Power Difference [kW]", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax.set_title("Grid Power: Safety Margin from Robustness")
    ax.set_ylabel("Grid Power [kW]"); ax.set_xlabel("Hour")
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=9)
    ax.grid(alpha=0.3); ax.set_xlim(1, 24)

    plt.tight_layout()
    fig5.savefig("/mnt/user-data/outputs/fig5_robust_analysis.png", dpi=150, bbox_inches="tight")
    print("  ✓ Saved fig5_robust_analysis.png")

    # ─────────────────────────────────────────────
    # 6. SUMMARY TABLE
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Metric':<35} {'QP':>10} {'ADMM':>10} {'Robust':>10}")
    print("-" * 65)

    def pct(a, b):
        return f"{(a-b)/b*100:+.1f}%"

    for lab, res in [("QP", qp), ("ADMM", admm), ("Robust", robust)]:
        pg   = res["P_grid"]
        ps   = res["P_solar"]
        pw   = res["P_wind"]
        pd   = res["P_discharge"]
        drs  = res["D_shift"]
        ren  = ps + pw
        tot  = pg + ren + pd

    def metric_row(name, fn):
        vals = [fn(r) for r in results]
        print(f"{name:<35} {vals[0]:>10.3f} {vals[1]:>10.3f} {vals[2]:>10.3f}")

    metric_row("Total cost [$]",
               lambda r: r["cost"])
    metric_row("Comp. time [ms]",
               lambda r: r["time"] * 1000)
    metric_row("Avg grid power [kW]",
               lambda r: np.mean(r["P_grid"]))
    metric_row("Total renewable used [kWh]",
               lambda r: np.sum(r["P_solar"] + r["P_wind"]))
    metric_row("Total grid energy [kWh]",
               lambda r: np.sum(r["P_grid"]))
    metric_row("Total battery cycles [kWh]",
               lambda r: np.sum(r["P_discharge"]))
    metric_row("Max demand shift [kW]",
               lambda r: np.max(np.abs(r["D_shift"])))
    metric_row("Renewable fraction [%]",
               lambda r: 100 * np.sum(r["P_solar"] + r["P_wind"])
                             / np.sum(r["P_grid"] + r["P_solar"] + r["P_wind"] + r["P_discharge"]))

    print("=" * 65)
    print("\nAll figures saved to /mnt/user-data/outputs/")


if __name__ == "__main__":
    main()
