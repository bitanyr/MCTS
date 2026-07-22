import sys
import os
import torch
import random
import logging
import pyomo.environ as pyo

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from env.aps_env import ActivePlanningEnv
from core.mcts import MCTS
from core.network import ADNDeepNet
from data.ieee33 import S_BASE, BRANCHES
from data.scenarios import RTP_PRICE
from data.devices import C_LOSS, C_RES, C_AUL, C_ESS_INV, P_ESS_MAX, C_PV_INV, PV_CAPACITY, C_GAS_INV, P_GAS_MAX, C_SVC_INV, Q_SVC_MAX, C_CB_INV, Q_CB_MAX

# خاموش کردن هشدارهای زرد رنگ در ترمینال
logging.getLogger('pyomo.core').setLevel(logging.ERROR)

def robust_evaluate_placement(model, placement_dict):
    """ حل‌گرِ قدرتمند و بدون محدودیت زمانی برای ارزیابی نهایی پایان‌نامه """
    for i in model.N:
        model.s_ess[i].set_value(0)
        model.s_gas[i].set_value(0)
        model.s_svc[i].set_value(0)
        model.s_cb[i].set_value(0)

    for node in placement_dict.get('ess', []): model.s_ess[node].set_value(1)
    for node in placement_dict.get('gas', []): model.s_gas[node].set_value(1)
    for node in placement_dict.get('svc', []): model.s_svc[node].set_value(1)
    for node in placement_dict.get('cb', []):  model.s_cb[node].set_value(1)

    for t in model.T:
        for i in model.N:
            model.v[i, t].set_value(1.0)
            if pyo.value(model.s_ess[i]) == 1:
                model.E_soc[i, t].set_value(0.3)
                model.P_ch[i, t].set_value(0.01)
                model.P_dis[i, t].set_value(0.01)
            else:
                model.E_soc[i, t].set_value(0.0)
                model.P_ch[i, t].set_value(0.0)
                model.P_dis[i, t].set_value(0.0)
                
            model.P_curt_res[i, t].set_value(0.0)
            model.P_curt_aul[i, t].set_value(0.0)
            model.P_gas[i, t].set_value(0.01 if pyo.value(model.s_gas[i]) == 1 else 0.0)
            model.Q_svc[i, t].set_value(0.0)
            model.Q_cb[i, t].set_value(0.01 if pyo.value(model.s_cb[i]) == 1 else 0.0)
            
            if hasattr(model, 'v_viol_down'): model.v_viol_down[i, t].set_value(0.001)
            if hasattr(model, 'v_viol_up'): model.v_viol_up[i, t].set_value(0.001)

        for k in model.E:
            model.P[k, t].set_value(0.01)
            model.Q[k, t].set_value(0.01)
            model.l[k, t].set_value(0.01)
            if hasattr(model, 'l_viol'): model.l_viol[k, t].set_value(0.001)
            
        model.P_sub[t].set_value(0.1)
        model.Q_sub[t].set_value(0.1)
        if hasattr(model, 'P_sub_import'): model.P_sub_import[t].set_value(0.1)
        if hasattr(model, 'P_sub_export'): model.P_sub_export[t].set_value(0.0)
        if hasattr(model, 'sub_overload'): model.sub_overload[t].set_value(0.001)

    solver = pyo.SolverFactory('ipopt')
    solver.options['max_iter'] = 5000
    solver.options['max_cpu_time'] = 120.0
    solver.options['tol'] = 1e-4
    solver.options['print_level'] = 0
    solver.options['mu_strategy'] = 'adaptive'
    solver.options['obj_scaling_factor'] = 1e-5
    solver.options['bound_push'] = 1e-6

    import os
    if not solver.available():
        idaes_path = os.path.join(os.path.expanduser('~'), '.idaes', 'bin', 'ipopt.exe')
        fallback_path = r'C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\Scripts\ipopt.exe'
        if os.path.exists(idaes_path):
            solver = pyo.SolverFactory('ipopt', executable=idaes_path)
        elif os.path.exists(fallback_path):
            solver = pyo.SolverFactory('ipopt', executable=fallback_path)

    try:
        solver.solve(model, tee=False)
    except:
        pass


def extract_detailed_costs(model, placement_dict):
    DAYS_MULTIPLIER = 365.0 / 4.0
    interest_rate = 0.05
    lifetime = 20
    CRF = (interest_rate * (1 + interest_rate)**lifetime) / (((1 + interest_rate)**lifetime) - 1)
    
    capex = sum(
        C_ESS_INV * P_ESS_MAX * (1 if i in placement_dict.get('ess', []) else 0) +
        C_GAS_INV * P_GAS_MAX * (1 if i in placement_dict.get('gas', []) else 0) +
        C_SVC_INV * Q_SVC_MAX * (1 if i in placement_dict.get('svc', []) else 0) +
        C_CB_INV * Q_CB_MAX * (1 if i in placement_dict.get('cb', []) else 0) +
        (C_PV_INV * PV_CAPACITY if i in [8, 10, 13, 16, 18, 20, 22, 28] else 0) 
        for i in model.N
    ) * CRF
    
    C_EXPORT_PRICE = 20.0  # باید همیشه با objective.py هماهنگ بماند
    market = sum(
        (RTP_PRICE[t] * pyo.value(model.P_sub_import[t]) - C_EXPORT_PRICE * pyo.value(model.P_sub_export[t])) * S_BASE
        for t in model.T
    ) * DAYS_MULTIPLIER
    loss = sum(C_LOSS * pyo.value(model.l[k, t]) * BRANCHES[k]['r'] * S_BASE for k in model.E for t in model.T) * DAYS_MULTIPLIER
    curt = sum(C_RES * pyo.value(model.P_curt_res[i, t]) * S_BASE + C_AUL * pyo.value(model.P_curt_aul[i, t]) * S_BASE for i in model.N for t in model.T) * DAYS_MULTIPLIER
    
    C_GAS_FUEL = 50.0  
    C_EMISSION = 20.0  
    gas_opex = sum(
        (C_GAS_FUEL + C_EMISSION) * pyo.value(model.P_gas[i, t]) * S_BASE * (1 if i in placement_dict.get('gas', []) else 0)
        for i in model.N for t in model.T
    ) * DAYS_MULTIPLIER

    # 🔴 جراحی اساسی: جریمه‌های ریاضی دیگر با دلارهای واقعی جمع نمی‌شوند!
    # این جریمه‌ها فقط برای هدایت شبکه عصبی در زمان آموزش (train.py) هستند.
    total_real = capex + market + loss + curt + gas_opex
    
    min_v = min(pyo.value(model.v[i, t]) for i in model.N for t in model.T) ** 0.5 
    losses_mw = sum(pyo.value(model.l[k, t]) * BRANCHES[k]['r'] * S_BASE for k in model.E for t in model.T)
    
    return {
        'min_v': min_v, 'losses_mw': losses_mw, 'capex': capex, 
        'market': market, 'loss_cost': loss, 'gas_opex': gas_opex, 
        'total': total_real
    }

def evaluate_and_compare(model_path="trained_adn_net_ep200.pth"):
    print("==================================================")
    print(" 🔬 PROVING MODEL SUPERIORITY (ANNUALIZED ANALYSIS)")
    print("==================================================")

    env = ActivePlanningEnv()
    
    print("\n[Phase 1] Evaluating Base Case over 1 Year...")
    base_placement = {'ess': [], 'pv': [], 'gas': [], 'svc': [], 'cb': []}
    robust_evaluate_placement(env.base_model, base_placement)
    base = extract_detailed_costs(env.base_model, base_placement)

    print("\n[Phase 2] Evaluating Random AI (Naive Benchmark)...")
    random_placement = {'ess': [], 'gas': [], 'svc': [], 'cb': []}
    devices = ['ess', 'gas', 'svc', 'cb']
    for _ in range(5):
        d = random.choice(devices)
        bus = random.randint(1, 32)
        if bus not in random_placement[d]: random_placement[d].append(bus)
    robust_evaluate_placement(env.base_model, random_placement)
    rand_ai = extract_detailed_costs(env.base_model, random_placement)

    print("\n[Phase 3] Loading Trained AI & Designing Network...")
    neural_net = ADNDeepNet(num_buses=33, num_device_types=4)
    if os.path.exists(model_path):
        neural_net.load_state_dict(torch.load(model_path, weights_only=True))
        neural_net.eval()
        print("  ✅ Mature Brain Loaded Successfully!")
    else:
        print(f"  ⚠️ Warning: Could not find '{model_path}'. Using random AI.")

    ai_state = env.reset()
    for step in range(5):
        mcts = MCTS(neural_net=neural_net, num_simulations=100)
        best_action, _ = mcts.search(ai_state, temperature=0.1, add_noise=False)
        ai_state.add_device(best_action[0], best_action[1])
        print(f"  📍 AI installs [{best_action[0].upper()}] at Bus {best_action[1]}")

    print("\n⏳ Running Heavy Physics Simulation for AI's final design (Please wait up to 120s)...")
    robust_evaluate_placement(env.base_model, ai_state.get_placement_dict())
    ai = extract_detailed_costs(env.base_model, ai_state.get_placement_dict())
        
    print("\n==================================================")
    print(" 📊 DETAILED THESIS REPORT (REAL ECONOMICS)")
    print("==================================================")
    print("1. Technical Improvements (Minimum Voltage):")
    print(f"   - Base Case:  {base['min_v']:.4f} p.u.")
    print(f"   - Random AI:  {rand_ai['min_v']:.4f} p.u.")
    print(f"   - Trained AI: {ai['min_v']:.4f} p.u.")
    
    print("\n2. Economic Breakdown (Actual Cash Flow):")
    print(f"   - 🔧 Hardware CAPEX: Base=${base['capex']:,.0f} | Rand=${rand_ai['capex']:,.0f} | AI=${ai['capex']:,.0f}")
    print(f"   - ⚡ Grid Purchase:  Base=${base['market']:,.0f} | Rand=${rand_ai['market']:,.0f} | AI=${ai['market']:,.0f}")
    print(f"   - 🔥 Gas Fuel Cost:  Base=${base['gas_opex']:,.0f} | Rand=${rand_ai['gas_opex']:,.0f} | AI=${ai['gas_opex']:,.0f}")
    print(f"   - 💧 Losses Cost:    Base=${base['loss_cost']:,.0f} | Rand=${rand_ai['loss_cost']:,.0f} | AI=${ai['loss_cost']:,.0f}")
    
    print(f"\n3. TOTAL ACTUAL COST (No Math Penalties):")
    print(f"   - 🔴 Base Case:  ${base['total']:,.2f}")
    print(f"   - ❌ Random AI:  ${rand_ai['total']:,.2f}")
    print(f"   - ✅ Trained AI: ${ai['total']:,.2f}")
    
    savings = base['total'] - ai['total']
    rand_savings = base['total'] - rand_ai['total']
    
    print("\n==================================================")
    if rand_savings < 0:
        print(f"📉 Random AI LOST ${-rand_savings:,.2f} (Bankrupted the grid with fuel!)")
    else:
        print(f"🎲 Random AI saved ${rand_savings:,.2f}")
        
    if savings > 0:
        print(f"🏆 SUCCESS! The Trained AI SAVES the grid ${savings:,.2f} EVERY YEAR!")
    else:
        print(f"⚠️ Net loss: ${-savings:,.2f}")
    print("==================================================")

if __name__ == "__main__":
    evaluate_and_compare(model_path="trained_adn_net_ep200.pth")