import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pyomo.environ as pyo
from optimization.variables import define_variables
from optimization.constraints import define_constraints
from optimization.objective import define_objective

def build_base_model():
    print("Building Base Pyomo Model (This takes a few seconds)...")
    model = pyo.ConcreteModel()
    
    model = define_variables(model)
    model = define_constraints(model)
    model = define_objective(model)
    
    return model

def evaluate_placement(model, placement_dict):
    # 1. Reset placements
    # 🔴 [FIXED] استفاده از set_value برای جلوگیری از تخریب اشیاء Pyomo
    for i in model.N:
        model.s_ess[i].set_value(0)
        model.s_gas[i].set_value(0)
        model.s_svc[i].set_value(0)
        model.s_cb[i].set_value(0)

    # 2. Apply new placements
    # 🔴 [FIXED] اعمال تجهیزات با حفظ پیوندهای ریاضی
    for node in placement_dict.get('ess', []): model.s_ess[node].set_value(1)
    for node in placement_dict.get('gas', []): model.s_gas[node].set_value(1)
    for node in placement_dict.get('svc', []): model.s_svc[node].set_value(1)
    for node in placement_dict.get('cb', []):  model.s_cb[node].set_value(1)

    # مقداردهی اولیه دقیق (Interior-Point Warm Start)
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
            
            if pyo.value(model.s_gas[i]) == 1:
                model.P_gas[i, t].set_value(0.01)
            else:
                model.P_gas[i, t].set_value(0.0)
                
            model.Q_svc[i, t].set_value(0.0)
            
            if pyo.value(model.s_cb[i]) == 1:
                model.Q_cb[i, t].set_value(0.01)
            else:
                model.Q_cb[i, t].set_value(0.0)
            
            if hasattr(model, 'v_viol_down'): model.v_viol_down[i, t].set_value(0.001)
            if hasattr(model, 'v_viol_up'): model.v_viol_up[i, t].set_value(0.001)

        for k in model.E:
            model.P[k, t].set_value(0.01)
            model.Q[k, t].set_value(0.01)
            # 🔴 [FIXED] تضاد ریاضی برطرف شد: l = P^2 + Q^2 = 0.0002
            model.l[k, t].set_value(0.0002)
            if hasattr(model, 'l_viol'): 
                model.l_viol[k, t].set_value(0.001)
            
        model.P_sub[t].set_value(0.1)
        model.Q_sub[t].set_value(0.1)
        if hasattr(model, 'P_sub_import'): model.P_sub_import[t].set_value(0.1)
        if hasattr(model, 'P_sub_export'): model.P_sub_export[t].set_value(0.0)
        
        if hasattr(model, 'sub_overload'): 
            model.sub_overload[t].set_value(0.001)

    # 3. Solver Configuration
    solver = pyo.SolverFactory('ipopt')
    # 🔴 [FIXED] افزایش زمان سالور برای حل یک شبکه غیرخطی بزرگ
    solver.options['max_iter'] = 3000
    solver.options['max_cpu_time'] = 120.0 
    solver.options['tol'] = 1e-4
    solver.options['print_level'] = 0
    
    # 🔴 جراحی ریاضی: تنظیمات فوق‌حیاتی برای جلوگیری از سینگولار شدن و تنظیم مقیاس تابع هدف
    solver.options['mu_strategy'] = 'adaptive'
    solver.options['obj_scaling_factor'] = 1e-5
    solver.options['bound_push'] = 1e-6

    if not solver.available():
        idaes_path = os.path.join(os.path.expanduser('~'), '.idaes', 'bin', 'ipopt.exe')
        if os.path.exists(idaes_path):
            solver = pyo.SolverFactory('ipopt', executable=idaes_path)
        else:
            fallback_path = r'C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\Scripts\ipopt.exe'
            if os.path.exists(fallback_path):
                solver = pyo.SolverFactory('ipopt', executable=fallback_path)
            else:
                 raise RuntimeError(f"Cannot find ipopt.exe in {idaes_path} or {fallback_path}")

    try:
        result = solver.solve(model, tee=False)
        if (result.solver.status == pyo.SolverStatus.ok) and \
           (result.solver.termination_condition == pyo.TerminationCondition.optimal):
            total_cost = pyo.value(model.obj)
            return True, total_cost
        else:
            return False, float('inf')
    except Exception as e:
        return False, float('inf')

if __name__ == "__main__":
    my_model = build_base_model()
    print("\n--- Test 1: No Extra Devices (Only Fixed PVs) ---")
    is_feasible, cost = evaluate_placement(my_model, {})
    if is_feasible:
        print(f"Network is Feasible. Total Cost: ${cost:,.2f}")
    else:
        print("Network is INFEASIBLE.")