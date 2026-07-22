# project_root/env/aps_env.py

import sys
import os

# تضمین شناسایی ریشه پروژه برای سیستم Import پایتون
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from env.state import NetworkState
from optimization.model_builder import build_base_model, evaluate_placement

class ActivePlanningEnv:
    def __init__(self, num_buses=33):
        self.num_buses = num_buses
        self.state = NetworkState(num_buses)
        
        print("Initializing Physics Engine (Pyomo) for the RL Environment...")
        # ساخت مدل پایه Pyomo (فقط یک‌بار در شروع پروژه انجام می‌شود)
        self.base_model = build_base_model()

    def reset(self):
        """شروع مجدد اپیزود: پاک کردن تمام تجهیزات نصب شده"""
        self.state = NetworkState(self.num_buses)
        return self.state

    def step(self, action):
        """
        اعمال یک اقدام واقعی روی محیط اصلی
        action: ('pv', 15) یا ('ess', 12)
        """
        device_type, bus_index = action

        # ثبت دستگاه در وضعیت شبکه
        is_valid = self.state.add_device(device_type, bus_index)

        if not is_valid:
            return self.state, -50000, False, {"msg": "Invalid: Already installed"}

        # ارزیابی وضعیت جدید با حل‌گر ریاضی
        placement_dict = self.state.get_placement_dict()
        is_feasible, total_cost = evaluate_placement(self.base_model, placement_dict)

        if is_feasible:
            reward = -total_cost
            info = {"msg": "Feasible", "cost": total_cost}
        else:
            reward = -1e9
            info = {"msg": "Infeasible", "cost": float('inf')}

        done = False
        return self.state, reward, done, info

    def base_model_evaluate(self, placement_dict):
        """
        این متد کلیدی به فایل core/mcts.py اجازه می‌دهد بدون تغییر دادن 
        وضعیت اصلی شبیه‌ساز، چیدمان‌های فرضی درخت را با Pyomo ارزیابی کند.
        """
        return evaluate_placement(self.base_model, placement_dict)

if __name__ == "__main__":
    env = ActivePlanningEnv()
    print("Environment test passed successfully.")