import pyomo.environ as pyo
import numpy as np
import torch.optim as optim
import logging
import torch
import os
from core.network import ADNDeepNet
from core.mcts import MCTS
from core.replay_buffer import ReplayBuffer
from env.aps_env import ActivePlanningEnv

# غیرفعال کردن هشدارهای زرد رنگ سالور در کنسول
logging.getLogger('pyomo.core').setLevel(logging.ERROR)

def verify_exact_physics(model):
    """ بررسی فیزیک دقیق شبکه بدون گپ SOCP """
    try:
        max_gap = 0.0
        from data.ieee33 import BRANCHES
        for k in model.E:
            for t in model.T:
                fb = BRANCHES[k]['from']
                l_val = pyo.value(model.l[k, t])
                v_val = pyo.value(model.v[fb, t])
                P_val = pyo.value(model.P[k, t])
                Q_val = pyo.value(model.Q[k, t])
                
                gap = abs((l_val * v_val) - (P_val**2 + Q_val**2))
                if gap > max_gap:
                    max_gap = gap
        return max_gap
    except Exception as e:
        return -1.0

def calculate_performance_index(model):
    """ محاسبه شاخص عملکرد واقعی (تخطی ولتاژ) از خروجی Pyomo """
    perf = 0.0
    try:
        for i in model.N:
            for t in model.T:
                perf += pyo.value(model.v_viol_down[i, t]) + pyo.value(model.v_viol_up[i, t])
    except Exception:
        pass
    return max(0.0, perf)

def self_play():
    env = ActivePlanningEnv()
    net = ADNDeepNet(num_buses=33, num_device_types=4)
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
    
    buffer = ReplayBuffer(capacity=10000) 
    
    # 🔴 [FIXED] تنظیمات قطعی برای ران اصلی
    episodes = 200 
    checkpoint_interval = 50 # ذخیره هر 50 اپیزود
    
    print("==================================================")
    print("🚀 Starting AlphaZero Self-Play Training Loop")
    print("   Physics Engine: EXACT NLP (Non-Convex DistFlow)")
    print(f"   Target Episodes: {episodes} (Checkpoints every {checkpoint_interval})")
    print("==================================================\n")

    for ep in range(episodes):
        print(f"\n--- Episode {ep+1}/{episodes} ---")
        state = env.reset()
        
        # اجرای MCTS با 100 شبیه‌سازی برای عمق جستجوی استاندارد
        mcts = MCTS(neural_net=net, num_simulations=100)
        
        episode_memory = []
        step = 0
        final_cost = 0.0 
        
        while True:
            # 🔴 [FIXED] بودجه ۵ استپ دقیقاً هماهنگ با evaluate_model.py
            if step >= 5:  
                break

            # 🔴 [FIXED] استفاده از Temperature = 0.5 برای تیز کردن توزیع و سقوط قطعی Policy Loss
            best_action, action_probs = mcts.search(state, temperature=0.5, add_noise=True)
            
            if best_action is None:
                print("   ⚠️ No valid actions left. Ending episode.")
                break
                
            state_tensor = mcts.state_to_tensor(state)
            
            episode_memory.append([state_tensor, action_probs, best_action, 0.0])

            print(f"   [Step {step+1}] AI chosen action -> Device: {best_action[0].upper()}, Bus: {best_action[1]}")
            state, reward, is_infeasible_done, info = env.step(best_action)
            
            if info["msg"] == "Feasible":
                gap = verify_exact_physics(env.base_model)
                actual_perf = calculate_performance_index(env.base_model)
                
                episode_memory[-1][3] = actual_perf
                final_cost = info['cost'] 
                
                print(f"      ✅ Feasible! Cost: ${final_cost:,.0f} | Gap: {gap:.2e} | Voltage Violations: {actual_perf:.4f}")
            else:
                print(f"      🚨 Blackout! Solver hit '{info['msg']}'. The grid collapsed. Pruning branch.")
                episode_memory[-1][3] = 100.0 # جریمه سنگین فیزیکی
                final_cost = 1000000.0 # جریمه یک میلیون دلاری
                break 
            
            step += 1

        # تخصیص پاداش واقعی (Value) به تمام گره‌های این اپیزود
        for seq in episode_memory:
            s_tensor, t_policy, _, target_perf = seq
            if target_perf >= 100.0:
                 scaled_value = -10.0 # بدترین حالت ممکن در صورت بلک‌اوت
            else:
                 # اسکیل کردن هزینه برای شبکه عصبی (100 هزار دلار معادل 1.0-)
                 scaled_value = max(-10.0, - (final_cost / 100000.0))
                
            buffer.push(s_tensor, t_policy, scaled_value, target_perf)

        if len(buffer) > 32:
            states, target_pis, target_values, target_perfs = buffer.sample(32)
            tot_loss, p_loss, v_loss = net.train_step(optimizer, states, target_pis, target_values, target_perfs)
            print(f"   🧠 [NN Update] Total Loss: {tot_loss:.4f} (Policy: {p_loss:.4f}, Value: {v_loss:.4f})")

        # 🔴 [FIXED] ذخیره چک‌پوینت‌ها مستقیماً در پوشه اصلی تا ارور ندهد
        if (ep + 1) % checkpoint_interval == 0:
            checkpoint_name = f"trained_adn_net_checkpoint_ep{ep+1}.pth"
            torch.save(net.state_dict(), checkpoint_name)
            print(f"   💾 [CHECKPOINT] Model saved safely to {checkpoint_name}")

    print("\n🎉 Training Complete! Saving final model...")
    # 🔴 [FIXED] این خط کامنت بود! از کامنت درآوردم تا فایل نهایی حتماً سیو شود.
    final_model_name = f"trained_adn_net_ep{episodes}.pth"
    torch.save(net.state_dict(), final_model_name)
    print(f"✅ Final model saved successfully as: {final_model_name}")

if __name__ == "__main__":
    self_play()