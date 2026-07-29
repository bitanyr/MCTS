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


logging.getLogger('pyomo.core').setLevel(logging.ERROR)

def verify_exact_physics(model):
    # SOCP and gap
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
    # Voltage violation
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
    
    episodes = 200
    checkpoint_interval = 50 
    
    
    INITIAL_TEMP = 1.0
    MIN_TEMP = 0.1
    DECAY_RATE = 0.98 
    
    print("==================================================")
    print("   Starting AlphaZero Self-Play Training Loop")
    print("   Physics Engine: EXACT NLP (Non-Convex DistFlow)")
    print(f"  Target Episodes: {episodes} (Checkpoints every {checkpoint_interval})")
    print("==================================================\n")

    # Calculating network costs in basic mode
    print("  Evaluating Base Network for Reward Scaling...")
    empty_placement = {'ess': [], 'pv': [], 'gas': [], 'svc': [], 'cb': []}
    is_base_feasible, base_cost = env.base_model_evaluate(empty_placement)
    
    if not is_base_feasible:
        print("   Warning: The network is infeasible in the Base Case!")
        base_cost = 1000000.0 # A default number to avoid division by zero errors
    else:
        print(f"   Base Cost established at: ${base_cost:,.2f}")
    print("--------------------------------------------------")

    for ep in range(episodes):
        print(f"\n--- Episode {ep+1}/{episodes} ---")
        state = env.reset()
        
        
        current_temp = max(MIN_TEMP, INITIAL_TEMP * (DECAY_RATE ** ep))
        print(f"  Current MCTS Temperature: {current_temp:.3f}")
        
        # Run MCTS with 400 simulations to give MCTS a chance to deepen
        mcts = MCTS(neural_net=net, num_simulations=400)
        
        episode_memory = []
        step = 0
        final_cost = 0.0 
        
        while True:
            if step >= 5:  
                break

            # Use dynamic temperature in search
            best_action, action_probs = mcts.search(state, temperature=current_temp, add_noise=True)
            
            if best_action is None:
                print("    No valid actions left. Ending episode.")
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
                
                print(f"       Feasible! Cost: ${final_cost:,.0f} | Gap: {gap:.2e} | Violations: {actual_perf:.4f}")
            else:
                print(f"       Blackout! Solver hit '{info['msg']}'. The grid collapsed. Pruning branch.")
                episode_memory[-1][3] = 100.0 
                final_cost = base_cost * 1.5 # Penalty equivalent to 50% worse than the baseline
                break 
            
            step += 1

        # Assign real reward (Value) using scaling formula
        for seq in episode_memory:
            s_tensor, t_policy, _, target_perf = seq
            
            if target_perf >= 100.0:
                 
                 scaled_value = -2.0 
            else:
                 
                 scaled_value = (base_cost - final_cost) / base_cost
                 
                 
                 scaled_value = max(-2.0, min(2.0, scaled_value))
                
            buffer.push(s_tensor, t_policy, scaled_value, target_perf)

        if len(buffer) > 32:
            epochs_per_episode = 10 
            for _ in range(epochs_per_episode):
                states, target_pis, target_values, target_perfs = buffer.sample(32)
                tot_loss, p_loss, v_loss = net.train_step(optimizer, states, target_pis, target_values, target_perfs)
                
            print(f"    [NN Update] Total Loss: {tot_loss:.4f} (Policy: {p_loss:.4f}, Value: {v_loss:.4f})")

        if (ep + 1) % checkpoint_interval == 0:
            checkpoint_name = f"trained_adn_net_checkpoint_ep{ep+1}.pth"
            torch.save(net.state_dict(), checkpoint_name)
            print(f"    [CHECKPOINT] Model saved safely to {checkpoint_name}")

    print("\n  Training Complete! Saving final model...")
    final_model_name = f"trained_adn_net_ep{episodes}.pth"
    torch.save(net.state_dict(), final_model_name)
    print(f" Final model saved successfully as: {final_model_name}")

if __name__ == "__main__":
    self_play()