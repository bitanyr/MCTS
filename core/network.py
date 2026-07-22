import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ADNDeepNet(nn.Module):
    def __init__(self, num_buses=33, num_device_types=4):
        """
        شبکه عصبی با 3 سر خروجی کاملاً مستقل
        فضای اکشن: 4 نوع دستگاه در 33 شین = 132 بُعد
        """
        super(ADNDeepNet, self).__init__()
        self.num_buses = num_buses
        self.num_device_types = num_device_types
        self.input_dim = num_buses * num_device_types  # 132

        # 🔴 [CRITICAL FIX] جداسازی کامل شبکه‌ها (Decoupling)
        # برای جلوگیری از له شدن Policy زیر گرادیان‌های سنگینِ Value، 
        # لایه‌های مشترک را حذف کردیم و به هر سر، لایه اختصاصی دادیم.

        # 1. Policy Network (مستقل)
        self.pol_fc1 = nn.Linear(self.input_dim, 256)
        self.pol_fc2 = nn.Linear(256, 128)
        self.pol_out = nn.Linear(128, self.input_dim)

        # 2. Value Network (مستقل)
        self.val_fc1 = nn.Linear(self.input_dim, 256)
        self.val_fc2 = nn.Linear(256, 64)
        self.val_out = nn.Linear(64, 1)

        # 3. Performance Network (مستقل)
        self.prf_fc1 = nn.Linear(self.input_dim, 128)
        self.prf_fc2 = nn.Linear(128, 64)
        self.prf_out = nn.Linear(64, 1)

    def forward(self, x):
        # Policy Head
        p = F.relu(self.pol_fc1(x))
        p = F.relu(self.pol_fc2(p))
        policy_probs = F.softmax(self.pol_out(p), dim=-1)

        # Value Head
        v = F.relu(self.val_fc1(x))
        v = F.relu(self.val_fc2(v))
        value = 10.0 * torch.tanh(self.val_out(v))
        
        # Performance Head
        pr = F.relu(self.prf_fc1(x))
        pr = F.relu(self.prf_fc2(pr))
        perf_index = F.relu(self.prf_out(pr))

        return policy_probs, value, perf_index

    def predict(self, state_tensor):
        self.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_tensor)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
                
            policy_probs, value, perf_index = self.forward(state_tensor)
            
        return policy_probs.squeeze(0).numpy(), value.item(), perf_index.item()

    def train_step(self, optimizer, states, target_pis, target_values, target_perfs=None):
        self.train()
        optimizer.zero_grad()

        states = torch.FloatTensor(np.array(states))
        target_pis = torch.FloatTensor(np.array(target_pis))
        target_values = torch.FloatTensor(np.array(target_values)).unsqueeze(1)
        
        if target_perfs is None:
            target_perfs = torch.zeros_like(target_values)
        else:
            target_perfs = torch.FloatTensor(np.array(target_perfs)).unsqueeze(1)

        policy_probs, values, perfs = self.forward(states)

        # 1. Policy Loss (Cross-Entropy)
        loss_policy = -torch.sum(target_pis * torch.log(policy_probs + 1e-8)) / states.size(0)
        
        # 🔴 [FIXED] استفاده از Smooth L1 (Huber Loss) به جای MSE
        # این کار از انفجار اعداد (مثل Loss=42) به شدت جلوگیری می‌کند
        loss_value = F.smooth_l1_loss(values, target_values)
        loss_perf = F.smooth_l1_loss(perfs, target_perfs)

        # ترکیب نهایی زیان‌ها
        total_loss = loss_policy + loss_value + 0.1 * loss_perf
        
        total_loss.backward()
        optimizer.step()

        return total_loss.item(), loss_policy.item(), loss_value.item()