import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    # 🔴 آپدیت: اضافه شدن پارامتر perf (Performance Index)
    def push(self, state, pi, value, perf):
        """ذخیره تجربه جدید در حافظه (شامل شاخص عملکرد فیزیکی)"""
        self.buffer.append((state, pi, value, perf))

    def sample(self, batch_size):
        """استخراج تصادفی یک مینی‌بچ برای آموزش شبکه عصبی"""
        batch = random.sample(self.buffer, batch_size)
        # 🔴 آپدیت: استخراج perf در کنار بقیه داده‌ها
        states, pis, values, perfs = zip(*batch)
        return list(states), list(pis), list(values), list(perfs)

    def __len__(self):
        """برگرداندن تعداد تجربیات موجود در بافر"""
        return len(self.buffer)