# project_root/env/state.py

import copy

class NetworkState:
    def __init__(self, num_buses=33):
        self.num_buses = num_buses
        # این دیکشنری نگه می‌دارد که در کدام شین‌ها چه تجهیزاتی نصب شده است
        # از set استفاده کردیم تا از نصب تکراری یک دستگاه در یک شین جلوگیری شود
        self.placements = {
            'ess': set(),
            'pv': set(),
            'gas': set(),
            'svc': set(),
            'cb': set()
        }

    def add_device(self, device_type, bus_index):
        """تلاش برای نصب یک تجهیز جدید در یک شین"""
        if bus_index not in self.placements[device_type]:
            self.placements[device_type].add(bus_index)
            return True # نصب موفق
        return False # از قبل اینجا نصب شده بود

    def get_placement_dict(self):
        """تبدیل به فرمت دیکشنری ساده برای ارسال به Pyomo"""
        return {k: list(v) for k, v in self.placements.items()}

    def clone(self):
        """
        کپی کردن وضعیت فعلی.
        این تابع برای درخت جستجوی مونت‌کارلو (MCTS) بی‌نهایت مهم است!
        """
        new_state = NetworkState(self.num_buses)
        new_state.placements = copy.deepcopy(self.placements)
        return new_state
