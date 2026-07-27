# project_root/env/state.py

import copy

class NetworkState:
    def __init__(self, num_buses=33):
        self.num_buses = num_buses
        
        self.placements = {
            'ess': set(),
            'pv': set(),
            'gas': set(),
            'svc': set(),
            'cb': set()
        }

    def add_device(self, device_type, bus_index):
    
        if bus_index not in self.placements[device_type]:
            self.placements[device_type].add(bus_index)
            return True 
        return False 

    def get_placement_dict(self):
        return {k: list(v) for k, v in self.placements.items()}

    def clone(self):
        new_state = NetworkState(self.num_buses)
        new_state.placements = copy.deepcopy(self.placements)
        return new_state
