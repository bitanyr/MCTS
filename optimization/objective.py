import pyomo.environ as pyo
from data.ieee33 import BRANCHES, S_BASE
from data.devices import *
from data.scenarios import RTP_PRICE

FIXED_PV_NODES = [8, 10, 13, 16, 18, 20, 22, 28]

def define_objective(model):
    def objective_rule(m):
        interest_rate = 0.05
        lifetime = 20
        CRF = (interest_rate * (1 + interest_rate)**lifetime) / (((1 + interest_rate)**lifetime) - 1)
        
        inv_cost = sum(
            C_ESS_INV * P_ESS_MAX * m.s_ess[i] +
            C_GAS_INV * P_GAS_MAX * m.s_gas[i] +
            C_SVC_INV * Q_SVC_MAX * m.s_svc[i] +
            C_CB_INV * Q_CB_MAX * m.s_cb[i] +
            (C_PV_INV * PV_CAPACITY if i in FIXED_PV_NODES else 0) 
            for i in m.N
        )
        annual_inv_cost = inv_cost * CRF
        
        DAYS_MULTIPLIER = 365.0 / 4.0
        
        C_EXPORT_PRICE = 19.99 
        market_cost = sum(
            (RTP_PRICE[t] * m.P_sub_import[t] - C_EXPORT_PRICE * m.P_sub_export[t]) * S_BASE
            for t in m.T
        ) * DAYS_MULTIPLIER
        
        C_DEG = 15.0 
        degradation_cost = sum(
            C_DEG * (m.P_ch[i, t] + m.P_dis[i, t]) * S_BASE
            for i in m.N for t in m.T
        ) * DAYS_MULTIPLIER

        curtailment_cost = sum(
            C_RES * m.P_curt_res[i, t] * S_BASE + 
            C_AUL * m.P_curt_aul[i, t] * S_BASE 
            for i in m.N for t in m.T
        ) * DAYS_MULTIPLIER
        
        C_GAS_FUEL = 50.0  
        C_EMISSION = 20.0  
        gas_opex_cost = sum(
            (C_GAS_FUEL + C_EMISSION) * m.P_gas[i, t] * S_BASE 
            for i in m.N for t in m.T
        ) * DAYS_MULTIPLIER

        PENALTY_VOLT = 1e5
        volt_penalty_cost = sum(
            PENALTY_VOLT * (m.v_viol_down[i, t] + m.v_viol_up[i, t]) * S_BASE
            for i in m.N for t in m.T
        ) * DAYS_MULTIPLIER
        
        PENALTY_OVERLOAD = 1e6
        overload_penalty_cost = sum(
            PENALTY_OVERLOAD * m.sub_overload[t] 
            for t in m.T
        ) * DAYS_MULTIPLIER

        PENALTY_THERMAL = 1e5
        thermal_penalty_cost = sum(
            PENALTY_THERMAL * m.l_viol[k, t] 
            for k in m.E for t in m.T
        ) * DAYS_MULTIPLIER

        PENALTY_SOC = 1e5
        soc_penalty_cost = sum(
            PENALTY_SOC * (m.soc_viol_down[i] + m.soc_viol_up[i]) * S_BASE
            for i in m.N
        ) * DAYS_MULTIPLIER

        return (annual_inv_cost + market_cost + curtailment_cost + gas_opex_cost + 
                degradation_cost + volt_penalty_cost + overload_penalty_cost + 
                thermal_penalty_cost + soc_penalty_cost) 

    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
    return model