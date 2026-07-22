import pyomo.environ as pyo
from data.ieee33 import BRANCHES, S_BASE
from data.devices import *
from data.scenarios import RTP_PRICE

FIXED_PV_NODES = [8, 10, 13, 16, 18, 20, 22, 28]

def define_objective(model):
    def objective_rule(m):
        # ----------------------------------------------------------------
        # 1. Investment Costs (CAPEX)
        # ----------------------------------------------------------------
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
        
        # ----------------------------------------------------------------
        # 2. Operational Costs (OPEX)
        # ----------------------------------------------------------------
        DAYS_MULTIPLIER = 365.0 / 4.0
        
        # 🟢 [FIXED] Dual Pricing: واردات با نرخ گرانِ RTP، صادرات با نرخ ارزانِ
        # feed-in tariff (نه همون قیمت خرید!) -- دقیقاً مثل قراردادهای واقعیِ اتصال
        # DER با شبکهٔ بالادست. این جلوی سودآوریِ آربیتراژِ گاز رو می‌گیره: تولید با
        # C_GAS_FUEL+C_EMISSION=۷۰$/MWh و فروش با ۲۰$/MWh یعنی هر MWh که فقط برای
        # صادرات تولید بشه، ۵۰$ ضرر خالص داره.
        # نکته: چون کف قیمت RTP (در scenarios.py) دقیقاً ۲۰$/MWh است، در همون ساعاتِ
        # کف‌قیمت این دو عدد برابر می‌شن -- یعنی در آن ساعات خاص، جواب بهینه ممکن است
        # هم‌زمان P_sub_import و P_sub_export را غیرصفر نشان دهد (چون فرقی نمی‌کند)،
        # بدون این‌که روی هزینه‌ی کل یا مقدار واقعیِ P_sub اثر بگذارد. اگر خواستید این
        # حالت مرزی را کاملاً حذف کنید، کافی‌ست C_EXPORT_PRICE را کمی کمتر از ۲۰ بگذارید.
        C_EXPORT_PRICE = 20.0  # $/MWh - نرخ عمده‌فروشی/feed-in، نه نرخ خرده‌فروشیِ RTP
        market_cost = sum(
            (RTP_PRICE[t] * m.P_sub_import[t] - C_EXPORT_PRICE * m.P_sub_export[t]) * S_BASE
            for t in m.T
        ) * DAYS_MULTIPLIER
        
        # 🔴 [FIXED] بخش loss_cost حذف شد تا از Double-Counting هزینه تلفات جلوگیری شود.
        
        # هزینه استهلاک باتری به ازای هر سیکل شارژ و دشارژ
        C_DEG = 15.0 # $/MWh 
        degradation_cost = sum(
            C_DEG * (m.P_ch[i, t] + m.P_dis[i, t]) * S_BASE
            for i in m.N for t in m.T
        ) * DAYS_MULTIPLIER

        # جریمه اقتصادی برای قطع بار یا قطع تولید خورشیدی
        curtailment_cost = sum(
            C_RES * m.P_curt_res[i, t] * S_BASE + 
            C_AUL * m.P_curt_aul[i, t] * S_BASE 
            for i in m.N for t in m.T
        ) * DAYS_MULTIPLIER
        
        # هزینه سوخت و آلایندگی توربین‌های گازی
        C_GAS_FUEL = 50.0  
        C_EMISSION = 20.0  
        gas_opex_cost = sum(
            (C_GAS_FUEL + C_EMISSION) * m.P_gas[i, t] * S_BASE 
            for i in m.N for t in m.T
        ) * DAYS_MULTIPLIER

        # ----------------------------------------------------------------
        # 3. Soft Constraint Penalties (Physics Violations)
        # ----------------------------------------------------------------
        # 🔴 [FIXED] حذف try/exceptها به دلیل تعریف صریح متغیرها در variables.py
        
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

        # ----------------------------------------------------------------
        # Total Objective Function
        # ----------------------------------------------------------------
        return (annual_inv_cost + market_cost + curtailment_cost + gas_opex_cost + 
                degradation_cost + volt_penalty_cost + overload_penalty_cost + thermal_penalty_cost)

    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
    return model