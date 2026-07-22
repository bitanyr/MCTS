# project_root/data/devices.py

# --- Financial Parameters ---
C_LOSS = 200        # $/MWh
C_RES = 500         # $/MW 
C_AUL = 5000        # $/MW 

# Investment Costs (CapEx)
C_ESS_INV = 200000  # $/MW
C_GAS_INV = 30000   # $/MW
C_SVC_INV = 500000  # $/MVar
C_CB_INV = 40000    # $/MVar
C_PV_INV = 100000   # $/MW 

# --- Technical Parameters (Per-Unit based on 1MVA base) ---
P_ESS_MAX = 0.1     
E_ESS_MAX = 0.5     
E_ESS_MIN = 0.1     
EFF_CH = 0.9        
# 🟢 [FIXED] EFF_DIS باید یک راندمان واقعی (<=1) باشد، چون در constraints.py با
# فرمول "P_dis / EFF_DIS" استفاده می‌شود (تقسیم، نه ضرب). مقدار قبلی (1.1) با این
# فرمول یعنی دشارژ کردن باتری انرژی از هیچ تولید می‌کرد (تست شد: راندمان رفت‌وبرگشت
# واقعی 116% درمی‌آمد، که از نظر فیزیکی غیرممکن است — بقای انرژی را نقض می‌کند).
# با EFF_CH=0.9 و EFF_DIS=0.9 (راندمان یک‌طرفه‌ی متقارن)، راندمان رفت‌وبرگشت کل
# می‌شود 0.9*0.9=81%، که با مقادیر معمول باتری‌های لیتیومی در ادبیات سازگار است.
EFF_DIS = 0.9       

PV_CAPACITY = 0.5   

P_GAS_MAX = 0.5     
Q_SVC_MAX = 0.5     
Q_CB_MAX = 0.5      

# Grid Limits
V_MIN_SQ = 0.95**2
V_MAX_SQ = 1.05**2
# 🔴 اصلاح شد: حد حرارتی به 25.0 تغییر کرد (معادل 5 MVA). 
# اکنون ریشه شبکه خفه نمی‌شود و هوش مصنوعی مجبور است برای رفع افت ولتاژ، تجهیزات را به انتهای فیدر ببرد.
I_MAX_SQ = 25.0
