import math

class MCTSNode:
    def __init__(self, state, parent=None, action=None, prior_prob=1.0):
        self.state = state          # وضعیت شبکه در این گره
        self.parent = parent        # گره پدر
        self.action = action        # اکشنی که باعث رسیدن به این گره شد
        self.children = {}          # دیکشنری از فرزندان {action: MCTSNode}
        
        # مقادیر مهم برای الگوریتم MCTS
        self.visit_count = 0        # تعداد دفعات بازدید (N)
        self.value_sum = 0.0        # مجموع پاداش‌های کسب شده (W)
        self.prior_prob = prior_prob # احتمال اولیه پیشنهاد شده توسط شبکه عصبی (P)
        
    @property
    def q_value(self):
        """میانگین پاداش کسب شده در این مسیر"""
        if self.visit_count == 0:
            # 🔴 اصلاح طلایی (First Play Urgency)
            # جلوگیری از توهم هزینه صفر در مسائل Cost-Minimization
            # اگر گره هنوز بازدید نشده، ارزش شبکه فعلی (پدر) را به عنوان تخمین اولیه قرض می‌گیرد
            if self.parent and self.parent.visit_count > 0:
                return self.parent.q_value
            return -10.0 # یک عدد منفی پایه برای اولین گره ریشه
        
        return self.value_sum / self.visit_count

    def is_expanded(self):
        """آیا این گره قبلاً باز شده است؟"""
        return len(self.children) > 0

    def get_ucb(self, c_puct=1.0):
        """
        محاسبه مقدار Upper Confidence Bound (معادله PUCT در مقاله)
        ترکیبی از استخراج (Q) و اکتشاف (U)
        """
        # جلوگیری از خطای احتمالی NoneType در صورتی که گره ریشه اشتباهاً ارزیابی شود
        parent_visits = self.parent.visit_count if self.parent else 1
        
        # U = c_puct * P * sqrt(N_parent) / (1 + N_child)
        u_value = (c_puct * self.prior_prob * math.sqrt(parent_visits) / (1 + self.visit_count))
        
        return self.q_value + u_value

    def best_child(self, c_puct=1.0):
        """انتخاب فرزندی که بیشترین مقدار PUCT را دارد"""
        if not self.children:
            return None
            
        # برگرداندن (action, node) با بیشترین ارزش
        return max(self.children.items(), 
                   key=lambda child_item: child_item[1].get_ucb(c_puct))