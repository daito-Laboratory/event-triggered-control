import torch
from My_module.plant_base import PlantBase

# 連続時間システムなら0,離散時間システムなら1
system_type = 1
# システムの時間(+1)
T = 31

# システムのステップ数(離散時間システムなら適当でよい)
Step = 1

# 刻み幅
if system_type == 0:
    h = (T-1)/Step
else:
    h = 1

# 状態の次元
N = 2

# 入力の次元
M = 1

# 評価関数のパラメータ
Q = 10*torch.eye(2, dtype=torch.double)
Qf = 10*torch.eye(2, dtype=torch.double)
R = torch.eye(1, dtype=torch.double)

# 離散時間線形システムのA,B(他のシステムなら適当でもよい)
A = torch.tensor([[1, 0.8], [0, 1.1]], dtype=torch.double)
B = torch.tensor([[1], [-1]], dtype=torch.double)

# 外乱 x(k+1) = A x + B u + w(k)、w の各成分 ~ U[W_MIN, W_MAX]
# 設定は parameter.py（disturbance_enabled, w_min, w_max）。無ければ外乱なし（元の動作）
try:
    import parameter
    W_ON = bool(getattr(parameter, "disturbance_enabled", False))
    W_MIN = float(getattr(parameter, "w_min", 0.0))
    W_MAX = float(getattr(parameter, "w_max", 0.0))
except ImportError:
    W_ON, W_MIN, W_MAX = False, 0.0, 0.0


# システムの状態方程式を連続時間風に記述(離散時間と連続時間の両立のため)
class Plant_over(PlantBase):  # 継承！
    def __init__(self, h):
        self.h = h

    def x_next(self, x, u):
        # プライドは捨てる
        # x1 = x[0]
        # x2 = x[1]
        # x_next = torch.zeros_like(x, dtype=torch.double)
        # x_next[0] = 1.0 * x1 + 0.8 * x2 + 1.0 * u     # ← 非線形でもOK
        # x_next[1] = 1.1 * x2 - 1.0 * u
        x_next = A@x+B@u
        if W_ON:
            w = W_MIN + (W_MAX - W_MIN) * torch.rand(x.shape, dtype=x_next.dtype, device=x_next.device)
            x_next = x_next + w

        return x_next