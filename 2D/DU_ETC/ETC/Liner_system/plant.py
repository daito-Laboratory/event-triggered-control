import sys
import importlib.util
from contextlib import contextmanager
from pathlib import Path

import torch
from My_module.plant_base import PlantBase
from My_module.my_function_ET import derive_seed

# 連続時間システムなら0,離散時間システムなら1
system_type = 1
# システムの時間(+1)
T = 31

# システムのステップ数(離散時間システムなら適当でよい)
Step = 1

# 刻み幅
if system_type == 0:
    h = (T - 1) / Step
else:
    h = 1

# 状態の次元
N = 2

# 入力の次元
M = 1

# 評価関数のパラメータ
Q = 10 * torch.eye(2, dtype=torch.double)
Qf = 10 * torch.eye(2, dtype=torch.double)
R = torch.eye(1, dtype=torch.double)

# 離散時間線形システムのA,B(他のシステムなら適当でもよい)
A = torch.tensor([[1, 0.8], [0, 1.1]], dtype=torch.double)
B = torch.tensor([[1], [-1]], dtype=torch.double)


# ---------- parameter.py の読み込み ----------
def _load_parameter():
    # simulation.py / DETC.py が先に import していれば同じモジュールを共有する
    if "parameter" in sys.modules:
        return sys.modules["parameter"]
    path = Path(__file__).resolve().parent / "NN" / "parameter.py"
    if not path.exists():
        raise FileNotFoundError(f"parameter.py が見つかりません: {path}")
    spec = importlib.util.spec_from_file_location("parameter", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["parameter"] = module
    spec.loader.exec_module(module)
    return module


parameter = _load_parameter()

_REQUIRED = ["disturbance_enabled", "w_min", "w_max"]
_missing = [k for k in _REQUIRED if not hasattr(parameter, k)]
if _missing:
    raise AttributeError(f"parameter.py に未定義の項目があります: {_missing}")

# 外乱 x(k+1) = A x + B u + w(k)、w の各成分 ~ U[W_MIN, W_MAX]
W_ON = bool(parameter.disturbance_enabled)
W_MIN = float(parameter.w_min)
W_MAX = float(parameter.w_max)
if W_MIN > W_MAX:
    raise ValueError(f"w_min > w_max です: {W_MIN}, {W_MAX}")

# 乱数の seed（parameter.py）。学習中の外乱は train_seed、評価時の外乱は val_seed / test_seed で決まる
TRAIN_SEED = int(getattr(parameter, "train_seed", 1))
VAL_SEED = int(getattr(parameter, "val_seed", 1001))
TEST_SEED = int(getattr(parameter, "test_seed", 2001))
if VAL_SEED == TEST_SEED:
    raise ValueError("val_seed と test_seed は別の値にしてください")

print(f"[plant] disturbance={W_ON}, w~U[{W_MIN}, {W_MAX}], "
      f"seed(val/test)={VAL_SEED}/{TEST_SEED}, "
      f"from {Path(parameter.__file__).resolve()}")


# システムの状態方程式を連続時間風に記述(離散時間と連続時間の両立のため)
class Plant_over(PlantBase):  # 継承！
    def __init__(self, h, disturbance=None, train_seed=None):
        """
        disturbance: None なら parameter.py の設定に従う。True/False で個別に上書き
        train_seed : 学習中の外乱を決める seed。None なら parameter.py の train_seed
        """
        self.h = h
        self.disturbance = W_ON if disturbance is None else bool(disturbance)
        self.train_seed = TRAIN_SEED if train_seed is None else int(train_seed)
        self.gen = torch.Generator().manual_seed(derive_seed("train_noise", self.train_seed))

    @contextmanager
    def fixed_noise(self, index, stream="val"):
        """
        評価用。初期値の番号 index ごとに毎回同じ外乱系列を使う。
        stream="val" は best epoch の選択用（val_seed）、"test" は最終評価用（test_seed）。
        train_seed には依存しないので、学習の seed を変えても評価の外乱は変わらない。
        ブロックを抜けると学習用の乱数状態に戻す。
        """
        if stream == "val":
            seed = derive_seed("val_noise", VAL_SEED, int(index))
        elif stream == "test":
            seed = derive_seed("test_noise", TEST_SEED, int(index))
        else:
            raise ValueError(f"stream は 'val' か 'test': {stream!r}")
        state = self.gen.get_state()
        self.gen.manual_seed(seed)
        try:
            yield
        finally:
            self.gen.set_state(state)

    def x_next(self, x, u):
        # プライドは捨てる
        # x1 = x[0]
        # x2 = x[1]
        # x_next = torch.zeros_like(x, dtype=torch.double)
        # x_next[0] = 1.0 * x1 + 0.8 * x2 + 1.0 * u     # ← 非線形でもOK
        # x_next[1] = 1.1 * x2 - 1.0 * u
        x_next = A @ x + B @ u
        if self.disturbance:
            w = W_MIN + (W_MAX - W_MIN) * torch.rand(
                x_next.shape, generator=self.gen, dtype=x_next.dtype)
            x_next = x_next + w.to(x_next.device)
        return x_next
