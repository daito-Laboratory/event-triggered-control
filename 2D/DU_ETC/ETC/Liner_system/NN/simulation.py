import os
import re
import sys
import importlib.util

import torch
import matplotlib.pyplot as plt

# ---------- パス設定（起動場所に依存させない） ----------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                                 # NN/（parameter.py）
sys.path.append(os.path.abspath(os.path.join(_HERE, "..", "..", "..")))   # My_module
sys.path.append(os.path.abspath(os.path.join(_HERE, "..")))               # plant.py

import parameter                      # plant より先に読み込み、plant 側と共有する
from My_module import my_function_ET
import plant
from plant import Plant_over
from My_module.my_function_ET import derive_seed

# ---------- ハイパーパラメータの検証 ----------
_REQUIRED = ["epoch", "lr", "Folderpath", "min_val", "max_val", "lam", "Batch",
             "datasize", "AD", "evol", "Switch", "hidden_layer", "hidden_node"]
_missing = [k for k in _REQUIRED if not hasattr(parameter, k)]
if _missing:
    raise AttributeError(f"parameter.py に未定義の項目があります: {_missing}")
print(f"[simulation] parameter: {os.path.abspath(parameter.__file__)}")


# ---------- 評価する学習結果フォルダの決定 ----------
def _sig_tag():
    """DETC.py と同じ規則でフォルダ名を作る（DETC.py 側を変えたらここも合わせる）"""
    k_mode = getattr(parameter, "k_mode", "fixed")
    k_start = float(getattr(parameter, "k_start", 1.0))
    k_end = float(getattr(parameter, "k_end", 2.0))
    k_rate = float(getattr(parameter, "k_rate", 1.0))
    hard_forward = bool(getattr(parameter, "hard_forward", False))
    tag = f"T{plant.T - 1}_sig_{k_mode}_s{k_start:g}"
    if k_mode == "multiply":
        tag += f"_r{k_rate:g}"
    elif k_mode == "linear":
        tag += f"_end{k_end:g}"
    if hard_forward:
        tag += "_ste"
    return tag


if len(sys.argv) > 1:
    # 例: python simulation.py "<...>\T30_sig_fixed_s1_ste\20261011_235900_123456"
    run_dir = os.path.abspath(sys.argv[1])
else:
    # 引数が無ければ、現在の parameter.py の条件で最も新しい実行結果を使う
    base = os.path.join(parameter.Folderpath, _sig_tag())
    runs = sorted(d for d in os.listdir(base)
                  if os.path.exists(os.path.join(base, d, "result.py"))) if os.path.isdir(base) else []
    if not runs:
        raise FileNotFoundError(f"学習結果 (result.py) が見つかりません: {base}")
    run_dir = os.path.join(base, runs[-1])
run_dir = run_dir.rstrip("\\/") + os.sep
module_path = os.path.join(run_dir, "result.py")
if not os.path.exists(module_path):
    raise FileNotFoundError(f"result.py が見つかりません: {module_path}")
print(f"[simulation] result   : {module_path}")

spec = importlib.util.spec_from_file_location("result", module_path)
result = importlib.util.module_from_spec(spec)
spec.loader.exec_module(result)

K_best = torch.tensor(result.Best_K, dtype=torch.double)
num_hidden_best = int(result.Best_num_hidden)
node_hidden_best = int(result.Best_node_hidden)
hidden_W_best = [torch.tensor(w, dtype=torch.double) for w in result.Best_hidden_weights]
hidden_b_best = [torch.tensor(b, dtype=torch.double) for b in result.Best_hidden_biases]
out_W_best = torch.tensor(result.Best_out_weight, dtype=torch.double)
out_b_best = torch.tensor(result.Best_out_bias, dtype=torch.double)
# NN の入力の種類と最終評価の本数は、学習時の記録に合わせる（古い result.py には無いので従来の値）
nn_input = getattr(result, "NN_input", "x_xhat")
test_size = int(getattr(result, "Test_size", parameter.evol))
fig_count = int(getattr(parameter, "fig_count", 100))

# ---------- 学習時の条件と現在の設定の整合チェック ----------
if len(hidden_W_best) != num_hidden_best:
    raise ValueError(f"隠れ層の数が不一致: num_hidden={num_hidden_best}, 重みの数={len(hidden_W_best)}")
if (num_hidden_best, node_hidden_best) != (int(parameter.hidden_layer), int(parameter.hidden_node)):
    print(f"[warning] 学習済みNN ({num_hidden_best}層, {node_hidden_best}ノード) と "
          f"parameter.py ({parameter.hidden_layer}層, {parameter.hidden_node}ノード) が一致しません")
with open(module_path, encoding="utf-8", errors="ignore") as fp:
    result_text = fp.read()
for expected in (f"# disturbance_enabled = {plant.W_ON}",
                 f"# w_range = [{plant.W_MIN}, {plant.W_MAX}]"):
    if expected not in result_text:
        print(f"[warning] 学習時の外乱設定と現在の設定が異なる可能性があります（期待: {expected}）")

# ---------- 設定 ----------
T = plant.T
h = plant.h
N = plant.N
M = plant.M
systemtype = plant.system_type
Q = plant.Q
Qf = plant.Qf
R = plant.R

epoch = parameter.epoch
lr = parameter.lr
min_val = parameter.min_val
max_val = parameter.max_val
lam = parameter.lam
Batch = parameter.Batch
datasize = parameter.datasize
AD = parameter.AD
evolution = parameter.evol
Switch = parameter.Switch

out_dir = run_dir + "simulation" + os.sep   # DETC.py の図を上書きしない

Plant = Plant_over(h=h)
cstr = my_function_ET.CSTR(T, h, N, M, Q, Qf, R, Plant, systemtype, epoch, lr, out_dir,
                           min_val, max_val, lam, Batch, datasize, AD, evolution, Switch,
                           nn_input=nn_input)
cstr.hidden_layer = num_hidden_best
cstr.hidden_node = node_hidden_best

print(f"[simulation] T={T}, h={h}, N={N}, M={M}, Switch={Switch}, lam={lam}, "
      f"test_size={test_size}, NN 入力={nn_input}, x0~U[{min_val}, {max_val}], disturbance={Plant.disturbance}")

# ---------- 評価（DETC.py の Best 評価と同じ初期値・同じ外乱） ----------
total_cost_Best = 0
total_event_Best = 0
# 最終評価用の初期状態（DETC.py と同じく test_seed だけで決まる）
x0 = my_function_ET.Mike_x0(min_val, max_val, test_size, N,
                            seed=derive_seed("test_data", int(getattr(parameter, "test_seed", 2001))))
for i in range(test_size):
    with Plant.fixed_noise(i, stream="test"):
        cost, event = my_function_ET.Evolute_NN(
            cstr, T, x0[i], i, K_best,
            hidden_W_best, hidden_b_best, out_W_best, out_b_best,
            N, M, Q, R, num_hidden_best, node_hidden_best, out_dir,
            Qf=Qf, lam=lam, save_fig=(i < fig_count))
    total_cost_Best += cost
    total_event_Best += event
Average_cost_Best = total_cost_Best / test_size
Average_event_Best = total_event_Best / test_size
# コストは学習時と同じ評価関数 Σ(x'Qx + u'Ru + λδ) + x_T' Qf x_T
print("Average_cost", Average_cost_Best)
print("Average_event", Average_event_Best)

# ---------- DETC.py の記録との一致確認 ----------
cond_path = os.path.join(run_dir, "condition.txt")
if os.path.exists(cond_path):
    with open(cond_path, encoding="utf-8") as fp:
        m = re.search(r"# Best: Average_cost = ([^,]+), Average_event = (\S+)", fp.read())
    if m:
        rec_cost, rec_event = float(m.group(1)), float(m.group(2))
        ok = (abs(rec_cost - float(Average_cost_Best)) <= 1e-9 * max(1.0, abs(rec_cost))
              and abs(rec_event - float(Average_event_Best)) <= 1e-9)
        print(f"[simulation] DETC.py の記録: cost={rec_cost}, event={rec_event} -> "
              f"{'一致' if ok else '不一致（条件か外乱の扱いが学習時と異なる）'}")

plt.show()