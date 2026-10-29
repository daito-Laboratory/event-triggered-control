import parameter
from datetime import datetime
import torch  # 機械学習用のライブラリ
import torch.nn as nn  # ニューラルネットワーク構築
import torch.optim as optim  # 最適化
import matplotlib.pyplot as plt  # グラフのプロット
from torch.utils.data import Dataset  # 訓練5データの生成
from torch.utils.data import DataLoader  # 訓練データのロード
import os  # 位置参照に使用
from openpyxl import Workbook, load_workbook  # Excelにデータを保存
import winsound  # 音を鳴らす
import sys
# my_module のパスを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from My_module import my_function_ET
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import plant
from plant import Plant_over
# my_module のモジュールをインポート
# ?----------------------------------------------------------------パラメータ入手

# TODO----------------------------------------------------------------プラントのパラメータ
T= plant.T
h=plant.h
N=plant.N
M= plant.M
systemtype= plant.system_type
Q = plant.Q
Qf = plant.Qf
R = plant.R
# TODO----------------------------------------------------------------ハイパーパラメータ
k_mode = getattr(parameter, "k_mode", "fixed")
k_start = float(getattr(parameter, "k_start", 1.0))
k_end = float(getattr(parameter, "k_end", 2.0))
k_rate = float(getattr(parameter, "k_rate", 1.0))
hard_forward = bool(getattr(parameter, "hard_forward", False))
if k_mode not in ("fixed", "multiply", "linear"):
    raise ValueError(f"k_mode must be fixed, multiply or linear, got {k_mode!r}")
if k_start <= 0 or k_end <= 0 or k_rate <= 0:
    raise ValueError("k_start, k_end and k_rate must all be positive")
# k の設定と毎回異なる日時フォルダで分ける（同じ条件の再実行も上書きしない）。
sig_tag = f"T{T-1}_sig_{k_mode}_s{k_start:g}"
if k_mode == "multiply":
    sig_tag += f"_r{k_rate:g}"
elif k_mode == "linear":
    sig_tag += f"_end{k_end:g}"
if hard_forward:
    sig_tag += "_ste"
run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
path = os.path.join(parameter.Folderpath, sig_tag, run_id) + os.sep
os.makedirs(path, exist_ok=False)
epoch = parameter.epoch
lr = parameter.lr
min_val = parameter.min_val
max_val = parameter.max_val
lam = parameter.lam
Batch = parameter.Batch
datasize = parameter.datasize
AD = parameter.AD
evolution= parameter.evol
Switch = parameter.Switch

# 学習条件（plant.py で実際に使われる外乱の設定を記録する）
cond_text = (
    "# ===== Training conditions =====\n"
    f"# disturbance_enabled = {plant.W_ON}\n"
    f"# w_range = [{plant.W_MIN}, {plant.W_MAX}]\n"
    f"# lam = {lam}, T = {T}, epoch = {epoch}, lr = {lr}, AD = {AD}\n"
    f"# x0_range = [{min_val}, {max_val}], datasize = {datasize}, Batch = {Batch}, evol = {evolution}\n"
    f"# Q = {Q.diag().tolist()}, Qf = {Qf.diag().tolist()}, R = {R.diag().tolist()}\n"
    f"# hidden_layer = {parameter.hidden_layer}, hidden_node = {parameter.hidden_node}, seed = 1\n"
    f"# sigmoid: mode = {k_mode}, k_start = {k_start}, k_rate = {k_rate}, k_end = {k_end}\n"
    f"# hard_forward (straight-through) = {hard_forward}\n"
    f"# sigmoid: first_k = {k_start}, last_k = "
    f"{(k_start if k_mode == 'fixed' else k_start * (k_rate ** (epoch - 1)) if k_mode == 'multiply' else k_end):.6g}\n"
)
print(cond_text)

my_function_ET.seedinit(1)
# TODO----------------------------------------------------------------訓練開始
Plant=Plant_over(h=h)
cstr = my_function_ET.CSTR(T,h,N,M,Q,Qf,R,Plant,systemtype,epoch,lr,path,min_val,max_val,lam,Batch,datasize,AD,evolution,Switch)
cstr.hidden_layer=parameter.hidden_layer
cstr.hidden_node=parameter.hidden_node
cstr.k_mode = k_mode
cstr.k_start = k_start
cstr.k_end = k_end
cstr.k_rate = k_rate
cstr.hard_forward = hard_forward
data = cstr.utrain()
# print(data)
(
    K_best,
    num_hidden_best,
    node_hidden_best,
    hidden_W_best,
    hidden_b_best,
    out_W_best,
    out_b_best,
    K_last,
    num_hidden_last,
    node_hidden_last,
    hidden_W_last,
    hidden_b_last,
    out_W_last,
    out_b_last,
    training_time,
) = data

# print(hidden_W_best)
# print(hidden_b_best)

# print("best_A_w", best_A_w)
# print("best_B_w", best_B_w)
# print("best_C_w", best_C_w)
# print("best_A_b", best_A_b)
# print("best_B_b", best_B_b)
# print("best_C_b", best_C_b)
# print("best_K", best_K)

# print("last_A_w", last_A_w)
# print("last_B_w", last_B_w)
# print("last_C_w", last_C_w)
# print("last_A_b", last_A_b)
# print("last_B_b", last_B_b)
# print("last_C_b", last_C_b)
# print("last_K", last_K)
# TODO----------------------------------------------------------------検証

total_cost_Last = 0
total_event_Last = 0
my_function_ET.seedinit(1)
x0 = my_function_ET.Mike_x0(min_val, max_val, evolution, N)
print("Last_parameter")
for i in range(evolution):
    cost, event = my_function_ET.Evolute_NN(cstr,
                                                T, x0[i], i,K_last, hidden_W_last,hidden_b_last,out_W_last,out_b_last, N, M, Q, R,num_hidden_last,node_hidden_last, path)
    total_cost_Last += cost
    total_event_Last += event
Average_cost_Last = total_cost_Last/evolution
Average_event_Last = total_event_Last/evolution
print("Average_cost", Average_cost_Last)
print("Average_event", Average_event_Last)
# # TODO----------------------------------------------------------------検証2
print("Best_parameter")
total_cost_Best = 0
total_event_Best = 0
for i in range(evolution):
    cost, event = my_function_ET.Evolute_NN(cstr,
                                                T, x0[i], i,K_best, hidden_W_best,hidden_b_best,out_W_best,out_b_best, N, M, Q, R,num_hidden_best,node_hidden_best, path)
    total_cost_Best += cost
    total_event_Best += event
Average_cost_Best = total_cost_Best/evolution
Average_event_Best = total_event_Best/evolution
print("Average_cost", Average_cost_Best)
print("Average_event", Average_event_Best)
# # # ?----------------------------------------------------------------データの保存
parameters_content = cond_text + f"""
# Generated Parameters (Variable-layer NN)

# ===== Best Parameters =====
Best_K = {K_best.tolist()}
Best_num_hidden = {num_hidden_best}
Best_node_hidden = {node_hidden_best}

Best_hidden_weights = { [w.tolist() for w in hidden_W_best] }
Best_hidden_biases  = { [b.tolist() for b in hidden_b_best] }

Best_out_weight = {out_W_best.tolist()}
Best_out_bias   = {out_b_best.tolist()}

# ===== Last Parameters =====
K_last = {K_last.tolist()}
num_hidden_last = {num_hidden_last}
node_hidden_last = {node_hidden_last}

hidden_weights_last = {[w.tolist() for w in hidden_W_last]}
hidden_biases_last  = {[b.tolist() for b in hidden_b_last]}

out_weight_last = {out_W_last.tolist()}
out_bias_last   = {out_b_last.tolist()}
"""
parameters_file_path = path+"result.py"

with open(parameters_file_path, "w") as file:
    file.write(parameters_content)
# 検証結果も条件と一緒にテキストで残す
with open(path + "condition.txt", "w", encoding="utf-8") as file:
    file.write(cond_text)
    file.write(f"# Last: Average_cost = {float(Average_cost_Last)}, Average_event = {float(Average_event_Last)}\n")
    file.write(f"# Best: Average_cost = {float(Average_cost_Best)}, Average_event = {float(Average_event_Best)}\n")
# ===== 実験フォルダに簡潔な README.md を自動作成 =====
# この README の数値は、この実行で実際に用いた条件と計算結果から取得する。
# Evolute_NN の Average_cost は x^T Q x + u^T R u のステージ和のみ。
# 通信費 lambda*送信回数 と終端コストは含まれないので、総コストとは表記しない。
optimizer_name = "AdamW" if AD == 1 else "Adam"
k_last_used = float(cstr.k)  # 最後のエポックで実際に使用した係数
if k_mode == "fixed":
    sigmoid_setting = f"固定 (k={k_start:g})"
elif k_mode == "multiply":
    sigmoid_setting = f"倍率増加 (初期k={k_start:g}、毎エポック×{k_rate:g})"
else:
    sigmoid_setting = f"線形変更 (初期k={k_start:g}、最終k={k_end:g})"
readme_text = f"""# NNイベントトリガ制御：実験記録

- **実行日時**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **対象系**：離散時間・状態{N}次元／入力{M}次元、A={[[round(float(v), 6) for v in row] for row in plant.A.tolist()]}、B={[[round(float(v), 6) for v in row] for row in plant.B.tolist()]}
- **学習期間**：{T-1}ステップ（`plant.T={T}`、終端状態 x_{T-1}）
- **評価関数**：Σ(xᵀQx + uᵀRu + λd) + x_TᵀQf x_T（T={T-1}）、Q={Q.diag().tolist()}、Qf={Qf.diag().tolist()}、R={R.diag().tolist()}、λ={lam}
- **外乱**：{'あり' if plant.W_ON else 'なし'}（{'各成分 一様分布 ['+str(plant.W_MIN)+', '+str(plant.W_MAX)+']' if plant.W_ON else '適用なし'}）
- **初期状態**：各成分 [{min_val}, {max_val}]、学習データ{datasize}件／バッチ{Batch}件／Best選択用{evolution}件
- **学習**：{optimizer_name}、学習率{lr}、{epoch}エポック、乱数シード1
- **NN**：隠れ層{num_hidden_best}層×{node_hidden_best}ノード、NNトリガとフィードバックゲインKを同時学習
- **Sigmoid**：{sigmoid_setting}、実際の最終係数 `k={k_last_used:.6g}`
- **通信判定**：学習時 {'Hard（値は0/1、勾配はSigmoid：ストレートスルー推定）' if hard_forward else 'Soft（Sigmoid）'}、評価時 Hard（NN出力 > 0）

## この実験の結果（Hard通信、{evolution}試行）

- **Best K**：{[[round(float(v), 6) for v in row] for row in K_best.tolist()]}
- **Best**：平均QRステージコスト {float(Average_cost_Best):.6g}、平均通信回数 {float(Average_event_Best):.4f}
- **Last**：平均QRステージコスト {float(Average_cost_Last):.6g}、平均通信回数 {float(Average_event_Last):.4f}
- **学習時間**：{float(training_time):.1f}秒

> **注意**：上記の「平均QRステージコスト」は現行 `Evolute_NN()` の値であり、通信費と終端コストは含みません。理論UUBとの総コスト比較には使わないでください。

詳細：`result.py`（重み・K）、`condition.txt`（条件・平均値）、`sigmoid_k_history.csv`（傾き推移）、各種 `.svg`（学習曲線・軌道）。
"""
readme_path = os.path.join(path, "README.md")
# 同じフォルダにすでに存在するREADMEは変更しない。
with open(readme_path, "x", encoding="utf-8") as readme_file:
    readme_file.write(readme_text)
print("README保存先:", readme_path)

my_function_ET.Save(parameter.Folder, epoch, lr, min_val,
                    max_val, lam, Batch, datasize, AD, evolution, Average_cost_Last, Average_event_Last, Average_cost_Best, Average_event_Best, path)
winsound.Beep(1000, 1000)