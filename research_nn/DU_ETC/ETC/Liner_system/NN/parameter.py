import os

Folder = os.path.dirname(__file__)

#!ハイパーパラメータ設定

# エポック数
epoch = 100
# 学習率
lr = 0.003
lam = int(os.environ.get("ETC_LAMBDA", "50"))
#lam= 200

# 初期値の最大値
max_val = 5

# 初期値の最小値
min_val = -5
# 訓練データのサイズ
datasize = 64
# ミニバッチ学習のバッチサイズ
Batch = 8

# 検証データの数
evol = 100

# 外乱（True で x(k+1) = A x + B u + w、w の各成分 ~ U[w_min, w_max]。False で外乱なし）
disturbance_enabled = True
w_min = -0.2
w_max = 0.2
# 保存先フォルダ名・result.py に書く外乱の表記
dist_tag = ("dist_" + str(w_min) + "_" + str(w_max)) if disturbance_enabled else "dist_none"

# NN の入力："x_e" で [x; e]（e = x̂ − x）、"x_xhat" で従来の [x; x̂]
nn_input = "x_e"

# 最終評価
test_size = 1000
test_T = 100

# ===== 乱数の seed =====
# 学習側: NN の初期重み・学習用初期状態・ミニバッチの並び順・学習中の外乱を決める。
#   seed を変えて複数回学習するときは、ここを 1, 2, 3, … と変えて DETC.py を実行する
train_seeds = [1]
# 評価側: 全実験で固定する（変えると実験どうしを比較できなくなる）
val_seed = 1001    # best epoch 選択用の初期状態と外乱
test_seed = 2001   # 最終評価用の初期状態と外乱

# 計算手法、AD=0でADAM,1でADAMW
AD = 1

# ===== 学習時のシグモイド傾き k（NN構造と学習率 lr には影響しない） =====
# "fixed"    : k_start で100エポックずっと固定
# "multiply" : 各エポック k = k_start * k_rate**(epoch_index)（従来の1.03倍）
# "linear"   : k_start から k_end まで100エポックで直線的に増加
k_mode = "multiply"  # fixed / multiply / linear
k_start = 1.0
k_rate = 1.01        # multiply を選択した場合のみ使用
k_end = 2.0          # linear を選択した場合のみ使用
# True: 学習中も送信判定を 0/1 で計算し、勾配だけ sigmoid から流す（ストレートスルー推定）。False で従来の Soft 学習
hard_forward = False
# 従来の1.03倍を再現: k_mode="multiply", k_start=1.0, k_rate=1.03
# 以前の標準Sigmoid: k_mode="fixed", k_start=1.0
Switch = "NN"
hidden_node=8
hidden_layer=5

# フィードバックゲイン K の初期値：LQR ゲイン（Q=10I、R=1）。安定側から学習を始める
# 小林法（定理1）の K_koba を使う場合は下の行に切り替える。None にすると 0 から学習する
K_init = [[0.4378, 1.9107]]  # LQR のゲイン
# K_init = [[0.2463, 1.2051]]  # K_koba（小林論文 2.3節の計算例、σ=0.16）

# 結果が収納されるフォルダの指定
#Path = Folder + "\\Result\\"+"e" + str(epoch) + "\\lam" + str(lam) + "\\w"+str(min_val)+"_"+str(max_val)+"\\DataSize"+str(datasize)+"_"+"Batch"+str(Batch) + "\\lr" + str(lr)+"\\layer_"+str(hidden_layer)+"node_"+str(hidden_node)

Path = (
    Folder
    + "\\Result\\"
    + "e" + str(epoch)
    + "\\lam" + str(lam)
    + "\\x0_" + str(min_val) + "_" + str(max_val)
    + "\\" + dist_tag
    + "\\DataSize" + str(datasize) + "_Batch" + str(Batch)
    + "\\lr" + str(lr)
    + "\\layer_" + str(hidden_layer) + "node_" + str(hidden_node)

)

if AD == 0:
    Folderpath = Path + "\\Adam\\"
elif AD == 1:
    Folderpath = Path + "\\AdamW\\"

# 軌道の図（Best/Fig、Last/Fig）を保存する本数。test_size 本すべてを保存すると時間と容量がかかるため
fig_count = 100

Folderpath = os.path.join(Folderpath, "stagecost_fixed") + os.sep
os.makedirs(Folderpath, exist_ok=True)
# 検証の大きさ

num_trials = 1000

# 検証回数

Test_count = 1

# 検証の初期値の範囲(回数分)
Test_min_1 = -20
Test_max_1 = 20

Test_min_2 = -100
Test_max_2 = 100