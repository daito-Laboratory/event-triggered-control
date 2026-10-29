import os

Folder = os.path.dirname(__file__)

#!ハイパーパラメータ設定

# エポック数
epoch = 100
# 学習率
lr =  0.006
lam = 200
#lam= 50

# 初期値の最小値
max_val =20

# 初期値の最大値
min_val = -20
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

# 最終評価
test_size = 1000
test_T = 100

# 計算手法、AD=0でADAM,1でADAMW
AD = 1

# ===== 学習時のシグモイド傾き k（NN構造と学習率 lr には影響しない） =====
# "fixed"    : k_start で100エポックずっと固定
# "multiply" : 各エポック k = k_start * k_rate**(epoch_index)（従来の1.03倍）
# "linear"   : k_start から k_end まで100エポックで直線的に増加
k_mode = "fixed"
k_start = 1.0
k_rate = 1.03         # multiply を選択した場合のみ使用
k_end = 2.0          # linear を選択した場合のみ使用
# True: 学習中も送信判定を 0/1 で計算し、勾配だけ sigmoid から流す（ストレートスルー推定）。False で従来の Soft 学習
hard_forward = True
# 従来の1.03倍を再現: k_mode="multiply", k_start=1.0, k_rate=1.03
# 以前の標準Sigmoid: k_mode="fixed", k_start=1.0
Switch = "NN"
hidden_node=8
hidden_layer=5

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