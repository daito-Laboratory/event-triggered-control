import torch
import torch.nn as nn  # ニューラルネットワーク構築
import torch.optim as optim  # 最適化
from torch.utils.data import Dataset  # 訓練データの生成
from torch.utils.data import DataLoader  # 訓練データのロード
import os  # ファイル、フォルダ操作
import matplotlib.pyplot as plt  # グラフ、表の作成
from openpyxl import Workbook, load_workbook  # Excelにデータを保存
import time
import contextlib
import hashlib
def seedinit(seed):  # シード値を固定する関数
    torch.manual_seed(int(seed))


def derive_seed(*parts):
    """
    用途名と seed から、用途ごとに独立した seed を作る（同じ引数なら常に同じ値）。
      derive_seed("nn_init", train_seed)       NN の初期重み
      derive_seed("train_data", train_seed)    学習用初期状態
      derive_seed("shuffle", train_seed)       ミニバッチの並び順
      derive_seed("train_noise", train_seed)   学習中の外乱
      derive_seed("val_data", val_seed)        best epoch 選択用の初期状態
      derive_seed("val_noise", val_seed, i)    best epoch 選択用の外乱（軌道 i）
      derive_seed("test_data", test_seed)      最終評価用の初期状態
      derive_seed("test_noise", test_seed, i)  最終評価用の外乱（軌道 i）
    """
    key = "/".join(str(p) for p in parts)
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:15], 16)
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
# dtype=torch.float32
dtype=torch.double


# ?----------------------------------------------------------------クラス
class DisturbanceDataset(Dataset):  # ミニバッチ学習用の学習データ作成プログラム
    def __init__(self, num_samples, N, min_init, max_init, generator=None):
        # 初期値を各成分 [min_init, max_init] の一様分布で生成
        # generator を渡すと、その乱数生成器だけを使う（用途ごとに seed を分けるため）
        if generator is None:
            self.initial_conditions = [torch.rand(
                N, 1) * (max_init - min_init) + min_init for _ in range(num_samples)]
        else:
            self.initial_conditions = [torch.rand(
                N, 1, generator=generator, dtype=dtype) * (max_init - min_init) + min_init
                for _ in range(num_samples)]

    def __len__(self):
        return len(self.initial_conditions)

    def __getitem__(self, idx):
        return self.initial_conditions[idx]

# class NeuralNet(nn.Module):  #ニューラルネットワークの定義
#     def __init__(self,N,mid_number1,mid_number2):
#         super(NeuralNet, self).__init__()
        
#         self.fc1 = nn.Linear(N*2,mid_number1 ,dtype=dtype)
#         self.fc2 = nn.Linear(mid_number1,mid_number2 ,dtype=dtype)
#         self.relu = nn.ReLU()
#         self.fc3 = nn.Linear(mid_number2, 1,dtype=dtype)
# \
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc3(x)
#         # x = self.sig(x)
#         return x
class NeuralNet(nn.Module):
    def __init__(self, N, num_hidden, node_hidden):
        super(NeuralNet, self).__init__()

        self.relu = nn.ReLU()

        layers = []

        in_dim = N * 2            # 入力次元
        hidden_dim = node_hidden  # 隠れ層のノード数

        # 1層目（入力 → 隠れ層）
        if num_hidden >= 1:
            layers.append(nn.Linear(in_dim, hidden_dim, dtype=dtype))
            in_dim = hidden_dim

        # 2層目以降（隠れ層 → 隠れ層）
        for _ in range(num_hidden - 1):
            layers.append(nn.Linear(in_dim, hidden_dim, dtype=dtype))
            in_dim = hidden_dim

        self.hidden_layers = nn.ModuleList(layers)

        # 出力層（最後の隠れ層 → 出力1次元）
        self.fc_out = nn.Linear(in_dim, 1, dtype=dtype)

    def forward(self, x):
        for fc in self.hidden_layers:
            x = fc(x)
            x = self.relu(x)
        
        x = self.fc_out(x)
        return x
# class NeuralNet_evol(nn.Module):  #ニューラルネットワークの定義
#     def __init__(self,N,mid_number1,mid_number2,a_weight,a_bias,b_weight,b_bias,c_weight,c_bias):
#         super().__init__()
#         a_w = a_weight.clone().detach().to(dtype=dtype)
#         a_b = a_bias.clone().detach().to(dtype=dtype)
#         b_w = b_weight.clone().detach().to(dtype=dtype)
#         b_b = b_bias.clone().detach().to(dtype=dtype)
#         c_w = c_weight.clone().detach().to(dtype=dtype)
#         c_b = c_bias.clone().detach().to(dtype=dtype)

#         self.fc1 = nn.Linear(N * 2, mid_number1, dtype=dtype)
#         self.fc1.weight = nn.Parameter(a_w)
#         self.fc1.bias   = nn.Parameter(a_b)

#         self.fc2 = nn.Linear(mid_number1, mid_number2, dtype=dtype)
#         self.fc2.weight = nn.Parameter(b_w)
#         self.fc2.bias   = nn.Parameter(b_b)

#         self.fc3 = nn.Linear(mid_number2, 1, dtype=dtype)
#         self.fc3.weight = nn.Parameter(c_w)
#         self.fc3.bias   = nn.Parameter(c_b)
#         self.relu = nn.ReLU()
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc3(x)
#         # x = self.sig(x)
#         return x
class NeuralNet_evol(nn.Module):
    """
    学習後のパラメータを固定して使う NN
    - 構造: 入力 2N → 隠れ層 num_hidden 層（各 node_hidden ノード）→ 出力 1
    - 各層の重み・バイアスは引数で渡す
    """
    def __init__(
        self,
        N: int,
        num_hidden: int,
        node_hidden: int,
        hidden_weights,  # list[Tensor] 長さ num_hidden, 各 shape = (node_hidden, in_dim)
        hidden_biases,   # list[Tensor] 長さ num_hidden, 各 shape = (node_hidden,)
        out_weight,      # Tensor shape = (1, last_dim)
        out_bias         # Tensor shape = (1,) or ()
    ):
        super().__init__()

        assert len(hidden_weights) == len(hidden_biases), \
            "hidden_weights と hidden_biases の長さが一致していません"
        assert len(hidden_weights) == num_hidden, \
            "num_hidden と hidden_weights の数が一致していません"

        self.N = N
        self.num_hidden = num_hidden
        self.node_hidden = node_hidden

        self.relu = nn.ReLU()
        self.hidden_layers = nn.ModuleList()

        # ---- 隠れ層 ----
        in_dim = N * 2
        for W, b in zip(hidden_weights, hidden_biases):
            W = W.clone().detach().to(dtype=dtype)
            b = b.clone().detach().to(dtype=dtype)

            fc = nn.Linear(in_dim, node_hidden, dtype=dtype)
            fc.weight = nn.Parameter(W)
            fc.bias   = nn.Parameter(b)

            self.hidden_layers.append(fc)
            in_dim = node_hidden  # 次層の入力次元

        # ---- 出力層 ----
        out_weight = out_weight.clone().detach().to(dtype=dtype)
        out_bias   = out_bias.clone().detach().to(dtype=dtype)

        self.fc_out = nn.Linear(in_dim, 1, dtype=dtype)
        self.fc_out.weight = nn.Parameter(out_weight)
        self.fc_out.bias   = nn.Parameter(out_bias)

    def forward(self, x):
        for fc in self.hidden_layers:
            x = fc(x)
            x = self.relu(x)
        x = self.fc_out(x)
        return x

# NN トリガの入力の種類
#   "x_xhat": 入力 = [x; x̂]（従来）
#   "x_e"   : 入力 = [x; e]、e = x̂ − x（誤差を直接入力する）
NN_INPUTS = ("x_xhat", "x_e")


def nn_features(x, x_hat, N, nn_input="x_xhat"):
    """NN トリガの入力ベクトル（2N×1）を作る。学習・選択用評価・最終評価で共通に使う"""
    if nn_input not in NN_INPUTS:
        raise ValueError(f"nn_input は {NN_INPUTS} のどれか: {nn_input!r}")
    second = x_hat if nn_input == "x_xhat" else x_hat - x
    Rx = torch.zeros(2 * N, 1, dtype=dtype, device=device)
    for i in range(N):
        Rx[i] = x[i]
    for i in range(N):
        Rx[i + N] = second[i]
    return Rx


class CSTR(nn.Module):  # 深層展開のメインクラス
    def __init__(self, T, h, N, M, Q, Qf, R, plant,  system_type, epoch, lr, path, min_init, max_init, lam, Batch, datasize, AD, evolution, Switch,
                 train_seed=1, val_seed=1001, K_init=None, nn_input="x_xhat"):  # パラメータの定義
        super(CSTR, self).__init__()
        # NN トリガの入力の種類（DETC.py が parameter.py の nn_input を渡す）
        if nn_input not in NN_INPUTS:
            raise ValueError(f"nn_input は {NN_INPUTS} のどれか: {nn_input!r}")
        self.nn_input = nn_input
        self.device = device
        self.T = T
        self.N = N
        self.M = M
        self.plant = plant
# ?----------------------------------------------------------------最適化パラメータ
        # K の初期値（DETC.py が parameter.py の K_init を渡す）。指定がなければ 0 から学習する
        if K_init is None:
            K0 = torch.zeros(1, self.N, device=device, dtype=dtype)
        else:
            K0 = torch.tensor(K_init, device=device, dtype=dtype).reshape(1, self.N)
        self.K = nn.Parameter(K0)

# TODO----------------------------------------------------------------Normal_DETC
        self.L = nn.Parameter(torch.zeros(
            2*N, 2*N, dtype=dtype, requires_grad=True, device=device))

# TODO----------------------------------------------------------------4para_DETC
        self.a = nn.Parameter(torch.zeros(
            2*N, 2*N, dtype=dtype, requires_grad=True, device=device))
        self.b = nn.Parameter(torch.zeros(
            1, 2*N, dtype=dtype, requires_grad=True, device=device))
        self.c = nn.Parameter(torch.zeros(
            1, 1, dtype=dtype, requires_grad=True, device=device))

# TODO----------------------------------------------------------------L1_DETC
        self.L_1 = nn.Parameter(torch.zeros(
            self.N, self.N, dtype=dtype, requires_grad=True, device=device))
        self.L_big = torch.tensor([[1.0]], dtype=dtype, device=device)#上書きする
        self.L_mini = nn.Parameter(torch.zeros(
            self.N, self.N, dtype=dtype, requires_grad=True, device=device))
# TODO----------------------------------------------------------------L1_DETC
        self.sigma = nn.Parameter(torch.zeros(
            1, 1, dtype=dtype, requires_grad=True, device=device))
        self.L_sigma = torch.tensor([[1.0]],dtype=dtype, device=device)#上書きする

# ToDO----------------------------------------------------------------L_K
        self.L_K =torch.tensor([[-0.0256,0,0,0],[0,-0.0256,0,0],[0,0,1,0],[0,0,0,1]],dtype=dtype, device=device)#上書きする
        
# TODO----------------------------------------------------------------noK
        self.K_2 =torch.tensor([[1.0]], dtype=dtype, device=device)#上書きする


        #Now_state_dict用
        self.Now_K=torch.tensor([[1.0,-1.0]], dtype=dtype, device=device)
        self.Now_hidden_W = []
        self.Now_hidden_B = []
        self.Now_out_W=torch.tensor([[1.0]], dtype=dtype, device=device)
        self.Now_out_B=torch.tensor([[1.0]],dtype=dtype, device=device)

        #Best_state_dict用
        # self.Best_K=torch.tensor([[1.0]], dtype, device=device)
        # self.Best_A_W=torch.tensor([[1.0]], dtype, device=device)
        # self.Best_A_B=torch.tensor([[1.0]], dtype, device=device)
        # self.Best_B_W=torch.tensor([[1.0]], dtype, device=device)
        # self.Best_B_B=torch.tensor([[1.0]], dtype, device=device)
        # self.Best_C_W=torch.tensor([[1.0]], dtype, device=device)
        # self.Best_C_B=torch.tensor([[1.0]], dtype, device=device)

# !----------------------------------------------------------------Switch


        self.switch_configs = {
            "L": {
                "Torigger": self.ET_L,
                "eval_func": lambda d: self.J_W(d, self.L, self.K),
                "best_params": {"best_K": self.K, "best_L": self.L},
                "log_params":{"K":self.K,"L":self.L}

            },
            "L1": {
                "Torigger": self.ET_L1,
                "eval_func": lambda d: self.J_W(d, self.L_big, self.K),
                "best_params": {"best_K": self.K, "best_L": self.compute_L_big},
                "log_params":{"K":self.K,"L":self.compute_L_big}
            },
            "4para": {
                "Torigger": self.ET_4para1,
                "eval_func": lambda d: self.J_W_4para(d, self.a, self.b, self.c, self.K),
                "best_params": {"best_K": self.K, "best_M2": self.a, "best_M1": self.b, "best_M0": self.c},
                "log_params":{"K":self.K, "M2": self.a, "M1": self.b, "M0": self.c}
            },
            "sigma": {
                "Torigger": self.ET_sigma,
                "eval_func": lambda d: self.J_W(d, self.L_sigma, self.K),
                "best_params": {"best_K": self.K, "best_L": self.compute_L_sigma},
                "log_params":{"K":self.K,"L":self.compute_L_sigma}
            },
            "K": {
                "Torigger": self.ET_K,
                "eval_func": lambda d: self.J_W(d, self.L_K, self.K),
                "log_params":{"K":self.K}
            },
            "4para2": {
                "Torigger": self.ET_4para2,
                "eval_func": lambda d: self.J_W_4para2(d, self.a, self.b, self.c, self.K),
                "best_params": {"best_K": self.K, "best_a": self.a, "best_b": self.b, "best_c": self.c},
                "log_params":{"K":self.K, "M2": self.a, "M1": self.b, "M0": self.c}
            },
            "noK": {
                "Torigger": self.ET_sigma,
                "eval_func": lambda d: self.J_W(d, self.L_sigma, self.K_2),
                "best_params": {"best_K": self.K_2, "best_L": self.compute_L_sigma},
                "log_params":{"K":self.K,"L":self.compute_L_sigma}
            },
            "NN": {
                "Torigger": self.ET_NN,
                "eval_func": lambda d: self.J_W_NN(
                                d,
                                self.Now_K,
                                self.Now_hidden_W,   # [W1, W2, ..., WL]
                                self.Now_hidden_B,   # [b1, b2, ..., bL]
                                self.Now_out_W,      # 出力層 W
                                self.Now_out_B       # 出力層 b
                            ),
                "best_params": {
                        "best_K":          lambda: self.Now_K.clone().detach(),
                        "num_hidden":      lambda: self.hidden_layer,
                        "node_hidden":     lambda: self.hidden_node,
                        "hidden_weights":  lambda: [w.clone().detach() for w in self.Now_hidden_W],
                        "hidden_biases":   lambda: [b.clone().detach() for b in self.Now_hidden_B],
                        "out_weight":      lambda: self.Now_out_W.clone().detach(),
                        "out_bias":        lambda: self.Now_out_B.clone().detach(),
                    },
                "log_params": {
                    "K":           lambda: self.Now_K.clone().detach().cpu(),

                    # 隠れ層（リスト）
                    "hidden_weights": lambda: [
                        w.clone().detach().cpu() for w in self.Now_hidden_W
                    ],
                    "hidden_biases":  lambda: [
                        b.clone().detach().cpu() for b in self.Now_hidden_B
                    ],

                    # 出力層
                    "out_weight": lambda: self.Now_out_W.clone().detach().cpu(),
                    "out_bias":   lambda: self.Now_out_B.clone().detach().cpu(),
                }

            },

        }
        self.switch = Switch
        self.config = self.switch_configs[Switch]
        
# ?----------------------------------------------------------------訓練データ
        self.min_init = min_init
        self.max_init = max_init

# ?----------------------------------------------------------------コスト関数
        self.Q = Q
        self.Qf = Qf
        self.R = R
        self.lam = lam
# ?----------------------------------------------------------------その他
        self.batchSize = Batch
        self.epoch = epoch
        self.lr = lr
        self.delta = torch.zeros(1, self.T-1, dtype=dtype)
        self.h = h
        self.sig = nn.Sigmoid()
        # NN本体の構造は変えず、Soft送信判定 sigmoid(k * NN出力) の傾きだけ変える。
        # mode: fixed（固定）/ multiply（毎epoch倍率）/ linear（線形補間）
        self.k_mode = "fixed"
        self.k_start = 1.0
        self.k_end = 2.0
        self.k_rate = 1.0
        self.k = self.k_start
        # True: 学習中も送信判定をハード（0/1）で計算し、勾配だけ sigmoid から流す（ストレートスルー推定）
        self.hard_forward = False
        self.best = 0
        self.path = path
        self.datasize = datasize
        self.AD = AD
        self.system_type = system_type
        self.evolution = evolution
        

#-------------------------------------------------------
        self.hidden_node=16
        self.hidden_layer=1
        self.model = NeuralNet(self.N,self.hidden_layer,self.hidden_node)
        # 乱数の seed（学習側は複数回変えて試す、検証側は全実験で固定）
        self.train_seed = int(train_seed)
        self.val_seed = int(val_seed)
        self.best_epoch = None
        

        self.log=0
        self.log_snap=[]
        self.log_snap_ALL=[]
# ----------------------------------------------------------------関数

    def f(self, x, u):
    
        x_next =self.plant.x_next(x, u)
        
        return x_next

    def _make_loaders(self):
        # 学習用・選択用の初期状態とミニバッチの並び順を、用途ごとの seed で作る
        g_train = torch.Generator().manual_seed(derive_seed("train_data", self.train_seed))
        g_val = torch.Generator().manual_seed(derive_seed("val_data", self.val_seed))
        g_shuffle = torch.Generator().manual_seed(derive_seed("shuffle", self.train_seed))
        dataset = DisturbanceDataset(num_samples=self.datasize, N=self.N,
                                     min_init=self.min_init, max_init=self.max_init, generator=g_train)
        dataloader = DataLoader(dataset, batch_size=self.batchSize, shuffle=True, generator=g_shuffle)
        evoldataset = DisturbanceDataset(num_samples=self.evolution, N=self.N,
                                         min_init=self.min_init, max_init=self.max_init, generator=g_val)
        evoldataloader = DataLoader(evoldataset, batch_size=1, shuffle=False)
        return dataloader, evoldataloader

    def _eval_one(self, idx, data):
        # 評価データ idx ごとに外乱系列を固定する（全エポックで同じ外乱で比較するため）
        fixed = getattr(self.plant, "fixed_noise", None)
        ctx = fixed(idx) if fixed is not None else contextlib.nullcontext()
        with ctx:
            return self.config["eval_func"](data)

# ? ---------------------------------------------------------------イベントトリガ
    def ET_L(self, x, x_hat):  # KとLをパラメータに持つイベントトリガ
        Rx = torch.zeros(2*self.N, 1, dtype=dtype).to(self.device)
        e = x_hat-x
        for i in range(0, self.N):
            Rx[i] = x[i]
        for i in range(0, self.N):
            Rx[i+self.N] = e[i]
        
        return Rx.T@self.L@Rx

    def ET_L1(self, x, x_hat):  # KとL1をパラメータにもつイベントトリガ
        I = torch.eye(self.N, dtype=dtype,
                      requires_grad=True, device=self.device)
        L = torch.cat([torch.cat([self.L_1, torch.zeros_like(self.L_1)], dim=1), torch.cat(
            [torch.zeros_like(self.L_1), I], dim=1)], dim=0)
        Rx = torch.zeros(2*self.N, 1, dtype=dtype).to(self.device)
        e = x_hat-x
        for i in range(0, self.N):
            Rx[i] = x[i]
        for i in range(0, self.N):
            Rx[i+self.N] = e[i]
        self.L_big = L
        return Rx.T@L@Rx

    def ET_sigma(self, x, x_hat):  # Kとsigmaをパラメータにもつイベントトリガ
        I = torch.eye(self.N, dtype=dtype,
                      requires_grad=True, device=self.device)
        L = torch.cat([torch.cat([self.sigma*I, torch.zeros_like(self.L_1)], dim=1), torch.cat(
            [torch.zeros_like(self.L_1), I], dim=1)], dim=0)
        Rx = torch.zeros(2*self.N, 1, dtype=dtype).to(self.device)
        e = x_hat-x
        for i in range(0, self.N):
            Rx[i] = x[i]
        for i in range(0, self.N):
            Rx[i+self.N] = e[i]
        self.L_sigma = L
        return Rx.T@L@Rx

    def ET_K(self, x, x_hat):  # Kをパラメータにもつイベントトリガ
        Rx = torch.zeros(2*self.N, 1, dtype=dtype).to(self.device)
        e = x_hat-x
        for i in range(0, self.N):
            Rx[i] = x[i]
        for i in range(0, self.N):
            Rx[i+self.N] = e[i]
        return Rx.T@self.L_K@Rx

    def ET_4para1(self, x, x_hat):  # KとM2,M1,M0,をパラメータにもつイベントトリガ
        Rx = torch.zeros(2*self.N, 1, dtype=dtype).to(self.device)
        e = x_hat-x
        for i in range(0, self.N):
            Rx[i] = x[i]
        for i in range(0, self.N):
            Rx[i+self.N] = e[i]
        return Rx.T@self.a@Rx+self.b@Rx+self.c  # a,b,cとM2,M1,M0は形が同じなので流用

    def ET_4para2(self, x, x_hat):  # Kとa,b,c,をパラメータにもつイベントトリガ
        Rx = torch.zeros(2*self.N, 1, dtype=dtype).to(self.device)
        e = x_hat-x
        for i in range(0, self.N):
            Rx[i] = x[i]
        for i in range(0, self.N):
            Rx[i+self.N] = e[i]
        # print(x)
        return (Rx-self.b.T).T@self.a@(Rx-self.b.T)+self.c

    def ET_NN(self, x, x_hat):
        Rx = nn_features(x, x_hat, self.N, self.nn_input)

        O = self.model(Rx.T)

        self._snapshot_nn()
        return O

    def _snapshot_nn(self):
        # 現在のNNの重みとKを、評価・保存用に写し取る
        self.Now_hidden_W = []
        self.Now_hidden_B = []

        for fc in self.model.hidden_layers:
            self.Now_hidden_W.append(fc.weight.detach().clone().cpu())
            self.Now_hidden_B.append(fc.bias.detach().clone().cpu())

        # 出力層の重み・バイアス
        self.Now_out_W = self.model.fc_out.weight.detach().clone().cpu()
        self.Now_out_B = self.model.fc_out.bias.detach().clone().cpu()
        self.Now_K = self.K.detach().clone()

    def ET_NN_W(
        self,
        x,
        x_hat,
        hidden_W_list,
        hidden_B_list,
        out_W,
        out_B,
    ):
        # 入力ベクトル Rx の作成（学習時と同じ入力）
        Rx = nn_features(x, x_hat, self.N, self.nn_input)

        # 評価用モデルの構築
        model2 = NeuralNet_evol(
            N=self.N,
            num_hidden=self.hidden_layer,      # どこかで保持している層数
            node_hidden=self.hidden_node,    # どこかで保持しているノード数
            hidden_weights=hidden_W_list,
            hidden_biases=hidden_B_list,
            out_weight=out_W,
            out_bias=out_B,
        ).to(self.device)

        O = model2(Rx.T)
        return O

# ? ----------------------------------------------------------------コスト関数
    def BatchJ(self, T, data, trigger_func):
        cost = torch.zeros(self.batchSize,
                           dtype=dtype).to(self.device)
    
        cost = self.J(T, data, trigger_func)

        return cost

    def J(self, T, x0, trigger_func):
        B_size=x0.size(0)
        cost = torch.zeros(B_size, dtype=dtype).to(self.device)
        x, u, delta = self.makex(T, x0,B_size, trigger_func)
        Q  = self.Q.to(device)
        Qf = self.Qf.to(device)
        R  = self.R.to(device)
        lam = self.lam

    # -------- ステージコスト --------
        if T >= 1:
            # x_t: (B, N, T-1) → (B, T-1, N)
            x_t = x[:, :, 0:T]             # (B, N, T-1)
            x_tT = x_t.transpose(1, 2)       # (B, T-1, N)

            # u_t: (B, M, T-1) → (B, T-1, M)
            u_t = u[:, :, 0:T]             # (B, M, T-1)
            u_tT = u_t.transpose(1, 2)       # (B, T-1, M)

            # x^T Q x = diag( x_tT @ Q @ x_tT^T )
            # (B, T-1, N) @ (N, N) -> (B, T-1, N)
            xQ = torch.matmul(x_tT, Q)       # (B, T-1, N)
            # 要素ごとの積を取って和を取ると x^T Q x
            xQx = (xQ * x_tT).sum(dim=2)     # (B, T-1)

            # u^T R u も同様
            uR = torch.matmul(u_tT, R)       # (B, T-1, M)
            uRu = (uR * u_tT).sum(dim=2)     # (B, T-1)

            d_t = delta[:, 0, 0:T]         # (B, T-1)
            stage = xQx + uRu + lam * d_t    # (B, T-1)
            stage_sum = stage.sum(dim=1)     # (B,)
        else:
            # T=1 のときはステージコストなし
            stage_sum = torch.zeros(B_size, dtype=dtype, device=device)

        # -------- 終端コスト --------
        # xT: (B, N, 1)
        xT = x[:, :, T:T+1]                  # (B, N, 1)
        xT_T = xT.transpose(1, 2)            # (B, 1, N)
        # xT^T Qf xT
        xTQ = torch.matmul(xT_T, Qf)         # (B, 1, N)
        xTQfxT = (xTQ * xT_T).sum(dim=2).squeeze(1)  # (B,)

        # -------- 合計 --------
        cost = stage_sum + xTQfxT            # (B,)
        return cost

    def makex(self, T, x0, B_size, trigger_func):
        device = self.device

        # x0: (B, N, 1) 前提
        x0 = x0.to(device=device, dtype=dtype)

        # K: (1, N) → (B, 1, N)
        K = self.K.to(device=device, dtype=dtype)          # (1, N)
        K_batched = K.unsqueeze(0).expand(B_size, -1, -1)  # (B, 1, N)

        # ---- 時系列リスト ----
        x_list = []      # 各要素: (B, N, 1)
        u_list = []      # 各要素: (B, M, 1)
        delta_list = []  # 各要素: (B, 1, 1)

        # 初期状態
        x_t = x0.clone()      # (B, N, 1)
        x_hat = x0.clone()    # (B, N, 1)

        # t = 0 の状態を保存
        x_list.append(x_t)

        # t=0 のイベントは「必ず送信」として δ_0 = 1
        delta_0 = torch.ones(B_size, 1, 1, device=device, dtype=dtype)
        delta_list.append(delta_0)

        # t=0 の入力 u_0
        u_0 = torch.bmm(K_batched, x_hat)  # (B, 1, N) x (B, N, 1) -> (B, 1, 1)
        u_list.append(u_0)

        # 次の状態 x_1 を計算
        x_next_batch = []
        for b in range(B_size):
            x_next_batch.append(self.f(
                x_t[b, :, :],   # (N, 1)
                u_0[b, :, :]    # (1, 1)
            ))
        x_t = torch.stack(x_next_batch, dim=0)  # (B, N, 1)
        x_list.append(x_t)

        # ---- t = 1 ... T-1 ----
        for i in range(1, T):
            # 現在の状態 x_t に対して δ_i, x_hat を更新
            d_batch = []
            x_hat_new_batch = []

            for b in range(B_size):
                o = self.k * trigger_func(
                    x_t[b, :, :],   # (N, 1)
                    x_hat[b, :, :]  # (N, 1)
                )                   # (1, 1) 想定
                d_soft = self.sig(o)
                if self.hard_forward:
                    # 値は 0/1（実際の制御と同じ）、勾配は d_soft と同じ
                    d = (o > 0).to(d_soft.dtype) + d_soft - d_soft.detach()
                else:
                    d = d_soft

                d_batch.append(d)
                x_hat_new_batch.append(
                    d * x_t[b, :, :] + (1.0 - d) * x_hat[b, :, :]
                )

            delta_i = torch.stack(d_batch, dim=0)          # (B, 1, 1)
            x_hat = torch.stack(x_hat_new_batch, dim=0)    # (B, N, 1)

            delta_list.append(delta_i)

            # 入力 u_i
            u_i = torch.bmm(K_batched, x_hat)              # (B, 1, 1)
            u_list.append(u_i)

            # 次の状態 x_{i+1}
            x_next_batch = []
            for b in range(B_size):
                x_next_batch.append(self.f(
                    x_t[b, :, :],   # (N, 1)
                    u_i[b, :, :]    # (1, 1)
                ))
            x_t = torch.stack(x_next_batch, dim=0)         # (B, N, 1)
            x_list.append(x_t)

        # ---- リストをテンソルにまとめる ----
        # x_list: 長さ T+1, 各要素 (B, N, 1) → (B, N, T+1)
        x = torch.cat(x_list, dim=2)

        # u_list: 長さ T, 各要素 (B, M, 1) → (B, M, T)
        u = torch.cat(u_list, dim=2)

        # delta_list: 長さ T, 各要素 (B, 1, 1) → (B, 1, T)
        delta = torch.cat(delta_list, dim=2)

        # もし元の実装どおり (B, N, T), (B, M, T-1), (B, 1, T-1) に合わせたいなら、
        # T を self.T と一致させて、最後の一歩を調整してもOK

        return x, u, delta

#!--------------------------------------------------------------------------------------------------------------------------
    def utrain(self):  # 最重要、ここで訓練
        print("Start Training")
        directory = os.path.dirname(self.path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        seedinit(derive_seed("nn_init", self.train_seed))  # NN の初期重み
        self.model=NeuralNet(self.N,self.hidden_layer,self.hidden_node)
        if self.AD == 0:
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        elif self.AD == 1:
            optimizer = optim.AdamW(self.parameters(), lr=self.lr)

        target = torch.zeros(self.batchSize, dtype=dtype).to(self.device)
        loss_fn = nn.L1Loss()
        min_cost = float('inf')
        best_params = None
        final_params = {}
        loss_history = []
        lr_history = []
        eval_cost_history = []
        sigmoid_k_history = []
        log_params = self.config["log_params"]
        best_epoch = -1  # 初期化
        dataloader, evoldataloader = self._make_loaders()

        trigger_func = self.config.get("Torigger")
        printed_marks = 0
        if self.system_type == 0:
            T = self.T
            indices = []
            k = 1
            while k < T:
                indices.append(k)      # 1, 6, 11, 16, ...
                # k += 10                 # ★ 5刻み
                k*=2

            if indices[-1] != T - 1:
                indices.append(T - 1)  # 最後だけ T-1 に合わせる
        else:
            indices = list(range(1, self.T))  # 従来通り全部

        print(indices)
        train_start_time = time.perf_counter()
        for e in range(self.epoch):
            # エポック番号 e=0 は最初のエポック。倍率方式なら初回は k_start。
            if self.k_mode == "fixed":
                self.k = self.k_start
            elif self.k_mode == "multiply":
                self.k = self.k_start * (self.k_rate ** e)
            elif self.k_mode == "linear":
                progress_k = e / max(self.epoch - 1, 1)
                self.k = self.k_start + (self.k_end - self.k_start) * progress_k
            else:
                raise ValueError(f"Unknown sigmoid k mode: {self.k_mode!r}")
            if not (0 < self.k < float("inf")):
                raise ValueError(f"Sigmoid coefficient must be finite and > 0: {self.k}")
            sigmoid_k_history.append(float(self.k))
            if e != 0:
                self.log=1

            for data in dataloader:


                for i in indices:
                    # print(i)
                    optimizer.zero_grad()
                    costJ = self.BatchJ(i, data, trigger_func)
                    loss = loss_fn(costJ, target)
                    loss.backward()
                    optimizer.step()
                    if self.switch == "NN":
                        self._snapshot_nn()  # ログには更新後の値を記録する
                    if self.log == 0:
                        snap = {}
                        for name, getter in log_params.items():
                            value = getter() if callable(getter) else getter
                            snap[name] = to_log_value(value)
                        self.log_snap.append(snap)

                    snap_ALL = {}
                    for name, getter in log_params.items():
                        value = getter() if callable(getter) else getter
                        snap_ALL[name] = to_log_value(value)
                    self.log_snap_ALL.append(snap_ALL)
            torch.save(self.log_snap, self.path+"params.pt")
            torch.save(self.log_snap_ALL, self.path+"params_all.pt")
                
            progress = (e + 1) / self.epoch
            target_marks = int(progress * 10)
            if target_marks > printed_marks:
                print("*", end="", flush=True)
                printed_marks = target_marks
            loss_history.append(loss.item())
            lr_history.append(optimizer.param_groups[0]['lr'])

# 評価コスト計算と記録部分
            if self.switch == "NN":
                self._snapshot_nn()  # エポック終了時点（最後の更新後）の値で評価する
            eval_cost = sum(self._eval_one(i, data) for i, data in enumerate(evoldataloader)) / self.evolution
            eval_cost_history.append(eval_cost.detach().item())  # ← .detach().item() を追加


            if eval_cost < min_cost:
                min_cost = eval_cost
                best_epoch = e 
                best_params = {
                    key: (param.clone() if isinstance(param, torch.Tensor) else param())
                    for key, param in self.config["best_params"].items()
                }
        if best_params is None:
            raise RuntimeError("全エポックで選択用の評価値が NaN または Inf でした（学習が発散しています）")
        self.best_epoch = best_epoch          # 0 始まり（e=0 が 1 エポック目）
        self.best_eval_cost = float(min_cost)  # Best 選択に使った評価値（選択用の初期状態・外乱）
        train_end_time = time.perf_counter()
        elapsed_sec = train_end_time - train_start_time
        print(f"\nTraining time: {elapsed_sec:.2f} [sec]")
        if self.switch == "NN":
            self._snapshot_nn()  # 学習終了時点（最終エポックの最後の更新後）の値
        final_params = {
            key: (param.clone() if isinstance(param, torch.Tensor) else param())
            for key, param in self.config["best_params"].items()
        }
        loss = torch.tensor(loss_history, dtype=dtype)
        eval_cost = torch.tensor(eval_cost_history, dtype=dtype)
        lr = torch.tensor(lr_history, dtype=dtype)
        epochs = torch.arange(len(loss), dtype=dtype)
        # ===============================
        # 1. 学習損失（全エポック・線形軸）
        # ===============================
        plt.figure()
        plt.plot(_to_np(loss), label='Training Loss')
        plt.axvline(best_epoch, linestyle='--', color='gray',
                    label=f'Best Epoch = {best_epoch}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('loss')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.path + 'loss_train_full_linear.svg')

        # ===============================
        # 2. 学習損失（後半20%＋外れ値カット）
        # ===============================
        zoom_start = int(len(loss) * 0.8)
        tail_epochs = epochs[zoom_start:]
        tail_loss = loss[zoom_start:]

        plt.figure()
        plt.plot(_to_np(epochs), _to_np(loss),
                alpha=0.3, label='Training Loss (all)')
        plt.plot(_to_np(tail_epochs), _to_np(tail_loss),
                color='red', label='Training Loss (tail)')
        plt.axvline(best_epoch, linestyle='--', color='gray',
                    label=f'Best Epoch = {best_epoch}')

        ylim = robust_ylim_torch(tail_loss, lower=0.0,
                                upper_percentile=95.0, margin=1.2)
        if ylim is not None:
            plt.ylim(*ylim)

        plt.xlim(zoom_start, len(loss) - 1)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('loss-cat')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.path + 'loss_train_tail_linear.svg')

        # ===============================
        # 3. 学習損失（全エポック・対数軸）
        # ===============================
        eps = 1e-12
        loss_for_log = loss + eps

        plt.figure()
        plt.plot(_to_np(loss_for_log), label='Training Loss')
        plt.axvline(best_epoch, linestyle='--', color='gray',
                    label=f'Best Epoch = {best_epoch}')
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('loss-log')
        plt.grid(True, which='both')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.path + 'loss_train_full_log.svg')

        # ===============================
        # 4. 評価コスト（全エポック・線形軸）
        # ===============================
        plt.figure()
        plt.plot(_to_np(eval_cost), label='Eval Cost', color='blue')
        plt.axvline(best_epoch, linestyle='--', color='gray',
                    label=f'Best Epoch = {best_epoch}')
        plt.xlabel('Epoch')
        plt.ylabel('Evaluation Cost')
        plt.title('evol')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.path + 'eval_cost_full_linear.svg')

        # ===============================
        # 5. 評価コスト（後半20%＋外れ値カット）
        # ===============================
        zoom_start_eval = int(len(eval_cost) * 0.8)
        eval_epochs = epochs
        tail_eval_epochs = eval_epochs[zoom_start_eval:]
        tail_eval_cost = eval_cost[zoom_start_eval:]

        plt.figure()
        plt.plot(_to_np(eval_epochs), _to_np(eval_cost),
                alpha=0.3, label='Eval Cost (all)')
        plt.plot(_to_np(tail_eval_epochs), _to_np(tail_eval_cost),
                color='blue', label='Eval Cost (tail)')
        plt.axvline(best_epoch, linestyle='--', color='gray',
                    label=f'Best Epoch = {best_epoch}')

        ylim_eval = robust_ylim_torch(tail_eval_cost, lower=0.0,
                                    upper_percentile=95.0, margin=1.2)
        if ylim_eval is not None:
            plt.ylim(*ylim_eval)

        plt.xlim(zoom_start_eval, len(eval_cost) - 1)
        plt.xlabel('Epoch')
        plt.ylabel('Evaluation Cost')
        plt.title('evol-cat')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.path + 'eval_cost_tail_linear.svg')

        # ===============================
        # 6. 評価コスト（全エポック・対数軸）
        # ===============================
        eps_eval = 1e-12
        eval_for_log = eval_cost + eps_eval

        plt.figure()
        plt.plot(_to_np(eval_for_log), label='Eval Cost', color='blue')
        plt.axvline(best_epoch, linestyle='--', color='gray',
                    label=f'Best Epoch = {best_epoch}')
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Evaluation Cost (log scale)')
        plt.title('evol-log')
        plt.grid(True, which='both')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.path + 'eval_cost_full_log.svg')

        # 学習率のプロットも torch → np 変換だけ挟めば OK
        self.plt_lr(_to_np(lr), _to_np(loss))

        # シグモイド係数 k の実際の推移を出力（学習率の推移とは別）。
        import csv
        with open(self.path + "sigmoid_k_history.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "sigmoid_k"])
            writer.writerows((i + 1, k) for i, k in enumerate(sigmoid_k_history))
        plt.figure()
        plt.plot(range(1, len(sigmoid_k_history) + 1), sigmoid_k_history)
        plt.xlabel("Epoch")
        plt.ylabel("Sigmoid slope coefficient k")
        plt.title("Sigmoid k schedule")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.path + "sigmoid_k_history.svg")

        plt.close('all')
        print()
        return (*best_params.values(), *final_params.values(),elapsed_sec) 

    def utrain_opt(self, optimizer_name):  # Optuna用の関数,基本はutrainと同じ
        seedinit(derive_seed("nn_init", self.train_seed))  # NN の初期重み
        self.model = NeuralNet(self.N, self.hidden_layer, self.hidden_node)
        # Optimizerを文字列で指定
        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        target = torch.zeros(
            self.batchSize, dtype=dtype).to(self.device)

        loss_fn = nn.L1Loss()
        min_cost = float('inf')
        best_params = None
        loss_history = []
        lr_history = []
        # 訓練用・評価用データセットの作成（utrain と同じ seed の扱い）
        dataloader, evoldataloader = self._make_loaders()
        trigger_func = self.config.get("Torigger")
        for e in range(self.epoch):  # epoch分繰り返す
            
            for data in dataloader:  # 訓練ループ
                step = int((self.T - 1) / 10)  # =10
                indices = list(range(1, self.T, step))

                if indices[-1] != self.T - 1:
                    indices.append(self.T - 1)

                # for i in indices:
                    
                for i in range(1, self.T):
                    
                    optimizer.zero_grad()
                    costJ = self.BatchJ(i, data, trigger_func)
                    loss = loss_fn(costJ, target)
                    loss.backward()
                    optimizer.step()
            loss_history.append(loss.item())
            current_lr = optimizer.param_groups[0]['lr']
            lr_history.append(current_lr)
            # 評価ループ
            if self.switch == "NN":
                self._snapshot_nn()
            eval_cost = sum(self._eval_one(i, data)
                            for i, data in enumerate(evoldataloader)) / self.evolution

            # 最適なパラメータを保存
            if eval_cost < min_cost:
                min_cost = eval_cost
                best_params = {
                    key: (param.clone() if isinstance(
                        param, torch.Tensor) else param())
                    for key, param in self.config["best_params"].items()
                }
        if best_params is None:
            raise RuntimeError("全エポックで選択用の評価値が NaN または Inf でした（学習が発散しています）")
        return tuple(best_params.values())

#!----------------------------------------------------------------評価

# ----------------------------------------------------------------- 2パラメータのイベントトリガ
    def ET_W(self, x, x_hat, L):
        Rx = torch.zeros(2*self.N, 1, dtype=dtype).to(self.device)
        e = x_hat-x
        for i in range(0, self.N):
            Rx[i] = x[i]
        for i in range(0, self.N):
            Rx[i+self.N] = e[i]
        return Rx.T@L@Rx

    def J_W(self, x0, L, K):
        T = self.T-1
        cost = torch.zeros(1, dtype=dtype).to(self.device)

        x, u, delta = self.makex_W(T, x0, L, K)

        for i in range(0, T):
            cost = cost+self.StageCost(x[:, [i]],
                                       u[:, [i]], delta[:, [i]])
        cost = cost+self.FinalCost(x[:, [T]])
        return cost

    def J_W_NN(self, x0, K, hidden_W_list, hidden_B_list, out_W, out_B):

        T = self.T-1
        cost = torch.zeros(1, dtype=dtype).to(self.device)

        x, u, delta = self.makex_W_NN(T, x0, K, hidden_W_list, hidden_B_list, out_W, out_B)
    
        for i in range(0, T):
            cost = cost+self.StageCost(x[:, [i]],
                                       u[:, [i]], delta[:, [i]])
        cost = cost+self.FinalCost(x[:, [T]])
        return cost
    
    def makex_W(self, T, x0, L, K):
        x = torch.zeros(self.N, self.T, dtype=dtype).to(self.device)
        x_hat = torch.zeros(
            self.N, 1, dtype=dtype).to(self.device)
        delta = torch.zeros(1, self.T-1, dtype=dtype).to(self.device)
        u = torch.zeros(1, self.T-1, dtype=dtype).to(self.device)
        x[:, [0]] = x0.to(dtype=dtype)
        x_hat = x0.to(dtype=dtype)
        for i in range(0, T):
            if i == 0:
                delta[:, [i]] = 1
                x_hat = x[:, [i]]
            else:
                if self.ET_W(x[:, [i]], x_hat, L) > 0:
                    x_hat = x[:, [i]]
                    delta[:, [i]] = 1
            u[:, [i]] = K@x_hat
            x[:, [i+1]] = self.f(x[:, [i]], u[:, [i]])
        return x, u, delta

    def makex_W_NN(self, T, x0, K, hidden_W_list, hidden_B_list, out_W, out_B):
        x = torch.zeros(self.N, self.T, dtype=dtype).to(self.device)
        x_hat = torch.zeros(
            self.N, 1, dtype=dtype).to(self.device)
        delta = torch.zeros(1, self.T-1, dtype=dtype).to(self.device)
        u = torch.zeros(1, self.T-1, dtype=dtype).to(self.device)
        x[:, [0]] = x0.to(dtype=dtype)
        x_hat = x0.to(dtype=dtype)
        for i in range(0, T):
            if i == 0:
                delta[:, [i]] = 1
                x_hat = x[:, [i]]
            else:
                if self.ET_NN_W(x[:, [i]], x_hat, hidden_W_list, hidden_B_list, out_W, out_B) > 0:
                    x_hat = x[:, [i]]
                    delta[:, [i]] = 1
            u[:, [i]] = K@x_hat
            x[:, [i+1]] = self.f(x[:, [i]], u[:, [i]])
        return x, u, delta
# ----------------------------------------------------------------- 4パラメータのイベントトリガ


    def ET_W_4para(self, x, x_hat, M2, M1, M0):  # イベントトリガ条件
        Rx = torch.zeros(2*self.N, 1, dtype=dtype).to(self.device)
        e = x_hat-x
        for i in range(0, self.N):
            Rx[i] = x[i]
        for i in range(0, self.N):
            Rx[i+self.N] = e[i]
        return Rx.T@M2@Rx+M1@Rx+M0

    def ET_W_4para2(self, x, x_hat, a, b, c):  # イベントトリガ条件
        Rx = torch.zeros(2*self.N, 1, dtype=dtype).to(self.device)
        e = x_hat-x
        for i in range(0, self.N):
            Rx[i] = x[i]
        for i in range(0, self.N):
            Rx[i+self.N] = e[i]
        return (Rx-b.T).T@a@(Rx-b.T)+c

    def J_W_4para(self, x0, M2, M1, M0, K):
        T = self.T-1
        cost = torch.zeros(1, dtype=dtype).to(self.device)

        x, u, delta = self.makex_W_4para(T, x0, M2, M1, M0, K)

        for i in range(0, T):
            cost = cost+self.StageCost(x[:, [i]],
                                       u[:, [i]], delta[:, [i]])
        cost = cost+self.FinalCost(x[:, [T]])
        return cost

    def J_W_4para2(self, x0, a, b, c, K):
        T = self.T-1
        cost = torch.zeros(1, dtype=dtype).to(self.device)

        x, u, delta = self.makex_W_4para2(T, x0, a, b, c, K)

        for i in range(0, T):
            cost = cost+self.StageCost(x[:, [i]],
                                       u[:, [i]], delta[:, [i]])
        cost = cost+self.FinalCost(x[:, [T]])
        return cost

    def makex_W_4para(self, T, x0, M2, M1, M0, K):
        x = torch.zeros(self.N, self.T, dtype=dtype).to(self.device)
        x_hat = torch.zeros(
            self.N, 1, dtype=dtype).to(self.device)
        delta = torch.zeros(1, self.T-1, dtype=dtype).to(self.device)
        u = torch.zeros(1, self.T-1, dtype=dtype).to(self.device)
        x[:, [0]] = x0.to(dtype=dtype)
        x_hat = x0.to(dtype=dtype)
        for i in range(0, T):
            if i == 0:
                delta[:, [i]] = 1
                x_hat = x[:, [i]]
            else:
                if self.ET_W_4para(x[:, [i]], x_hat, M2, M1, M0) > 0:
                    x_hat = x[:, [i]]
                    delta[:, [i]] = 1
            u[:, [i]] = K@x_hat
            x[:, [i+1]] = self.f(x[:, [i]], u[:, [i]])
        return x, u, delta

    def makex_W_4para2(self, T, x0, a, b, c, K):
        x = torch.zeros(self.N, self.T, dtype=dtype).to(self.device)
        x_hat = torch.zeros(
            self.N, 1, dtype=dtype).to(self.device)
        delta = torch.zeros(1, self.T-1, dtype=dtype).to(self.device)
        u = torch.zeros(1, self.T-1, dtype=dtype).to(self.device)
        x[:, [0]] = x0.to(dtype=dtype)
        x_hat = x0.to(dtype=dtype)
        for i in range(0, T):
            if i == 0:
                delta[:, [i]] = 1
                x_hat = x[:, [i]]
            else:
                if self.ET_W_4para2(x[:, [i]], x_hat, a, b, c) > 0:
                    x_hat = x[:, [i]]
                    delta[:, [i]] = 1
            u[:, [i]] = K@x_hat
            x[:, [i+1]] = self.f(x[:, [i]], u[:, [i]])
        return x, u, delta
# ?----------------------------------------------------------------コスト計算

    def StageCost(self, x, u, delta):  # !ステージコスト
        # print(delta)
        return x.T @ self.Q @ x + u.T @ self.R @ u+self.lam*delta

    def FinalCost(self, x):  # !終端コスト
        return x.T @ self.Qf @ x

# ?------------------------------------------------------その他の関数

    def compute_L_sigma(self):
        I = torch.eye(self.N, dtype=dtype, device=self.device)
        return torch.cat([
            torch.cat([self.sigma * I, torch.zeros_like(I)], dim=1),
            torch.cat([torch.zeros_like(I), I], dim=1)
        ], dim=0)

    def compute_L_big(self):

        I = torch.eye(self.N, dtype=dtype, device=self.device)
        return torch.cat([torch.cat([self.L_1, torch.zeros_like(self.L_1)], dim=1), torch.cat(
            [torch.zeros_like(self.L_1), I], dim=1)], dim=0)
    def plt_lr(self, lr_history, loss_history):  # コスト関数、学習率の時系列データをプロット
        plt.figure(figsize=(10, 6))
        plt.plot(lr_history, loss_history, marker='o', linestyle='-')
        plt.title('Loss vs. Learning Rate')
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.grid(True)
        path = self.path
        plt.savefig(path+'loss_history.svg')

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(lr_history) + 1),
                 lr_history, marker='o', linestyle='-')
        plt.title('Learning Rate over Epochs')
        plt.xlabel('Epoch')
        plt.yscale('log')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        plt.savefig(path+'lr_history.svg')



#!----------------------------------------------------------------関数


def Evolute(cstr, T, x0, k, K, L, N, M, Q, R, Result_path):
    if not os.path.exists(Result_path+"Fig\\"):
        os.makedirs(Result_path+"Fig\\")
    x = torch.zeros(N, T, dtype=dtype).to(device)
    x_hat = torch.zeros(N, 1, dtype=dtype).to(device)
    u = torch.zeros(M, T-1, dtype=dtype).to(device)
    delta = torch.zeros(1, T-1, dtype=dtype)
    x[:, [0]] = x0
    cost = 0

    # シミュレーション実行
    for i in range(T-1):
        
        if i==0:
                x_hat = x[:, [i]]
                delta[:, [i]] = 1  # イベント発生時に1を記録
        else:
            if ET(x[:, [i]], x_hat, L, N) > 0:
                x_hat = x[:, [i]]
                delta[:, [i]] = 1  # イベント発生時に1を記録
        u[:, [i]] = K @ x_hat
        x[:, [i + 1]] = cstr.f(x[:, [i]], u[:, [i]])
    for i in range(T-1):
        cost += COST(x[:, [i]], u[:, [i]], Q, R)

    # イベント発生回数を計算して表示
    event_count = int(torch.sum(delta).item())
    time = range(T)

    plt.figure()

    # x_0 のプロット
    plt.subplot(3, 1, 1)
    for i in range(0, N):
        if i == 0:
            iro = "red"
            name = "x1"
        elif i == 1:
            iro = "green"
            name = "x2"
        elif i == 2:
            iro = "blue"
            name = "x3"
        elif i == 3:
            iro = "yellow"
            name = "x4"
        plt.plot(time, x[i].squeeze().detach().cpu().tolist(),
                 "-", color=iro, label=name)
    plt.ylabel("x_0")
    plt.grid(True)
    plt.legend()  # 凡例を表示

    # u のプロット
    plt.subplot(3, 1, 2)
    plt.plot(range(T-1), u[0, :].detach().cpu().numpy(),
             label="u", color="green")
    for i, val in enumerate(delta.squeeze().cpu().tolist()):
        if val == 1:
            plt.plot(i, u.squeeze().detach().cpu().numpy()[
                     i], 'o', color="black", markersize=9)  # detach を追加
    plt.ylabel("u")
    plt.grid(True)
    plt.legend()  # 凡例を表示

    # mark の推移のプロット
    plt.subplot(3, 1, 3)
    plt.step(range(T-1), delta[0, :].detach().cpu().numpy(),
             where='mid', label="mark", color="purple")
    plt.ylabel("mark")
    plt.xlabel("Time")
    plt.grid(True)
    plt.legend()  # 凡例を表示

    # グラフ表示
    plt.tight_layout()
    plt.savefig(Result_path + "Fig\\" + str(k) + ".svg")

    return cost, event_count


def Evolute_4para1(cstr, T, x0, k, K, M2, M1, M0, N, M, Q, R, Result_path):
    if not os.path.exists(Result_path+"Fig\\"):
        os.makedirs(Result_path+"Fig\\")
    x = torch.zeros(N, T, dtype=dtype).to(device)
    x_hat = torch.zeros(N, 1, dtype=dtype).to(device)
    u = torch.zeros(M, T-1, dtype=dtype).to(device)
    delta = torch.zeros(1, T-1, dtype=dtype)
    x[:, [0]] = x0
    cost = 0

    # シミュレーション実行
    for i in range(T-1):
        if i==0:
                x_hat = x[:, [i]]
                delta[:, [i]] = 1  # イベント発生時に1を記録
        else:
            if ET_4para(x[:, [i]], x_hat, M2, M1, M0, N) > 0:
                x_hat = x[:, [i]]
                delta[:, [i]] = 1  # イベント発生時に1を記録

        u[:, [i]] = K @ x_hat
        x[:, [i + 1]] = cstr.f(x[:, [i]], u[:, [i]])
    for i in range(T-1):
        cost += COST(x[:, [i]], u[:, [i]], Q, R)

    # イベント発生回数を計算して表示
    event_count = int(torch.sum(delta).item())
    time = range(T)

    plt.figure()

    # x_0 のプロット
    plt.subplot(3, 1, 1)
    for i in range(0, N):
        if i == 0:
            iro = "red"
            name = "x1"
        elif i == 1:
            iro = "green"
            name = "x2"
        elif i == 2:
            iro = "blue"
            name = "x3"
        elif i == 3:
            iro = "yellow"
            name = "x4"
        plt.plot(time, x[i].squeeze().detach().cpu().tolist(),
                 "-", color=iro, label=name)
    plt.ylabel("x_0")
    plt.grid(True)
    plt.legend()  # 凡例を表示

    # u のプロット
    plt.subplot(3, 1, 2)
    plt.plot(range(T-1), u[0, :].detach().cpu().numpy(),
             label="u", color="green")
    for i, val in enumerate(delta.squeeze().cpu().tolist()):
        if val == 1:
            plt.plot(i, u.squeeze().detach().cpu().numpy()[
                     i], 'o', color="black", markersize=9)  # detach を追加
    plt.ylabel("u")
    plt.grid(True)
    plt.legend()  # 凡例を表示

    # mark の推移のプロット
    plt.subplot(3, 1, 3)
    plt.step(range(T-1), delta[0, :].detach().cpu().numpy(),
             where='mid', label="mark", color="purple")
    plt.ylabel("mark")
    plt.xlabel("Time")
    plt.grid(True)
    plt.legend()  # 凡例を表示

    # グラフ表示
    plt.tight_layout()
    plt.savefig(Result_path + "Fig\\" + str(k) + ".svg")
    plt.close('all')
    return cost, event_count


def Evolute_4para2(cstr, T, x0, k, K, a, b, c, N, M, Q, R, Result_path):
    if not os.path.exists(Result_path+"Fig\\"):
        os.makedirs(Result_path+"Fig\\")
    x = torch.zeros(N, T, dtype=dtype).to(device)
    x_hat = torch.zeros(N, 1, dtype=dtype).to(device)
    u = torch.zeros(M, T-1, dtype=dtype).to(device)
    delta = torch.zeros(1, T-1, dtype=dtype)
    x[:, [0]] = x0
    cost = 0

    # シミュレーション実行
    for i in range(T-1):
        if i==0:
                x_hat = x[:, [i]]
                delta[:, [i]] = 1  # イベント発生時に1を記録
        else:
            if ET_4para2(x[:, [i]], x_hat, a, b, c, N) > 0:
                x_hat = x[:, [i]]
                delta[:, [i]] = 1  # イベント発生時に1を記録

        u[:, [i]] = K @ x_hat
        x[:, [i + 1]] = cstr.f(x[:, [i]], u[:, [i]])
    for i in range(T-1):
        cost += COST(x[:, [i]], u[:, [i]], Q, R)

    # イベント発生回数を計算して表示
    event_count = int(torch.sum(delta).item())
    time = range(T)

    plt.figure()

    # x_0 のプロット
    plt.subplot(3, 1, 1)
    for i in range(0, N):
        if i == 0:
            iro = "red"
            name = "x1"
        elif i == 1:
            iro = "green"
            name = "x2"
        elif i == 2:
            iro = "blue"
            name = "x3"
        elif i == 3:
            iro = "yellow"
            name = "x4"
        plt.plot(time, x[i].squeeze().detach().cpu().tolist(),
                 "-", color=iro, label=name)
    plt.ylabel("x_0")
    plt.grid(True)
    plt.legend()  # 凡例を表示

    # u のプロット
    plt.subplot(3, 1, 2)
    plt.plot(range(T-1), u[0, :].detach().cpu().numpy(),
             label="u", color="green")
    for i, val in enumerate(delta.squeeze().cpu().tolist()):
        if val == 1:
            plt.plot(i, u.squeeze().detach().cpu().numpy()[
                     i], 'o', color="black", markersize=9)  # detach を追加
    plt.ylabel("u")
    plt.grid(True)
    plt.legend()  # 凡例を表示

    # mark の推移のプロット
    plt.subplot(3, 1, 3)
    plt.step(range(T-1), delta[0, :].detach().cpu().numpy(),
             where='mid', label="mark", color="purple")
    plt.ylabel("mark")
    plt.xlabel("Time")
    plt.grid(True)
    plt.legend()  # 凡例を表示

    # グラフ表示
    plt.tight_layout()
    plt.savefig(Result_path + "Fig\\" + str(k) + ".svg")
    plt.close('all')
    return cost, event_count


def Evolute_4para2_J(cstr, T, x0, k, K, a, b, c, N, M, Q, R, Result_path):
    if not os.path.exists(Result_path+"Fig\\"):
        os.makedirs(Result_path+"Fig\\")
    x = torch.zeros(N, T, dtype=dtype).to(device)
    x_hat = torch.zeros(N, 1, dtype=dtype).to(device)
    u = torch.zeros(M, T-1, dtype=dtype).to(device)
    delta = torch.zeros(1, T-1, dtype=dtype)
    x[:, [0]] = x0
    cost = 0
    J_log=[]

    # シミュレーション実行
    for i in range(T-1):
        if i==0:
                x_hat = x[:, [i]]
                delta[:, [i]] = 1  # イベント発生時に1を記録
        else:
            if ET_4para2(x[:, [i]], x_hat, a, b, c, N) > 0:
                x_hat = x[:, [i]]
                delta[:, [i]] = 1  # イベント発生時に1を記録

        u[:, [i]] = K @ x_hat
        x[:, [i + 1]] = cstr.f(x[:, [i]], u[:, [i]])
    for i in range(T-1):
        cost += COST(x[:, [i]], u[:, [i]], Q, R)
        J_log.append(cost.item()*cstr.h)
    # イベント発生回数を計算して表示
    event_count = int(torch.sum(delta).item())
    time = range(T)


    return x,u,cost, event_count,J_log

def Evolute_NN(
    cstr, T, x0, k,
    K,
    hidden_W_list, hidden_B_list,
    out_W, out_B,
    N, M, Q, R,
    num_hidden, node_hidden,
    Result_path,
    Qf=None, lam=None, save_fig=True
):
    # Qf と lam を渡すと、学習時と同じ評価関数
    #   Σ(x'Qx + u'Ru + λδ) + x_T' Qf x_T
    # を返す。渡さない場合は従来どおり Σ(x'Qx + u'Ru) のみ。
    if not os.path.exists(Result_path+"Fig\\"):
        os.makedirs(Result_path+"Fig\\")
    x = torch.zeros(N, T, dtype=dtype).to(device)
    x_hat = torch.zeros(N, 1, dtype=dtype).to(device)
    u = torch.zeros(M, T-1, dtype=dtype).to(device)
    delta = torch.zeros(1, T-1, dtype=dtype)
    x[:, [0]] = x0
    cost = 0
    
    # シミュレーション実行
    for i in range(T-1):
        
        if i==0:
                x_hat = x[:, [i]]
                delta[:, [i]] = 1  # イベント発生時に1を記録
        else:
            o=ET_NN(
                x[:, [i]],
                x_hat,
                N,
                num_hidden, node_hidden,
                hidden_W_list, hidden_B_list,
                out_W, out_B,
                nn_input=getattr(cstr, "nn_input", "x_xhat")
            )
            if o > 0:
                x_hat = x[:, [i]]
                delta[:, [i]] = 1  # イベント発生時に1を記録
        u[:, [i]] = K @ x_hat
        x[:, [i + 1]] = cstr.f(x[:, [i]], u[:, [i]])
    for i in range(T-1):
        cost += COST(x[:, [i]], u[:, [i]], Q, R)
    if lam is not None:
        cost = cost + lam * delta.sum().to(dtype=dtype)       # 通信コスト λΣδ
    if Qf is not None:
        cost = cost + x[:, [T-1]].T @ Qf @ x[:, [T-1]]       # 終端コスト

    # イベント発生回数を計算して表示
    event_count = int(torch.sum(delta).item())
    if not save_fig:   # 評価本数が多いときは図を一部だけ保存する
        return cost, event_count
    time = range(T)

    plt.figure()

    # x_0 のプロット
    plt.subplot(3, 1, 1)
    for i in range(0, N):
        if i == 0:
            iro = "red"
            name = "x1"
        elif i == 1:
            iro = "green"
            name = "x2"
        elif i == 2:
            iro = "blue"
            name = "x3"
        elif i == 3:
            iro = "yellow"
            name = "x4"
        plt.plot(time, x[i].squeeze().detach().cpu().tolist(),
                 "-", color=iro, label=name)
    plt.ylabel("x_0")
    plt.grid(True)
    plt.legend()  # 凡例を表示

    # u のプロット
    plt.subplot(3, 1, 2)
    plt.plot(range(T-1), u[0, :].detach().cpu().numpy(),
             label="u", color="green")
    for i, val in enumerate(delta.squeeze().cpu().tolist()):
        if val == 1:
            plt.plot(i, u.squeeze().detach().cpu().numpy()[
                     i], 'o', color="black", markersize=9)  # detach を追加
    plt.ylabel("u")
    plt.grid(True)
    plt.legend()  # 凡例を表示

    # mark の推移のプロット
    plt.subplot(3, 1, 3)
    plt.step(range(T-1), delta[0, :].detach().cpu().numpy(),
             where='mid', label="mark", color="purple")
    plt.ylabel("mark")
    plt.xlabel("Time")
    plt.grid(True)
    plt.legend()  # 凡例を表示

    # グラフ表示
    plt.tight_layout()
    plt.savefig(Result_path + "Fig\\" + str(k) + ".svg")
    plt.close('all')
    return cost, event_count

def Evolute_rand(cstr, x0, T, K, L, N, M, Q, R):
    x = torch.zeros(N, T, dtype=dtype).to(device)
    x_hat = torch.zeros(N, 1, dtype=dtype).to(device)
    u = torch.zeros(M, T-1, dtype=dtype).to(device)
    delta = torch.zeros(1, T-1, dtype=dtype)
    x[:, [0]] = x0
    cost = 0

    # シミュレーション実行
    for i in range(T-1):
        max_norm = 0  # 最大状態ノルム
        total_norm = 0  # 平均計算用
        if i==0:
                x_hat = x[:, [i]]
                delta[:, [i]] = 1  # イベント発生時に1を記録
        else:
            if ET(x[:, [i]], x_hat, L, N) > 0:
                x_hat = x[:, [i]]
                delta[:, [i]] = 1  # イベント発生時に1を記録
                
        u[:, [i]] = K @ x_hat
        x[:, [i + 1]] = cstr.f(x[:, [i]], u[:, [i]])
        current_norm = torch.norm(x[:, [i]])
        max_norm = max(max_norm, current_norm)
        total_norm += current_norm
    for i in range(T-1):
        cost += COST(x[:, [i]], u[:, [i]], Q, R)

    return x, u, cost, torch.sum(delta).item()


def Evolute_rand_4para(cstr, x0, T, K, L2, L1, L0, N, M, Q, R):
    x = torch.zeros(N, T, dtype=dtype).to(device)
    x_hat = torch.zeros(N, 1, dtype=dtype).to(device)
    u = torch.zeros(M, T-1, dtype=dtype).to(device)
    delta = torch.zeros(1, T-1, dtype=dtype)
    x[:, [0]] = x0
    cost = 0

    # シミュレーション実行
    for i in range(T-1):
        max_norm = 0  # 最大状態ノルム
        total_norm = 0  # 平均計算用
        if i==0:
                x_hat = x[:, [i]]
                delta[:, [i]] = 1  # イベント発生時に1を記録
        else:
            if ET_4para(x[:, [i]], x_hat, L2, L1, L0, N) > 0:
                x_hat = x[:, [i]]
                delta[:, [i]] = 1  # イベント発生時に1を記録
        
        u[:, [i]] = K @ x_hat
        x[:, [i + 1]] = cstr.f(x[:, [i]], u[:, [i]])
        current_norm = torch.norm(x[:, [i]])
        max_norm = max(max_norm, current_norm)
        total_norm += current_norm

    for i in range(T-1):
        cost += COST(x[:, [i]], u[:, [i]])
    return x, u, cost, torch.sum(delta).item()


def Evolute_rand_4para2(cstr, x0, T, K, a, b, c, N, M, Q, R):
    x = torch.zeros(N, T, dtype=dtype).to(device)
    x_hat = torch.zeros(N, 1, dtype=dtype).to(device)
    u = torch.zeros(M, T-1, dtype=dtype).to(device)
    delta = torch.zeros(1, T-1, dtype=dtype)
    x[:, [0]] = x0
    cost = 0

    # シミュレーション実行
    for i in range(T-1):
        max_norm = 0  # 最大状態ノルム
        total_norm = 0  # 平均計算用
        if i==0:
                x_hat = x[:, [i]]
                delta[:, [i]] = 1  # イベント発生時に1を記録
        else:
            if ET_4para2(x[:, [i]], x_hat, a, b, c, N) > 0:
                x_hat = x[:, [i]]
                delta[:, [i]] = 1  # イベント発生時に1を記録
        u[:, [i]] = K @ x_hat
        x[:, [i + 1]] = cstr.f(x[:, [i]], u[:, [i]])
        current_norm = torch.norm(x[:, [i]])
        max_norm = max(max_norm, current_norm)
        total_norm += current_norm

    for i in range(T-1):
        cost += COST(x[:, [i]], u[:, [i]], Q, R)

    return x, u, cost, torch.sum(delta).item()


def Evolute_rand_NN(
    cstr, T, x0, k,
    K,
    hidden_W_list, hidden_B_list,
    out_W, out_B,
    N, M, Q, R,
    num_hidden, node_hidden,
    Result_path
):
    x = torch.zeros(N, T, dtype=dtype).to(device)
    x_hat = torch.zeros(N, 1, dtype=dtype).to(device)
    u = torch.zeros(M, T-1, dtype=dtype).to(device)
    delta = torch.zeros(1, T-1, dtype=dtype)
    x[:, [0]] = x0
    cost = 0

    # シミュレーション実行
    for i in range(T-1):
        max_norm = 0  # 最大状態ノルム
        total_norm = 0  # 平均計算用
        if i==0:
                x_hat = x[:, [i]]
                delta[:, [i]] = 1  # イベント発生時に1を記録
        else:
            if ET_NN(
                x[:, [i]],
                x_hat,
                N,
                num_hidden, node_hidden,
                hidden_W_list, hidden_B_list,
                out_W, out_B
            ) > 0:
                x_hat = x[:, [i]]
                delta[:, [i]] = 1  # イベント発生時に1を記録
                
        u[:, [i]] = K @ x_hat
        x[:, [i + 1]] = cstr.f(x[:, [i]], u[:, [i]])
        current_norm = torch.norm(x[:, [i]])
        max_norm = max(max_norm, current_norm)
        total_norm += current_norm
    for i in range(T-1):
        cost += COST(x[:, [i]], u[:, [i]], Q, R)

    return x, u, cost, torch.sum(delta).item()



def f(x, u):
    x_next = CSTR.plant.x_next(x, u)

    return x_next


def COST(x, u, Q, R):
    return x.T@Q@x+u.T@R@u
def COST_delta(x, u, Q, R,delta,lam):
    return x.T@Q@x+u.T@R@u+delta*lam

def Mike_x0(min_val, max_val, evol, N, seed=None):
    # seed を渡すと、その seed だけで初期値が決まる（学習側の乱数の影響を受けない）
    g = None if seed is None else torch.Generator().manual_seed(int(seed))
    initial_conditions = [
        torch.empty((N, 1), dtype=dtype).uniform_(min_val, max_val, generator=g)
        for _ in range(evol)
    ]
    return initial_conditions


def ET(x, x_hat, L, N):
    Rx = torch.zeros(2*N, 1, dtype=dtype).to(device)
    e = x_hat-x
    for i in range(0, N):
        Rx[i] = x[i]
    for i in range(0, N):
        Rx[i+N] = e[i]
    return Rx.T@L@Rx


def ET_4para(x, x_hat, L2, L1, L0, N):
    Rx = torch.zeros(2*N, 1, dtype=dtype).to(device)
    e = x_hat-x
    for i in range(0, N):
        Rx[i] = x[i]
    for i in range(0, N):
        Rx[i+N] = e[i]
    return Rx.T@L2@Rx+L1@Rx+L0


def ET_4para2(x, x_hat, L2, L1, L0, N):
    Rx = torch.zeros(2*N, 1, dtype=dtype).to(device)
    e = x_hat-x
    for i in range(0, N):
        Rx[i] = x[i]
    for i in range(0, N):
        Rx[i+N] = e[i]
    return (Rx-L1.T).T@L2@(Rx-L1.T)+L0

def ET_NN(x, x_hat,
               N,
               num_hidden, node_hidden,
               hidden_W_list, hidden_B_list,
               out_W, out_B, nn_input="x_xhat"):

    Rx = nn_features(x, x_hat, N, nn_input)  # 学習時と同じ入力
    model = NeuralNet_evol(
        N=N,
        num_hidden=num_hidden,
        node_hidden=node_hidden,
        hidden_weights=hidden_W_list,
        hidden_biases=hidden_B_list,
        out_weight=out_W,
        out_bias=out_B,
    ).to(device)
    O = model(Rx.T)
    return O



def Save(Result_file, epoch, lr, min_val, max_val, lam, Batch, datasize, AD, evolution, Average_cost_Last, Average_event_Last, Average_cost_Best, Average_event_Best, path, extra=None):
    # extra: 追加で記録する条件の辞書（外乱、NN構造、k、seed など）。列として Link の前に入る
    # Result_file で指定されたフォルダが存在しない場合は作成
    os.makedirs(Result_file, exist_ok=True)

    # 保存する Excel ファイルのパスを設定
    file_path = os.path.join(Result_file, "result.xlsx")

    if AD == 0:
        method = "Adam"
    elif AD == 1:
        method = "AdamW"
    else:
        method = "Unknown"  # 念のためのデフォルト値

    # 親フォルダにある Result フォルダのパスを計算
    result_dir = os.path.join(path, "Result", path.rstrip("/\\"))

    # HYPERLINK 関数を作成
    Hyper_link = f'=HYPERLINK("{result_dir}")'
    lr_str = f"{lr}"  # learning-Rateを文字列化

    # 新しい行のデータ
    new_row = [
        epoch,
        lr_str,
        lam,
        method,
        min_val,
        max_val,
        Batch,
        datasize,
        Average_cost_Last.item(),
        Average_event_Last.item() if isinstance(
            Average_event_Last, torch.Tensor) else Average_event_Last,
        Average_cost_Best.item(),
        Average_event_Best.item() if isinstance(
            Average_event_Best, torch.Tensor) else Average_event_Best,
        *[str(v) if isinstance(v, (list, tuple)) else v for v in (extra or {}).values()],
        Hyper_link
    ]
    header = ["Epoch", "learning-Rate", "Lambda", "Method", "x0_min", "x0_max", "MiniBatch_size", "Data_Size",
              "Last_Cost", "Last_Event_count", "Best_Cost", "Best_Event_count",
              *list((extra or {}).keys()), "Link"]

    # ファイルが存在する場合と存在しない場合で処理を分ける
    if os.path.exists(file_path):
        try:
            # 既存ファイルをロード
            wb = load_workbook(file_path)
            # 列構成が同じシートに追記する。無ければ新しいシートを作る（古い形式の行と混ざらないように）
            ws = None
            for sh in wb.worksheets:
                if [c.value for c in sh[1]] == header:
                    ws = sh
                    break
            if ws is None:
                ws = wb.create_sheet(f"v{len(wb.worksheets) + 1}")
                ws.append(header)
        except Exception as e:
            print(f"ファイル読み込みエラー: {e}")
            wb = None
    else:
        # 新しいワークブックを作成
        wb = Workbook()
        ws = wb.active
        # ヘッダーを書き込む
        ws.append(header)

    if wb is not None:
        # 新しい行を追加
        ws.append(new_row)
        # ファイルを指定した場所に保存
        wb.save(file_path)

    

def _to_np(x):
    # torch.Tensor → numpy に変換
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def plot_histories(self, best_epoch: int):
    loss = self.loss_history          # torch.Tensor [epoch]
    eval_cost = self.eval_cost_history
    lr = self.lr_history
    
def robust_ylim_torch(values: torch.Tensor,
                      lower: float = 0.0,
                      upper_percentile: float = 95.0,
                      margin: float = 1.2):
    """
    values: 1次元 torch.Tensor を想定
    """
    vals = values.detach().cpu()
    if vals.numel() == 0:
        return None
    # パーセンタイル (0-100) → quantile では 0-1
    q = torch.quantile(vals, upper_percentile / 100.0)
    return lower, float(q * margin)


def to_log_value(value):
    """Tensor / list[Tensor] / それ以外で場合分けしてログ用に変換"""
    if torch.is_tensor(value):
        # 単体 Tensor
        return value.detach().cpu().clone()

    if isinstance(value, list):
        # list（例: hidden_weights, hidden_biases）
        new_list = []
        for v in value:
            if torch.is_tensor(v):
                new_list.append(v.detach().cpu().clone())
            else:
                new_list.append(v)  # 念のためそのまま
        return new_list

    # int, float, str などはそのまま
    return value