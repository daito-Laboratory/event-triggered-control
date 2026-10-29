# Neural Network-Based Event-Triggered Control

## 目的

ニューラルネットワーク（NN）で状態送信のタイミングを決定するイベントトリガ制御の学習・評価コードです。このフォルダは元の研究実装であり、リポジトリ直下の教育用デモとは異なります。

## 制御系と通信判断

2次元の離散時間系を対象とします。

$$
x_{k+1}=Ax_k+Bu_k+w_k,\qquad u_k=K\hat{x}_k
$$

$$
A=\begin{bmatrix}1&0.8\\0&1.1\end{bmatrix},\quad
B=\begin{bmatrix}1\\-1\end{bmatrix}
$$

- $x_k$：現在の状態（2次元）
- $\hat{x}_k$：制御器が最後に受信し保持している状態
- $u_k$：制御入力
- $w_k$：外乱
- $K$：NNトリガと同時に学習するフィードバックゲイン

この版のNN入力は $[x_k^\top,e_k^\top]^\top$ で、$e_k=\hat{x}-x_k$ と定義します。NNの出力が通信判断に使われ、学習時のSoft/Hard処理は `parameter.py` の設定で切り替えます。

評価関数は状態・入力・通信・終端コストを含みます。

$$
J=\sum_{k=0}^{T-1}
\left(x_k^\top Qx_k+u_k^\top Ru_k+\lambda\delta_k\right)
+x_T^\top Q_fx_T
$$

ここで、$\delta_k$ は送信判断、$\lambda$ は通信のコスト係数です。今回のソースでは最終評価時の通信回数に初回送信を含みます。

## 実行するファイルと配置

```text
DU_ETC/
├── My_module/
│   ├── my_function_ET.py      # 学習・評価・グラフ化
│   └── plant_base.py
└── ETC/
    └── Liner_system/
        ├── plant.py           # NN/parameter.py を参照
        └── NN/
            ├── parameter.py   # ハイパーパラメータ
            ├── DETC.py        # 学習・評価・結果出力
            ├── simulation.py  # 学習済みNNの再評価
            └── Result/        # 実験別の出力
```

**階層を変えないこと**：`DETC.py` は3階層上の `My_module/` を、`plant.py` は同じ `Liner_system/` の `NN/parameter.py` を参照します。

### 実行例（Windows）

```powershell
cd research_nn\DU_ETC\ETC\Liner_system\NN
python DETC.py
```

学習したNNの別途評価は `python simulation.py "結果フォルダのフルパス"` として実行できます。依存ライブラリは `torch`・`matplotlib`・`openpyxl` です。

## 今回整理する実験

| 項目 | 設定 |
|---|---|
| エポック数 | 100 |
| 学習率 | 0.003 |
| 通信重み $\lambda$ | 50 |
| NN | 隠れ層5層、各8ノード |
| NN入力 | $[x;e]$ |
| 初期値範囲 | 各成分 $[-5,5]$ |
| 外乱 | 各成分 一様分布 $[-0.2,0.2]$ |
| データ数 / バッチ | 64 / 8 |
| Sigmoid傾き | 初期1、各エポック1.01倍 |
| 学習 / 検証 / テストseed | 1 / 1001 / 2001 |

## 結果フォルダの階層

以下はユーザーが指定した**既存の実験フォルダ**です。GitHubではWindowsの絶対パスではなく、`NN/` からの相対パスで保存します。

```text
Result/
└── e100/lam50/x0_-5_5/dist_-0.2_0.2/
    └── DataSize64_Batch8/lr0.003/layer_5node_8/
        └── AdamW/stagecost_fixed/T30_sig_multiply_s1_r1.01/
            └── 20261027_193214_072517_seed1/
                ├── Best/
                │   └── Fig/
                │       └── 0.svg ...       # Bestの状態・通信軌道
                ├── Last/
                │   └── Fig/
                │       └── 0.svg ...       # Lastの状態・通信軌道
                ├── result.py               # 学習済みNNパラメータ
                ├── condition.txt           # 実験条件・数値
                └── *.svg                   # 学習曲線・評価曲線など
```

### READMEに掲載する結果

**実験結果は、添付ファイルと実験条件の対応を確認してから掲載します。** READMEに表示するのは代表的な2〜3枚程度のグラフと主要数値に限定し、残りの図は正しい `Result/` 以下に整理して保存します。

> **注意：** 今回の学習一式をこのGitHub環境で再実行したわけではありません。未発表成果・研究コードをPublicに変更する前に、公開許可を確認してください。
