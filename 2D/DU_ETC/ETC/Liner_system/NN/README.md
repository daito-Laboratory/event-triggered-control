# NN training and evaluation

本ディレクトリにはニューラルネットワークを用いたイベントトリガ制御の学習・評価スクリプトを配置しています。

## Files

| File | Description |
|---|---|
| [DETC.py](DETC.py) | NNの学習、Best/Lastの最終評価、結果の保存 |
| [parameter.py](parameter.py) | 実験条件・ハイパーパラメータ・乱数シード |
| [simulation.py](simulation.py) | 保存済みNNの再評価 |
| [../plant.py](../plant.py) | 2次元線形制御対象および外乱生成 |
| [../../../My_module/my_function_ET.py](../../../My_module/my_function_ET.py) | NNの定義、コスト計算、学習、描画処理 |

`My_module` は `DU_ETC/` 直下にあり、`plant.py` はこのディレクトリの1階層上です。Pythonファイル単位で移動するとインポートが解決できなくなるため、この構造のまま使用してください。

## Model

$$
x_{k+1}=Ax_k+Bu_k+w_k,\quad
u_k=K\hat{x}_k
$$

$$
A=\begin{bmatrix}1&0.8\\0&1.1\end{bmatrix},
\quad B=\begin{bmatrix}1\\-1\end{bmatrix}
$$

通信トリガ用NNの入力は `parameter.py` で設定します。今回の実験では `x_e`（$[x_k;e_k]$、$e_k=\hat{x}_{k-1}-x_k$）を使用します。

$$
J=\sum_{k=0}^{T-1}
\left(x_k^\top Qx_k+u_k^\top Ru_k+\lambda\delta_k\right)
+x_T^\top Q_f x_T
$$

`DETC.py` は複数のtrain seedに対応し、検証用と最終評価用の乱数系列を分離します。結果は実験条件と実行日時を含むディレクトリへ保存されます。

## Reproducing the run

Windows PowerShell（リポジトリのルートで依存関係を導入した後）：

```powershell
cd research_nn\DU_ETC\ETC\Liner_system\NN
python DETC.py
python simulation.py
```

別の学習seedを指定する場合：

```powershell
python DETC.py --train_seed 2
```

## Recorded run: 2026-10-27

| Setting | Value |
|---|---|
| Horizon / Epochs | 30 / 100 |
| Initial states | $[-5,5]^2$ |
| Disturbance | $[-0.2,0.2]^2$ |
| $\lambda$ | 50 |
| Learning rate | 0.003 |
| NN | 5 hidden layers, 8 nodes each |
| NN input | `x_e` |
| Sigmoid parameter | $k_0=1$; ×1.01/epoch |
| Training/validation/test seed | 1 / 1001 / 2001 |
| Best gain | $K=[0.326254,1.688476]$ |

**結果の保存先：** [20261027_193214_072517_seed1/](Result/e100/lam50/x0_-5_5/dist_-0.2_0.2/DataSize64_Batch8/lr0.003/layer_5node_8/AdamW/stagecost_fixed/T30_sig_multiply_s1_r1.01/20261027_193214_072517_seed1/)

- [評価コスト（全期間・線形表示）](Result/e100/lam50/x0_-5_5/dist_-0.2_0.2/DataSize64_Batch8/lr0.003/layer_5node_8/AdamW/stagecost_fixed/T30_sig_multiply_s1_r1.01/20261027_193214_072517_seed1/eval_cost_full_linear.svg)
- [評価コスト（全期間・対数表示）](Result/e100/lam50/x0_-5_5/dist_-0.2_0.2/DataSize64_Batch8/lr0.003/layer_5node_8/AdamW/stagecost_fixed/T30_sig_multiply_s1_r1.01/20261027_193214_072517_seed1/eval_cost_full_log.svg)
- [評価コスト（終盤）](Result/e100/lam50/x0_-5_5/dist_-0.2_0.2/DataSize64_Batch8/lr0.003/layer_5node_8/AdamW/stagecost_fixed/T30_sig_multiply_s1_r1.01/20261027_193214_072517_seed1/eval_cost_tail_linear.svg)
- [損失と学習率](Result/e100/lam50/x0_-5_5/dist_-0.2_0.2/DataSize64_Batch8/lr0.003/layer_5node_8/AdamW/stagecost_fixed/T30_sig_multiply_s1_r1.01/20261027_193214_072517_seed1/loss_history.svg)
- [Lastモデルの状態軌道・通信タイミング（0〜25）](Result/e100/lam50/x0_-5_5/dist_-0.2_0.2/DataSize64_Batch8/lr0.003/layer_5node_8/AdamW/stagecost_fixed/T30_sig_multiply_s1_r1.01/20261027_193214_072517_seed1/Last/Fig/)
- [学習済みパラメータ](Result/e100/lam50/x0_-5_5/dist_-0.2_0.2/DataSize64_Batch8/lr0.003/layer_5node_8/AdamW/stagecost_fixed/T30_sig_multiply_s1_r1.01/20261027_193214_072517_seed1/result.py)

実験結果の代表図は[リポジトリのトップページ](../../../../../README.md)に掲載しています。
