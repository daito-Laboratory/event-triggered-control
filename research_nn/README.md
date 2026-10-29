# NN-Based Event-Triggered Control — 研究コード（非公開・整理中）

このフォルダには、提供されたPythonコードを**元の処理内容を変更せず**に配置しています。研究室の公開許可を確認するまでは、リポジトリを **Private** にしてください。

## フォルダ構成

\`\`\`text
event-triggered-control/
├── README.md                     # リポジトリ全体の説明（別の教育用デモあり）
├── src/                          # 独立した教育用デモ（研究原本とは別）
└── research_nn/
    ├── README.md                 # この説明書
    ├── DETC.py                   # NN学習・検証・グラフ生成の起点
    ├── parameter.py              # 学習設定、保存先
    ├── plant.py                  # 2次元制御対象、外乱モデル
    ├── simulation.py             # 保存済みNNによるシミュレーション（要保存先調整）
    ├── beep.py                   # 通知音
    ├── requirements.txt
    └── My_module/
        ├── __init__.py
        ├── my_function_ET.py     # NN、コスト計算、学習、評価、作図処理
        └── plant_base.py         # 制御対象の基底クラス
\`\`\`

**階層上の重要点:** \`plant.py\` の \`from My_module.plant_base import PlantBase\` と、\`DETC.py\`・\`simulation.py\` の \`from My_module import my_function_ET\` が解決するように、\`My_module/\` は \`research_nn/\` **直下**に置いています。プロジェクトのルートではなく \`research_nn/\` を実行起点にしてください。

元のファイル名 \`parameter(4).py\` と \`simulation(1).py\` は、既存の \`import parameter\` や実行手順に合わせて \`parameter.py\`、\`simulation.py\` として登録しました。

## 制御モデル

離散時間線形システム：

$$
x_{k+1}=Ax_k+Bu_k+w_k,\qquad u_k=K\hat{x}_k
$$

今回添付された設定は、状態 $x_k\in\mathbb{R}^2$、入力 $u_k\in\mathbb{R}$ の**2次元系**です。

$$
A=\begin{bmatrix}1&0.8\\0&1.1\end{bmatrix},\qquad
B=\begin{bmatrix}1\\-1\end{bmatrix}
$$

NNは状態 $x_k$ と保持状態 $\hat{x}$ を入力に受け、通信判断に使うスカラーを出力します。シミュレーションでは正の出力を通信イベントとして扱います。学習時は設定によりSigmoidまたはストレートスルー推定を使用します。

最適化の評価関数には、状態コスト、入力コスト、通信コスト、終端コストが含まれます。

$$
J=\sum_{k=0}^{T-1}
\left(x_k^\top Qx_k+u_k^\top Ru_k+\lambda\delta_k\right)
+x_T^\top Q_fx_T
$$

※ \`Evolute_NN()\` が返す評価用コストは上式の全項ではなく、QRステージコストです。数値を混同しないでください。

## 実行の流れ（Windowsでの元コード想定）

1. Python環境に \`requirements.txt\` の依存関係を導入する。
2. ターミナルのカレントディレクトリを **\`research_nn/\`** にする。
3. \`parameter.py\`、\`plant.py\` で条件を確認する。
4. \`python DETC.py\` で学習・最良パラメータの検証・グラフ出力を実行する。

\`\`\`powershell
cd research_nn
python -m pip install -r requirements.txt
python DETC.py
\`\`\`

\`DETC.py\` はWindows専用の \`winsound\` を利用し、出力ディレクトリの構築にもWindows形式の区切り文字を含んでいます。ほかのOSでは調整が必要です。

### 学習時の出力

\`parameter.py\` の \`Folderpath\` 以下に、さらに **Sigmoidの設定を表すフォルダと実行日時フォルダ**が生成されます。

\`\`\`text
<Folderpath>/
└── <sig_tag>/
    └── <run_id>/
        ├── result.py                      # 学習したK、NNの重み・バイアス
        ├── condition.txt                  # 学習・評価条件と平均値
        ├── README.md                      # 実行ごとの実験メモ
        ├── sigmoid_k_history.csv
        ├── loss_train_full_linear.svg
        ├── eval_cost_full_linear.svg
        ├── sigmoid_k_history.svg
        ├── params.pt
        ├── params_all.pt
        └── Fig/                           # 状態、入力、通信判断の軌道図
\`\`\`

このほかにも学習曲線の対数表示、末尾区間の拡大図など複数のSVGが生成されます。上記の実行結果は**このリポジトリには未掲載**です。

### 別スクリプト \`simulation.py\` の注意点

そのままの \`simulation.py\` は \`parameter.Folderpath/result.py\` を読み込みますが、実際の \`DETC.py\` は \`<Folderpath>/<sig_tag>/<run_id>/result.py\` に保存します。**そのため、保存先を合わせないと学習結果を読み込めません。**

今回のアップロードでは元コードを変更していません。別スクリプトを動作させるには、対象となる実験の \`result.py\` のパスを指定する修正が必要です。

## 現状・公開時の注意

- 添付されたコードは保存しましたが、**学習ジョブ全体は実行していません**。実行結果や性能を検証済みとは主張しません。
- 元コードには実験固有の設定・出力先、研究室と関連する処理があるため、**外部公開前に指導教員や共同研究者との権利・公開範囲の確認が必要**です。
- 学習済み重み・実験ログ・生成ファイルを意図せず公開しないよう、\`.gitignore\` を追加しています。
- リポジトリ直下の \`src/\` はAI補助で作った独立デモです。この \`research_nn/\` とは区別してください。
