# Event-Triggered Control — Python Demo

> **研究コードについて：** 提供されたNN学習・検証・グラフ生成のPythonコードは、[research_nn/](research_nn/) に分けて保存しています。以下の `src/` はそれとは異なる教育用デモです。リポジトリは現在Privateです。

> **公開準備中 / Draft.** This is an independently generated **educational demonstration**, not the applicant's original laboratory code or unpublished research results.

## 概要 / Overview

離散時間の一次元線形システムについて、毎ステップ状態を送信する方式と、誤差が閾値を超えた場合だけ送信する方式を比較する小規模なPythonデモです。大学院の研究テーマに関連する基礎概念の紹介を目的としています。

**注意:** 本デモの閾値判定は、実際の研究で使用するニューラルネットワークやシンボリック回帰によるトリガ判断を再現していません。研究室の未公開データや既存コードを含みません。

## 制御モデルと数式 / Mathematical Model

### 1. 制御対象の状態方程式

以下の**離散時間・一次元の線形システム**を考えます。

$$
x_{k+1} = a x_k + b u_k
$$

| 記号 | 意味 | このデモの値 |
|---|---|---:|
| $x_k$ | 時刻 $k$ の制御対象の状態 | 初期値 $x_0=4.0$ |
| $u_k$ | 時刻 $k$ の制御入力 | 下記のフィードバック則で計算 |
| $a$ | 状態の係数 | $1.05$ |
| $b$ | 入力の係数 | $1.0$ |
| $K$ | フィードバックゲイン | $-0.65$ |

### 2. 通信があるとき・ないときの制御

制御器は、**最後に受信した状態**を $\hat{x}_k$ として保持します。

$$
u_k = K\hat{x}_k
$$

- **送信あり**：$\hat{x}_k=x_k$ に更新してから、$u_k$ を計算します。
- **送信なし**：$\hat{x}_k=\hat{x}_{k-1}$ とし、前回受信した状態をそのまま使います。

### 3. イベントトリガ条件（いつ送信するか）

現在の状態と、最後に受信した状態との差を

$$
e_k = x_k-\hat{x}_{k-1}
$$

と定義します。初回 $k=0$ は必ず送信し、$k\geq1$ では次の条件で送信します。

$$
\boxed{\;|e_k|\ \geq\ \sigma |x_k|+\varepsilon\;}
$$

つまり、**状態のずれが許容範囲以上になったときだけ送信する**方式です。

| 記号 | 意味 | このデモの値 |
|---|---|---:|
| $e_k$ | 更新前の状態推定誤差 | $x_k-\hat{x}_{k-1}$ |
| $\sigma$ | 状態の大きさに比例する閾値の係数 | $0.2$ |
| $\varepsilon$ | 絶対閾値 | $0.02$ |

**毎回送信する方式（Periodic）**では、全時刻で $\hat{x}_k=x_k$ とします。これを**イベントトリガ方式（Event-triggered）**と比較します。

> **研究モデルとの違い：** ここでのトリガは単純な誤差閾値です。現在取り組んでいるニューラルネットワークやシンボリック回帰によるトリガ判断の式・学習結果を再現したものではありません。研究内容の公開にあたっては別途確認が必要です。

## 実装 / Files

- `src/simulate.py` — 2方式のシミュレーション・結果CSVの生成
- `src/plot_results.py` — 状態と送信タイミングの可視化
- `tests/test_simulate.py` — 4つの簡単な自動テスト
- `requirements.txt` — グラフ作成に必要なパッケージ

## 実行 / Run

Python 3.10以上で動作する構成です。

```bash
python src/simulate.py
python -m unittest discover -s tests -v
python -m pip install -r requirements.txt
python src/plot_results.py
```

`src/simulate.py` はPython標準ライブラリのみで動作します。CSV・グラフは `results/` に生成されます。

## 動作例 / Sample output

デフォルト設定 (`steps=30`) を使うと、次の結果になります。

| Method | State transmissions |
|---|---:|
| Periodic | 30 / 30 |
| Event-triggered | 13 / 30 |

送信回数は設定したゲインや閾値に依存します。このサンプルは一般的な安定性を保証するものではありません。

## 研究成果・著作について

この初期デモは生成AIの補助で公開準備用に作成したひな形です。**応募者本人の開発実績や研究実装であるとは主張しません。** 本人が検証・改良した範囲を明確にし、研究成果を追加する場合は必要な公開許可を得てから掲載します。
