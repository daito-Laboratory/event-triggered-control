# Event-Triggered Control — Python Demo

> **公開準備中 / Draft.** This is an independently generated **educational demonstration**, not the applicant's original laboratory code or unpublished research results.

## 概要 / Overview

離散時間の一次元線形システムについて、毎ステップ状態を送信する方式と、誤差が閾値を超えた場合だけ送信する方式を比較する小規模なPythonデモです。大学院の研究テーマに関連する基礎概念の紹介を目的としています。

**注意:** 本デモの閾値判定は、実際の研究で使用するニューラルネットワークやシンボリック回帰によるトリガ判断を再現していません。研究室の未公開データや既存コードを含みません。

## Model

```text
x[k+1] = a*x[k] + b*u[k]
u[k] = K*x_last_sent[k]

Event-triggered:
send if k == 0 or |x[k] - x_last_sent| >= sigma*|x[k]| + epsilon
```

状態が送信されないとき、制御器は最後に受け取った状態を使います。

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
