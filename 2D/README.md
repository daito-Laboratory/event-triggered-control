# 2D — Neural Network-Based Event-Triggered Control

状態2次元・入力1次元の離散時間線形システムを対象に、ニューラルネットワークによるイベントトリガ判定と状態フィードバックゲインの学習・評価を行います。

## Source code

- [NN/DETC.py](DU_ETC/ETC/Liner_system/NN/DETC.py)：学習、モデル選択、最終評価
- [NN/parameter.py](DU_ETC/ETC/Liner_system/NN/parameter.py)：実験設定
- [NN/simulation.py](DU_ETC/ETC/Liner_system/NN/simulation.py)：学習済みモデルの評価
- [Liner_system/plant.py](DU_ETC/ETC/Liner_system/plant.py)：制御対象と外乱
- [My_module/my_function_ET.py](DU_ETC/My_module/my_function_ET.py)：NN・コスト関数・学習・可視化の実装

**[NNの実行方法と実験条件](DU_ETC/ETC/Liner_system/NN/README.md)**

## Results

学習曲線・評価コスト・状態軌道などは、元の条件別ディレクトリ構造を維持して[NN/Result/](DU_ETC/ETC/Liner_system/NN/Result/)に保存しています。代表的な結果は[ルートREADME](../README.md#2d-experiment-results)をご覧ください。

## Directory layout

```text
2D/
└── DU_ETC/
    ├── My_module/
    │   ├── my_function_ET.py
    │   └── plant_base.py
    └── ETC/
        └── Liner_system/
            ├── plant.py
            └── NN/
                ├── DETC.py
                ├── parameter.py
                ├── simulation.py
                └── Result/
```

`NN/DETC.py` からのモジュール参照に必要な相対階層を保っています。
