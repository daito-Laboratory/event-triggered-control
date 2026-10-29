# NNによるイベントトリガ制御 — 研究コード

研究室で作成した最新版のPythonコードを、**実際のモジュール参照関係を保った階層**に整理しています。公開許可の確認が済むまではGitHubを**非公開（Private）**に維持します。

## 現行ソース

```text
research_nn/
└── DU_ETC/
    ├── My_module/
    │   ├── __init__.py
    │   ├── my_function_ET.py          # NN・学習・評価・描画処理
    │   └── plant_base.py
    └── ETC/
        └── Liner_system/
            ├── plant.py               # 制御対象
            └── NN/
                ├── DETC.py            # NN学習・最終評価
                ├── parameter.py       # 実験設定
                ├── simulation.py      # 学習済みNNの評価
                ├── beep.py
                ├── README.md
                └── Result/            # 実験ごとにパラメータ・グラフを分類
```

このリポジトリではPC上の先頭フォルダ（デスクトップや「01_ETC」など）を省略してありますが、**NNから3階層上にMy_moduleを置くこと、plant.pyがNNの1階層上にあること**は元のコードに合わせています。

最新のソースは [NN/](DU_ETC/ETC/Liner_system/NN/) 以下です。先に登録した旧版の重複ソースは現行ツリーから外しました。Gitの履歴には旧版が残ります。

## 実験・結果

**詳細は [NN/README.md](DU_ETC/ETC/Liner_system/NN/README.md) に記載します。** READMEには代表的な結果のみを掲載し、各グラフは元の実験フォルダとBest/Last別の場所に保存します。

> このフォルダは研究コードです。リポジトリ直下の `src/` にある教育用デモとは区別してください。GitHubへの登録は、学習ジョブの実行成功や公開許可を意味しません。
