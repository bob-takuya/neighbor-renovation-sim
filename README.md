# neighbor-renovation-sim

卒業論文「集合住宅の近隣住戸協働型改修の可能性を探るシミュレーション方法の研究」（板橋拓也、東京大学工学部建築学科、2026）の実験編で使った模型・実験・集計のコードです。論文の付録 B（再実行の手順）は、このリポジトリをクローンしたところから始まります。

## 必要なもの

- Python 3.10 以上（Python 3.13 で動作を確かめた）
- NumPy、pandas、Matplotlib、seaborn、SciPy

```
git clone https://github.com/bob-takuya/neighbor-renovation-sim.git
cd neighbor-renovation-sim
python3 -m pip install -r requirements.txt
mkdir -p out/e0 out/e1 out/figs out/s1 out/s1dev out/s2 out/s3 out/s4 out/s5a out/s6 out/s7 out/s8 out/s8a out/s8b
```

以下のコマンドはすべてリポジトリの最上位で実行します。出力は `out/` の下に書き出されます（出力先のフォルダは上の `mkdir` で作っておきます）。実行時間は 2 コアの計算機での目安で、論文の付録 B に書いてあります。

## 構成

| 場所 | 中身 |
|---|---|
| `trade/` | 模型（s0・s1・s3〜s8）と推定器（vstar・vstar2〜vstar5）。版の対応は論文の付録 C.1 と C.7 |
| `experiments/` | 各実験の実行・集計・図のスクリプトと、事前登録の文（`prereg_*.json`） |
| `tests/` | 模型の試験（版を足しても既定の設定で一つ前の版と同じ記録を返すこと など） |
| `out/*_frozen.sha256` | 事前登録で凍結した時点のファイルのハッシュ |
| `out/frozen_code.sha256` | 凍結したファイルの、コメントと docstring を除いた字句の列のハッシュ |
| `tools/check_frozen.py` | 凍結したファイルとの照合 |

## 試験と、凍結したファイルとの照合

```
PYTHONPATH=. python3 -m tests.test_s0
PYTHONPATH=. python3 -m tests.test_s1
PYTHONPATH=. python3 -m tests.test_s3
python3 tools/check_frozen.py
```

公開にあたり、`.py` のコメントと docstring から、実装に関係のない覚え書きを除きました。そのため `.py` はファイルのハッシュでは `out/*_frozen.sha256` と一致しません。`tools/check_frozen.py` は、`.py` については、コメント・docstring・空行を除いた字句の列が凍結したファイルと同じであることを `out/frozen_code.sha256` で確かめ、事前登録の文（`.json`）については、ファイルのハッシュが凍結の時点と同じであることを `out/*_frozen.sha256` で確かめます。

## 実験の実行

各章の実験のコマンドは論文の付録 B.3〜B.7 にあります。たとえば観察 1 の検証の実験は次のとおりです。

```
PYTHONPATH=. python3 -m experiments.s1_main cache
PYTHONPATH=. python3 -m experiments.s1_main main
PYTHONPATH=. python3 -m experiments.analyze_s1
```

名付けで行った推定器の別の実装との照合は次の順に実行します。

```
PYTHONPATH=. python3 -m experiments.s2_mine_for_indep   # trade/vstar.py の値 → out/s2/mine_for_indep.json
PYTHONPATH=. python3 -m experiments.s2_runs_for_indep   # 同じ条件の局の記録 → out/s2/runs.pkl
PYTHONPATH=. python3 -m experiments.s2_indep_vstar      # 別の実装の値 → out/s2/indep_values.csv
PYTHONPATH=. python3 -m experiments.s2_indep_compare    # 照合 → out/s2/indep_compare.json
```

`s2_runs_for_indep.py` と `s2_indep_compare.py` は、公開にあたり、論文の値を出したときの手順を書き起こしたものです。これで作った `mine_for_indep.json`・`indep_values.csv`・`indep_compare.json` は、論文に使ったものとファイルの中身まで一致することを確かめています。
