# audio-separator-rs

[English](README.md) | [中文](README.zh-CN.md) | **日本語** | [한국어](README.ko.md)

Rust によるボーカル / 伴奏分離ツール。コアロジックは再利用可能な crate として分離され、3 つの利用形態を提供します:

- **CLI**（`asep`）: 単発実行 — ローカル推論、MVSEP クラウド API、または自分の asep-server へ接続;
- **HTTP サーバー**（`audio-separator-server`）: 常駐プロセス。アップロード / タスク / ダウンロード / キャンセルの REST API を提供;
- **crate**（`audio-separator-core`）: ライブラリとして他の Rust プログラムに組み込み。

## アーキテクチャ

```
crates/
├── audio-separator-core    コア: モデル管理、推論エンジン、音声 IO、MVSEP クライアント
├── audio-separator-cli     CLI エントリ（バイナリ asep）
└── audio-separator-server  axum HTTP サーバー（タスクキュー + デュアルバックエンド）
```

### ローカル推論エンジン（バックエンド `local`）

| アーキテクチャ | エンジン | 状態 |
| --- | --- | --- |
| mdx（UVR-MDX-NET） | ONNX Runtime | ✅ |
| bs_roformer | candle（純 Rust、f64 STFT + band-split + 双軸アテンション） | ✅ |
| mel_band_roformer | candle（Slaney mel フィルタバンク + MSST 構造） | ✅ |
| bs_polarformer | candle（PoPE 極座標位置埋め込み） | ✅ |

Roformer の重みは `ckpt`（pickle）または `safetensors` で提供されます。`ckpt` は初回使用時に自動で `safetensors` に変換・キャッシュされます（遅延変換）。CPU 推論で、性能は参考実装と同程度です（詳細は `PLAN.md` §10）。

### MVSEP クラウドバックエンド（バックエンド `mvsep`）

[MVSEP API](https://mvsep.com/zh/full_api) 経由で分離を実行します。非 Premium アカウントは同時実行 1 タスク・日次クォータ制限で、コード内でセマフォによるレート制限を行います。

## クイックスタート

### ビルド

Rust ツールチェーンが必要です。ローカル ORT（ONNX Runtime 1.28）の設定は下記「ローカルビルド手順（Windows / ONNX Runtime）」を参照。

```sh
cargo build --release --workspace
```

成果物: `target/release/asep.exe`、`target/release/audio-separator-server.exe`。

### CLI での分離（ローカル）

```sh
# モデル初回使用時にモデルリスト（models.json 参照）から自動ダウンロード
asep separate input.wav -o out/ --model model_bs_polarformer_float16

# 出力フォーマット指定（wav/wav32/flac/flac24/mp3/m4a）。ローカルバックエンドの M4A は
# 外部 ffmpeg プロセスでエンコード（AAC 320 kbps）
asep separate input.wav -o out/ --model UVR_MDXNET_9482 --format flac

# ffmpeg のパスを明示（探索順: --ffmpeg > config ffmpeg.path > PATH）
asep separate input.wav -o out/ --model UVR_MDXNET_9482 --format m4a --ffmpeg "C:\path\to\ffmpeg.exe

# モデル重み + パラメータ設定（yaml/json）を直接指定し、models.json を完全にバイパス:
#   --model       ダウンロード URL / ローカルパス（+ --arch でアーキテクチャ指定）
#   --config-url  パラメータファイルの URL / ローカルパス。アーキテクチャパラメータの正典ソース
asep separate input.wav -o out/ --model ./model.ckpt --arch bs_polarformer --config-url ./config.yaml
asep separate input.wav -o out/ --model https://.../model.ckpt --arch mel_band_roformer --config-url https://.../config.yaml
```

### M4A 出力（ローカルバックエンド）

M4A は外部 `ffmpeg` プロセスでエンコードされます（AAC-LC 320 kbps）。ffmpeg の探索順序: `--ffmpeg <path>`（CLI）→ 設定 `ffmpeg.path`（TOML `[ffmpeg] path = "..."`）→ `PATH`。ffmpeg が見つからない場合は、パスの設定かインストールを促す明確なエラーを返します。M4A の*入力*ファイルは symphonia でネイティブにデコードされ、ffmpeg は不要です。

### CLI での分離（MVSEP クラウド）

```sh
asep separate input.wav -o out/ --backend mvsep \
  --model model_bs_polarformer_float16 --api-key YOUR_MVSEP_KEY --format flac

# 環境変数で Key を渡す場合
ASEP_MVSEP_API_KEY=YOUR_KEY asep separate input.wav -o out/ --backend mvsep --model ...
```

### CLI での分離（自分のサーバー）

まず自分の asep-server を起動し、CLI からドライブします — CLI がファイルのアップロード・タスクのポーリング・ステムのダウンロードを担当します:

```sh
# ターミナル 1: サーバーを起動（デフォルト 127.0.0.1:8080）
audio-separator-server

# ターミナル 2: サーバー経由で分離
asep separate input.wav -o out/ --backend server --model model_bs_polarformer_float16

# リモートサーバー / Bearer 認証
asep separate input.wav -o out/ --backend server \
  --server-url http://10.0.0.5:8080 --auth-token secret --model UVR_MDXNET_9482

# サーバーが提供するモデルリストの取得 / サーバータスクの確認
asep models --backend server --server-url http://127.0.0.1:8080
asep job-status <task_id> --server-url http://127.0.0.1:8080
```

補足: ローカル入力ファイルはサーバーへアップロードされます。URL 入力は `audio_url` フィールドとして透過されます（URL 入力はサーバー側では MVSEP バックエンドのみ対応）。サーバーは事前に起動しておく必要があり、CLI は起動しません。クライアントは `--server-url` へ直接接続し、ダウンロードプロキシは使いません。

### モデルリストとランキング

マニフェスト内の全モデルを一覧表示し、必要に応じて MUSDB18-HQ SDR またはコミュニティ推奨でランク付けします:

```sh
# 全モデルを一覧（名前順）
asep models

# ボーカル分離品質でランク付け（MUSDB18-HQ 中央値 SDR、降順）
asep models --sort sdr

# SDR 上位 10 モデル
asep models --sort sdr --top 10

# コミュニティ推奨順でランク付け（deton24 ガイド、昇順；カテゴリ+順位を表示）
asep models --sort community

# 単一モデルの詳細を確認（SDR スコアとランキング情報があれば表示）
asep model-info model_bs_roformer_ep_368_sdr_12.9628

# ランキングデータは別ファイル（rankings.json）にあります。直接確認:
asep rankings                            # コミュニティガイド節、カテゴリ+順位でグループ化
asep rankings --section mvsep            # MVSEP multisong リーダーボードスナップショット（instrum 順位）
asep rankings --section mvsep --view vocals --top 10
asep rankings --section community --sort sdr

# アーキテクチャ / ステム数 / ステム名でフィルタ（--sort / --top と組み合わせ可）
asep models --arch bs_roformer           # bs_roformer の全モデル
asep models --arch mdx,vr                # mdx または vr のいずれか
asep models --stems 4                    # ちょうど 4 ステム
asep models --stem drums                 # ドラム分離が可能なモデル
asep models --arch mdx --stem vocals --top 5
```

スコアとランキングは3つのソースに由来します：python-audio-separator のベンチマーク（MUSDB18-HQ 中央値 SDR、`models.json` に保持）、deton24 UVR-MDX-Demucs-GSEP コミュニティガイド（カテゴリ順位 + fullness/bleedless/SDR メトリクス）、MVSEP multisong リーダーボードスナップショット（ステム別 SDR）。後者2つは独立した `rankings.json` に格納されます（下記参照）。SDR ソート時、スコアのないモデルは最後に配置されます。コミュニティソート時、コミュニティ推奨のないモデルは最後に配置されます。mdx / bs_roformer / mel_band_roformer / bs_polarformer 以外のアーキテクチャは MVSEP クラウドバックエンド用に登録されており、ローカルでは実行できません。アーキテクチャ（`--arch`、カンマ区切り複数値・いずれか一致）、ステム数（`--stems`、完全一致）、ステム名（`--stem`）でソート前にリストを絞り込めます。サーバー API でも同じフィルタを受け付けます（`/api/v1/models?arch=…&stems=…&stem=…`）。

### サーバー

```sh
# 起動（デフォルト 127.0.0.1:8080）
audio-separator-server --api-key YOUR_MVSEP_KEY --workers 1

# 認証を有効化（Bearer Token、デフォルトはオフ）
audio-separator-server --auth-token secret
```

REST エンドポイント:

| メソッド | パス | 説明 |
| --- | --- | --- |
| POST | `/api/v1/separate` | multipart でタスク送信（`audio` ファイルまたは `audio_url` + `model` + `backend` + `format` + 任意の `config_url`） |
| GET | `/api/v1/tasks/{id}` | タスク状態（queued/running/done/failed/cancelled + 進捗） |
| GET | `/api/v1/tasks/{id}/download?stem=` | ステム結果のダウンロード |
| DELETE | `/api/v1/tasks/{id}` | 実行中タスクのキャンセル / 終了タスクの削除 |
| GET | `/api/v1/models?backend=local\|mvsep[&arch=&stems=&stem=]` | モデルリスト（アーキテクチャ / ステム数 / ステム名の任意フィルタ、local のみ）/ MVSEP アルゴリズムカタログ |
| GET | `/api/v1/rankings` | ランキングデータ（コミュニティガイド / MVSEP リーダーボードスナップショット） |
| GET | `/api/v1/health` | ヘルスチェック |

アップロード例:

```sh
curl -F "audio=@input.wav" -F "model=model_bs_polarformer_float16" \
  -F "backend=local" -F "format=flac" http://127.0.0.1:8080/api/v1/separate
```

## モデルリスト（models.json）

モデルのメタ情報（ダウンロード URL、sha256、アーキテクチャパラメータ、MVSEP マッピングなど）は JSON ファイルで管理されます:

- デフォルトではリポジトリ内の `models.json` を読み込み;
- `--models-url <url>` / 設定 `models.list = { url = "..." }` でリモートリストを取得（ローカルパス / URL どちらも可）;
- リストは GitHub リポジトリ [delusion-lab/asep-models](https://github.com/delusion-lab/asep-models) で独立管理（マニフェスト JSON のみをホストし、重みは公式ソースに残します）。

エントリ例:

```json
{
  "name": "model_bs_polarformer_float16",
  "architecture": "bs_polarformer",
  "source_url": "https://huggingface.co/.../model_bs_polarformer_float16.ckpt",
  "sha256": "省略可、未設定時は検証スキップ",
  "config_url": "省略可、yaml/json パラメータファイル（URL 対応）",
  "mvsep": { "sep_type": 123 }
}
```

`config_url` により、オープンソースモデル作者が重みとともに公開しているパラメータファイル（yaml/json）を URL で直接参照し、アーキテクチャパラメータの正典ソースとできます。

**models.json を完全にバイパス**: `--model <URL|ローカルパス> --config-url <URL|ローカルパス>`（CLI）、または multipart の `config_url` フィールド（サーバー）
で、重みとパラメータファイルをマニフェスト登録なしで直接指定できます。名前で参照する場合、`config_url` はマニフェストエントリの同名設定を上書きします。
サーバー `POST /api/v1/separate` の `config_url` フィールドも同様に有効です。

## ランキングデータ（rankings.json）

ランキング/推奨データはモデルリストとは別に管理されます。そのソース（コミュニティガイド、MVSEP リーダーボード）はカタログとは独立して更新されるためです:

- `community` 節：deton24 UVR-MDX-Demucs-GSEP コミュニティガイドエントリ（`name` はモデルリストと一致。rank / category / metrics / source / url を含む）;
- `mvsep` 節：MVSEP multisong リーダーボードスナップショット（プラットフォームのアルゴリズム名、ステム別 SDR、各ソートビューの順位、品質チェックエントリ URL）。

読み込みはモデルリストと同じ仕組みです：デフォルトでリポジトリ内の `rankings.json` を読み込み（`models.json` と同じディレクトリフォールバック）；`--rankings-url <url>` / 設定 `models.rankings = { url = "..." }` でリモートファイルを取得；`--rankings-file <path>` / `{ path = "..." }` でローカルファイルを指定。ランキングファイルは任意です——なくてもすべて動作します（コミュニティ表示とソートは空にフォールバック）。

MVSEP スナップショットはいつでも更新できます（community 節は変更されません）:
`python scripts/fetch-mvsep-rankings.py --proxy <プロキシ>`（プロキシは任意、環境変数にフォールバック）。

## ネットワークとプロキシ

すべてのネットワーク操作（モデルダウンロード、リスト取得、MVSEP 呼び出し）は HTTP プロキシ経由で統一されます。解決順:

`config.network.proxy`（明示設定）→ `ALL_PROXY` → `HTTPS_PROXY` → `HTTP_PROXY`

## ローカルビルド手順（Windows / ONNX Runtime）

`ort` crate はネイティブの ONNX Runtime DLL を必要とします（crates.io の `ort` は RC 版のみ公開。デフォルトの静的ライブラリは MSVC STL と競合します）。

```powershell
powershell -ExecutionPolicy Bypass -File scripts/fetch-ort.ps1
```

スクリプトは公式 `onnxruntime-win-x64-1.28.0` を `%LOCALAPPDATA%\asep-ort` にダウンロードし、`scripts/` のコメントに従って `onnxruntime.dll` を `target/<profile>` へ配布します（実行時は exe ディレクトリが優先ロードされます。古い DLL は使用しないでください）。`scripts/fetch-ort.ps1` は `.cargo/config.toml` も生成します（マシンローカル、コミット対象外）。

## テスト

```sh
cargo test --workspace
```

対象: モデルリスト解析、アーキテクチャパラメータ、エンコーダラウンドトリップ（WAV16/32、FLAC16/24、MP3）、サーバーのヘルスチェック / 認証 / モデルリスト。

## Docker

```sh
docker build -t audio-separator-server .
docker run -p 8080:8080 -e ASEP_MVSEP_API_KEY=... audio-separator-server
```

イメージは `local`（ONNX Runtime Linux ライブラリ含む）と `mvsep` の両バックエンドに対応。Roformer の重みは初回使用時にネットワークアクセスが必要です。
