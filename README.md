# DiffSinger Duration CPU

OpenVPI版DiffSingerをCPUのみで動かし、公式VarianceモデルのDuration predictorを3方式で学習・比較するためのセットアップです。DiffSinger本体は固定コミットで取得し、`third_party/`以下に置きます。

## セットアップ

LinuxとPython 3.10～3.12が必要です。次のコマンドでCPU版PyTorch、公式依存関係、DiffSinger本体を仮想環境に構築します。利用可能ならPython 3.12を選びます。

```bash
bash scripts/setup_cpu.sh
source .venv-cpu312/bin/activate
```

CPU学習はGPUよりかなり遅くなります。モデルサイズとbatchを小さめにした初期設定を使います。

## データ

`data/dataset.zip`にはWAV 3件と時間付き音素ラベル（HTK `.lab`）が含まれています。次のコマンドでDiffSingerのraw形式へ変換します。

```bash
python scripts/prepare_lab_dataset.py data/dataset.zip data/duration_cpu/raw_mora
```

変換後の音声は`data/duration_cpu/raw_mora/wavs/`、ラベルは`transcriptions.csv`、音素辞書は`data/duration_cpu/phonemes.txt`に作られます。既定では`data/sho2.zip`の`sho2/phonemes.json`から日本語音素41種も辞書へ加えます。教師データにない音素も語彙に保持でき、SHO事前学習モデルなら41音素を推論できます。自作モデルの未出現音素embeddingは学習されないため、精度は保証できません。`.lab`に単語境界はないので、`ph_num`は子音列＋母音を1モーラとする近似です。単語/音符境界付きデータでの再評価が必要です。

## 音素スペクトログラム

各WAVの音素境界をスペクトログラムに重ね、音素ごとの出現を時間方向に96フレームへ正規化して平均した画像を作成できます。

```bash
.venv/bin/python scripts/plot_phoneme_spectrogram.py data/dataset.zip
```

画像は`data/phoneme_spectrogram.png`に出力されます。上段にWAVごとのスペクトログラムと音素区間、下段に音素別の平均スペクトログラムを表示します。

## 音素データGUI

ブラウザーで教師データと推論入力の音素を確認できます。教師データではZIP内のWAV再生、スペクトログラム、`.lab`の時間境界、音素一覧を表示します。推論データは`data/`以下の`.ds`を一覧から選べるほか、手元の`.ds` / `.json` / `.lab`も読み込めます。`.ds`の`ph_seq`を音素一覧・タイムラインに表示し、Duration推論後は選択した音響モデルでWAV合成、再生、保存ができます。

```bash
python scripts/phoneme_viewer.py --host 0.0.0.0
```

ブラウザーで`http://localhost:8765`を開きます。ポートを変える場合は`--port`を指定してください。別の教師ZIPや推論データの場所は`--training-archive`と`--inference-dir`で指定できます。`.ds`に`ph_dur`がない場合、GUIは音素の順序だけを示し、durationを予測値として表示しません。

推論タブではSHO ZIP内の事前学習済みDuration ONNX（41音素対応）または自作checkpointを選び、`.ds`の音素長を予測できます。`data/awaki_New Part_00.ds`はSHOモデルで推論できます。自作checkpointは辞書にある音素なら推論できますが、教師例がない音素の精度は保証されません。スペクトログラムには対応するWAVも読み込ませます。Durationモデルは音素長を予測するモデルで、音声波形そのものは生成しません。推論後は教師データの音素別平均を参考に句頭の最初の発音音素を補正し、0.30秒を上限にします。短縮分は次の母音音素へ配分し、フレーズ総時間を保ちます。後続に母音がない場合は、次の発音音素へ配分します。音素境界はスペクトログラムとタイムライン上の縦線で示されます。

音声合成用に公式PC-NSF-HiFiGAN（44.1 kHz / hop 512 / 128 mel bins）を配置済みです。checkpointは`third_party/DiffSinger/checkpoints/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.ckpt`です。配布元は[DiffSinger Community Vocoders](https://github.com/openvpi/vocoders/releases/tag/pc-nsf-hifigan-44.1k-hop512-128bin-2025.02)で、同梱の`NOTICE.txt`にあるCC BY-NC-SA 4.0ライセンスに従ってください。

合成中はログと進捗を逐次表示し、完成したフレーズから順にプレビュー再生しながらスペクトログラムを更新します。最後に、フレーズのoffsetと重なりを反映した全体WAVを再生・保存できます。Checkpoint / ONNXのどちらもフレーズ単位で処理します。

推論画面はピアノロール中心の歌声編集画面です。.ds/.jsonを複数選択してトラックとして読み込み、後から「トラックを追加」できます。BPM・拍子、グリッド、ズーム、音符追加/削除、音高ドラッグ、右端ドラッグでの長さ変更、歌詞テキストとDS音素列の編集、Undo/Redoに対応します。音符の音高・長さを変更すると、音符列に加えてph_durとF0カーブも更新します。編集したトラックはDSとして、トラック構成とBPM/拍子は独自の`.dsproj`として保存できます（Ctrl+S）。合成・Duration推論は選択中のトラックに対して実行します。歌詞から音素への自動変換は行わないため、音素列はモデル辞書にある記号で入力してください。BPM/拍子は編集グリッドとプロジェクト設定であり、DiffSingerへ渡す音符・音素の秒単位durationを一括変換するものではありません。

外部プロジェクトはUST、USTX、MIDI、VSQXとOpenUtauの`.oup`から読み込めます。歌詞は基本的な日本語かな/ローマ字を音素へ変換しますが、OpenUtau本体のphonemizerと同等ではありません。読みへ変換できない歌詞はエラーになります。ポリフォニックなMIDIは歌声向けの単旋律へ変換し、同時発音では高い音、音符の重なりでは後続音を優先します。MIDIのSMPTEタイムベースは未対応です。BGMを読み込むと波形レーン上で位置と音量を調整でき、複数の歌声トラックとBGMを同時にミックスしてWAVへ保存できます。位置や音量を変えた後は「ミックスを更新」で再合成せずに反映できます。`.dsproj`には読み込んだBGMも埋め込まれます。

GUIは音響モデルをCheckpointかONNXから選べます。Checkpointは`third_party/DiffSinger/checkpoints/<実験名>/`に`config.yaml`と`model_ckpt_steps_*.ckpt`を置きます。ONNXは公式DiffSinger exportの`<実験名>.onnx`、`dsconfig.yaml`、音素・言語辞書JSON（複数歌手モデルの場合は`.emb`も）を一緒に`third_party/DiffSinger/artifacts/`または`data/acoustic_models/`に置いてください。ONNXは配布vocoderと同じ44.1 kHz / hop 512 / 128 mel bins / natural-log mel設定が必要です。GUIは合成可能なモデルを自動検出します。合成する`.ds`には`ph_seq`、`ph_dur`、`f0_seq`、`f0_timestep`が必要です。`ph_dur`がなければ先にGUIでDuration推論を実行します。

### Windowsローカル実行アプリ

DiffSingerソース、`data/dataset.zip`、vocoder、少なくとも1つの音響CheckpointまたはONNXモデルが配置された状態で、Python 3.12を用意し、64-bit Windows上で次を実行するとローカルGUIアプリを作成します。ビルド時にCPU版PyTorch、ONNX Runtime、DiffSinger依存パッケージをダウンロードし、推論用ランタイムとモデル/データをアプリへ同梱します。

```powershell
.\scripts\build_windows.ps1
```

完成した`dist\DiffSingerGUI\DiffSingerGUI.exe`を起動すると、ローカルWeb GUIがブラウザーで開きます。`.exe`単体ではなく`dist\DiffSingerGUI\`フォルダー全体を一緒に移動してください。ビルドと初回の依存ダウンロードにはインターネット接続が必要ですが、ビルド後の起動・合成はオフラインで行えます。大きなcheckpoint、vocoder、ONNXモデルもビルド時に同梱されるため、配布先のライセンスとモデル利用条件を確認してください。Windows向けバイナリはWindows上でビルドする必要があります。

GitHub Actionsから作る場合は、モデルと`data/`をGit LFSで公開リポジトリへ登録し、`build-windows-exe`ブランチへpushします。`.github/workflows/build-windows-exe.yml`がWindows runner上でビルドし、`DiffSingerGUI-Windows-x64` artifactを7日間保存します。これはクロスコンパイルではなく、Windows上でのネイティブビルドです。

## 軽量音響モデル

小さい音響モデルを3サイズ（hidden size 32 / 64 / 96）で用意できます。各モデルのLYNXNet2拡散バックボーンは1層で、depthwise Conv1Dのkernel sizeは31です。設定だけ生成する場合:

```bash
python scripts/run_acoustic_models.py \
  --raw-data-dir data/duration_cpu/raw_mora \
  --binary-data-dir data/acoustic_cpu/binary \
  --dictionary data/duration_cpu/phonemes.txt \
  --language ja \
  --speaker my_singer \
  --spk-id 0 \
  --test-prefix data-kaza-001
```

設定ファイルは`third_party/DiffSinger/configs/acoustic_cpu_<サイズ>_<日時>.yaml`に作られます。学習も実行する場合は`--train`を追加します。学習回数は`--max-updates`と`--val-check-interval`で指定できます（既定値は1000 / 100）。vocoder検証を無効にしているため学習にはvocoder checkpointは不要ですが、合成には配布済みのPC-NSF-HiFiGAN checkpointを使います。少数録音での学習結果は品質を保証しません。

比較に使うvalidation utteranceのprefixを`--test-prefix`で指定します。同じprefixを3方式で使うため、方式間の比較条件は揃います。以下の例では`data-kaza-001`を保留します。

## 学習と比較

```bash
python scripts/run_duration_sweep.py \
	--raw-data-dir data/duration_cpu/raw_mora \
	--binary-data-dir data/duration_cpu/binary_allphones \
	--dictionary data/duration_cpu/phonemes.txt \
	--language ja \
	--speaker my_singer \
	--spk-id 0 \
	--test-prefix data-kaza-001
```

スクリプトは小型の`attn`、`fs2`、`resnet`構成を含む6候補を同一条件で学習します。既定は5候補を30 updateで比較し、`attn_small`を60 updateまで学習する段階候補も含みます。各候補を10 updateごとに検証します。精度が最良候補から2ポイント以内のモデルから、checkpointサイズ×1 update秒が最小のものを採用し、比較結果JSONへ精度・速度・容量を保存します。学習ステップ数は`--max-updates`、検証間隔は`--val-check-interval`で変更できます。設定だけ先に作る場合は`--prepare-only`を追加してください。

まずvalidationの`metrics/ph_dur_acc`が最高の候補を特定し、その精度から2ポイント以内の候補ではcheckpoint容量×1 update秒が最小のものを採用します。結果JSONに採用実験名とcheckpointパス、全候補の指標を記録します。採用モデルは公式のVariance checkpointとして保存されます。

validation録音以外の2件を学習に使います。3録音だけの比較結果は暫定的なため、候補の優劣は追加録音でも再検証してください。
