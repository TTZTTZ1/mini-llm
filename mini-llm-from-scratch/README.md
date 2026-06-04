# 从零训练中文 Mini LLM

这是一个面向学习和实验复现的小型中文 Decoder-only Transformer 项目。目标不是追求大模型效果，而是用较短周期完整走通 LLM 的核心链路：中文数据准备、Tokenizer 训练、Decoder-only Transformer 实现、训练、生成、上下文长度消融、位置编码消融、采样策略对比和 KV Cache 推理加速实验。

项目默认按 Linux 服务器训练来设计。本地 Mac 主要负责写代码、提交 Git、运行 CPU 级别的快速测试；腾讯 CloudStudio Linux/A800 服务器负责下载数据、预处理、正式训练和推理基准测试。

## 项目结构

```text
mini-llm-from-scratch/
├── configs/                 # 训练和消融实验配置
├── experiments/             # 实验记录模板
├── report/                  # 实验报告草稿
├── scripts/                 # 一键下载、预处理、训练和评测脚本
├── src/mini_llm/            # 核心源码
├── tests/                   # 单元测试和脚本测试
├── environment.yml          # Conda 环境配置
├── requirements.txt         # Python 依赖
└── README.md                # 项目说明
```

核心源码职责：

- `src/mini_llm/config.py`：读取配置、校验参数、解析项目路径。
- `src/mini_llm/data.py`：构建训练样本、读写 token 二进制文件、创建 DataLoader。
- `src/mini_llm/tokenizer_train.py`：训练中文 BPE Tokenizer。
- `src/mini_llm/model.py`：实现 Decoder-only Transformer、因果注意力、KV Cache。
- `src/mini_llm/rope.py`：实现 RoPE 旋转位置编码。
- `src/mini_llm/train.py`：训练入口，负责训练循环、验证、保存 checkpoint 和指标。
- `src/mini_llm/generate.py`：文本生成入口，支持不同采样策略。
- `src/mini_llm/benchmark_kv_cache.py`：KV Cache 推理速度对比。
- `src/mini_llm/metrics.py`：loss、perplexity 等指标工具。
- `src/mini_llm/checkpoint.py`：模型保存和加载。

## 本地 Mac 开发

本地用于代码开发和快速验证，不建议在 Mac 上跑正式实验。

```bash
cd mini-llm-from-scratch
conda env create -f environment.yml
conda activate mini-llm
PYTHONPATH=src pytest -q
./scripts/run_smoke_test.sh
```

如果不使用 Conda，也可以继续用 venv：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

预期结果：

- `pytest` 输出全部测试通过。
- `run_smoke_test.sh` 能完成极小配置训练，并在 `results/smoke/` 下生成 checkpoint 和指标文件。

## GitHub 同步方式

推荐流程是：Mac 本地开发并提交，推送到 GitHub；CloudStudio 服务器从 GitHub 拉取代码后运行实验。

Mac 本地提交：

```bash
git add mini-llm-from-scratch docs/superpowers/plans .gitignore
git commit -m "feat: implement mini llm from scratch"
git push origin HEAD
```

CloudStudio 服务器拉取：

```bash
git clone <你的 GitHub 仓库地址>
cd <仓库目录>/mini-llm-from-scratch
```

如果服务器已经 clone 过：

```bash
cd <仓库目录>
git pull origin HEAD
cd mini-llm-from-scratch
```

## CloudStudio Linux/A800 环境

服务器上运行正式实验。代码和脚本均按 Linux shell 环境设计。

```bash
cd mini-llm-from-scratch
conda env create -f environment.yml
conda activate mini-llm
nvidia-smi
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
```

预期结果：

- `conda env create` 成功创建名为 `mini-llm` 的环境。
- `nvidia-smi` 能看到 A800 GPU。
- `cuda_available` 输出 `True`，并能打印 GPU 名称。

如果 `nvidia-smi` 正常但 `cuda_available` 是 `False`，通常是 PyTorch CUDA 轮子和服务器驱动不匹配。此时先保持 Conda 环境不变，只在 `mini-llm` 环境中按 PyTorch 官网给出的 Linux/CUDA 安装命令重装 `torch`。

## 数据集下载与预处理

项目保留 THUCNews/cnews 作为快速闭环数据，也新增了正式预训练数据管线。正式实验默认使用 HuggingFace `HuggingFaceFW/fineweb-edu` 的 `sample-10BT`，通过 `datasets` streaming 读取文本，训练 32k BPE tokenizer，并流式写入 `uint32` token 二进制文件。这个流程不会把几十亿 token 一次性读入内存。

正式 212M 主实验数据：

```bash
./scripts/prepare_fineweb_edu.sh
```

默认配置：

```text
DATASET_NAME=HuggingFaceFW/fineweb-edu
DATASET_CONFIG=sample-10BT
VOCAB_SIZE=32000
TARGET_TRAIN_TOKENS=3200000000
TARGET_VAL_TOKENS=20000000
```

预期结果：

- `data/tokenizer/fineweb_edu_bpe_32000.json`：32k BPE tokenizer。
- `data/processed/fineweb_edu_train.bin`：约 3.2B 训练 token。
- `data/processed/fineweb_edu_val.bin`：约 20M 验证 token。
- `data/processed/fineweb_edu_manifest.json`：本次数据准备元信息。

如果只想用本地小文本测试数据准备流程：

```bash
LOCAL_TEXT_FILE="tests/fixtures/tiny_corpus.txt" \
TARGET_TRAIN_TOKENS=2000 \
TARGET_VAL_TOKENS=200 \
TOKENIZER_TRAIN_DOCS=100 \
TOKENIZER_TRAIN_CHARS=10000 \
MIN_CHARS=5 \
FORCE_TOKENIZER=1 \
./scripts/prepare_fineweb_edu.sh
```

THUCNews/cnews 仍可用于 smoke test 和中文快速实验。该子集在 HuggingFace 上只有 `cnews.train.txt`；项目会在预处理阶段从 token 序列中自动切出训练集和验证集。

```bash
./scripts/download_thucnews.sh
./scripts/prepare_thucnews.sh
```

预期结果：

- 原始文本数据写入 `data/raw/thucnews/`。
- Tokenizer、训练 token、验证 token 写入 `data/processed/`。

如果希望下载完整 THUCNews 压缩包：

```bash
THUCNEWS_SOURCE=full ./scripts/download_thucnews.sh
./scripts/prepare_thucnews.sh
```

完整数据较大。若只想从完整数据中抽取一部分做快速实验：

```bash
THUCNEWS_SOURCE=full THUCNEWS_MAX_FILES=5000 ./scripts/download_thucnews.sh
./scripts/prepare_thucnews.sh
```

## 一键实验命令

建议先跑快速闭环，再跑正式实验。

```bash
./scripts/download_thucnews.sh
./scripts/prepare_thucnews.sh
./scripts/run_smoke_test.sh
```

在决定 large 模型规模和 batch size 前，先用随机 token 压测 A800 训练速度和显存：

```bash
./scripts/benchmark_train_speed.sh
```

如果想在 100M-220M 参数范围内找一个训练更充分、吞吐更快的主模型，运行：

```bash
./scripts/benchmark_midrange_train_speed.sh
```

脚本会依次测试 `large_250m`、`large_320m`、`large_370m`、`large_450m`、`large_560m`，输出文件：

```text
results/benchmarks/train_speed_large_250m.csv
results/benchmarks/train_speed_large_320m.csv
results/benchmarks/train_speed_large_370m.csv
results/benchmarks/train_speed_large_450m.csv
results/benchmarks/train_speed_large_560m.csv
```

中等模型脚本会测试 `mid_134m`、`mid_163m`、`mid_191m`、`mid_219m`、`mid_212m`，输出到同一目录下的 `train_speed_${preset}.csv`。

重点看 `tokens_per_second` 和 `peak_memory_reserved_mb`。该压测不依赖真实数据集，测的是目标模型在不同 batch size 下的训练吞吐和峰值显存。若某个 batch size OOM，脚本会记录 `status=oom` 并停止该模型后续更大的 batch，然后继续测试下一个模型。

212M 正式主实验：

```bash
./scripts/prepare_fineweb_edu.sh
./scripts/run_final_212m.sh
```

该配置为：

```text
模型：16 layers, 14 heads, 896 hidden, RoPE, ctx 1024
参数量：约 212M
batch_size：24
max_steps：130209
训练 token：约 3.2B
tokens/param：约 15.1
```

完整实验套件：

```bash
./scripts/run_experiment_suite.sh
```

默认会依次运行：

- `configs/final_212m_rope_ctx1024.yaml`：212M 主模型。
- `configs/ablation_212m_pos_abs.yaml`：212M + learned absolute position。
- `configs/ablation_212m_pos_none.yaml`：212M + no position encoding。
- `configs/ablation_212m_context_512.yaml`：212M + ctx 512。
- `configs/ablation_212m_context_1536.yaml`：212M + ctx 1536。
- `configs/scale_134m_rope_ctx1024.yaml`：134M 规模对比模型。

如果只想先试跑每个配置 20 步：

```bash
MAX_STEPS_OVERRIDE=20 ./scripts/run_experiment_suite.sh
```

训练后评估：

```bash
./scripts/run_posttrain_evals.sh
```

会复用主模型 checkpoint，完成 greedy/temperature/top-k/top-p 采样对比和 KV Cache 推理基准测试。采样策略和 KV Cache 实验不需要重新训练模型。

旧版快速实验脚本仍保留：

```bash
./scripts/run_train_tiny.sh
./scripts/run_train_small.sh
./scripts/run_ablation_context.sh
./scripts/run_ablation_position.sh
./scripts/run_sampling.sh
./scripts/run_kv_cache_benchmark.sh
```

各实验含义：

- `run_train_tiny.sh`：训练小模型，验证完整训练链路。
- `run_train_small.sh`：训练较大模型，用于和小模型对比规模收益。
- `run_ablation_context.sh`：对比 128、256、512 三档上下文长度。
- `run_ablation_position.sh`：对比无位置编码、绝对位置编码、RoPE。
- `run_sampling.sh`：对比不同采样策略生成效果。
- `run_kv_cache_benchmark.sh`：对比使用和不使用 KV Cache 的推理速度。

## 配置文件说明

- `configs/smoke.yaml`：本地和服务器快速冒烟测试配置。
- `configs/tiny.yaml`：小模型训练配置。
- `configs/small.yaml`：较大模型训练配置。
- `configs/final_212m_rope_ctx1024.yaml`：212M 正式主模型，3.2B train tokens。
- `configs/ablation_212m_pos_abs.yaml`：212M 绝对位置编码消融。
- `configs/ablation_212m_pos_none.yaml`：212M 无位置编码消融。
- `configs/ablation_212m_context_512.yaml`：212M 短上下文消融。
- `configs/ablation_212m_context_1536.yaml`：212M 长上下文消融。
- `configs/scale_134m_rope_ctx1024.yaml`：134M 规模对比模型。
- `configs/ablation_context_128.yaml`：上下文长度 128。
- `configs/ablation_context_256.yaml`：上下文长度 256。
- `configs/ablation_context_512.yaml`：上下文长度 512。
- `configs/ablation_pos_none.yaml`：不使用位置编码。
- `configs/ablation_pos_abs.yaml`：使用绝对位置编码。
- `configs/ablation_pos_rope.yaml`：使用 RoPE 位置编码。

## 是否需要重构

不需要完整重构。当前项目是配置驱动结构：

- 改模型大小、上下文长度、位置编码、batch size、训练步数：改 `configs/*.yaml`。
- 改数据来源、token 数、tokenizer 规模：改 `scripts/prepare_fineweb_edu.sh` 的环境变量，或直接调用 `python -m mini_llm.pretrain_data`。
- 改训练算法、checkpoint 策略、评估指标：才需要改 `src/mini_llm/train.py`。
- 改 attention、RoPE、KV Cache：才需要改 `src/mini_llm/model.py` 或 `src/mini_llm/rope.py`。

## 测试命令

开发或修改代码后运行：

```bash
conda activate mini-llm
PYTHONPATH=src pytest -q
bash -n scripts/*.sh
python -m compileall -q src tests
```

预期结果：

- pytest 全部通过。
- shell 脚本语法检查无输出或无报错。
- Python 编译检查无报错。

## 结果与 Git 管理

训练产生的数据、checkpoint、日志和实验输出默认不提交到 Git：

- `data/processed/`
- `results/`
- `checkpoints/`
- `.venv/`

建议提交到 Git 的内容：

- 源码：`src/mini_llm/`
- 配置：`configs/`
- 环境配置：`environment.yml`、`requirements.txt`
- 脚本：`scripts/`
- 测试：`tests/`
- 实验记录：`experiments/`
- 报告草稿：`report/`
- 少量经过筛选的图表或关键结果摘要

这样可以保证 GitHub 仓库保持轻量，服务器实验结果也不会把仓库撑大。
