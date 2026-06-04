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
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src pytest -q
./scripts/run_smoke_test.sh
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
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
nvidia-smi
```

预期结果：

- `pip install` 成功安装 PyTorch、tokenizers、PyYAML、pytest 等依赖。
- `nvidia-smi` 能看到 A800 GPU。

## 数据集下载与预处理

数据集可以自动下载。默认下载较小的 THUCNews 派生中文 cnews 子集，更适合快速完成实验闭环。

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

正式实验：

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
- `configs/ablation_context_128.yaml`：上下文长度 128。
- `configs/ablation_context_256.yaml`：上下文长度 256。
- `configs/ablation_context_512.yaml`：上下文长度 512。
- `configs/ablation_pos_none.yaml`：不使用位置编码。
- `configs/ablation_pos_abs.yaml`：使用绝对位置编码。
- `configs/ablation_pos_rope.yaml`：使用 RoPE 位置编码。

## 测试命令

开发或修改代码后运行：

```bash
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
- 脚本：`scripts/`
- 测试：`tests/`
- 实验记录：`experiments/`
- 报告草稿：`report/`
- 少量经过筛选的图表或关键结果摘要

这样可以保证 GitHub 仓库保持轻量，服务器实验结果也不会把仓库撑大。
