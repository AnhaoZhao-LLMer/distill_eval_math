# Distillation Eval Bundle (vLLM)

这个目录是一个独立评测包，给另一台 8 卡服务器直接跑 16 个蒸馏 checkpoint 用。默认后端是 `vLLM`，默认会：

- 先从 ModelScope 下载 16 个模型
- 按 `8 GPU × 每卡串行 2 模型` 跑完全部评测
- 评测 4 个数据集：
  - `gsm8k`：每题 `n=3`
  - `math500`：每题 `n=3`
  - `amc23`：每题 `n=5`
  - `aime24`：每题 `n=5`
- 汇总每个模型在每个数据集上的：
  - `avg@n`
  - `pass@n`
  - `avg_len`

## 目录

- `data/`：4 个 parquet 测评集
- `configs/models.yaml`：16 模型 manifest
- `configs/datasets.yaml`：数据集和生成参数
- `src/`：下载、评测、汇总、SwanLab 上报代码
- `scripts/run_all.sh`：主入口

## 2 步跑起来

### 1. 准备环境

```bash
cd /path/to/distill_eval_vllm
bash scripts/prepare_env.sh
source .venv/bin/activate
```

如果你已经有合适的 Python / CUDA 环境，也可以直接：

```bash
pip install -r requirements.txt
```

### 2. 配好 token 后一键运行

```bash
export MODELSCOPE_API_TOKEN="your_modelscope_token"
export SWANLAB_API_KEY="your_swanlab_key"   # optional
bash scripts/run_all.sh
```

跑完以后主要看：

- `results/summary_per_model_dataset.csv`
- `results/summary_wide.csv`

## 常用覆盖

### 只做小样本 smoke test

```bash
MAX_SAMPLES_PER_DATASET=20 GPU_IDS="0 1" bash scripts/run_all.sh
```

### 如果模型已经下载过，跳过下载

```bash
SKIP_DOWNLOAD=1 bash scripts/run_all.sh
```

### 自定义模型缓存和结果目录

```bash
MODEL_ROOT=/path/to/models RESULTS_ROOT=/path/to/results bash scripts/run_all.sh
```

## 输出格式

每个模型每个数据集会生成：

- `results/<model_alias>/<dataset>/predictions.jsonl`
- `results/<model_alias>/<dataset>/metrics.json`

其中：

- `predictions.jsonl`：每题一行，含 `samples`、每个 sample 的 `token_len` 和 `is_correct`
- `metrics.json`：该模型在该数据集上的 `avg_at_n`、`pass_at_n`、`avg_len`

全局汇总：

- `summary_per_model_dataset.csv`
  - 一行 = 一个模型 × 一个数据集
- `summary_wide.csv`
  - 一行 = 一个模型
  - 展开为 `gsm8k_avg@3`、`gsm8k_pass@3`、`gsm8k_avg_len` 这类列

## 当前默认口径

- `enable_thinking=False`
- `temperature=0.6`
- `top_p=0.95`
- `top_k=20`
- `max_new_tokens=8192`
- `tensor_parallel_size=1`
- `gpu_memory_utilization=0.85`

这些默认值都在 [datasets.yaml](/C:/Users/Administrator/Desktop/analysis_kl/distill_eval_vllm/configs/datasets.yaml) 里，可以直接改。
