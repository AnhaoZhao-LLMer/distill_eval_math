# Distillation Eval Bundle (vLLM)

这个仓库用于在另一台 8 卡服务器上，批量评测 16 个蒸馏 checkpoint。默认后端是 `vLLM`，默认流程是：

- 从 ModelScope 下载模型
- 按 `8 GPU x 每卡串行 2 模型` 调度
- 跑 4 个数据集：
  - `gsm8k`：每题 `n=3`
  - `math500`：每题 `n=3`
  - `amc23`：每题 `n=5`
  - `aime24`：每题 `n=5`
- 汇总每个模型在每个数据集上的：
  - `avg@n`
  - `pass@n`
  - `avg_len`

## 目录结构

- `data/`：4 个 parquet 评测集
- `configs/models.yaml`：16 个模型 manifest
- `configs/datasets.yaml`：数据集与生成参数
- `src/`：下载、评测、汇总代码
- `scripts/run_all.sh`：主入口
- `scripts/prepare_env.sh`：可选的 conda 环境辅助脚本

## 最简使用

### 1. 手动创建 conda 环境

```bash
git clone https://github.com/AnhaoZhao-LLMer/distill_eval_math
cd distill_eval_math
conda create -n distill_eval_vllm python=3.12
conda activate distill_eval_vllm
pip install -r requirements.txt
```

### 2. 一键运行

```bash
bash scripts/run_all.sh
```

跑完以后主要看：

- `results/summary_per_model_dataset.csv`
- `results/summary_wide.csv`

## 输出格式

每个模型每个数据集会生成：

- `results/<model_alias>/<dataset>/predictions.jsonl`
- `results/<model_alias>/<dataset>/metrics.json`

其中：

- `predictions.jsonl`：每题一行，包含 `samples`、每个 sample 的 `token_len` 和 `is_correct`
- `metrics.json`：该模型在该数据集上的 `avg_at_n`、`pass_at_n`、`avg_len`

全局汇总：

- `summary_per_model_dataset.csv`
  - 一行 = 一个模型 x 一个数据集
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

这些默认值都在 `configs/datasets.yaml` 里，可以直接改。
