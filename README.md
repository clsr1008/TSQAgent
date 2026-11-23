# TSQAgent: Dedicated Agentic Reasoning for Time-Series Data Quality

<p align="center">
  <a href="#quick-start">Quick Start</a> |
  <a href="REPRODUCE.md">Reproduction</a> |
  <a href="#citation">Citation</a>
</p>

> 🔥 **2026.09** — TSQAgent was accepted to NeurIPS 2026 as a Spotlight paper.

## TSQAgent

TSQAgent is a multi-agent framework for pairwise time-series data-quality assessment. Given two time-series blocks and their dataset context, it identifies the quality dimensions relevant to the comparison, gathers quantitative evidence, and returns an interpretable judgment of which series has higher overall quality. These pairwise judgments can then be converted into data-quality scores for ranking samples and selecting higher-quality training data.

<p align="center">
  <img src="assets/framework.png" alt="TSQAgent framework" width="100%">
</p>

Its core capabilities are:

- **🧠 Reasoning-driven planning.** A GRPO-trained Perceiver focuses the analysis on the quality dimensions that are actually relevant to each pair.
- **📊 Tool-grounded inspection.** The Inspector combines LLM reasoning with statistical tools for missingness, noise, outliers, trends, frequency, amplitude, and pattern consistency.
- **⚖️ Interpretable decisions.** The Adjudicator returns a winner, confidence, and dimension-level evidence, with re-checking and re-planning when needed.
- **🎯 From judgments to better data.** Pairwise preferences train a lightweight rater for ranking samples and selecting higher-quality data for downstream tasks.

## TSQBench

TSQBench is a controlled synthetic benchmark for evaluating time-series quality reasoning. Each pair begins with a shared template and receives dimension-specific degradations, yielding interpretable quality differences across seven dimensions: missing values, noise level, rare patterns, trend, frequency, amplitude, and pattern consistency. It evaluates whether a model identifies the relevant dimensions and correctly compares quality within each dimension. See the [TSQBench code](benchmark/) for generation and evaluation scripts.

## Results at a Glance

We evaluate quality-aware data selection on 11 real-world datasets across long-term forecasting, short-term forecasting, and classification. Each method scores the training samples, selects the top 50% to train a downstream model (Linear, CNN, or PatchTST), and is evaluated on a held-out test set over 10 random seeds. The chart summarizes TSQAgent's ranking across 33 dataset-model settings: dark blue denotes Rank 1, light blue denotes Rank 2, and TSQAgent places in the top two for 31 settings.

<p align="center">
  <img src="assets/TSQAgent_Rank12.png" alt="TSQAgent ranking results across downstream settings" width="100%">
</p>

## Quick Start

```bash
conda create -n tsquality python=3.11 -y
conda activate tsquality
pip install -r requirements.txt

export OPENAI_API_KEY="<your_api_key>"
python main.py --model gpt-4o-mini --case noise
```

The command runs a built-in pairwise quality comparison and writes an HTML report with the reasoning trace. To use a local OpenAI-compatible server, train the Perceiver, annotate datasets, train a rater, or reproduce the paper's evaluations, follow the [reproduction guide](REPRODUCE.md).

## Using TSQAgent

```python
from config import Config, build_llm
from workflow import run_pipeline

cfg = Config(
    model="gpt-4o-mini",
    base_url="https://api.openai.com/v1",
    api_key="",  # Read OPENAI_API_KEY from the environment.
    enable_thinking=False,
    max_steps_per_dimension=6,
    max_recheck=2,
    max_replan=1,
)

result = run_pipeline(
    input_data={
        "dataset_description": "Industrial temperature sensor, 1-min sampling",
        "series_A": [...],
        "series_B": [...],
        "timestamps": [...],
    },
    llm=build_llm(cfg),
    config=cfg,
)

print(result["winner"])       # "A" | "B" | "tie"
print(result["confidence"])   # 0.0 - 1.0
print(result["explanation"])  # dimension-level evidence
```

## Reproducibility

The full research pipeline is documented in [REPRODUCE.md](REPRODUCE.md):

1. Generate synthetic TSQBench-style training pairs.
2. Fine-tune the Perceiver with quality-tailored GRPO.
3. Collect pairwise judgments over real-world datasets.
4. Train a dataset-specific or meta-learning quality rater.
5. Run data-selection experiments for forecasting and classification.

## Repository Map

| Directory | Purpose |
| --- | --- |
| `agents/` | Perceiver, Inspector, and Adjudicator workflow |
| `tools/` | Statistical tools used for evidence collection |
| `training/` | Synthetic data construction and Perceiver GRPO training |
| `annotation/` | Pairwise quality annotation workflows |
| `meta_learning_rater/` | Dataset-specific and cross-dataset quality raters |
| `evaluation/` | Downstream forecasting and classification experiments |

## Citation

If TSQAgent or TSQBench is useful for your research, please cite:

```bibtex
@inproceedings{wu2026tsqagent,
  title={TSQAgent: Rating Time Series Data Quality via Dedicated Agentic Reasoning},
  author={Wu, Shunyu and Li, Dan and Ye, Haozheng and Feng, Weibin and Lou, Jian and Zhang, Bo and Feng, Wenjie and Guo, Chenjuan and Ng, See-Kiong},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```
