# SmolVLM 本地空间推理评测

本仓库记录三个 4 位量化 SmolVLM 检查点在 VSR 零样本测试子集上的本地实验。六类空间关系各 10 道题，共 60 道题、59 张不同图像、180 条逐模型预测。

| 检查点 | 严格评分 | 接受 Yes/No | 单题中位耗时 |
|---|---:|---:|---:|
| 256M | 31/60（51.7%） | 31/60（51.7%） | 0.47 s |
| 500M | 14/60（23.3%） | 34/60（56.7%） | 0.52 s |
| 2B | 36/60（60.0%） | 36/60（60.0%） | 1.64 s |

严格规则只接受 True/False。接受 Yes/No 是在同一批输出上重算的敏感性分析。耗时在 Apple M5、16 GB 内存的 Mac 上测得。

- `run.py` 实现下载与核验样本、推理和评分。
- `sample.csv`、`sample.json` 记录固定的 60 道题、原图地址和校验值。
- `vsr-test.jsonl` 是上游测试标注的原始快照。
- `predictions.csv`、`summary.csv`、`paired.csv` 分别提供逐题预测、汇总和配对统计。
- 解压 `results.zip` 可得到正式实验目录 `results/`，含原始 JSONL 与模型元数据。随后用 `python run.py score` 重算。
- 重新推理请见 [复现实验指南](复现实验.md)。运行 `python run.py prepare` 按上游地址获取 59 张图像。模型权重由脚本按相应模型仓库获取。

VSR 标注来自 Liu、Emerson 和 Collier 的 [Visual Spatial Reasoning（TACL 2023）](https://aclanthology.org/2023.tacl-1.37/)，使用其 [官方零样本划分](https://huggingface.co/datasets/cambridgeltl/vsr_zeroshot)，标注许可为 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。上游标注快照按原样保存，采样清单从中抽样并新增行号、来源与校验信息。源图像由 COCO 提供，按原始地址获取。
