# DG-RTDiff

Direction-Guided Residual-Target Diffusion for image-domain GPR clutter suppression.

本工程对应 `D:\Clutter\DG-RTDiff_实验方案.md`。模型输入单通道 GPR PNG，直接预测稀疏目标响应图；它不处理原始雷达矩阵，也不进行介电常数或物理深度反演。

## 当前状态

- 代码用于后续复制到 RTX 5090D 工作站训练。
- 默认配置不会自动运行；只有显式执行 `train.py` 才会训练。
- `--dry-run` 只执行少量前向/反向，用于检查数据、模型和损失能否衔接。
- 固定组合数据仍标记为 `PENDING_USER_QC`，正式训练前必须完成人工质检。

## 目录

```text
DG_RTDiff/
├─ configs/                    # 正式配置与 CPU 冒烟配置
├─ dg_rtdiff/
│  ├─ data/                    # 固定组合、M 弱标注、R2/R1 数据集
│  ├─ diffusion/               # 余弦调度和 DDIM
│  ├─ models/                  # U-Net、方向条件器和基线
│  ├─ evaluation.py
│  ├─ losses.py
│  ├─ metrics.py
│  └─ utils.py
├─ tools/build_r1_windows.py   # 根据人工确认的 ROI 建立深圳滑窗清单
├─ tests/
├─ train.py
├─ evaluate.py
├─ infer.py
└─ smoke_test.py
```

## 工作站环境

推荐 Python 3.11 或 3.12。RTX 50 系列需要支持 Blackwell 的 CUDA 版 PyTorch。例如：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install -r requirements.txt
```

验证 GPU：

```powershell
.\.venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 数据路径

配置默认假定工程与 `data` 同级：

```text
Clutter/
├─ data/
│  ├─ fixed_composite_dataset_v2_gain18_lower/
│  ├─ M/
│  ├─ R1/
│  └─ R2/
└─ DG_RTDiff/
```

换工作站后只需修改 YAML 的 `data.*_root`，不要修改数据读取代码。

## 只检查可运行性

```powershell
python -m pytest -q
python smoke_test.py --config configs/smoke.yaml
python train.py --config configs/smoke.yaml --dry-run
```

这些命令不会执行正式 epoch。

## 正式训练顺序

数据质检通过并确认 GPU 环境后：

```powershell
python train.py --config configs/unet_baseline.yaml --seed 20260716
python train.py --config configs/vanilla_ddpm.yaml --seed 20260716
python train.py --config configs/dg_rtdiff_stage_a.yaml --seed 20260716
python train.py --config configs/dg_rtdiff_stage_b.yaml --resume runs/dg_rtdiff_stage_a_s20260716/checkpoints/best_ema.pt --seed 20260716
```

建议先只运行一个种子。确认 DG-RTDiff 相比 Vanilla DDPM 有收益后，再补三个种子和消融。

正式配置含 `max_hours` 硬超时。训练过程每个 epoch 更新 `runs/<实验>/heartbeat.json`，并写入 `logs/train.csv` 与 `checkpoints/latest.pt`，便于工作站监控和中断恢复。

## 深圳 R1 处理

R1 是宽幅软件截图，禁止直接压缩成 `320×320`。先人工确认统一雷达 ROI，再生成只记录坐标、不复制图像的窗口清单：

```powershell
python tools/build_r1_windows.py --input-root ../data/R1 --output manifests/r1_windows.csv --roi LEFT TOP RIGHT BOTTOM --window-size 320 --stride 160
```

检查清单和若干裁剪图后，把路径填入配置 `data.r1_window_manifest`。R1 只允许最终跨域测试。

## 传统基线与统计比较

传统方法只能从项目根目录以模块方式运行：

```powershell
python -m tools.run_traditional_baseline --config configs/dg_rtdiff_stage_a.yaml --method mean_scan --dataset composite --split val --output runs/traditional/mean_scan_val
python -m tools.run_traditional_baseline --config configs/dg_rtdiff_stage_a.yaml --method svd --rank 3 --dataset composite --split val --output runs/traditional/svd_r3_val
python -m tools.compare_methods --method-a runs/vanilla/per_image.csv --method-b runs/dg_rtdiff/per_image.csv --metric nmae_roi --lower-is-better --output runs/comparisons/dg_vs_vanilla_nmae.json
```

SVD 的秩和 RPCA 参数只能在 val 上选择，冻结后再运行 test。

## 输出解释

- `target_signed16.png`：有符号目标响应，32768 表示零；
- `target_preview.png`：仅用于显示的 8 位预览；
- `removed_preview.png`：输入减去目标响应的诊断图；
- 这些结果均是 PNG 图像域产物，不是恢复后的原始雷达信号。