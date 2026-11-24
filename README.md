# PhysFlow-SPI

**PhysFlow: Factorized Physics-Aware Pixel-Space Flow Matching for Extreme-Compression Single-Pixel Face Reconstruction**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-ee4c2c.svg)](https://pytorch.org/)

[English](#english) | [中文说明](#中文说明)

---

<p align="center">
  <img src="assets/teaser_m64.png" width="98%" alt="PhysFlow Extreme-Compression Single-Pixel Face Reconstruction at M=64 (SR=0.098%)" />
</p>

## English

### Overview
**PhysFlow** is an open-source PyTorch implementation for variable-rate single-pixel face reconstruction and reconstruction-free facial analysis under extreme sub-Nyquist binary sampling ($M \in \{16, 32, 64, 128, 256\}$ patterns over $256 \times 256$ optics, i.e., $0.02\%$--$0.39\%$ sampling rate).

<p align="center">
  <img src="assets/architecture.png" width="98%" alt="PhysFlow Architecture: Optical Sensing, Dual-Domain Factorizer, Slot-Residual Pixel Flow, and Blind Kalman-Wiener Solver" />
</p>

The framework combines three lightweight, physically grounded components:
1. **Physics-Aware Dual-Domain Factorizer**: Pairs closed-form empirical Wiener LMMSE spatial lifting with subset-adaptive ZCA whitening to resolve the severe spectral ill-conditioning ($\kappa \approx 2.38 \times 10^{8}$) of non-negative binary DMD masks, extracting continuous semantics ($\mathbf{C}_{\mathrm{sem}}$), identity ($\mathbf{C}_{\mathrm{id}}$), and 106-point canonical geometry ($\mathbf{C}_{\mathrm{geo}}$) across variable sampling rates without retraining per rate.
2. **VAE-Free Pixel-Space Flow Conditioning**: Injects identity and geometry embeddings into trailing padding slots of a frozen MiniT2I-B/16 multimodal flow transformer via RMSNorm-safe zero-initialized residual gates and rank-16 Q/K/V/O LoRA, allowing direct single-step measurement consistency in RGB pixel space.
3. **Blind Spectral Noise-Adaptive Kalman-Wiener FLOWER**: Replaces speckle-inducing Euclidean projections at inference with a covariance-coupled Kalman-Wiener proximal step whose regularization adapts blindly from trailing covariance eigenmodes.

### Key Benchmark Results (Single RTX 3090 24 GB)
- **Training Cost**: `1.39` GPU-hours total (`83.29` min across Stage 1A + 1B + E2E), `2.80 GB` peak VRAM, `42.73M` trainable parameters (`13.68%` of `312.26M`).
- **Inference Throughput**: `55.9 FPS` (`17.88 ms`, `5.55 GFLOPs`) for reconstruction-free 106-point landmark & 40-attribute inference; `2.25 FPS` (`0.44 s/img`) for 8-step and `1.15 FPS` (`0.87 s/img`) for 16-step $512 \times 512$ synthesis (`2.95x` faster than LSI at `2.57 s/img`).
- **Full-Cohort CelebA-HQ (`N = 479` Strictly Identity-Disjoint Portraits, Inception-v3 FID)**:

| Method | $M=64$ FID $\downarrow$ | $M=64$ AdaFace $\uparrow$ | $M=64$ NME106 $\downarrow$ | $M=64$ LPIPS $\downarrow$ | $M=64$ PSNR $\uparrow$ | $M=256$ FID $\downarrow$ | $M=256$ AdaFace $\uparrow$ | $M=256$ PSNR $\uparrow$ | $M=256$ RelRes $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| TV-FISTA (CS) | 373.91 | 0.0137 | ---* | 0.8561 | 13.74 | 349.62 | 0.0236 | 14.08 | 0.0100 |
| Wiener-LMMSE | 200.17 | 0.0295 | 0.0869 | 0.7276 | 17.30 | 175.28 | 0.1485 | 19.46 | 0.0034 |
| PnP-Flow (ICLR 2025) | 146.94 | 0.0051 | 0.4090 | 0.6899 | 12.60 | 147.65 | 0.0212 | 15.86 | 0.0101 |
| ZS-FLOWER (ICLR 2026) | 149.67 | 0.0037 | 0.4121 | 0.6606 | 11.75 | 181.94 | 0.0060 | 14.02 | 0.0101 |
| LSI (CVPR 2025) | 167.03 | -0.0067 | 0.1365 | 0.7253 | 7.70 | 119.29 | 0.1462 | 16.74 | 0.0382 |
| **PhysFlow (Ours)** | **113.15** | **0.0311** | **0.0805** | **0.6283** | **17.51** | **106.03** | **0.1641** | **19.62** | **0.0032** |

\*TV-FISTA yields 0/479 face detections at $M=64$ and $M=256$. At $M=256$, PhysFlow beats LSI on AdaFace by `+12.2%` ($p = 0.0005$), FID by `-13.26`, NME106 by `-20.1%` ($p < 10^{-4}$), and PSNR by `+2.87 dB` ($p < 10^{-4}$), and transfers to standard 2D sequency-ordered Walsh-Hadamard `{0,1}` DMD masks (`19.97 dB` PSNR, `0.1168` AdaFace at $M=256$) with **zero generator retraining**.

### Installation

```bash
# 0. Clone the repository
git clone https://github.com/WYH302/PhysFlow-SPI.git
cd PhysFlow-SPI

# 1. Create a clean Python 3.12 virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate

# 2. Install PyTorch with CUDA matching your GPU driver
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4. Verify GPU and module smoke tests
python scripts/check_gpu.py
python scripts/smoke_modules.py
```

### Data & Pretrained Assets Preparation
Place public benchmarks and third-party pretrained weights under the following local directories (excluded from git via `.gitignore`):
- `data/raw/{ffhq, celeba, celeba_hq, lapa, wflw}` (with `identity_CelebA.txt` and `list_attr_celeba.txt` under `celeba/`)
- `third_party/latent-imaging/paper_checkpoints/` (official LSI binary mask $\Phi$ and StyleGAN-XL baseline weights)
- `checkpoints/` (MiniT2I-B/16, FLAN-T5-Large, DINOv2-Small, InsightFace AntelopeV2 / Buffalo\_L, and AdaFace IR-101)

### Reproducing Experiments

```bash
# Step 0: Build aligned 512px identity-disjoint dataset cache and verify physical forward model
python scripts/build_real_dataset_cache.py
python scripts/assert_phi_official.py

# Step 1A: Train & evaluate dual-domain measurement factorizer (1,479 disjoint test faces)
python scripts/run_phase1a_real.py

# Step 1B: Oracle slot-residual conditioning verification (O0 -> O1 -> O2)
python scripts/run_phase1b_oracle_real.py

# Step 2: End-to-end PhysFlow training & multi-rate / 6-axis noise / Pareto benchmark
python scripts/run_phase2_e2e_physflow.py

# Step 3: Reconstruction-free downstream transfer, external LaPa/WFLW, PnP-Flow & 5-fold CV
python scripts/run_phase3_downstream_and_paper.py
python scripts/eval_lapa_wflw.py
python scripts/run_journal_audit_extras.py

# Step 4: Blind spectral ablation, hyperparameter sensitivity, Hadamard/Bernoulli DMD transfer & full-479 FID
python scripts/run_reviewer_shield_suite.py
python scripts/finish_module_e_full479.py
```

### Design Insights & Practical Lessons Learned
Building PhysFlow involved several instructive dead ends that shaped the final lightweight architecture; we share them candidly in case they help fellow researchers working on computational optics and flow matching:
1. **Why naive 1D SPI encoders stall near the mean face**: Non-negative binary DMD patterns share a large DC floor ($\kappa(\Sigma_y) \approx 2.38 \times 10^{8}$). Before adding subset-adaptive ZCA whitening and closed-form Wiener spatial lifting, a standard 1D MLP barely beat predicting the mean portrait (`NME = 0.0971` vs. `0.0975`). Equalizing the spectrum first immediately cut landmark error by `35.2%` with zero extra trainable weights.
2. **Why RMSNorm breaks naive zero-initialized adapters**: In MiniT2I-B/16 (MM-JiT), every text block applies RMSNorm to its 256-slot sequence. Appending small-norm tokens fails because $\mathrm{RMSNorm}(\alpha \mathbf{h}) = \mathrm{RMSNorm}(\mathbf{h})$ amplifies tiny noise to full magnitude at step 0. Injecting unit-RMS identity/geometry tokens strictly into unused trailing padding slots `[224:256]` with zero-initialized $\tanh(\gamma_l)$ residual gates preserved exact `0.0` step-0 deviation.
3. **Why mid-trajectory VGG perceptual losses and over-long fine-tuning hurt pixel flows**: Adding a VGG-16 perceptual loss on mid-$t$ one-step clean estimates (`t >= 0.45`) severely destabilized the straight-line velocity field (validation LPIPS degraded `0.5349 -> 0.6739`, SSIM `0.6574 -> 0.3586`). Likewise, extending Stage-2 fine-tuning `2.5x` from `600` to `1,500` steps gently overfit the 5k training split, lowering zero-shot LaPa identity (`0.1785 -> 0.1614`). Keeping exact velocity matching for `600` steps (`1.39` GPU-hours) proved the true generalization optimum.
4. **Why optical mask structure still matters at `0.1%–0.4%` compression**: Plugging our frozen generator onto standard 2D sequency-ordered Walsh-Hadamard `{0,1}` DMD masks recovers strong identity (`0.1168` AdaFace, `19.97 dB` PSNR), whereas unstructured random Bernoulli `{0,1}` masks spread energy across all 65,536 pixels, preserving pose/luminance (`0.0908` NME, `17.07 dB` PSNR) but limiting fine biometric SNR.

### Citation

```bibtex
@article{wu2026physflow,
  author  = {Wu, Yonghao},
  title   = {PhysFlow: Factorized Physics-Aware Pixel-Space Flow Matching for Extreme-Compression Single-Pixel Face Reconstruction},
  journal = {Optics Communications},
  year    = {2026},
  note    = {Manuscript submitted for review}
}
```

### Acknowledgements
We gratefully acknowledge the authors of MiniT2I, Latent Space Imaging (LSI), FLOWER, PnP-Flow, DINOv2, InsightFace, and AdaFace for releasing public models and evaluation tools that made this study possible.

---

## 中文说明

### 项目简介
**PhysFlow** 是面向极端亚奈奎斯特采样率（$256 \times 256$ 光学口径下仅采集 $M \in \{16, 32, 64, 128, 256\}$ 组二值散斑测量，对应 $0.02\%$--$0.39\%$ 采样率）的单像素人脸重建与免重建视觉分析开源 PyTorch 实现。

本框架包含三个核心物理协同模块：
1. **物理感知双域分解编码器**：通过零参数闭式 Wiener-LMMSE 空间提升与子集自适应 ZCA 白化，克服非负二值 DMD 掩膜导致的极端谱病态性（条件数 $\kappa \approx 2.38 \times 10^{8}$），在单一模型下实现可变采样率的连续语义 $\mathbf{C}_{\mathrm{sem}}$、身份 $\mathbf{C}_{\mathrm{id}}$ 与 106 点标准几何 $\mathbf{C}_{\mathrm{geo}}$ 分解表征。
2. **无 VAE 像素空间流匹配条件注入**：采用 RMSNorm 安全的零初始化槽位残差注意力与 Q/K/V/O 秩-16 LoRA 微调冻结的 MiniT2I-B/16 像素流骨干，无需经过非线性 VAE 解码即可直接在 RGB 像素空间施加单步物理测量一致性监督。
3. **盲谱噪声自适应 Kalman-Wiener FLOWER 求解器**：在推理阶段从测量协方差尾部特征模态盲估计未知传感器噪声方差，以协方差耦合的 Kalman-Wiener 像素近端投影替代传统欧氏投影，显著抑制二值掩膜散斑并提升六类未校准噪声下的鲁棒性。

### 算力成本与核心性能（单张 RTX 3090 24 GB）
- **总训练开销**：全三阶段仅需 **1.39 GPU 小时**（83.29 分钟），峰值显存 **2.80 GB**，可训练参数仅 **42.73M**（占总参数 312.26M 的 13.68%）。
- **推理吞吐**：免重建 106 点关键点与 40 属性分析达 **55.9 FPS**（17.88 ms / 5.55 GFLOPs）；8 步 / 16 步 $512 \times 512$ 图像重建分别达 **2.25 FPS**（0.44 s）与 **1.15 FPS**（0.87 s，较 CVPR 2025 LSI 快 **2.95 倍**）。
- **跨数据集泛化**：在 CelebA-HQ、CelebA、LaPa 以及 WFLW 六类困难子集（姿态、表情、光照、妆容、遮挡、模糊）上综合超越 LSI (CVPR 2025)、PnP-Flow (ICLR 2025) 与 Zero-Shot FLOWER (ICLR 2026)；在 69 位未见名人的严格跨身份 5 折交叉验证中，免重建关键点误差较 LSI 降低 28.9%（$p = 2.8 \times 10^{-5}$），40 属性 Macro-F1 提升 47.7%（$p = 3.8 \times 10^{-4}$）。

### 核心思考与踩坑记录
在研发与调优 PhysFlow 的过程中，我们曾走过几次非常有启发性的弯路，特此坦诚记录，希望能为做计算光学与流匹配逆问题的同行节省排雷时间：
1. **为什么直接用 1D 神经网络编码二值单像素读数容易塌缩**：非负二值 DMD 掩膜带有极强的公共直流（DC）分量，导致测量协方差条件数高达 $\kappa(\Sigma_y) \approx 2.38 \times 10^{8}$。在引入子集自适应 ZCA 白化与闭式 Wiener 空间提升之前，标准 1D MLP 的关键点误差几乎与直接输出平均脸无异（`0.0971` vs. `0.0975`）；仅靠零参数物理白化便立刻将误差降低 `35.2%`。
2. **RMSNorm 架构下“零初始化失效”陷阱**：MiniT2I-B/16 (MM-JiT) 在文本流使用 RMSNorm，由于 $\mathrm{RMSNorm}(\alpha \mathbf{h}) = \mathrm{RMSNorm}(\mathbf{h})$，直接拼接小幅值外部 token 会在第 0 步被放大并破坏预训练先验。将单位 RMS 的身份/几何向量仅以零初始化 $\tanh(\gamma_l)$ 残差门控注入原生 256 槽位的尾部空闲区 `[224:256]`，才真正实现了第 0 步 `0.0` 数值偏差的平滑微调。
3. **中段感知损失与过长微调的反效果**：在像素流匹配中段（`t >= 0.45`）直接对单步预测加 VGG-16 感知损失会破坏直线速度场（验证集 LPIPS 从 `0.5349` 恶化至 `0.6739`）；将二阶段微调步数盲目拉长 2.5 倍（`600 -> 1500` 步）也会导致对 5,000 张训练集轻微过拟合，使外部 LaPa 身份相似度从 `0.1785` 降至 `0.1614`。最终证明 `600` 步纯精确速度匹配（总耗时 `1.39` GPU 小时）才是泛化与算力的最优点。
4. **极端压缩下光学掩膜结构的物理边界**：在不重训生成器的前提下，迁移到标准 2D 序率排序 Walsh-Hadamard `{0,1}` DMD 掩膜依然能保持高身份保真（`M=256` 达 `19.97 dB` PSNR、`0.1168` AdaFace）；而完全无结构的随机 Bernoulli `{0,1}` 掩膜将能量均匀分散到 65,536 个像素，虽能准确恢复姿态与光照（`0.0908` NME、`17.07 dB` PSNR），但在极低采样率下的高频生物特征信噪比存在物理上限。

### 快速复现步骤
请参照上方英文说明完成 Python 3.12 + CUDA PyTorch 环境安装，按目录结构放置公开数据集与预训练权重后，依次运行 `scripts/smoke_modules.py`、`build_real_dataset_cache.py`、`run_phase1a_real.py`、`run_phase1b_oracle_real.py`、`run_phase2_e2e_physflow.py`、`run_phase3_downstream_and_paper.py`、`eval_lapa_wflw.py`、`run_journal_audit_extras.py`、`run_reviewer_shield_suite.py` 与 `finish_module_e_full479.py` 即可完整复现论文全部结果。

### 许可证与联系
本项目基于 [MIT License](LICENSE) 开源。如有学术交流或复现问题，欢迎联系 `yonghaowu369@gmail.com`。
