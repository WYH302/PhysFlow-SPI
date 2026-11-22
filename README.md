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

The framework combines three lightweight, physically grounded components:
1. **Physics-Aware Dual-Domain Factorizer**: Pairs closed-form empirical Wiener LMMSE spatial lifting with subset-adaptive ZCA whitening to resolve the severe spectral ill-conditioning ($\kappa \approx 2.38 \times 10^{8}$) of non-negative binary DMD masks, extracting continuous semantics ($\mathbf{C}_{\mathrm{sem}}$), identity ($\mathbf{C}_{\mathrm{id}}$), and 106-point canonical geometry ($\mathbf{C}_{\mathrm{geo}}$) across variable sampling rates without retraining per rate.
2. **VAE-Free Pixel-Space Flow Conditioning**: Injects identity and geometry embeddings into trailing padding slots of a frozen MiniT2I-B/16 multimodal flow transformer via RMSNorm-safe zero-initialized residual gates and rank-16 Q/K/V/O LoRA, allowing direct single-step measurement consistency in RGB pixel space.
3. **Blind Spectral Noise-Adaptive Kalman-Wiener FLOWER**: Replaces speckle-inducing Euclidean projections at inference with a covariance-coupled Kalman-Wiener proximal step whose regularization adapts blindly from trailing covariance eigenmodes.

### Key Benchmark Results (Single RTX 3090 24 GB)
- **Training Cost**: `1.39` GPU-hours total (`83.29` min across Stage 1A + 1B + E2E), `2.80 GB` peak VRAM, `42.73M` trainable parameters (`13.68%` of `312.26M`).
- **Inference Throughput**: `55.9 FPS` (`17.88 ms`, `5.55 GFLOPs`) for reconstruction-free 106-point landmark & 40-attribute inference; `2.25 FPS` (`0.44 s/img`) for 8-step and `1.15 FPS` (`0.87 s/img`) for 16-step $512 \times 512$ synthesis (`2.95x` faster than LSI at `2.57 s/img`).

| Method | CelebA-HQ $M=64$ AdaFace $\uparrow$ | NME106 $\downarrow$ | LPIPS $\downarrow$ | PSNR (dB) $\uparrow$ | RelRes $\downarrow$ | CelebA $M=256$ AdaFace $\uparrow$ | PSNR (dB) $\uparrow$ | Latency (s/img) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Tikhonov Back-Proj. | 0.0085 | 0.2595* | 0.8186 | 12.40 | 0.0069 | --- | --- | 0.0001 |
| Wiener-LMMSE | 0.0248 | 0.0939 | 0.7315 | 17.06 | 0.0052 | --- | --- | 0.0001 |
| Text Bridge (B0-Disc.) | 0.0088 | 0.8130 | 0.7434 | 9.80 | 0.3369 | 0.0171 | 10.44 | 0.74 |
| PnP-Flow (ICLR 2025) | 0.0285 | 0.3434 | 0.6788 | 12.66 | 0.0122 | --- | --- | 1.01 |
| ZS-FLOWER (ICLR 2026) | -0.0034 | 0.3751 | 0.6691 | 11.68 | 0.0134 | 0.0080 | 14.37 | 0.70 |
| LSI (CVPR 2025) | -0.0125 | 0.1388 | 0.7264 | 7.46 | 0.3742 | 0.1351 | 17.53 | 2.57 |
| **PhysFlow (Ours)** | **0.0300** | **0.0869** | **0.6329** | **17.29** | **0.0046** | **0.1761** | **20.48** | **0.87** |

\*Tikhonov back-projection enables face detection on only 1/45 portraits at $M=64$ due to binary speckle.

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
- `checkpoints/` (MiniT2I-B/16, FLAN-T5-Base, DINOv2-Small, InsightFace AntelopeV2 / Buffalo\_L, and AdaFace IR-101)

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
```

### Citation

```bibtex
@article{wu2026physflow,
  author  = {Wu, Yonghao},
  title   = {PhysFlow: Factorized Physics-Aware Pixel-Space Flow Matching for Extreme-Compression Single-Pixel Face Reconstruction},
  journal = {Journal of Visual Communication and Image Representation},
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

### 快速复现步骤
请参照上方英文说明完成 Python 3.12 + CUDA PyTorch 环境安装，按目录结构放置公开数据集与预训练权重后，依次运行 `scripts/smoke_modules.py`、`build_real_dataset_cache.py`、`run_phase1a_real.py`、`run_phase1b_oracle_real.py`、`run_phase2_e2e_physflow.py`、`run_phase3_downstream_and_paper.py`、`eval_lapa_wflw.py` 与 `run_journal_audit_extras.py` 即可完整复现论文全部结果。

### 许可证与联系
本项目基于 [MIT License](LICENSE) 开源。如有学术交流或复现问题，欢迎联系 `yonghaowu369@gmail.com`。
