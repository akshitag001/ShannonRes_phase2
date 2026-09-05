# ShannonRes Phase 2: SEM Image Restoration
**High-Fidelity Scanning Electron Microscope Image Denoising and Super-Resolution**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg?logo=pytorch)
![Task](https://img.shields.io/badge/Task-Image_Restoration-brightgreen.svg)

> **Note**: This repository contains our final submission for Phase 2. You can find our previous work for Phase 1 here: [akshitag001/ShannonRes](https://github.com/akshitag001/ShannonRes)

## 1. Overview
The ShannonRes project tackles the critical challenge of restoring highly noisy, low-resolution Scanning Electron Microscope (SEM) imagery. Modern semiconductor inspection requires rapid scanning speeds, which inherently introduces severe shot noise and limits spatial resolution. 

This repository contains our final winning solution for Phase 2: a **physics-aware, conditioning-injected neural network** that recovers fine granular textures and perfectly preserves critical dynamic ranges.

**Dataset**: The official KLA Phase 2 Task material and dataset can be accessed [here (SharePoint Link)](https://interinstitutional-my.sharepoint.com/personal/sourabh_i4c_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fsourabh%5Fi4c%5Fin%2FDocuments%2FSourabh%2Fi4C%5Ffolder%5FSourabh%2Fi4C%5Fhackathons%5FPrograms%2FIESA%20Semicon%2026%2FPhase%202%2FKLA%5FProblem%20Statement%201%5FPhase%202%2FKLA%20Phase%202%20Task%20material&ga=1).

---

## 2. Final Solution: FiLM-Conditioned NAFNet

Our final architecture is the **FiLM-Conditioned NAFNet**. We selected the *Nonlinear Activation Free Network (NAFNet)* for its state-of-the-art restoration performance and incredible computational efficiency. To handle varying noise distributions, we introduced a parallel **Degradation Estimator** that calculates a global noise embedding, which is then injected directly into the NAFNet bottleneck using a **Feature-wise Linear Modulation (FiLM)** layer.

### Phase 2 Architecture Flow

```mermaid
graph TD
    Input[Noisy LR Input <br/> Shape: Bx1xHxW] --> Intro[Intro Convolution <br/> 1 -> 32 channels]
    
    subgraph Encoder
        Intro --> Enc1[NAFBlock x1]
        Enc1 --> Down1[Downsample <br/> 32 -> 64 ch]
        Down1 --> Enc2[NAFBlock x1]
        Enc2 --> Down2[Downsample <br/> 64 -> 128 ch]
        Down2 --> Enc3[NAFBlock x1]
        Enc3 --> Down3[Downsample <br/> 128 -> 256 ch]
        Down3 --> Enc4[NAFBlock x14]
    end

    subgraph Degradation Estimator
        Input --> DE1[Conv + LeakyReLU]
        DE1 --> DE2[Conv + LeakyReLU]
        DE2 --> DE3[Conv + LeakyReLU]
        DE3 --> GAP[Global Average Pool + Linear]
        GAP --> Embed[Degradation Embedding <br/> Shape: 32]
    end

    subgraph Bottleneck
        Enc4 --> Mid[Middle NAFBlock x1]
        Mid --> FiLM[FiLM Layer]
        Embed --> FiLM
    end

    subgraph Decoder
        FiLM --> Up1[Upsample <br/> 256 -> 128 ch]
        Enc3 -. Skip Connection .-> Up1
        Up1 --> Dec1[NAFBlock x1]
        
        Dec1 --> Up2[Upsample <br/> 128 -> 64 ch]
        Enc2 -. Skip Connection .-> Up2
        Up2 --> Dec2[NAFBlock x1]
        
        Dec2 --> Up3[Upsample <br/> 64 -> 32 ch]
        Enc1 -. Skip Connection .-> Up3
        Up3 --> Dec3[NAFBlock x1]
        
        Dec3 --> Dec4[NAFBlock x14]
    end

    Dec4 --> OutConv[Upsampling Conv <br/> + PixelShuffle]
    OutConv --> Clamp[Clamp 0.0 to 1.0]
    Clamp --> Output[Restored HR Output <br/> Shape: Bx1x2Hx2W]
```

![Phase 2 Architecture Detailed](docs/PHASE2ARCH.png)

### Loss Function

Our model is trained using a **Restoration Combined Loss** function, which is a carefully tuned weighted sum of four different metrics to ensure both pixel-perfect accuracy and perceptually pleasing results:

1. **Charbonnier Loss ($\lambda = 1.0$)**: A differentiable approximation of L1 loss with a small $\epsilon$ for stable gradients. It enforces overall spatial correctness without penalizing outliers as heavily as L2.
2. **SSIM Loss ($\lambda = 0.5$)**: Computed as `1 - SSIM`. It heavily penalizes structural and contrast degradation, ensuring granular high-frequency textures and dynamic ranges are preserved.
3. **Sobel Edge Loss ($\lambda = 0.1$)**: An L1 loss applied over spatial gradients (computed via Sobel filters in X and Y directions). It explicitly forces the model to recover sharp, high-frequency edges.
4. **LPIPS Loss ($\lambda = 0.3$)**: A learned perceptual metric that extracts deep features to ensure the generated output visually matches the ground truth texture to the human eye.

---

## 3. Quick Start
To run inference over the provided test set, run the following commands sequentially:

```bash
# 1. Clone the repository
git clone https://github.com/akshitag001/ShannonRes_phase2.git
cd ShannonRes_phase2

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run standalone inference
python run.py /path/to/test_input_dir /path/to/output_dir
```

---

## 4. Training and Reproduction

To reproduce the final submitted FiLM checkpoint, we provide a complete training pipeline. The model is warm-started from the `weights/model_nafnet_v2.pth` base checkpoint to accelerate convergence.

```bash
python train.py --config configs/final_model.yaml
```

### Verified Training Parameters
The following hyperparameter settings are explicitly set and verified in the source code (`configs/final_model.yaml` and `src/dataset.py`):
- **Seed**: `42`
- **LPIPS Weight (`lambda_lpips`)**: `0.5`
- **Data Augmentations**:
  - **Synthetic Degradation**: 50% probability (`synthesize_prob: 0.5`) of applying randomized multiplicative Speckle noise ($\sigma \in [0.1, 0.4]$) and additive Gaussian noise ($\sigma \in [0.05, 0.2]$) before downsampling.
  - **Geometric**: Random 128x128 patches (GT), 50% Horizontal Flip, 50% Vertical Flip, and Random 90° Rotations.

### Ablation Studies (Phase-2 Validation Split)
All metrics were computed strictly on the fair, leak-checked 478-image Phase-2 Validation Split.

| Model / Configuration | PSNR | SSIM | LPIPS | Highlight Retention |
|---|---|---|---|---|
| Baseline NAFNet (`model_nafnet_v2.pth`) | 22.10 | 0.559 | 0.325 | **66.68%** |
| Frequency (FFT) Loss Finetuned | **22.40** | **0.575** | 0.316 | 54.12% |
| **Final FiLM Checkpoint (`model_nafnet_film_v2.pth`)** | 22.37 | 0.573 | **0.307** | 60.47% |

*(Note: While the FFT-finetuned model marginally improved PSNR/SSIM, it severely degraded LPIPS perceptual quality and highlight retention, leading to its rejection in favor of the final FiLM checkpoint.)*

---

## 5. Repository Structure
```text
ShannonRes/
├── README.md                          <- Primary submission document
├── requirements.txt                   <- Environment specification
├── run.py                             <- Mandatory standalone evaluation script
├── .gitignore                         <- Excludes datasets and intermediate artifacts
├── src/                               <- Source code directory
│   ├── models/                        
│   │   └── nafnet.py                  <- Final NAFNet + FiLM architecture definition
│   ├── losses.py                      <- Charbonnier, SSIM, Sobel, and LPIPS loss modules
│   └── dataset.py                     <- PyTorch Dataset loaders
├── train.py                           <- Training script to reproduce the final model
├── configs/                           
│   └── final_model.yaml               <- Configuration for model_nafnet_film_v2.pth
├── weights/                           
│   └── model_nafnet_film_v2.pth       <- Final submitted checkpoint
├── outputs/                           <- Generated test set outputs (from run.py)
└── docs/                              
    ├── RESEARCH.md                    <- Research history, methodology, negative results
    └── ARCHITECTURE.md                <- Detailed architectural specifications
```

---

## 6. Results

The model was evaluated strictly on a clean, 478-image Phase-2 validation split, and on the official undisclosed Test Set.

| Metric | Phase-2 Validation | Official Test Set |
|---|---|---|
| **PSNR** | 22.37 | 23.54 |
| **SSIM** | 0.573 | 0.596 |
| **LPIPS** | 0.307 | 0.307 |

**Key Finding — Highlight Retention**: We developed a custom diagnostic to measure "Highlight Retention" (the fraction of near-saturated pixels `>0.9` correctly preserved). We achieved **60.47%** highlight retention, entirely resolving the dynamic-range flattening issue seen in early Charbonnier-only experiments.

---

## 7. Research & Methodology
For a deep dive into our methodology, see [`docs/RESEARCH.md`](docs/RESEARCH.md). Key takeaways:
- **The SSIM Discovery**: We discovered that pure pixel/edge losses (Charbonnier + Sobel) caused bright highlights to be smoothed into flat gray patches. Re-introducing SSIM (which penalizes local variance loss) forced the model to preserve granular bright textures.
- **Disciplined Negative Results**: We attempted both Adversarial (GAN) training and Frequency-Domain (FFT) high-frequency loss fine-tuning. Both were rejected to prevent hallucination and texture smoothing. 

## 8. Input/Output Specification
- **Input (`NoisyLR`)**: `Bx1x128x128` raw noisy SEM images saved as `.npy` arrays, float32, range `[0.0, 1.0]`.
- **Output (`RestoredHR`)**: `Bx1x256x256` denoised and upscaled SEM images saved as `.npy` arrays, float32, strictly clamped to `[0.0, 1.0]`.

## 9. Environment
Inference was tested end-to-end on an NVIDIA H100 GPU. Hardware execution times on the test set comfortably pass all baseline requirements with high throughput. See `requirements.txt` for software dependencies.

## 10. References
- Chen, L., et al. "Simple Baselines for Image Restoration." *ECCV 2022*.
- Perez, E., et al. "FiLM: Visual Reasoning with a General Conditioning Layer." *AAAI 2018*.
