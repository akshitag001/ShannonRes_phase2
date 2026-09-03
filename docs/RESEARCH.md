# Research & Methodology

This document details the iterative research, experimentation, and design decisions made to arrive at the final FiLM-Conditioned NAFNet model for the ShannonRes hackathon.

## Core Inspiration
Our final model architecture is built on two primary papers:
1. **NAFNet**: *Simple Baselines for Image Restoration* (Chen et al., ECCV 2022). NAFNet removes complex non-linear activations (like GELU/ReLU) in favor of SimpleGate (element-wise multiplication) and Simplified Channel Attention, proving that computational efficiency and state-of-the-art restoration performance can go hand-in-hand.
2. **FiLM**: *Feature-wise Linear Modulation* (Perez et al., AAAI 2018). FiLM layers allow a model to be conditioned on external metadata or global image statistics. We used this to inject global degradation embeddings directly into the bottleneck of the NAFNet architecture.

## The SSIM Highlight Discovery
During Phase 2 validation, we encountered a severe highlight-retention regression (highlight preservation plummeted from ~78% to 39%). Through disciplined ablation studies, we determined that this was **not** an architectural failure, but a loss-function side effect. 

We had temporarily removed the SSIM loss in favor of a pure Charbonnier + Sobel Edge loss. While Charbonnier mathematically penalizes absolute error and Sobel penalizes edge gradients, **SSIM penalizes local variance loss**. When a bright, textured highlight gets smoothed into a flat gray patch, SSIM strongly penalizes it. Re-introducing SSIM alongside Charbonnier and Sobel immediately restored highlight retention to competitive levels while maintaining sharp edges.

## Negative Results (What We Ruled Out)
We believe in reporting disciplined negative results. The following experiments were attempted but ultimately rejected for the final submission:

### 1. Adversarial (GAN) Training
We implemented a PatchGAN discriminator and trained the model using a combined Charbonnier + Adversarial loss to encourage sharper, more realistic textures.
- **Result**: The GAN loss introduced high-frequency structural artifacts (hallucinations) that were severely penalized by PSNR and LPIPS on the validation set. Given the strict fidelity requirements of SEM imagery, generative hallucinations were deemed too risky.

### 2. Frequency-Domain (FFT) Loss Fine-Tuning
To address dynamic-range compression in flat, texture-heavy regions, we fine-tuned the model using a high-frequency FFT-based loss.
- **Result**: The FFT loss narrowly improved PSNR (22.37 -> 22.40) but degraded LPIPS (0.307 -> 0.316) and significantly worsened bright highlight retention (60% -> 54%). The model learned to smooth over noise rather than reconstruct it. We halted this approach.

### 3. MambaIR (State Space Models)
We architected a Mamba-based image restoration model based on recent SSM literature.
- **Result**: Ruled out due to severe environment/hardware constraints (failing CUDA compilations for the `mamba_ssm` package on the provided hackathon instances). We pivoted fully to optimizing NAFNet.
