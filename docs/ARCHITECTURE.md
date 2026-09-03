# Architecture Specification

This document outlines the architecture of the **FiLM-Conditioned NAFNet**, which is the final model chosen for the ShannonRes submission.

## Overview
The architecture is based on the Nonlinear Activation Free Network (NAFNet), augmented with a Feature-wise Linear Modulation (FiLM) layer at the bottleneck to allow the model to condition its restoration process on the global degradation characteristics of the input image.

## Model Flow

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

## Key Components

1. **NAFBlock**: The core building block. It uses LayerNorm, Pointwise Convolutions, and Depthwise Convolutions. Instead of ReLU/GELU, it uses a `SimpleGate` (splitting feature maps into two halves and multiplying them element-wise) and a Simplified Channel Attention module.
2. **Degradation Estimator**: A small parallel CNN that looks at the raw input image and computes a 32-dimensional embedding representing the specific noise/blur profile of that image.
3. **FiLM Layer**: Situated at the deepest part of the network (the bottleneck). It applies a learned affine transformation (scale and shift) to the feature maps based on the degradation embedding. This allows the model to dynamically adjust its behavior for different noise distributions.
4. **Output Head**: Since this is a super-resolution task (scale=2), the final layer is a Convolution followed by a PixelShuffle operation to upscale the spatial resolution by 2x, clamped to `[0.0, 1.0]` to guarantee valid image ranges.
