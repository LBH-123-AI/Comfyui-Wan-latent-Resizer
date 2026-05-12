### English README

<div align="center">

# ComfyUI-Wan-latent-Resizer

**Learned Latent Resizer for Wan2.1 & Wan2.2**
Lightweight · High-fidelity · Unified Upscaling & Downscaling

</div>

---

### ✨ Overview

A dedicated ComfyUI node that replaces naive interpolation with a trained neural network to resize Wan video latents. A single, lightweight model handles both high-fidelity upscaling and downscaling for Wan2.1 and Wan2.2, supporting arbitrary scale factors.

### 📰 News

-   **[2026-05-12]** 🎉 **Wan2.2 Support Added!** A new dedicated model for Wan2.2 is now available, providing unified upscaling and downscaling.
-   **[2026-05-12]** ✨ **Unified Model Architecture**: Upscaling and downscaling capabilities are now merged into a single, more capable model file for each version.

### 📸 Comparison Results

**Comparison 1 (wan2.1_latent_upscaler_comparison_001.png):**

![Comparison 1](examples/wan2.1_latent_upscaler_comparison_001.png)

**Comparison 2 (wan2.1_latent_upscaler_comparison_002.png):**

![Comparison 2](examples/wan2.1_latent_upscaler_comparison_002.png)

**Comparison 3 (wan2.1_latent_upscaler_comparison_003.mp4):**

<video src="examples/wan2.1_latent_upscaler_comparison_003.mp4" controls width="100%"></video>

Left: Latent Interpolation Upscaling (blurry) | Right: Learned Latent Resizing (sharp, our method)

### 📁 Project Structure
```text
Comfyui-Wan-latent-Resizer/
├── examples/
├── models/
│   ├── wan2.1_latent_resizer_fp16.safetensors
│   └── wan2.2_latent_resizer_fp16.safetensors
├── nodes/
│   ├── __init__.py
│   ├── wan2_1_latent_Resizer_nodes.py
│   └── wan2_2_latent_Resizer_nodes.py
├── workflows/
├── __init__.py
└── README.md
```

### 🚀 Key Features

-   ✅ **Unified Resizing**: A single model performs both upscaling and downscaling, simplifying your workflow.
-   ✅ **Neural Precision**: Learned specifically for Wan latents, vastly outperforming traditional interpolation.
-   ✅ **Arbitrary Scale Factors**: Supports continuous scaling across a wide range (e.g., 1.25× – 10×).
-   ✅ **Broad Compatibility**: Seamlessly handles (B,C,T,H,W) video latents and (B,C,H,W) image latents.
-   ✅ **Minimal Overhead**: An ultra-lightweight architecture ensures fast inference with negligible memory cost.
-   ✅ **Plug-and-Play**: Works as a standard ComfyUI node without altering your existing workflow.

### 🧠 Training Data

-   3,000 high-resolution video clips
-   1,000 high-quality images
Diverse content ensures robust performance on a wide range of resizing tasks.

### 📦 Installation

1.  Clone the repository into ComfyUI's `custom_nodes` folder:
    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/LBH-123-AI/ComfyUI-Wan-latent-Resizer.git
    ```
2.  All required dependencies (torch, einops, safetensors) are already present in a standard ComfyUI environment.

### 🧩 Usage

Add the "Wan Latent Resizer" node from the menu, select the model version (Wan2.1 or Wan2.2), choose Upscale or Downscale mode, and set your target resolution or scale factor.

**Typical workflows:**
-   **Quick Preview**: `[Wan Latent] → [Wan Latent Resizer] → [Decode]`
-   **High-Quality (Recommended)**: `[Low-res Latent] → [Wan Latent Resizer] → [High-res Refinement] → [Decode]`
    This generates a low-res latent first for speed, upscales it, and then refines it for superior detail, drastically reducing VRAM usage and generation time.

### 🧪 Model Details

| Model | Parameters | Size | Purpose |
| :--- | :--- | :--- | :--- |
| wan2.1_latent_resizer_fp16.safetensors | 21.80M | 41.5 MB | Unified resizing for Wan2.1 |
| wan2.2_latent_resizer_fp16.safetensors | 51.14M | 97.5 MB | Unified resizing for Wan2.2 |

Architecture: A lightweight ResBlock-based network with scale/target-size conditioning. Input and output are 16-channel Wan latents.

### 🙏 Acknowledgments

This project is inspired by and builds upon ComfyUi_NNLatentUpscale (https://github.com/Ttl/ComfyUi_NNLatentUpscale). Special thanks to Ttl (https://github.com/Ttl) for the excellent work and open-source contribution.

---

### 中文版 README

<div align="center">

# ComfyUI-Wan-latent-Resizer

**Wan2.1 & Wan2.2 专用 Latent 智能缩放节点**
轻量 · 高保真 · 放大缩小一体化

</div>

---

### ✨ 简介

为 Wan 视频/图像生成流程提供的专用 ComfyUI 节点，用训练好的神经网络替代简单的插值算法来完成 latent 缩放。一个轻量级模型即可同时处理 Wan2.1 和 Wan2.2 的高保真放大与缩小任务，并支持任意缩放倍数。

### 📰 新闻动态

-   **[2026-05-12]** 🎉 **新增 Wan2.2 支持！** 现已提供全新的 Wan2.2 专用模型，支持放大与缩小一体化。
-   **[2026-05-12]** ✨ **统一模型架构**：每个版本的放大与缩小功能现已合并到一个更强大的模型文件中。

### 📸 对比效果

**对比示例 1（wan2.1_latent_upscaler_comparison_001.png）：**

![对比示例 1](examples/wan2.1_latent_upscaler_comparison_001.png)

**对比示例 2（wan2.1_latent_upscaler_comparison_002.png）：**

![对比示例 2](examples/wan2.1_latent_upscaler_comparison_002.png)

**对比示例 3（wan2.1_latent_upscaler_comparison_003.mp4）：**

<video src="examples/wan2.1_latent_upscaler_comparison_003.mp4" controls width="100%"></video>

左：Latent 插值放大（画面模糊） | 右：学习型 Latent 缩放（画面清晰，即本方法）

### 📁 项目结构
```text
Comfyui-Wan-latent-Resizer/
├── examples/
├── models/
│   ├── wan2.1_latent_resizer_fp16.safetensors
│   └── wan2.2_latent_resizer_fp16.safetensors
├── nodes/
│   ├── __init__.py
│   ├── wan2_1_latent_Resizer_nodes.py
│   └── wan2_2_latent_Resizer_nodes.py
├── workflows/
├── __init__.py
└── README.md
```

### 🚀 核心特性

-   ✅ **一体式缩放**：单模型同时支持放大与缩小，简化工作流。
-   ✅ **神经网络高保真**：专门针对 Wan latent 学习，效果远超传统插值。
-   ✅ **任意缩放倍数**：支持连续自定义倍数（如 1.25× – 10×）。
-   ✅ **广泛兼容**：无缝处理 (B,C,T,H,W) 视频和 (B,C,H,W) 图像 latent。
-   ✅ **极低开销**：超轻量架构，推理速度极快，资源占用可忽略不计。
-   ✅ **即插即用**：作为标准 ComfyUI 节点工作，不改变现有流程。

### 🧠 训练数据

-   3,000 段高清视频素材
-   1,000 张高质量图像素材
多样化的内容确保了模型在各类缩放任务上都有稳健表现。

### 📦 安装

1.  将仓库克隆到 ComfyUI 的 `custom_nodes` 文件夹：
    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/LBH-123-AI/ComfyUI-Wan-latent-Resizer.git
    ```
2.  所有必需的依赖（torch、einops、safetensors）在标准 ComfyUI 环境中已自带。

### 🧩 使用方法

从节点菜单中添加 "Wan Latent Resizer"，选择模型版本（Wan2.1 或 Wan2.2），选择放大或缩小模式，并设置目标分辨率或缩放倍数即可。

**典型流程示例：**
-   **快速预览**：`[Wan Latent] → [Wan Latent Resizer] → [解码]`
-   **高品质输出（推荐）**：`[低清 Latent] → [Wan Latent Resizer] → [高清重绘] → [解码]`
    先生成低分辨率 latent 以提高速度，放大后再进行高清细节修复。此流程在保证画质的同时，能显著降低显存占用和生成时间。

### 🧪 模型详情

| 模型 | 参数量 | 大小 | 用途 |
| :--- | :--- | :--- | :--- |
| wan2.1_latent_resizer_fp16.safetensors | 21.80M | 41.5 MB | Wan2.1 一体化智能缩放 |
| wan2.2_latent_resizer_fp16.safetensors | 51.14M | 97.5 MB | Wan2.2 一体化智能缩放 |

架构：基于残差块的轻量网络，带有缩放倍数/目标尺寸条件输入。输入和输出均为 16 通道 Wan latent。

### 🙏 致谢

本项目受 ComfyUi_NNLatentUpscale (https://github.com/Ttl/ComfyUi_NNLatentUpscale) 启发并基于其构建，特别感谢原作者 Ttl (https://github.com/Ttl) 的优秀工作和开源贡献。