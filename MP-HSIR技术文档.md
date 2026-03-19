# MP-HSIR 高光谱图像超分辨率任务技术文档

> 本文档基于代码逻辑，详细说明 MP-HSIR 框架如何执行高光谱图像超分辨率（及通用复原）任务。

---

## 目录

1. [项目概述](#1-项目概述)
2. [整体执行流程](#2-整体执行流程)
3. [数据准备与加载](#3-数据准备与加载)
4. [降质模拟（退化合成）](#4-降质模拟退化合成)
5. [网络模型架构](#5-网络模型架构)
   - 5.1 [OverlapPatchEmbed（图像嵌入）](#51-overlappatchembedimage-embedding)
   - 5.2 [Text_Prompt（文本提示编码器）](#52-text_prompt文本提示编码器)
   - 5.3 [TVSP（文本-视觉协同提示）](#53-tvsp文本视觉协同提示)
   - 5.4 [PGSSTB（提示引导空谱变换块）](#54-pgsstb提示引导空谱变换块)
   - 5.5 [编解码器主干网络（U-Net）](#55-编解码器主干网络u-net)
   - 5.6 [完整前向传播逻辑](#56-完整前向传播逻辑)
6. [训练流程](#6-训练流程)
7. [测试与评估流程](#7-测试与评估流程)
8. [超分辨率任务的具体实现](#8-超分辨率任务的具体实现)
9. [多提示机制详解](#9-多提示机制详解)
10. [关键设计决策](#10-关键设计决策)

---

## 1. 项目概述

**MP-HSIR**（Multi-Prompt Hyperspectral Image Restoration）是一种**多提示通用高光谱图像复原框架**，由 Zhehui Wu、Yong Chen、Naoto Yokoya、Wei He 提出（arXiv:2503.09131）。

### 核心问题

高光谱图像（HSI）在成像过程中会遭受多种未知退化：

| 退化类型 | 表现 |
|---------|------|
| 高斯噪声 | 每个像素的随机加性噪声 |
| 复合噪声 | 高斯噪声 + 条纹/死线/脉冲噪声 |
| 高斯模糊 | 空间细节丢失 |
| 运动模糊 | 方向性运动模糊 |
| **空间超分辨率** | 低分辨率 → 高分辨率重建 |
| 图像修复（Inpainting） | 随机像素缺失 |
| 波段缺失 | 整条光谱带丢失 |
| 去雾 | 大气散射引起的色彩失真 |
| 泊松噪声 | 光子计数噪声 |

传统方法针对单一任务设计，**MP-HSIR** 用一个统一模型处理上述所有任务。

### 关键创新

- **三类提示融合**：文本提示（CLIP语义）+ 视觉提示（可学习空间特征）+ 谱提示（低秩谱模式）
- **提示引导空谱变换块（PGSSTB）**：双分支谱注意力（全局 + 局部提示引导）
- **U-Net 多尺度架构**：三级编解码器，在不同尺度注入提示

---

## 2. 整体执行流程

```
┌─────────────────────────────────────────────────────────────┐
│                        训练阶段                              │
│                                                             │
│  原始HSI数据库(LMDB)                                         │
│       ↓                                                     │
│  随机采样64×64图像块                                          │
│       ↓                                                     │
│  随机选择退化类型(超分/去噪/去模糊等)并合成退化图              │
│       ↓                                                     │
│  数据增强（随机翻转/旋转）                                    │
│       ↓                                                     │
│  MP_HSIR_Net前向传播(退化图 + 任务ID)→ 复原图                │
│       ↓                                                     │
│  L1损失(复原图, 原始清晰图)                                  │
│       ↓                                                     │
│  AdamW优化 + 线性预热余弦退火学习率调度                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                        测试阶段                              │
│                                                             │
│  加载预训练权重                                              │
│       ↓                                                     │
│  指定退化模式(--mode)和测试数据目录                          │
│       ↓                                                     │
│  对每张测试图像合成对应退化                                   │
│       ↓                                                     │
│  MP_HSIR_Net推理(退化图 + 任务ID) → 复原图                  │
│       ↓                                                     │
│  计算 PSNR / SSIM 评价指标                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 数据准备与加载

### 3.1 数据集

**自然场景数据集**（`natural_scene` 模式）：
- ICVL（31波段）
- ARAD（31波段，前900张训练，后50张测试）

**遥感高光谱数据集**（`remote_sensing` 模式）：
- BerlinUrGrad（184波段）、Chikusei（128波段）、Eagle（50波段）
- Xiongan（150波段）、Houston（144波段）、PaviaU/C（103/102波段）、WDC（191波段）

### 3.2 LMDB 高效数据库

训练数据预先被裁剪为 **64×64 的图像块**，并以 LMDB 格式存储，以提高 I/O 效率：

```python
# utils/lmdb_patch.py
# 将原始 .mat 格式的 HSI 裁剪为 64×64 图像块并写入 LMDB
```

数据库中每条记录包含：
- **键**：图像块索引字符串
- **值**：`float32` 类型的 `(C, 64, 64)` numpy 数组（归一化到 `[0, 1]`）
- **元信息**：`meta_info.txt`，记录尺寸和来源文件名

### 3.3 LMDBDataset 类

```python
class LMDBDataset(data.Dataset):
    def __init__(self, args):
        # 打开 LMDB 数据库
        self.env = lmdb.open(args.db_path, max_readers=64, readonly=True)
        
        # 过滤指定数据集（如只使用遥感数据集）
        self.dataset_names = ['BerlinUrGrad', 'Chikusei', 'Eagle', ...]
        # 仅保留属于目标数据集的图像块索引
        self.valid_idxs = [idx for idx if source matches dataset_names]
    
    def __getitem__(self, index):
        # 从 LMDB 读取二进制数据
        data = txn.get(index_str.encode('ascii'))
        # 重建为 (C, H, W) numpy 数组
        X = np.frombuffer(data, dtype=np.float32).reshape(C, H, W)
        return X, source_file
```

### 3.4 ImageTransformDataset 类（训练集）

```python
class ImageTransformDataset(Dataset):
    def __getitem__(self, idx):
        img, name = self.dataset[idx]          # 读取干净图像块
        
        # 1. 随机选择退化类型
        de_id = random.randint(0, len(self.de_type) - 1)
        
        # 2. 应用退化（含超分辨率降采样/重采样）
        degrad_patch, _ = self.D.single_degrade(img.copy(), de_type, de_range)
        
        # 3. 数据增强（随机翻转/旋转）
        degrad_patch, clean_patch = random_augmentation(degrad_patch, img)
        
        # 4. 转为 Tensor
        return [name, de_id], degrad_patch_tensor, clean_patch_tensor, torch.tensor([de_id])
```

**退化类型与任务 ID 的对应关系：**

| 任务 ID | 退化类型 | 说明 |
|---------|---------|------|
| 0 | `gaussianN` | 高斯噪声（σ ∈ [30, 70]） |
| 1 | `complexN` | 复合噪声（高斯 + 条纹/死线/脉冲） |
| 2 | `blur` | 高斯模糊（核尺寸 7/11/15） |
| 3 | `sr` | **超分辨率**（下采样因子 2/4/8） |
| 4 | `inpaint` | 图像修复（掩码率 70%/80%/90%） |
| 5 | `haze` | 去雾（ω ∈ [0.5, 0.75, 1.0]） |
| 6 | `bandmiss` | 波段缺失（缺失率 10%/20%/30%） |

---

## 4. 降质模拟（退化合成）

核心类 `Degradation`（`utils/degradation_utils.py`）负责合成所有退化类型。

### 4.1 超分辨率退化（`sr`）

超分辨率任务的退化是通过**双三次插值下采样**再**上采样**实现的：

```python
def _bicubic_downsample(self, clean_patch, downsample_factor):
    """双三次下采样"""
    H, W = clean_patch.shape[1], clean_patch.shape[2]
    new_h, new_w = H // downsample_factor, W // downsample_factor
    
    # 使用 F.interpolate 进行双三次插值降采样
    ms = F.interpolate(clean_patch_tensor, size=(new_h, new_w),
                       mode='bicubic', align_corners=True)
    return ms  # shape: (C, H/factor, W/factor)

def _upsample(self, clean_patch, upsample_factor):
    """双三次上采样恢复原始尺寸"""
    lms = F.interpolate(clean_patch_tensor, size=(H*factor, W*factor),
                        mode='bicubic', align_corners=True)
    return lms  # shape: (C, H, W)
```

**超分辨率合成步骤：**

```
原始清晰 HSI (C, 64, 64)
    ↓ _bicubic_downsample(factor=2/4/8)
低分辨率 HSI (C, 32/16/8, 32/16/8)
    ↓ _upsample(factor=2/4/8)  ← 双三次上采样恢复尺寸
退化图 (C, 64, 64)  ← 模糊/低频的低分辨率图像
```

> 注：先下采样再上采样的目的是保持与原始图像相同的空间尺寸，以便网络直接预测残差。

### 4.2 其他退化类型

```python
# 高斯噪声
noise = np.random.randn(*clean_patch.shape) * (sigma / 255)
degraded = clean_patch + noise

# 复合噪声（随机组合）
degraded = gaussian_noise(clean_patch)
degraded = random.choice([deadline_noise, impulse_noise, stripe_noise])(degraded)

# 高斯模糊（每波段独立卷积）
kernel_2d = gaussian_kernel(kernel_size)
degraded = F.conv2d(input, kernel_2d, groups=C)  # 深度可分离卷积

# 去雾（大气散射模型）
transmission = exp((λ₀/λ_band)^γ × log(1 - ω × cirrus_band))
hazy_hsi = hsi × transmission + atmospheric_light × (1 - transmission)

# 波段缺失
lost_bands = np.random.choice(B, num_bands_to_loss, replace=False)
hsi[lost_bands] = 0

# 图像修复
mask = np.random.rand(C, H, W) > mask_ratio
degraded = hsi * mask
```

---

## 5. 网络模型架构

`net/MP_HSIR.py` 中的 `MP_HSIR_Net` 是整个框架的核心，采用**三级 U-Net 编解码架构**，并在每个尺度注入多类型提示。

### 5.1 OverlapPatchEmbed（图像嵌入）

```python
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=100, embed_dim=96, bias=False):
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        return self.proj(x)  # (B, 100, H, W) → (B, 96, H, W)
```

使用 **3×3 卷积**（步长=1，填充=1）将光谱通道数 `C` 映射到特征维度 `dim=96`，保持空间分辨率不变。

### 5.2 Text_Prompt（文本提示编码器）

```python
class Text_Prompt(nn.Module):
    def __init__(self, task_classes=7):
        # 7 个任务的文本描述（英文）
        self.task_text_prompts = [
            "A hyperspectral image corrupted by Gaussian noise.",
            "A hyperspectral image affected by complex noise patterns.",
            "A hyperspectral image degraded by Gaussian blur.",
            "A hyperspectral image with reduced spatial resolution.",  # 超分辨率任务
            "A hyperspectral image compressed to a certain ratio.",
            "A hyperspectral image degraded by atmospheric haze.",
            "A hyperspectral image with missing spectral bands.",
        ]
        # 使用 CLIP ViT-B/32 编码文本，得到 (7, 512) 的固定向量
        clip_model, _ = clip.load("ViT-B/32", device="cpu")
        self.clip_prompt = clip_model.encode_text(clip.tokenize(prompts))
    
    def forward(self, x, de_class):
        # de_class: 任务 ID 张量（如 [3] 表示超分辨率）
        # one-hot 编码任务 ID
        prompt_weights = F.one_hot(de_class, num_classes=7).float()  # (B, 7)
        
        # 加权平均 CLIP 嵌入
        clip_prompt = prompt_weights.unsqueeze(-1) * self.clip_prompt  # (B, 7, 512)
        clip_prompt = torch.mean(clip_prompt, dim=1)                   # (B, 512)
        
        return clip_prompt, prompt_weights
```

**逻辑说明**：根据输入的任务 ID，使用 one-hot 权重从预先编码的 CLIP 文本向量中**选取对应任务的语义嵌入**（512维），作为全局语义提示。

### 5.3 TVSP（文本-视觉协同提示）

`TVSP`（Text-Visual Synergistic Prompt）将文本语义提示与可学习的视觉空间特征融合：

```python
class TVSP(nn.Module):
    def __init__(self, task_classes=7, prompt_size=64, prompt_dim=96):
        # 可学习的任务文本向量（每个任务一个，在微调中优化）
        self.text_prompt_learnable = nn.Parameter(
            torch.randn(1, task_classes, prompt_dim, 1, 1))  # (1, 7, 96, 1, 1)
        
        # 可学习的视觉空间提示
        self.visual_prompt = nn.Parameter(
            torch.randn(1, prompt_dim, prompt_size, prompt_size))  # (1, 96, 64, 64)
        
        # 交叉注意力融合模块
        self.cross_transformer = CrossTransformer(dim=prompt_dim, num_heads=2)
    
    def forward(self, x, clip_prompt, prompt_weights):
        # 1. 用任务权重选择并融合可学习文本提示
        text_prompt = prompt_weights.view(B, 7, 1, 1, 1) * self.text_prompt_learnable  # (B, 7, 96, 1, 1)
        text_prompt = torch.mean(text_prompt, dim=1)  # (B, 96, 1, 1)
        
        # 2. 乘以 CLIP 语义权重，将预训练 CLIP 知识注入
        text_prompt = text_prompt * clip_prompt.unsqueeze(-1).unsqueeze(-1)  # (B, 96, 1, 1)
        
        # 3. 插值扩展到空间尺寸
        text_prompt = F.interpolate(text_prompt, size=(prompt_size, prompt_size))  # (B, 96, 64, 64)
        
        # 4. 交叉注意力：文本提示(Q) ↔ 视觉提示(K/V) 融合
        prompts = self.cross_transformer(text_prompt, self.visual_prompt.repeat(B, 1, 1, 1))
        
        # 5. 插值到特征图尺寸并投影
        output_prompt = F.interpolate(prompts, (H, W), mode='bilinear')
        return self.conv_last(output_prompt)  # (B, 96, H, W)
```

### 5.4 PGSSTB（提示引导空谱变换块）

`PGSSTB`（Prompt-Guided Spatial-Spectral Transformer Block）是网络的**基本处理单元**，每个块同时处理空间和谱维度的注意力：

```
输入特征 x (B, C, H, W)
    ↓ 转换为序列格式 (B, H×W, C)
    ↓ LayerNorm
    ↓ 划分为局部窗口（窗口大小 = 8）
    ↓
   ┌─────────────────────────────────────┐
   │     空间注意力（局部窗口 MHSA）       │
   │  + 相对位置偏置                       │
   │  + 交替使用移位窗口（SW-MSA）         │
   └───────────────┬─────────────────────┘
                   │ (B, nW, win², C)
       ┌───────────┴────────────┐
       │                        │
  局部谱注意力               全局谱注意力
  PG_Spectral_Attention     Spectral_Attention
  （提示引导低秩）            （全局通道注意力）
       │                        │
       └───────────┬────────────┘
                   ↓ 求和融合
    ┌──────────────────────────┐
    │   残差连接 + GatedMLP     │
    └──────────────────────────┘
    输出特征 (B, C, H, W)
```

#### PG_Spectral_Attention（提示引导局部谱注意力）

```python
class PG_Spectral_Attention(nn.Module):
    def __init__(self, dim, compress_ratio=8, prompt_len=128):
        # 压缩维度：96 → 12（compress_ratio=8）
        self.linear_down = nn.Linear(dim, dim // compress_ratio)
        self.linear_up = nn.Linear(dim // compress_ratio, dim)
        
        # 谱提示库：128个提示 × 压缩后维度（12）
        self.prompt_param = nn.Parameter(torch.rand(1, 1, 128, dim // compress_ratio))
        
        # 用于从输入特征生成提示权重
        self.linear_prompt = nn.Linear(dim, 128)
    
    def forward(self, x_kv):  # (B, win², C)
        shortcut = x_kv
        
        # 1. 沿空间维度平均池化得到谱描述符
        x_kv = x_kv.mean(dim=1, keepdim=True)  # (B, 1, C)
        
        # 2. 用谱描述符生成提示权重（从128个谱提示中加权选择）
        prompt_weights = F.softmax(self.linear_prompt(x_kv), dim=-1)  # (B, 1, 128)
        
        # 3. 压缩谱描述符
        x_kv = self.linear_down(x_kv)  # (B, 1, C//8)
        
        # 4. 加权混合谱提示（软检索）
        spectral_prompt = prompt_weights.unsqueeze(-1) * self.prompt_param  # (B, 1, 128, C//8)
        spectral_prompt = torch.sum(spectral_prompt, dim=2)                 # (B, 1, C//8)
        
        # 5. Q来自谱提示，K/V来自数据特征（交叉注意力）
        q = self.q(spectral_prompt)        # (B, 1, C//8)
        k, v = self.kv(x_kv).chunk(2, -1) # (B, 1, C//8) each
        
        attn = softmax(q.T @ k / sqrt(C//8))  # (B, C//8, 1)
        out = (attn @ v.T).linear_up()         # (B, 1, C)
        
        # 6. 逐元素相乘（门控残差）
        return out * shortcut  # (B, win², C)
```

#### GatedMlp（门控前馈网络）

```python
class GatedMlp(nn.Module):
    def forward(self, x):
        x_fc1, x_gate = self.fc1(x).chunk(2, dim=-1)  # 分成两半
        x = x_fc1 * self.act(x_gate)                   # 门控激活（类似 SwiGLU）
        return self.fc2(x)
```

### 5.5 编解码器主干网络（U-Net）

`MP_HSIR_Net` 采用三级对称 U-Net 结构：

```
输入 HSI (B, 100, 64, 64)
    ↓ OverlapPatchEmbed
(B, 96, 64, 64)  ← 编码器第1级
    ↓ BaseBlock(depth=2, heads=2)  [2个PGSSTB块]
(B, 96, 64, 64)  → 保存跳跃连接 enc1
    ↓ Downsample [3×3Conv + PixelUnshuffle(2)]
(B, 192, 32, 32) ← 编码器第2级
    ↓ BaseBlock(depth=4, heads=4)  [4个PGSSTB块]
(B, 192, 32, 32) → 保存跳跃连接 enc2
    ↓ Downsample
(B, 384, 16, 16) ← 瓶颈层（Latent）
    ↓ BaseBlock(depth=6, heads=8)  [6个PGSSTB块]
(B, 384, 16, 16)
    ↓ Upsample [3×3Conv + PixelShuffle(2)]
(B, 192, 32, 32) ← 解码器第2级
    ↑ + enc2（经TVSP提示增强后）
    ↓ concat → reduce_chan(1×1 Conv)
(B, 192, 32, 32)
    ↓ BaseBlock(depth=4, heads=4)
(B, 192, 32, 32)
    ↓ Upsample
(B, 96, 64, 64)  ← 解码器第1级
    ↑ + enc1（经TVSP提示增强后）
    ↓ concat
(B, 192, 64, 64)
    ↓ BaseBlock(depth=2, heads=2)
(B, 192, 64, 64)
    ↓ Refinement BaseBlock(depth=4, heads=2)
(B, 192, 64, 64)
    ↓ 3×3 Conv 投影
(B, 100, 64, 64)
    ↓ + inp_img（全局残差连接）
输出 HSI (B, 100, 64, 64)
```

**下采样/上采样模块：**

```python
class Downsample(nn.Module):
    def __init__(self, n_feat):
        # 3×3 Conv 先减半通道数，再用 PixelUnshuffle(2) 空间→通道
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat//2, kernel_size=3, padding=1),
            nn.PixelUnshuffle(2)  # (B, C//2, H, W) → (B, C*2, H/2, W/2)
        )

class Upsample(nn.Module):
    def __init__(self, n_feat):
        # 3×3 Conv 先翻倍通道数，再用 PixelShuffle(2) 通道→空间
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat*2, kernel_size=3, padding=1),
            nn.PixelShuffle(2)    # (B, C*2, H, W) → (B, C//2, H*2, W*2)
        )
```

### 5.6 完整前向传播逻辑

```python
def forward(self, inp_img, task_id):
    B, C, H, W = inp_img.shape  # e.g. (8, 100, 64, 64)
    
    # ① 文本提示编码
    text_prompt, prompt_weights = self.text_prompt(inp_img, task_id)
    # text_prompt: (B, 512)  prompt_weights: (B, 7)
    
    # ② 图像嵌入
    inp_enc_level1 = self.patch_embed(inp_img)        # (B, 96, 64, 64)
    
    # ③ 编码器
    out_enc_level1 = self.encoder_level1(inp_enc_level1)  # (B, 96, 64, 64)
    inp_enc_level2 = self.down1_2(out_enc_level1)          # (B, 192, 32, 32)
    out_enc_level2 = self.encoder_level2(inp_enc_level2)   # (B, 192, 32, 32)
    inp_enc_level3 = self.down2_3(out_enc_level2)          # (B, 384, 16, 16)
    
    # ④ 瓶颈层
    latent = self.latent(inp_enc_level3)               # (B, 384, 16, 16)
    
    # ⑤ 解码器（含提示融合）
    inp_dec_level2 = self.up3_2(latent)                # (B, 192, 32, 32)
    
    # 在 32×32 尺度注入提示
    prompt2 = self.prompt2(out_enc_level2, text_prompt, prompt_weights)
    out_enc_level2 = self.fusion2(out_enc_level2, prompt2)  # PromptFusion
    
    inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)  # (B, 384, 32, 32)
    inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)          # (B, 192, 32, 32)
    out_dec_level2 = self.decoder_level2(inp_dec_level2)
    
    inp_dec_level1 = self.up2_1(out_dec_level2)        # (B, 96, 64, 64)
    
    # 在 64×64 尺度注入提示
    prompt1 = self.prompt1(out_enc_level1, text_prompt, prompt_weights)
    out_enc_level1 = self.fusion1(out_enc_level1, prompt1)  # PromptFusion
    
    inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)  # (B, 192, 64, 64)
    
    out_dec_level1 = self.decoder_level1(inp_dec_level1)
    out_dec_level1 = self.refinement(out_dec_level1)   # 细化块（4个PGSSTB）
    
    # ⑥ 输出投影 + 全局残差连接
    out_dec_level1 = self.output(out_dec_level1) + inp_img  # (B, 100, 64, 64)
    
    return out_dec_level1
```

---

## 6. 训练流程

### 6.1 训练脚本（`train.py`）

使用 **PyTorch Lightning** 管理训练循环：

```python
class PromptIRModel(pl.LightningModule):
    def __init__(self, args):
        self.net = MP_HSIR_Net(in_channel=100, out_channel=100, dim=96, task_classes=7)
        self.loss_fn = nn.L1Loss()
    
    def training_step(self, batch, batch_idx):
        [clean_name, de_type], degrad_patch, clean_patch, prompt = batch
        
        # 前向传播
        restored = self.net(degrad_patch, prompt)
        restored = torch.clamp(restored, 0, 1)   # 截断到合法范围
        
        # L1 损失（均绝对误差）
        loss = self.loss_fn(restored, clean_patch)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.args.lr)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=int(0.1 * self.args.epochs),  # 10% 线性预热
            max_epochs=self.args.epochs,
            eta_min=1e-6
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
```

### 6.2 学习率调度（`utils/schedulers.py`）

```
学习率
  ↑
lr │          ╭─────────────╮
   │         /               \
   │        /                 \
   │       /                   \________
   │      /   线性预热  →  余弦退火
   │─────/
 0 └──────────────────────────────→ epoch
     0   10%      100% (epochs)
```

- **线性预热阶段**（前 10% epoch）：从 0 线性增加到 `base_lr`
- **余弦退火阶段**（后 90% epoch）：从 `base_lr` 余弦衰减到 `eta_min=1e-6`

### 6.3 训练参数

| 参数 | 自然场景 | 遥感 |
|------|---------|------|
| 训练 epoch | 100 | 300 |
| 学习率 | 2e-4 | 1e-4 |
| 批大小 | 32 | 32 |
| 优化器 | AdamW | AdamW |
| 精度 | 混合精度（fp16） | 混合精度（fp16） |
| 输入波段 | 31 | 100 |
| 隐藏维度 | 64 | 96 |
| 任务类别数 | 6 | 7 |

### 6.4 训练命令

```bash
# 遥感高光谱（7类退化）
python train.py \
    --epochs 300 \
    --lr 1e-4 \
    --data_type remote_sensing \
    --db_path /data/Train/Remote_sensing_minmax_patch_64.db

# 自然场景高光谱（6类退化）
python train.py \
    --epochs 100 \
    --lr 2e-4 \
    --data_type natural_scene \
    --db_path /data/Train/Natural_scene_minmax_patch_64.db
```

---

## 7. 测试与评估流程

### 7.1 测试模式说明（`test.py`）

`--mode` 参数指定测试的退化任务：

| `--mode` | 退化任务 | 提示 ID |
|----------|---------|---------|
| 0 | 高斯去噪（σ=10/30/50/70） | 0 |
| 1 | 高斯去噪（Non-IID） | 0 |
| 2 | 条纹/死线/脉冲噪声 | 1 |
| 3 | 高斯去模糊 | 2 |
| 4 | 复合去噪 | 1 |
| 5 | 运动去模糊 | 2 |
| 6 | 泊松去噪 | 0 |
| 7 | **超分辨率（2×/4×/8×）** | 3 |
| 8 | 图像修复 | 4 |
| 9 | 去雾 | 5 |

### 7.2 测试流程

```python
def test_super_resolution(net, dataset, sr_factor=4, device=None):
    testloader = DataLoader(dataset, batch_size=1)
    psnr = AverageMeter()
    ssim = AverageMeter()
    
    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch = degrad_patch.to(device)   # 低分辨率图像（已上采样至原尺寸）
            clean_patch  = clean_patch.to(device)    # 原始高分辨率图像
            
            prompt = torch.tensor([3]).to(device)    # task_id=3 代表超分辨率
            
            # 模型推理
            restored = net(degrad_patch, prompt)
            restored = torch.clamp(restored, 0, 1)
            
            # 计算逐波段平均 PSNR/SSIM
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
    
    print(f"[SR×{sr_factor}] PSNR: {psnr.avg:.2f} dB, SSIM: {ssim.avg:.4f}")
```

### 7.3 评价指标

```python
def compute_psnr_ssim(restored, clean):
    """对每个波段分别计算 PSNR 和 SSIM，最后取平均"""
    C = clean.shape[-3]
    for batch_idx in range(B):
        for band in range(C):
            x = restored[batch_idx, band]  # 单波段灰度图
            y = clean[batch_idx, band]
            
            # PSNR = 20 × log10(1 / sqrt(MSE))
            psnr += peak_signal_noise_ratio(x, y, data_range=1)
            
            # SSIM：结构相似性指标
            ssim += structural_similarity(x, y, data_range=1)
    
    return psnr / (B * C), ssim / (B * C), B
```

### 7.4 测试命令

```bash
# 超分辨率测试（遥感）
python test.py \
    --mode 7 \
    --test_dir /data/Test/Chikusei \
    --ckpt_path /ckpt/Remote_sensing.ckpt

# 高斯去噪测试（自然场景）
python test.py \
    --mode 0 \
    --test_dir /data/Test/ICVL \
    --ckpt_path /ckpt/Natural_scene.ckpt

# 去雾测试（遥感）
python test.py \
    --mode 9 \
    --test_dir /data/Test/Eagle \
    --ckpt_path /ckpt/Remote_sensing.ckpt
```

---

## 8. 超分辨率任务的具体实现

本节重点描述**超分辨率（SR）任务**在 MP-HSIR 中的完整处理逻辑。

### 8.1 SR 任务流程图

```
┌──────────────────────────────────────────────────────────────────┐
│                    超分辨率任务（以4×为例）                        │
│                                                                  │
│  原始HSI:  (100, 64, 64)   ← 存储于LMDB，值域[0,1]              │
│      ↓                                                           │
│  双三次下采样 (factor=4):  (100, 16, 16)                          │
│      ↓                                                           │
│  双三次上采样 (factor=4):  (100, 64, 64)   ← 低频模糊版           │
│      ↓                                                           │
│  数据增强（翻转/旋转）                                            │
│      ↓                                                           │
│  task_id = 3（超分辨率类别）                                      │
│      ↓                                                           │
│  Text_Prompt: 取"reduced spatial resolution"的CLIP嵌入 (512-d)  │
│      ↓                                                           │
│  MP_HSIR_Net前向传播：                                            │
│    ① OverlapPatchEmbed: (100,64,64)→(96,64,64)                  │
│    ② 编码器（3级）: 提取多尺度特征 + 空谱注意力                  │
│    ③ 瓶颈层: 6个PGSSTB捕获全局上下文                             │
│    ④ TVSP提示生成: 文本×视觉提示融合→任务引导空间提示            │
│    ⑤ PromptFusion: 将SR任务提示注入跳跃连接                      │
│    ⑥ 解码器（3级）: 逐步恢复空间分辨率和谱细节                   │
│    ⑦ 全局残差: output = net_output + 低分辨率输入                │
│      ↓                                                           │
│  复原HSI (100, 64, 64)   ← 高频细节已恢复                        │
│      ↓                                                           │
│  评估: PSNR / SSIM（与原始64×64图像对比）                        │
└──────────────────────────────────────────────────────────────────┘
```

### 8.2 SR 退化合成代码路径

```python
# utils/dataset_utils.py
def __getitem__(self, idx):
    img, name = self.dataset[idx]  # (100, 64, 64) 清晰图
    
    # 对于 de_type='sr'：
    degrad_patch, _ = self.D.single_degrade(img.copy(), 'sr', [(2, 4, 8)])
    # 等价于：
    #   factor = random.choice([2, 4, 8])
    #   low_res = bicubic_downsample(img, factor)      # (100, 32/16/8, 32/16/8)
    #   degrad  = bicubic_upsample(low_res, factor)    # (100, 64, 64) 模糊版

# utils/degradation_utils.py
def _degrade_by_type(self, clean_patch, degrade_type, degrade_range):
    if degrade_type == 'sr':
        factor = random.choice(degrade_range)
        self.downsample_factor = factor
        degraded = self._bicubic_downsample(clean_patch, factor)  # 下采样
        return degraded, factor  # 注意：此时尺寸已缩小

def single_degrade(self, clean_patch, degrade_type, degrade_range, name=None):
    degrad_patch, _ = self._degrade_by_type(clean_patch, degrade_type, degrade_range[0])
    
    if degrade_type == 'sr':
        # 额外的上采样步骤：将低分辨率图像插值回原始尺寸
        degrad_patch, _ = self._degrade_by_type(
            degrad_patch, 'resize', [self.downsample_factor])
        # resize 调用 _resize 进行最近邻填充，或使用 _upsample 进行双三次插值
    
    return degrad_patch, self.intensity  # (100, 64, 64) 低频图
```

### 8.3 网络如何处理 SR 任务

**谱提示的作用**（`PG_Spectral_Attention`）：

高分辨率 HSI 具有特定的低秩光谱结构。超分辨率退化后，各波段之间的光谱相关性被削弱。`prompt_param`（128×12 的可学习参数）学习了各种高光谱场景的**典型低秩谱模式**，通过软检索为网络提供谱先验，帮助恢复丢失的谱细节。

**文本提示的作用**：

"A hyperspectral image with reduced spatial resolution."这一文本通过 CLIP 编码为 512 维语义向量，使网络了解当前任务是**空间超分**而非噪声去除，从而调整内部注意力权重的分配方式（更注重空间高频重建）。

**全局残差连接**：

```python
out_dec_level1 = self.output(out_dec_level1) + inp_img
```

对于 SR 任务，`inp_img` 是模糊的低分辨率上采样图，网络只需**预测高频残差**，大幅降低了学习难度，这是图像复原领域的经典技巧（残差学习）。

---

## 9. 多提示机制详解

### 9.1 三类提示的分工

```
任务ID (e.g., 3=超分辨率)
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│                      提示生成阶段                           │
│                                                            │
│  ┌─────────────────────────────────┐                       │
│  │       文本提示（语义层）          │ 512-d CLIP 向量      │
│  │  - 编码任务语义                  │ 固定预训练权重        │
│  │  - "reduced spatial resolution"  │ + 可学习任务向量     │
│  └────────────────┬────────────────┘                       │
│                   │                                        │
│  ┌────────────────▼────────────────┐                       │
│  │       视觉提示（结构层）          │ 可学习参数           │
│  │  - 64×64 / 32×32 特征图          │ 随任务联合优化       │
│  │  - 捕获任务相关空间结构           │                      │
│  └────────────────┬────────────────┘                       │
│                   │                                        │
│  ┌────────────────▼────────────────┐                       │
│  │       谱提示（物理层）            │ 可学习参数           │
│  │  - 128个低秩谱模式               │ 通用先验知识         │
│  │  - 8-d 压缩谱表示                │                      │
│  │  - 每块独立软检索               │                      │
│  └─────────────────────────────────┘                       │
└────────────────────────────────────────────────────────────┘
         │
         ▼
   引导网络处理过程
```

### 9.2 提示注入位置

| 位置 | 提示类型 | 尺度 | 目的 |
|------|---------|------|------|
| 每个 PGSSTB 块内部 | 谱提示（隐式） | 各级 | 增强谱重建 |
| 解码器第2级之前 | TVSP（文本+视觉） | 32×32 | 中等粒度任务引导 |
| 解码器第1级之前 | TVSP（文本+视觉） | 64×64 | 细粒度空间引导 |

### 9.3 PromptFusion（提示融合模块）

```python
class PromptFusion(nn.Module):
    def forward(self, x, prompt):
        out = torch.cat([x, prompt], dim=1)  # 拼接：(B, dim×2, H, W)
        out = self.transformer(out)           # 全局注意力融合
        out = self.conv(out)                  # 投影回原始维度
        return out
```

通道拼接 + Transformer 融合的方式，让网络**自适应地决定**提示信息对特征的影响程度。

---

## 10. 关键设计决策

### 10.1 为何使用 L1 损失而非 MSE？

L1 损失对异常值（如强噪声、极端退化）更鲁棒，且在实践中往往产生更清晰的视觉重建结果，避免 MSE 导致的过度平滑。

### 10.2 为何使用双三次插值而非学习型上采样？

在**退化合成阶段**使用双三次插值是为了模拟真实传感器采集低分辨率图像后的标准插值流程，与真实应用场景一致。实际复原仍由神经网络完成。

### 10.3 窗口注意力为何使用 shift？

标准窗口注意力（W-MSA）每次只能看到 8×8 的局部区域，引入**移位窗口**（SW-MSA，shift=4）后，相邻窗口间可以交换信息，弥补了边界信息孤立的问题，同时保持线性计算复杂度。

### 10.4 为何在跳跃连接处注入提示？

跳跃连接携带了编码器的多尺度细节信息。在此处注入任务提示，可以让解码器在**利用这些细节时**就已经知道当前任务（如 SR 还是去噪），避免任务无关的细节干扰。

### 10.5 谱提示的低秩设计原因

高光谱图像的光谱维度高度相关，大量信息集中在少数几个主成分上。将谱描述符压缩到 `dim//8`（如 12 维），并用 128 个可学习提示组成的**提示字典**来检索先验模式，既减少了计算量，又通过提示库引入了跨场景的谱结构先验知识。

---

## 附录：目录结构速查

```
MP-HSIR/
├── train.py                    # 主训练脚本（PyTorch Lightning）
├── test.py                     # 主测试脚本（9种退化模式）
├── options.py                  # 命令行参数配置
├── requirements.txt            # Python 依赖
│
├── net/
│   ├── MP_HSIR.py              # 🔑 核心网络架构（TVSP/PGSSTB/MP_HSIR_Net）
│   ├── classifier.py           # 退化分类器（辅助）
│   └── comparison_methods/     # 对比方法（SERT/SST/PromptIR等）
│
├── utils/
│   ├── dataset_utils.py        # 数据集类（LMDBDataset/ImageTransformDataset）
│   ├── degradation_utils.py    # 🔑 退化合成（含SR下采样/上采样）
│   ├── image_utils.py          # 图像处理工具（裁剪/增强/分块）
│   ├── val_utils.py            # 评价指标（PSNR/SSIM）
│   ├── loss_utils.py           # 损失函数
│   ├── schedulers.py           # 学习率调度器（线性预热+余弦退火）
│   ├── lmdb_patch.py           # LMDB数据库生成脚本
│   └── mat_data.py             # MATLAB数据处理
│
├── data_dir/
│   └── README.md               # 数据集下载链接
└── ckpt/
    └── README.md               # 预训练权重下载链接
```

---

*文档生成时间：2026-03-19 | 基于代码逻辑分析自动生成*
