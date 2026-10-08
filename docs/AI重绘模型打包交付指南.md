# 🎨 AI 重绘模型（SD15npu QNN）打包交付指南

> 适用对象：为 AI 重绘（`--type sd15npu`，QNN 推理）制作/交付 **SD1.5 模型压缩包**。
> 硬性依据：`AiRedrawEngine.kt`（引擎进程 + 模型部署）、`Prefs.aiRedrawHfUrl`（云下载链接）。
> 本文教你**从零把一个 SD1.5 底模打包成 App / LocalDream 可直接部署的 QNN zip**，并给出最易踩的 **VAE 格式兼容**排障步骤。

---

## 1. 包内结构要求（SD1.5 QNN，7 个文件平铺）

```
<模型名>.zip                          ← 通常 1.1~1.3GB
├── unet.bin          ← UNet context binary（~880MB）
├── vae_encoder.bin   ← VAE 编码器
├── vae_decoder.bin   ← VAE 解码器
├── clip_v2.mnn       ← CLIP 文本编码器（MNN 格式，~156MB）
├── tokenizer.json    ← 分词器
├── token_emb.bin     ← token embedding
└── pos_emb.bin       ← position embedding
```

| 项 | 要求 |
|---|---|
| **平铺** | 文件在 zip **根目录**，不要套子目录（引擎按文件名取用）|
| **压缩法** | **必须 `store`（不压缩）**——模型 bin 已是大文件，压缩既不省空间，还拖慢部署/下载，甚至可能因重算 CRC 损坏 bin |
| **CLIP 是 MNN** | `clip_v2.mnn` 头 2 字节 = `20 00`，引擎专门用 MNN 跑文本编码，**MNN 正常**；仅 VAE 必须为 QNN |
| **unet 自留** | UNet context binary 头 4 字节 = `00 00 00 02`，保留各底模自己的即可 |
| `.patch` | 高分辨率补丁，可选带 |

---

## 2. ⚠️ VAE 必须是 QNN（最易踩坑）

引擎以 **QNN 后端**（`--type sd15npu`）加载整个模型，**要求 VAE 必须是 QNN context binary**。
若 VAE 实为 **MNN 格式**，引擎会当 QNN 强跑，导致 VAE 推理直接失败。

### 2.1 症状

| 现象 | 说明 |
|---|---|
| 引擎日志出现 `QNN VAE enc exec failed` | VAE encoder 执行失败 |
| 重绘输出"根本没动" | VAE 编/解码失败，无法重建图像，图无变化 |

### 2.2 怎么判 VAE 是不是 QNN（看大小即可，不用解包）

| VAE 文件 | 标准 QNN（SD1.5 通用，✅）| MNN（❌ 不兼容）|
|---|---|---|
| `vae_encoder.bin` | **41,438,176 B**（≈40MB）| 75,815,080 B（≈72MB）|
| `vae_decoder.bin` | **59,949,944 B**（≈58MB）| 110,170,224 B（≈105MB）|

> 看到越大号（约 1.8×）的 VAE 就是 MNN，不可直接用，需按 §2.3 替换。

### 2.3 替换为标准 QNN VAE

SD1.5 的 VAE 是**所有底模通用**的。任取一个正规 QNN SD1.5 包（如 MeinaMixV12 / AnythingV5）里的 VAE 换上即可，**unet 保留各模型自己的**：

```bash
# ① 从标准 QNN 包取出它的两个 VAE
unzip -j MeinaMixV12_qnn2.28_8gen2.zip '*/vae_encoder.bin' '*/vae_decoder.bin' -d std/

# ② 重打包：逐项复制源 zip 的所有文件，但 vae_encoder.bin / vae_decoder.bin
#    改写成 std/ 里的标准版（python zipfile，ZIP_STORED 不压缩）
```

替换后 VAE 回到标准 QNN 格式，`QNN VAE enc exec failed` 消失，重绘正常生效。

---

## 3. 校验清单（打包完自查）

```
□ zip 内 7 个文件平铺根目录
□ unet.bin / vae_*.bin 头 4 字节 = 00 00 00 02（QNN context binary）
□ clip_v2.mnn 头 2 字节 = 20 00（MNN，正常）
□ vae_encoder.bin = 41,438,176 B（标准 QNN）
□ vae_decoder.bin = 59,949,944 B（标准 QNN）
□ zip 用 store 压缩法（非 deflate）
```

校验命令速查：

```bash
ZIP=你的模型.zip
# ① 清单（应有 unet / vae_encoder / vae_decoder / clip_v2 / tokenizer / token_emb / pos_emb）
unzip -l "$ZIP"
# ② VAE 是否标准尺寸（40MB / 58MB）
unzip -l "$ZIP" | grep -iE 'vae_(en|de)coder'
# ③ 头 magic：QNN=00 00 00 02；CLIP(MNN)=20 00
for f in unet.bin vae_encoder.bin vae_decoder.bin clip_v2.mnn; do
  echo "== $f =="; unzip -p "$ZIP" "$f" | head -c 4 | od -A n -t x1
done
```

---

## 4. 交付/集成到 App

| 路径 | 做法 | 改代码？ |
|---|---|---|
| **本地 zip 导入** | 设置页 → AI 重绘 → 选 zip → 自动解压部署（`AiRedrawEngine` 按 `modelIdFor` 定位目录，`unet.bin` 存在即就绪）| 否 |
| **HF 云下载 zip** | `Prefs.aiRedrawHfUrl` 填某模型仓库的 **resolve 直链**（`https://huggingface.co/<user>/<repo>/resolve/main/<模型>.zip`）+ `Prefs.aiRedrawHfToken` → `downloadAiRedrawZip` 走镜像循环 + 断点续传 | 否（设置页填链接）|

> 云发布建议：新建一个 **model** 仓库（非 space），把 zip 传上去，再用上面的 resolve 直链配到 app 设置页即可让用户免本地导入、直接云下载。