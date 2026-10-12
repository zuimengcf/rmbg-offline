# RMBG 离线抠图

一个**完全离线**的 Android AI 抠图应用：选一张图片，AI 自动识别人物 / 动漫角色 / 商品等主体并一键抠出透明背景。所有推理都在手机本地完成，不依赖任何云端服务，照片不出设备。

基于 ONNX Runtime 本地推理，提供 CPU、NNAPI、QNN（骁龙 DSP 硬件加速）三条加速路径；界面使用 Jetpack Compose 构建。

---

## ✨ 功能特性

| 能力 | 说明 |
|---|---|
| 🖼 **一键离线抠图** | 本地 AI 推理，模型就绪后全程零联网，图片不出手机 |
| ⚡ **三重加速** | CPU 通用推理 / NNAPI 自动加速 / QNN 走骁龙 HTP 硬件加速 |
| 🧠 **多种模型可选** | RMBG-1.4、RMBG-2.0、MODNet、Anime-Seg，按场景选择（人像 / 动漫 / 通用） |
| 🔍 **SAM 智能选区** | 点选或框选图片中的任意对象，不只限「人像」，支持多目标分别处理 |
| ✂️ **精修工作台** | 画笔擦除 / 恢复边缘、区域后处理、长按对比原图、导出前预览 |
| 🎨 **AI 重绘** | 联动 LocalDream 本地生图服务，支持 img2img / inpaint 二次创作 |
| 🕘 **历史记录** | 按原图自动分组，多选批量保存 / 删除，多种排序与列表 / 网格视图 |
| 🔗 **系统分享直达** | 从相册或其他 App「分享」图片，直接进入抠图 |
| 📤 **导入导出** | 备份 / 恢复全部模型与设置，换机迁移方便 |
| 🌐 **智能模型下载** | 国内镜像（hf-mirror）优先、官方源兜底自动切换，多线程分块 + 断点续传 |

## 📦 两个版本

| 版本 | 包名 | 模型策略 | 适合谁 |
|---|---|---|---|
| **normal 完整版** | `com.rmbg.offline` | **开箱即用**：Anime-Seg、SAM、超分模型全部内置，含骁龙 QNN 硬件加速 | 空间充足，装上就能用 |
| **lite 精简版** | `com.rmbg.offline.lite` | **轻装上阵**：不内置模型，首次使用时按需联网下载，用 CPU / NNAPI 推理 | 想省空间，或设备不支持 QNN |

> 两个版本包名不同，可**共存安装**。lite 版首次使用需联网下载所选模型，之后即可离线使用；normal 版开箱即用无需下载。

## 🚀 快速开始

1. 前往 [Releases 页面](https://github.com/zuimengcf/rmbg-offline/releases) 下载最新 APK：
   - `rmbg-normal.apk` — 完整版，开箱即用
   - `rmbg-lite.apk` — 精简版，按需下载模型
2. 安装后打开 App，在「模型」页确认模型状态（normal 版已就绪；lite 版点击下载）。
3. 从相册选择图片，或从其他 App 分享图片进来，即可开始抠图。
4. 精修满意后保存，支持 PNG 透明背景导出。

## 🧠 支持的模型

**CPU 模型**（CPU / NNAPI 推理，normal 内置 Anime-Seg，其余按需下载）：

| 模型 | 特长 |
|---|---|
| RMBG-1.4 | 通用抠图，官方原版 |
| RMBG-2.0 | BiRefNet，最强效果 |
| MODNet | 人像抠图，轻量快速 |
| Anime-Seg | 动漫人物，ISNet |

**QNN 模型**（EPContext 离线编译产物，骁龙 HTP 硬件加速，无需在线编译）：

| 模型 |
|---|
| RMBG-1.4 QNN |
| Anime-Seg QNN |
| MODNet QNN |

**附加能力**：SAM 智能选区（Encoder + Decoder）、Real-ESRGAN 超分。

## 📱 设备要求

- **最低**：Android 8.0（API 27）以上，arm64 设备
- **QNN 硬件加速**：需骁龙 8 Gen2+ 且 Android 12+；无 DSP 的设备会自动回退 CPU / NNAPI，不影响使用
- **lite 版**：首次下载模型需要联网

## 🙏 致谢

- 抠图模型：briaai（RMBG-1.4 / RMBG-2.0）、Xenova（MODNet）、skytnt（Anime-Seg）
- 智能选区：Meta（Segment Anything Model）
- 超分模型：xinntao（Real-ESRGAN）
- AI 重绘：LocalDream 本地生图服务（基于 Stable Diffusion）
- 推理框架：Microsoft ONNX Runtime
- QNN 集成参考 Qualcomm / ONNX Runtime 官方文档
- 模型许可遵循各原始仓库规定，详见各自页面