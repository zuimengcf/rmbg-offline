# RMBG 离线抠图

Android 端完全离线的 AI 抠图应用。基于 **ONNX Runtime** 本地推理，支持 **CPU / NNAPI / QNN（骁龙 DSP）** 三种加速路径，Jetpack Compose 构建 UI，不依赖任何云端服务即可完成抠图。

## ✨ 特性

- **完全离线推理**：模型内置或下载到本地后，抠图过程零联网
- **多加速路径**：CPU 通用、NNAPI 自动加速、QNN EP 走骁龙 DSP（HTP）获得硬件加速
- **多模型支持**：RMBG-1.4 / RMBG-2.0 / MODNet / Anime-Seg，另有 QNN 离线编译版本
- **多镜像下载**：hf-mirror.com（国内优先）+ huggingface.co（官方兜底）自动切换，分块下载 + 断点续传
- **本地模型导入**：SAF 导入任意 `.onnx`、支持 zip 导入 EPContext（QNN）模型包
- **高级编辑**：区域选择（AreaSelect）、后处理（PostProcess）、画笔精修、按住对比/预览
- **AI 重绘**：联动 LocalDream 本地生图服务（img2img / inpaint）
- **历史管理**：分组折叠、多选、批量保存/删除、排序、网格/列表视图
- **Share 接入**：系统分享图片直达抠图

## 📦 双版本（flavor）

| Flavor | 包名 | 模型策略 | APK 体积 |
|---|---|---|---|
| **normal** | `com.rmbg.offline` | QNN 模型 bin 内置（开箱即用） | ~289MB |
| **lite** | `com.rmbg.offline.lite` | 不内置模型，全部 HF 云下载 | ~148MB |

> 两版本可共存安装。lite 版首次使用需联网下载模型（CPU 模型 + QNN 模型均从 HF 镜像拉取）。

## 🧠 模型清单

### CPU 模型（普通 ONNX，CPU / NNAPI 推理）

| ID | 名称 | 来源仓库 | 体积 |
|---|---|---|---|
| `rmbg14_fp32` | RMBG-1.4 | `briaai/RMBG-1.4` | 176MB |
| `rmbg20` | RMBG-2.0（BiRefNet） | `briaai/RMBG-2.0` | 513MB |
| `modnet` | MODNet（人像） | `Xenova/modnet` | 25MB |
| `anime_seg` | Anime-Seg（动漫） | `skytnt/anime-seg` | 176MB |

### QNN 模型（EPContext 离线编译产物，骁龙 HTP 加速，无需在线编译）

| ID | 名称 | 来源仓库 | 体积 |
|---|---|---|---|
| `qnn_rmbg14_v2` | RMBG-1.4 QNN v2 | `zuimengqm/rmbg-qnn-rmbg14` | 99MB |
| `qnn_animeseg` | Anime-Seg QNN | `zuimengqm/rmbg-qnn-animeseg` | 87MB |
| `qnn_modnet` | MODNet QNN | `zuimengqm/rmbg-qnn-modnet` | 15MB |

> QNN 模型需骁龙 8 Gen2+（FastRPC）+ `libcdsprpc.so`（Android 12+ 自动加载）。无 DSP 设备会自动回退 CPU/NNAPI。

## 🔧 技术栈

- **推理**：`onnxruntime-android 1.30.0` + `onnxruntime-android-qnn 2.5.0`（QNN EP 插件）+ `qnn-runtime 2.50.0`（与 EPContext bin 编译版本对齐）
- **UI**：Jetpack Compose（BOM 2024.10.01）+ Material 3
- **网络**：OkHttp 4.12.0（分块下载）+ Coil（图片加载）
- **构建**：AGP + Kotlin 2.x + Compose 插件，`compileSdk 35 / minSdk 27 / targetSdk 35`

## 🏗️ 目录结构

```
app/src/
├── main/
│   ├── assets/qnn/          # QNN 模型 onnx（ep_cache_context 描述文件）
│   ├── assets/qnnlibs/      # QNN 运行时（HTP backend/stub/skel 全套 .so）
│   ├── java/com/rmbg/offline/
│   │   ├── MainActivity.kt           # 主界面 / 抠图流程 / 模型页 / 历史 / 设置
│   │   ├── ml/ModelManager.kt        # 模型管理：下载 / 部署 / 镜像切换 / 本地导入
│   │   ├── ml/RmbgOnnxEngine.kt      # ONNX Runtime 推理引擎（CPU/QNN/多模型）
│   │   ├── ml/AiRedrawEngine.kt      # AI 重绘引擎（LocalDream 联动）
│   │   ├── ml/LocalDreamClient.kt    # LocalDream HTTP 客户端
│   │   ├── EditorScreen.kt           # 画笔精修
│   │   ├── AreaSelectScreen.kt       # 区域选择
│   │   ├── PostProcessScreen.kt      # 后处理
│   │   └── HistoryManager.kt         # 历史记录持久化
│   └── jniLibs/             # stable-diffusion 原生库（AI 重绘用）
└── normal/assets/qnn/       # ★ normal 版：QNN bin（EPContext 二进制，不入库）
```

## 🛠️ 构建

```bash
# 编译（flavor 化后需分别指定）
./gradlew :app:compileNormalDebugKotlin :app:compileLiteDebugKotlin

# 打包 release（需 keystore.properties 签名配置）
./gradlew :app:assembleNormalRelease :app:assembleLiteRelease
```

### 签名

- 签名证书与密码存于 `keystore.properties`（**已被 .gitignore 忽略，绝不入库**）
- 未配置签名文件时 release 构建产出 unsigned APK

### 模型资产说明

- `normal` 版所需的 QNN bin（`app/src/normal/assets/qnn/*.bin`）**不纳入本仓库**：
  由 HF 仓库下载（`zuimengqm/rmbg-qnn-*`）或本地准备后放入，
  否则 normal 版 APK 不含 QNN 模型，运行时会走云下载。
- `lite` 版无需任何内置模型资产，首次使用自动从 HF 镜像下载。

## 📄 License

请遵循各模型原始仓库（briaai / skytnt / Xenova）的许可条款。