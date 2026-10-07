# RMBG 离线抠图

一个 **完全离线** 的 Android AI 抠图应用。基于 ONNX Runtime 本地推理，支持 **CPU / NNAPI / QNN（骁龙 DSP）** 三种加速路径，Jetpack Compose 构建界面。抠图、精修、历史记录全部在本机完成，不依赖任何云端服务。

---

## 目录

- [功能特性](#功能特性)
- [双版本说明](#双版本说明)
- [模型矩阵](#模型矩阵)
- [环境要求](#环境要求)
- [快速构建](#快速构建)
- [模型资产说明](#模型资产说明)
- [目录结构](#目录结构)
- [技术架构](#技术架构)
- [常见问题 FAQ](#常见问题-faq)
- [许可证与致谢](#许可证与致谢)

---

## 功能特性

| 能力 | 说明 |
|---|---|
| 🖼 离线抠图 | ONNX Runtime 本地推理，模型就绪后全程零联网 |
| ⚡ 三重加速 | CPU 通用、NNAPI 自动加速、QNN 走骁龙 HTP 硬件加速 |
| 🧠 多模型 | RMBG-1.4 / RMBG-2.0 / MODNet / Anime-Seg，另含 QNN 离线编译版 |
| 🌐 智能下载 | hf-mirror.com（国内优先）+ huggingface.co（官方兜底）自动切换，多线程分块 + 断点续传 |
| 📥 本地模型导入 | SAF 导入任意 `.onnx`，支持 zip 导入 EPContext（QNN）模型包 |
| ✂️ 高级精修 | 区域选择、后处理、画笔擦除/恢复、长按对比原图、预览导出效果 |
| 🎨 AI 重绘 | 联动 LocalDream 本地生图服务（img2img / inpaint） |
| 🕘 历史管理 | 按原图分组折叠、多选批量保存/删除、7 种排序、列表/网格视图 |
| 🔗 系统分享 | 从相册 / 其他 App「分享」图片直达抠图 |

## 双版本说明

| Flavor | 包名 | 模型策略 | APK 体积 | 适用场景 |
|---|---|---|---|---|
| **normal**（完整版） | `com.rmbg.offline` | QNN 模型 bin **构建时自动从 HF 下载并内置**，开箱即用 | ~289MB | 骁龙 8 Gen2+ 设备，追求 QNN 硬件加速 |
| **lite**（精简版） | `com.rmbg.offline.lite` | 不内置模型，**首次运行联网从 HF 下载** | ~148MB | 小体积、CPU/NNAPI 即可、骁龙旧款或无 DSP |

> 两个版本可**共存安装**（包名不同）。lite 版首次使用需联网下载所选模型（默认 Anime-Seg CPU，可切换 QNN 模型）。

## 模型矩阵

### CPU 模型（普通 ONNX，CPU / NNAPI 推理）

| ID | 名称 | 来源（HF 仓库） | 体积 | 特长 |
|---|---|---|---|---|
| `rmbg14_fp32` | RMBG-1.4 | [`briaai/RMBG-1.4`](https://huggingface.co/briaai/RMBG-1.4) | 176MB | 通用抠图，官方原版 |
| `rmbg20` | RMBG-2.0 | [`briaai/RMBG-2.0`](https://huggingface.co/briaai/RMBG-2.0) | 513MB | BiRefNet，最强效果 |
| `modnet` | MODNet | [`Xenova/modnet`](https://huggingface.co/Xenova/modnet) | 25MB | 人像抠图，轻量快速 |
| `anime_seg` | Anime-Seg | [`skytnt/anime-seg`](https://huggingface.co/skytnt/anime-seg) | 176MB | 动漫人物，ISNet |

### QNN 模型（EPContext 离线编译产物，骁龙 HTP 加速，无需在线编译）

| ID | 名称 | 来源（HF 仓库） | 体积 |
|---|---|---|---|
| `qnn_rmbg14_v2` | RMBG-1.4 QNN v2 | [`zuimengqm/rmbg-qnn-rmbg14`](https://huggingface.co/zuimengqm/rmbg-qnn-rmbg14) | 99MB |
| `qnn_animeseg` | Anime-Seg QNN | [`zuimengqm/rmbg-qnn-animeseg`](https://huggingface.co/zuimengqm/rmbg-qnn-animeseg) | 87MB |
| `qnn_modnet` | MODNet QNN | [`zuimengqm/rmbg-qnn-modnet`](https://huggingface.co/zuimengqm/rmbg-qnn-modnet) | 15MB |

> QNN 模型需**骁龙 8 Gen2+（FastRPC）**并在 Android 12+ 加载 OEM 库 `libcdsprpc.so`。无 DSP 设备会自动回退 CPU / NNAPI。

## 环境要求

- **JDK** 17
- **Android SDK**：`platforms;android-35` + `build-tools;35.0.0`
- **Gradle**：使用仓库内置 `gradlew` wrapper（无需单独安装）
- **硬件参考**：QNN 加速需骁龙 8 Gen2+；CPU 版任意 arm64 设备
- 仅构建 **arm64-v8a** 架构（QNN HTP 仅支持 64 位骁龙 DSP）

## 快速构建

```bash
# 1. 编译（flavor 化后需分别指定任务）
./gradlew :app:compileNormalDebugKotlin :app:compileLiteDebugKotlin

# 2. 打包 Debug APK
./gradlew :app:assembleNormalDebug :app:assembleLiteDebug

# 3. 打包 Release APK
./gradlew :app:assembleNormalRelease :app:assembleLiteRelease
```

### 构建完整版（normal）会自动下载 QNN bin

normal 版的 QNN 模型二进制（`.bin`，约 197MB）**不入 git 仓库**（体积大）。构建 normal 版时 Gradle 会自动检测 `app/src/normal/assets/qnn/` 下是否已有 `.bin`：

- **未下载** → 自动从 HF 镜像下载（hf-mirror.com 优先，huggingface.co 兜底），成功后继续打包
- **已下载** → 直接打包（支持断点续传，重复构建不会重复下载）
- **下载失败** → 构建中断并提示（请检查网络后重试）

> Lite 版不内置任何模型，构建无需下载。

## 模型资产说明

| 资产 | 存放位置 | 是否入库 | 说明 |
|---|---|---|---|
| QNN 模型 `.onnx`（描述文件，<1KB） | `app/src/main/assets/qnn/` | ✅ 入库 | EPContext 引用 bin 的文件名 |
| QNN 模型 `.bin`（EPContext，197MB） | `app/src/normal/assets/qnn/` | ❌ 忽略 | normal 构建时自动从 HF 拉取 |
| QNN 运行时 `.so`（133MB） | `app/src/main/assets/qnnlibs/` | ✅ 入库 | HTP backend/stub/skel v68-v81 全套 |
| SD 原生库 | `app/src/main/jniLibs/` | ✅ 入库 | AI 重绘用 |

## 目录结构

```
rmbg-offline-apk/
├── app/
│   └── src/
│       ├── main/
│       │   ├── assets/
│       │   │   ├── qnn/                # QNN 模型 .onnx 描述文件
│       │   │   └── qnnlibs/            # QNN 运行时 .so（backend/stub/skel）
│       │   ├── java/com/rmbg/offline/
│       │   │   ├── MainActivity.kt     # 主界面 / 抠图 / 模型页 / 历史 / 设置
│       │   │   ├── ml/
│       │   │   │   ├── ModelManager.kt     # 模型管理：下载/部署/镜像/本地导入
│       │   │   │   ├── RmbgOnnxEngine.kt   # ONNX Runtime 推理引擎（CPU/QNN）
│       │   │   │   ├── AiRedrawEngine.kt   # AI 重绘逻辑
│       │   │   │   └── LocalDreamClient.kt # LocalDream HTTP 客户端
│       │   │   ├── EditorScreen.kt     # 画笔精修
│       │   │   ├── AreaSelectScreen.kt # 区域选择
│       │   │   ├── PostProcessScreen.kt# 后处理
│       │   │   └── HistoryManager.kt   # 历史持久化
│       │   ├── jniLibs/                # stable-diffusion 原生库
│       │   └── res/                    # 资源
│       └── normal/assets/qnn/          # ★ normal 版 QNN .bin（构建时自动下载）
├── .github/workflows/                  # CI：release-apk.yml
├── gradle/                             # Gradle wrapper
└── build.gradle.kts
```

## 技术架构

```
┌─────────────────────────────────────────────┐
│                  Compose UI                  │
├─────────────────────────────────────────────┤
│         ModelManager（模型下载/部署）          │
│   多镜像切换 · 分块下载 · 断点续传 · 本地导入   │
├─────────────────────────────────────────────┤
│            RmbgOnnxEngine（推理引擎）          │
│   ┌─────────┐   ┌──────────────────────┐    │
│   │ CPU EP  │   │   QNN EP (HTP/DSP)   │    │
│   │ NNAPI   │   │  EPContext 离线编译   │    │
│   └─────────┘   └──────────────────────┘    │
├─────────────────────────────────────────────┤
│   onnxruntime-android 1.30.0                 │
│   + onnxruntime-android-qnn 2.5.0 (QNN EP)   │
│   + qnn-runtime 2.50.0 (与 EPContext 对齐)    │
└─────────────────────────────────────────────┘
```

- **QNN 三件套**：`onnxruntime-android`（核心）+ `onnxruntime-android-qnn`（EP 插件）+ `qnn-runtime`（运行时）。三者必须显式声明——QNN 插件 AAR 的 POM **不会**传递引入 ORT 核心或运行时。
- **版本对齐**：`qnn-runtime 2.50.0` 必须与 EPContext bin 的编译版本（QNN SDK 2.50.0，见 onnx 的 `ep_sdk_version` 属性）一致，否则无法加载 context binary。
- **legacy JNI 打包**：QNN HTP 加载器从 `ApplicationInfo.nativeLibraryDir` 按路径发现 backend/stub/skel 库，必须 `useLegacyPackaging = true`（不压缩 native 库）。
- **Android 12+ 隐藏库**：`libcdsprpc.so` 是 OEM 库不随 APK 分发，通过 manifest `<uses-native-library required="false">` 显式请求，无 FastRPC 设备可正常回退 CPU/NNAPI。

## 常见问题 FAQ

**Q：QNN 模型怎么没生效？**
A：需要骁龙 8 Gen2+ 且 Android 12+。可在「模型」页查看引擎状态；无 DSP 自动回退 CPU/NNAPI。

**Q：lite 版下载模型很慢 / 失败？**
A：App 默认 hf-mirror.com（国内镜像）优先、huggingface.co 兜底，支持断点续传。网络波动时到「设置」手动切换镜像源重试。

**Q：normal 版构建报 bin 下载失败？**
A：确认网络可达 hf-mirror.com / huggingface.co；可手动把 `.bin` 放到 `app/src/normal/assets/qnn/`（文件名与 `.onnx` 同名）跳过下载。

**Q：两个版本能一起装吗？**
A：可以，包名不同（`com.rmbg.offline` / `com.rmbg.offline.lite`）。

**Q：构建 OOM / 很慢？**
A：Gradle 已配置 `-Xmx4608m` 并关闭并行；首次构建需下载依赖 + 压缩 200MB assets，耐心等待。

## 许可证与致谢

- 本项目代码按各自文件头部声明授权
- 模型许可遵循各原始仓库：briaai（RMBG）、skytnt（Anime-Seg）、Xenova（MODNet）
- QNN 相关集成参考 Qualcomm / Microsoft ONNX Runtime 官方文档