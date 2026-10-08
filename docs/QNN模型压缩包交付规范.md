# 📦 QNN 抠图模型压缩包交付规范

> 适用对象：本 App（rmbg-offline）内置/导入的 **QNN EPContext 抠图模型**（onnx + bin）。
> 硬性依据：`ModelManager.kt`（识别/导入/部署）、`RmbgOnnxEngine.kt`（加载）、`AiRedrawEngine.kt`（AI 重绘引擎）。

---

## 1. 命名规范（三层：模型标识 → 文件 → 压缩包）

### 1.1 模型标识命名

```
qnn_<模型>_<平台>[_<变体>...]
```

| 段 | 规则 | 示例 |
|---|---|---|
| 前缀 | **必须 `qnn_`**（App 识别 QNN EPContext 的硬标记，见 `qnnContextFileFor`/`modelFileName`） | `qnn_animeseg_...` |
| 模型 | 小写字母数字下划线，语义化 | `animeseg`、`rmbg14`、`modnet` |
| 平台 | **必写**，二选一格式：`sm8550`\|`8gen2` / `sm8650`\|`8gen3` / `sm8750`\|`8gen4` | `sm8650` |
| 变体 | 可选，下划线连接：`soc57`、`v75`、`512`、`qairt250`、`fp32` 等 | `_soc57_v75_qairt250_fp32` |

> ⚠️ **平台段决定型号锁定**（`qnnTargetSoc` 解析，见 `ModelManager.kt`）：
> - 含 `sm8650`/`8gen3` → 只在 8 Gen 3（SM8650）设备可选
> - 含 `sm8550`/`8gen2` → 只在 8 Gen 2（SM8550）设备可选
> - 不写平台段 = **通用**（所有骁龙设备可选，不做锁定）

### 1.2 文件命名（onnx 与 bin 必须同名同目录）

```
<标识>.onnx     ← EPContext 头，约 0.8~1KB
<标识>.bin      ← context binary，15MB~100MB+
```

✅ 例：`AnimeSeg ISNetIS`（SM8650 / soc57 / V75 / QAIrt250 / FP32）规范化名：

```
qnn_animeseg_sm8650_soc57_v75_qairt250_fp32.onnx
qnn_animeseg_sm8650_soc57_v75_qairt250_fp32.bin
```

### 1.3 压缩包命名

```
qnn_<模型>_<平台>_<sdk>[_<变体>].zip
```

✅ 示例：`qnn_animeseg_sm8650_qairt250_fp32.zip`

---

## 2. 压缩包内部结构（硬性要求）

```
qnn_animeseg_sm8650_qairt250_fp32.zip
├── qnn_animeseg_sm8650_soc57_v75_qairt250_fp32.onnx   ← 必须有 .onnx
└── qnn_animeseg_sm8650_soc57_v75_qairt250_fp32.bin    ← 必须有配套 .bin
```

| 要求 | 说明 |
|---|---|
| 平铺 | 文件在 zip **根目录**，不要套子目录（App 解压只取 basename） |
| 无隐藏文件 | 不要 `__MACOSX`、`.DS_Store`（App 解压会跳过，但避免混淆） |
| 仅 onnx+bin | 其余 meta/readme 可放但忽略；**禁止多个 .bin 互相混淆**（App 匹配 bin 时用精确→包含→唯一兜底） |

---

## 3. 数据内容要求（bin 版本修改时必读）

### 3.1 BIN（编译产物，本身不可改）

| 项 | 要求 |
|---|---|
| 格式 | `qnn-context-binary-generator` 直接输出的 **context binary v2** |
| ❌ 排除 | POSIX tar 权重包（偏移 257 处有 `ustar` 标记 → App 检测并提示拒载） |
| SDK 版本 | **必须 QNN 2.50**（App runtime 已锁定 `qnn-runtime:2.50.0`；2.39 加载 2.50 模型报 `ORT_NOT_IMPLEMENTED`） |
| 目标 | soc_model 57（SM8650）HTP v75 → 对应 `libQnnHtpV75Skel.so`（App 已内置 V68~V81 全套） |

### 3.2 ONNX（App 可自动修复，但建议导出即合规）

| 项 | 要求 |
|---|---|
| `ep_cache_context` 属性 | 值 = **bin 的纯文件名**（如 `qnn_animeseg_sm8650_..._fp32.bin`） |
| ❌ 排除 | 绝对路径 / 带目录引用（ORT 1.30 加载报 `ORT_INVALID_GRAPH: External mode should set ep_cache_context field with a relative path`） |
| 备注 | App 导入 zip 时**自动**把绝对路径改写为 basename（`normalizeEpContextPath`，protobuf 类型感知重编码），但导出时就写相对路径最稳 |

---

## 4. 校验清单（交付前自查）

```
□ 文件名符合 qnn_<模型>_<平台>[_<变体>] 规范
□ onnx 与 bin 同名同目录
□ zip 内文件平铺根目录、无隐藏文件
□ onnx 内 ep_cache_context = bin 纯文件名
□ bin 是 context binary v2（非 tar）
□ bin 由 QNN 2.50 编译（版本串 v2.50.0.260828221209）
□ 目标 soc_model/型号 与文件名平台段一致
□ bin 大小与登记 sizeBytes 一致
```

---

> 本文档教你**从零做出一个 App 直接可用的 QNN 抠图模型 zip**（onnx + bin 打包）。
> 按 §5 的 6 步走完，产物即符合 App 的本地导入 / HF 云下载两条路径的要求。

---

## 5. 制作指南（6 步产出合规 zip）

### 5.1 准备编译环境

| 项 | 要求 |
|---|---|
| QNN SDK | **2.50**（`ep_sdk_version` 必须 `v2.50.0.260828221209`，与 App runtime `qnn-runtime:2.50.0` 匹配；用 2.39 编译的 bin 加载报 `ORT_NOT_IMPLEMENTED`） |
| 目标设备 | 确认 soc_model + HTP 版本（例：SM8650 = soc_model 57 = HTP v75） |
| ONNX 工具 | Python `onnx` 包（校验/修改 `ep_cache_context` 用） |

### 5.2 编译 context binary（核心命令）

在 **bin 输出目录下**执行（关键：当前目录决定 onnx 里引用路径的形态）：

```bash
cd <你的bin输出目录>          # ← 必须先 cd，保证导出引用为纯文件名
qnn-context-binary-generator \
  --model <你的模型.onnx> \
  --backend htp \
  --binary_file <输出名>.bin \
  --soc_model 57 \            # SM8650=57；SM8550 用 55/对应值
  --htp_socs 75 \             # HTP 版本：V75（SM8650）/ V73（SM8550）
  --debug \
  --saver 1
```

> ⚠️ **不要在命令行里写输出路径**（如 `/home/xxx/qnn_out/xx.bin`）——那会让 onnx 的 `ep_cache_context` 变成绝对路径，App 加载报 `ORT_INVALID_GRAPH`。**cd 进目录后只写文件名**。

### 5.3 规范化命名（onnx + bin 同名同目录）

按 §1 规则重命名：

```bash
# 模型标识：qnn_<模型>_<平台>[_<变体>]
# 例：AnimeSeg ISNetIS（SM8650 / soc57 / V75 / QAIrt250 / FP32）
mv <工具链输出.onnx> qnn_animeseg_sm8650_soc57_v75_qairt250_fp32.onnx
mv <工具链输出.bin>  qnn_animeseg_sm8650_soc57_v75_qairt250_fp32.bin
```

**bin 名必须与 onnx 内 `ep_cache_context` 引用一致**（改名后检查，见 5.4）。

### 5.4 校验 / 修正 onnx 的 `ep_cache_context`

用 Python 确认引用是**纯文件名**（无 `/`、无目录）：

```bash
python3 - <<'EOF'
import onnx
m = onnx.load("qnn_animeseg_sm8650_soc57_v75_qairt250_fp32.onnx")
for node in m.graph.node:
    if node.op_type == "EPContext":
        for a in node.attribute:
            if a.name == "ep_cache_context":
                print("引用:", a.s.decode())
                ok = b"/" not in a.s and b"\\" not in a.s
                print("合规(纯文件名):", "✅" if ok else "❌ 需修改")
                if not ok:
                    a.s = b"qnn_animeseg_sm8650_soc57_v75_qairt250_fp32.bin"
                    print("已改写为:", a.s.decode())
onnx.save(m, "qnn_animeseg_sm8650_soc57_v75_qairt250_fp32.onnx")
EOF
```

> 若显示合规（纯文件名）即跳过修改；App 导入时也会自动 normalize，但导出时就合规最稳。

### 5.5 打包 zip（平铺根目录）

```bash
zip -j qnn_animeseg_sm8650_qairt250_fp32.zip \
  qnn_animeseg_sm8650_soc57_v75_qairt250_fp32.onnx \
  qnn_animeseg_sm8650_soc57_v75_qairt250_fp32.bin
```

- `-j`：**junk paths**，强制平铺根目录（App 解压只取 basename）
- 包内只允许 onnx + bin 两个文件；不要带 `__MACOSX/`、`.DS_Store`、子目录

### 5.6 交付前校验（对照 §4 清单逐项过）

用 §6 的现成命令复核一遍（zip 结构 / tar 标记 / SDK 版本 / 引用路径），全绿即可交付。

---

## 6. 校验命令速查（交付前跑一遍）

```bash
ZIP=你的包.zip
# ① 包内结构（应只有 onnx+bin，平铺）
unzip -l "$ZIP"
# ② 解压
TMP=$(mktemp -d) && unzip -o -q "$ZIP" -d "$TMP" && cd "$TMP"
# ③ bin 是否 context binary v2（头 4 字节 = 00 00 00 02）
head -c 4 *.bin | od -A x -t x1z
# ④ 是否 tar 权重包（偏移 257 处应无 ustar）
dd if=*.bin bs=1 skip=257 count=5 2>/dev/null | od -c
# ⑤ ep_cache_context 是否为纯文件名（无 /）
python3 -c "import onnx;m=onnx.load('$(ls *.onnx)');print([a.s for n in m.graph.node if n.op_type=='EPContext' for a in n.attribute if a.name=='ep_cache_context'])"
# ⑥ SDK 版本
strings $(ls *.onnx) | grep ep_sdk_version -A1
```

---

## 7. 初始化路径（zip 就位后）

| 路径 | 做法 | 改代码？ |
|---|---|---|
| **本地 zip 导入** | 模型页 →「添加自定义 NPU 模型」→ 选 zip → App 自动解压/规范化/部署 | 否 |
| **HF 云下载 zip**（内置化） | `BuiltinModel.hfZip` 填仓库内 zip 路径 → `ensureQnnContextDual` 自动走整包下载→解压部署（`downloadQnnZip`/`deployQnnZip`，镜像循环+断点续传） | 是（ModelManager 注册） |
| 传统 onnx/bin 双文件 | `hfZip` 不设置时的历史路径，仍可用 | 是 |

---

## 8. 已注册模型一览（截至 v1.1.1）

| 模型 id | onnx / bin 文件名 | 目标平台 | 大小 |
|---|---|---|---|
| `qnn_rmbg14_v2` | `qnn_rmbg14_sdk250_v2.onnx` / `.bin` | 通用（SDK250） | 99MB |
| `qnn_animeseg` | `qnn_animeseg_v73.onnx` / `.bin` | SM8550（描述声明） | 87MB |
| `qnn_modnet` | `qnn_modnet_sm8550_512.onnx` / `.bin` | SM8550 | 15MB |
| `qnn_animeseg_sm8650` | `qnn_animeseg_sm8650_soc57_v75_qairt250_fp32.onnx` / `.bin` | SM8650 | 87MB（已核验合规 ✅） |
