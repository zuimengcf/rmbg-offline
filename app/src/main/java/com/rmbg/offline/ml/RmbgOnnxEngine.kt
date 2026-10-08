package com.rmbg.offline.ml

import android.graphics.Bitmap
import android.graphics.Color
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * RMBG-1.4 ONNX 推理引擎
 *
 * 参考 HuggingFace briaai/RMBG-1.4 的预处理/后处理流程：
 * - 输入: 1x3x1024x1024 RGB float (归一化到 [0,1])
 * - 输出: 1x1x1024x1024 float 前景概率图（Sigmoid 输出）
 * - 后处理: 概率图 → alpha mask → 与原图合成透明 PNG
 *
 * 输出读取方式参考 ChuBaichuan-TagAI 的 TaggerEngine：
 * 遍历 OrtSession.Result 的 iterator，用 entry.value.value 取底层数据，
 * 兼容 onnxruntime-android 的 FloatArray/DoubleArray 等不同返回类型。
 */
class RmbgOnnxEngine(
    private val modelFile: File,
    private val enableNnapi: Boolean = false,
    private val enableQnn: Boolean = false,
    private val threads: Int = 4,
    private val nativeLibDir: String? = null
) {

    /** 当前绑定的模型文件绝对路径（供上层判断是否需要重建引擎） */
    val modelPath: String = modelFile.absolutePath

    /** 当前使用的 CPU 线程数（供上层判断线程数变化时是否需要重建引擎） */
    val threadCount: Int = threads

    /** 最近一次加载时 NNAPI 是否成功启用（供 UI 反馈） */
    @Volatile
    var nnapiActive: Boolean = false
        private set

    /** NNAPI 启用/回退的说明（供 UI 反馈） */
    @Volatile
    var nnapiStatus: String = "未启用"
        private set

    /** 背景硬化阈值（alpha 下限，0~255）：
     *  低于此值的半透明背景像素被硬化为全透明(磨砂残影清除)，主体/边缘保留柔和渐变。
     *  可调：值越高越"干净"(背景更彻底透明)，越低越保留半透明发丝等细节。默认 192。 */
    @Volatile
    var bgCleanAlpha: Int = 192

    /** 最近一次加载时 QNN 是否成功启用（供 UI 反馈） */
    @Volatile
    var qnnActive: Boolean = false
        private set

    /** QNN 启用/回退的说明（供 UI 反馈） */
    @Volatile
    var qnnStatus: String = "未启用"
        private set

    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null

    /** 是否正在推理（runRmbg 设置，供 onStop 判断能否立即释放引擎） */
    @Volatile
    var isRunning: Boolean = false

    /** 关闭请求：onStop 时若正在推理，仅标记此标志，等推理结束再真正 close（避免 close 与 native run 冲突闪退） */
    @Volatile
    var closeRequested: Boolean = false

    /** 模型输入名（从模型读取） */
    private var inputName: String = "input"
    /** 模型输入尺寸（从模型读取，RMBG 为 1024） */
    private var inputSize: Int = 1024
    /** 输入布局 NCHW / NHWC */
    private var inputLayout: InputLayout = InputLayout.NCHW
    /** 输入通道顺序：QNN SDK 离线编译的部分模型(如 rmbg14_sdk250)期望 BGR，引擎默认 RGB */
    private var bgrInput: Boolean = false
    /** 输出是否需要 sigmoid：该模型的原生输出是 logits(未过 sigmoid)，直接当概率会导致
     *  人物内部(logits 小正)被压到阈值下抠空成"线条画"。对 rmbg14_sdk250 应用 sigmoid。 */
    private var sigmoidOutput: Boolean = false

    private enum class InputLayout { NHWC, NCHW }

    // ---- 复用缓冲（降低峰值内存）----
    private var reuseTile: Bitmap? = null
    private var reusePixels: IntArray? = null

    /** 取消标志：分块推理时每块之间检查，为 true 则提前终止当前分块循环 */
    @Volatile
    private var cancelFlag: Boolean = false

    /** 最近一次抠图的未硬化结果（保留半透明边缘 alpha，供查看器后处理去色边/柔化使用）。
     *  常规 resultBitmap 是硬化后的(alpha 0/255)，后处理无中间 alpha 可处理；
     *  保留这份未硬化版即可让后处理真正生效。关闭查看器后可置 null 释放。 */
    @Volatile
    var lastSoftResult: Bitmap? = null

    /** 请求取消当前进行中的分块推理（线程安全；单次推理无法中断，分块之间生效） */
    fun cancelCurrentRun() { cancelFlag = true }

    /** 模型是否已加载 */
    val isLoaded: Boolean get() = session != null

    /** 加载模型（线程安全，可重复调用） */
    fun load() {
        if (session != null) return
        // ★★ QNN/NNAPI 健壮策略（不靠"关闭回避"，靠"预检判定 + 真尝试 + 失败自动降级 CPU"）：
        //   1) QNN：HTP 硬件硬性要求 QDQ 量化 + 静态 shape。
        //      - EPContext 形态（qnn_*.onnx + .bin）：已验证可跑，必尝试
        //      - QDQ 量化模型（int8/uint8/quant/s8w/16a8w）：值得尝试，失败自动降级
        //      - FP32/FP16 普通模型：HTP 不支持（在线编译会 native abort），预检判定不适用 → 走 CPU/NNAPI
        //   2) NNAPI：QDQ 量化子图会让 partitioner 在 createSession 时 native abort，
        //      预检判定不适用 → 纯 CPU。
        //   3) 任何加速器失败（可捕获异常）→ 自动重建纯 CPU session，绝不闪退。
        // EPContext 模型：是否已部署配套 .bin（onnx 内 ep_cache_context 引用 bin，embed_mode=0 要求 onnx/bin 同目录同名）
        // ★ 兼容三类 QNN 部署名：qnn_modnet_*、qnn_rmbg14_*、qnn_animeseg_*（App 规范化命名统一 qnn_ 前缀），
        //   只要当前选中模型配置了 deployBinName 且 bin 已就位，即按 EPContext 处理走 QNN HTP。
        // ★ 本地导入（zip/onnx）的 EPContext 模型：文件名是 model_local_*.onnx，虽不以 qnn_ 开头，
        //   但 onnx 内部含 EPContext 标记且配套 .bin 已就位（zip 导入时自动部署），同样按 QNN EPContext 处理。
        // ★ 兼容开发机导出：ep_cache_context 可能是绝对路径，extractEpCacheContext 会提取 basename，
        //   ★★ ORT 1.30 强制 ep_cache_context 必须是相对路径（绝对路径直接 createSession 失败
        //       ORT_INVALID_GRAPH: "External mode should set ep_cache_context field with a relative path"）。
        //       因此在加载前自动规范化：把本地 EPContext onnx 内嵌的绝对/子目录路径改写为 basename
        //       （zip 导入时已复制 bin 为 basename，embed_mode=0 下 runtime 按 basename 在 onnx 同目录找 bin）。
        //       这样旧版已导入的模型（内嵌仍是绝对路径）无需重新导入，加载时自动修复。
        val localBinRef = if (modelFile.name.startsWith("model_local_")) {
            com.rmbg.offline.ml.ModelManager.normalizeEpContextPath(modelFile)
            com.rmbg.offline.ml.ModelManager.extractEpCacheContext(modelFile)
        } else null
        val isQnnEpContextModel =
            (modelFile.name.startsWith("qnn_") && com.rmbg.offline.ml.ModelManager.qnnContextBinFileFor()?.exists() == true) ||
                (localBinRef != null && File(com.rmbg.offline.ml.ModelManager.modelDir(), localBinRef).exists())
        val isQdqModel = listOf("qnn", "qdq", "int8", "uint8", "quant", "s8w", "16a8w")
            .any { modelFile.name.contains(it, ignoreCase = true) }
        // QNN 是否值得尝试（只有 HTP 理论上能吃的模型才试，其余预检判定不适用）
        val qnnWorthTry = enableQnn && (isQnnEpContextModel || isQdqModel)
        val nnapiWorthTry = enableNnapi && !isQdqModel
        val qnnSkipReason = when {
            !enableQnn -> "未开启 QNN"
            isQnnEpContextModel -> ""
            isQdqModel -> ""
            else -> "HTP 需 QDQ 量化模型，当前模型不适用"
        }
        if (enableQnn && qnnSkipReason.isNotEmpty()) {
            android.util.Log.w("RMBG-ENGINE", "QNN 预检不适用: $qnnSkipReason (${modelFile.name})")
        }
        if (enableNnapi && !nnapiWorthTry) {
            android.util.Log.w("RMBG-ENGINE", "NNAPI 预检不适用: QDQ 量化模型 (${modelFile.name})")
        }
        val e = OrtEnvironment.getEnvironment()
        // ★ QNN 诊断文件日志：写到 modelDir/qnn_debug.log，便于用 run-as 读取
        val dbgFile = File(com.rmbg.offline.ml.ModelManager.modelDir(), "qnn_debug.log")
        fun dbg(msg: String) {
            val ts = System.currentTimeMillis()
            val line = "[${java.text.SimpleDateFormat("HH:mm:ss.SSS", java.util.Locale.getDefault()).format(java.util.Date(ts))}] $msg\n"
            try { dbgFile.appendText(line) } catch (_: Exception) {}
            android.util.Log.i("RMBG-ENGINE", msg)
        }
        try { dbgFile.writeText("=== QNN Debug Log ${java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", java.util.Locale.getDefault()).format(java.util.Date())} ===\n") } catch (_: Exception) {}
        dbg("load() start, enableQnn=$enableQnn, qnnWorthTry=$qnnWorthTry, enableNnapi=$enableNnapi, nnapiWorthTry=$nnapiWorthTry, modelFile=${modelFile.absolutePath}")
        dbg("modelFile exists=${modelFile.exists()} size=${if (modelFile.exists()) modelFile.length() else -1}")
        // SessionOptions：参考 TagAI 的内存优化配置（可重建：加速器失败时降级纯 CPU 重试）
        fun buildOptions(useNnapi: Boolean, useQnn: Boolean): OrtSession.SessionOptions {
            return OrtSession.SessionOptions().apply {
            setMemoryPatternOptimization(false)
            addConfigEntry("session.use_env_allocators", "1")
            setIntraOpNumThreads(threads.coerceIn(1, 8))
            setInterOpNumThreads(1)
            // ★ 纯 CPU 路径开图优化（对齐 Kortex ALL_OPT）：QNN/NNAPI 路径保持默认，
            //   避免对离线编译的 EPContext 图融合造成冲突。
            if (!useNnapi && !useQnn) {
                try { setOptimizationLevel(ai.onnxruntime.OrtSession.SessionOptions.OptLevel.ALL_OPT) } catch (_: Exception) {}
            }
            // 诊断：QNN verbose 日志（0=VERBOSE）
            try { setSessionLogVerbosityLevel(0) } catch (_: Exception) {}
            // ONNX 加速：NNAPI 硬件加速（实验性，算子不支持时自动回退 CPU）
            if (useNnapi) {
                try {
                    addNnapi()
                    nnapiActive = true
                    nnapiStatus = "NNAPI 已启用"
                } catch (e: Exception) {
                    nnapiActive = false
                    nnapiStatus = "NNAPI 不可用，已回退 CPU（${e.message ?: "未知错误"}）"
                    dbg("NNAPI 不可用: ${e.message}")
                }
            } else {
                nnapiActive = false
                nnapiStatus = if (isQdqModel) "QDQ 量化模型不适用 NNAPI" else "未启用（设置中可打开）"
            }
            // QNN 加速：Qualcomm Hexagon HTP DSP（ORT 1.30.0 插件机制：registerExecutionProviderLibrary + getEpDevices）
            if (useQnn) {
                try {
                    // ★ Android 上 addQnn() 无效（libonnxruntime.so 未编译 QNN EP，addQnn 报 not supported）
                    // 正确方式：用 1.28+ 插件机制注册 libonnxruntime_providers_qnn.so
                    // ★★ 插件路径必须用 nativeLibDir（/data/app/.../lib/arm64）！
                    //   Android linker 的 class loader namespace 只允许从 nativeLibraryDir dlopen native 库，
                    //   从外部存储（/storage/emulated/0/Android/data/...）dlopen 会报
                    //   "not accessible for the namespace clns-9"。
                    //   QNN runtime AAR 已把插件 + 全套 stub/skel 装进 nativeLibraryDir。
                    val pluginPath = when {
                        nativeLibDir != null -> {
                            val f = File(nativeLibDir, "libonnxruntime_providers_qnn.so")
                            if (f.exists()) f else null
                        }
                        else -> null
                    }
                    dbg("QNN pluginPath=$pluginPath exists=${pluginPath?.exists() ?: false} size=${if (pluginPath?.exists() == true) pluginPath.length() else -1}")
                    if (pluginPath == null || !pluginPath.exists()) {
                        throw RuntimeException("QNN EP 插件未找到: $pluginPath（请先在模型页部署 QNN EPContext）")
                    }
                    // ★★ 关键：设置 ADSP_LIBRARY_PATH 指向 QNN 库所在目录。
                    //   QNN HTP backend 通过它 dlopen libQnnHtpV73Stub.so（stub→skel），
                    //   正确做法：指向 ApplicationInfo.nativeLibraryDir（QNN runtime AAR 解压的 native lib 目录）。
                    val adspCandidates = listOfNotNull(nativeLibDir)
                    var adspSet = ""
                    for (cand in adspCandidates) {
                        val f = File(cand)
                        if (f.isDirectory && f.listFiles()?.any { it.name.contains("libQnnHtp") } == true) {
                            try {
                                android.system.Os.setenv("ADSP_LIBRARY_PATH", cand, true)
                                adspSet = cand
                                dbg("ADSP_LIBRARY_PATH=$cand (set OK, contains libQnnHtp*)")
                                break
                            } catch (e3: Throwable) {
                                dbg("ADSP_LIBRARY_PATH=$cand set FAILED: ${e3.message}")
                            }
                        } else {
                            dbg("ADSP_LIBRARY_PATH candidate skipped (no libQnnHtp*): $cand")
                        }
                    }
                    if (adspSet.isEmpty()) {
                        // 兜底：仍指向 modelDir（哪怕没有 libQnnHtp，也至少让日志可诊断）
                        android.system.Os.setenv("ADSP_LIBRARY_PATH", com.rmbg.offline.ml.ModelManager.modelDir().absolutePath, true)
                        dbg("ADSP_LIBRARY_PATH fallback=modelDir (no libQnnHtp* found)")
                    }
                    // ★ 某些 OEM 构建只有 CPU 类注册设备；ORT_QNN_ENABLE_CPU_BACKEND 让插件暴露句柄，
                    //   backend_type 仍选择实际 HTP backend（官方教程做法）
                    try { android.system.Os.setenv("ORT_QNN_ENABLE_CPU_BACKEND", "1", true); dbg("ORT_QNN_ENABLE_CPU_BACKEND=1 (set OK)") } catch (_: Throwable) {}
                    // 列出 modelDir 下所有 .so 文件
                    val soFiles = com.rmbg.offline.ml.ModelManager.modelDir().listFiles()?.filter { it.name.endsWith(".so") }?.map { "${it.name}(${it.length()})" } ?: emptyList()
                    dbg("modelDir .so files: $soFiles")
                    if (nativeLibDir != null) {
                        val nativeSos = File(nativeLibDir).listFiles()?.filter { it.name.endsWith(".so") }?.map { "${it.name}(${it.length()})" } ?: emptyList()
                        dbg("nativeLibDir .so files (${nativeLibDir}): $nativeSos")
                    }
                    // 1. 注册插件库（QNNExecutionProvider 注册名 + 插件 .so 绝对路径）
                    // 先检查是否已注册（同一进程重复注册会报 "already registered"）
                    val existingQnn = try { e.epDevices.any { it.epName == "QNNExecutionProvider" } } catch (_: Exception) { false }
                    dbg("existingQnn=$existingQnn")
                    if (!existingQnn) {
                        e.registerExecutionProviderLibrary("QNNExecutionProvider", pluginPath.absolutePath)
                        dbg("QNN EP 插件注册成功: ${pluginPath.absolutePath}")
                    } else {
                        dbg("QNN EP 插件已注册，复用现有设备")
                    }
                    // 2. 找到 QNN EP 设备
                    val qnnDevices = e.epDevices.filter { it.epName == "QNNExecutionProvider" }
                    dbg("qnnDevices count=${qnnDevices.size}")
                    if (qnnDevices.isEmpty()) {
                        throw RuntimeException("QNN EP 设备未找到（插件已注册但无设备）")
                    }
                    // 诊断：打印 QNN 设备的元数据 / 选项（看插件暴露了哪些默认 option）
                    try {
                        val dev = qnnDevices.first()
                        dbg("QNN EP 设备: ${dev.epName}, vendor=${dev.epVendor}")
                        dbg("QNN EP metadata: ${dev.epMetadata}")
                        dbg("QNN EP options: ${dev.epOptions}")
                    } catch (derr: Exception) {
                        dbg("QNN EP 设备信息读取失败: ${derr.message}")
                    }
                    // 3. 把 QNN 设备加到 SessionOptions
                    //    ★★ 官方正确 provider options：backend_type=htp（而不是 backend_path 指向自拷库）
                    //    QNN runtime AAR 会安装 libQnnHtp.so 到 nativeLibraryDir，backend 按类型自动发现；
                    //    htp_performance_mode=burst 加速；graph_finalization_optimization_mode=3 更优图。
                    //    soc_model/htp_arch 不传 → backend 自动检测 SoC（读 /sys/devices/soc0/soc_id），
                    //    EPContext 缓存里已固化 soc_model/htp_arch，运行时无需再设。
                    val qnnOptions = mapOf(
                        "backend_type" to "htp",
                        "htp_performance_mode" to "burst",
                        "htp_graph_finalization_optimization_mode" to "3",
                        "offload_graph_io_quantization" to "0"
                    )
                    // ★★ 只通过 device-level API 添加 QNN EP，且只传 1 个设备。
                    //   此前同时用了反射 addExecutionProvider("QNN", options) + device-level addExecutionProvider(qnnDevices)
                    //   + 手动 addConfigEntry("ep.qnn.*")，三路叠加导致 session 出现 2 个 QNNExecutionProvider，
                    //   EPContext 节点匹配歧义报 "not compatible with any execution provider"。
                    //   修复：去掉反射和手动 config entries，只走 device-level 单设备路径。
                    try {
                        addExecutionProvider(listOf(qnnDevices.first()), qnnOptions)
                        dbg("device-level addExecutionProvider([single QNN device], options) - OK")
                    } catch (devErr: Exception) {
                        dbg("device-level addExecutionProvider failed: ${devErr.message}")
                        throw devErr
                    }
                    // 列出所有 EP
                    try {
                        val allEps = e.epDevices.map { "${it.epName}(${it.epVendor})" }
                        dbg("all EP devices: $allEps")
                    } catch (_: Exception) {}
                    qnnActive = true
                    qnnStatus = "QNN (HTP DSP) 已启用"
                } catch (e2: Exception) {
                    qnnActive = false
                    qnnStatus = "QNN 不可用，已回退 CPU（${e2.message ?: "未知错误"}）"
                    dbg("QNN 不可用，回退 CPU: ${e2.message}")
                    e2.printStackTrace()
                }
            } else {
                qnnActive = false
                qnnStatus = if (qnnSkipReason.isNotEmpty()) "QNN 未启用: $qnnSkipReason" else "未启用（设置中可打开）"
            }
            }
        }

        // 逐级降级：QNN(+NNAPI) → 纯 CPU；任一失败自动重建 session，绝不闪退
        var useQnn = qnnWorthTry
        var useNnapi = nnapiWorthTry && !useQnn // QNN 优先，避免双加速器互抢
        // ★★ EPContext 模型（QNN 专用，onnx 内嵌 QNNExecutionProvider 节点）只能用 QNN EP 加载；
        //   纯 CPU session 无法解析其 EPContext 节点 → 抛 ORT_NOT_IMPLEMENTED。
        //   因此此类模型 QNN 失败时【不能降级纯 CPU】，直接报错，避免"抠图失败: ORT_NOT_IMPLEMENTED"。
        val isEpContextOnly = isQnnEpContextModel
        var options = buildOptions(useNnapi, useQnn)
        // ★★ EPContext 专用模型：若 QNN EP 添加失败（qnnActive=false，options 里无 QNN EP），
        //   绝不能用纯 CPU session 去加载（会撞 EPContext 节点报 ORT_NOT_IMPLEMENTED），
        //   直接在此抛出明确错误提示，而不是等到 createSession 报晦涩的 ORT 错误。
        if (isEpContextOnly && !qnnActive) {
            dbg("EPContext 模型但 QNN EP 未就绪（qnnActive=$qnnActive, qnnStatus=$qnnStatus），直接报错不尝试 CPU")
            try { options.close() } catch (_: Exception) {}
            throw IllegalStateException("QNN 不可用（${qnnStatus}），EPContext 模型必须使用 QNN EP 才能加载")
        }
        dbg("createSession start, useQnn=$useQnn, useNnapi=$useNnapi, isEpContextOnly=$isEpContextOnly, modelFile=${modelFile.absolutePath}")
        var s: OrtSession? = null
        var attempt = 0
        while (true) {
            try {
                s = e.createSession(modelFile.absolutePath, options)
                break
            } catch (csErr: Exception) {
                attempt++
                dbg("createSession FAILED (attempt=$attempt, qnn=$useQnn, nnapi=$useNnapi): ${csErr.message}")
                if (isEpContextOnly || csErr.message?.contains("EPContext") == true) {
                    // ★ EPContext 专用模型（或异常本身含 EPContext 标记）：
                    //   QNN 是唯一可用 EP，失败即报错，绝不降级 CPU（降级会报 ORT_NOT_IMPLEMENTED）
                    dbg("EPContext 模型 QNN createSession 失败（不可降级 CPU）: ${csErr.message}")
                    qnnActive = false
                    qnnStatus = "QNN 创建失败: ${csErr.message ?: "未知错误"}"
                    try { options.close() } catch (_: Exception) {}
                    throw csErr
                }
                if (useQnn || useNnapi) {
                    // 加速器失败 → 降级：关闭全部加速器，纯 CPU 重试（托底）
                    dbg("加速器创建失败，降级纯 CPU 重试...")
                    useQnn = false
                    useNnapi = false
                    qnnActive = false
                    qnnStatus = "QNN 创建失败，已降级 CPU（${csErr.message ?: "未知错误"}）"
                    nnapiActive = false
                    nnapiStatus = "NNAPI 创建失败，已降级 CPU"
                    try { options.close() } catch (_: Exception) {}
                    options = buildOptions(false, false)
                    if (attempt >= 2) { dbg("降级已达上限，放弃重试"); throw csErr }
                    continue
                }
                dbg("纯 CPU createSession 也失败，抛出: ${csErr.message}")
                throw csErr
            }
        }
        val finalSession = s ?: throw IllegalStateException("createSession 返回空 session")
        dbg("createSession OK, session=${finalSession != null}")
        env = e
        session = finalSession

        // 从模型元数据读取输入名 / 尺寸 / 布局（TagAI 同款逻辑）
        finalSession.inputNames.firstOrNull()?.let { inputName = it }
        finalSession.inputInfo[inputName]?.info?.let { info ->
            val shape = (info as? ai.onnxruntime.TensorInfo)?.shape
            if (shape != null && shape.size == 4) {
                when {
                    shape[1] == 3L -> {
                        inputLayout = InputLayout.NCHW
                        val spatial = listOf(shape[2], shape[3]).firstOrNull { it > 0 } ?: 1024L
                        inputSize = spatial.toInt()
                    }
                    shape[3] == 3L -> {
                        inputLayout = InputLayout.NHWC
                        val spatial = listOf(shape[1], shape[2]).firstOrNull { it > 0 } ?: 1024L
                        inputSize = spatial.toInt()
                    }
                    else -> {
                        val spatial = shape.firstOrNull { it > 3 } ?: 1024L
                        inputSize = spatial.toInt()
                    }
                }
            }
        }
        // ★ 该 rmbg14_sdk250 量化模型的正确推理约定就是标准 RGB 输入 + 原生输出(已含 Sigmoid/概率)，
        //   在 Linux 上用 onnxruntime 标准加载即可出正确 mask。安卓端不应做任何通道交换或额外 sigmoid，
        //   否则会破坏模型结果（此前误加 BGR/sigmoid 导致背景错乱、整图变暗）。保持纯标准。
        bgrInput = false
        sigmoidOutput = false

        // ★ 预热（对齐 Kortex 首次推理流畅）：创建 session 后立刻用空白输入跑一次推理，
        //   把 QNN HTP 权重加载/图编译/算子初始化开销前移到模型加载阶段（此时通常有进度提示），
        //   避免用户点"抠图"后才卡那几百 ms。失败静默，不影响主流程。
        try {
            val warmShape = if (inputLayout == InputLayout.NHWC)
                longArrayOf(1, inputSize.toLong(), inputSize.toLong(), 3)
            else
                longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
            val warmBuf = ByteBuffer.allocateDirect(inputSize * inputSize * 3 * 4)
                .order(ByteOrder.nativeOrder()).asFloatBuffer()
            // 全零即可，无需真实像素
            warmBuf.put(FloatArray(inputSize * inputSize * 3)).rewind()
            val warmTensor = OnnxTensor.createTensor(e, warmBuf, warmShape)
            try { finalSession.run(mapOf(inputName to warmTensor)) } finally { warmTensor.close() }
            dbg("warm-up OK (inputSize=${inputSize}, layout=${inputLayout})")
        } catch (wuErr: Exception) {
            dbg("warm-up 跳过（不影响后续）: ${wuErr.message}")
        }
    }

    /** 释放 session（★ 不关 env：OrtEnvironment.getEnvironment() 是进程级全局单例，
     *   RmbgOnnxEngine 与 SuperResEngine 共用；关闭它会销毁 EPContext 运行环境，
     *   之后超分/QNN 任一方再建 session 即报 ORT_NOT_IMPLEMENTED "EPContext ... not compatible"） */
    fun close() {
        closeRequested = true
        try { session?.close() } catch (_: Exception) {}
        session = null
        env = null
        closeRequested = false
    }

    // ★ 峰值内存保护：一次处理的最大像素数。超过则先把超大图等比降采样到该上限内再抠，
    //   完成后把透明结果放大回原尺寸（牺牲极端大图的一点边缘精度，换取稳定不 OOM）。
    //   4000×4000≈16M px；降采样后 alphaFull+weightFull 峰值被限制在 16M×8B≈128MB 以内。
    private val maxWorkingPixels: Long = 16_000_000L

    /**
     * 执行抠图
     * @param input 原始图片（任意尺寸）
     * @param threshold 前景阈值 0~1（默认 0.5，越低保留越多边缘）
     * @param tileMode 拆分抠图模式：true=大图分块推理保留细节，false=整体缩放到 inputSize
     * @return 透明背景 PNG Bitmap（与原图同尺寸）
     */
    fun removeBackground(
        input: Bitmap,
        threshold: Float = 0.5f,
        tileMode: Boolean = false,
        onProgress: ((Int, Int) -> Unit)? = null
    ): Bitmap {
        // ★ 运行锁：标记本次推理进行中（供 onStop 判断延迟 close，避免 native session 被释放后仍被调用闪退）
        isRunning = true
        // 若 close 已被请求（onStop 触发），等待的推理不应再跑
        if (closeRequested) throw java.util.concurrent.CancellationException("引擎已停止")
        try {
        // ★★ 峰值内存保护：原图超限则降采样到工作上限内抠图，完成后放大回原尺寸
        val w0 = input.width
        val h0 = input.height
        val px0 = w0.toLong() * h0.toLong()
        var work = input
        var downscaled = false
        if (px0 > maxWorkingPixels) {
            val scale = kotlin.math.sqrt(maxWorkingPixels.toDouble() / px0)
            val nw = (w0 * scale).toInt().coerceAtLeast(1)
            val nh = (h0 * scale).toInt().coerceAtLeast(1)
            work = Bitmap.createScaledBitmap(input, nw, nh, true)
            downscaled = true
            // 内存保护日志（未用 load 内的局部 dbg，避免跨作用域调用）
            android.util.Log.d("RMBGEngine", "内存保护：大图 ${w0}x${h0}(${px0}px) 降采样到 ${nw}x${nh} 抠图，完成后再放大回原尺寸")
        }

        // 大图自动分块精细推理（>inputSize 触发，保留细节）；小图整体推理
        val isLarge = work.width > inputSize || work.height > inputSize
        val result = if (!isLarge) {
            removeBackgroundSingle(work, threshold)
        } else {
            removeBackgroundTiled(work, threshold, onProgress)
        }

        // 若降采样过，把透明结果等比放大回原图尺寸
        if (downscaled) {
            val restored = Bitmap.createScaledBitmap(result, w0, h0, true)
            // ★ 同步放大未硬化副本到原尺寸（供查看器后处理）
            val softRestored = lastSoftResult?.let { Bitmap.createScaledBitmap(it, w0, h0, true) }
            if (result !== work) result.recycle()
            lastSoftResult?.recycle()
            work.recycle()
            lastSoftResult = softRestored
            return restored
        }
        return result
        } finally {
            // ★ 推理结束：释放运行锁；若期间 onStop 请求了延迟 close，则此刻真正关闭
            isRunning = false
            if (closeRequested) { close() }
        }
    }

    /** 整体推理（等比 letterbox 到 1:1，避免非方形图拉伸变形） */
    private fun removeBackgroundSingle(input: Bitmap, threshold: Float): Bitmap {
        val s = session ?: throw IllegalStateException("Model not loaded")
        val e = env ?: throw IllegalStateException("Env not loaded")

        // ★ 停止检查：整体推理无法中断 native run，但至少在预处理前快速响应取消
        if (cancelFlag) throw java.util.concurrent.CancellationException("抠图已停止")

        // ---- 预处理：等比 letterbox 到 inputSize×inputSize（居中、白底填充），内容不变形 ----
        val size = inputSize
        val scale = minOf(size.toFloat() / input.width, size.toFloat() / input.height)
        val contentW = maxOf(1, (input.width * scale).toInt())
        val contentH = maxOf(1, (input.height * scale).toInt())
        val padL = (size - contentW) / 2
        val padT = (size - contentH) / 2
        val square = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val cv = android.graphics.Canvas(square)
        cv.drawColor(android.graphics.Color.WHITE) // 白底
        val scaled = Bitmap.createScaledBitmap(input, contentW, contentH, true)
        cv.drawBitmap(scaled, padL.toFloat(), padT.toFloat(), null)
        scaled.recycle()
        val tensorData = preprocess(square)
        square.recycle()

        // ---- 推理 ----
        val shape = if (inputLayout == InputLayout.NHWC)
            longArrayOf(1, size.toLong(), size.toLong(), 3)
        else
            longArrayOf(1, 3, size.toLong(), size.toLong())
        val inputTensor = OnnxTensor.createTensor(e, tensorData, shape)
        val output = try {
            s.run(mapOf(inputName to inputTensor))
        } finally {
            inputTensor.close()
        }

        // ---- 读取输出（TagAI 同款方式：遍历 iterator 取底层数据）----
        val probArray = extractOutputProbs(output)
        output.close()

        // ---- 后处理：概率 → alpha mask → 原图合成（按 letterbox content 区域映射，避免拉伸错位）----
        // ★ 默认不硬化：保留半透明边缘 alpha（背景硬化作为查看器后处理的可选项，不再自动做）
        lastSoftResult = postprocess(input, probArray, size, threshold, padL, padT, contentW, contentH, harden = false)
        return lastSoftResult ?: postprocess(input, probArray, size, threshold, padL, padT, contentW, contentH, harden = false)
    }

    /**
     * 拆分抠图：大图切成多个 inputSize×inputSize 块分别推理（串行，控制内存），
     * 边缘重叠 OVERLAP 像素做线性渐变融合，再拼回原尺寸。
     * 相比整体缩放，能保留更多细节、更精确的边缘。
     *
     * @param onProgress 进度回调 (已完成块数, 总块数)，在调用线程执行
     */
    private fun removeBackgroundTiled(input: Bitmap, threshold: Float, onProgress: ((Int, Int) -> Unit)? = null): Bitmap {
        val s = session ?: throw IllegalStateException("Model not loaded")
        val e = env ?: throw IllegalStateException("Env not loaded")
        val size = inputSize
        val overlap = 64
        val w = input.width
        val h = input.height
        cancelFlag = false // 重置取消标志（本次分块推理可被 cancelCurrentRun 打断）

        // 计算分块网格
        val cols = ((w - 1) / (size - overlap) + 1).coerceAtLeast(1)
        val rows = ((h - 1) / (size - overlap) + 1).coerceAtLeast(1)

        // 每块在原图中的起始位置
        val tileXs = IntArray(cols)
        val tileYs = IntArray(rows)
        for (c in 0 until cols) {
            val x = c * (size - overlap)
            tileXs[c] = if (x + size > w) w - size else x
            tileXs[c] = tileXs[c].coerceAtLeast(0)
        }
        for (r in 0 until rows) {
            val y = r * (size - overlap)
            tileYs[r] = if (y + size > h) h - size else y
            tileYs[r] = tileYs[r].coerceAtLeast(0)
        }

        // 全尺寸 alpha 累积（FloatArray：加权累计）
        // ★ 修复缝隙：旧版 alpha 按 p·wt 加权、weight 却恒定 +1，归一化 alpha/count 把
        //   重叠区削成一半 → 接缝变暗/透明。这里 weight 累计 wt，归一化 alpha/Σwt；
        //   重叠区两侧 wt 之和 = 1，得到真实加权均值 → 无缝融合。
        // ★ 内存优化：alpha 用 FloatArray；weight 用 ByteArray（权重归一化到 0..255 存储），
        //   峰值从 w*h*8B 降到 w*h*5B（4000×3000：96MB→60MB）。
        val alphaFull = FloatArray(w * h)
        val weightFull = ByteArray(w * h)

        val totalTiles = cols * rows
        var doneTiles = 0

        // 复用裁剪位图（避免每块 new Bitmap，显著降内存）
        val tile = reuseTile ?: Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888).also { reuseTile = it }

        // 逐块推理（串行，防内存爆）
        for (r in 0 until rows) {
            for (c in 0 until cols) {
                // ★ 停止检查：每块开始前若收到取消请求则立即终止（直接杀死，不保留半成品）
                if (cancelFlag) break
                val ox = tileXs[c]
                val oy = tileYs[r]

                // 裁剪块（复用 tile 位图）
                val canvas = android.graphics.Canvas(tile)
                canvas.drawBitmap(input, -ox.toFloat(), -oy.toFloat(), null)

                // 预处理 + 推理
                val tensorData = preprocess(tile)
                val shape = if (inputLayout == InputLayout.NHWC)
                    longArrayOf(1, size.toLong(), size.toLong(), 3)
                else
                    longArrayOf(1, 3, size.toLong(), size.toLong())
                val inputTensor = OnnxTensor.createTensor(e, tensorData, shape)
                val output = try {
                    s.run(mapOf(inputName to inputTensor))
                } finally {
                    inputTensor.close()
                }
                val probs = extractOutputProbs(output)
                output.close()

                // 概率图写入全尺寸 alpha + 权重（边缘 64px 重叠区线性渐变融合）
                // ★ 越界保护：QNN EPContext 的 HTP 编译输出 shape 可能与 CPU 不同（padding/裁剪），
                //   probs.size 可能 < size²，越界索引会抛 ArrayIndexOutOfBoundsException
                for (ty in 0 until size) {
                    for (tx in 0 until size) {
                        val gx = ox + tx
                        val gy = oy + ty
                        if (gx >= w || gy >= h) continue
                        val idx = gy * w + gx
                        val pi = ty * size + tx
                        val p = if (pi < probs.size) probs[pi] else 0f
                        // 边缘 64px 线性衰减权重（中心 1 → 边界 0；重叠区两侧 wt 和 = 1）
                        val wx = minOf(tx, size - 1 - tx, overlap).toFloat() / overlap
                        val wy = minOf(ty, size - 1 - ty, overlap).toFloat() / overlap
                        val wt = minOf(wx, wy).coerceIn(0f, 1f)
                        alphaFull[idx] += p.coerceIn(0f, 1f) * wt
                        // ★ ByteArray 存储：字节累加（0..255 表示 0..1，允许溢出饱和）
                        val prev = weightFull[idx].toInt() and 0xFF
                        val next = (prev + (wt * 255).toInt()).coerceAtMost(0xFF)
                        weightFull[idx] = next.toByte()
                    }
                }
                doneTiles++
                onProgress?.invoke(doneTiles, totalTiles)
            }
        }

        // 归一化：加权均值 alpha = Σ(p·wt) / Σwt
        for (i in alphaFull.indices) {
            // ★ ByteArray 权重转回 0..1（0..255 表示 0..1）
            val wsum = (weightFull[i].toInt() and 0xFF) / 255f
            if (wsum > 0f) {
                alphaFull[i] = (alphaFull[i] / wsum).coerceIn(0f, 1f)
            }
        }

        // 阈值化 → 生成透明 PNG（按行处理，避免整图大 IntArray 峰值翻倍）
        // ★ 与区域抠图一致：低置信背景(半透明残影)硬化为全透明，消除磨砂雾感；
        //   主体与边缘(高alpha)保留柔和渐变
        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        // ★ 默认不硬化：保留全部半透明边缘 alpha（背景硬化作为查看器后处理的可选项，不再自动做）
        // ★ 越界修复：reusePixels 是引擎级共享缓存，仅为空时创建→若之前用小图抠过(长度偏小)，
        //   现在载入大图(行宽 w 更大)复用它会越界崩溃。必须按当前行宽检查重建。
        val rp = reusePixels
        val src = if (rp == null || rp.size < w) IntArray(w).also { reusePixels = it } else rp
        val dst = IntArray(w)
        for (y in 0 until h) {
            input.getPixels(src, 0, w, 0, y, w, 1)
            for (x in 0 until w) {
                val i = y * w + x
                val a = alphaFull[i]
                val alphaByte = probToAlpha(a, threshold)
                val color = src[x]
                val r = (color shr 16) and 0xFF
                val g = (color shr 8) and 0xFF
                val b = color and 0xFF
                // 默认不硬化：保留全部半透明边缘（背景硬化作为查看器后处理选项）
                dst[x] = Color.argb(alphaByte, r, g, b)
            }
            out.setPixels(dst, 0, w, 0, y, w, 1)
        }
        lastSoftResult = out // 未硬化结果，供查看器后处理
        return out
    }

    /**
     * 局部重新抠图（选区）：把整图交给模型推理（保留全局上下文，边缘更准），
     * 但只把选区内的新 mask 覆盖到当前结果上，选区外保留现有结果。
     * @param region 原图像素坐标区域 [left, top, right, bottom]
     */
    fun removeBackgroundRegion(
        input: Bitmap,
        threshold: Float = 0.5f,
        region: IntArray, // [left, top, right, bottom]
        currentResult: Bitmap? = null
    ): Bitmap {
        val left = region[0].coerceIn(0, input.width - 1)
        val top = region[1].coerceIn(0, input.height - 1)
        val right = region[2].coerceIn(left + 1, input.width)
        val bottom = region[3].coerceIn(top + 1, input.height)
        val rw = right - left
        val rh = bottom - top
        if (rw <= 0 || rh <= 0) return currentResult ?: input

        // ★ 尺寸保护：currentResult 若与 input 尺寸不一致，视为 null（否则 getPixels 越界崩溃）
        val base = if (currentResult != null &&
            currentResult.width == input.width &&
            currentResult.height == input.height) currentResult else null

        // ★★★ 关键修复：推理必须用「不透明版原图」，绝不能把透明图直接给模型。
        //   若 input 本身是透明 PNG（如把抠图结果重新载入再精修），透明区的 RGB 是残留色(白或黑)，
        //   模型会把整片透明区误判成背景 → 推理结果把透明区重新填成不透明。
        //   做法：在不透明背景(白)上合成 input 得到 inferSrc(干净 RGB)；选区内的 RGB 也一律从 inferSrc 取，
        //   避免用透明原图的残留色(黑/白)导致选区内主体边缘出现黑边/白边。
        val hasTransparency = input.hasAlpha()
        val inferSrc = if (hasTransparency) {
            val op = Bitmap.createBitmap(input.width, input.height, Bitmap.Config.ARGB_8888)
            val c = android.graphics.Canvas(op)
            c.drawColor(android.graphics.Color.WHITE) // 白色不透明底
            c.drawBitmap(input, 0f, 0f, null)
            op
        } else input

        // ★ 整图推理（用不透明版原图：大图自动分块、小图整体缩放，得到全图 alpha mask）
        //   好处：选区内的主体有完整上下文，人物边缘即使跨过选区边界也能正确识别，
        //   不会像"只裁剪选区"那样因局部上下文不足而把边缘误判。
        val fullRmbg = removeBackground(inferSrc, threshold, tileMode = true)
        // fullRmbg 是整图抠好的透明 PNG（RGB 来自干净白底合成版，alpha=抠图结果）

        // 拼接回整图（逐行处理，避免整图 IntArray×3 峰值 144MB → 行缓冲 48KB）
        //   选区内 RGB 用 inferSrc(干净背景)，alpha 用推理结果；选区外保留 base(当前透明结果)
        val w = input.width
        val h = input.height
        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val rowIn = IntArray(w)      // 干净合成版 RGB（选区内用）
        val rowCur = IntArray(w)     // 当前结果（选区外用）
        val rowFull = IntArray(w)    // 整图推理结果（取 alpha）
        val rowDst = IntArray(w)
        for (y in 0 until h) {
            inferSrc.getPixels(rowIn, 0, w, 0, y, w, 1)
            fullRmbg.getPixels(rowFull, 0, w, 0, y, w, 1)
            if (base != null) base.getPixels(rowCur, 0, w, 0, y, w, 1)
            for (x in 0 until w) {
                val inRegion = x >= left && x < right && y >= top && y < bottom
                if (inRegion) {
                    // ★ 选区内：alpha 取推理结果。清除两类残影：
                    //   1) alpha 低于 bgCleanAlpha 的半透明残影
                    //   2) 白底合成版带来的白色残影：alpha 未满(非不透明) 且 RGB 接近纯白(白底)
                    //      这种是"透明背景在白底上合成"产生的白色块，必须透明掉
                    val a = (rowFull[x] ushr 24) and 0xFF
                    val pxRgb = rowIn[x] and 0x00FFFFFF
                    if (a < bgCleanAlpha) {
                        // ★★★ 遮罩模型核心：抠空只清 alpha，RGB 保留原图色！
                        //   旧版写 0x00000000 把 RGB 一起清 0 → 恢复取不回原色（恢复失效根因）。
                        //   透明像素 RGB 保留 inferSrc 原图色，恢复画笔取回的就是原背景色。
                        rowDst[x] = pxRgb // alpha=0, RGB=原图色
                    } else {
                        // 检查是否为白色残影：RGB 三分量都高(接近白) 且 alpha 非满(未真正不透明)
                        val r2 = (pxRgb shr 16) and 0xFF
                        val g2 = (pxRgb shr 8) and 0xFF
                        val b2 = pxRgb and 0xFF
                        val nearWhite = r2 >= 235 && g2 >= 235 && b2 >= 235
                        if (nearWhite && a < 255) {
                            // 白底残影：透明（RGB 保留原图色）
                            rowDst[x] = pxRgb
                        } else {
                            rowDst[x] = (a shl 24) or pxRgb // 真前景/边缘：保留
                        }
                    }
                } else {
                    // ★ 选区外：仅保留当前结果(base)；无当前结果(base=null)时用原图 rowIn(不透明)，
                    //   保证「只抠选区内」，选区外保持原样。绝不能退回 rowFull(整图已被抠成透明)——那会让选区外也被抠掉。
                    rowDst[x] = if (base != null) rowCur[x] else rowIn[x]
                }
            }
            out.setPixels(rowDst, 0, w, 0, y, w, 1)
        }
        if (hasTransparency) inferSrc.recycle()
        fullRmbg.recycle()
        return out
    }

    /**
     * 结果背景强度事后调整（无需重新抠图）：
     * 对已生成的透明结果，把 alpha 低于 strength 的半透明残影硬化为全透明，
     * 保留主体/边缘的柔和渐变。与抠图时的 bgCleanAlpha（作用于 alpha mask 前）互补，
     * 这里直接重映射结果像素的 alpha 通道，适合"抠完再微调背景干净度"。
     * 返回新的透明 PNG（不修改原图，可逆）。
     */
    fun applyBgStrength(result: Bitmap, strength: Int): Bitmap {
        val w = result.width
        val h = result.height
        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val row = IntArray(w)
        val dst = IntArray(w)
        val s = strength.coerceIn(0, 255)
        for (y in 0 until h) {
            result.getPixels(row, 0, w, 0, y, w, 1)
            for (x in 0 until w) {
                val p = row[x]
                val a = (p ushr 24) and 0xFF
                // ★★★ 遮罩模型核心：低 alpha 残影只清 alpha，RGB 保留（恢复取回原色）
                dst[x] = if (a < s) (p and 0x00FFFFFF) else p
            }
            out.setPixels(dst, 0, w, 0, y, w, 1)
        }
        return out
    }

    /** 单次推理返回概率数组（inputSize×inputSize） */
    private fun inferProbs(bitmap: Bitmap): FloatArray {
        val s = session ?: throw IllegalStateException("Model not loaded")
        val e = env ?: throw IllegalStateException("Env not loaded")
        val size = inputSize
        val tensorData = preprocess(bitmap)
        val shape = if (inputLayout == InputLayout.NHWC)
            longArrayOf(1, size.toLong(), size.toLong(), 3)
        else
            longArrayOf(1, 3, size.toLong(), size.toLong())
        val inputTensor = OnnxTensor.createTensor(e, tensorData, shape)
        val output = try {
            s.run(mapOf(inputName to inputTensor))
        } finally {
            inputTensor.close()
        }
        val probs = extractOutputProbs(output)
        output.close()
        return probs
    }

    /** 将 inputSize×inputSize 概率图缩放到 w×h */
    private fun scaleProbsToSize(probs: FloatArray, size: Int, w: Int, h: Int): FloatArray {
        val out = FloatArray(w * h)
        for (y in 0 until h) {
            for (x in 0 until w) {
                val sx = (x * size / w).coerceIn(0, size - 1)
                val sy = (y * size / h).coerceIn(0, size - 1)
                val pi = sy * size + sx
                out[y * w + x] = if (pi < probs.size) probs[pi] else 0f
            }
        }
        return out
    }

    /**
     * 从 OrtSession.Result 提取概率数组。
     * 参考 TagAI 的 extractBestScores + flattenOutputValues：
     * 遍历 result 的 iterator，entry.value.value 是底层数据对象，
     * 可能是 FloatArray / DoubleArray / IntArray / LongArray 等，统一展平。
     */
    private fun extractOutputProbs(results: OrtSession.Result): FloatArray {
        val values = mutableListOf<Float>()
        val iterator = results.iterator()
        while (iterator.hasNext()) {
            val entry = iterator.next()
            flattenOutputValues(entry.value.value, values)
        }
        if (values.isEmpty()) {
            throw IllegalStateException("No output values from ONNX model")
        }
        // 输出是 1x1x1024x1024 概率图，取整个数组
        val arr = values.toFloatArray()
        // rmbg14_sdk250 原生输出是 logits，需过 sigmoid 才得到 [0,1] 前景概率，
        // 否则人物内部(logits 小正)被直接当概率压到阈值下，抠空成"线条画"。
        if (sigmoidOutput) {
            for (i in arr.indices) arr[i] = 1f / (1f + Math.exp(-arr[i].toDouble()).toFloat())
        }
        return arr
    }

    /** 递归展平各种 ONNX 输出类型（TagAI flattenOutputValues 同款） */
    private fun flattenOutputValues(value: Any?, out: MutableList<Float>) {
        when (value) {
            is FloatArray -> out.addAll(value.toList())
            is DoubleArray -> value.forEach { out.add(it.toFloat()) }
            is IntArray -> value.forEach { out.add(it.toFloat()) }
            is LongArray -> value.forEach { out.add(it.toFloat()) }
            is Array<*> -> value.forEach { flattenOutputValues(it, out) }
            is Number -> out.add(value.toFloat())
        }
    }

    /** 归一化到 [0,1]，CHW 或 NHWC 布局（按模型实际要求） */
    private fun preprocess(bitmap: Bitmap): FloatBuffer {
        val size = inputSize
        val pixels = IntArray(size * size)
        bitmap.getPixels(pixels, 0, size, 0, 0, size, size)

        val data = FloatArray(size * size * 3)
        if (inputLayout == InputLayout.NHWC) {
            // NHWC: R,G,B 连续排
            for (i in pixels.indices) {
                val p = pixels[i]
                val r = ((p shr 16) and 0xFF) / 255f
                val g = ((p shr 8) and 0xFF) / 255f
                val b = (p and 0xFF) / 255f
                if (bgrInput) {
                    data[i * 3] = b
                    data[i * 3 + 1] = g
                    data[i * 3 + 2] = r
                } else {
                    data[i * 3] = r
                    data[i * 3 + 1] = g
                    data[i * 3 + 2] = b
                }
            }
        } else {
            // NCHW: 先所有 R，再所有 G，再所有 B
            for (i in pixels.indices) {
                val p = pixels[i]
                val r = ((p shr 16) and 0xFF) / 255f
                val g = ((p shr 8) and 0xFF) / 255f
                val b = (p and 0xFF) / 255f
                if (bgrInput) {
                    data[i] = b
                    data[size * size + i] = g
                    data[2 * size * size + i] = r
                } else {
                    data[i] = r
                    data[size * size + i] = g
                    data[2 * size * size + i] = b
                }
            }
        }
        val buffer = ByteBuffer.allocateDirect(data.size * 4).order(ByteOrder.nativeOrder())
        val fb = buffer.asFloatBuffer()
        fb.put(data)
        fb.rewind()
        return fb
    }

    /** 概率 [0,1] → alpha 字节 [0,255]（阈值化）
     *  a >= threshold → 255（不透明）；否则按比例淡出。
     *  threshold=0 时任何 a>0 都全不透明（保留最多），避免除零。 */
    private fun probToAlpha(a: Float, threshold: Float): Int {
        val t = threshold.coerceIn(0f, 1f)
        if (a >= t) return 255
        if (t <= 0f) return (a * 255).toInt().coerceIn(0, 255)
        return (a / t * 255).toInt().coerceIn(0, 255)
    }

    /** 概率图 → alpha mask，与原图合成（按行处理，降低峰值内存）
 *  @param contentL/contentT/contentW/contentH 推理图(1:1)中实际内容所在的几何；若非 letterbox(拉伸旧路径)传 0,0,size,size
 *  @param harden 是否应用 bgCleanAlpha 背景硬化（把半透明背景砍成真透明、消除磨砂雾感）。
 *     true=常规抠图结果（alpha 趋向 0/255，显示干净）；false=保留全部半透明边缘（供查看器后处理用）。 */
    private fun postprocess(
        original: Bitmap,
        probs: FloatArray,
        size: Int,
        threshold: Float,
        contentL: Int = 0,
        contentT: Int = 0,
        contentW: Int = size,
        contentH: Int = size,
        harden: Boolean = true
    ): Bitmap {
// 1. 构建 alpha 图（缩放到 size）
        val alpha = FloatArray(size * size)
        for (i in 0 until minOf(probs.size, size * size)) {
            alpha[i] = probs[i]
        }
        // 2. 把 content 区域等比映射到原图尺寸（不做拉伸：原图 content 比例=推理 content 比例）
        val w = original.width
        val h = original.height
        val alphaFull = FloatArray(w * h)
        // content 映射：原图整幅对应推理图的 content 矩形
        val scaleX = contentW.toFloat() / w
        val scaleY = contentH.toFloat() / h
        for (y in 0 until h) {
            for (x in 0 until w) {
                val sx = (contentL + (x * scaleX).toInt()).coerceIn(contentL, (contentL + contentW - 1).coerceAtMost(size - 1))
                val sy = (contentT + (y * scaleY).toInt()).coerceIn(contentT, (contentT + contentH - 1).coerceAtMost(size - 1))
                val pi = sy * size + sx
                alphaFull[y * w + x] = if (pi < alpha.size) alpha[pi] else 0f
            }
        }

        // 3. 阈值化 → 生成透明 PNG（按行写，避免整图大数组）
        // ★ 背景残影硬化：alpha 低于下限(bgCleanAlpha)的像素完全透明(RGB 清零)，去掉磨砂雾感
        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val src = reusePixels ?: IntArray(w).also { reusePixels = it }
        val dst = IntArray(w)
        for (y in 0 until h) {
            original.getPixels(src, 0, w, 0, y, w, 1)
            for (x in 0 until w) {
                val i = y * w + x
                val a = alphaFull[i]
                val alphaByte = probToAlpha(a, threshold)
                val color = src[x]
                val r = (color shr 16) and 0xFF
                val g = (color shr 8) and 0xFF
                val b = color and 0xFF
                dst[x] = if (harden && alphaByte < bgCleanAlpha) {
                    // ★★★ 遮罩模型核心：硬化只清 alpha，RGB 保留原图色（恢复取回原色）
                    (r shl 16) or (g shl 8) or b
                } else {
                    // 保留柔和过渡（含半透明边缘；harden=false 时完整保留供后处理去色边/柔化）
                    Color.argb(alphaByte, r, g, b)
                }
            }
            out.setPixels(dst, 0, w, 0, y, w, 1)
        }
        return out
    }

    /**
     * 边缘后处理（作用在已生成的透明结果像素上，无需重新抠图，可逆）：
     * 综合做三类边缘优化：
     *  1) 去色边 decontaminate[0~1]：去掉边缘残留背景色(白边/黑边/彩边)。
     *     原理：对 alpha 0<a<255 的边缘像素，已知背景色 bg，还原未被污染的前景色：
     *     fg = (rgb - (1-a)*bg) / a。强度按 decontaminate 线性混合。
     *  2) 柔化 feather[半径px]：对边缘 alpha 做高斯平滑，让硬边过渡柔和（0=关闭）。
     *  3) 收缩/扩展 shrink[px, 负=扩展]：对 alpha 做形态学腐蚀(+)/膨胀(-)，去白边常收缩。
     * @param bgColor 背景色 ARGB（去色边用），如白色 0xFFFFFFFF
     */
    fun postProcess(
        result: Bitmap,
        decontaminate: Float = 0f,
        feather: Int = 0,
        shrink: Int = 0,
        bgColor: Int = -1,  // 0xFFFFFFFF 白色 ARGB
        harden: Boolean = false // ★ 背景硬化（后处理选项）：alpha<bgCleanAlpha 的半透明残影设为全透明，去磨砂雾
    ): Bitmap {
        val w = result.width
        val h = result.height
        val n = w * h
        val pixels = IntArray(n)
        result.getPixels(pixels, 0, w, 0, 0, w, h)
        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)

        // ---- 阶段0：背景硬化（可选，后处理面板开关）----
        if (harden) {
            val ha = bgCleanAlpha
            for (i in 0 until n) {
                val a = (pixels[i] ushr 24) and 0xFF
                // ★★★ 遮罩模型核心：硬化只清 alpha，RGB 保留原图色（恢复取回原色）
                if (a < ha) pixels[i] = pixels[i] and 0x00FFFFFF
            }
        }

        val bgR = (bgColor shr 16) and 0xFF
        val bgG = (bgColor shr 8) and 0xFF
        val bgB = bgColor and 0xFF
        val dc = decontaminate.coerceIn(0f, 1f)

        // ---- 阶段1：逐像素去色边（改 RGB，保留 alpha）----
        if (dc > 0f) {
            for (i in 0 until n) {
                val p = pixels[i]
                val a = (p ushr 24) and 0xFF
                if (a in 1..254) { // 仅边缘半透明像素
                    val ar = a / 255f
                    val r = ((p shr 16) and 0xFF).toFloat()
                    val g = ((p shr 8) and 0xFF).toFloat()
                    val b = (p and 0xFF).toFloat()
                    // 还原前景色：fg = (rgb - (1-a)*bg) / a
                    val fr = (r - (1f - ar) * bgR) / ar
                    val fg = (g - (1f - ar) * bgG) / ar
                    val fb = (b - (1f - ar) * bgB) / ar
                    // 与原色按强度混合（避免过度拉伸噪点）
                    val nr = (fr * dc + r * (1f - dc)).coerceIn(0f, 255f).toInt()
                    val ng = (fg * dc + g * (1f - dc)).coerceIn(0f, 255f).toInt()
                    val nb = (fb * dc + b * (1f - dc)).coerceIn(0f, 255f).toInt()
                    pixels[i] = (a shl 24) or (nr shl 16) or (ng shl 8) or nb
                }
            }
        }

        // ---- 阶段2：柔化（对 alpha 做高斯平滑，仅边缘）----
        if (feather > 0) {
            val src = pixels.copyOf()
            val radius = feather.coerceIn(1, 8)
            val kernel = IntArray((2 * radius + 1) * (2 * radius + 1))
            var ksum = 0
            for (dy in -radius..radius) for (dx in -radius..radius) {
                val g = (radius - kotlin.math.abs(dx)) * (radius - kotlin.math.abs(dy)) + 1
                kernel[(dy + radius) * (2 * radius + 1) + (dx + radius)] = g
                ksum += g
            }
            for (y in 0 until h) {
                for (x in 0 until w) {
                    val a = (src[y * w + x] ushr 24) and 0xFF
                    if (a in 1..254) {
                        var acc = 0
                        var wsum = 0
                        for (dy in -radius..radius) {
                            val yy = y + dy
                            if (yy !in 0 until h) continue
                            for (dx in -radius..radius) {
                                val xx = x + dx
                                if (xx !in 0 until w) continue
                                val k = kernel[(dy + radius) * (2 * radius + 1) + (dx + radius)]
                                acc += ((src[yy * w + xx] ushr 24) and 0xFF) * k
                                wsum += k
                            }
                        }
                        val na = (acc / wsum).coerceIn(0, 255)
                        pixels[y * w + x] = (na shl 24) or (pixels[y * w + x] and 0x00FFFFFF)
                    }
                }
            }
        }

        // ---- 阶段3：收缩/扩展（形态学，对 alpha）----
        if (shrink != 0) {
            val src = pixels.copyOf()
            val rad = kotlin.math.abs(shrink).coerceIn(1, 20)
            val erode = shrink > 0 // 收缩=腐蚀
            for (y in 0 until h) {
                for (x in 0 until w) {
                    var mn = 255
                    var mx = 0
                    for (dy in -rad..rad) {
                        val yy = y + dy
                        if (yy !in 0 until h) continue
                        for (dx in -rad..rad) {
                            val xx = x + dx
                            if (xx !in 0 until w) continue
                            val a = (src[yy * w + xx] ushr 24) and 0xFF
                            if (a < mn) mn = a
                            if (a > mx) mx = a
                        }
                    }
                    val na = if (erode) mn else mx
                    pixels[y * w + x] = (na shl 24) or (pixels[y * w + x] and 0x00FFFFFF)
                }
            }
        }

        out.setPixels(pixels, 0, w, 0, 0, w, h)
        return out
    }

    companion object {
        const val INPUT_SIZE = 1024
    }
}