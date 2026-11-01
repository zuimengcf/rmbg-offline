package com.rmbg.offline.ml
import com.rmbg.offline.R
import com.rmbg.offline.OperitApp

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.ceil
import kotlin.math.min

/**
 * Real-ESRGAN anime6B 4x 超分引擎（CPU 或 QNN HTP 加速）。
 *
 * 设计要点（基于实测验证 v4）：
 *  1) 白底压平输入：RGB = 前景色*a + 255*(1-a)，让模型在亮背景工作，边缘混合出亮过渡（不产生黑边/黑点）。
 *  2) alpha 通道单独双线性放大（边缘平滑），不喂给超分模型脑补。
 *  3) 保护性还原：fg = (out - 255*(1-a)) / a，但只对 alpha>=25 像素做除法还原；
 *     alpha<25 的极淡残影直接清透明（避免除以小 alpha 放大噪声 → 这是"黑点/黑边"的根源）。
 *
 * 加载策略：
 *  - 从文件加载（导入的超分模型：普通 ONNX 走纯 CPU；EPContext(QNN) 走 QNN HTP 加速）。
 *  - 从 assets 字节加载（内置默认模型，纯 CPU，兼容旧调用）。
 *  - QNN EPContext 的 bin 必须与 onnx 同目录（embed_mode=0），动态读 input/output 名与 shape。
 *
 * 内存保护：输入边长>512 先等比降采样（4x 输出边长<=2048），防 OOM。
 */
class SuperResEngine(
    private val threads: Int = 4,
    private val modelFile: File? = null,
    private val modelBytes: ByteArray? = null,
    private val nativeLibDir: String? = null
) {
    companion object {
        /** 模型输入期望 0~1 归一化（Real-ESRGAN ONNX 导出约定） */
        private const val INPUT_SCALE = 1f / 255f
        /** 保护性还原阈值：alpha 低于此值的像素不除法还原，直接清透明 */
        private const val PROTECT_ALPHA = 25
        /** 超分安全工作上限：输入边长>此值先降采样（4x 输出边长上限 = 上限*4） */
        const val MAX_INPUT_SIDE = 512
        /** 极速模式（缩小图片）上限：更小输入 ≥ 更快，牺牲细节换速度（4x 输出≤1280） */
        const val TURBO_MAX_INPUT_SIDE = 320
        /** 分块推理的分块重叠量（源像素，与抠图 removeBackgroundTiled 一致整倍数） */
        private const val TILE_OVERLAP = 64
        /** 分块推理输出像素上限（4x 输出 ≤ 此值才走分块，防超长超大图 OOM；4096²≈16.7M px） */
        const val MAX_TILED_OUT_PIXELS = 16_000_000L

        /** 从 assets 加载模型文件字节（内置默认，纯 CPU） */
        fun loadFromAssets(context: Context, assetPath: String): ByteArray? {
            return try {
                context.assets.open(assetPath).use { it.readBytes() }
            } catch (_: Exception) { null }
        }

        /** 从文件加载导入的超分模型（普通 ONNX=CPU / EPContext=QNN），自动检测。 */
        fun loadFromFile(context: Context, file: File, threads: Int = 4): SuperResEngine {
            return SuperResEngine(
                threads = threads,
                modelFile = file,
                nativeLibDir = context.applicationInfo.nativeLibraryDir
            )
        }
    }

    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var inputName: String = "input"
    private var outputName: String? = null

    /** 模型固定输入宽高（QNN EPContext 常为静态 512×512；动态 shape 时为 0） */
    private var fixedInputW: Int = 0
    private var fixedInputH: Int = 0
    /** 是否 EPContext(QNN) 模型 */
    @Volatile
    var isQnn: Boolean = false
        private set
    @Volatile
    var qnnActive: Boolean = false
        private set

    @Volatile
    var isLoaded: Boolean = false
        private set

    /** 加载模型（失败时抛异常，由调用方决定降级）。
     *  ★ onnxruntime-android 1.30 只有进程级全局单例 OrtEnvironment.getEnvironment()
     *   （无 createEnvironment，按 name 复用同一原生 env）。所有引擎（QNN 抠图、超分）
     *   共用这个单例：各自只持有并关闭自己的 OrtSession，绝不关闭 env。
     *   若任一引擎关闭 env（进程唯一），其余引擎再建 session 会因原生环境被销毁而失败
     *   （QNN EPContext 报 ORT_NOT_IMPLEMENTED "not compatible"；超分/CPU 报错建不起来）。 */
    fun load() {
        if (session != null) return
        val e = OrtEnvironment.getEnvironment()

        // ---- 判断是否为 EPContext(QNN) 模型：onnx 内嵌 ep_cache_context 引用 bin ----
        val isEpContext = modelFile != null &&
            modelFile.exists() &&
            ModelManager.extractEpCacheContext(modelFile) != null
        isQnn = isEpContext

        // QNN EPContext 必须从文件加载（bin 与 onnx 同目录），且需要 QNN 运行时插件。
        if (isEpContext) {
            // ★ 规范化 ep_cache_context 为相对路径（绝对路径 createSession 会失败）
            try { ModelManager.normalizeEpContextPath(modelFile!!) } catch (_: Exception) {}
        }

        val opts = OrtSession.SessionOptions().apply {
            setMemoryPatternOptimization(false)
            addConfigEntry("session.use_env_allocators", "1")
            setIntraOpNumThreads(threads.coerceIn(1, 8))
            setInterOpNumThreads(1)
            if (isEpContext) {
                // ---- QNN 路径：注册插件 + ADSP_LIBRARY_PATH + device-level EP（在 apply 块内调用 SessionOptions 方法）----
                try {
                    val pluginPath = if (nativeLibDir != null) {
                        val f = File(nativeLibDir, "libonnxruntime_providers_qnn.so")
                        if (f.exists()) f else null
                    } else null
                    if (pluginPath == null || !pluginPath.exists()) {
                        throw RuntimeException(OperitApp.appContext.getString(R.string.qnn_5))
                    }
                    // ADSP_LIBRARY_PATH 指向 nativeLibDir（含 libQnnHtp*）
                    if (nativeLibDir != null && File(nativeLibDir).listFiles()?.any { it.name.contains("libQnnHtp") } == true) {
                        try { android.system.Os.setenv("ADSP_LIBRARY_PATH", nativeLibDir, true) } catch (_: Throwable) {}
                    }
                    try { android.system.Os.setenv("ORT_QNN_ENABLE_CPU_BACKEND", "1", true) } catch (_: Throwable) {}
                    // 注册插件（进程级只注册一次）
                    val existingQnn = try { e.epDevices.any { it.epName == "QNNExecutionProvider" } } catch (_: Exception) { false }
                    if (!existingQnn) {
                        e.registerExecutionProviderLibrary("QNNExecutionProvider", pluginPath.absolutePath)
                    }
                    val qnnDevices = e.epDevices.filter { it.epName == "QNNExecutionProvider" }
                    if (qnnDevices.isEmpty()) {
                        throw RuntimeException(OperitApp.appContext.getString(R.string.qnn_6))
                    }
                    // ★★ 必须传全部 QNN 设备（ORT_QNN_ENABLE_CPU_BACKEND=1 下暴露 CPU READY + HTP 两个设备，
                    //   只取 first() 会取到错误 backend 导致 EPContext "not compatible"）
                    addExecutionProvider(qnnDevices, mapOf(
                        "backend_type" to "htp",
                        "htp_performance_mode" to "burst",
                        "htp_graph_finalization_optimization_mode" to "3",
                        "offload_graph_io_quantization" to "0"
                    ))
                    qnnActive = true
                } catch (qErr: Exception) {
                    qnnActive = false
                    android.util.Log.w("RMBG-SUPER", "QNN EP 配置失败，超分将回退纯 CPU: ${qErr.message}")
                    // EPContext 模型 QNN 失败不能纯 CPU 加载（会 ORT_NOT_IMPLEMENTED），抛错
                    close()  // 关闭 apply 中的部分资源（无 session，仅安全）
                    throw IllegalStateException("超分 QNN 模型加载失败（${qErr.message ?: "未知错误"}）。请确认设备支持 QNN 或改用 CPU 超分模型")
                }
            } else {
                // ---- CPU 路径：普通 ONNX 模型 ----
                try { setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT) } catch (_: Exception) {}
            }
        }

        val s = try {
            if (modelFile != null) {
                e.createSession(modelFile.absolutePath, opts)
            } else if (modelBytes != null) {
                e.createSession(modelBytes, opts)
            } else {
                throw IllegalStateException(OperitApp.appContext.getString(R.string.str_307))
            }
        } catch (ex: Exception) {
            try { opts.close() } catch (_: Exception) {}
            throw ex
        }

        // 动态读取输入/输出名与 shape
        s.inputNames.firstOrNull()?.let { inputName = it }
        s.outputNames.firstOrNull()?.let { outputName = it }
        try {
            val info = s.inputInfo[inputName]?.info as? TensorInfo
            val shape = info?.shape
            if (shape != null && shape.size == 4 && shape[2] > 0 && shape[3] > 0) {
                fixedInputH = shape[2].toInt()
                fixedInputW = shape[3].toInt()
            }
        } catch (_: Exception) {}

        env = e
        session = s
        isLoaded = true
    }

    /**
     * 4x 超分带透明通道的抠图结果。
     * 大图自动降采样：输入边长>512 先等比例缩到 512 以内（4x 输出<=2048），防 OOM。
     * @return 新位图（4x 尺寸，ARGB_8888）
     */
    fun upscale4x(src: Bitmap): Bitmap = upscale4x(src, tileMode = false, turboMode = false)

    /**
     * 4x 超分带透明通道的抠图结果，支持分块推理与极速模式（抠图/超分共用语义）。
     *
     * 模式分派（与抠图 removeBackground 的 tileMode/turboMode 对齐）：
     *  - turboMode = true（极速模式·缩小图片）：优先降采样到 turboLevel 目标边长再整张推理，
     *    缩得越小越快（128最快/最糊，768最慢/最接近原细节），保证最大图也不 OOM。
     *  - tileMode = true && !turbo（精细·大图分块）：把原图按 tileLevel（2/4/9/16 块）目标分块，
     *    逐块 4x 推理，融合拼接保细节——大图不降采样也能出全细节，代价是慢（分块越多越慢）。
     *  - 都 false：默认行为=输入边长>MAX_INPUT_SIDE(512) 先等比降采样再整张推理（原逻辑）。
     *
     * @param tileLevel 分块档位（2/4/9/16），仅当 tileMode=true 生效
     * @param turboLevel 极速档位（128/256/512/768），仅当 turboMode=true 生效
     * @return 新位图（4x 尺寸，ARGB_8888）。
     */
    fun upscale4x(src: Bitmap, tileMode: Boolean, turboMode: Boolean, tileLevel: Int = 4, turboLevel: Int = 256): Bitmap {
        if (turboMode) {
            return upscaleTurbo(src, turboLevel)
        }
        if (tileMode) {
            // ★ 分块只对大图有意义（单块就装得下的直接走整张，避免额外开销与接缝）；
            //   QNN 固定输入模型切块代价高且几何会失真，也走整张兜底。
            //   分块输出超 MAX_TILED_OUT_PIXELS 上限时 upscaleTiled 抛 IllegalArgumentException，
            //   捕获后降级整张路径（整张有降采样保护，内存可控）。
            val tooSmall = src.width <= MAX_INPUT_SIDE && src.height <= MAX_INPUT_SIDE
            if (!tooSmall && fixedInputW <= 0 && fixedInputH <= 0) {
                try {
                    return upscaleTiled(src, tileLevel)
                } catch (_: IllegalArgumentException) {
                    // fall through → 整张路径
                }
            }
        }
        return upscaleWhole(src)
    }

    /** 极速模式路径：先等比缩到 turboLevel 目标最长边（128/256/512/768），再整张 4x 超分后放大还原。 */
    private fun upscaleTurbo(src: Bitmap, turboLevel: Int): Bitmap {
        ensureLoaded()
        val w0 = src.width
        val h0 = src.height
        val maxSide = maxOf(w0, h0)
        val side = turboLevel.coerceIn(128, 768)
        // 已足够小（≤side）直接整张超分，避免无意义放大
        var work = src
        if (maxSide > side && !(fixedInputW > 0 && fixedInputH > 0)) {
            val scale = side.toFloat() / maxSide
            val nw = (w0 * scale).toInt().coerceAtLeast(128)
            val nh = (h0 * scale).toInt().coerceAtLeast(128)
            work = Bitmap.createScaledBitmap(src, nw, nh, true)
        }
        val small = upscaleWhole(work)
        // 极速：把 4x 结果放大回「原图边长×4」，保持输出尺寸与整张一致（缩放损失可接受）
        if (work !== src) { try { work.recycle() } catch (_: Exception) {} }
        if (small.width == w0 * 4 && small.height == h0 * 4) return small
        val restored = Bitmap.createScaledBitmap(small, w0 * 4, h0 * 4, true)
        if (small !== restored) { try { small.recycle() } catch (_: Exception) {} }
        return restored
    }

    /** 分块路径：把原图按 tileLevel（2/4/9/16 块）目标分块网格切块，逐块 4x 推理，重叠融合拼接。 */
    private fun upscaleTiled(src: Bitmap, tileLevel: Int): Bitmap {
        ensureLoaded()
        val w = src.width
        val h = src.height
        val overlap = TILE_OVERLAP

        // 由目标分块总数推导行列网格（对齐抠图 removeBackgroundTiled）
        val root = kotlin.math.sqrt(tileLevel.toDouble()).toInt()
        val cols: Int
        val rows: Int
        if (root * root == tileLevel) {
            cols = root; rows = root
        } else {
            if (w >= h) { cols = 2; rows = 1 } else { cols = 1; rows = 2 }
        }
        val tileW = maxOf(64, (w + (cols - 1) * overlap + cols - 1) / cols)
        val tileH = maxOf(64, (h + (rows - 1) * overlap + rows - 1) / rows)

        // 每块起始坐标（均匀步进 tileW - overlap，最后一块强制贴右/下边缘：
        // 与抠图 removeBackgroundTiled 同一修复——旧公式 w/cols 会让右/下边缘漏一块 overlap 带）
        val xs = IntArray(cols)
        for (c in 0 until cols) {
            var x = c * (tileW - overlap)
            if (c == cols - 1 || x + tileW > w) x = w - tileW
            xs[c] = x.coerceAtLeast(0)
        }
        val ys = IntArray(rows)
        for (r in 0 until rows) {
            var y = r * (tileH - overlap)
            if (r == rows - 1 || y + tileH > h) y = h - tileH
            ys[r] = y.coerceAtLeast(0)
        }

        // 输出尺寸（4x）
        val ow = w * 4
        val oh = h * 4
        // ★ 内存上限检查：MAX_TILED_OUT_PIXELS 已定义却从未引用（旧代码直接分配，
        //   2000×1500 4x → 输出 48M px，rgbaFull+weightFull ≈ 960MB → 必 OOM）。
        //   超限抛异常，调用方（upscale4x 分派器）捕获后降级到整张路径。
        require(ow.toLong() * oh <= MAX_TILED_OUT_PIXELS) {
            "Tiled output ${ow}x$oh exceeds MAX_TILED_OUT_PIXELS=$MAX_TILED_OUT_PIXELS"
        }
        val rgbaFull = FloatArray(ow * oh * 4)   // RGBA 累积（含 alpha 权重，按 4 分量加权）
        val weightFull = FloatArray(ow * oh)     // 权重累积（重叠带 Σwt 可 >1）

        // 逐块 4x 推理
        for (r in 0 until rows) {
            for (c in 0 until cols) {
                val ox = xs[c]
                val oy = ys[r]
                val tw = minOf(tileW, w - ox).coerceAtLeast(1)
                val th = minOf(tileH, h - oy).coerceAtLeast(1)
                // 裁剪块
                val srcRect = android.graphics.Rect(ox, oy, minOf(ox + tileW, w), minOf(oy + tileH, h))
                val tileBmp = Bitmap.createBitmap(src, srcRect.left, srcRect.top, srcRect.width(), srcRect.height())
                val up = upscaleWhole(tileBmp) // 4x：尺寸 tw*4 × th*4
                tileBmp.recycle()

                // 块输出像素与全图输出的坐标映射（重叠带权重）
                val hasL = c > 0; val hasR = c < cols - 1; val hasT = r > 0; val hasB = r < rows - 1
                val upW = up.width
                val upH = up.height
                // ★ 动态 CPU 模型的 upscaleWhole 对 >512px 的 tile 会先降采样再 4x，
                //   up 实际尺寸 = 降采样尺寸×4 < tw*4×th*4。旧代码按 dy*4+sy / dx*4+sx
                //   直接索引 → 越界全部 continue → 缺块/透明区域。
                //   修复：按 up 实际尺寸与块应有尺寸 (tw*4, th*4) 的比例做坐标映射，
                //   降采样融合会损失该块部分细节（由覆盖同一区域的邻块补偿），但位置正确、不缺块。
                val scaleX = upW.toFloat() / (tw * 4)
                val scaleY = upH.toFloat() / (th * 4)
                val upPx = IntArray(upW * upH)
                up.getPixels(upPx, 0, upW, 0, 0, upW, upH)
                up.recycle()

                for (dy in 0 until th) {
                    val gy = oy + dy
                    // 垂直权重（源像素 overlap 带，相对 4x 前）
                    var wty = 1f
                    if (hasT && dy < overlap) wty = dy.toFloat() / overlap
                    if (hasB && th - 1 - dy < overlap) {
                        val rb = (th - 1 - dy).toFloat() / overlap
                        if (rb < wty) wty = rb
                    }
                    val outY = gy * 4
                    // 该源像素行对应的 up 中实际行范围（比例映射，向下取整+clamp）
                    val upY0 = (dy * 4 * scaleY).toInt().coerceIn(0, upH - 1)
                    val upY1 = ((dy + 1) * 4 * scaleY).toInt().coerceIn(upY0, upH)
                    for (dx in 0 until tw) {
                        val gx = ox + dx
                        var wtx = 1f
                        if (hasL && dx < overlap) wtx = dx.toFloat() / overlap
                        if (hasR && tw - 1 - dx < overlap) {
                            val rr = (tw - 1 - dx).toFloat() / overlap
                            if (rr < wtx) wtx = rr
                        }
                        val wt = minOf(wtx, wty).coerceIn(0f, 1f)
                        // 该源像素对应的 up 实际列范围（比例映射）
                        val upX0 = (dx * 4 * scaleX).toInt().coerceIn(0, upW - 1)
                        val upX1 = ((dx + 1) * 4 * scaleX).toInt().coerceIn(upX0, upW)
                        // 把 up 的 [upX0,upX1)×[upY0,upY1) 邻域采样写入对应 4x 输出邻域
                        for (sy in 0 until 4) {
                            val oy2 = outY + sy
                            if (oy2 >= oh) continue
                            // 在 [upY0, upY1) 内按 sy 均匀取样（区间空时取 upY0）
                            val upY = if (upY1 > upY0)
                                upY0 + (upY1 - upY0 - 1) * sy / 3
                            else upY0
                            val rowO = oy2 * ow
                            val rowU = upY * upW
                            for (sx in 0 until 4) {
                                val ox2 = gx * 4 + sx
                                if (ox2 >= ow) continue
                                val upX = if (upX1 > upX0)
                                    upX0 + (upX1 - upX0 - 1) * sx / 3
                                else upX0
                                val srcPx = upPx[rowU + upX]
                                val a = (srcPx ushr 24) and 0xFF
                                val r8 = (srcPx shr 16) and 0xFF
                                val g8 = (srcPx shr 8) and 0xFF
                                val b8 = srcPx and 0xFF
                                val oi = rowO + ox2
                                rgbaFull[oi * 4] += r8 * wt
                                rgbaFull[oi * 4 + 1] += g8 * wt
                                rgbaFull[oi * 4 + 2] += b8 * wt
                                rgbaFull[oi * 4 + 3] += a * wt
                                weightFull[oi] += wt
                            }
                        }
                    }
                }
            }
        }

        // 归一化（逐输出像素除以权重和）
        val out = Bitmap.createBitmap(ow, oh, Bitmap.Config.ARGB_8888)
        val outPx = IntArray(ow * oh)
        for (i in 0 until ow * oh) {
            val wsum = weightFull[i]
            if (wsum > 0f) {
                val r = (rgbaFull[i * 4] / wsum).coerceIn(0f, 255f).toInt()
                val g = (rgbaFull[i * 4 + 1] / wsum).coerceIn(0f, 255f).toInt()
                val b = (rgbaFull[i * 4 + 2] / wsum).coerceIn(0f, 255f).toInt()
                val a = (rgbaFull[i * 4 + 3] / wsum).coerceIn(0f, 255f).toInt()
                outPx[i] = (a shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        out.setPixels(outPx, 0, ow, 0, 0, ow, oh)
        return out
    }

    /** 整张推理路径（原 upscale4x 逻辑）。 */
    private fun upscaleWhole(src: Bitmap): Bitmap {
        ensureLoaded()
        // 大图降采样保护（给足余量：512→2048 是 4x，2048 位图约 16MB，+RGB/alpha 数组峰值可控）
        var work = src
        val maxSide = MAX_INPUT_SIDE
        if (src.width > maxSide || src.height > maxSide) {
            val scale = maxSide.toFloat() / maxOf(src.width, src.height)
            val nw = (src.width * scale).toInt().coerceAtLeast(1)
            val nh = (src.height * scale).toInt().coerceAtLeast(1)
            work = Bitmap.createScaledBitmap(src, nw, nh, true)
        }

        // ---- 输入尺寸决策 ----
        // QNN EPContext 常为静态 512×512：等比缩放内容 + 白底 letterbox 居中填满固定尺寸。
        // CPU 动态 shape 模型：直接用 work 尺寸。
        val useLetterbox = fixedInputW > 0 && fixedInputH > 0
        val inW: Int
        val inH: Int
        var contentScale = 1f        // work→模型输入的缩放（输出裁剪还原纵横比用）
        var padL = 0
        var padT = 0
        if (useLetterbox) {
            inW = fixedInputW; inH = fixedInputH
            val scale = min(inW.toFloat() / work.width, inH.toFloat() / work.height)
            contentScale = scale
            val cw = (work.width * scale).toInt().coerceAtLeast(1)
            val ch = (work.height * scale).toInt().coerceAtLeast(1)
            padL = (inW - cw) / 2
            padT = (inH - ch) / 2
            // 等比缩放 + 居中放入 letterbox 画布
            val scaled = Bitmap.createScaledBitmap(work, cw, ch, true)
            val canvas = Bitmap.createBitmap(inW, inH, Bitmap.Config.ARGB_8888)
            val c = android.graphics.Canvas(canvas)
            c.drawColor(android.graphics.Color.WHITE) // 白底 padding
            c.drawBitmap(scaled, padL.toFloat(), padT.toFloat(), null)
            // ★ 只回收降采样副本（scaled 和上一级 work，若 work 是降采样得来），
            //   绝不回收输入 src（它可能是 viewerBitmap/resultBitmap/originalBitmap，回收即导致
            //   "Canvas: trying to use a recycled bitmap" 崩溃）。
            //   注意：旧逻辑 `if (canvas !== work) work.recycle()` 在小图场景 work===src 时会误回收输入，
            //   必须先让 work 指向 canvas 再回收中间副本。
            if (scaled !== work) scaled.recycle()
            val oldWork = work          // 记录回收前的 work（可能是 src 或降采样副本）
            work = canvas               // ★ 先切换 work 到新 letterbox 画布
            if (oldWork !== src && oldWork !== canvas && !oldWork.isRecycled) {
                oldWork.recycle()
            }
        } else {
            inW = work.width; inH = work.height
        }
        val w = work.width
        val h = work.height

        val pixels = IntArray(w * h)
        work.getPixels(pixels, 0, w, 0, 0, w, h)
        if (work !== src) work.recycle()

        // ---- 构建 alpha 与白底压平 RGB ----
        val alpha = FloatArray(w * h)
        val rgb = FloatArray(w * h * 3)
        for (i in 0 until w * h) {
            val p = pixels[i]
            val a = (p ushr 24) and 0xFF
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            alpha[i] = a.toFloat()
            val ar = a / 255f
            // 白底压平
            rgb[i * 3] = r * ar + 255f * (1f - ar)
            rgb[i * 3 + 1] = g * ar + 255f * (1f - ar)
            rgb[i * 3 + 2] = b * ar + 255f * (1f - ar)
        }

        // ---- 模型推理：NCHW 1x3xHxW，0~1 ----
        // ★ 必须传 FloatBuffer（元素数 = w*h*3），传 ByteBuffer 会被 onnxruntime 按字节数计（w*h*3*4），
        //   与 shape 要求的元素数 w*h*3 不匹配 → "Shape [1,3,H,W] requires X elements but buffer has 4X" 崩溃。
        val buf = ByteBuffer.allocateDirect(w * h * 3 * 4).order(ByteOrder.nativeOrder())
        val fb = buf.asFloatBuffer()
        for (c in 0 until 3) {
            for (i in 0 until w * h) {
                fb.put(rgb[i * 3 + c] * INPUT_SCALE)
            }
        }
        fb.rewind()
        val shape = longArrayOf(1L, 3L, h.toLong(), w.toLong())
        val tensor = OnnxTensor.createTensor(env, fb, shape)
        val result = session!!.run(mapOf(inputName to tensor))
        tensor.close()

        // ---- 读输出（内存安全 + 兼容嵌套类型）----
        // ★★ 真机实测：onnxruntime-android 的 entry.value.value 对超分模型返回的往往是
        //   【嵌套 Array】（如 [1][3][H][W]），而非扁平 FloatArray。RmbgOnnxEngine 靠
        //   flattenOutputValues 的 is Array<*> 递归分支能正常抠图，正是佐证。
        //   若只处理 FloatArray/DoubleArray，其它类型被丢弃 → outRGB 全 0 → 主体黑影。
        //   ★ 但 flattenOutputValues 用 toList()/addAll 会为 2048²×3≈1200万 float 创建海量
        //   ArrayList → OOM。故这里改为【递归叶节点直接顺序填入预分配 outRGB】，
        //   既兼容嵌套 Array，又绝不过度分配中间 List。输出是 NCHW planar（RRR…GGG…BBB），
        //   outRGB 按 planar 顺序存（outRGB[0..plane)=R, [plane..2plane)=G, [2plane..3plane)=B），
        //   下游用 planar 索引（i, plane+i, 2*plane+i）读取。
        // ★ 输出尺寸：优先从输出 tensor shape 动态读取（QNN 超分为 4x 固定 2048×2048；
        //   CPU 动态模型也是 4x）。若读取失败，回退输入 4x。
        var sw = w * 4
        var sh = h * 4
        try {
            val outInfo = if (outputName != null) session!!.outputInfo[outputName]?.info as? TensorInfo else null
            val oshape = outInfo?.shape
            if (oshape != null && oshape.size == 4 && oshape[2] > 0 && oshape[3] > 0) {
                sh = oshape[2].toInt(); sw = oshape[3].toInt()
            }
        } catch (_: Exception) {}
        val plane = sw * sh
        val need = sw * sh * 3
        val outRGB = FloatArray(need)
        var fillPos = 0
        fun fillLeaves(v: Any?) {
            if (fillPos >= need) return
            when (v) {
                is FloatArray -> {
                    val n = min(v.size, need - fillPos)
                    for (j in 0 until n) outRGB[fillPos++] = v[j].coerceIn(0f, 1f) * 255f
                }
                is DoubleArray -> {
                    val n = min(v.size, need - fillPos)
                    for (j in 0 until n) outRGB[fillPos++] = v[j].toFloat().coerceIn(0f, 1f) * 255f
                }
                is IntArray -> {
                    val n = min(v.size, need - fillPos)
                    for (j in 0 until n) outRGB[fillPos++] = v[j].coerceIn(0, 255).toFloat()
                }
                is LongArray -> {
                    val n = min(v.size, need - fillPos)
                    for (j in 0 until n) outRGB[fillPos++] = v[j].coerceIn(0L, 255L).toFloat()
                }
                is Array<*> -> for (e in v) { if (fillPos >= need) break; fillLeaves(e) }
                is Number -> outRGB[fillPos++] = v.toFloat().coerceIn(0f, 1f) * 255f
                else -> { /* 其它类型忽略（不会出现） */ }
            }
        }
        val iter = result.iterator()
        while (iter.hasNext()) {
            val entry = iter.next()
            fillLeaves(entry.value.value)
        }
        result.close()

        // ---- alpha 双线性放大 ----
        // ★ 直接用双线性插值放大 alpha 数组（不依赖 ALPHA_8 位图 copyPixels，避免 Android 位图格式
        //   在 createScaledBitmap/copyPixelsFromBuffer 间的字节序/对齐差异导致 alpha 丢失变黑影）
        val aBigF = FloatArray(sw * sh)
        for (yy in 0 until sh) {
            val fy = (yy + 0.5f) * h / sh - 0.5f
            val y0 = fy.coerceIn(0f, h - 1f).toInt()
            val y1 = min(y0 + 1, h - 1)
            val wy = (fy - y0).coerceIn(0f, 1f)
            for (xx in 0 until sw) {
                val fx = (xx + 0.5f) * w / sw - 0.5f
                val x0 = fx.coerceIn(0f, w - 1f).toInt()
                val x1 = min(x0 + 1, w - 1)
                val wx = (fx - x0).coerceIn(0f, 1f)
                val p00 = alpha[y0 * w + x0]; val p01 = alpha[y0 * w + x1]
                val p10 = alpha[y1 * w + x0]; val p11 = alpha[y1 * w + x1]
                val top = p00 * (1f - wx) + p01 * wx
                val bot = p10 * (1f - wx) + p11 * wx
                aBigF[yy * sw + xx] = top * (1f - wy) + bot * wy
            }
        }

        // ---- 保护性还原 + 组装 RGBA ----
        val out = Bitmap.createBitmap(sw, sh, Bitmap.Config.ARGB_8888)
        val outPx = IntArray(sw * sh)
        for (i in 0 until sw * sh) {
            val a = aBigF[i].toInt().coerceIn(0, 255)
            val ar = a / 255f
            val safe = if (ar <= 0f) 1f else ar
            var r = (outRGB[i] - 255f * (1f - ar)) / safe
            var g = (outRGB[plane + i] - 255f * (1f - ar)) / safe
            var b = (outRGB[plane * 2 + i] - 255f * (1f - ar)) / safe
            val outA: Int
            if (a < PROTECT_ALPHA) {
                outA = 0; r = 0f; g = 0f; b = 0f
            } else {
                outA = a
            }
            outPx[i] = (outA shl 24) or
                (r.coerceIn(0f, 255f).toInt() shl 16) or
                (g.coerceIn(0f, 255f).toInt() shl 8) or
                b.coerceIn(0f, 255f).toInt()
        }
        out.setPixels(outPx, 0, sw, 0, 0, sw, sh)
        return out
    }

    private fun ensureLoaded() {
        if (session == null) load()
    }

    /** 释放 session（★ 不关 env：env 是进程级全局单例 OrtEnvironment.getEnvironment()，
     *   与 QNN 抠图引擎共用；关闭它会销毁共享原生环境，之后 QNN 再建 session
     *   即报 ORT_NOT_IMPLEMENTED "EPContext node ... not compatible"，超分也建不起来） */
    fun close() {
        try { session?.close() } catch (_: Exception) {}
        session = null
        env = null
        isLoaded = false
    }
}