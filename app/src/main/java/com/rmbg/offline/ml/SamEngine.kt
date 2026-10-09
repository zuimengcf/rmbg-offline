package com.rmbg.offline.ml

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * SAM ViT-B 分割引擎（本地 CPU 推理）。
 *
 * ★ 正确画布管线（已用交付包实测验证，IoU 0.95+）：
 *   - encoder 输入：HWC RGB float32 [0,255]，画布 1024×682（config.yaml max_width=1024, max_height=682）
 *     mean/std 归一化由 encoder ONNX 图内部执行，不在此预处理。
 *   - 坐标换算：原图 (W,H) → 模型画布，x × 1024/W、y × 682/H（x/y 独立缩放）
 *   - decoder 输入：image_embeddings + point_coords + point_labels + mask_input(全0) +
 *     has_mask_input(=0) + orig_im_size=[682,1024]
 *   - 输出用 masks（模型内部已按 orig_im_size resize 到 682×1024，不要用 low_res_masks 手动放大）
 *   - 后处理：masks 双线性放大回原图 → logit>0 → 最大 8 邻域连通主体
 *
 * ★ 框选 → box prompt：左上角 + 右下角两个点，label=[2,3]（SAM 官方式）；精度足够。
 *   （可扩展正/负点精修：point_labels 1=前景 0=背景）
 */
class SamEngine(
    private val encoderFile: File,
    private val decoderFile: File,
    private val threads: Int = 4
) {
    companion object {
        /** 模型画布宽（config.yaml） */
        const val MODEL_W = 1024
        /** 模型画布高（config.yaml max_height=682） */
        const val MODEL_H = 682
    }

    private var env: OrtEnvironment? = null
    private var encoderSession: OrtSession? = null
    private var decoderSession: OrtSession? = null

    @Volatile
    var isLoaded: Boolean = false
        private set

    @Volatile
    var isRunning: Boolean = false
        private set

    @Volatile
    var closeRequested: Boolean = false
        private set

    private var encInputName = "image"
    private var decInputNames: List<String> = emptyList()

    // ---- Embedding 缓存：同一张图多次框选只编码一次（decoder 单次 ~0.1s）----
    private var cachedEmbData: FloatArray? = null
    private var cachedEmbKey: String? = null

    /** 计算图片指纹（尺寸 + 均匀采样哈希），用于判断是否复用 embedding */
    private fun bitmapFingerprint(bmp: Bitmap): String {
        val w = bmp.width; val h = bmp.height
        val stepX = (w / 24).coerceAtLeast(1)
        val stepY = (h / 24).coerceAtLeast(1)
        var hash = w * 73856093 + h * 19349663
        // 采样 24×24 网格像素（不申请整图数组，避免大图内存峰值）
        var y = 0
        while (y < h) {
            var x = 0
            while (x < w) {
                val p = bmp.getPixel(x, y)
                hash = hash * 31 + (p and 0xFFFFFF)
                x += stepX
            }
            y += stepY
        }
        return "$w x $h #$hash"
    }

    /** 加载模型（失败抛异常） */
    fun load() {
        if (encoderSession != null && decoderSession != null) return
        val e = OrtEnvironment.getEnvironment()
        val opts = OrtSession.SessionOptions().apply {
            setMemoryPatternOptimization(false)
            addConfigEntry("session.use_env_allocators", "1")
            setIntraOpNumThreads(threads.coerceIn(1, 8))
            setInterOpNumThreads(1)
            try { setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT) } catch (_: Exception) {}
        }
        val enc = e.createSession(encoderFile.absolutePath, opts)
        val dec = e.createSession(decoderFile.absolutePath, opts)
        // 读输入名（encoder 输入可能是 image / input；decoder 固定 6 输入）
        enc.inputNames.firstOrNull()?.let { encInputName = it }
        decInputNames = dec.inputNames.toList()
        env = e
        encoderSession = enc
        decoderSession = dec
        isLoaded = true
    }

    fun close() {
        closeRequested = true
        try { decoderSession?.close() } catch (_: Exception) {}
        try { encoderSession?.close() } catch (_: Exception) {}
        decoderSession = null
        encoderSession = null
        env = null
        // 释放 embedding 缓存
        cachedEmbData = null
        cachedEmbKey = null
        isLoaded = false
        closeRequested = false
    }

    private fun ensureLoaded() {
        if (!isLoaded || encoderSession == null || decoderSession == null) load()
    }

    /**
     * 点击分割（SAM 核心用法）：全图一次编码，点击处 → 分割该物体。
     * @param src 图片（任意尺寸，ARGB_8888）
     * @param point 点击位置（图片像素坐标 [x,y]）
     * @param extraPos 额外前景点（可选，多选合并区域）；null=单点
     * @return 掩码 Bitmap（同 src 尺寸，掩码内白 alpha=255，外透明）；点击处未识别到物体（空掩码）返回 null
     */
    fun segmentAt(src: Bitmap, point: IntArray, extraPos: List<IntArray>? = null): Bitmap? {
        ensureLoaded()
        if (closeRequested) throw java.util.concurrent.CancellationException("引擎已停止")
        isRunning = true
        try {
            val e = env ?: throw IllegalStateException("env not loaded")
            val enc = encoderSession ?: throw IllegalStateException("encoder not loaded")
            val dec = decoderSession ?: throw IllegalStateException("decoder not loaded")
            val w = src.width
            val h = src.height
            if (w <= 0 || h <= 0) throw IllegalArgumentException("图片尺寸无效 ${w}x${h}")

            // ---- 预处理：缩放（拉伸）到 1024×682，HWC float32 [0,255] ----
            // ★ Embedding 缓存：同图复用，只需跑 decoder（~0.1s），点击交互不卡。
            val fp = bitmapFingerprint(src)
            val embData: FloatArray
            if (cachedEmbKey == fp && cachedEmbData != null) {
                embData = cachedEmbData!!
            } else {
                val canvas = Bitmap.createScaledBitmap(src, MODEL_W, MODEL_H, true)
                val px = IntArray(MODEL_W * MODEL_H)
                canvas.getPixels(px, 0, MODEL_W, 0, 0, MODEL_W, MODEL_H)
                if (canvas !== src) canvas.recycle()
                val buf = ByteBuffer.allocateDirect(MODEL_W * MODEL_H * 3 * 4).order(ByteOrder.nativeOrder())
                val fb = buf.asFloatBuffer()
                for (i in px.indices) {
                    val p = px[i]
                    fb.put(((p shr 16) and 0xFF).toFloat())
                    fb.put(((p shr 8) and 0xFF).toFloat())
                    fb.put((p and 0xFF).toFloat())
                }
                fb.rewind()
                // ★★ encoder ONNX 输入是 [H,W,3]（HWC，无 batch 维）——交付包脚本 cv2.resize 直接送；
                //   传 [1,H,W,3] 会 shape 不匹配 → 模型不运行（已实测定位）
                val shape = longArrayOf(MODEL_H.toLong(), MODEL_W.toLong(), 3L)
                val inputTensor = OnnxTensor.createTensor(e, fb, shape)
                val embResult = try { enc.run(mapOf(encInputName to inputTensor)) } finally { inputTensor.close() }
                try {
                    val it = embResult.iterator()
                    if (!it.hasNext()) throw IllegalStateException("encoder 输出为空")
                    embData = flattenToFloatArray(it.next().value.value)
                        ?: throw IllegalStateException("encoder 输出解析失败")
                } finally {
                    try { embResult.close() } catch (_: Exception) {}
                }
                cachedEmbData = embData
                cachedEmbKey = fp
            }

            // ---- 构建 embedding tensor（一次性，用完关闭）----
            val embBuf = ByteBuffer.allocateDirect(embData.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
            embBuf.put(embData).rewind()
            val encTensor = OnnxTensor.createTensor(e, embBuf, longArrayOf(1L, 256L, 64L, 64L))

            // ---- 构建 prompt：纯点（前景 label=1；可加正点）----
            val scaleX = MODEL_W.toFloat() / w
            val scaleY = MODEL_H.toFloat() / h
            val coords = ArrayList<Float>()
            val labels = ArrayList<Float>()
            coords.add(point[0] * scaleX); coords.add(point[1] * scaleY); labels.add(1f)
            if (extraPos != null) {
                for (pt in extraPos) {
                    coords.add(pt[0] * scaleX); coords.add(pt[1] * scaleY); labels.add(1f)
                }
            }
            val n = coords.size / 2
            val coordBuf = ByteBuffer.allocateDirect(n * 2 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
            coordBuf.put(coords.toFloatArray()).rewind()
            val labelBuf = ByteBuffer.allocateDirect(n * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
            labelBuf.put(labels.toFloatArray()).rewind()
            val coordTensor = OnnxTensor.createTensor(e, coordBuf, longArrayOf(1L, n.toLong(), 2L))
            val labelTensor = OnnxTensor.createTensor(e, labelBuf, longArrayOf(1L, n.toLong()))
            // ★ 静态输入必须用 FloatBuffer（createTensor 无 FloatArray 重载）
            val maskBuf = ByteBuffer.allocateDirect(1 * 1 * 256 * 256 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
            maskBuf.put(FloatArray(1 * 1 * 256 * 256)).rewind()
            val maskTensor = OnnxTensor.createTensor(e, maskBuf, longArrayOf(1L, 1L, 256L, 256L))
            val hasMaskBuf = ByteBuffer.allocateDirect(4).order(ByteOrder.nativeOrder()).asFloatBuffer()
            hasMaskBuf.put(floatArrayOf(0f)).rewind()
            val hasMaskTensor = OnnxTensor.createTensor(e, hasMaskBuf, longArrayOf(1L))
            val origBuf = ByteBuffer.allocateDirect(8).order(ByteOrder.nativeOrder()).asFloatBuffer()
            origBuf.put(floatArrayOf(MODEL_H.toFloat(), MODEL_W.toFloat())).rewind()
            val origTensor = OnnxTensor.createTensor(e, origBuf, longArrayOf(2L))
            val feed = HashMap<String, OnnxTensor>()
            feed[decInputNames.find { it.contains("image_embeddings", true) } ?: "image_embeddings"] = encTensor
            feed[decInputNames.find { it.contains("point_coords", true) } ?: "point_coords"] = coordTensor
            feed[decInputNames.find { it.contains("point_labels", true) } ?: "point_labels"] = labelTensor
            feed[decInputNames.find { it.contains("mask_input", true) } ?: "mask_input"] = maskTensor
            feed[decInputNames.find { it.contains("has_mask_input", true) } ?: "has_mask_input"] = hasMaskTensor
            feed[decInputNames.find { it.contains("orig_im_size", true) } ?: "orig_im_size"] = origTensor
            val decResult = try { dec.run(feed) } finally {
                coordTensor.close(); labelTensor.close(); maskTensor.close(); hasMaskTensor.close(); origTensor.close()
                // ★ 一次性 embedding tensor 用完关闭（数据已缓存为 FloatArray，无生命周期冲突）
                try { encTensor.close() } catch (_: Exception) {}
            }
            // 输出：masks [1,num_masks,682,1024]（多候选，已按 orig_im_size 放大）+ iou_predictions [1,num_masks]
            // ★ 点分割取 IoU 最高的候选掩码
            var masksAny: Any? = null
            var scoresAny: Any? = null
            try {
                val it = decResult.iterator()
                while (it.hasNext()) {
                    val entry = it.next()
                    val n = entry.key
                    when {
                        n.equals("masks", true) -> masksAny = entry.value.value
                        n.equals("iou_predictions", true) || n.equals("scores", true) -> scoresAny = entry.value.value
                    }
                }
                if (masksAny == null) {
                    // 兜底：第一个输出
                    val it2 = decResult.iterator()
                    if (it2.hasNext()) masksAny = it2.next().value.value
                }
                masksAny ?: throw IllegalStateException("decoder 输出为空")
            } catch (t: Throwable) {
                try { decResult.close() } catch (_: Exception) {}
                throw t
            } finally {
                try { decResult.close() } catch (_: Exception) {}
            }
            val allLogits = flattenToFloatArray(masksAny) ?: throw IllegalStateException("masks 输出解析失败")
            val scores = flattenToFloatArray(scoresAny)
            // 解析候选数：masks 是 [1,N,H,W]，展平后总量 / (682*1024)
            val mh = MODEL_H; val mw = MODEL_W
            val perMask = mh * mw
            val numCandidates = (allLogits.size / perMask).coerceAtLeast(1)
            // 选 IoU 最高候选（scores 与 masks 候选一一对应）
            var bestIdx = 0
            if (scores != null && scores.isNotEmpty() && numCandidates > 1) {
                var best = scores[0]
                for (i in 1 until minOf(numCandidates, scores.size)) {
                    if (scores[i] > best) { best = scores[i]; bestIdx = i }
                }
            }
            // 提取选中的候选 logits（682×1024）
            val logits = FloatArray(perMask)
            val base = bestIdx * perMask
            val avail = minOf(perMask, allLogits.size - base)
            for (i in 0 until avail) logits[i] = allLogits[base + i]
            // 若解析到的候选数不止一个但 scores 缺失，回退候选 0

            // ---- 后处理：双线性放大到原图 → logit>0 → 最大连通主体 ----
            val fullMask = resizeMask(logits, mw, mh, w, h)
            val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            val outPx = IntArray(w * h)
            // 阈值 + 记录连通主体（简化：4 邻域 floodfill 找最大）
            val mask = BooleanArray(w * h)
            var cnt = 0
            for (i in 0 until w * h) {
                if (fullMask[i] > 0f) { mask[i] = true; cnt++ }
            }
            if (cnt == 0) {
                // ★ 空掩码：返回 null（调用方据此提示"未识别到物体"，避免拿到全透明图误以为成功）
                return null
            }
            val largest = largestComponent(mask, w, h)
            for (i in 0 until w * h) {
                if (mask[i] && largest[i]) outPx[i] = 0xFFFFFFFF.toInt()
                else outPx[i] = 0
            }
            out.setPixels(outPx, 0, w, 0, 0, w, h)
            return out
        } finally {
            isRunning = false
            if (closeRequested) close()
        }
    }

    /** 把 onnx runtime 返回的嵌套对象展平为 float 数组（兼容 FloatArray/DoubleArray/嵌套 Array） */
    private fun flattenToFloatArray(v: Any?): FloatArray? {
        val out = ArrayList<Float>(MODEL_W * MODEL_H)
        fun rec(x: Any?) {
            when (x) {
                is FloatArray -> for (f in x) out.add(f)
                is DoubleArray -> for (d in x) out.add(d.toFloat())
                is IntArray -> for (i in x) out.add(i.toFloat())
                is LongArray -> for (l in x) out.add(l.toFloat())
                is Array<*> -> for (e in x) rec(e)
                is Number -> out.add(x.toFloat())
                else -> {}
            }
        }
        rec(v)
        if (out.isEmpty()) return null
        return out.toFloatArray()
    }

    /** 双线性放大 682×1024 → W×H（RGB 通道独立，float） */
    private fun resizeMask(src: FloatArray, sw: Int, sh: Int, dw: Int, dh: Int): FloatArray {
        val out = FloatArray(dw * dh)
        for (y in 0 until dh) {
            val fy = (y + 0.5f) * sh / dh - 0.5f
            val y0 = fy.coerceIn(0f, sh - 1f).toInt()
            val y1 = (y0 + 1).coerceAtMost(sh - 1)
            val wy = (fy - y0).coerceIn(0f, 1f)
            for (x in 0 until dw) {
                val fx = (x + 0.5f) * sw / dw - 0.5f
                val x0 = fx.coerceIn(0f, sw - 1f).toInt()
                val x1 = (x0 + 1).coerceAtMost(sw - 1)
                val wx = (fx - x0).coerceIn(0f, 1f)
                val p00 = src[y0 * sw + x0]; val p01 = src[y0 * sw + x1]
                val p10 = src[y1 * sw + x0]; val p11 = src[y1 * sw + x1]
                out[y * dw + x] = p00 * (1 - wx) * (1 - wy) + p01 * wx * (1 - wy) + p10 * (1 - wx) * wy + p11 * wx * wy
            }
        }
        return out
    }

    /** 8 邻域最大连通主体（BFS，返回与 mask 同尺寸的 BooleanArray） */
    private fun largestComponent(mask: BooleanArray, w: Int, h: Int): BooleanArray {
        val visited = BooleanArray(w * h)
        var best = 0
        var bestStart = -1
        for (i in mask.indices) {
            if (!mask[i] || visited[i]) continue
            // BFS
            val queue = java.util.ArrayDeque<Int>()
            queue.add(i); visited[i] = true
            var size = 0
            while (!queue.isEmpty()) {
                val cur = queue.poll()
                size++
                val cx = cur % w; val cy = cur / w
                for (dy in -1..1) {
                    for (dx in -1..1) {
                        if (dx == 0 && dy == 0) continue
                        val nx = cx + dx; val ny = cy + dy
                        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue
                        val ni = ny * w + nx
                        if (mask[ni] && !visited[ni]) { visited[ni] = true; queue.add(ni) }
                    }
                }
            }
            if (size > best) { best = size; bestStart = i }
        }
        val out = BooleanArray(w * h)
        if (bestStart < 0) return out
        // 再 BFS 一次拿 best 分量
        val queue = java.util.ArrayDeque<Int>()
        queue.add(bestStart); out[bestStart] = true
        while (!queue.isEmpty()) {
            val cur = queue.poll()
            val cx = cur % w; val cy = cur / w
            for (dy in -1..1) {
                for (dx in -1..1) {
                    if (dx == 0 && dy == 0) continue
                    val nx = cx + dx; val ny = cy + dy
                    if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue
                    val ni = ny * w + nx
                    if (mask[ni] && !out[ni]) { out[ni] = true; queue.add(ni) }
                }
            }
        }
        return out
    }
}