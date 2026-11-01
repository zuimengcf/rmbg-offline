package com.rmbg.offline

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.net.Uri
import kotlin.math.roundToInt

/**
 * 统一导出管线（所有保存出口共用）：
 *   1. 清脏 RGB：alpha=0 的像素 RGB 归零（实测樱花图 PNG -48%，并杜绝下游缩放彩边 halo）
 *   2. 主体收纳裁剪（可选）：canvas = alpha>128 最小外接矩形 + 边距（边距按主体长边百分比），
 *      主体自动居中；边距 0 = 紧贴主体。marginPct < 0 = 关闭（保留完整画布）。
 *   3. 格式编码：PNG 无损 / WEBP 有损(SDK>=30 稳定 alpha) / JPG 白底（透明区合成底色）。
 *
 * 注意：清 RGB 只影响 alpha==0 像素；画笔恢复取色走 orig_*.png 原图副本，不受影响。
 */
object ExportProcessor {

    enum class ExportFormat { PNG, WEBP, JPG }

    fun extOf(f: ExportFormat): String = when (f) {
        ExportFormat.PNG -> "png"
        ExportFormat.WEBP -> "webp"
        ExportFormat.JPG -> "jpg"
    }

    fun mimeOf(f: ExportFormat): String = when (f) {
        ExportFormat.PNG -> "image/png"
        ExportFormat.WEBP -> "image/webp"
        ExportFormat.JPG -> "image/jpeg"
    }

    /** 按导出格式生成建议文件名 */
    fun suggestedName(f: ExportFormat = ExportFormat.valueOf(Prefs.exportFormat)): String =
        "rmbg_${System.currentTimeMillis()}.${extOf(f)}"

    /** 从 uri 扩展名推断格式；识别不出返回 null（调用方回退 Prefs） */
    fun formatFromUri(uri: Uri): ExportFormat? = when {
        uri.lastPathSegment?.endsWith(".webp", true) == true -> ExportFormat.WEBP
        uri.lastPathSegment?.endsWith(".jpg", true) == true -> ExportFormat.JPG
        uri.lastPathSegment?.endsWith(".jpeg", true) == true -> ExportFormat.JPG
        uri.lastPathSegment?.endsWith(".png", true) == true -> ExportFormat.PNG
        else -> null
    }

    /**
     * 导出核心：清脏 RGB + 可选收纳裁剪。
     * @param marginPct 收纳边距（主体长边的百分比 0~50）；<0 = 不裁剪
     * @param center 是否把主体移到画布中心（收纳模式下恒为居中）
     * @return 处理后的新 Bitmap（调用方负责 recycle；不裁剪且无需清理时可能返回原引用）
     */
    fun buildExport(
        src: Bitmap,
        marginPct: Int,
        center: Boolean,
        cleanRgb: Boolean = true
    ): Bitmap {
        val w = src.width
        val h = src.height
        if (src.config != Bitmap.Config.ARGB_8888) {
            // 非 8888（如 JPEG 解码的 RGB_565）无 alpha 语义，直接走裁剪
            return if (marginPct >= 0) cropFit(src, marginPct, true) else src
        }
        val px = IntArray(w * h)
        src.getPixels(px, 0, w, 0, 0, w, h)
        // ---- 1. 清脏 RGB（含 premul 残留：alpha==0 时 RGB 一律归零）----
        var minX = w; var minY = h; var maxX = -1; var maxY = -1
        var hasFg = false
        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until w) {
                val i = row + x
                val a = (px[i] ushr 24) and 0xFF
                if (a == 0) {
                    if (cleanRgb) px[i] = 0
                } else {
                    hasFg = true
                    if (x < minX) minX = x
                    if (x > maxX) maxX = x
                    if (y < minY) minY = y
                    if (y > maxY) maxY = y
                }
            }
        }
        val cleaned = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        cleaned.setPixels(px, 0, w, 0, 0, w, h)
        if (marginPct < 0) {
            if (!center || !hasFg || (minX == 0 && minY == 0 && maxX == w - 1 && maxY == h - 1)) {
                return cleaned
            }
            // ---- 仅居中（不裁剪）：透明画布平移主体到中心 ----
            val cx = (minX + maxX) / 2.0
            val cy = (minY + maxY) / 2.0
            val dx = ((w - 1) / 2.0 - cx).roundToInt()
            val dy = ((h - 1) / 2.0 - cy).roundToInt()
            if (dx == 0 && dy == 0) return cleaned
            val moved = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            val cv = Canvas(moved)
            cv.drawBitmap(cleaned, dx.toFloat(), dy.toFloat(), null)
            cleaned.recycle()
            return moved
        }
        if (!hasFg) return cleaned
        // ---- 2. 主体收纳：以清理后的 bbox 为准裁剪（顺带裁掉画布外雾带）----
        val bw = maxX - minX + 1
        val bh = maxY - minY + 1
        val m = (maxOf(bw, bh) * (marginPct.coerceIn(0, 50)) / 100.0).roundToInt()
        var cw = bw + 2 * m
        var ch = bh + 2 * m
        // 上限保护：收纳画布不超过原画布 2 倍（超大边距时避免内存翻倍）
        cw = cw.coerceAtMost(w * 2)
        ch = ch.coerceAtMost(h * 2)
        val canvas = Bitmap.createBitmap(cw, ch, Bitmap.Config.ARGB_8888)
        val cv = Canvas(canvas)
        // 主体 bbox 中心 → 画布中心
        val bcx = (minX + maxX) / 2.0
        val bcy = (minY + maxY) / 2.0
        val dx = ((cw - 1) / 2.0 - bcx).roundToInt()
        val dy = ((ch - 1) / 2.0 - bcy).roundToInt()
        cv.drawBitmap(cleaned, dx.toFloat(), dy.toFloat(), null)
        cleaned.recycle()
        return canvas
    }

    /** 收纳裁剪（无像素级清理需求时直接按 bbox 裁；内部走 buildExport 统一语义） */
    private fun cropFit(src: Bitmap, marginPct: Int, center: Boolean): Bitmap {
        // 非 8888：先转 8888 再走统一管线
        val arg = src.copy(Bitmap.Config.ARGB_8888, false)
        val r = buildExport(arg, marginPct, center)
        if (r !== arg) arg.recycle()
        return r
    }

    /**
     * 编码写入 uri（按格式；JPG 透明区合成白色底）。
     * @return 是否成功
     */
    fun write(
        context: Context,
        bmp: Bitmap,
        uri: Uri,
        format: ExportFormat,
        quality: Int,
        marginPct: Int,
        center: Boolean,
        cleanRgb: Boolean = true
    ): Boolean {
        return try {
            val out = buildExport(bmp, marginPct, center, cleanRgb)
            val q = quality.coerceIn(50, 100)
            context.contentResolver.openOutputStream(uri)?.use { os ->
                when (format) {
                    ExportFormat.PNG -> out.compress(Bitmap.CompressFormat.PNG, 100, os)
                    ExportFormat.WEBP ->
                        if (android.os.Build.VERSION.SDK_INT >= 30) {
                            out.compress(Bitmap.CompressFormat.WEBP_LOSSY, q, os)
                        } else {
                            @Suppress("DEPRECATION")
                            out.compress(Bitmap.CompressFormat.WEBP, q, os)
                        }
                    ExportFormat.JPG -> {
                        // JPG 无 alpha：透明区合成白底（避免黑底）
                        val bg = Bitmap.createBitmap(out.width, out.height, Bitmap.Config.ARGB_8888)
                        val cv = Canvas(bg)
                        cv.drawColor(Color.WHITE)
                        cv.drawBitmap(out, 0f, 0f, null)
                        bg.compress(Bitmap.CompressFormat.JPEG, q, os)
                        bg.recycle()
                    }
                }
            }
            if (out !== bmp) out.recycle()
            true
        } catch (e: Exception) {
            false
        }
    }

    /** write + toast 由调用方处理；便捷布尔封装 */
    fun encode(
        context: Context,
        bmp: Bitmap,
        uri: Uri,
        format: ExportFormat,
        quality: Int,
        marginPct: Int,
        center: Boolean
    ): Boolean = write(context, bmp, uri, format, quality, marginPct, center)

    /**
     * 预览用：产出 (预览位图[最长边<=512]，预估字节数)。
     * 预览位图已应用 清脏 RGB + 收纳/居中；JPG 语义下透明区合成白底（所见即所得）。
     */
    fun buildPreview(
        src: Bitmap,
        format: ExportFormat,
        quality: Int,
        marginPct: Int,
        center: Boolean,
        cleanRgb: Boolean = true
    ): Pair<Bitmap, Long> {
        val exp = buildExport(src, marginPct, center, cleanRgb)
        // 预览降采样（最长边 512），大幅降低每次滑动的重算开销
        val longest = maxOf(exp.width, exp.height)
        val preview = if (longest <= 512) exp else {
            val s = 512f / longest
            val pb = Bitmap.createScaledBitmap(exp, (exp.width * s).roundToInt().coerceAtLeast(1),
                (exp.height * s).roundToInt().coerceAtLeast(1), true)
            if (exp !== src) exp.recycle()
            pb
        }
        val bytes = estimateSize(preview, format, quality)
        return Pair(preview, bytes)
    }

    /** 预估编码字节数（真实编码到内存流） */
    fun estimateSize(bmp: Bitmap, format: ExportFormat, quality: Int): Long = try {
        val q = quality.coerceIn(50, 100)
        java.io.ByteArrayOutputStream().use { bos ->
            when (format) {
                ExportFormat.PNG -> bmp.compress(Bitmap.CompressFormat.PNG, 100, bos)
                ExportFormat.WEBP ->
                    if (android.os.Build.VERSION.SDK_INT >= 30) {
                        bmp.compress(Bitmap.CompressFormat.WEBP_LOSSY, q, bos)
                    } else {
                        @Suppress("DEPRECATION")
                        bmp.compress(Bitmap.CompressFormat.WEBP, q, bos)
                    }
                ExportFormat.JPG -> {
                    val bg = Bitmap.createBitmap(bmp.width, bmp.height, Bitmap.Config.ARGB_8888)
                    val cv = Canvas(bg)
                    cv.drawColor(Color.WHITE)
                    cv.drawBitmap(bmp, 0f, 0f, null)
                    bg.compress(Bitmap.CompressFormat.JPEG, q, bos)
                    bg.recycle()
                }
            }
            bos.size().toLong()
        }
    } catch (e: Exception) {
        -1L
    }
}
