package com.rmbg.offline

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.io.File
import java.io.FileOutputStream

/**
 * 当前工程持久化：把正在编辑的原图 + alpha 遮罩存到内部存储 work/ 目录，
 * 后台被杀/切应用后回来可恢复，不丢失工作进度。
 *
 * ★ 架构纠正：不再保存「结果图（透明 PNG）」——结果图透明区 RGB 已丢，恢复永远不对。
 *   改为保存「原图 RGB（唯一颜色源）+ alpha 遮罩（唯一可编辑状态）」：
 *   - original.png：完整原图（RGB + 原始 alpha）
 *   - mask.png：灰度遮罩图（白=保留/不透明，黑=擦除/透明），即结果图 alpha 通道的独立存储
 *   加载时从原图 RGB + 遮罩 alpha 合成结果图，遮罩可随时编辑，原图 RGB 永不丢失。
 */
object ProjectStateManager {

    private const val ORIG = "work_original.png"
    private const val MASK = "work_mask.png"

    private fun workDir(context: Context): File =
        File(context.filesDir, "work").apply { mkdirs() }

    /** 保存当前工程（原图 + 结果遮罩），结果可为空（未抠图时） */
    fun save(context: Context, original: Bitmap?, result: Bitmap?) {
        try {
            val dir = workDir(context)
            if (original != null) {
                FileOutputStream(File(dir, ORIG)).use { os ->
                    original.compress(Bitmap.CompressFormat.PNG, 100, os)
                }
            }
            // ★ 从结果图提取 alpha 遮罩存为灰度 PNG（不存结果 RGB——透明区 RGB 已不可靠）
            if (result != null) {
                val mask = extractAlphaMask(result)
                FileOutputStream(File(dir, MASK)).use { os ->
                    mask.compress(Bitmap.CompressFormat.PNG, 100, os)
                }
                if (mask !== result) mask.recycle()
            }
        } catch (_: Exception) {}
    }

    /** 从位图提取 alpha 通道为独立灰度遮罩图（白=不透明/保留，黑=透明/擦除） */
    private fun extractAlphaMask(bmp: Bitmap): Bitmap {
        val w = bmp.width; val h = bmp.height
        val px = IntArray(w * h)
        bmp.getPixels(px, 0, w, 0, 0, w, h)
        for (i in px.indices) {
            val a = (px[i] ushr 24) and 0xFF
            // 灰度值 = alpha（255=白/保留，0=黑/擦除）
            px[i] = (0xFF shl 24) or (a shl 16) or (a shl 8) or a
        }
        val mask = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        mask.setPixels(px, 0, w, 0, 0, w, h)
        return mask
    }

    /** 恢复原图（可能为 null） */
    fun loadOriginal(context: Context): Bitmap? {
        return try {
            val f = File(workDir(context), ORIG)
            if (f.exists()) BitmapFactory.decodeFile(f.absolutePath) else null
        } catch (_: Exception) { null }
    }

    /** 恢复结果图（从原图 RGB + 遮罩 alpha 合成，原图不可用或无遮罩时返回 null） */
    fun loadResult(context: Context): Bitmap? {
        return try {
            val dir = workDir(context)
            val origFile = File(dir, ORIG)
            val maskFile = File(dir, MASK)
            if (!origFile.exists()) return null
            if (!maskFile.exists()) return null
            val orig = BitmapFactory.decodeFile(origFile.absolutePath) ?: return null
            val mask = BitmapFactory.decodeFile(maskFile.absolutePath) ?: return orig
            // 合成：原图 RGB + 遮罩 alpha → 结果图
            val w = orig.width; val h = orig.height
            // 尺寸不一致时以原图为准（遮罩可能不同尺寸，缩放取样）
            val maskAligned = if (mask.width == w && mask.height == h) mask
                              else Bitmap.createScaledBitmap(mask, w, h, true)
            val origPx = IntArray(w * h)
            val maskPx = IntArray(w * h)
            orig.getPixels(origPx, 0, w, 0, 0, w, h)
            maskAligned.getPixels(maskPx, 0, w, 0, 0, w, h)
            for (i in origPx.indices) {
                // alpha 取遮罩的灰度值（R 通道），RGB 取原图
                val a = maskPx[i] and 0xFF
                origPx[i] = (a shl 24) or (origPx[i] and 0x00FFFFFF)
            }
            val result = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            result.setPixels(origPx, 0, w, 0, 0, w, h)
            if (mask !== orig) orig.recycle()
            if (mask !== maskAligned) mask.recycle()
            if (maskAligned !== result) maskAligned.recycle()
            result
        } catch (_: Exception) { null }
    }

    /** 清除工程缓存 */
    fun clear(context: Context) {
        try {
            val dir = workDir(context)
            File(dir, ORIG).delete()
            File(dir, MASK).delete()
        } catch (_: Exception) {}
    }
}