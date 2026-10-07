package com.rmbg.offline

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * 抠图历史记录管理器
 * 保存抠图结果到应用内部存储 history/ 目录，带缩略图。
 * 每条记录附带模型名 / 耗时 / 加速方式等元数据（存 info.txt 旁文件）。
 */
object HistoryManager {

    // ★ 缩略图内存缓存（避免历史页每次进入/滚动都重新解码 PNG）
    //   key = 文件名，容量按内存估算（每张缩略图约 78×78×4 ≈ 24KB，200 张 ≈ 5MB）
    // ★ 注意：淘汰时【不能】立即 recycle()！同一缩略图可能同时被网格/列表视图引用
    //   （都从缓存拿到同一个 Bitmap），提前回收会导致 "Canvas: trying to use a recycled bitmap" 崩溃。
    //   交给 GC 回收即可（Android 8+ 对硬件位图/普通位图内存都能正常回收）。
    private val thumbCache = object : android.util.LruCache<String, Bitmap>(8 * 1024 * 1024) {
        override fun sizeOf(key: String, value: Bitmap): Int = value.byteCount
    }

    data class HistoryItem(
        val id: String,
        val fileName: String,
        val thumbName: String,
        val timestamp: Long,
        /** ★ 原图副本文件名（orig_$id.png），无则为空；供恢复画笔取原背景色 */
        val origFileName: String = "",
        /** 使用的模型显示名（如 RMBG-1.4 / QNN-HTP EPContext / 本地模型名） */
        val modelName: String = "",
        /** 推理耗时（毫秒，0=未知） */
        val durationMs: Long = 0L,
        /** 加速方式（CPU / NNAPI / QNN） */
        val accel: String = "",
        /** 是否极速模式 */
        val turbo: Boolean = false,
        /** 原图分组键（同一原图的多次抠图共享同一 key，用于归纳折叠分组） */
        val originalKey: String = "",
        /** 结果图文件大小（字节），用于大小排序 */
        val fileSize: Long = 0L
    )

    private fun historyDir(context: Context): File =
        File(context.filesDir, "history").apply { mkdirs() }

    /**
     * 保存抠图结果（原图 + 结果 + 缩略图 + 元数据）
     * @param modelName 模型显示名
     * @param durationMs 推理耗时（毫秒）
     * @param accel 加速方式
     * @param turbo 是否极速模式
     * @return 保存的 HistoryItem
     */
    fun saveResult(
        context: Context,
        result: Bitmap,
        original: Bitmap? = null,          // ★ 原图副本（供恢复画笔取原背景色）
        originalName: String? = null,
        modelName: String = "",
        durationMs: Long = 0L,
        accel: String = "",
        turbo: Boolean = false,
        originalKey: String = ""
    ): HistoryItem {
        val dir = historyDir(context)
        val id = System.currentTimeMillis().toString()
        val ts = System.currentTimeMillis()

        // 结果图（原尺寸）
        val fileName = "rmbg_$id.png"
        val file = File(dir, fileName)
        FileOutputStream(file).use { os ->
            result.compress(Bitmap.CompressFormat.PNG, 100, os)
        }
        // ★ 原图副本（同名 .orig.png）：供"恢复画笔"从原图取 RGB（原背景色永不丢失）。
        //   历史记录恢复正确性的关键——历史打开时不能用主界面当前原图（可能不是同一张）。
        //   兼容旧记录（无 orig 文件）：loadOriginal 返回 null，恢复退回结果自身提取。
        val origFileName = "orig_$id.png"
        if (original != null && !original.isRecycled) {
            try {
                FileOutputStream(File(dir, origFileName)).use { os ->
                    original.compress(Bitmap.CompressFormat.PNG, 100, os)
                }
            } catch (_: Exception) {}
        }

        // 缩略图（256px 宽）
        val thumbName = "thumb_$id.jpg"
        val scale = 256f / result.width
        val thumbW = 256
        val thumbH = (result.height * scale).toInt().coerceAtLeast(1)
        val thumb = Bitmap.createScaledBitmap(result, thumbW, thumbH, true)
        FileOutputStream(File(dir, thumbName)).use { os ->
            thumb.compress(Bitmap.CompressFormat.JPEG, 80, os)
        }
        thumb.recycle()

        // 元数据（模型 / 耗时 / 加速 / 极速 / 原图分组键）
        try {
            File(dir, "meta_$id.txt").writeText(
                listOf(
                    "model=$modelName",
                    "duration_ms=$durationMs",
                    "accel=$accel",
                    "turbo=$turbo",
                    "time=$ts",
                    "original_key=$originalKey"
                ).joinToString("\n")
            )
        } catch (_: Exception) {}

        return HistoryItem(
            id = id,
            fileName = fileName,
            thumbName = thumbName,
            timestamp = ts,
            origFileName = if (original != null && !original.isRecycled) origFileName else "",
            modelName = modelName,
            durationMs = durationMs,
            accel = accel,
            turbo = turbo,
            originalKey = originalKey,
            fileSize = file.length()
        )
    }

    /** ★ 更新历史记录的结果图 + 缩略图（后处理精修覆盖同一记录，原图 orig 存档与元数据保持不变）。
     *  用于"后处理/抠图统一为同一条历史记录"：不新建记录，只覆盖结果，原图始终复用同一存档。 */
    fun updateResult(context: Context, item: HistoryItem, newResult: Bitmap): HistoryItem? {
        if (newResult.isRecycled) return null
        val dir = historyDir(context)
        val f = File(dir, item.fileName)
        try {
            FileOutputStream(f).use { os -> newResult.compress(Bitmap.CompressFormat.PNG, 100, os) }
        } catch (_: Exception) { return null }
        // 缩略图（256px 宽）
        val thumbName = "thumb_${item.id}.jpg"
        try {
            val scale = 256f / newResult.width
            val thumbW = 256
            val thumbH = (newResult.height * scale).toInt().coerceAtLeast(1)
            val thumb = Bitmap.createScaledBitmap(newResult, thumbW, thumbH, true)
            FileOutputStream(File(dir, thumbName)).use { os -> thumb.compress(Bitmap.CompressFormat.JPEG, 80, os) }
            thumb.recycle()
        } catch (_: Exception) {}
        return item.copy(
            thumbName = thumbName,
            timestamp = System.currentTimeMillis(), // ★ 更新时间戳，使精修后的记录排到最前
            fileSize = f.length()
        )
    }

    /** 读取单条元数据（旧记录无文件则返回默认值） */
    private fun readMeta(context: Context, id: String): HistoryItem {
        val dir = historyDir(context)
        val meta = File(dir, "meta_$id.txt")
        var modelName = ""
        var durationMs = 0L
        var accel = ""
        var turbo = false
        var originalKey = ""
        try {
            if (meta.exists()) {
                meta.readLines().forEach { line ->
                    val kv = line.split("=", limit = 2)
                    if (kv.size == 2) {
                        when (kv[0]) {
                            "model" -> modelName = kv[1]
                            "duration_ms" -> durationMs = kv[1].toLongOrNull() ?: 0L
                            "accel" -> accel = kv[1]
                            "turbo" -> turbo = kv[1] == "true"
                            "original_key" -> originalKey = kv[1]
                        }
                    }
                }
            }
        } catch (_: Exception) {}
        return HistoryItem(
            id = id,
            fileName = "",
            thumbName = "",
            timestamp = 0L,
            modelName = modelName,
            durationMs = durationMs,
            accel = accel,
            turbo = turbo,
            originalKey = originalKey
        )
    }

    // ★ 历史列表缓存（避免每次切到历史页都全量扫目录+读 meta 磁盘 I/O）
    //   失效策略：目录最后修改时间或文件数变化才重扫
    private var histCache: List<HistoryItem>? = null
    private var histCacheDirMtime = 0L
    private var histCacheCount = -1

    /** 获取历史列表（按时间倒序）——带缓存，目录未变时直接返回 */
    fun getHistory(context: Context): List<HistoryItem> {
        val dir = historyDir(context)
        val files = dir.listFiles() ?: return emptyList()
        val count = files.count { it.name.startsWith("rmbg_") && it.name.endsWith(".png") }
        val mtime = dir.lastModified()
        // 目录没变 → 复用缓存（historyVersion 变化或新增记录时目录 mtime 会更新）
        if (histCache != null && histCacheCount == count && histCacheDirMtime == mtime) {
            return histCache!!
        }
        val items = files
            .filter { it.name.startsWith("rmbg_") && it.name.endsWith(".png") }
            .mapNotNull { f ->
                val id = f.name.removePrefix("rmbg_").removeSuffix(".png")
                val thumb = File(dir, "thumb_$id.jpg")
                val orig = File(dir, "orig_$id.png")
                val meta = readMeta(context, id)
                HistoryItem(
                    id = id,
                    fileName = f.name,
                    thumbName = if (thumb.exists()) thumb.name else "",
                    timestamp = f.lastModified(),
                    origFileName = if (orig.exists()) orig.name else "",
                    modelName = meta.modelName,
                    durationMs = meta.durationMs,
                    accel = meta.accel,
                    turbo = meta.turbo,
                    originalKey = meta.originalKey,
                    fileSize = f.length()
                )
            }
            .sortedByDescending { it.timestamp }
        histCache = items
        histCacheCount = count
        histCacheDirMtime = mtime
        return items
    }

    /** ★ 清空历史缓存（删除记录后调用，避免下次进入仍显示旧列表） */
    fun invalidateHistoryCache() {
        histCache = null
    }

    /** 加载历史结果图 */
    fun loadResult(context: Context, item: HistoryItem): Bitmap? {
        val f = File(historyDir(context), item.fileName)
        if (!f.exists()) return null
        return BitmapFactory.decodeFile(f.absolutePath)
    }

    /** ★ 加载历史对应的原图副本（供恢复画笔取原背景色），无则返回 null */
    fun loadOriginal(context: Context, item: HistoryItem): Bitmap? {
        if (item.origFileName.isEmpty()) return null
        val f = File(historyDir(context), item.origFileName)
        if (!f.exists()) return null
        return BitmapFactory.decodeFile(f.absolutePath)
    }

    /** 加载缩略图（带内存缓存，避免历史页滚动/切换重复解码） */
    fun loadThumb(context: Context, item: HistoryItem): Bitmap? {
        if (item.thumbName.isEmpty()) return null
        thumbCache.get(item.thumbName)?.let { return it }
        val f = File(historyDir(context), item.thumbName)
        if (!f.exists()) return null
        val bmp = BitmapFactory.decodeFile(f.absolutePath) ?: return null
        // 缓存（若超容量会淘汰最久未用）
        thumbCache.put(item.thumbName, bmp)
        return bmp
    }

    /** 删除一条历史记录 */
    fun delete(context: Context, item: HistoryItem): Boolean {
        val dir = historyDir(context)
        var ok = File(dir, item.fileName).delete()
        if (item.thumbName.isNotEmpty()) {
            ok = File(dir, item.thumbName).delete() && ok
        }
        if (item.origFileName.isNotEmpty()) {
            File(dir, item.origFileName).delete()
        }
        File(dir, "meta_${item.id}.txt").delete()
        return ok
    }

    /** 清空历史 */
    fun clearAll(context: Context): Boolean {
        val dir = historyDir(context)
        val files = dir.listFiles() ?: return true
        var ok = true
        files.forEach { ok = it.delete() && ok }
        return ok
    }

    /** 格式化时间 */
    fun formatTime(ts: Long): String =
        SimpleDateFormat("MM-dd HH:mm", Locale.getDefault()).format(Date(ts))
}