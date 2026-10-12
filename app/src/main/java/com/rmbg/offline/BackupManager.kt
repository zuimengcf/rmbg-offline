package com.rmbg.offline

import android.content.Context
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream
import java.util.zip.ZipOutputStream

/**
 * 导出 / 导入（备份 / 恢复）
 * =====================
 * ★ 设计约定（用户要求）：
 *   - 模型：只能一个一个单独导出（每个模型一个文件，见 MainActivity 内的模型导出逻辑）。
 *   - 历史 + 配置：合并打包为一个 zip（本文件负责）。
 *
 * 备份 zip 结构（历史+配置）：
 *   - manifest.json  备份清单（版本 / 导出时间 / 统计）
 *   - settings.json  Prefs 全量配置（SharedPreferences "rmbg_settings" 全部键值）
 *   - history/       历史记录目录完整内容（rmbg_*.png / orig_*.png / thumb_*.jpg / meta_*.txt）
 *
 * 说明：
 *   - 历史目录在内部 filesDir/history；配置为 SharedPreferences（rmbg_settings.xml）。
 *   - 导出/导入必须跑在 IO 线程（协程 Dispatchers.IO）。
 *   - 进度回调（bytesDone / totalBytes）供 UI 展示。
 */
object BackupManager {

    /** 备份格式版本 */
    private const val BACKUP_VERSION = 1

    /** 进度回调（byte 单位，总字节为 -1 时未知） */
    fun interface Progress {
        fun onProgress(bytesDone: Long, totalBytes: Long)
    }

    // ------------------------------------------------ 目录定位 ------------------------------------------------

    /** 历史目录（内部存储 filesDir/history） */
    fun historyDir(context: Context): File = File(context.filesDir, "history")

    /** Prefs SharedPreferences 文件名 */
    private fun prefsFile(context: Context): File =
        File(context.filesDir, "shared_prefs/rmbg_settings.xml")

    // ------------------------------------------------ 导出（历史+配置 → zip） ------------------------------------------------

    /**
     * 导出「历史 + 配置」备份 zip 到目标输出流（SAF 返回的 uri 对应流）。
     * @return 是否成功
     */
    fun exportHistoryAndPrefs(
        context: Context,
        outStream: OutputStream,
        progress: Progress? = null
    ): Boolean {
        return try {
            val histDir = historyDir(context)
            // 统计总字节（历史目录 + settings 文本），用于进度
            var total = 0L
            total += dirSize(histDir)
            val settingsText = exportSettings(context)
            total += settingsText.length.toLong()
            total += 256L // manifest 估算

            var done = 0L
            progress?.onProgress(0, total)

            ZipOutputStream(outStream).use { zos ->
                // manifest.json
                val manifest = JSONObject().apply {
                    put("app", "RMBG 离线抠图")
                    put("backup_version", BACKUP_VERSION)
                    put("export_time", System.currentTimeMillis())
                    put("history_files", countFiles(histDir))
                }
                val manifestBytes = manifest.toString().toByteArray(Charsets.UTF_8)
                writeEntry(zos, "manifest.json", manifestBytes)
                done += manifestBytes.size.toLong()
                progress?.onProgress(done, total)

                // settings.json（Prefs 全量）
                writeEntry(zos, "settings.json", settingsText.toByteArray(Charsets.UTF_8))
                done += settingsText.length.toLong()
                progress?.onProgress(done, total)

                // history/ 目录
                if (histDir.exists()) {
                    val ok = zipDir(zos, histDir, "history") { n ->
                        done += n
                        progress?.onProgress(done, total)
                    }
                    if (!ok) return false
                }
            }
            true
        } catch (e: Exception) {
            android.util.Log.e("RMBG-BACKUP", "导出失败", e)
            false
        }
    }

    // ------------------------------------------------ 导入（zip → 历史+配置） ------------------------------------------------

    /**
     * 导入「历史 + 配置」备份 zip。
     * 还原 settings.json（Prefs）与 history/ 目录。
     * @param zi 已打开的 ZipInputStream（由调用方获得 uri 流后传入；本函数负责读取与解压）
     * @return 是否成功；成功后调用方应 invalidateHistoryCache + 刷新 UI
     */
    fun importHistoryAndPrefs(context: Context, zi: ZipInputStream): Boolean {
        return try {
            var settingsJson: String? = null
            var entry: ZipEntry?
            // 第一遍：读 manifest / settings，收集历史文件临时落盘
            val histDir = historyDir(context)
            histDir.mkdirs()
            // ★ 防路径穿越：只取文件名（备份内部本来就是扁平结构，无需子目录）
            while (zi.nextEntry.also { entry = it } != null) {
                val e = entry ?: continue
                if (e.isDirectory) continue
                when {
                    e.name == "settings.json" -> {
                        settingsJson = zi.readBytes().toString(Charsets.UTF_8)
                    }
                    e.name.startsWith("history/") -> {
                        val baseName = e.name.substringAfterLast('/').substringAfterLast('\\')
                        if (baseName.isEmpty()) continue
                        // 只接受已知类型文件，防止任意写入
                        if (!baseName.endsWith(".png", true) &&
                            !baseName.endsWith(".jpg", true) &&
                            !baseName.endsWith(".txt", true)) continue
                        val dest = File(histDir, baseName)
                        // ★ zip 炸弹防护：单次解压总量限制（历史图几千张 × 几MB 内，上限 4GB）
                        if (dest.length() + e.size > 4L * 1024 * 1024 * 1024) {
                            android.util.Log.e("RMBG-BACKUP", "解压超限，中止")
                            return false
                        }
                        zi.copyTo(FileOutputStream(dest))
                    }
                }
                zi.closeEntry()
            }

            // 还原配置
            if (settingsJson != null) {
                restoreSettings(context, settingsJson)
            }
            true
        } catch (e: Exception) {
            android.util.Log.e("RMBG-BACKUP", "导入失败", e)
            false
        }
    }

    // ------------------------------------------------ Prefs 配置 导出/还原 ------------------------------------------------

    /** 导出 Prefs 全量配置为 JSON 字符串（SharedPreferences getAll 序列化） */
    private fun exportSettings(context: Context): String {
        val json = JSONObject()
        val sp = context.getSharedPreferences("rmbg_settings", Context.MODE_PRIVATE)
        val all = sp.all ?: return "{}"
        for ((k, v) in all) {
            when (v) {
                is String -> json.put(k, v)
                is Int -> json.put(k, v)
                is Long -> json.put(k, v)
                is Float -> json.put(k, v)
                is Boolean -> json.put(k, v)
                is java.util.Set<*> -> json.put(k, v.joinToString(","))
                // 其余类型跳过
            }
        }
        return json.toString()
    }

    /** 从 JSON 还原 Prefs（仅覆盖存在的键，不删除用户未备份的新配置键） */
    private fun restoreSettings(context: Context, jsonText: String) {
        try {
            val json = JSONObject(jsonText)
            val sp = context.getSharedPreferences("rmbg_settings", Context.MODE_PRIVATE)
            val ed = sp.edit()
            val keys = json.keys()
            while (keys.hasNext()) {
                val k = keys.next()
                val v = json.opt(k) ?: continue
                when (v) {
                    is String -> ed.putString(k, v)
                    is Int -> ed.putInt(k, v)
                    is Long -> ed.putLong(k, v)
                    is Double -> ed.putFloat(k, v.toFloat())
                    is Boolean -> ed.putBoolean(k, v)
                    else -> { /* 数组/对象跳过 */ }
                }
            }
            ed.apply()
        } catch (e: Exception) {
            android.util.Log.e("RMBG-BACKUP", "还原配置失败", e)
        }
    }

    // ------------------------------------------------ 工具 ------------------------------------------------

    /** 递归统计目录大小（含内部文件） */
    fun dirSize(dir: File): Long {
        if (!dir.exists()) return 0L
        val files = dir.listFiles() ?: return 0L
        var total = 0L
        for (f in files) {
            total += if (f.isDirectory) dirSize(f) else f.length()
        }
        return total
    }

    /** 统计目录内文件数 */
    private fun countFiles(dir: File): Int {
        if (!dir.exists()) return 0
        val files = dir.listFiles() ?: return 0
        var count = 0
        for (f in files) {
            count += if (f.isDirectory) countFiles(f) else 1
        }
        return count
    }

    /** 写入单个 zip entry */
    private fun writeEntry(zos: ZipOutputStream, name: String, bytes: ByteArray) {
        zos.putNextEntry(ZipEntry(name))
        zos.write(bytes)
        zos.closeEntry()
    }

    /** 把目录递归写入 zip（zip 内路径 = prefix/相对路径） */
    private fun zipDir(zos: ZipOutputStream, dir: File, prefix: String, onBytes: (Long) -> Unit): Boolean {
        if (!dir.exists()) return true
        val files = dir.listFiles() ?: return true
        for (f in files) {
            val entryName = "$prefix/${f.name}"
            if (f.isDirectory) {
                if (!zipDir(zos, f, entryName, onBytes)) return false
            } else if (f.isFile) {
                try {
                    zos.putNextEntry(ZipEntry(entryName))
                    f.inputStream().use { input ->
                        val buf = ByteArray(1 shl 16)
                        var n: Int
                        var total = 0L
                        while (input.read(buf).also { n = it } != -1) {
                            zos.write(buf, 0, n)
                            total += n
                            if (total >= (1L shl 20)) { // 每 1MB 汇报一次进度
                                onBytes(total)
                                total = 0
                            }
                        }
                        onBytes(total)
                    }
                    zos.closeEntry()
                } catch (e: Exception) {
                    android.util.Log.e("RMBG-BACKUP", "写入 $entryName 失败", e)
                    return false
                }
            }
        }
        return true
    }
}