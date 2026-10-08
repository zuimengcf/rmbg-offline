package com.rmbg.offline.ml

import android.content.Context
import android.os.Handler
import android.os.Looper
import com.rmbg.offline.BuildConfig
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong

/**
 * RMBG 模型下载管理器
 *
 * 支持：
 * 1. 多镜像源自动切换（hf-mirror.com 国内镜像优先，官方兜底）
 * 2. 多线程分块下载（并行 Range 请求，默认 4 连接）
 * 3. 断点续传（.part 文件自动续传）
 * 4. 内置多模型（RMBG-1.4 FP32/FP16/量化 + RMBG-2.0）
 * 5. 所有回调切主线程，避免 IO 线程操作 UI 闪退
 */
object ModelManager {

    /** 镜像源信息 */
    data class Mirror(
        val name: String,
        val baseUrl: String,
        val description: String
    )

    /** 模型类型：CPU（普通 ONNX，可 NNAPI 加速） / QNN（EPContext 离线编译产物，走骁龙 DSP） */
    enum class ModelKind { CPU, QNN }

    /** 内置模型信息 */
    data class BuiltinModel(
        val id: String,
        val name: String,
        val hfRepo: String,
        val hfFile: String,
        val sizeBytes: Long,
        val description: String,
        val kind: ModelKind = ModelKind.CPU,
        // ---- QNN EPContext 专用（kind=QNN 时有效）----
        val assetOnnx: String? = null,      // assets 内 onnx 路径（ep_cache_context 引用 bin 名）
        val assetBin: String? = null,       // assets 内 bin 路径
        val hfBinFile: String? = null,      // HF 仓库内 bin 路径（双轨下载用；null=由 hfFile 同目录推断）
        val hfZip: String? = null,          // ★ HF 仓库内 zip 路径（内含 bin+onnx；设置后云下载改为整包 zip 下载→解压部署，本地 zip 导入同样适用）
        val deployOnnxName: String? = null, // 部署到模型目录后的 onnx 文件名（null=用 asset 文件名）
        val deployBinName: String? = null   // 部署到模型目录后的 bin 文件名（★必须与 onnx 内 ep_cache_context 引用一致）
    ) {
        val isQnn: Boolean get() = kind == ModelKind.QNN
    }

    /** CPU 模型列表（普通 ONNX，CPU / NNAPI 推理） */
    val cpuModels = listOf(
        BuiltinModel(
            id = "rmbg14_fp32",
            name = "RMBG-1.4",
            hfRepo = "briaai/RMBG-1.4",
            hfFile = "onnx/model.onnx",
            sizeBytes = 176_153_355L,
            description = "通用抠图 · 官方原版 · 176MB"
        ),
        BuiltinModel(
            id = "rmbg20",
            name = "RMBG-2.0",
            hfRepo = "briaai/RMBG-2.0",
            hfFile = "onnx/model_fp16.onnx",
            sizeBytes = 513_576_499L,
            description = "BiRefNet 最强 · 513MB（需 HF Token）"
        ),
        BuiltinModel(
            id = "modnet",
            name = "MODNet",
            hfRepo = "Xenova/modnet",
            hfFile = "onnx/model.onnx",
            sizeBytes = 25_888_640L,
            description = "人像抠图 · 轻量快速 · 25MB"
        ),
        BuiltinModel(
            id = "anime_seg",
            name = "Anime-Seg",
            hfRepo = "skytnt/anime-seg",
            hfFile = "isnetis.onnx",
            sizeBytes = 176_069_933L,
            description = "动漫人物抠图 · ISNet · 176MB"
        )
    )

    /** QNN 模型列表（EPContext 离线编译产物，HTP DSP 加速，无需在线编译）
     *  ★ App 规范命名：qnn_<model>_<soc/后缀>，onnx 内 ep_cache_context 引用与 bin 同名 */
    val qnnModels = listOf(
        BuiltinModel(
            id = "qnn_rmbg14_v2",
            name = "RMBG-1.4 QNN v2",
            hfRepo = "zuimengqm/rmbg-qnn-rmbg14",
            hfFile = "qnn/qnn_rmbg14_sdk250_v2.onnx",
            hfBinFile = "qnn/qnn_rmbg14_sdk250_v2.bin",
            sizeBytes = 99_079_792L + 855L,
            description = "RMBG-1.4 · SDK250 离线编译 · 99MB（新版）",
            kind = ModelKind.QNN,
            assetOnnx = "qnn/qnn_rmbg14_sdk250_v2.onnx",
            assetBin = "qnn/qnn_rmbg14_sdk250_v2.bin",
            deployOnnxName = "qnn_rmbg14_sdk250_v2.onnx",
            deployBinName = "qnn_rmbg14_sdk250_v2.bin"
        ),
        BuiltinModel(
            id = "qnn_animeseg",
            name = "Anime-Seg QNN-HTP",
            hfRepo = "zuimengqm/rmbg-qnn-animeseg",
            hfFile = "qnn/qnn_animeseg_v73.onnx",
            hfBinFile = "qnn/qnn_animeseg_v73.bin",
            sizeBytes = 90_789_176L + 814L,
            description = "Anime-Seg · ISNet · SM8550/V73 离线编译 · 87MB",
            kind = ModelKind.QNN,
            assetOnnx = "qnn/qnn_animeseg_v73.onnx",
            assetBin = "qnn/qnn_animeseg_v73.bin",
            deployOnnxName = "qnn_animeseg_v73.onnx", // ★=onnx 内 ep_cache_context 引用一致
            deployBinName = "qnn_animeseg_v73.bin" // ★=onnx 内 ep_cache_context 引用（已规范化）
        ),
        BuiltinModel(
            id = "qnn_modnet",
            name = "MODNet QNN-HTP",
            hfRepo = "zuimengqm/rmbg-qnn-modnet",
            hfFile = "qnn/qnn_modnet_sm8550_512.onnx",
            hfBinFile = "qnn/qnn_modnet_sm8550_512.bin",
            sizeBytes = 15_930_872L + 841L,
            description = "MODNet · 512×512 · SM8550/V73 离线编译 · 15MB",
            kind = ModelKind.QNN,
            assetOnnx = "qnn/qnn_modnet_sm8550_512.onnx",
            assetBin = "qnn/qnn_modnet_sm8550_512.bin",
            deployOnnxName = "qnn_modnet_sm8550_512.onnx", // App 规范名
            deployBinName = "qnn_modnet_sm8550_512.bin" // ★=onnx 内 ep_cache_context 引用（已规范化）
        ),
        BuiltinModel(
            id = "qnn_realesrgan",
            name = "Real-ESRGAN QNN-HTP",
            hfRepo = "zuimengqm/rmbg-qnn-realesrgan",
            hfFile = "qnn/qnn_realesrgan_sm8550_qairt250_fp32.zip",
            hfZip = "qnn/qnn_realesrgan_sm8550_qairt250_fp32.zip",
            sizeBytes = 30_034_102L,
            description = "4x 超分 · SM8550/V73 QNN HTP · 30MB（512→2048）",
            kind = ModelKind.QNN
        )
    )

    /** 全部内置模型（CPU + QNN），保持旧引用兼容
     *  ★ Lite 精简版：只保留 anime_seg（CPU）+ 三个 QNN（onnx 内置、bin 云下载），
     *    其余 CPU 模型（rmbg14_fp32/rmbg20/modnet）本来就是 HF 云下载，不内置也能用，
     *    但为贴合「只内置 anime」策略，Lite 下直接隐藏这些 CPU 模型列表项 */
    val builtinModels: List<BuiltinModel> = if (BuildConfig.IS_LITE) {
        listOf(
            cpuModels.find { it.id == "anime_seg" }!!,
            qnnModels.find { it.id == "qnn_rmbg14_v2" }!!,
            qnnModels.find { it.id == "qnn_animeseg" }!!,
            qnnModels.find { it.id == "qnn_modnet" }!!
        )
    } else {
        cpuModels + qnnModels
    }

    /** 本地导入的模型（非内置，用户通过 SAF 导入的 .onnx） */
    data class LocalModel(
        val id: String,
        val displayName: String,
        val fileName: String,
        val sizeBytes: Long
    )

    @Volatile
    private var _localModels: List<LocalModel>? = null

    /** 扫描本地导入的模型文件（models 目录下 model_local_*.onnx） */
    fun localModels(): List<LocalModel> {
        _localModels?.let { return it }
        val dir = modelDir()
        val list = dir.listFiles()
            ?.filter { it.name.startsWith("model_local_") && it.name.endsWith(".onnx") }
            ?.map { f ->
                // ★ 模型总大小 = onnx + 配套 EPContext bin（onnx 本身仅几百字节，bin 可达 1GB+，
                //   只显示 onnx 会显示 0KB 误导用户）
                var total = f.length()
                try {
                    val binRef = extractEpCacheContext(f)
                    if (binRef != null) {
                        val bin = File(dir, binRef)
                        if (bin.exists()) total += bin.length()
                    }
                } catch (_: Exception) {}
                LocalModel(
                    id = "local_${f.name.removePrefix("model_local_").removeSuffix(".onnx")}",
                    displayName = f.name.removePrefix("model_local_").removeSuffix(".onnx"),
                    fileName = f.name,
                    sizeBytes = total
                )
            } ?: emptyList()
        _localModels = list
        return list
    }

    /** 导入本地模型文件到模型目录（返回 id，失败返回 null） */
    fun importLocalModel(src: File, displayName: String): String? {
        return try {
            val safeName = displayName.replace(Regex("[^a-zA-Z0-9_-]"), "_").ifBlank { "model" }
            val dest = File(modelDir(), "model_local_$safeName.onnx")
            src.copyTo(dest, overwrite = true)
            _localModels = null // 失效缓存
            "local_$safeName"
        } catch (e: Exception) {
            android.util.Log.e("RMBG-MODEL", "导入失败", e)
            null
        }
    }

    /**
     * 从 zip 导入 QNN 模型（onnx + 配套 .bin 一起打包）。
     * 解压后把 .onnx 导入为本地模型，并把 onnx 内部 ep_cache_context 引用的 .bin
     * 放到模型目录（与 onnx 同目录，embed_mode=0 要求）。
     *
     * @param zipFile zip 源文件
     * @param displayName 模型显示名（zip 主 onnx 的 basename）
     * @return 导入的本地模型 id（失败返回 null）
     */
    fun importLocalModelZip(zipFile: File, displayName: String): String? {
        var result: String? = null
        val tmpDir = File(zipFile.parentFile ?: cacheDirForImport(), "zip_import_${System.currentTimeMillis()}")
        try {
            tmpDir.mkdirs()
            // 1) 解压 zip（限制总量防 zip 炸弹；QNN EPContext 的 bin 可达 1GB+，上限放宽到 4GB）
            var totalBytes = 0L
            java.util.zip.ZipFile(zipFile).use { zf ->
                val entries = zf.entries()
                while (entries.hasMoreElements()) {
                    val entry = entries.nextElement()
                    val name = entry.name
                    if (entry.isDirectory) continue
                    // 只处理 onnx 与 bin（其余如 meta 忽略）
                    if (!name.endsWith(".onnx", ignoreCase = true) &&
                        !name.endsWith(".bin", ignoreCase = true)) continue
                    totalBytes += entry.size
                    if (totalBytes > 4L * 1024 * 1024 * 1024) { // 4GB 上限（容纳 1GB+ 的 QNN bin，防 zip 炸弹）
                        android.util.Log.e("RMBG-MODEL", "zip 解压超限，中止")
                        return@importLocalModelZip null
                    }
                    // 防路径穿越：只取文件名
                    val baseName = name.substringAfterLast('/').substringAfterLast('\\')
                    if (baseName.isEmpty()) continue
                    val outFile = File(tmpDir, baseName)
                    zf.getInputStream(entry).use { input ->
                        outFile.outputStream().use { out -> input.copyTo(out) }
                    }
                }
            }

            // 2) 找出主 onnx（优先用 displayName 对应的，否则第一个 .onnx）
            val onnxFiles = tmpDir.listFiles { _, n -> n.endsWith(".onnx", ignoreCase = true) }
                ?: emptyArray()
            if (onnxFiles.isEmpty()) {
                android.util.Log.e("RMBG-MODEL", "zip 内无 .onnx 文件")
                return@importLocalModelZip null
            }
            val mainOnnx = onnxFiles.firstOrNull { it.nameWithoutExtension == displayName }
                ?: onnxFiles[0]

            // ★ bin 引用规范化说明：QNN SDK 在开发机导出的 EPContext 其 ep_cache_context 可能是绝对路径，
            //   ORT 1.30 强制要求 ep_cache_context 必须是相对路径（否则 createSession 报 ORT_INVALID_GRAPH）。
            //   因此：① bin 以 basename 复制到模型目录（embed_mode=0 要求同目录同名）
            //        ② onnx 内嵌绝对路径在导入时用 normalizeEpContextPath 改写为 basename。

            // 3) 导入 onnx 为本地模型
            val safeName = displayName.replace(Regex("[^a-zA-Z0-9_-]"), "_").ifBlank { "model" }
            val destOnnx = File(modelDir(), "model_local_$safeName.onnx")
            mainOnnx.copyTo(destOnnx, overwrite = true)

            // 4) 若 onnx 是 EPContext，提取引用的 bin 名，并从解压内容中找到对应 bin 放置到模型目录
            //    ★ 匹配策略（QNN SDK 命名差异兼容）：
            //      ① 精确等于 binRef（如 rmbg20.bin）
            //      ② 文件名包含 binRef（如 rmbg20_net_qnn_ctx.bin 对应引用 rmbg20.bin）
            //      ③ 唯一 .bin 兜底（onnx 引用名与 zip 内命名差异较大时）
            val binRef = extractEpCacheContext(mainOnnx)
            val binFiles = tmpDir.listFiles { _, n -> n.endsWith(".bin", ignoreCase = true) } ?: emptyArray()
            val binSrc = when {
                binRef != null -> binFiles.firstOrNull { it.name.equals(binRef, ignoreCase = true) }
                    ?: binFiles.firstOrNull { it.name.contains(binRef, ignoreCase = true) }
                    ?: binFiles.firstOrNull { binRef.contains(it.nameWithoutExtension, ignoreCase = true) }
                else -> null
            } ?: binFiles.firstOrNull() // 无引用或匹配失败时，唯一 bin 兜底
            if (binSrc != null) {
                // bin 名必须以 onnx 内部引用为准（embed_mode=0 要求），复制并改名
                val destName = binRef ?: binSrc.name
                val destBin = File(modelDir(), destName)
                binSrc.copyTo(destBin, overwrite = true)
                if (binRef == null) {
                    android.util.Log.w("RMBG-MODEL", "EPContext 未提取到 bin 引用名，按解压名原样放置: ${binSrc.name}")
                }
                // ★★ 关键：把 onnx 内嵌的绝对/子目录路径改写为 basename（ORT 1.30 强制相对路径）
                if (normalizeEpContextPath(destOnnx)) {
                    android.util.Log.i("RMBG-MODEL", "zip 导入 EPContext 路径已规范化 -> $destName")
                }
                // ★ 检测 bin 是否为 QNN 工具链 tar 权重包（POSIX tar，偏移 257 处有 "ustar"）：
                //   ORT QNN EP 只认 qnn-context-binary-generator 直接输出的 context binary（v2 二进制），
                //   不认 tar 权重包 → createSession 报 "Failed to get context binary info"。
                //   此处提前识别并给出清晰提示（不影响导入，用户可据此换正确模型包）。
                try {
                    if (destBin.length() > 300) {
                        val head = destBin.readBytes().take(300).toByteArray()
                        val isTar = head[257] == 'u'.code.toByte() && head[258] == 's'.code.toByte() &&
                            head[259] == 't'.code.toByte() && head[260] == 'a'.code.toByte() && head[261] == 'r'.code.toByte()
                        if (isTar) {
                            android.util.Log.w("RMBG-MODEL",
                                "⚠️ 检测到 $destName 是 QNN 工具链 tar 权重包（POSIX tar），ORT QNN EP 无法加载（需要 context binary）。请使用 qnn-context-binary-generator 输出的 .bin")
                        }
                    }
                } catch (_: Exception) {}
            } else if (binRef != null) {
                android.util.Log.e("RMBG-MODEL", "EPContext 引用 bin=$binRef 但 zip 内无 .bin 文件")
            }

            _localModels = null // 失效缓存
            result = "local_$safeName"
        } catch (e: Exception) {
            android.util.Log.e("RMBG-MODEL", "zip 导入失败", e)
            result = null
        } finally {
            try { tmpDir.deleteRecursively() } catch (_: Exception) {}
        }
        return result
    }


    /** zip 导入时临时目录兜底（zipFile.parentFile 可能为 null 时的临时缓存） */
    private fun cacheDirForImport(): File = File(System.getProperty("java.io.tmpdir") ?: ".", "rmbg_import_cache").apply { mkdirs() }


    /**
     * 从 zip 导入【超分模型】（onnx + 配套 .bin 一起打包，QNN EPContext 或普通 ONNX 皆可）。
     * ★ 与抠图模型隔离：解压到 modelDir()/superres/ 子目录，onnx 固定名 superres_model.onnx，
     *    bin 按 onnx 内 ep_cache_context 引用名放置。绝不进入 localModels()（其只扫顶层 model_local_*）。
     *
     * @param zipFile zip 源文件
     * @return 部署后的 onnx 绝对路径（失败返回 null）
     */
    fun importSuperResModelZip(zipFile: File): String? {
        val srDir = File(modelDir(), "superres").apply { mkdirs() }
        val tmpDir = File(zipFile.parentFile ?: cacheDirForImport(), "sr_zip_import_${System.currentTimeMillis()}")
        return try {
            tmpDir.mkdirs()
            // 1) 解压 zip（限制总量；QNN bin 可达 1GB+，放宽 4GB）
            var totalBytes = 0L
            java.util.zip.ZipFile(zipFile).use { zf ->
                val entries = zf.entries()
                while (entries.hasMoreElements()) {
                    val entry = entries.nextElement()
                    if (entry.isDirectory) continue
                    val name = entry.name
                    if (!name.endsWith(".onnx", ignoreCase = true) && !name.endsWith(".bin", ignoreCase = true)) continue
                    totalBytes += entry.size
                    if (totalBytes > 4L * 1024 * 1024 * 1024) {
                        android.util.Log.e("RMBG-MODEL", "超分 zip 解压超限，中止")
                        return@importSuperResModelZip null
                    }
                    val baseName = name.substringAfterLast('/').substringAfterLast('\\')
                    if (baseName.isEmpty()) continue
                    val outFile = File(tmpDir, baseName)
                    zf.getInputStream(entry).use { input -> outFile.outputStream().use { out -> input.copyTo(out) } }
                }
            }

            // 2) 找主 onnx（第一个 .onnx）
            val onnxFiles = tmpDir.listFiles { _, n -> n.endsWith(".onnx", ignoreCase = true) } ?: emptyArray()
            if (onnxFiles.isEmpty()) {
                android.util.Log.e("RMBG-MODEL", "超分 zip 内无 .onnx 文件")
                return@importSuperResModelZip null
            }
            val mainOnnx = onnxFiles[0]

            // 3) 部署 onnx（固定名 superres_model.onnx）
            val destOnnx = File(srDir, "superres_model.onnx")
            mainOnnx.copyTo(destOnnx, overwrite = true)

            // 4) 若为 EPContext，提取 bin 引用并放置同目录（embed_mode=0 要求 onnx/bin 同目录）
            val binRef = extractEpCacheContext(mainOnnx)
            val binFiles = tmpDir.listFiles { _, n -> n.endsWith(".bin", ignoreCase = true) } ?: emptyArray()
            val binSrc = when {
                binRef != null -> binFiles.firstOrNull { it.name.equals(binRef, ignoreCase = true) }
                    ?: binFiles.firstOrNull { it.name.contains(binRef, ignoreCase = true) }
                    ?: binFiles.firstOrNull { binRef.contains(it.nameWithoutExtension, ignoreCase = true) }
                else -> null
            } ?: binFiles.firstOrNull()
            if (binSrc != null) {
                val destName = binRef ?: binSrc.name
                binSrc.copyTo(File(srDir, destName), overwrite = true)
                // ★ 规范化 onnx 内嵌 ep_cache_context 为相对路径
                normalizeEpContextPath(destOnnx)
                android.util.Log.i("RMBG-MODEL", "超分 EPContext 已部署 bin=$destName")
            } else if (binRef != null) {
                android.util.Log.e("RMBG-MODEL", "超分 EPContext 引用 bin=$binRef 但 zip 内无 .bin")
            }

            if (destOnnx.exists()) destOnnx.absolutePath else null
        } catch (e: Exception) {
            android.util.Log.e("RMBG-MODEL", "超分 zip 导入失败", e)
            null
        } finally {
            try { tmpDir.deleteRecursively() } catch (_: Exception) {}
        }
    }


    /** 超分模型内置仓库（HF）与 zip 路径 */
    const val SUPER_RES_HF_REPO = "zuimengqm/rmbg-qnn-realesrgan"
    const val SUPER_RES_HF_ZIP = "qnn/qnn_realesrgan_sm8550_qairt250_fp32.zip"
    /** 内置 CPU 超分资产路径（普通 ONNX，免下载） */
    const val SUPER_RES_CPU_ASSET = "models/realesrgan_anime6b.onnx"

    /** 超分模型部署目录（与抠图隔离：modelDir()/superres/） */
    fun superResDir(): File = File(modelDir(), "superres").apply { mkdirs() }

    /**
     * ★ 超分模型云下载（智能路径）：
     *  - QNN 可用且当前设备匹配 SM8550 → 从 HF 下载超分 QNN zip（onnx+bin）→ 部署到 superres/ → 返回 onnx 绝对路径
     *  - 否则（CPU/无 QNN）→ 返回 null（由调用方回退内置 CPU 资产 SUPER_RES_CPU_ASSET，免下载）
     * @return 部署后的超分 onnx 绝对路径（QNN）；null 表示回退 CPU
     */
    suspend fun downloadSuperResModel(
        context: Context,
        listener: ProgressListener? = null
    ): String? = withContext(Dispatchers.IO) {
        initContext(context)
        // ★ 已部署（superres/superres_model.onnx 存在）→ 直接返回，免重复下载
        val srDir = superResDir()
        val existing = File(srDir, "superres_model.onnx")
        if (existing.exists() && existing.length() > 100) {
            return@withContext existing.absolutePath
        }
        // 设备不匹配 SM8550 → 无法用当前 QNN 超分，返回 null 走 CPU
        if (!deviceSocModel().uppercase().contains("SM8550")) {
            return@withContext null
        }
        val zipDst = File(modelDir(), "qnn_realesrgan_sm8550_qairt250_fp32.zip")
        val ok = downloadModel(
            context = context,
            hfRepo = SUPER_RES_HF_REPO,
            hfFile = SUPER_RES_HF_ZIP,
            destFile = zipDst,
            listener = listener
        )
        if (!ok || !zipDst.exists() || zipDst.length() < 1_000_000L) {
            zipDst.delete()
            return@withContext null
        }
        val path = importSuperResModelZip(zipDst)
        zipDst.delete() // 部署后清理源 zip
        path
    }

    /** 超分模型是否已就绪（导入或部署到 superres/ 的 onnx 存在） */
    fun isSuperResReady(): Boolean {
        val f = File(superResDir(), "superres_model.onnx")
        return f.exists() && f.length() > 100
    }

    /** 删除已部署/导入的超分模型文件（superres/ 目录） */
    fun deleteSuperResModel(): Boolean {
        return try {
            val dir = superResDir()
            var ok = true
            dir.listFiles()?.forEach { ok = it.delete() && ok }
            ok
        } catch (_: Exception) { false }
    }

    // ================= Anime-Seg 多 NPU 变体下载（按设备 SoC 自动匹配）=================
    /** Anime-Seg 各 SoC 变体 → HF zip（SM8550 无 zip，走内置 v73 双文件） */
    fun animeSegZipForSoc(soc: String): String? = when (soc.uppercase()) {
        "SM8350" -> "qnn/qnn_animeseg_sm8350_qairt250_fp32.zip"
        "SM8450" -> "qnn/qnn_animeseg_sm8450_qairt250_fp32.zip"
        "SM8650" -> "qnn/qnn_animeseg_sm8650_qairt250_fp32.zip"
        "SM8750" -> "qnn/qnn_animeseg_sm8750_qairt250_fp32.zip"
        "SM8850" -> "qnn/qnn_animeseg_sm8850_qairt250_fp32.zip"
        else -> null   // SM8550 或未知 → 用内置 v73（assets 兜底 + 双文件下载）
    }

    /**
     * ★ Anime-Seg 多 NPU 变体下载（按设备 SoC 自动匹配对应 QNN 编译产物）：
     *  - SM8350/8450/8650/8750/8850 → 下载对应 SoC 的 qairt250 zip → 解压 → 统一部署为
     *    qnn_animeseg_v73.onnx/.bin（改写 onnx 内 ep_cache_context 引用），引擎固定加载 v73 名。
     *  - SM8550 / 未知 SoC → 用内置 assets 的 v73 双文件（ensureQnnContextDual 兜底）。
     * @return 是否部署成功
     */
    suspend fun downloadAnimeSeg(
        context: Context,
        listener: ProgressListener? = null
    ): Boolean = withContext(Dispatchers.IO) {
        initContext(context)
        val bm = builtinModels.find { it.id == "qnn_animeseg" } ?: return@withContext false
        val onnxDst = qnnContextFileFor(bm.id) ?: return@withContext false
        val binDst = qnnContextBinFileFor(bm.id) ?: return@withContext false

        // ① 已部署（v73 onnx+bin 就位）→ 直接完成
        if (onnxDst.exists() && onnxDst.length() > 100 && binDst.exists() && binDst.length() > 1_000_000) {
            return@withContext true
        }

        val soc = deviceSocModel().uppercase()
        val zip = animeSegZipForSoc(soc)
        // ② 无对应 SoC zip（SM8550/未知）→ 走内置 v73 assets + 双文件下载
        if (zip == null) {
            return@withContext ensureQnnContextDual(context, "qnn_animeseg", listener)
        }

        // ③ 有对应 SoC zip → 下载 + 解压 + 统一部署为 v73 名
        val dir = modelDir().also { it.mkdirs() }
        val zipDst = File(dir, "qnn_animeseg_${soc}.zip")
        val mainHandler = Handler(Looper.getMainLooper())
        val zipOk = downloadModel(
            context = context,
            hfRepo = bm.hfRepo,
            hfFile = zip,
            destFile = zipDst,
            listener = listener
        )
        if (!zipOk || !zipDst.exists() || zipDst.length() < 1_000_000L) {
            zipDst.delete()
            return@withContext false
        }

        // 解压 zip → 找 onnx/bin → 部署为 v73 名（改写 ep_cache_context 引用）
        val tmpDir = File(dir, "animeseg_zip_import_${System.currentTimeMillis()}")
        var deployed = false
        try {
            tmpDir.mkdirs()
            var totalBytes = 0L
            java.util.zip.ZipFile(zipDst).use { zf ->
                val entries = zf.entries()
                while (entries.hasMoreElements()) {
                    val entry = entries.nextElement()
                    if (entry.isDirectory) continue
                    val name = entry.name
                    if (!name.endsWith(".onnx", ignoreCase = true) && !name.endsWith(".bin", ignoreCase = true)) continue
                    totalBytes += entry.size
                    if (totalBytes > 2L * 1024 * 1024 * 1024) return@use
                    val baseName = name.substringAfterLast('/').substringAfterLast('\\')
                    if (baseName.isEmpty()) continue
                    zf.getInputStream(entry).use { input ->
                        File(tmpDir, baseName).outputStream().use { out -> input.copyTo(out) }
                    }
                }
            }
            val onnxFile = tmpDir.listFiles { _, n -> n.endsWith(".onnx", ignoreCase = true) }?.firstOrNull()
            val binFile = tmpDir.listFiles { _, n -> n.endsWith(".bin", ignoreCase = true) }?.firstOrNull()
            if (onnxFile != null && binFile != null) {
                val destOnnx = File(dir, "qnn_animeseg_v73.onnx")
                val destBin = File(dir, "qnn_animeseg_v73.bin")
                onnxFile.copyTo(destOnnx, overwrite = true)
                binFile.copyTo(destBin, overwrite = true)
                // ★ 改写 onnx 内 ep_cache_context 引用为 v73.bin（各 SoC 内部名不同）
                normalizeEpContextPath(destOnnx, "qnn_animeseg_v73.bin")
                deployed = destOnnx.length() > 100 && destBin.length() > 1_000_000
                if (deployed) {
                    android.util.Log.i("RMBG-MODEL", "Anime-Seg 已按 SoC=$soc 部署为 v73 名")
                }
            }
        } catch (e: Exception) {
            android.util.Log.e("RMBG-MODEL", "Anime-Seg SoC zip 解压部署失败", e)
        } finally {
            zipDst.delete()
            try { tmpDir.deleteRecursively() } catch (_: Exception) {}
        }
        deployed
    }

    /** 删除本地导入的模型 */
    fun deleteLocalModel(id: String): Boolean {
        val f = File(modelDir(), "model_$id.onnx")
        val ok = f.delete()
        _localModels = null
        _compatCache.remove(id) // 清兼容性缓存
        return ok
    }

    // ================= 模型加速兼容性探测（导入时标注：QNN/NNAPI 支不支持）=================
    /** 加速支持程度 */
    enum class AccelSupport {
        QNN_EPCONTEXT,  // EPContext 形态，QNN 强推荐
        QNN_TRY,        // QDQ 量化且无风险算子，QNN 可尝试（失败自动降级 CPU）
        QNN_RISKY,      // 含 HTP 风险算子，QNN 大概率不支持
        CPU_ONLY        // 无量化 / 完全不适用，仅 CPU/NNAPI
    }

    /** 探测结果（供 UI 标注展示） */
    data class ModelCompatibility(
        val support: AccelSupport,
        val qnnLabel: String,
        val nnapiLabel: String,
        val desc: String
    )

    /** 本地模型兼容性缓存（按 id） */
    private val _compatCache = java.util.concurrent.ConcurrentHashMap<String, ModelCompatibility>()

    /** 清缓存（导入/删除后调用） */
    fun clearCompatCache() { _compatCache.clear() }

    /**
     * 探测本地模型加速兼容性。
     * @param deep 是否做全文件字节扫描（IO 重，需后台线程）；false 只按文件名快判
     */
    fun probeLocalModelCompat(lm: LocalModel, deep: Boolean = true): ModelCompatibility {
        _compatCache[lm.id]?.let { return it }
        val compat = computeCompat(File(modelDir(), lm.fileName), deep)
        _compatCache[lm.id] = compat
        return compat
    }

    /** 探测任意 .onnx 文件 */
    fun probeModelCompat(file: File, deep: Boolean = true): ModelCompatibility = computeCompat(file, deep)

    private fun computeCompat(file: File, deep: Boolean): ModelCompatibility {
        // 1) EPContext：onnx 内容含 EPContext 节点 + 配套 .bin 存在 → QNN 强推荐（不再死认文件名前缀）
        val isEpContext = scanForBytes(file, listOf("EPContext", "ep_cache_context"))
        if (isEpContext) {
            val binName = extractEpCacheContext(file)
            val binOk = binName != null && File(modelDir(), binName).exists()
            if (binOk) {
                return ModelCompatibility(
                    AccelSupport.QNN_EPCONTEXT,
                    "✅ QNN (EPContext)",
                    "❌ NNAPI 不适用",
                    "骁龙 DSP 离线编译产物，最推荐"
                )
            }
            return ModelCompatibility(
                AccelSupport.QNN_EPCONTEXT,
                "✅ QNN (EPContext)",
                "❌ NNAPI 不适用",
                "EPContext 产物（缺 .bin，需部署）"
            )
        }
        val name = file.name.lowercase()
        // 2) 文件名快判：含量化关键词 → 疑似 QDQ，deep 扫描进一步确认
        val nameHintQdq = listOf("qnn", "qdq", "quant", "int8", "uint8", "s8w", "16a8w")
            .any { name.contains(it) }
        // 3) deep 扫描：QDQ 量化算子 + HTP 风险算子
        val hasQdq = nameHintQdq || scanForBytes(file, listOf("QuantizeLinear", "DequantizeLinear", "ConvInteger"))
        if (!hasQdq) {
            return ModelCompatibility(
                AccelSupport.CPU_ONLY,
                "❌ QNN 不适用",
                "✅ NNAPI 可尝试",
                "无量化（FP32/FP16），HTP 仅吃 QDQ 量化模型"
            )
        }
        // 4) QDQ 模型：深扫 HTP 风险算子（不支持就标激进，避免在线编译 abort）
        if (deep && scanForBytes(file, HTP_RISKY_OPS)) {
            return ModelCompatibility(
                AccelSupport.QNN_RISKY,
                "⚠️ QNN 有风险",
                "❌ NNAPI 不适用",
                "含 HTP 风险算子，在线编译可能失败，建议仅 CPU"
            )
        }
        return ModelCompatibility(
            AccelSupport.QNN_TRY,
            "⚠️ QNN 可尝试",
            "❌ NNAPI 不适用",
            "QDQ 量化模型，QNN 会尝试 HTP，失败自动降级 CPU"
        )
    }

    /** HTP 大概率不支持的算子（白名单外风险项） */
    private val HTP_RISKY_OPS = listOf(
        "RandomNormal", "RandomUniform", "Multinomial", "NonMaxSuppression", "RoiAlign",
        "If", "Loop", "Scan", "DynamicQuantizeLinear", "StringNormalizer", "Unique",
        "EyeLike", "Trilu", "BitShift", "TopK", "NonZero", "CumSum", "RandomNormalLike",
        "RandomUniformLike", "Det", "LpPool", "MaxRoiPool", "Mod", "Round", "Sign"
    )

    /** 从 EPContext onnx 内容提取 ep_cache_context 属性值（bin 文件名）。
     *  protobuf 结构：... 0a <len> "ep_cache_context" 22 <len> "<bin文件名>" a0 01 03 ...
     *  只取字段 2(s) 的 string，且校验是合法文件名，防止误读相邻字段。
     *  ★ 兼容绝对路径：QNN SDK 导出的 EPContext 常把 bin 引用写成完整路径
     *   （如 /run/csi/.../models/rmbg2/rmbg20.bin），此时提取 basename（rmbg20.bin）。 */
    fun extractEpCacheContext(file: File): String? {
        if (!file.exists() || file.length() == 0L || file.length() > 8L * 1024 * 1024) return null
        return try {
            val bytes = file.readBytes()
            val key = "ep_cache_context".toByteArray(Charsets.UTF_8)
            val start = indexOfBytes(bytes, key, 0, bytes.size)
            if (start < 0) return null
            // 从 key 后开始找 0x22（field 2, wire type 2 = length-delimited string）
            var i = start + key.size
            while (i < bytes.size - 2) {
                if (bytes[i] == 0x22.toByte()) {
                    val len = bytes[i + 1].toInt() and 0xFF
                    if (len > 0 && i + 2 + len <= bytes.size) {
                        val raw = String(bytes, i + 2, len, Charsets.UTF_8)
                        // 取 basename（兼容绝对路径），再校验合法文件名字符且以 .bin 结尾
                        val name = raw.substringAfterLast('/').substringAfterLast('\\')
                        if (name.endsWith(".bin") && name.length > 4 &&
                            name.all { it.isLetterOrDigit() || it == '_' || it == '-' || it == '.' }) {
                            return name
                        }
                    }
                }
                i++
            }
            null
        } catch (_: Exception) { null }
    }

        /**
     * 规范化 EPContext onnx 内嵌的 ep_cache_context 路径：绝对路径 → 相对 basename。
     *
     * ★★ 为什么必须做：ORT 1.30 加载 EPContext（embed_mode=0 外部模式）时强制要求
     *    ep_cache_context 必须是相对路径（报错 ORT_INVALID_GRAPH：
     *    "External mode should set ep_cache_context field with a relative path"）。
     *    QNN SDK 在开发机导出的 EPContext 常内嵌绝对路径（如 /run/csi/.../rmbg20.bin），
     *    直接用会 createSession 失败。runtime 按 basename 在 onnx 同目录找 bin（embed_mode=0），
     *    因此只需把绝对路径改写为 basename 即可，bin 复制为同名放在 onnx 同目录。
     *
     * ★★ 实现必须是"类型感知的 protobuf 重编码"，绝不能字符串级局部替换：
     *    protobuf 是嵌套结构，ep_cache_context 位于 AttributeProto(2a) 内，而 AttributeProto
     *    又位于 NodeProto(0a) 内，NodeProto 又位于 GraphProto(0a) 内，GraphProto 又在
     *    ModelProto(3a) 内。任何内层长度变化都必须同步更新所有外层的 length 前缀，
     *    局部替换会导致外层前缀失配 → ORT 报 ORT_INVALID_PROTOBUF（已踩坑验证）。
     *    本实现按 ONNX 类型层级（Model→Graph→Node→Attribute）逐层解析重编码：
     *      - Model(level0)：field 7(0x3a) 是 Graph → 递归
     *      - Graph(level1)：field 1(0x0a) 是 Node 列表 → 逐个递归；其余字段原样拷贝
     *      - Node(level2)：field 5(0x2a) 是 Attribute 列表 → 逐个递归；其余字段原样拷贝
     *      - Attribute(level3)：读 field 1(0x0a) name；若 name==ep_cache_context 且
     *        field 4(0x22) s 含目录分隔符 → 替换为 basename 并重新编码
     *    这样每层长度前缀都按重编码后真实长度重新计算，结构始终合法
     *    （与 onnx 库输出字节级一致，已用 python 验证 + 幂等）。
     *
     * @param targetBinName 可选：直接把 ep_cache_context 引用改写为指定 bin 文件名（用于各 SoC 变体 zip 部署时统一改名；
     *                       否则沿用"绝对路径 → basename"逻辑）
     * @return 是否发生了重写（true = 已改写）
     */
    fun normalizeEpContextPath(file: File, targetBinName: String? = null): Boolean {
        if (!file.exists() || file.length() == 0L || file.length() > 64L * 1024 * 1024) return false
        try {
            val bytes = file.readBytes()
            if (bytes.size < 16) return false
            val changed = BooleanArray(1)

            fun writeVarint(out: java.io.ByteArrayOutputStream, v: Int) {
                var n = v
                while (true) {
                    var b = (n and 0x7F)
                    n = n shr 7
                    if (n != 0) b = b or 0x80
                    out.write(b)
                    if (n == 0) break
                }
            }
            fun readVarint(buf: ByteArray, pos: Int): LongArray {
                var result = 0L
                var shift = 0
                var i = pos
                while (i < buf.size && shift < 64) {
                    val b = buf[i].toInt() and 0xFF
                    result = result or ((b.toLong() and 0x7F) shl shift)
                    if (b and 0x80 == 0) return longArrayOf(result, (i - pos + 1).toLong())
                    shift += 7
                    i++
                }
                return longArrayOf(result, (i - pos + 1).toLong())
            }
            fun copyField(out: java.io.ByteArrayOutputStream, field: Int, wire: Int, payload: ByteArray) {
                writeVarint(out, (field shl 3) or wire)
                if (wire == 2) {
                    writeVarint(out, payload.size)
                    out.write(payload)
                } else {
                    out.write(payload)
                }
            }
            // AttributeProto 重编码：读 name(field1)，若 ep_cache_context 替换 s(field4)
            fun rewriteAttribute(buf: ByteArray): ByteArray {
                val out = java.io.ByteArrayOutputStream()
                var name: String? = null
                var pos = 0
                while (pos < buf.size) {
                    val (tag, tagLen) = readVarint(buf, pos)
                    pos += tagLen.toInt()
                    val field = (tag shr 3).toInt()
                    val wire = (tag and 0x7L).toInt()
                    when (wire) {
                        0 -> {
                            val (v, vl) = readVarint(buf, pos)
                            writeVarint(out, (field shl 3) or 0)
                            writeVarint(out, v.toInt())
                            pos += vl.toInt()
                        }
                        1 -> { copyField(out, field, 1, buf.copyOfRange(pos, pos + 8)); pos += 8 }
                        5 -> { copyField(out, field, 5, buf.copyOfRange(pos, pos + 4)); pos += 4 }
                        2 -> {
                            val (len, lenLen) = readVarint(buf, pos)
                            val pStart = pos + lenLen.toInt()
                            val pEnd = pStart + len.toInt()
                            val payload = buf.copyOfRange(pStart, pEnd)
                            if (field == 1) {
                                name = String(payload, Charsets.UTF_8)
                                copyField(out, field, 2, payload)
                            } else if (field == 4 && name == "ep_cache_context") {
                                val s = String(payload, Charsets.UTF_8)
                                // ★ 指定目标 bin 名：直接把引用改写为目标名（用于各 SoC 变体统一部署名）
                                if (targetBinName != null) {
                                    if (s != targetBinName) {
                                        copyField(out, field, 2, targetBinName.toByteArray(Charsets.UTF_8))
                                        changed[0] = true
                                    } else {
                                        copyField(out, field, 2, payload)
                                    }
                                    pos = pEnd
                                    continue
                                }
                                if (s.contains('/') || s.contains('\\')) {
                                    val base = s.substringAfterLast('/').substringAfterLast('\\')
                                    if (base.endsWith(".bin") && base.length > 4 &&
                                        base.all { it.isLetterOrDigit() || it == '_' || it == '-' || it == '.' }) {
                                        copyField(out, field, 2, base.toByteArray(Charsets.UTF_8))
                                        changed[0] = true
                                        pos = pEnd
                                        continue
                                    }
                                }
                                copyField(out, field, 2, payload)
                            } else {
                                copyField(out, field, 2, payload)
                            }
                            pos = pEnd
                        }
                        else -> { out.write(buf, pos - tagLen.toInt(), buf.size - (pos - tagLen.toInt())); pos = buf.size }
                    }
                }
                return out.toByteArray()
            }
            // NodeProto 重编码：field5(0x2a) attribute 列表递归；其余原样
            fun rewriteNode(buf: ByteArray): ByteArray {
                val out = java.io.ByteArrayOutputStream()
                var pos = 0
                while (pos < buf.size) {
                    val (tag, tagLen) = readVarint(buf, pos)
                    pos += tagLen.toInt()
                    val field = (tag shr 3).toInt()
                    val wire = (tag and 0x7L).toInt()
                    when (wire) {
                        0 -> {
                            val (v, vl) = readVarint(buf, pos)
                            writeVarint(out, (field shl 3) or 0)
                            writeVarint(out, v.toInt())
                            pos += vl.toInt()
                        }
                        1 -> { copyField(out, field, 1, buf.copyOfRange(pos, pos + 8)); pos += 8 }
                        5 -> { copyField(out, field, 5, buf.copyOfRange(pos, pos + 4)); pos += 4 }
                        2 -> {
                            val (len, lenLen) = readVarint(buf, pos)
                            val pStart = pos + lenLen.toInt()
                            val pEnd = pStart + len.toInt()
                            val payload = buf.copyOfRange(pStart, pEnd)
                            val newPayload = if (field == 5) rewriteAttribute(payload) else payload
                            copyField(out, field, 2, newPayload)
                            pos = pEnd
                        }
                        else -> { out.write(buf, pos - tagLen.toInt(), buf.size - (pos - tagLen.toInt())); pos = buf.size }
                    }
                }
                return out.toByteArray()
            }
            // GraphProto 重编码：field1(0x0a) node 列表递归；其余原样
            fun rewriteGraph(buf: ByteArray): ByteArray {
                val out = java.io.ByteArrayOutputStream()
                var pos = 0
                while (pos < buf.size) {
                    val (tag, tagLen) = readVarint(buf, pos)
                    pos += tagLen.toInt()
                    val field = (tag shr 3).toInt()
                    val wire = (tag and 0x7L).toInt()
                    when (wire) {
                        0 -> {
                            val (v, vl) = readVarint(buf, pos)
                            writeVarint(out, (field shl 3) or 0)
                            writeVarint(out, v.toInt())
                            pos += vl.toInt()
                        }
                        1 -> { copyField(out, field, 1, buf.copyOfRange(pos, pos + 8)); pos += 8 }
                        5 -> { copyField(out, field, 5, buf.copyOfRange(pos, pos + 4)); pos += 4 }
                        2 -> {
                            val (len, lenLen) = readVarint(buf, pos)
                            val pStart = pos + lenLen.toInt()
                            val pEnd = pStart + len.toInt()
                            val payload = buf.copyOfRange(pStart, pEnd)
                            val newPayload = if (field == 1) rewriteNode(payload) else payload
                            copyField(out, field, 2, newPayload)
                            pos = pEnd
                        }
                        else -> { out.write(buf, pos - tagLen.toInt(), buf.size - (pos - tagLen.toInt())); pos = buf.size }
                    }
                }
                return out.toByteArray()
            }
            // ModelProto 重编码：field7(0x3a) graph 递归；其余原样
            fun rewriteModel(buf: ByteArray): ByteArray {
                val out = java.io.ByteArrayOutputStream()
                var pos = 0
                while (pos < buf.size) {
                    val (tag, tagLen) = readVarint(buf, pos)
                    pos += tagLen.toInt()
                    val field = (tag shr 3).toInt()
                    val wire = (tag and 0x7L).toInt()
                    when (wire) {
                        0 -> {
                            val (v, vl) = readVarint(buf, pos)
                            writeVarint(out, (field shl 3) or 0)
                            writeVarint(out, v.toInt())
                            pos += vl.toInt()
                        }
                        1 -> { copyField(out, field, 1, buf.copyOfRange(pos, pos + 8)); pos += 8 }
                        5 -> { copyField(out, field, 5, buf.copyOfRange(pos, pos + 4)); pos += 4 }
                        2 -> {
                            val (len, lenLen) = readVarint(buf, pos)
                            val pStart = pos + lenLen.toInt()
                            val pEnd = pStart + len.toInt()
                            val payload = buf.copyOfRange(pStart, pEnd)
                            val newPayload = if (field == 7) rewriteGraph(payload) else payload
                            copyField(out, field, 2, newPayload)
                            pos = pEnd
                        }
                        else -> { out.write(buf, pos - tagLen.toInt(), buf.size - (pos - tagLen.toInt())); pos = buf.size }
                    }
                }
                return out.toByteArray()
            }

            val newBytes = rewriteModel(bytes)
            if (!changed[0]) return false
            // 原子写回（同目录 tmp + rename，避免写一半损坏）
            val tmp = File(file.parentFile, file.name + ".tmp_norm")
            try {
                tmp.writeBytes(newBytes)
                if (tmp.renameTo(file) || (file.delete() && tmp.renameTo(file))) {
                    android.util.Log.i("RMBG-MODEL", "EPContext 路径已规范化 (${file.name})")
                    return true
                }
                tmp.delete()
            } catch (_: Exception) {
                try { tmp.delete() } catch (_: Exception) {}
            }
            return false
        } catch (_: Exception) {
            return false
        }
    }

    /** protobuf varint 编码字节数 */
    private fun varintSize(v: Int): Int {
        var n = v
        var c = 1
        while (n >= 0x80) { n = n shr 7; c++ }
        return c
    }

    /** 分块扫描文件字节，查找任一字面量（ONNX 算子 op_type 为明文 UTF-8） */
    private fun scanForBytes(file: File, needles: List<String>): Boolean {
        if (!file.exists() || file.length() == 0L) return false
        val patterns = needles.mapNotNull { it.toByteArray(Charsets.UTF_8).takeIf { b -> b.isNotEmpty() } }
        if (patterns.isEmpty()) return false
        val overlap = patterns.maxOf { it.size } - 1
        val chunk = 1 shl 20 // 1MB
        return try {
            java.io.BufferedInputStream(java.io.FileInputStream(file), chunk).use { input ->
                val buf = ByteArray(chunk + overlap)
                var carry = 0
                while (true) {
                    val n = input.read(buf, carry, chunk)
                    if (n <= 0) break
                    val total = carry + n
                    for (p in patterns) {
                        if (indexOfBytes(buf, p, 0, total) >= 0) return true
                    }
                    // 保留尾部 overlap 字节作为下一次的头部（跨块匹配）
                    carry = if (total > overlap) overlap else total
                    System.arraycopy(buf, total - carry, buf, 0, carry)
                }
                // 处理最后残段
                for (p in patterns) if (indexOfBytes(buf, p, 0, carry) >= 0) return true
                false
            }
        } catch (_: Exception) { false }
    }

    /** 朴素字节查找（hay[pos] 与 needle 完全匹配） */
    private fun indexOfBytes(hay: ByteArray, needle: ByteArray, start: Int, end: Int): Int {
        if (needle.isEmpty() || end - start < needle.size) return -1
        outer@ for (i in start..end - needle.size) {
            for (j in needle.indices) {
                if (hay[i + j] != needle[j]) continue@outer
            }
            return i
        }
        return -1
    }

    /** 分块下载并发数（★ 8 并发：更快利用带宽；QNN bin 大文件收益明显） */
    private const val DOWNLOAD_THREADS = 8

    // ---- 下载取消支持 ----
    /** 请求取消当前下载：置位后下载循环在每个分块的读循环里检测并中断 */
    @Volatile
    var cancelRequested: Boolean = false

    /** 重置取消标志（每次启动下载前调用） */
    fun resetCancel() { cancelRequested = false }

    /** 请求取消下载 */
    fun requestCancelDownload() { cancelRequested = true }

    /** 下载是否被请求取消 */
    fun isCancelRequested(): Boolean = cancelRequested

    // ---- 下载前连通性检查 ----
    /**
     * 快速探测镜像源/URL 是否可达（HEAD 请求，短超时）。
     * 用于下载前/镜像切换前预检，避免每个镜像白白等待 connectTimeout(30s)。
     * ★ 线程安全 + 可重入：不阻塞主线程，内部自带 client（不共享下载连接池）。
     * @param url 要探测的地址（域名根或完整 URL 均可）
     * @param timeoutMs 探测超时（默认 4s）
     * @return true=可达
     */
    fun quickProbe(url: String, timeoutMs: Long = 4000): Boolean {
        return try {
            val probeClient = OkHttpClient.Builder()
                .connectTimeout(timeoutMs, TimeUnit.MILLISECONDS)
                .readTimeout(timeoutMs, TimeUnit.MILLISECONDS)
                .callTimeout(timeoutMs + 1000, TimeUnit.MILLISECONDS)
                .build()
            val req = Request.Builder().url(url).head().build()
            probeClient.newCall(req).execute().use { resp -> resp.code in 200..499 }
        } catch (_: Exception) { false }
    }

    /** 镜像源列表：国内优先，官方兜底 */
    val mirrors = listOf(
        Mirror("hf-mirror.com", "https://hf-mirror.com", "国内 HuggingFace 镜像 · 推荐"),
        Mirror("huggingface.co", "https://huggingface.co", "官方源 · 备用")
    )

    @Volatile
    var selectedMirrorIndex: Int = -1   // ★ -1 = 自动（启动/下载前探测最优源）；0/1 = 手动指定

    @Volatile
    var activeMirrorIndex: Int = -1

    // ---- 自动镜像探测缓存（避免每次下载都探测，TTL 5 分钟）----
    @Volatile
    private var autoMirrorBest: Int = -1
    @Volatile
    private var autoMirrorProbedAt: Long = 0L
    private const val AUTO_MIRROR_TTL_MS = 5 * 60 * 1000L

    /**
     * 自动探测最优镜像源（国内 hf-mirror / 国外 huggingface），返回镜像 index。
     * 并行 HEAD 探测两个源：可达且耗时最短者胜出；都不可达返回 -1。
     * 结果缓存 5 分钟（避免频繁探测浪费流量/电量）。
     */
    suspend fun detectBestMirror(): Int = withContext(Dispatchers.IO) {
        val now = System.currentTimeMillis()
        if (autoMirrorBest >= 0 && now - autoMirrorProbedAt < AUTO_MIRROR_TTL_MS) {
            return@withContext autoMirrorBest
        }
        val results = mirrors.mapIndexed { idx, m ->
            async {
                val t0 = System.currentTimeMillis()
                val ok = quickProbe(m.baseUrl, 6000)
                val ms = System.currentTimeMillis() - t0
                if (ok) idx to ms else null
            }
        }.mapNotNull { it.await() }
        val best = results.minByOrNull { it.second }?.first ?: -1
        autoMirrorBest = best
        autoMirrorProbedAt = now
        best
    }

    /** 清除自动镜像探测缓存（手动切换镜像后调用，下次下载重新探测） */
    fun invalidateAutoMirror() {
        autoMirrorBest = -1
        autoMirrorProbedAt = 0L
    }

    /** 当前实际应使用的镜像起始 index：自动(-1) → 探测；手动 → 固定 */
    suspend fun effectiveMirrorIndex(): Int = if (selectedMirrorIndex < 0) {
        detectBestMirror().coerceIn(0, mirrors.size - 1)
    } else {
        selectedMirrorIndex.coerceIn(0, mirrors.size - 1)
    }

    // ================= 设备能力自动识别（QNN/CPU + 锁定型号）=================
    /** 当前设备 SoC 型号（如 SM8550）；Android 8+ 有 SOC_MODEL，老设备回退 hardware */
    fun deviceSocModel(): String = runCatching {
        android.os.Build.SOC_MODEL?.trim().orEmpty()
    }.getOrDefault("").ifBlank {
        runCatching { android.os.Build.HARDWARE }.getOrDefault("")
    }

    /** 是否为骁龙（Qualcomm）SoC：SM/QCS/QCM/SDM/MSM 前缀均视为高通平台 */
    fun isQualcommSoc(): Boolean {
        val soc = deviceSocModel().uppercase()
        return listOf("SM", "QCS", "QCM", "SDM", "MSM", "SM8", "SM7", "SM6")
            .any { soc.startsWith(it) } || soc.isBlank().not() && soc.contains("SM8550")
    }

    /** 是否支持 QNN HTP：骁龙 SoC + FastRPC 通信库 libcdsprpc.so 存在（QNN HTP 硬性依赖） */
    fun isQnnSupported(context: Context): Boolean {
        if (!isQualcommSoc()) return false
        return try {
            val candidates = listOf(
                "/vendor/lib64/libcdsprpc.so",
                "/vendor/lib/libcdsprpc.so",
                "/system/lib64/libcdsprpc.so",
                "/system/lib/libcdsprpc.so"
            )
            candidates.any { File(it).exists() } || runCatching {
                System.loadLibrary("cdsprpc"); true
            }.getOrDefault(false)
        } catch (_: Exception) { false }
    }

    /**
     * QNN 模型声明的目标 SoC（如 SM8550/8GEN2 → "SM8550"）；未声明返回 null（通用）。
     * 从模型 id / HF 文件名 / 描述中识别（App 规范命名 qnn_<model>_<soc/后缀>）。
     */
    fun qnnTargetSoc(bm: BuiltinModel): String? {
        if (!bm.isQnn) return null
        val text = (bm.id + " " + bm.hfFile + " " + bm.hfBinFile + " " + bm.description).uppercase()
        return when {
            text.contains("SM8350") || text.contains("8GEN1") -> "SM8350"
            text.contains("SM8450") || text.contains("8GEN1+") || text.contains("8PLUSGEN1") -> "SM8450"
            text.contains("SM8550") || text.contains("8GEN2") -> "SM8550"
            text.contains("SM8650") || text.contains("8GEN3") -> "SM8650"
            text.contains("SM8750") || text.contains("8GEN4") -> "SM8750"
            text.contains("SM8850") || text.contains("8GEN5") || text.contains("8ELITE") -> "SM8850"
            else -> null   // 未声明 → 通用，适用
        }
    }

    /**
     * QNN 模型是否匹配当前设备型号（★ 锁定型号）：
     * 模型声明了目标 SoC（如 sm8550/8gen2）时，必须与当前设备 SoC 一致；
     * 未声明目标型号（通用 v2/SDK250 格式）视为适用。
     */
    fun qnnModelMatchesDevice(bm: BuiltinModel): Boolean {
        if (!bm.isQnn) return true
        val soc = deviceSocModel().uppercase()
        val target = qnnTargetSoc(bm) ?: return true   // 未声明 → 通用，适用
        return soc == target
    }

    /**
     * 自动选择默认模型（QNN/CPU 识别 + 锁定型号）：
     * 设备支持 QNN（骁龙 + libcdsprpc）时：
     *   ★ 优先选「明确声明了当前设备型号」的 QNN 模型（如 SM8550 → qnn_animeseg）；
     *     再退到未声明型号的通用 QNN 模型（如 qnn_rmbg14_v2）；
     *   否则回退 CPU 默认模型（anime_seg）。
     */
    fun autoPickDefaultModel(context: Context): String {
        if (!isQnnSupported(context)) {
            return builtinModels.firstOrNull { !it.isQnn }?.id ?: "anime_seg"
        }
        val soc = deviceSocModel().uppercase()
        // ① 明确声明匹配当前型号的 QNN 模型（按内置列表顺序，qnn_animeseg 在 qnn_modnet 前）
        val declaredMatch = builtinModels.firstOrNull {
            it.isQnn && qnnTargetSoc(it) == soc && qnnModelMatchesDevice(it)
        }
        // ② 未声明型号的通用 QNN 模型
        val genericQnn = builtinModels.firstOrNull {
            it.isQnn && qnnTargetSoc(it) == null && qnnModelMatchesDevice(it)
        }
        // ③ CPU 兜底
        return declaredMatch?.id
            ?: genericQnn?.id
            ?: builtinModels.firstOrNull { !it.isQnn }?.id
            ?: "anime_seg"
    }

    /** QNN 可用且当前设备匹配该模型 → 应默认开 QNN 加速 */
    fun shouldAutoEnableQnn(context: Context, modelId: String): Boolean {
        val bm = builtinModels.find { it.id == modelId } ?: return false
        return bm.isQnn && isQnnSupported(context) && qnnModelMatchesDevice(bm)
    }

    /** 可配置的模型仓库与文件路径（UI 可修改） */
    @Volatile
    var hfRepo: String = "briaai/RMBG-1.4"

    @Volatile
    var hfFile: String = "onnx/model.onnx"

    @Volatile
    var selectedModelId: String = "qnn_rmbg14_v2"

    /** HuggingFace Token（用于下载 gated 模型，如 RMBG-2.0） */
    @Volatile
    var hfToken: String = ""

    /** 自定义模型直链下载地址（UI 配置；非空且已下载时切换为 custom_model 走 CPU 推理） */
    @Volatile
    var modelUrl: String = ""

    /** 自定义直链模型对应的 selectedModelId 标记（custom_model → model_custom.onnx） */
    const val CUSTOM_MODEL_ID = "custom_model"

    /** 模型文件名：内置按 id 映射；QNN EPContext 按部署文件名；本地导入的按 id（local_xxx → model_local_xxx.onnx） */
    fun modelFileName(): String {
        // 自定义直链模型：固定部署名 model_custom.onnx
        if (selectedModelId == CUSTOM_MODEL_ID) return "model_custom.onnx"
        // QNN EPContext：用 EPContext onnx 部署名（与 .bin 同目录，embed_mode=0 要求）
        if (selectedModelId.startsWith("qnn_")) {
            val bm = builtinModels.find { it.id == selectedModelId }
            if (bm?.deployOnnxName != null) return bm.deployOnnxName
        }
        // 本地导入模型：id 形如 local_xxx，文件为 model_local_xxx.onnx
        if (selectedModelId.startsWith("local_")) {
            return "model_$selectedModelId.onnx"
        }
        val bm = builtinModels.find { it.id == selectedModelId } ?: return "model.onnx"
        return "model_${bm.id}.onnx"
    }

    /** QNN EPContext 模型文件（按内置 id 查部署 onnx，找不到该 id 时用当前选中模型） */
    fun qnnContextFileFor(id: String? = null): File? {
        val bm = id?.let { i -> builtinModels.find { it.id == i } } ?: builtinModels.find { it.id == selectedModelId }
        val name = bm?.deployOnnxName ?: return null
        return File(modelDir(), name)
    }

    /** QNN EPContext 配套 context binary（须与 onnx 同一目录，文件名与 onnx 内 ep_cache_context 引用一致） */
    fun qnnContextBinFileFor(id: String? = null): File? {
        val bm = id?.let { i -> builtinModels.find { it.id == i } } ?: builtinModels.find { it.id == selectedModelId }
        val name = bm?.deployBinName ?: return null
        return File(modelDir(), name)
    }

    // ---- 兼容旧引用（RMBG-1.4 QNN EPContext）----
    val qnnContextFile: File get() = qnnContextFileFor("qnn_rmbg14_v2") ?: File(modelDir(), "qnn_rmbg14_sdk250_v2.onnx")
    val qnnContextBinFile: File get() = qnnContextBinFileFor("qnn_rmbg14_v2") ?: File(modelDir(), "qnn_rmbg14_sdk250_v2.bin")

    /** 模型文件（按当前选中模型动态指向） */
    val modelFile: File get() = File(modelDir(), modelFileName())

    /** 模型目录（★外部存储，用户可直接访问：/storage/emulated/0/Android/data/com.rmbg.offline/files/models）
     *  默认兜底同路径（initContext 会用 getExternalFilesDir 覆盖） */
    @Volatile
    private var _modelDir: File? = null

    fun modelDir(): File {
        _modelDir?.let { return it }
        // ★ 兜底路径：normal=com.rmbg.offline；lite=com.rmbg.offline.lite
        //   仅 initContext 未调用时兜底（正常流程 initContext 用 getExternalFilesDir 已覆盖）
        return File(
            android.os.Environment.getExternalStorageDirectory(),
            "Android/data/${if (BuildConfig.IS_LITE) "com.rmbg.offline.lite" else "com.rmbg.offline"}/files/models"
        )
    }

    /** 用 Context 初始化模型目录路径（★外部存储：/storage/emulated/0/Android/data/<pkg>/files/models，
     *  用户文件管理器可直接访问，App 无需存储权限；这是标准 getExternalFilesDir 用法） */
    fun initContext(context: Context) {
        _modelDir = context.getExternalFilesDir(null)?.let { File(it, "models") }
            ?: File(context.filesDir, "models") // 兜底：外部不可用时退回内部
        _modelDir?.mkdirs()
    }

    /** 确保 QNN EPContext 产物已就位（从 assets 拷贝到模型目录，.onnx 与 .bin 须同目录）
     *  @param modelId 内置 QNN 模型 id（默认当前选中；传 null 表示当前选中模型不是 QNN 时用 RMBG-1.4） */
    fun ensureQnnContext(context: Context, modelId: String? = null): Boolean {
        val bm = (modelId?.let { builtinModels.find { m -> m.id == it } }
            ?: builtinModels.find { it.id == selectedModelId })
            ?: builtinModels.find { it.isQnn }
            ?: return false
        val onnxAsset = bm.assetOnnx ?: return false
        val binAsset = bm.assetBin ?: return false
        val onnxDst = qnnContextFileFor(bm.id) ?: return false
        val binDst = qnnContextBinFileFor(bm.id) ?: return false
        val dir = modelDir().also { it.mkdirs() }
        return try {
            // onnx（EPContext 头，约 0.8~1KB）
            // ★ 注意：不能用 openFd() 取 assets 大小！debug 构建 AAPT 会压缩大文件，
            //   openFd 会抛 "can not be opened as a file descriptor; it is probably compressed"。
            //   因此不比较大小：v1/v2 用独立文件名天然隔离，文件存在且有效即跳过。
            if (!onnxDst.exists() || onnxDst.length() < 100) {
                context.assets.open(onnxAsset).use { src ->
                    onnxDst.outputStream().use { dst -> src.copyTo(dst) }
                }
            }
            // bin（context binary，15~99MB）
            if (!binDst.exists() || binDst.length() < 1_000_000) {
                context.assets.open(binAsset).use { src ->
                    binDst.outputStream().use { dst -> src.copyTo(dst) }
                }
            }
            // ★ 不再往 modelDir 复制 QNN EP 插件（libonnxruntime_providers_qnn.so）：
            //   RmbgOnnxEngine 明确只从 ApplicationInfo.nativeLibraryDir 加载插件
            //   （Android linker namespace 不允许从外部存储 dlopen，会报 clns-9 错误），
            //   modelDir 里的插件复制是死冗余，已废弃。
            // ★ 不再往 modelDir 复制 QNN runtime 库（含 libQnnHtpPrepare.so 85MB）：
            //   RmbgOnnxEngine 已从 ApplicationInfo.nativeLibraryDir 直接加载全套 QNN 库（qnn-runtime AAR），
            //   modelDir 复制属历史遗留老方案，会在 Android/data 下白占 ~96MB，已废弃。
            onnxDst.exists() && binDst.exists() &&
                onnxDst.length() > 100 && binDst.length() > 1_000_000
        } catch (e: Exception) {
            android.util.Log.e("RMBG-MODEL", "QNN context 拷贝失败", e)
            false
        }
    }

    /**
     * ★ 双轨：确保 QNN EPContext 产物就位（assets 兜底优先，缺失时才走 HF 下载）
     * assets 内置 onnx+bin 打包在 APK 内，无网/离线优先用内置；内置缺失或损坏时，
     * 从 HF 仓库下载 onnx+bin（hfRepo/hfBinFile）到部署名，实现「下载型」模型。
     * @param modelId 内置 QNN 模型 id
     * @param listener 下载进度回调（仅 HF 下载阶段触发）
     */
    suspend fun ensureQnnContextDual(
        context: Context,
        modelId: String,
        listener: ProgressListener? = null
    ): Boolean = withContext(Dispatchers.IO) {
        val bm = builtinModels.find { it.id == modelId } ?: return@withContext false
        if (!bm.isQnn) return@withContext false
        val onnxDst = qnnContextFileFor(bm.id) ?: return@withContext false
        val binDst = qnnContextBinFileFor(bm.id) ?: return@withContext false
        val dir = modelDir().also { it.mkdirs() }

        // ① assets 兜底拷贝（离线优先；ensureQnnContext 内部已做大小/存在校验）
        ensureQnnContext(context, bm.id)

        // ② assets 已齐 → 完成
        if (onnxDst.exists() && onnxDst.length() > 100 && binDst.exists() && binDst.length() > 1_000_000) {
            return@withContext true
        }

        // ③ assets 缺失/损坏 → HF 下载
        val hfRepo = bm.hfRepo

        // ★ zip 整包模式：hfZip 非空 → 下载整个 zip（内含 bin+onnx）→ 解压部署
        //   与本地 zip 导入（importLocalModelZip）共用解压/路径规范化/校验逻辑
        if (bm.hfZip != null) {
            return@withContext downloadQnnZip(context, bm, hfRepo, listener)
        }

        // 双文件模式（onnx + bin 分开下载，hfZip 未设置时的历史路径）
        val hfOnnx = bm.hfFile
        val hfBin = bm.hfBinFile ?: hfOnnx.removeSuffix(".onnx") + ".bin"
        val mainHandler = Handler(Looper.getMainLooper())
        val client = buildClient()

        // 从 HF 镜像下载单个文件到部署目标（镜像循环 + 分块 + 断点续传，与 CPU downloadModel 同策略）
        suspend fun downloadOne(repoId: String, repoFile: String, target: File): Boolean {
            val start = effectiveMirrorIndex()
            val order = (start until mirrors.size) + (0 until start)
            for (idx in order) {
                // ★ 支持取消：已请求取消则不继续尝试其他镜像
                if (cancelRequested) return@downloadOne false
                val mirror = mirrors[idx]
                val fu = "${mirror.baseUrl}/$repoId/resolve/main/$repoFile"
                try {
                    // ★ 下载前快速连通性检查：不可达镜像直接跳过
                    if (!quickProbe(mirror.baseUrl)) {
                        mainHandler.post { listener?.onMirrorError(mirror.name, "网络不可达（预检失败）") }
                        continue
                    }
                    mainHandler.post { listener?.onMirrorSwitch(idx, mirror.name) }
                    fun buildReq(u: String): Request {
                        val b = Request.Builder().url(u)
                        if (hfToken.isNotBlank()) b.header("Authorization", "Bearer $hfToken")
                        return b.build()
                    }
                    // ★ 第一步：手动解析重定向（client 已关自动跟随）拿到真实下载 URL
                    //   HF resolve 直链会 302 → cas-bridge.xethub.hf.co 签名 CDN URL。
                    var realUrl = fu
                    try {
                        client.newCall(buildReq(fu)).execute().use { p ->
                            if (p.code == 302 || p.code == 307 || p.code == 308) {
                                p.header("Location")?.takeIf { it.startsWith("http") }?.let { realUrl = it }
                            } else {
                                android.util.Log.w("RMBG-MODEL", "resolve 未重定向: HTTP ${p.code}（非 302/307/308，直接用原 URL）")
                            }
                        }
                    } catch (e: Exception) {
                        android.util.Log.w("RMBG-MODEL", "解析重定向失败: ${e.message}，直接用原 URL")
                    }

                    // 第二步：对真实下载 URL 发 Range: bytes=0-0 探测总大小（206 + Content-Range）
                    var probed = 0L
                    try {
                        client.newCall(buildReq(realUrl).newBuilder().header("Range", "bytes=0-0").build())
                            .execute().use { resp ->
                                if (resp.isSuccessful || resp.code == 206) {
                                    val cr = resp.header("Content-Range")
                                    probed = cr?.substringAfter('/')?.toLongOrNull()
                                        ?: resp.body?.contentLength() ?: 0L
                                } else {
                                    android.util.Log.w("RMBG-MODEL", "大小探测 HTTP ${resp.code} <- $mirror.name")
                                }
                            }
                    } catch (e: Exception) {
                        android.util.Log.w("RMBG-MODEL", "大小探测失败(${mirror.name}): ${e.message}")
                    }
                    // 用元数据大小兜底：onnx≈sizeBytes-bin，bin=binExpected
                    val expectSize = if (target == onnxDst) (bm.sizeBytes - binExpected(bm)).coerceAtLeast(100L)
                        else binExpected(bm)
                    val totalBytes = if (probed in 1 until expectSize / 2) expectSize else probed
                    if (target == binDst && totalBytes < 1_000_000) {
                        android.util.Log.e("RMBG-MODEL", "bin 大小异常(${totalBytes}B)，跳过 $mirror.name")
                        continue // bin 至少 1MB
                    }
                    if (target == onnxDst && totalBytes < 100) {
                        android.util.Log.e("RMBG-MODEL", "onnx 大小异常(${totalBytes}B)，跳过 $mirror.name")
                        continue
                    }
                    // 断点续传
                    val partBase = target.name
                    val existing = File(dir, "$partBase.part")
                    if (existing.exists() && existing.length() >= totalBytes) {
                        existing.renameTo(target)
                        return@downloadOne true
                    }
                    // 分块下载（直接对签名 CDN URL 发 Range）
                    val downloaded = downloadChunks(
                        client = client, baseUrl = realUrl, totalBytes = totalBytes,
                        dir = dir, partPrefix = partBase, token = hfToken,
                        threads = DOWNLOAD_THREADS,
                        onProgress = { d, t, sp -> mainHandler.post { listener?.onProgress(d, t, sp) } }
                    )
                    if (downloaded != totalBytes) {
                        android.util.Log.e("RMBG-MODEL", "分块下载不完整 ${downloaded}/${totalBytes} <- $mirror.name")
                        continue
                    }
                    mergeChunks(dir, partBase, totalBytes, target)
                    return@downloadOne true
                } catch (e: Exception) {
                    android.util.Log.e("RMBG-MODEL", "下载失败(${mirror.name}): ${e.javaClass.simpleName}: ${e.message}")
                    mainHandler.post { listener?.onMirrorError(mirror.name, "${e.javaClass.simpleName}: ${e.message}") }
                    continue
                }
            }
            return@downloadOne false
        }

        // 下载 onnx（小）→ bin（大）
        val onnxOk = if (onnxDst.exists() && onnxDst.length() > 100) true
            else downloadOne(hfRepo, hfOnnx, onnxDst)
        val binOk = if (binDst.exists() && binDst.length() > 1_000_000) true
            else downloadOne(hfRepo, hfBin, binDst)

        onnxOk && binOk &&
            onnxDst.exists() && onnxDst.length() > 100 &&
            binDst.exists() && binDst.length() > 1_000_000
    }

    /**
     * ★ zip 整包模式：从 HF 云下载 QNN 模型 zip（内含 bin+onnx），解压部署到模型目录。
     *   与本地 zip 导入（importLocalModelZip）共用解压/路径规范化/校验逻辑：
     *   - 下载：镜像循环 + 302 解析 + Range 分块 + 断点续传（复用 downloadModel 的 destFile 通道）
     *   - 部署：解压 zip → 找主 onnx → 提取 ep_cache_context 引用 bin → 复制部署 → normalize 路径
     *
     * @param bm  内置 QNN 模型
     * @param hfRepo HF 仓库 id
     * @param listener 进度回调（下载阶段触发）
     */
    private suspend fun downloadQnnZip(
        context: Context,
        bm: BuiltinModel,
        hfRepo: String,
        listener: ProgressListener? = null
    ): Boolean = withContext(Dispatchers.IO) {
        initContext(context)
        val dir = modelDir().also { it.mkdirs() }
        val zipDst = File(dir, "${bm.id}.zip")
        val mainHandler = Handler(Looper.getMainLooper())

        // ① 已有解压产物 → 直接完成
        val onnxDst = qnnContextFileFor(bm.id)
        val binDst = qnnContextBinFileFor(bm.id)
        if (onnxDst != null && binDst != null &&
            onnxDst.exists() && onnxDst.length() > 100 &&
            binDst.exists() && binDst.length() > 1_000_000
        ) {
            return@withContext true
        }

        // ② 下载 zip（复用 downloadModel 的 destFile 通道：镜像循环 + 302 + 分块 + 断点续传）
        val zipOk = downloadModel(
            context = context,
            hfRepo = hfRepo,
            hfFile = bm.hfZip!!,
            destFile = zipDst,
            listener = listener
        )

        // ③ 解压部署（zip → bin+onnx → 模型目录，与本地导入同构）
        if (!zipOk || !zipDst.exists()) {
            mainHandler.post { listener?.onMirrorError("zip", "zip 下载失败或文件不存在") }
            return@withContext false
        }
        val deployed = deployQnnZip(zipDst, bm, mainHandler)
        if (!deployed) {
            mainHandler.post { listener?.onMirrorError("zip", "zip 解压部署失败（检查包内 onnx/bin 结构）") }
            return@withContext false
        }
        true
    }

    /**
     * 解压 QNN zip 并部署到模型目录（onnx+bin 同目录，ep_cache_context 规范化）。
     * 与 importLocalModelZip 的部署段共用逻辑，但部署名固定为内置模型的 deployOnnxName/deployBinName。
     */
    private fun deployQnnZip(
        zipFile: File,
        bm: BuiltinModel,
        mainHandler: Handler
    ): Boolean {
        try {
            val dir = modelDir().also { it.mkdirs() }
            val tmpDir = File(dir, "zip_deploy_${System.currentTimeMillis()}")
            tmpDir.mkdirs()
            try {
                // 1) 解压 zip（只取 onnx + bin）
                var totalBytes = 0L
                java.util.zip.ZipFile(zipFile).use { zf ->
                    val entries = zf.entries()
                    while (entries.hasMoreElements()) {
                        val entry = entries.nextElement()
                        if (entry.isDirectory) continue
                        val name = entry.name
                        if (!name.endsWith(".onnx", ignoreCase = true) &&
                            !name.endsWith(".bin", ignoreCase = true)) continue
                        totalBytes += entry.size
                        if (totalBytes > 4L * 1024 * 1024 * 1024) { // 4GB 上限
                            android.util.Log.e("RMBG-MODEL", "zip 解压超限，中止")
                            return false
                        }
                        val baseName = name.substringAfterLast('/').substringAfterLast('\\')
                        if (baseName.isEmpty()) continue
                        val outFile = File(tmpDir, baseName)
                        zf.getInputStream(entry).use { input ->
                            outFile.outputStream().use { out -> input.copyTo(out) }
                        }
                    }
                }

                // 2) 找主 onnx（优先 hfFile 文件名对应的，否则第一个 .onnx）
                val onnxFiles = tmpDir.listFiles { _, n -> n.endsWith(".onnx", ignoreCase = true) }
                    ?: emptyArray()
                if (onnxFiles.isEmpty()) {
                    android.util.Log.e("RMBG-MODEL", "zip 内无 .onnx 文件")
                    return false
                }
                val wantOnnx = bm.hfFile.substringAfterLast('/').substringAfterLast('\\')
                val mainOnnx = onnxFiles.firstOrNull { it.name == wantOnnx } ?: onnxFiles[0]

                // 3) 部署 onnx（固定内置部署名）
                val destOnnx = File(dir, bm.deployOnnxName ?: mainOnnx.name)
                mainOnnx.copyTo(destOnnx, overwrite = true)

                // 4) 提取 bin 引用 + 匹配 bin（精确 → 包含 → 唯一兜底，与 importLocalModelZip 同构）
                val binRef = extractEpCacheContext(mainOnnx)
                val binFiles = tmpDir.listFiles { _, n -> n.endsWith(".bin", ignoreCase = true) } ?: emptyArray()
                val binSrc = when {
                    binRef != null -> binFiles.firstOrNull { it.name.equals(binRef, ignoreCase = true) }
                        ?: binFiles.firstOrNull { it.name.contains(binRef, ignoreCase = true) }
                        ?: binFiles.firstOrNull { binRef.contains(it.nameWithoutExtension, ignoreCase = true) }
                    else -> null
                } ?: binFiles.firstOrNull()
                if (binSrc == null) {
                    android.util.Log.e("RMBG-MODEL", "EPContext 引用 bin=$binRef 但 zip 内无 .bin 文件")
                    return false
                }
                // bin 名必须以 onnx 内部引用为准（embed_mode=0 要求），复制并改名
                val destName = binRef ?: (bm.deployBinName ?: binSrc.name)
                val destBin = File(dir, destName)
                binSrc.copyTo(destBin, overwrite = true)

                // 5) 路径规范化（绝对路径 → basename）+ tar 权重包检测
                if (normalizeEpContextPath(destOnnx)) {
                    android.util.Log.i("RMBG-MODEL", "zip 部署 EPContext 路径已规范化 -> $destName")
                }
                try {
                    if (destBin.length() > 300) {
                        val head = destBin.readBytes().take(300).toByteArray()
                        val isTar = head[257] == 'u'.code.toByte() && head[258] == 's'.code.toByte() &&
                            head[259] == 't'.code.toByte() && head[260] == 'a'.code.toByte() && head[261] == 'r'.code.toByte()
                        if (isTar) {
                            android.util.Log.w("RMBG-MODEL",
                                "⚠️ 检测到 $destName 是 QNN 工具链 tar 权重包（POSIX tar），ORT QNN EP 无法加载（需要 context binary）。请使用 qnn-context-binary-generator 输出的 .bin")
                        }
                    }
                } catch (_: Exception) {}

                return destOnnx.exists() && destBin.exists() &&
                    destOnnx.length() > 100 && destBin.length() > 1_000_000
            } finally {
                try { tmpDir.deleteRecursively() } catch (_: Exception) {}
            }
        } catch (e: Exception) {
            android.util.Log.e("RMBG-MODEL", "QNN zip 部署失败", e)
            return false
        }
    }

    /** QNN bin 元数据大小（供 onnx 探测大小反推：sizeBytes - bin） */
    private fun binExpected(bm: BuiltinModel): Long {
        // sizeBytes = bin + onnx(≈1KB)，bin 约等于 sizeBytes 扣掉 onnx 头
        return (bm.sizeBytes - 2048L).coerceAtLeast(1_000_000L)
    }

    // ★ 已废弃：qnnPluginFile / qnnHtpFile / copyQnnRuntimeLibs()
    //   历史遗留——曾把 libonnxruntime_providers_qnn.so / libQnnHtp.so / libQnnHtpV73Stub.so /
    //   libQnnSystem.so / libQnnHtpPrepare.so(85MB) 从 APK 复制到 modelDir（Android/data/.../files/models/），
    //   在外部存储白占 ~96MB。
    //   RmbgOnnxEngine 实际从 nativeLibraryDir 加载 QNN 库，此复制从未被使用，已全部移除。

    fun isModelDownloaded(): Boolean {
        // QNN EPContext：检查 onnx + bin 是否都就位（按当前选中 QNN 模型）
        if (selectedModelId.startsWith("qnn_")) {
            val o = qnnContextFileFor() ?: return false
            val b = qnnContextBinFileFor() ?: return false
            return o.exists() && o.length() > 100 && b.exists() && b.length() > 1_000_000
        }
        // ★ 本地导入的 EPContext 模型：onnx 可能只有几百字节（EPContext 头，权重在 .bin）。
        //   只要 onnx 含 EPContext 标记且配套 bin 已就位 → 视为已下载（不走通用大小校验）。
        if (selectedModelId.startsWith("local_")) {
            val f = modelFile
            if (f.exists() && f.length() > 0) {
                val binRef = try { extractEpCacheContext(f) } catch (_: Exception) { null }
                if (binRef != null) {
                    val bin = File(modelDir(), binRef)
                    return bin.exists() && bin.length() > 1_000_000
                }
            }
        }
        val f = modelFile
        return f.exists() && f.length() > minValidSize()
    }

    /** 当前模型最小有效大小：内置元数据的 5%，至少 1MB（MODNet 25MB 也能通过校验） */
    fun minValidSize(): Long {
        // 自定义直链模型：至少 100KB（任意 .onnx）
        if (selectedModelId == CUSTOM_MODEL_ID) return 100_000L
        // QNN EPContext：onnx 本身只有 ~0.8~1KB（EPContext 头），用 bin 判断
        if (selectedModelId.startsWith("qnn_")) {
            return 100L
        }
        // 本地导入模型：按实际文件大小 5% 兜底（至少 100KB）
        if (selectedModelId.startsWith("local_")) {
            val f = modelFile
            return if (f.exists()) maxOf(100_000L, f.length() / 20) else 100_000L
        }
        val meta = builtinModels.find { it.id == selectedModelId }?.sizeBytes ?: 176_153_355L
        return maxOf(1_000_000L, meta / 20)
    }

    /** 检查指定内置模型是否已下载（QNN：onnx+bin 双文件；CPU：单 onnx；超分：superres/ 目录） */
    fun isBuiltinModelDownloaded(bm: BuiltinModel): Boolean {
        if (bm.id == "qnn_realesrgan") return isSuperResReady()
        if (bm.isQnn) {
            val o = qnnContextFileFor(bm.id) ?: return false
            val b = qnnContextBinFileFor(bm.id) ?: return false
            return o.exists() && o.length() > 100 && b.exists() && b.length() > 1_000_000
        }
        val f = File(modelDir(), "model_${bm.id}.onnx")
        return f.exists() && f.length() > maxOf(1_000_000L, bm.sizeBytes / 20)
    }

    /** 删除指定内置模型文件（QNN 删 onnx+bin，CPU 删 model_<id>.onnx；超分删 superres/） */
    fun deleteBuiltinModel(bm: BuiltinModel): Boolean {
        if (bm.id == "qnn_realesrgan") return deleteSuperResModel()
        return try {
            if (bm.isQnn) {
                var ok = true
                qnnContextFileFor(bm.id)?.let { if (it.exists()) ok = it.delete() && ok }
                qnnContextBinFileFor(bm.id)?.let { if (it.exists()) ok = it.delete() && ok }
                ok
            } else {
                val f = File(modelDir(), "model_${bm.id}.onnx")
                if (f.exists()) f.delete() else true
            }
        } catch (_: Exception) { false }
    }

    /** 内置模型状态详情（供 UI 诊断：为什么没显示已下载） */
    data class BuiltinModelStatus(
        val bm: BuiltinModel,
        val fileName: String,
        val exists: Boolean,
        val sizeBytes: Long,
        val minValid: Long,
        val downloaded: Boolean,
        val reason: String
    )

    /** 逐个模型计算状态：文件名 / 是否存在 / 实际大小 / 最小有效大小 / 判定 / 原因 */
    fun builtinModelStatus(bm: BuiltinModel): BuiltinModelStatus {
        // QNN EPContext：特殊处理（onnx + bin 双文件）
        if (bm.isQnn) {
            val o = qnnContextFileFor(bm.id) ?: return BuiltinModelStatus(bm, "未知", false, 0L, 1_000_000L, false, "EPContext 配置缺失")
            val b = qnnContextBinFileFor(bm.id) ?: o
            val ok = o.exists() && o.length() > 100 && b.exists() && b.length() > 1_000_000
            if (ok) {
                return BuiltinModelStatus(bm, "${o.name} + ${b.name}", true, b.length(), 1_000_000L, true, "OK")
            }
            return BuiltinModelStatus(bm, "${o.name} + ${b.name}", o.exists() && b.exists(), b.length(), 1_000_000L, false,
                "EPContext 文件缺失: ${if (!o.exists()) o.name else ""} ${if (!b.exists()) b.name else ""}")
        }
        val fileName = "model_${bm.id}.onnx"
        val f = File(modelDir(), fileName)
        val minValid = maxOf(1_000_000L, bm.sizeBytes / 20)
        val exists = f.exists()
        val size = if (exists) f.length() else 0L
        val downloaded = exists && size > minValid
        val reason = when {
            !exists -> "文件缺失: $fileName"
            size <= minValid -> "大小不足: ${size / 1024 / 1024}MB < ${minValid / 1024 / 1024}MB"
            else -> "OK"
        }
        return BuiltinModelStatus(bm, fileName, exists, size, minValid, downloaded, reason)
    }

    /** 全量诊断报告：模型目录 + 每个内置模型状态（供 UI 诊断卡片直接展示） */
    fun modelStatusReport(): String {
        val sb = StringBuilder()
        val dir = modelDir()
        sb.append("模型目录: ").append(dir.absolutePath).append('\n')
        sb.append("目录存在: ").append(dir.exists()).append('\n')
        sb.append("目录内 .onnx 文件:\n")
        dir.listFiles()?.filter { it.name.endsWith(".onnx") }?.sortedBy { it.name }?.forEach { f ->
            sb.append("  · ").append(f.name).append(" (").append(f.length() / 1024 / 1024).append("MB)\n")
        } ?: sb.append("  （无 .onnx 文件）\n")
        sb.append("内置模型判定:\n")
        builtinModels.forEach { bm ->
            val st = builtinModelStatus(bm)
            val tag = if (st.downloaded) "✅已下载" else "❌未下载"
            sb.append("  · ").append(bm.name).append(" [").append(tag).append("] ")
                .append(st.fileName).append(" → ").append(st.reason).append('\n')
        }
        return sb.toString()
    }

    /**
     * ★ 内部 → 外部 模型目录迁移：
     * 早期版本模型目录在内部 filesDir/models（/data/user/0/...，用户文件管理器进不去），
     * 现在改用 getExternalFilesDir（/storage/emulated/0/Android/data/<pkg>/files/models，用户可访问）。
     * App 运行时（真实 app 域）有权限写自己的外部目录，启动时调用一次把旧模型搬过来。
     */
    fun migrateModelsFromInternal(context: Context): Int {
        return try {
            val oldDir = File(context.filesDir, "models")
            val newDir = modelDir()
            if (!oldDir.exists() || oldDir == newDir) return 0
            newDir.mkdirs()
            var moved = 0
            oldDir.listFiles()?.forEach { src ->
                if (src.isFile) {
                    val dst = File(newDir, src.name)
                    if (!dst.exists() || dst.length() != src.length()) {
                        src.copyTo(dst, overwrite = true)
                        moved++
                    }
                }
            }
            android.util.Log.i("RMBG-MODEL", "模型迁移完成: 内部→外部 共 $moved 个文件 -> $newDir")
            moved
        } catch (e: Exception) {
            android.util.Log.w("RMBG-MODEL", "模型迁移失败: ${e.message}")
            0
        }
    }

    interface ProgressListener {
        fun onProgress(bytesDownloaded: Long, totalBytes: Long, speedBps: Long)
        fun onMirrorSwitch(mirrorIndex: Int, mirrorName: String)
        fun onMirrorError(mirrorName: String, error: String)
        fun onDone(file: File)
        fun onError(e: Exception)
    }

    private fun buildClient(readTimeoutSec: Long = 300): OkHttpClient =
        OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(readTimeoutSec, TimeUnit.SECONDS)
            .writeTimeout(60, TimeUnit.SECONDS)
            // ★ 连接池复用：并发分块共享连接（8 并发时避免重复握手，提速明显）
            .connectionPool(okhttp3.ConnectionPool(8, 5, TimeUnit.MINUTES))
            // ★ 关闭自动跟随重定向：HF resolve 直链会 302/307/308 到签名 CDN URL，
            //   手动解析 Location 拿到真实下载地址再分块（curl 实测最稳）。
            //   若让 OkHttp 自动跟随，跨 host（hf-mirror.com → cas-bridge.xethub.hf.co）
            //   时 Range 头可能不被保留，CDN 返回 200 全文件 → 8 线程各自抢整个文件
            //   互相限速断流，part 卡在 1~2MB（已实测复现）。
            .followRedirects(false)
            .followSslRedirects(false)
            .build()

    /**
     * 多线程分块下载模型
     * @param destFile 指定下载目标文件（默认=modelFile，即当前选中模型的部署路径）
     * @param tokenOverride 指定 HF Token（默认=null，用 ModelManager.hfToken）
     */
    suspend fun downloadModel(
        context: Context,
        hfRepo: String = ModelManager.hfRepo,
        hfFile: String = ModelManager.hfFile,
        destFile: File? = null,
        tokenOverride: String? = null,
        listener: ProgressListener? = null
    ) = withContext(Dispatchers.IO) {
        initContext(context)
        val dir = destFile?.parentFile ?: modelDir()
        if (!dir.exists()) dir.mkdirs()
        val target = destFile ?: modelFile
        val token = tokenOverride ?: hfToken

        val mainHandler = Handler(Looper.getMainLooper())
        val client = buildClient()

        var lastError: Exception? = null

        // ★ 下载自动重试：整体最多尝试 5 次（每轮走完所有镜像源），网络波动时自动恢复
        val maxAttempts = 5
        var attempt = 0
        while (attempt < maxAttempts) {
            attempt++
            // 取消检查：用户点击取消立即终止，不再重试
            if (cancelRequested) {
                activeMirrorIndex = -1
                val cancelErr = CancellationException("下载已取消")
                mainHandler.post { listener?.onError(cancelErr) }
                return@withContext false
            }
            // 重试前等待（递增退避：2s/4s/6s/8s），并提示当前尝试次数
            if (attempt > 1) {
                delay(2000L * attempt)
                mainHandler.post { listener?.onMirrorSwitch(-1, "网络波动，自动重试 $attempt/$maxAttempts ...") }
            }

            val start = effectiveMirrorIndex()
            val order = (start until mirrors.size) + (0 until start)

            for (idx in order) {
            // ★ 支持取消：已请求取消则不继续尝试其他镜像
            if (cancelRequested) {
                activeMirrorIndex = -1
                val cancelErr = CancellationException("下载已取消")
                mainHandler.post { listener?.onError(cancelErr) }
                return@withContext false
            }
            val mirror = mirrors[idx]
            activeMirrorIndex = idx
            val baseUrl = "${mirror.baseUrl}/$hfRepo/resolve/main/$hfFile"

            fun mirrorFailed(msg: String) {
                lastError = IllegalStateException(msg)
                mainHandler.post { listener?.onMirrorError(mirror.name, msg) }
            }

            try {
                // ★ 下载前快速连通性检查：不可达的镜像直接跳过，避免等待 connectTimeout(30s)
                if (!quickProbe(mirror.baseUrl)) {
                    mirrorFailed("网络不可达（${mirror.name}）")
                    continue
                }
                mainHandler.post { listener?.onMirrorSwitch(idx, mirror.name) }

                // 请求构建：带 HF Token（下载 gated 模型如 RMBG-2.0 必需）
                fun buildReq(url: String): Request {
                    val b = Request.Builder().url(url)
                    if (token.isNotBlank()) {
                        b.header("Authorization", "Bearer $token")
                    }
                    return b.build()
                }

                // ★ 第一步：手动解析重定向（client 已关自动跟随）拿到真实下载 URL
                //   HF resolve 直链（hf-mirror/官方）会 302 → cas-bridge.xethub.hf.co 签名 CDN URL。
                //   签名 URL 是 Range 分块真正能生效的地址；OkHttp 自动跟随跨 host 时
                //   Range 头可能丢失导致 CDN 回 200 全文件，所以这里必须手动拿 Location。
                var realUrl = baseUrl
                try {
                    client.newCall(buildReq(baseUrl)).execute().use { p ->
                        if (p.code == 302 || p.code == 307 || p.code == 308) {
                            p.header("Location")?.takeIf { it.startsWith("http") }?.let { realUrl = it }
                        }
                    }
                } catch (_: Exception) { /* 直接用 baseUrl */ }

                // 第二步：对真实下载 URL 发 Range: bytes=0-0 探测总大小（206 + Content-Range）
                val probeReq = buildReq(realUrl).newBuilder()
                    .header("Range", "bytes=0-0")
                    .build()
                var probedBytes = 0L
                try {
                    client.newCall(probeReq).execute().use { resp ->
                        if (!resp.isSuccessful && resp.code != 206) {
                            mirrorFailed("HTTP ${resp.code} from ${mirror.name}")
                            return@use
                        }
                        // 从 Content-Range: bytes 0-0/176153355 解析总大小
                        val cr = resp.header("Content-Range")
                        probedBytes = if (cr != null) {
                            cr.substringAfter('/').toLongOrNull() ?: 0L
                        } else {
                            resp.body?.contentLength() ?: 0L
                        }
                    }
                } catch (e: Exception) {
                    mirrorFailed("探测失败: ${e.message}")
                    continue
                }
                // 镜像可能返回异常小值（如 LFS 指针/错误重定向），以内置模型元数据兜底
                val builtinMeta = builtinModels.find { it.id == selectedModelId }
                val metaBytes = builtinMeta?.sizeBytes ?: 0L
                val totalBytes = if (probedBytes in 1 until metaBytes / 2) metaBytes else probedBytes
                // 当前模型最小有效大小：内置元数据的 5%，至少 1MB（MODNet 25MB 也能通过校验）
                // ★ 去掉 destFile 强制 100MB 门槛：QNN zip（90MB）与 SD 大包一视同仁，按各自模型阈值校验
                val minValid = minValidSize()
                if (totalBytes < minValid) {
                    mirrorFailed("文件过小(${totalBytes / 1024 / 1024}MB)")
                    continue
                }

                // 断点续传：已有完整 .part 则跳过
                val partBase = destFile?.name ?: modelFileName()
                val existing = File(dir, "$partBase.part")
                if (existing.exists() && existing.length() >= totalBytes) {
                    existing.renameTo(target)
                    activeMirrorIndex = idx
                    mainHandler.post { listener?.onDone(target) }
                    return@withContext true
                }

                // 分块下载前先解析真实下载 URL（跟随 302 拿到 CDN 签名 URL，Range 才生效）
                // ★ 已在上面"第一步"手动解析过（client 关闭了自动跟随），realUrl 已是签名 CDN URL，
                //   直接用于分块；若解析失败则回退 resolve 直链（OkHttp 会跟随但 Range 可能失效）。
                val downloadedAll = downloadChunks(
                    client = client,
                    baseUrl = realUrl,
                    totalBytes = totalBytes,
                    dir = dir,
                    partPrefix = partBase,
                    token = token,
                    threads = DOWNLOAD_THREADS,
                    onProgress = { done, total, speed ->
                        mainHandler.post { listener?.onProgress(done, total, speed) }
                    }
                )

                if (downloadedAll != totalBytes) {
                    mirrorFailed("下载不完整(${downloadedAll / 1024 / 1024}/${totalBytes / 1024 / 1024}MB)")
                    continue
                }

                // 合并分块
                mergeChunks(dir, partBase, totalBytes, target)
                activeMirrorIndex = idx
                mainHandler.post { listener?.onDone(target) }
                return@withContext true
            } catch (e: Exception) {
                mirrorFailed("${e.javaClass.simpleName}: ${e.message}")
            }
            } // ---- for 镜像循环结束 ----
        } // ---- while 重试循环结束 ----

        val err = lastError ?: IllegalStateException("所有镜像源下载失败，已自动重试 $maxAttempts 次")
        activeMirrorIndex = -1
        mainHandler.post { listener?.onError(err) }
        false
    }

    /**
     * ★ 从自定义直链下载 ONNX 模型（模型配置面板"下载链接"入口）。
     *   下载到 model_custom.onnx 并自动切换 selectedModelId=custom_model（走 CPU 推理）。
     *   支持直接 .onnx 文件，或 .zip（内含 .onnx，自动解压提取首个 .onnx）。
     * @param url 完整下载链接（http/https，可为 HF resolve 直链、CDN、任意托管）
     */
    suspend fun downloadModelFromUrl(
        context: Context,
        url: String,
        listener: ProgressListener? = null
    ): Boolean = withContext(Dispatchers.IO) {
        initContext(context)
        val dir = modelDir().also { it.mkdirs() }
        val zipDst = File(dir, "model_custom.zip")
        val target = File(dir, "model_custom.onnx")

        val mainHandler = Handler(Looper.getMainLooper())
        val client = buildClient()

        // 请求构建（带 HF Token，gated/私有托管可能需要）
        fun buildReq(u: String): Request {
            val b = Request.Builder().url(u)
            if (hfToken.isNotBlank()) b.header("Authorization", "Bearer $hfToken")
            return b.build()
        }

        try {
            // ★ ① 手动解析重定向（client 已关自动跟随）拿到真实下载 URL
            //   HF resolve 直链 / 其他托管 302 → 签名 CDN URL，Range 分块只有对真实 URL 才生效。
            var realUrl = url
            try {
                client.newCall(buildReq(url)).execute().use { p ->
                    if (p.code == 302 || p.code == 307 || p.code == 308) {
                        p.header("Location")?.takeIf { it.startsWith("http") }?.let { realUrl = it }
                    }
                }
            } catch (_: Exception) { /* 直接用 url */ }

            // ② 探测总大小（Range: bytes=0-0 对真实下载 URL 生效，206 + Content-Range）
            val probeReq = buildReq(realUrl).newBuilder().header("Range", "bytes=0-0").build()
            var totalBytes = 0L
            try {
                client.newCall(probeReq).execute().use { resp ->
                    if (resp.isSuccessful || resp.code == 206) {
                        val cr = resp.header("Content-Range")
                        totalBytes = cr?.substringAfter('/')?.toLongOrNull()
                            ?: resp.body?.contentLength() ?: 0L
                    } else {
                        mainHandler.post { listener?.onError(IllegalStateException("HTTP ${resp.code}")) }
                        return@withContext false
                    }
                }
            } catch (e: Exception) {
                mainHandler.post { listener?.onError(e) }
                return@withContext false
            }
            if (totalBytes < 100L) {
                mainHandler.post { listener?.onError(IllegalStateException("文件过小或无效链接")) }
                return@withContext false
            }

            // ③ 断点续传：已有完整 model_custom.onnx 则跳过
            if (target.exists() && target.length() >= totalBytes) {
                selectedModelId = CUSTOM_MODEL_ID
                mainHandler.post { listener?.onDone(target) }
                return@withContext true
            }
            // 已有 .zip 完成则直接解压
            if (zipDst.exists() && zipDst.length() >= totalBytes) {
                val okZip = unzipCustomModel(zipDst, target)
                if (okZip) {
                    selectedModelId = CUSTOM_MODEL_ID
                    mainHandler.post { listener?.onDone(target) }
                } else {
                    mainHandler.post { listener?.onError(IllegalStateException("压缩包解压失败（无 .onnx）")) }
                }
                return@withContext okZip
            }

            // ④ 分块下载到 .part（直接对签名 CDN URL 发 Range）
            val partBase = "model_custom"
            val downloadedAll = downloadChunks(
                client = client, baseUrl = realUrl, totalBytes = totalBytes,
                dir = dir, partPrefix = partBase, token = hfToken,
                threads = DOWNLOAD_THREADS,
                onProgress = { d, t, sp -> mainHandler.post { listener?.onProgress(d, t, sp) } }
            )
            if (downloadedAll < totalBytes) {
                mainHandler.post { listener?.onError(IllegalStateException("下载不完整")) }
                return@withContext false
            }
            // 合并分块 → 临时完整文件
            mergeChunks(dir, partBase, totalBytes, zipDst)

            // ⑤ 单 .onnx 直接 rename；.zip 解压提取 .onnx
            val ok: Boolean
            val lower = url.lowercase()
            val isZipMagic = try {
                val head = zipDst.inputStream().use { it.readNBytes(4) }
                head.size == 4 && head[0] == 0x50.toByte() && head[1] == 0x4B.toByte()
            } catch (_: Exception) { false }
            if (zipDst.length() >= 100_000L && (lower.endsWith(".zip") || isZipMagic)) {
                ok = unzipCustomModel(zipDst, target)
            } else {
                target.delete()
                ok = zipDst.renameTo(target) || zipDst.copyTo(target, overwrite = true).let { true }
            }
            zipDst.delete()

            if (!ok || !target.exists()) {
                mainHandler.post { listener?.onError(IllegalStateException("模型文件处理失败")) }
                return@withContext false
            }
            // ⑥ 切换自定义模型标记，走 CPU 链路
            selectedModelId = CUSTOM_MODEL_ID
            mainHandler.post { listener?.onDone(target) }
            return@withContext true
        } catch (e: Exception) {
            mainHandler.post { listener?.onError(e) }
            return@withContext false
        }
    }

    /** 从 zip 中提取首个 .onnx 到 target（解压） */
    private fun unzipCustomModel(zip: File, target: File): Boolean {
        return try {
            val zipf = java.util.zip.ZipFile(zip)
            try {
                val entry = zipf.entries().asSequence().filter { it.name.endsWith(".onnx") }.firstOrNull()
                    ?: return false
                zipf.getInputStream(entry).use { src ->
                    val out = FileOutputStream(target)
                    val buf = ByteArray(1 shl 20)
                    var read: Int
                    while (src.read(buf).also { read = it } != -1) out.write(buf, 0, read)
                    out.flush()
                }
                target.length() > 100_000L
            } finally { zipf.close() }
        } catch (_: Exception) { false }
    }

    /** 分块下载：每块独立 Range 请求，并发写入独立 .part.N 文件 */
    private suspend fun downloadChunks(
        client: OkHttpClient,
        baseUrl: String,
        totalBytes: Long,
        dir: File,
        partPrefix: String,
        token: String,
        threads: Int,
        onProgress: (Long, Long, Long) -> Unit
    ): Long = coroutineScope {
        val chunkSize = (totalBytes + threads - 1) / threads
        val done = AtomicLong(0L)
        val startTime = System.currentTimeMillis()

        val results = (0 until threads).map { i ->
            async(Dispatchers.IO) {
                val start = i * chunkSize
                val end = minOf(start + chunkSize - 1, totalBytes - 1)
                if (start > end) return@async 0L
                val partFile = File(dir, "$partPrefix.part.$i")
                val reqBuilder = Request.Builder()
                    .url(baseUrl)
                    .header("Range", "bytes=$start-$end")
                if (token.isNotBlank()) {
                    reqBuilder.header("Authorization", "Bearer $token")
                }
                val req = reqBuilder.build()
                try {
                    client.newCall(req).execute().use { resp ->
                        if (resp.code != 206 && resp.code != 200) {
                            return@use 0L
                        }
                        resp.body?.byteStream()?.use { input ->
                            val out = FileOutputStream(partFile)
                            val buf = ByteArray(512 * 1024)
                            var read: Int
                            var written = 0L
                            while (input.read(buf).also { read = it } != -1) {
                                // ★ 支持取消：检测到取消请求立即停止读取
                                if (cancelRequested) {
                                    out.flush()
                                    out.close()
                                    return@use 0L
                                }
                                out.write(buf, 0, read)
                                written += read
                                val d = done.addAndGet(read.toLong())
                                val now = System.currentTimeMillis()
                                val elapsed = (now - startTime).coerceAtLeast(1)
                                val speed = (d * 1000L / elapsed).coerceAtLeast(0)
                                onProgress(d, totalBytes, speed)
                            }
                            out.flush()
                        }
                    }
                    val f = partFile.length()
                    f
                } catch (e: Exception) {
                    0L
                }
            }
        }
        results.awaitAll().sum()
    }

    /** 合并分块到最终文件 */
    private fun mergeChunks(dir: File, partPrefix: String, totalBytes: Long, target: File) {
        target.delete()
        RandomAccessFile(target, "rw").use { raf ->
            raf.setLength(totalBytes)
            var offset = 0L
            var i = 0
            while (offset < totalBytes) {
                val part = File(dir, "$partPrefix.part.$i")
                if (part.exists()) {
                    part.inputStream().use { ins ->
                        raf.seek(offset)
                        val buf = ByteArray(1 shl 20)
                        var read: Int
                        while (ins.read(buf).also { read = it } != -1) {
                            raf.write(buf, 0, read)
                        }
                    }
                    offset += part.length()
                    part.delete()
                } else {
                    break
                }
                i++
            }
        }
        // 清理残留分块
        for (j in 0..32) {
            File(dir, "$partPrefix.part.$j").delete()
        }
    }

    fun deleteModel(): Boolean = modelFile.delete()

    // ================= AI 重绘模型 zip 云端下载（HF）=================

    /**
     * ★ AI 重绘 SD 模型 zip 的 HF 云下载。
     * 直接复用 [downloadModel]（镜像循环+探测+302+4线程分块+断点续传+merge）。
     *
     * @param repoId  HF 仓库 id，如 xororz/sd-qnn
     * @param fileName 仓库内 zip 文件名，如 MeinaMixV12_qnn2.28_8gen2.zip
     * @param destFile 保存目标（建议 filesDir/ai_models/ 下）
     * @param token    HF Token（私有/gated 仓库必需，可空）
     * @param listener 进度回调（主线程）
     * @return 成功 true
     */
    suspend fun downloadAiRedrawZip(
        context: Context,
        repoId: String,
        fileName: String,
        destFile: File,
        token: String,
        listener: ProgressListener? = null
    ): Boolean {
        initContext(context)
        val dir = destFile.parentFile ?: File(context.filesDir, "ai_models")
        if (!dir.exists()) dir.mkdirs()
        if (destFile.exists() && destFile.length() >= 100_000_000L) return true
        return downloadModel(context, hfRepo = repoId, hfFile = fileName,
            destFile = destFile, tokenOverride = token, listener = listener)
    }

    /**
     * ★ AI 重绘 SD 模型 zip 的【直链】云下载（HF resolve / CDN / 任意托管）。
     *   - HF resolve 直链：自动拆 repoId/fileName 走镜像循环（国内镜像更快更稳）；
     *   - 其他直链：多线程分块原样下载 zip（不解压、不提取）。
     *   保存文件名见 [destFile]（调用方取自 URL 末段 → modelId 即真实模型名，多模型不串目录）。
     *
     * @param url      完整直链（http/https）
     * @param destFile 保存目标（建议 filesDir/ai_models/ 下；文件名取自 URL 末段）
     * @param token    HF Token（私有/gated 仓库必需，可空）
     * @param listener 进度回调（主线程）
     * @return 成功 true
     */
    suspend fun downloadAiRedrawZipFromUrl(
        context: Context,
        url: String,
        destFile: File,
        token: String,
        listener: ProgressListener? = null
    ): Boolean {
        initContext(context)
        val dir = destFile.parentFile ?: File(context.filesDir, "ai_models")
        if (!dir.exists()) dir.mkdirs()
        if (destFile.exists() && destFile.length() >= 100_000_000L) return true

        // ★ HF resolve 直链：拆 repoId/fileName 走镜像循环（多镜像重试，更快更稳）
        val hfMatch = Regex("https?://[^/]+/([^/]+/[^/]+)/resolve/([^/]+)/(.+)")
            .find(url.trim())
        if (hfMatch != null) {
            val repoId = hfMatch.groupValues[1]
            val fileName = hfMatch.groupValues[3]
            if (repoId.isNotBlank() && fileName.isNotBlank()) {
                return downloadModel(context, hfRepo = repoId, hfFile = fileName,
                    destFile = destFile, tokenOverride = token, listener = listener)
            }
        }
        // 其他直链：zip 原样下载
        return downloadUrlZipToFile(context, url, destFile, token, listener)
    }

    /**
     * ★ 直链分块下载 AI 重绘 zip 到目标文件（探测+302+4线程分块+断点续传+merge）。
     *   zip 原样保存（不做 onnx 提取——那是抠图模型用的 downloadModelFromUrl 逻辑）。
     */
    private suspend fun downloadUrlZipToFile(
        context: Context,
        url: String,
        destFile: File,
        token: String,
        listener: ProgressListener? = null
    ): Boolean = withContext(Dispatchers.IO) {
        initContext(context)
        val dir = destFile.parentFile ?: File(context.filesDir, "ai_models")
        if (!dir.exists()) dir.mkdirs()

        val mainHandler = Handler(Looper.getMainLooper())
        val client = buildClient()

        // 请求构建（带 HF Token，gated/私有托管可能需要）
        fun buildReq(u: String): Request {
            val b = Request.Builder().url(u)
            if (token.isNotBlank()) b.header("Authorization", "Bearer $token")
            return b.build()
        }

        try {
            // ★ ① 手动解析重定向（client 已关自动跟随）拿到真实下载 URL
            //   HF resolve 直链 / 其他托管 302 → 签名 CDN URL，Range 分块只有对真实 URL 才生效。
            var realUrl = url
            try {
                client.newCall(buildReq(url)).execute().use { p ->
                    if (p.code == 302 || p.code == 307 || p.code == 308) {
                        p.header("Location")?.takeIf { it.startsWith("http") }?.let { realUrl = it }
                    }
                }
            } catch (_: Exception) { /* 直接用 url */ }

            // ② 探测总大小（Range: bytes=0-0 对真实下载 URL 生效，206 + Content-Range）
            val probeReq = buildReq(realUrl).newBuilder().header("Range", "bytes=0-0").build()
            var totalBytes = 0L
            try {
                client.newCall(probeReq).execute().use { resp ->
                    if (resp.isSuccessful || resp.code == 206) {
                        val cr = resp.header("Content-Range")
                        totalBytes = cr?.substringAfter('/')?.toLongOrNull()
                            ?: resp.body?.contentLength() ?: 0L
                    } else {
                        mainHandler.post { listener?.onError(IllegalStateException("HTTP ${resp.code}")) }
                        return@withContext false
                    }
                }
            } catch (e: Exception) {
                mainHandler.post { listener?.onError(e) }
                return@withContext false
            }
            if (totalBytes < 100_000_000L) {
                mainHandler.post { listener?.onError(IllegalStateException("文件过小或无效链接（需 SD 模型 zip ≥100MB）")) }
                return@withContext false
            }

            // ③ 断点续传：已有完整 zip 则跳过
            if (destFile.exists() && destFile.length() >= totalBytes) {
                mainHandler.post { listener?.onDone(destFile) }
                return@withContext true
            }

            // ④ 分块下载到 .part（直接对签名 CDN URL 发 Range）
            val partBase = destFile.name
            val downloadedAll = downloadChunks(
                client = client, baseUrl = realUrl, totalBytes = totalBytes,
                dir = dir, partPrefix = partBase, token = token,
                threads = DOWNLOAD_THREADS,
                onProgress = { d, t, sp -> mainHandler.post { listener?.onProgress(d, t, sp) } }
            )
            if (downloadedAll < totalBytes) {
                mainHandler.post { listener?.onError(IllegalStateException("下载不完整")) }
                return@withContext false
            }
            // ⑤ 合并分块 → 最终 zip（zip 原样保存）
            mergeChunks(dir, partBase, totalBytes, destFile)
            for (j in 0..32) File(dir, "$partBase.part.$j").delete()

            mainHandler.post { listener?.onDone(destFile) }
            return@withContext true
        } catch (e: Exception) {
            mainHandler.post { listener?.onError(e) }
            return@withContext false
        }
    }
}