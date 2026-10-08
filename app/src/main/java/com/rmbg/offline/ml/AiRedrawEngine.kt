package com.rmbg.offline.ml

import android.content.Context
import android.util.Log
import com.rmbg.offline.Prefs
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.util.zip.ZipInputStream

/**
 * AI 重绘本地引擎管理器（LocalDream stable_diffusion_core 自加载）。
 *
 * 链路：
 *  1. 部署模型：把 AnythingV5_qnn2.28_8gen2.zip 解压平铺到 filesDir/models/anythingv5/
 *     （unet.bin / vae_decoder.bin / vae_encoder.bin / clip_v2.mnn / tokenizer.json /
 *       pos_emb.bin / token_emb.bin / *.patch）
 *  2. 部署 QNN 运行时：assets/qnnlibs 下的 .so → filesDir/qnnlibs/（stub/skel 全套）
 *  3. 启动 native 引擎进程（exec）：
 *     libstable_diffusion_core.so --type sd15npu --model_dir <模型目录> --lib_dir <qnnlibs> --port 8081
 *  4. 轮询 http://127.0.0.1:8081 就绪
 *  5. 由 LocalDreamClient 调 /generate 做 img2img
 *
 * 引擎进程是独立 native 可执行文件（非 JNI 库），通过 ProcessBuilder 启动，
 * 自带 HTTP server。停止时 kill 进程。
 */
object AiRedrawEngine {
    private const val TAG = "AI-REDRAW-ENGINE"
    private const val ENGINE_NAME = "libstable_diffusion_core.so"
    private const val MODEL_ID = "anythingv5"
    private const val PORT = 8081
    private const val API_URL = "http://127.0.0.1:$PORT"

    // 模型 zip 内要解压的关键文件（LocalDream 平铺到 modelDir 根）
    // ★ 两种 VAE/CLIP 格式并存：
    //   - .bin        ：QNN/MNN 单文件（AnythingV5 等老包）
    //   - .mnn+.weight：MNN 分离权重（HentaiGapeMix V10 等新包：vae_decoder.mnn + vae_decoder.mnn.weight）
    private val REQUIRED_MODEL_FILES = listOf(
        "unet.bin",
        "vae_decoder.bin", "vae_decoder.mnn", "vae_decoder.mnn.weight",
        "vae_encoder.bin", "vae_encoder.mnn", "vae_encoder.mnn.weight",
        "clip_v2.mnn", "clip_v2.mnn.weight",
        "tokenizer.json", "pos_emb.bin", "token_emb.bin"
    )

    @Volatile
    private var process: Process? = null

    /** 当前已启动进程服务的模型目录（多模型：detectModelDir 变化时须 stop→start 换模型） */
    @Volatile
    private var activeModelDir: String? = null

    /** 当前已启动进程服务的分辨率（宽×高；切分辨率也须重启） */
    @Volatile
    private var activeRes: Pair<Int, Int>? = null

    /** 引擎是否已启动（进程活着且 8081 响应） */
    @Volatile
    var running: Boolean = false
        private set

    /** 最近状态信息（供 UI 展示） */
    @Volatile
    var status: String = "未启动"
        private set

    /** 引擎日志尾部（供 UI 诊断 QNN 是否生效） */
    @Volatile
    var logTail: String = ""
        private set

    private val logLock = Any()

    /** 引擎日志文件（filesDir/engine.log） */
    fun logFile(context: Context): File = File(context.filesDir, "engine.log")

    // ★ 日志写盘缓冲：appendLog 只累积到内存 + 更新 logTail，
    //   由后台线程每秒 flush 一次（避免每行都打开-写-关文件，拖垮 UI/IO）
    private val logBuf = StringBuilder()
    @Volatile private var logFlusherStarted = false
    @Volatile private var logFlusherContext: Context? = null

    /** 追加一行引擎日志：写文件（缓冲） + 保留尾部（供 UI 展示） */
    private fun appendLog(context: Context, line: String) {
        val clean = line.trim()
        if (clean.isEmpty()) return
        synchronized(logLock) {
            logBuf.append(clean).append('\n')
            val t = (logTail + clean + "\n").takeLast(2000)
            logTail = t
        }
        Log.i(TAG, "[engine] $clean")
        // 确保后台 flush 线程已启动（只启动一次，绑定第一个 context）
        if (!logFlusherStarted) {
            synchronized(logLock) {
                if (!logFlusherStarted) {
                    logFlusherStarted = true
                    logFlusherContext = context.applicationContext
                    Thread {
                        while (logFlusherStarted) {
                            try { Thread.sleep(1000) } catch (_: Exception) { break }
                            flushLogToFile()
                        }
                    }.start()
                }
            }
        }
    }

    /** 把缓冲日志批量写盘（后台线程调用） */
    private fun flushLogToFile() {
        val ctx = logFlusherContext ?: return
        var chunk: String? = null
        synchronized(logLock) {
            if (logBuf.isNotEmpty()) {
                chunk = logBuf.toString()
                logBuf.setLength(0)
            }
        }
        chunk?.let {
            try { logFile(ctx).appendText(it) } catch (_: Exception) {}
        }
    }

    /** 清空引擎日志（每次启动前） */
    private fun clearLog(context: Context) {
        synchronized(logLock) {
            logTail = ""
            logBuf.setLength(0)
            try { logFile(context).writeText("") } catch (_: Exception) {}
        }
    }

    /** 模型 zip 文件（外部传入，用户选择；启动时从 Prefs 恢复；部署成功后 zip 会被删除） */
    var modelZipFile: File? = null

    /** 当前活动模型 id（部署目录名；zip 删除后仍能定位模型，多模型切换用） */
    @Volatile
    var activeModelId: String = ""
        private set

    /** 设置当前活动模型（选包/下载完成时调用） */
    fun selectModel(zipFile: File) {
        modelZipFile = zipFile
        activeModelId = modelIdFor(zipFile)
        Prefs.aiRedrawModelId = activeModelId
    }

    /** 当前模型目录（按活动模型 id 定位，不依赖 zip 文件是否还在 → zip 删除后引擎照常启动） */
    fun modelDir(context: Context): File {
        val id = activeModelId.ifBlank { modelIdFor(modelZipFile) }
        return File(File(context.filesDir, "models"), id)
    }

    /** 指定 zip 的模型目录（多模型：按 zip 文件名生成独立子目录，多个模型并存互不覆盖） */
    fun modelDirFor(context: Context, zipFile: File?): File =
        File(File(context.filesDir, "models"), modelIdFor(zipFile))

    /**
     * ★ 已部署模型列表（扫描 filesDir/models/ 下含 unet.bin 的目录）。
     * zip 部署后会删除源包，但解压目录保留；多模型共存靠这个列表切换，无需重新导入 zip。
     * @return 已部署的模型 id 列表（按最后修改时间倒序，最新在前）
     */
    fun deployedModels(context: Context): List<String> {
        val modelsDir = File(context.filesDir, "models")
        return try {
            modelsDir.listFiles { f -> f.isDirectory && File(f, "unet.bin").exists() }
                ?.sortedByDescending { it.lastModified() }
                ?.map { it.name } ?: emptyList()
        } catch (_: Exception) { emptyList() }
    }

    /**
     * ★ 切换到已部署模型（不依赖 zip，直接按部署目录 id 切换）。
     * @return 是否切换成功（目录存在且含 unet.bin）
     */
    fun selectDeployedModel(context: Context, modelId: String): Boolean {
        val dir = File(File(context.filesDir, "models"), modelId)
        if (!File(dir, "unet.bin").exists()) return false
        activeModelId = modelId
        Prefs.aiRedrawModelId = modelId
        modelZipFile = null  // zip 已删，靠部署目录定位
        return true
    }

    /**
     * ★ 删除已部署的 AI 重绘模型（解压目录 + 残留 zip 源包一并清理）。
     * 若删的是当前活动模型，清空活动 id 并回退到空状态。
     * @return 是否删除成功（目录或 zip 至少清掉一个）
     */
    fun deleteDeployedModel(context: Context, modelId: String): Boolean {
        var deleted = false
        try {
            val dir = File(File(context.filesDir, "models"), modelId)
            if (dir.exists()) {
                dir.deleteRecursively()
                deleted = true
            }
        } catch (_: Exception) {}
        // 顺带清理同名残留 zip（ai_models 下同名或同 id 的包）
        try {
            val zipDir = File(context.filesDir, "ai_models")
            zipDir.listFiles { f ->
                f.isFile && f.name.endsWith(".zip") &&
                    (f.name.substringBeforeLast('.') == modelId || f.name == "$modelId.zip")
            }?.forEach { it.delete() }
        } catch (_: Exception) {}
        // 删的是当前活动模型 → 清空活动状态
        if (activeModelId == modelId) {
            activeModelId = ""
            modelZipFile = null
            try { Prefs.aiRedrawModelId = "" } catch (_: Exception) {}
            try { Prefs.aiModelZipPath = "" } catch (_: Exception) {}
        }
        return deleted
    }

    /** 从 zip 文件名生成模型 id（多模型共存：AnythingV5/MeinaMix 各自独立目录） */
    fun modelIdFor(zipFile: File?): String {
        val name = zipFile?.name?.substringBeforeLast('.') ?: MODEL_ID
        val safe = name.replace(Regex("[^A-Za-z0-9._-]"), "_").trim('_')
        return safe.ifBlank { MODEL_ID }
    }

    /** 模型部署指纹文件（记录来源 zip 路径+大小，换包自动重部署；按模型目录隔离） */
    private fun modelStampFile(context: Context): File = File(modelDir(context), ".zip_stamp")

    /**
     * 模型是否已部署（unet.bin 存在 + 指纹匹配目标 zip）。
     * 按传入 zip 独立判定（多模型共存：每个 zip 各自目录，互不干扰）。
     * 重绘时用它判断是否跳过 1GB 解压——部署一次后续直接复用。
     */
    fun isModelDeployed(context: Context, zipFile: File?): Boolean {
        val dir = if (zipFile != null) modelDirFor(context, zipFile) else modelDir(context)
        if (!File(dir, "unet.bin").exists()) return false
        val zip = zipFile ?: return true // 无 zip 时只要 unet.bin 在就算已部署
        // ★ zip 已删除（部署后自动清理）：unet.bin 在就算已部署，不强制重部署
        if (!zip.exists()) return true
        try {
            val stamp = File(dir, ".zip_stamp").readText()
            val expect = "${zip.absolutePath}|${zip.length()}"
            return stamp == expect
        } catch (_: Exception) {
            return true // 指纹读取失败：unet.bin 在，视为已部署（不强制重解压）
        }
    }

    /**
     * ★ 模型是否可用（可启动引擎）：zip 存在 或 已部署（unet.bin 在部署目录）。
     * zip 部署成功后会被删除，靠 activeModelId 定位部署目录；此判断兼容两种情况。
     */
    fun isModelReady(context: Context): Boolean {
        if (modelZipFile?.exists() == true) return true
        if (activeModelId.isNotBlank() && File(modelDir(context), "unet.bin").exists()) return true
        // 兜底：活动 id 为空但 Prefs 有记录
        val savedId = Prefs.aiRedrawModelId
        if (savedId.isNotBlank()) {
            val dir = File(File(context.filesDir, "models"), savedId)
            if (File(dir, "unet.bin").exists()) return true
        }
        return false
    }

    /** QNN 运行时目录（filesDir/qnnlibs/） */
    fun runtimeDir(context: Context): File =
        File(context.filesDir, "qnnlibs")

    /**
     * 模型包类型（仅 SD1.5 系，统一 sd15npu）。
     * 不同 SD1.5 底模包（AnythingV5/MeinaMix/DarkSushi 等）差异只在文件命名：
     *  clip 可能是 clip_v2.mnn / clip.mnn；tokenizer 可能内置在 clip 或独立 tokenizer.json。
     *  校验按 zip 内实际存在的文件判断，缺一不可的都报缺失。
     */
    data class ModelPackInfo(
        val type: String,          // 恒为 sd15npu
        val keyFiles: List<String> // 该校验的关键文件（实际存在的）
    )

    /** 扫描 zip 内条目名（不读取内容，仅列名，1GB 包扫描快） */
    private fun scanZipEntries(zipFile: File): List<String> = try {
        java.util.zip.ZipFile(zipFile).use { zf ->
            val names = zf.entries().asSequence().map { it.name.substringAfterLast('/') }.toList()
            names
        }
    } catch (e: Exception) {
        Log.e(TAG, "scanZipEntries failed", e)
        emptyList()
    }

    /** 识别 SD1.5 模型包并校验关键文件（恒 sd15npu，兼容不同底模包命名） */
    fun detectModelPack(zipFile: File): ModelPackInfo {
        val names = scanZipEntries(zipFile).toSet()
        // 必需：unet.bin（所有 SD1.5 QNN 包都有）
        val must = mutableListOf("unet.bin")
        // VAE：vae_decoder.bin（QNN 单文件）或 vae_decoder.mnn + vae_decoder.mnn.weight（MNN 分离权重）任一存在即可
        val hasDecoderBin = names.any { it == "vae_decoder.bin" || it == "vae_decoder.mnn" || it == "vae_decoder.mnn.weight" }
        if (!hasDecoderBin) must.add("vae_decoder.bin") // 标记缺失（兼容旧包名）
        // CLIP：clip_v2.mnn / clip.mnn 至少一个
        val hasClip = names.any { it == "clip_v2.mnn" || it == "clip.mnn" }
        if (!hasClip) {
            // 若无 .mnn，允许 text_encoder.bin（部分包用 onnx/mnn 混合）
            if (names.any { it.startsWith("text_encoder") }) {
                must.add(names.first { it.startsWith("text_encoder") })
            } else {
                must.add("clip_v2.mnn") // 标记缺失
            }
        }
        // tokenizer：tokenizer.json（部分包内置 BPE 不需要独立文件）
        if (names.any { it == "tokenizer.json" }) must.add("tokenizer.json")
        return ModelPackInfo("sd15npu", must)
    }

    /** 校验模型包关键文件齐全（按类型动态） */
    fun validateModelPack(zipFile: File): String? {
        val info = detectModelPack(zipFile)
        val names = scanZipEntries(zipFile).toSet()
        val missing = info.keyFiles.filter { !names.contains(it) }
        return if (missing.isEmpty()) null else "缺少关键文件: ${missing.joinToString(",")}"
    }

    /** 从 Prefs 恢复持久化的模型 zip 路径 + 活动模型 id（启动/重绘前调用）
     *  zip 已删除（部署成功后清理了）不报错：模型目录按 activeModelId 定位，不依赖 zip */
    fun restoreModelZip(context: Context) {
        // ★ 优先恢复活动模型 id（zip 删除后仍能定位部署目录）
        activeModelId = Prefs.aiRedrawModelId
        // ★ 已有活动模型 id 且其部署目录有 unet.bin（已部署/zip 已清理）：
        //   直接视为就绪，不挂任何 zip——避免残留的未部署 zip 干扰切换（modelZipFile 会串目录）。
        if (activeModelId.isNotBlank()) {
            val deployedDir = File(File(context.filesDir, "models"), activeModelId)
            if (File(deployedDir, "unet.bin").exists()) {
                modelZipFile = null
                return
            }
        }
        val p = Prefs.aiModelZipPath
        if (p.isNotEmpty()) {
            val f = File(p)
            if (f.exists() && f.length() >= 100_000_000L) {
                modelZipFile = f
                // 活动 id 为空时按 zip 名推导（首次迁移/旧版本）
                if (activeModelId.isBlank()) activeModelId = modelIdFor(f)
                return
            }
        }
        // 兜底：扫描 ai_models 目录（zip 还在的场景）
        try {
            val dir = File(context.filesDir, "ai_models")
            val zips = dir.listFiles()
                ?.filter { it.isFile && it.name.endsWith(".zip") && it.length() >= 100_000_000L }
                ?: emptyList()
            // ★ 优先选"已部署过"的 zip（其部署目录有 unet.bin），避免挑到残留未部署包（时间戳 zip）
            val deployedZip = zips.firstOrNull { f ->
                val modelName = f.name.substringBeforeLast('.')
                File(File(context.filesDir, "models"), modelName).let { File(it, "unet.bin").exists() }
            }
            val chosen = deployedZip ?: zips.maxByOrNull { it.lastModified() }
            chosen?.let { f ->
                modelZipFile = f
                if (activeModelId.isBlank()) activeModelId = modelIdFor(f)
            }
        } catch (_: Exception) {}
    }

    /** 从 engine.log 文件读回日志尾部（App 重启后日志 UI 仍可见） */
    fun restoreLogTail(context: Context) {
        if (logTail.isNotBlank()) return // 已有内存日志
        try {
            val f = logFile(context)
            if (f.exists() && f.length() > 0) {
                val all = f.readText()
                logTail = all.takeLast(2000)
            }
        } catch (_: Exception) {}
    }

    /**
     * 部署模型 zip 到模型目录（平铺 basename；多模型：按 zip 名独立目录，不删除其他模型）。
     * @param activate 部署后是否立即切换为活动模型：
     *   true  = 导入/下载完成自动部署（激活该模型，写 Prefs）
     *   false = 启动补全部署（仅部署目录入列表，不改变当前活动模型；部署完成后由 restoreModelZip 恢复定位）
     */
    suspend fun deployModel(context: Context, zipFile: File, activate: Boolean = true): Boolean = withContext(Dispatchers.IO) {
        try {
            val dir = modelDirFor(context, zipFile)
            // ★ 仅清理当前 zip 对应目录（多模型共存：不 deleteRecursively models/ 整个父目录）
            if (dir.exists()) dir.deleteRecursively()
            dir.mkdirs()

            ZipInputStream(zipFile.inputStream().buffered()).use { zis ->
                var entry = zis.nextEntry
                while (entry != null) {
                    if (!entry.isDirectory) {
                        val fileName = entry.name.substringAfterLast('/')
                        if (fileName.isNotEmpty() && !fileName.startsWith(".") && !fileName.startsWith("__MACOSX")) {
                            val out = File(dir, fileName)
                            FileOutputStream(out).use { os -> zis.copyTo(os) }
                        }
                    }
                    zis.closeEntry()
                    entry = zis.nextEntry
                }
            }
            // 校验关键文件齐全（按模型包类型动态识别，兼容不同 SD 模型包）
            val missing = validateModelPack(zipFile)
            if (missing != null) {
                status = "模型文件缺失: $missing"
                return@withContext false
            }
            // 写部署指纹（来源 zip 路径+大小，换包自动重部署）
            try {
                File(dir, ".zip_stamp").writeText("${zipFile.absolutePath}|${zipFile.length()}")
            } catch (_: Exception) {}
            status = "模型已部署: ${dir.absolutePath}"
            modelZipFile = zipFile
            if (activate) {
                // ★ 导入/下载自动部署：立即激活该模型
                activeModelId = modelIdFor(zipFile)
                Prefs.aiRedrawModelId = activeModelId
                Prefs.aiModelZipPath = zipFile.absolutePath
            }
            // activate=false（启动补全部署）：保持当前 activeModelId/Prefs 不变，
            //   仅让部署目录入列表；zip 删除后 restoreModelZip 会把 modelZipFile 清回 null。
            // ★ 部署成功后删除 zip 源包（引擎只认解压目录，zip 留着白占 ~1GB/模型；
            //   模型定位已切到 activeModelId + 部署目录，删 zip 不影响后续启动）
            try {
                if (zipFile.exists() && zipFile.delete()) {
                    Log.i(TAG, "部署完成，已删除模型 zip 源包: ${zipFile.name}")
                }
            } catch (_: Exception) {}
            true
        } catch (e: Exception) {
            status = "模型部署失败: ${e.message}"
            Log.e(TAG, "deployModel failed", e)
            false
        }
    }

    /** 部署 QNN 运行时（nativeLibraryDir → filesDir/qnnlibs）
     *  ★ 合并两套 QNN so：QNN 库（libQnnHtp/libQnnSystem/V68~V81 Skel/Stub）已由 qnn-runtime AAR
     *    打包进 nativeLibraryDir（抠图进程加载同一份），此处不再从 assets 二次打包，直接复制
     *    nativeLibraryDir 已有的库到 filesDir/qnnlibs 供 AI 重绘引擎进程 LD_LIBRARY_PATH 使用，
     *    APK 约省 100MB。 */
    suspend fun deployRuntime(context: Context): Boolean = withContext(Dispatchers.IO) {
        try {
            val dir = runtimeDir(context)
            if (!dir.exists()) dir.mkdirs()
            val nativeDir = File(context.applicationInfo.nativeLibraryDir)
            // 引擎需要的 QNN 库：nativeLibraryDir 下所有 libQnn*.so（Htp + System + V68~V81 Skel/Stub 全套）
            val srcSo = nativeDir.listFiles { f -> f.isFile && f.name.endsWith(".so") && f.name.startsWith("libQnn") }
                ?: emptyArray()
            if (srcSo.isEmpty()) {
                status = "nativeLibraryDir 无 QNN 运行时库（未打包 qnn-runtime AAR）"
                return@withContext false
            }
            var ok = 0
            for (src in srcSo) {
                val target = File(dir, src.name)
                val srcSize = src.length()
                if (!target.exists() || target.length() != srcSize) {
                    src.copyTo(target, overwrite = true)
                }
                target.setReadable(true, true)
                target.setExecutable(true, true)
                ok++
            }
            status = "QNN 运行时已部署: $ok 个库（来自 nativeLibraryDir）"
            true
        } catch (e: Exception) {
            status = "QNN 运行时部署失败: ${e.message}"
            Log.e(TAG, "deployRuntime failed", e)
            false
        }
    }

    /** 启动 native 引擎进程 */
    suspend fun start(context: Context, width: Int = 512, height: Int = 512): Boolean = withContext(Dispatchers.IO) {
        // ★ 立即清空并记录启动日志（无论成败都留痕，方便诊断"没跑起来"）
        clearLog(context)
        appendLog(context, "==== 引擎启动 ${java.text.SimpleDateFormat("HH:mm:ss", java.util.Locale.US).format(java.util.Date())} ====")
        try {
            val targetModelDir = modelDir(context).absolutePath
            // ★ 多模型/分辨率检测：进程活着但服务的是旧模型或旧分辨率 → 必须 stop→start，
            //   否则切 MeinaMix 后旧 AnythingV5 的 unet 继续服务（之前只认 isAlive 直接跳过的 bug）。
            if (process?.isAlive == true) {
                val sameModel = activeModelDir == targetModelDir
                val sameRes = activeRes == (width to height)
                if (sameModel && sameRes) {
                    status = "引擎已在运行（模型/分辨率匹配）"
                    running = true
                    appendLog(context, "引擎已在运行，跳过启动")
                    return@withContext true
                }
                appendLog(context, "引擎运行中但模型/分辨率变化（${activeModelDir ?: "?"} @ ${activeRes ?: "?"} → $targetModelDir @ ${width}x$height），重启引擎")
                stop(context)
            }

            // 1. 引擎可执行文件（jniLibs → nativeLibraryDir）
            val nativeDir = File(context.applicationInfo.nativeLibraryDir)
            val executable = File(nativeDir, ENGINE_NAME)
            if (!executable.exists()) {
                status = "引擎未找到: $ENGINE_NAME（APK 未打包 native 引擎）"
                appendLog(context, "❌ 引擎未找到: $ENGINE_NAME")
                return@withContext false
            }
            // 引擎 so 需可执行权限
            executable.setExecutable(true, true)

            // 2. 模型目录
            val modelsDir = modelDir(context)
            if (!modelsDir.exists()) {
                status = "模型目录不存在: ${modelsDir.absolutePath}，请先部署模型"
                appendLog(context, "❌ 模型目录不存在: ${modelsDir.absolutePath}")
                return@withContext false
            }
            // 3. QNN 运行时目录
            val libDir = runtimeDir(context)
            if (!libDir.exists()) {
                status = "QNN 运行时目录不存在，请先部署运行时"
                appendLog(context, "❌ QNN 运行时目录不存在")
                return@withContext false
            }

            // 4. 拼启动命令（对齐 LocalDream BackendService）
            // ★ 按模型包类型动态选 --type（兼容 SD1.5 / SDXL 等不同模型包）
            val packType = detectModelPack(modelZipFile ?: File(modelsDir, "unet.bin")).type
            val cmd = mutableListOf(
                executable.absolutePath,
                "--type", packType,
                "--model_dir", modelsDir.absolutePath,
                "--port", PORT.toString()
            )
            // 非 CPU 类型必须传 --lib_dir（QNN 运行时目录）
            if (packType != "sd15cpu") cmd += listOf("--lib_dir", libDir.absolutePath)

            // ★ 对齐 LocalDream：非 512×512 档位必须传 --patch（QNN context 高分辨率补丁）
            if (width != 512 || height != 512) {
                val patchFile = if (width == height) {
                    val squarePatch = File(modelsDir, "$width.patch")
                    if (squarePatch.exists()) squarePatch else File(modelsDir, "${width}x$height.patch")
                } else {
                    File(modelsDir, "${width}x$height.patch")
                }
                if (patchFile.exists()) {
                    cmd += listOf("--patch", patchFile.absolutePath)
                    Log.i(TAG, "Using patch file: ${patchFile.name}")
                } else {
                    Log.w(TAG, "Patch file not found: ${patchFile.absolutePath}, falling back to 512×512")
                }
            }

            // 5. 环境变量（QNN 必需，对齐 LocalDream BackendService）
            val env = HashMap<String, String>()
            // LD_LIBRARY_PATH：运行时目录 + 系统库路径（GPU/EGL 等）
            val systemLibPaths = mutableListOf(
                libDir.absolutePath,
                "/system/lib64",
                "/vendor/lib64",
                "/vendor/lib64/egl",
            )
            try {
                val maliSymlink = File("/system/vendor/lib64/egl/libGLES_mali.so")
                if (maliSymlink.exists()) {
                    val realPath = maliSymlink.canonicalPath
                    val soc = realPath.split("/").getOrNull(realPath.split("/").size - 2)
                    if (soc != null) {
                        listOf("/vendor/lib64/$soc", "/vendor/lib64/egl/$soc").forEach { p ->
                            if (!systemLibPaths.contains(p)) systemLibPaths.add(p)
                        }
                    }
                }
            } catch (_: Exception) {}
            env["LD_LIBRARY_PATH"] = systemLibPaths.joinToString(":")
            // DSP 搜索路径：运行时目录（LocalDream sd15npu 分支只设 DSP_LIBRARY_PATH）
            env["DSP_LIBRARY_PATH"] = libDir.absolutePath
            // ★ 额外补 FastRPC 平台路径（分号分隔），保证 skel 能解析
            val adspPath = listOf(
                libDir.absolutePath,
                "/vendor/lib/rfsa/adsp",
                "/vendor/dsp/cdsp",
                "/dsp",
            ).joinToString(";")
            env["ADSP_LIBRARY_PATH"] = adspPath

            Log.i(TAG, "启动命令: ${cmd.joinToString(" ")}")
            Log.i(TAG, "DSP_LIBRARY_PATH=${libDir.absolutePath}")
            Log.i(TAG, "ADSP_LIBRARY_PATH=$adspPath")

            appendLog(context, "CMD: ${cmd.joinToString(" ")}")
            appendLog(context, "DSP_LIBRARY_PATH=${libDir.absolutePath} | ADSP_LIBRARY_PATH=$adspPath")

            val pb = ProcessBuilder(cmd)
            // ★ 对齐 LocalDream：working directory = nativeDir
            pb.directory(nativeDir)
            pb.environment().putAll(env)
            pb.redirectErrorStream(true)
            process = pb.start()

            // 读进程日志（后台）→ 写文件 + logTail
            val p = process!!
            Thread {
                try {
                    p.inputStream.bufferedReader().forEachLine { line ->
                        appendLog(context, line)
                    }
                } catch (_: Exception) {}
            }.start()

            // 6. 轮询 8081 就绪（最多 180 秒：加载 880MB unet + CLIP + shader 编译较慢）
            val deadline = System.currentTimeMillis() + 180_000
            var ready = false
            var lastHealthFail = ""
            while (System.currentTimeMillis() < deadline) {
                if (!p.isAlive) {
                    status = "引擎进程提前退出"
                    appendLog(context, "❌ 引擎进程提前退出（exit code ${p.exitValue()}）")
                    running = false
                    return@withContext false
                }
                if (checkHealth()) {
                    ready = true
                    break
                }
                lastHealthFail = "等待引擎就绪 ${(deadline - System.currentTimeMillis()) / 1000}s..."
                status = lastHealthFail
                Thread.sleep(500)
            }
            if (!ready) {
                status = "引擎启动超时（8081 未就绪）"
                appendLog(context, "❌ 引擎启动超时（180s 内 8081 未就绪）")
                running = false
                return@withContext false
            }
            running = true
            status = "引擎就绪（端口 $PORT）"
            appendLog(context, "==== 引擎就绪 ====")
            // ★ 记录当前服务模型/分辨率（多模型切换时 start() 据此判断是否需重启）
            activeModelDir = targetModelDir
            activeRes = width to height
            true
        } catch (e: Exception) {
            status = "引擎启动失败: ${e.message}"
            appendLog(context, "❌ 引擎启动失败: ${e.message}")
            Log.e(TAG, "start failed", e)
            running = false
            false
        }
    }

    /** 健康检查（GET /） */
    fun checkHealth(): Boolean = try {
        val conn = java.net.URL(API_URL).openConnection() as java.net.HttpURLConnection
        conn.connectTimeout = 2000
        conn.readTimeout = 2000
        val code = conn.responseCode
        conn.disconnect()
        code in 200..499
    } catch (_: Exception) { false }

    /** 停止引擎进程 */
    suspend fun stop(context: Context): Boolean = withContext(Dispatchers.IO) {
        val p = process
        if (p == null) {
            running = false
            status = "引擎未运行"
            return@withContext true
        }
        try {
            p.destroy()
            if (!p.waitFor(3, java.util.concurrent.TimeUnit.SECONDS)) p.destroyForcibly()
            running = false
            status = "引擎已停止"
            process = null
            activeModelDir = null
            activeRes = null
            true
        } catch (e: Exception) {
            status = "停止引擎失败: ${e.message}"
            false
        }
    }
}
