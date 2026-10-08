package com.rmbg.offline

import android.app.Activity
import android.content.ContentValues
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.waitForUpOrCancellation
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.TransformOrigin
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.runtime.produceState
import kotlinx.coroutines.delay
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.window.Dialog
import androidx.core.content.FileProvider
import com.rmbg.offline.ml.AiRedrawEngine
import com.rmbg.offline.ml.LocalDreamClient
import com.rmbg.offline.ml.ModelManager
import com.rmbg.offline.ml.RmbgOnnxEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.text.DecimalFormat
import kotlin.math.roundToInt

/** 最近一次抠图保存到历史的结果 PNG 文件（分享复用磁盘文件，避免重新编码） */
@Volatile
private var latestHistoryFile: File? = null

// ★ 模型包 zip 内容探测结果（下载后分流：AI 重绘 SD 包 / 抠图 QNN 包）
private const val ZIP_AI_REDRAW = 1   // 含 unet.bin → Stable Diffusion AI 重绘模型
private const val ZIP_MATTING = 2     // 含 .onnx(+.bin) → 抠图 QNN/ONNX 模型
private const val ZIP_UNKNOWN = 0     // 无法识别

/** ★ 探测 zip 包内容：只列条目名（不读取内容，1GB 包扫描快），判断模型包类型 */
private fun probeZipKind(zipFile: File): Int {
    return try {
        java.util.zip.ZipFile(zipFile).use { zf ->
            var hasUnet = false
            var hasOnnx = false
            val entries = zf.entries()
            while (entries.hasMoreElements()) {
                val name = entries.nextElement().name.substringAfterLast('/').lowercase()
                if (name == "unet.bin" || name.startsWith("unet_")) hasUnet = true
                if (name.endsWith(".onnx")) hasOnnx = true
                if (name.endsWith(".safetensors")) hasUnet = true // SD 权重包
                if (hasUnet) return@use ZIP_AI_REDRAW
            }
            if (hasOnnx) ZIP_MATTING else ZIP_UNKNOWN
        }
    } catch (_: Exception) { ZIP_UNKNOWN }
}

@OptIn(ExperimentalMaterial3Api::class, androidx.compose.foundation.ExperimentalFoundationApi::class)
class MainActivity : ComponentActivity() {

    /** 后台时释放引擎，降低常驻内存防被杀 */
    override fun onStop() {
        super.onStop()
        try {
            val eng = RmbgScreenState.engine
            if (eng != null) {
                // ★ 闪退修复：若推理进行中，不能立即 close native session（否则 IO 线程仍访问已释放 session 闪退）。
                //   只标记 closeRequested，等 removeBackground 的 finally 推理结束再真正 close。
                eng.cancelCurrentRun()   // 请求中断推理
                if (!eng.isRunning) {
                    eng.close()
                    RmbgScreenState.engine = null
                }
            }
        } catch (_: Exception) {}
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // 支持系统分享图片导入（ACTION_SEND 单图 / ACTION_SEND_MULTIPLE 多图批量）
        val sharedUris: List<Uri> = when (intent?.action) {
            Intent.ACTION_SEND -> {
                if (android.os.Build.VERSION.SDK_INT >= 33) {
                    val u = intent?.getParcelableExtra(Intent.EXTRA_STREAM, Uri::class.java)
                    if (u != null) listOf(u) else emptyList()
                } else {
                    @Suppress("DEPRECATION")
                    val u = intent?.getParcelableExtra<Uri>(Intent.EXTRA_STREAM)
                    if (u != null) listOf(u) else emptyList()
                }
            }
            Intent.ACTION_SEND_MULTIPLE -> {
                if (android.os.Build.VERSION.SDK_INT >= 33) {
                    intent?.getParcelableArrayListExtra(Intent.EXTRA_STREAM, Uri::class.java)
                        ?: emptyList()
                } else {
                    @Suppress("DEPRECATION")
                    intent?.getParcelableArrayListExtra<Uri>(Intent.EXTRA_STREAM)
                        ?: emptyList()
                }
            }
            else -> emptyList()
        }

        // 模型目录初始化 + 旧模型迁移（内部→外部）
        ModelManager.initContext(this)
        try {
            kotlinx.coroutines.runBlocking {
                withContext(Dispatchers.IO) { ModelManager.migrateModelsFromInternal(this@MainActivity) }
            }
        } catch (_: Exception) {}

        setContent {
            RmbgScreen(sharedUris)
        }
    }
}

object RmbgScreenState {
    @Volatile
    var engine: com.rmbg.offline.ml.RmbgOnnxEngine? = null
}

// ===== 关于页（独立二级页面）=====
@OptIn(ExperimentalMaterial3Api::class, androidx.compose.foundation.ExperimentalFoundationApi::class)
@Composable
private fun AboutPage(
    context: android.content.Context,
    onBack: () -> Unit
) {
    // ★ 版本号动态获取（不依赖 BuildConfig，内部测试版本号跟随 packageManager）
    val versionName = remember {
        runCatching {
            context.packageManager.getPackageInfo(context.packageName, 0).versionName
        }.getOrNull() ?: "1.0.0"
    }
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFFF5F7FA))
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        // 顶部：返回 + 标题
        Row(verticalAlignment = Alignment.CenterVertically) {
            IconButton(onClick = onBack) {
                Icon(Icons.Filled.ArrowBack, contentDescription = "返回", tint = MaterialTheme.colorScheme.primary)
            }
            Spacer(Modifier.width(8.dp))
            Text("关于", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
        }

        // App 图标 + 名称 + 版本号
        Card(
            colors = CardDefaults.cardColors(containerColor = Color(0xFFFFFFFF)),
            shape = RoundedCornerShape(16.dp)
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Box(
                    modifier = Modifier
                        .size(64.dp)
                        .clip(RoundedCornerShape(16.dp))
                        .background(MaterialTheme.colorScheme.primary),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(Icons.Filled.ContentCut, contentDescription = null, tint = Color.White, modifier = Modifier.size(32.dp))
                }
                Text("RMBG 离线抠图", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                // ★ 版本 + 版本类型（完整版/精简版 Lite）
                Text(
                    text = "版本 v$versionName${if (BuildConfig.IS_LITE) "（Lite 精简版）" else "（完整版）"}",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }

        // 特性说明
        Card(
            colors = CardDefaults.cardColors(containerColor = Color(0xFFFFFFFF)),
            shape = RoundedCornerShape(16.dp)
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Text("完全本地离线推理", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
                Text(
                    "图片不会上传到任何服务器，隐私安全。抠图、重绘、历史记录全部在设备本地完成。",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }

        // 更新日志
        Card(
            colors = CardDefaults.cardColors(containerColor = Color(0xFFFFFFFF)),
            shape = RoundedCornerShape(16.dp)
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Text("更新日志", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
                ChangelogRow("v1.1.2", listOf(
                    "超分模型云下载：按设备 SoC/NPU 自动匹配对应版本（QNN HTP / CPU 自适应）",
                    "Anime-Seg 支持多 NPU 变体，按设备自动选择下载链接",
                    "模型页新增超分模型下载入口，与内置模型同格式",
                    "内置模型列表按设备能力智能匹配"
                ))
                ChangelogRow("v1.1.1", listOf(
                    "双版本：完整版（内置 QNN 模型）/ 精简版（云下载）",
                    "QNN EP（骁龙 DSP）硬件加速，抠图更快更省电",
                    "AI 重绘（Stable Diffusion）本地生成",
                    "下载可中途停止，下载前自动检查网络连通性",
                    "精简安装体积：不再向存储区复制重复运行库"
                ))
                ChangelogRow("v1.1.0", listOf(
                    "多镜像源下载，断点续传 + 进度通知",
                    "模型管理：导入本地 / 切换 / 删除",
                    "历史记录与分享"
                ))
                ChangelogRow("v1.0.0", listOf(
                    "RMBG 离线抠图上线",
                    "ONNX Runtime CPU 推理，支持批量抠图"
                ))
            }
        }

        // 技术栈
        Card(
            colors = CardDefaults.cardColors(containerColor = Color(0xFFFFFFFF)),
            shape = RoundedCornerShape(16.dp)
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Text("技术栈", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
                Text(
                    "ONNX Runtime · QNN EP（骁龙 DSP 加速）· Jetpack Compose",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }

        // GitHub 项目（独立卡片，醒目可点击跳转）
        Card(
            colors = CardDefaults.cardColors(containerColor = Color(0xFFFFFFFF)),
            shape = RoundedCornerShape(16.dp),
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(16.dp))
                .clickable {
                    runCatching {
                        val i = android.content.Intent(android.content.Intent.ACTION_VIEW)
                        i.data = android.net.Uri.parse("https://github.com/zuimengcf/rmbg-offline")
                        context.startActivity(i)
                    }
                }
        ) {
            Row(
                modifier = Modifier.padding(16.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Box(
                    modifier = Modifier
                        .size(44.dp)
                        .clip(RoundedCornerShape(12.dp))
                        .background(Color(0xFF24292F)),
                    contentAlignment = Alignment.Center
                ) {
                    Text("🐙", style = MaterialTheme.typography.titleMedium)
                }
                Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    Text("GitHub 项目", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
                    Text(
                        "github.com/zuimengcf/rmbg-offline",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.primary,
                        fontWeight = FontWeight.Medium
                    )
                }
                Spacer(Modifier.weight(1f))
                Icon(Icons.Filled.OpenInNew, contentDescription = "打开", tint = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.size(18.dp))
            }
        }

        // 开源信息
        Card(
            colors = CardDefaults.cardColors(containerColor = Color(0xFFFFFFFF)),
            shape = RoundedCornerShape(16.dp)
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Text("开源", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
                Text(
                    "开源协议：Apache License 2.0",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Text(
                    "模型仓库：Hugging Face（详见项目 README）",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

/** 更新日志的一行：版本号 + 变更列表 */
@Composable
private fun ChangelogRow(version: String, items: List<String>) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Text(version, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Bold)
        items.forEach { item ->
            Text(
                "· $item",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

/** 高级编辑 · 快捷切模型（临时切换）恢复信息：重抠完成后自动恢复主模型配置 */
data class QuickModelRestore(
    val id: String,          // 主模型 id
    val repo: String,        // 主模型 hfRepo
    val file: String,        // 主模型 hfFile
    val enableQnn: Boolean   // 主模型 QNN 开关
)

@OptIn(ExperimentalMaterial3Api::class, androidx.compose.foundation.ExperimentalFoundationApi::class)
@Composable
fun RmbgScreen(sharedUris: List<Uri>) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    // ---- 状态（从 Prefs 恢复，防重启重置）----
    // 工程恢复：后台被杀后回来恢复原图/结果
    var originalBitmap by remember { mutableStateOf(ProjectStateManager.loadOriginal(context)) }
    var resultBitmap by remember { mutableStateOf(ProjectStateManager.loadResult(context)) }
    // ★ 历史分组键（当前会话）：记录"这棵历史树属于哪张原始原图"。
    //   选新图/批量下一张时 = computeOriginalKey(新图)（新图开新分支）；
    //   对结果"再次抠图"时保持不变（结果归原图分支：A→B→C 同一条链）。
    //   保存历史时用它作 originalKey（不再对每次输入重算，避免结果图另立分支）。
    var currentHistoryKey by remember { mutableStateOf("") }
    var threshold by remember { mutableFloatStateOf(Prefs.threshold) }
    var isProcessing by remember { mutableStateOf(false) }
    var isDownloading by remember { mutableStateOf(false) }
    var downloadingId by remember { mutableStateOf<String?>(null) }
    // ★ 可取消下载：保存当前下载协程引用，供「取消下载」按钮 cancel
    var downloadingJob by remember { mutableStateOf<kotlinx.coroutines.Job?>(null) }
    var downloadProgress by remember { mutableStateOf(0f) }
    var downloadSpeed by remember { mutableStateOf(0L) }
    var downloadBytes by remember { mutableStateOf(0L) }
    var downloadTotal by remember { mutableStateOf(0L) }
    val restoredModelId = Prefs.selectedModelId
    // ★ 设备能力自动识别（QNN/CPU + 锁定型号）：
    //   1) 检测当前设备 SoC 型号（骁龙？）与 QNN 支持（libcdsprpc 存在？）
    //   2) 若 QNN 可用且存在匹配当前型号的 QNN 模型 → 自动选中该 QNN 模型（如 8gen2 → qnn_animeseg）；
    //      否则回退 CPU 模型（anime_seg）
    //   3) 用户手动选择过的模型仍优先（Prefs 非空且存在时尊重手动选择）
    // ★ 旧版升级兼容：Prefs 存的旧默认值 anime_seg 会被当成"用户选择"→ 额外判断：
    //   若当前设备支持 QNN 且 Prefs 存的模型是 CPU 模型（anime_seg 等），则按设备能力自动升级到 QNN 模型
    val autoModelId = ModelManager.autoPickDefaultModel(context)
    val restoredModelIdSafe = run {
        val stored = ModelManager.builtinModels.find { it.id == restoredModelId }
        val userChosen = restoredModelId.isNotEmpty() &&
            stored != null &&
            ModelManager.qnnModelMatchesDevice(stored)
        if (userChosen) {
            // ★ 旧版遗留：用户没真正手动选过（存的是旧默认 anime_seg 之类 CPU 模型），
            //   且设备支持 QNN → 自动升级到 QNN 模型，让 8gen2 默认用 QNN
            if (stored!!.isQnn || !ModelManager.isQnnSupported(context)) {
                restoredModelId
            } else {
                Prefs.selectedModelId = autoModelId   // 写回 Prefs，之后尊重此选择
                autoModelId
            }
        } else {
            Prefs.selectedModelId = autoModelId   // ★ 写回 Prefs，之后重启尊重此选择
            autoModelId
        }
    }
    ModelManager.selectedModelId = restoredModelIdSafe
    var modelReady by remember { mutableStateOf(ModelManager.isModelDownloaded()) }
    // ★ 默认状态文案：按当前选中模型动态生成（QNN 模型=部署，CPU 模型=下载，体积取真实值）
    val initialStatus = if (modelReady) {
        "模型已就绪，选择图片开始抠图"
    } else {
        val curBm = ModelManager.builtinModels.find { it.id == restoredModelIdSafe }
        if (curBm?.isQnn == true) {
            "需要部署 ${curBm.name}（约${curBm.sizeBytes / 1024 / 1024}MB，点击部署）"
        } else {
            "需要下载模型（约${(curBm?.sizeBytes ?: 176_153_355L) / 1024 / 1024}MB）"
        }
    }
    var statusText by remember { mutableStateOf(initialStatus) }
    // ★ 最近一次抠图/选区处理的耗时（毫秒），用于常驻显示
    var lastElapsedMs by remember { mutableStateOf<Long?>(null) }
    // ★ 处理中的实时秒数（每秒跳动，完成时记录最终值）
    var runningSeconds by remember { mutableStateOf(0L) }
    // ★ 实时计时：isProcessing 为 true 时每秒递增；停止时归零
    LaunchedEffect(isProcessing) {
        if (isProcessing) {
            runningSeconds = 0
            var sec = 0L
            // 每秒 +1，直到处理结束
            while (isProcessing) {
                kotlinx.coroutines.delay(1000L)
                sec++
                runningSeconds = sec
            }
        }
    }

    // ---- 镜像源状态（从 Prefs 恢复）----
    var selectedMirrorIndex by remember { mutableStateOf(Prefs.mirrorIndex) }
    ModelManager.selectedMirrorIndex = selectedMirrorIndex
    var activeMirrorName by remember { mutableStateOf<String?>(null) }

    // ---- 模型配置（从 Prefs 恢复）----
    var modelRepo by remember { mutableStateOf(Prefs.hfRepo) }
    var modelFile by remember { mutableStateOf(Prefs.hfFile) }
    var modelUrlInput by remember { mutableStateOf(Prefs.modelUrl) }
    ModelManager.hfRepo = modelRepo
    ModelManager.hfFile = modelFile
    ModelManager.modelUrl = modelUrlInput.trim()
    var showModelConfig by remember { mutableStateOf(false) }
    var hfTokenInput by remember { mutableStateOf(Prefs.hfToken) }
    ModelManager.hfToken = hfTokenInput
    var selectedBuiltinId by remember { mutableStateOf(restoredModelIdSafe) }
    var showBuiltinPicker by remember { mutableStateOf(false) }
    var showModelDiag by remember { mutableStateOf(false) }
    var tileMode by remember { mutableStateOf(Prefs.tileMode) }
    var turboMode by remember { mutableStateOf(Prefs.turboMode) }
    // ★ 本地模型列表版本号：导入/删除后自增，驱动列表响应式刷新
    var localModelsVersion by remember { mutableStateOf(0) }
    // ★ 内置模型下载状态版本号：下载/删除后自增，驱动模型页"已下载/未下载"徽标刷新
    var builtinModelsVersion by remember { mutableStateOf(0) }
    // ★ 存储占用检查：各目录占用报告文本（空=尚未检查）
    var storageReport by remember { mutableStateOf("") }
    var storageChecking by remember { mutableStateOf(false) }

    // ---- 设置：ONNX 加速（从 Prefs 恢复 + 设备能力自动识别）----
    var enableNnapi by remember { mutableStateOf(Prefs.enableNnapi) }
    // ★ QNN 自动识别：设备支持 QNN（骁龙 + libcdsprpc）且选中模型匹配当前型号 → 自动开启；
    //   用户手动开关优先（Prefs.enableQnn 非默认时尊重手动值）
    var enableQnn by remember {
        mutableStateOf(
            if (Prefs.enableQnn) true   // 用户手动开过 → 保持
            else ModelManager.shouldAutoEnableQnn(context, restoredModelIdSafe)
        )
    }
    var onnxThreads by remember { mutableStateOf(Prefs.onnxThreads) }
    var keepAlive by remember { mutableStateOf(Prefs.keepAlive) }
    // ★ 自动保存抠图结果开关（默认开；勾选后首次抠图即自动保存 原图→结果 到历史/相册）
    var autoSaveResult by remember { mutableStateOf(Prefs.autoSaveResult) }
    var tileProgress by remember { mutableStateOf(0f) }
    var tileProgressText by remember { mutableStateOf("") }
    var isTiling by remember { mutableStateOf(false) }
    var currentTab by remember { mutableStateOf(0) }
    // ★ 关于页：独立二级页面（设置 Tab 的"关于"卡片点击后进入）
    var showAboutPage by remember { mutableStateOf(false) }
    var historyVersion by remember { mutableStateOf(0) }
    // ★ 历史预加载：produceState 提升到 if 外层，切 Tab 不销毁、不重新加载。
    //   historyVersion 变化时自动重查（HistoryManager 内部有磁盘缓存，命中时直接返回）。
    val rawHistory by produceState<List<HistoryManager.HistoryItem>?>(initialValue = null, historyVersion) {
        value = withContext(Dispatchers.IO) { HistoryManager.getHistory(context) }
    }
    // ★ 历史排序方式（0=最新 1=最旧 2=耗时升 3=耗时降 4=模型名 5=文件最小 6=文件最大），持久化
    var historySort by remember { mutableIntStateOf(Prefs.historySort) }
    // ★ 历史分组折叠：记录已折叠的组（按 originalKey），持久化到 Prefs
    var collapsedHistoryGroups by remember {
        mutableStateOf(if (Prefs.historyCollapsedGroups.isEmpty())
            emptySet() else Prefs.historyCollapsedGroups.split(",").filter { it.isNotEmpty() }.toSet())
    }
    // ★ 历史视图：0=列表 1=网格，持久化
    var historyView by remember { mutableIntStateOf(Prefs.historyView) }
    // ★ 历史多选：选中的记录 id 集合（非空=多选模式）；长按进入多选，批量保存/删除
    var historySelectedIds by remember { mutableStateOf<Set<String>>(emptySet()) }
    // ★ 批量保存中的结果文件（多选保存到相册）
    var historyBatchSaving by remember { mutableStateOf(false) }
    // ★ 待确认删除的整组记录（确认弹窗用）
    var pendingDeleteGroup by remember { mutableStateOf<List<HistoryManager.HistoryItem>?>(null) }
    var showEditor by remember { mutableStateOf(false) }
    // ★ 编辑器目标：true=编辑原图（画笔在原图上抠空/标记，从原图观看/精修）
    //                false=编辑结果图（在抠好的透明结果上画笔恢复/扣空）
    var editorOnOriginal by remember { mutableStateOf(true) }
    // ★ 高级编辑内部子页：false=区域选择，true=后处理/画笔（从区域选择页顶部进入）
    var showBrushFromArea by remember { mutableStateOf(false) }
    // ★ 高级编辑 · 快捷切模型：不跳转模型页，就地弹出模型列表选择，选完立刻重抠当前图
    var showQuickModelPicker by remember { mutableStateOf(false) }
    var quickModelSwitching by remember { mutableStateOf(false) }
    // ★ 快捷切模型=临时切换：记录主模型配置，重抠完成后自动恢复（不改变主模型/不写 Prefs）
    var quickModelRestore by remember { mutableStateOf<QuickModelRestore?>(null) }
    var viewerItem by remember { mutableStateOf<HistoryManager.HistoryItem?>(null) }
    var viewerBitmap by remember { mutableStateOf<Bitmap?>(null) }
    // ★ 后处理编辑页关联的历史记录：进入编辑页时暂存 viewerItem（查看器 Dialog 已关闭），
    //   应用后处理时 updateResult 写回同一条记录（后处理/抠图统一同一条历史记录、同一个原图存档）。
    var postProcessHistoryItem by remember { mutableStateOf<HistoryManager.HistoryItem?>(null) }
    // ★ 结果背景强度事后调整（查看器内实时预览，可逆）：
    //   viewerBgOrigin 缓存查看器打开时的原始结果像素（IntArray），拖动滑杆基于它实时重映射，
    //   不破坏原始，可反复调整；点「应用」才写回 resultBitmap 并入撤销栈。
    var viewerBgOrigin by remember { mutableStateOf<IntArray?>(null) }
    var viewerBgStrength by remember { mutableStateOf(0) }
    // ★ 边缘后处理（查看器内，可逆实时预览；复用 viewerBgOrigin 缓存原始像素）
    var showPostProcess by remember { mutableStateOf(false) }  // 是否展开后处理面板
    var postDecontaminate by remember { mutableStateOf(0f) }   // 去色边强度 0~1
    var postFeather by remember { mutableStateOf(0) }          // 柔化半径 0~8
    var postShrink by remember { mutableStateOf(0) }           // 收缩 px（-20~20，负=扩展）
    // ★ 画笔手动扣空/恢复（后处理面板内，直接改 viewerBitmap 像素，GPU drawPath 高性能）
    var postBrushMode by remember { mutableStateOf(false) }    // 画笔模式开关
    var postBrushEraser by remember { mutableStateOf(true) }   // true=扣空(清alpha) false=恢复(填alpha+原图色)
    var postBrushSize by remember { mutableStateOf(50f) }      // 笔刷大小 px
    var postBrushVersion by remember { mutableStateOf(0) }     // 画笔像素变化计数，触发重组刷新显示
    var batchQueue by remember { mutableStateOf<List<Bitmap>>(emptyList()) }
    var batchIndex by remember { mutableStateOf(0) }
    var showOriginalViewer by remember { mutableStateOf(false) }
    // ★ 查看器拆分方案A：独立全屏"后处理/画笔"编辑页（从查看器拆出的新窗口）
    var showPostProcessScreen by remember { mutableStateOf(false) }
    var postProcessResult by remember { mutableStateOf<Bitmap?>(null) }
    // ★ 4x 超分（独立按钮，不入历史）： viewerBitmap 超分后的临时放大图
    //   仅当前查看会话有效，可保存/分享；不写历史记录、不进撤销栈（用户要求）
    var superResBusy by remember { mutableStateOf(false) }
    var superResLastW by remember { mutableIntStateOf(0) }
    var superResLastH by remember { mutableIntStateOf(0) }
    // ★ 查看器当前显示结果对应的原图（历史记录打开时加载该记录的原图副本，
    //   供后处理编辑页的"恢复画笔"从正确的原图取原背景色，避免用错主界面当前原图）
    var viewerOriginBitmap by remember { mutableStateOf<Bitmap?>(null) }

    // ---- AI 重绘（LocalDream img2img，一键执行）----
    var aiRedrawRequestId by remember { mutableIntStateOf(0) }    // 查看器按钮触发计数（>0 时执行）
    var aiRedrawRunning by remember { mutableStateOf(false) }     // 重绘进行中
    var aiRedrawStatus by remember { mutableStateOf("") }         // 重绘状态/错误信息
    // ★ 引擎空闲超时停止 Job：重绘完成后不立即停（可能有第二次重绘），
    //   延迟 AI_REDRAW_IDLE_STOP_MS 后再停；期间再次重绘会取消旧 Job 重新计时。
    var aiEngineStopJob by remember { mutableStateOf<kotlinx.coroutines.Job?>(null) }
    // ★ 自加载引擎状态（AiRedrawEngine）
    var aiEngineStatus by remember { mutableStateOf("") }         // 引擎状态提示
    var aiEngineBusy by remember { mutableStateOf(false) }        // 部署/启动/停止进行中
    // ★ 模型是否可用（zip 已选 或 已部署）：App 重启从 Prefs 恢复；
    //   部署成功后 zip 自动删除，此时靠 activeModelId 定位部署目录也算已选。
    //   ★ Lite 精简版：AI 重绘固定走 LocalDream App（复用其模型），本 App 无需 SD 模型包 → 恒 true
    var aiModelZipReady by remember {
        mutableStateOf(
            if (BuildConfig.IS_LITE) true
            else runCatching {
                val p = Prefs.aiModelZipPath
                val zipOk = p.isNotEmpty() && File(p).exists() && File(p).length() >= 100_000_000L
                if (zipOk) true
                else {
                    // zip 已删（部署后清理）：判断活动模型目录是否有 unet.bin
                    val savedId = Prefs.aiRedrawModelId
                    savedId.isNotEmpty() && File(File(context.filesDir, "models"), savedId).let { File(it, "unet.bin").exists() }
                }
            }.getOrDefault(false)
        )
    }
    // ★ HF 云下载（AI 重绘模型 zip）状态
    var aiRedrawDownloading by remember { mutableStateOf(false) }        // 下载中
    var aiRedrawDlProgress by remember { mutableStateOf(0L) }            // 已下载字节
    var aiRedrawDlTotal by remember { mutableStateOf(0L) }               // 总字节
    var aiRedrawDlSpeed by remember { mutableStateOf(0L) }               // 速度 B/s
    var aiRedrawDlMirror by remember { mutableStateOf("") }              // 当前镜像
    // ★ 已部署模型列表版本号（部署/切换后自增刷新）
    var deployedModelsVersion by remember { mutableIntStateOf(0) }
    // ★ AI 重绘预设（设置页配置，查看器一键执行）
    var aiRedrawPrompt by remember { mutableStateOf(Prefs.aiRedrawPrompt) }
    var aiRedrawNegative by remember { mutableStateOf(Prefs.aiRedrawNegative) }
    // ★ AI 重绘预设卡片折叠（设置页收纳，默认折叠；占屏大，折叠后设置页更清爽）
    var showAiRedrawPreset by remember { mutableStateOf(false) }
    var aiRedrawSteps by remember { mutableIntStateOf(Prefs.aiRedrawSteps) }
    var aiRedrawCfg by remember { mutableStateOf(Prefs.aiRedrawCfg) }
    var aiRedrawDenoise by remember { mutableStateOf(Prefs.aiRedrawDenoise) }
    var aiRedrawWidth by remember { mutableIntStateOf(Prefs.aiRedrawWidth) }
    var aiRedrawHeight by remember { mutableIntStateOf(Prefs.aiRedrawHeight) }
    var aiRedrawScheduler by remember { mutableStateOf(Prefs.aiRedrawScheduler) }
    // ★ AI 重绘推理后端：local=本地自加载引擎 / dream=已安装的 LocalDream App（复用其 8081）
    var aiRedrawBackend by remember { mutableStateOf(Prefs.aiRedrawBackend) }
    // ★ 高级编辑 · 区域重绘运行中（AreaSelectScreen 内"AI 区域重绘"按钮执行中，禁止重复触发）
    var aiRedrawAreaRunning by remember { mutableStateOf(false) }

    // ---- 撤销/前进（结果位图历史栈）----
    val undoStack = remember { mutableStateListOf<Bitmap>() }   // 已撤销/可重做的状态
    val redoStack = remember { mutableStateListOf<Bitmap>() }   // 撤销出的状态（可前进）
    val MAX_UNDO = 20
    // 抠图/区域编辑产生新结果时压入撤销栈
    fun pushUndo(prev: Bitmap?) {
        // ★ 链底补原图：首次操作（prev==null 且栈空）时用 originalBitmap 作链底，
        //   保证撤销链从原图 → 每步结果逐步推进，能一路撤回原图（之前 prev==null 直接 return 导致链首缺失）。
        if (prev == null) {
            val base = originalBitmap
            if (base == null || base.isRecycled) return
            if (undoStack.isEmpty()) {
                // 复制一份入栈，避免链底引用与当前 originalBitmap 绑死（后续原图可能被替换）
                try { undoStack.add(Bitmap.createBitmap(base)) } catch (_: Exception) { return }
            } else {
                return // 非首次且 prev 为 null：无事可压
            }
        } else {
            undoStack.add(prev)
        }
        if (undoStack.size > MAX_UNDO) {
            // ★ 内存保护（开源做法）：撤销栈弹出最旧状态时立即释放其 native 内存，避免峰值堆积
            try { undoStack.removeAt(0).recycle() } catch (_: Exception) {}
        }
        // ★ 内存保护：新操作使 redo 失效，清空 redo 栈并释放其中位图
        redoStack.forEach { try { it.recycle() } catch (_: Exception) {} }
        redoStack.clear()
    }
    fun doUndo(): Bitmap? {
        if (undoStack.isEmpty()) return null
        val prev = undoStack.removeAt(undoStack.size - 1)
        resultBitmap?.let { redoStack.add(it) }
        if (redoStack.size > MAX_UNDO) {
            // ★ 内存保护：redo 栈溢出时释放最旧
            try { redoStack.removeAt(0).recycle() } catch (_: Exception) {}
        }
        return prev
    }
    fun doRedo(): Bitmap? {
        if (redoStack.isEmpty()) return null
        val next = redoStack.removeAt(redoStack.size - 1)
        resultBitmap?.let { undoStack.add(it) }
        if (undoStack.size > MAX_UNDO) {
            // ★ 内存保护：undo 栈溢出时释放最旧
            try { undoStack.removeAt(0).recycle() } catch (_: Exception) {}
        }
        return next
    }
    // ★ 重置查看器后处理/画笔状态（关闭/切换查看对象时调用）
    fun resetPostVars() {
        showPostProcess = false
        postDecontaminate = 0f; postFeather = 0; postShrink = 0
        postBrushMode = false; postBrushEraser = true
        postBrushVersion++
    }

    // ★★ 4x 超分（独立按钮）：超分当前查看图并替换显示。
    //   不写历史记录、不进撤销栈（用户要求：超分图是临时的，可保存/分享但不入库）。
    //   大图自动降采样防 OOM（SuperResEngine 内部处理），失败静默降级为原图并提示。
    // ★ 禁止无限超分：引擎固定 输入≤512 → 输出≤2048。当前图已是超分产物（尺寸≥2048×2048，
    //   或等于上次超分尺寸）时拒绝再次超分（重复点只会空耗 CPU + 反复降采样损画质）。
    @Suppress("UNUSED_EXPRESSION")
    fun doSuperRes() {
        val cur = viewerBitmap ?: resultBitmap ?: originalBitmap ?: return
        if (cur.isRecycled || superResBusy) return
        // ★ 无限超分闸：当前图已是上次超分结果（尺寸匹配）或已超分上限（≥2048）→ 拒绝
        val alreadySuper = (superResLastW == cur.width && superResLastH == cur.height && superResLastW > 0)
        if (alreadySuper || cur.width >= 2048 || cur.height >= 2048) {
            Toast.makeText(context, "当前图已是最佳分辨率（2048），不再叠加超分", Toast.LENGTH_SHORT).show()
            return
        }
        scope.launch {
            superResBusy = true
            var failMsg = ""
            val up = withContext(Dispatchers.IO) {
                try {
                    val imported = com.rmbg.offline.Prefs.superResModelPath
                    if (imported.isNotEmpty() && java.io.File(imported).exists()) {
                        // ★ 优先用导入的超分模型（QNN EPContext 或普通 ONNX，自动检测）
                        val eng = com.rmbg.offline.ml.SuperResEngine.loadFromFile(context, java.io.File(imported), threads = 4)
                        try { eng.upscale4x(cur) } finally { eng.close() }
                    } else {
                        val model = com.rmbg.offline.ml.SuperResEngine.loadFromAssets(context, "models/realesrgan_anime6b.onnx")
                        if (model == null) {
                            failMsg = "模型加载失败（assets 缺失）"
                            null
                        } else {
                            val eng = com.rmbg.offline.ml.SuperResEngine(threads = 4, modelBytes = model)
                            try { eng.upscale4x(cur) } finally { eng.close() }
                        }
                    }
                } catch (e: Exception) {
                    // ★ 记录具体失败原因（真机排查超分失败用）
                    android.util.Log.e("RMBG-SUPER", "超分失败", e)
                    failMsg = "超分失败：${e.message?.take(80) ?: e.javaClass.simpleName}"
                    null
                }
            }
            superResBusy = false
            if (up != null && !up.isRecycled && up.width > 0) {
                // ★ 不入历史：直接替换查看器显示图；记录当前尺寸用于 UI 提示 + 防重复超分
                superResLastW = up.width; superResLastH = up.height
                // 旧 viewerBitmap 若非历史加载，回收旧图避免泄漏（历史图不回收，它由历史栈持有）
                val old = viewerBitmap
                if (old != null && old !== cur && old !== up && old.isRecycled.not() && viewerItem == null) {
                    try { old.recycle() } catch (_: Exception) {}
                }
                viewerBitmap = up
                statusText = "已 4x 超分：${up.width}×${up.height}（临时图，不写入历史，可保存/分享）"
                Toast.makeText(context, "已 4x 超分 ${up.width}×${up.height}", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(context, if (failMsg.isNotEmpty()) failMsg else "超分失败（模型不可用或内存不足），已保留原图", Toast.LENGTH_LONG).show()
            }
        }
    }

    // ---- 停止标志（运行中可取消）----
    var stopRequested by remember { mutableStateOf(false) }

    // ---- 对比按住（按住显示原图，松开恢复结果）----
    var comparePressed by remember { mutableStateOf(false) }

    // ---- 查看器背景色选项 ----
    val viewerBgChoices = remember { listOf(
        "棋盘", "白色", "黑色", "浅灰", "绿色"
    ) }
    var viewerBgColor by remember { mutableStateOf(Prefs.viewerBgColor) }

    // ---- 通知权限申请 ----
    val notifPermLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            Toast.makeText(context, "通知权限已开启", Toast.LENGTH_SHORT).show()
        } else {
            Toast.makeText(context, "未授予通知权限，抠图完成将不提醒", Toast.LENGTH_LONG).show()
        }
    }

    // ★ QNN 自动加载（双轨）：内置 QNN 模型的 onnx+bin 打包在 APK assets 内（离线兜底）；
    //   选中任一 QNN 模型时先尝试 assets 部署，assets 缺失/损坏则自动走 HF 仓库下载（下载型）。
    //   ensureQnnContextDual 内部：assets 优先 → 不足则 HF 下载 onnx+bin。
    LaunchedEffect(selectedBuiltinId) {
        val selBm = ModelManager.builtinModels.find { it.id == selectedBuiltinId }
        if (selBm?.isQnn == true) {
            val deployed = ModelManager.ensureQnnContextDual(context, selBm.id)
            modelReady = deployed && ModelManager.isModelDownloaded()
            statusText = if (modelReady) "模型已就绪，选择图片开始抠图" else "${selBm.name} 部署失败，可点重新部署"
        }
    }

    LaunchedEffect(Unit) {
        if (sharedUris.isNotEmpty()) {
            val bmps = sharedUris.mapNotNull { uri ->
                try { decodeBitmap(context, uri) } catch (_: Exception) { null }
            }
            if (bmps.isNotEmpty()) {
                originalBitmap = bmps.first()
                resultBitmap = null
                // ★ 新图开新历史分支
                currentHistoryKey = computeOriginalKey(bmps.first())
                batchQueue = bmps.drop(1)
                batchIndex = 0
                ProjectStateManager.save(context, bmps.first(), null)
                statusText = if (bmps.size > 1)
                    "已导入 ${bmps.size} 张图片，点击立即抠图批量处理"
                else "已导入分享图片，点击抠图开始处理"
                Toast.makeText(context, if (bmps.size > 1) "已导入 ${bmps.size} 张图片" else "已导入分享图片", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(context, "导入分享图片失败", Toast.LENGTH_LONG).show()
            }
        }
        if (Prefs.keepAlive) {
            try { KeepAliveService.start(context) } catch (_: Exception) {}
        }
        if (!Notifications.hasPermission(context)) {
            try { notifPermLauncher.launch(android.Manifest.permission.POST_NOTIFICATIONS) } catch (_: Exception) {}
        }
        // ★ AI 重绘模型 zip 恢复 + 引擎日志恢复 + 自动补全部署
        //   ★ Lite 精简版：AI 重绘固定走 LocalDream App（复用其模型），本 App 无本地引擎/模型包，整段跳过
        if (!BuildConfig.IS_LITE) {
        AiRedrawEngine.restoreModelZip(context)
        // ★ 引擎日志恢复（App 重启后日志 UI 仍显示上次引擎日志）
        AiRedrawEngine.restoreLogTail(context)
        if (AiRedrawEngine.isModelReady(context)) {
            aiModelZipReady = true
        }
        // ★ 启动自动补全部署：扫描 ai_models/ 下未部署的 AI 重绘 zip，
        //   逐个自动解压部署（不切换当前活动模型），让"已部署模型"列表一次性显示全部模型。
        //   （部署成功自动删 zip 源包；本机已有 AnythingV5/MeinaMix 两个包时，第二个也能自动入列）
        scope.launch {
            val pendingZips = withContext(Dispatchers.IO) {
                runCatching {
                    val dir = File(context.filesDir, "ai_models")
                    if (!dir.exists()) emptyList() else dir.listFiles { f ->
                        f.isFile && f.name.endsWith(".zip", ignoreCase = true) && f.length() >= 100_000_000L
                    }?.filter { zip ->
                        // ★ 已部署判定：对应部署目录已有 unet.bin（指纹匹配不强制，zip 还在=允许重部署）
                        val dir2 = AiRedrawEngine.modelDirFor(context, zip)
                        !File(dir2, "unet.bin").exists()
                    }?.sortedByDescending { it.lastModified() } ?: emptyList()
                }.getOrDefault(emptyList())
            }
            for (zip in pendingZips) {
                // ★ 逐个自动部署（activate=false：不改变当前活动模型，仅让部署目录入列表）
                val ok = AiRedrawEngine.deployModel(context, zip, activate = false)
                if (ok) {
                    deployedModelsVersion++ // ★ 刷新已部署模型列表
                    aiEngineStatus = "✅ 已自动部署: ${zip.nameWithoutExtension}"
                } else {
                    aiEngineStatus = "❌ 自动部署失败 ${zip.name}: ${AiRedrawEngine.status}"
                    break
                }
            }
            // ★ 补全部署后统一恢复活动模型定位（zip 已被删，modelZipFile 清空，靠 activeModelId 定位）
            AiRedrawEngine.restoreModelZip(context)
            if (AiRedrawEngine.isModelReady(context)) aiModelZipReady = true
        }
        }
        // ★ 历史预加载：App 启动时后台预查历史列表 + 预热前 20 条缩略图缓存。
        //   produceState 在外层已声明，此处仅预热 LruCache，切到 Tab 2 时缩略图直接命中缓存、零解码。
        scope.launch(Dispatchers.IO) {
            val items = HistoryManager.getHistory(context)
            items.take(20).forEach { item ->
                HistoryManager.loadThumb(context, item)
            }
        }
    }

    // ---- 引擎（懒加载，绑定全局引用供 onStop 释放）----
    var engine by remember { mutableStateOf(RmbgScreenState.engine) }
    fun getEngine(): com.rmbg.offline.ml.RmbgOnnxEngine {
        val target = ModelManager.modelFile
        val cached = engine
        // ★ 重建条件：模型路径变化 或 CPU 线程数变化（线程数改变需重建 session 才生效）
        if (cached != null && cached.isLoaded &&
            cached.modelPath == target.absolutePath &&
            cached.threadCount == onnxThreads) {
            return cached
        }
        try { cached?.close() } catch (_: Exception) {}
        val selBm = ModelManager.builtinModels.find { it.id == selectedBuiltinId }
        // ★ QNN 自动加载：与主界面 LaunchedEffect 一致，统一走【双轨】ensureQnnContextDual
        //   （assets 兜底优先，缺失/损坏时自动走 HF 仓库下载，兼容 zip 交付的多-新模型）。
        //   此前用旧单轨 ensureQnnContext（仅 assets），对 zip/HF 交付模型（无 assetOnnx/assetBin）
        //   返回 false → load 时 QNN 不可用抛异常 → 区域抠图报"模型未加载"，此处已对齐修复。
        if (selBm?.isQnn == true) {
            runBlocking { withContext(Dispatchers.IO) { ModelManager.ensureQnnContextDual(context, selBm.id) } }
        }
        val e = com.rmbg.offline.ml.RmbgOnnxEngine(
            target,
            enableNnapi = enableNnapi,
            enableQnn = enableQnn,
            threads = onnxThreads,
            nativeLibDir = try {
                context.applicationInfo.nativeLibraryDir
            } catch (_: Exception) { null }
        ).also {
            it.bgCleanAlpha = Prefs.bgCleanAlpha // ★ 背景硬化阈值：持久化可调
            it.load()
            engine = it
            RmbgScreenState.engine = it
        }
        return e
    }

    // ---- 选图（支持多选批量）----
    val imagePicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetMultipleContents()
    ) { uris: List<Uri> ->
        if (uris.isEmpty()) return@rememberLauncherForActivityResult
        val bmps = uris.mapNotNull { uri ->
            try { decodeBitmap(context, uri) } catch (_: Exception) { null }
        }
        if (bmps.isEmpty()) {
            Toast.makeText(context, "没有可用的图片", Toast.LENGTH_SHORT).show()
            return@rememberLauncherForActivityResult
        }
        originalBitmap = bmps.first()
        resultBitmap = null
        // ★ 新图开新历史分支
        currentHistoryKey = computeOriginalKey(bmps.first())
        batchQueue = bmps.drop(1)
        batchIndex = 0
        ProjectStateManager.save(context, bmps.first(), null)
        statusText = if (bmps.size > 1) "已选 ${bmps.size} 张图片，点击立即抠图逐张处理" else "已选择图片，点击立即抠图"
    }

    // ---- 保存 ----
    val saveLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.CreateDocument("image/png")
    ) { uri: Uri? ->
        val bmp = resultBitmap ?: return@rememberLauncherForActivityResult
        uri ?: return@rememberLauncherForActivityResult
        try {
            context.contentResolver.openOutputStream(uri)?.use { os ->
                bmp.compress(Bitmap.CompressFormat.PNG, 100, os)
            }
            Toast.makeText(context, "已保存", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Toast.makeText(context, "保存失败: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    // ---- 查看器另存为（SAF 系统文档保存，可自定义文件名/位置）----
    val viewerSaveLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.CreateDocument("image/png")
    ) { uri: Uri? ->
        val bmp = viewerBitmap ?: return@rememberLauncherForActivityResult
        uri ?: return@rememberLauncherForActivityResult
        try {
            context.contentResolver.openOutputStream(uri)?.use { os ->
                bmp.compress(Bitmap.CompressFormat.PNG, 100, os)
            }
            Toast.makeText(context, "已另存为", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Toast.makeText(context, "另存失败: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    // ---- 本地模型导入（SAF 选 .onnx）----
    // ★ GetContent() 而非 OpenDocument()：前者是"分享式 Picker"，侧边栏会列出第三方文件管理器
    //   （MT 管理器等实现了 ACTION_GET_CONTENT 接收方），方便在 SAF 里进入 MT 选模型包。
    val importModelLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri ?: return@rememberLauncherForActivityResult
        val displayName = try {
            context.contentResolver.query(uri, arrayOf(
                android.provider.OpenableColumns.DISPLAY_NAME
            ), null, null, null)?.use { c ->
                if (c.moveToFirst()) c.getString(0) else "model"
            } ?: "model"
        } catch (_: Exception) { "model" }
        val baseName = displayName.removeSuffix(".onnx").removeSuffix(".ONNX")
        val id = try {
            context.contentResolver.openInputStream(uri)?.use { input ->
                val tmp = File(context.cacheDir, "import_$baseName.onnx")
                tmp.outputStream().use { out -> input.copyTo(out) }
                ModelManager.importLocalModel(tmp, baseName)?.also {
                    tmp.delete()
                    localModelsVersion++ // ★ 触发本地模型列表刷新
                    selectedBuiltinId = it
                    ModelManager.selectedModelId = it
                    Prefs.selectedModelId = it
                    val lm = ModelManager.LocalModel(it, baseName, "model_$it.onnx", ModelManager.modelDir().let { d -> File(d, "model_$it.onnx").length() })
                    scope.launch {
                        val compat = withContext(Dispatchers.IO) {
                            runCatching { ModelManager.probeLocalModelCompat(lm, deep = true) }
                                .getOrElse { ModelManager.ModelCompatibility(ModelManager.AccelSupport.CPU_ONLY, "", "", "") }
                        }
                        val qnnOk = compat.support == ModelManager.AccelSupport.QNN_EPCONTEXT ||
                            compat.support == ModelManager.AccelSupport.QNN_TRY
                        enableQnn = qnnOk
                        Prefs.enableQnn = qnnOk
                        try { engine?.close() } catch (_: Exception) {}
                        engine = null
                        modelReady = ModelManager.isModelDownloaded()
                        statusText = "已导入模型: $baseName · ${compat.qnnLabel} · ${compat.nnapiLabel} · ${compat.desc}"
                    }
                }
            }
        } catch (e: Exception) {
            android.util.Log.e("RMBG-IMPORT", "导入失败", e)
            null
        }
        if (id == null) {
            Toast.makeText(context, "导入失败，请确认是有效的 .onnx 模型", Toast.LENGTH_SHORT).show()
        } else {
            Toast.makeText(context, "本地模型已导入并选中", Toast.LENGTH_SHORT).show()
        }
    }

    // ---- 本地模型 zip 导入（SAF 选 .zip：onnx + bin 一起打包）----
    // ★ GetContent() 同样让侧边栏列出第三方文件管理器（MT），能选中系统无法识别 MIME 的 zip
    val importZipLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri ?: return@rememberLauncherForActivityResult
        val displayName = try {
            context.contentResolver.query(uri, arrayOf(
                android.provider.OpenableColumns.DISPLAY_NAME
            ), null, null, null)?.use { c ->
                if (c.moveToFirst()) c.getString(0) else "model.zip"
            } ?: "model.zip"
        } catch (_: Exception) { "model.zip" }
        val baseName = displayName.removeSuffix(".zip").removeSuffix(".ZIP")
            .removeSuffix(".onnx").removeSuffix(".ONNX").ifBlank { "model" }
        // ★ 整个 zip 导入（复制 554MB + 解压 1GB+）放 IO 线程，避免主线程 ANR/卡死
        Toast.makeText(context, "正在导入 zip 模型（大文件需数秒~数十秒）...", Toast.LENGTH_SHORT).show()
        scope.launch {
            val imported = withContext(Dispatchers.IO) {
                try {
                    context.contentResolver.openInputStream(uri)?.use { input ->
                        val tmp = File(context.cacheDir, "import_${System.currentTimeMillis()}.zip")
                        tmp.outputStream().use { out -> input.copyTo(out) }
                        ModelManager.importLocalModelZip(tmp, baseName)?.also {
                            tmp.delete()
                        }
                    }
                } catch (e: Exception) {
                    android.util.Log.e("RMBG-IMPORT", "zip 导入失败", e)
                    null
                }
            }
            val id = imported
            if (id == null) {
                Toast.makeText(context, "zip 导入失败，请确认包含 .onnx 与配套 .bin", Toast.LENGTH_SHORT).show()
            } else {
                localModelsVersion++ // ★ 触发本地模型列表刷新
                selectedBuiltinId = id
                ModelManager.selectedModelId = id
                Prefs.selectedModelId = id
                val lm = ModelManager.LocalModel(id, baseName, "model_$id.onnx", ModelManager.modelDir().let { d -> File(d, "model_$id.onnx").length() })
                scope.launch {
                    // ★ 模型可用性校验：深扫描 onnx（EPContext/bin/算子）+ 确认文件有效
                    val validation = withContext(Dispatchers.IO) {
                        val file = File(ModelManager.modelDir(), lm.fileName)
                        if (!file.exists() || file.length() == 0L) {
                            "❌ 模型文件无效或为空"
                        } else {
                            val compat = runCatching { ModelManager.probeLocalModelCompat(lm, deep = true) }
                                .getOrElse { ModelManager.ModelCompatibility(ModelManager.AccelSupport.CPU_ONLY, "", "", "") }
                            // QNN EPContext 需确认 bin 就位
                            if (compat.support == ModelManager.AccelSupport.QNN_EPCONTEXT) {
                                val binRef = ModelManager.extractEpCacheContext(file)
                                val binOk = binRef != null && File(ModelManager.modelDir(), binRef).exists()
                                if (binOk) {
                                    "✅ 可用（QNN HTP DSP）· ${compat.desc}"
                                } else {
                                    "⚠️ 已识别为 QNN 模型，但配套 .bin 缺失，将回退 CPU"
                                }
                            } else {
                                "可用 · ${compat.qnnLabel} · ${compat.nnapiLabel} · ${compat.desc}"
                            }
                        }
                    }
                    val qnnOk = validation.startsWith("✅ 可用（QNN")
                    enableQnn = qnnOk
                    Prefs.enableQnn = qnnOk
                    try { engine?.close() } catch (_: Exception) {}
                    engine = null
                    modelReady = ModelManager.isModelDownloaded()
                    statusText = "已导入模型: $baseName\n$validation"
                }
                Toast.makeText(context, "zip 模型已导入并选中", Toast.LENGTH_SHORT).show()
            }
        }
    }

    // ---- 超分模型导入（SAF 选 .zip 或 .onnx）----
    // ★ 与抠图模型隔离：走 importSuperResModelZip 存到 modelDir()/superres/ 子目录，
    //   绝不进入抠图 localModels() 列表。CPU 超分可导入单 .onnx；QNN 超分导入 onnx+bin zip。
    val importSuperResLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri ?: return@rememberLauncherForActivityResult
        val displayName = try {
            context.contentResolver.query(uri, arrayOf(
                android.provider.OpenableColumns.DISPLAY_NAME
            ), null, null, null)?.use { c ->
                if (c.moveToFirst()) c.getString(0) else "super_res.zip"
            } ?: "super_res.zip"
        } catch (_: Exception) { "super_res.zip" }
        Toast.makeText(context, "正在导入超分模型（onnx+bin，大文件需数秒~数十秒）...", Toast.LENGTH_SHORT).show()
        scope.launch {
            val result = withContext(Dispatchers.IO) {
                try {
                    context.contentResolver.openInputStream(uri)?.use { input ->
                        val tmp = File(context.cacheDir, "sr_import_${System.currentTimeMillis()}.${if (displayName.endsWith(".zip", true)) "zip" else "onnx"}")
                        tmp.outputStream().use { out -> input.copyTo(out) }
                        if (!tmp.exists() || tmp.length() == 0L) { tmp.delete(); null } else {
                            if (displayName.endsWith(".zip", true)) {
                                val path = ModelManager.importSuperResModelZip(tmp)
                                tmp.delete()
                                path
                            } else {
                                val srDir = File(ModelManager.modelDir(), "superres").apply { mkdirs() }
                                val dest = File(srDir, "superres_model.onnx")
                                tmp.copyTo(dest, overwrite = true)
                                tmp.delete()
                                if (dest.exists()) dest.absolutePath else null
                            }
                        }
                    }
                } catch (e: Exception) {
                    android.util.Log.e("RMBG-IMPORT", "超分模型导入失败", e)
                    null
                }
            }
            if (result != null && File(result).exists()) {
                Prefs.superResModelPath = result
                Toast.makeText(context, "超分模型已导入（${File(result).name}）", Toast.LENGTH_LONG).show()
            } else {
                Toast.makeText(context, "超分模型导入失败：请选择包含 onnx(+bin) 的 zip 或单个 onnx", Toast.LENGTH_LONG).show()
            }
        }
    }

    // ---- AI 重绘模型 zip 选择（AnythingV5_qnn2.28_8gen2.zip）----
    // ★ GetContent() 而非 OpenDocument()：侧边栏能列出并进入第三方文件管理器（MT）
    //   launch 传 "*/*" 显示所有文件、不按类型过滤（可在 MT 里选中系统识别不了 MIME 的 zip）
    val aiModelZipPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri ?: return@rememberLauncherForActivityResult
        // ★ 复制到 filesDir/ai_models/ 持久目录（App 重启不丢，不再用 cacheDir 临时目录）
        Toast.makeText(context, "正在复制模型包（约 1GB，请稍候）...", Toast.LENGTH_SHORT).show()
        scope.launch {
            val copied = withContext(Dispatchers.IO) {
                try {
                    val dir = File(context.filesDir, "ai_models")
                    if (!dir.exists()) dir.mkdirs()
                    // ★ 文件名取 SaF 原始文件名（真实模型 zip 名）→ modelId 即真实模型名，
                    //   多模型共存不串目录。旧版本曾用时间戳命名导致 modelId 混乱，这里彻底修正。
                    val displayName = try {
                        context.contentResolver.query(
                            uri, arrayOf(android.provider.OpenableColumns.DISPLAY_NAME),
                            null, null, null
                        )?.use { c -> if (c.moveToFirst()) c.getString(0) else null } ?: null
                    } catch (_: Exception) { null }
                    val rawName = displayName?.let { it.substringAfterLast('/') } ?: "model.zip"
                    var safeName = rawName.replace(Regex("[^A-Za-z0-9._-]"), "_")
                    if (!safeName.endsWith(".zip", ignoreCase = true)) safeName += ".zip"
                    val name = safeName.ifBlank { "model.zip" }
                    val tmp = File(dir, name)
                    // ★ 同名已存在（重复选同一包）：覆盖写入，避免多份残留
                    if (tmp.exists()) tmp.delete()
                    context.contentResolver.openInputStream(uri)?.use { input ->
                        tmp.outputStream().use { out -> input.copyTo(out) }
                    }
                    tmp
                } catch (e: Exception) {
                    android.util.Log.e("AI-REDRAW", "复制模型包失败", e)
                    null
                }
            }
            if (copied == null || copied.length() < 100_000_000L) {
                aiEngineStatus = "❌ 模型包复制失败或文件过小"
                copied?.delete()
            } else {
                // ★ 智能识别：unet.bin → AI 重绘 SD 包（立即自动部署激活）；.onnx → 抠图 QNN 包（走本地导入）
                val zipKind = withContext(Dispatchers.IO) { probeZipKind(copied) }
                if (zipKind == ZIP_AI_REDRAW) {
                    // ★ 导入即部署：复制完成立即解压部署（约 1GB，1~3 分钟），
                    //   部署成功后 zip 自动删除、目录入"已部署模型"列表，无需等查看器
                    aiEngineStatus = "正在自动部署 AI 重绘模型（约 1GB，请稍候）..."
                    aiEngineBusy = true
                    val deployed = AiRedrawEngine.deployModel(context, copied, activate = true)
                    aiEngineBusy = false
                    if (deployed) {
                        aiModelZipReady = true
                        deployedModelsVersion++ // ★ 刷新已部署模型列表
                        aiEngineStatus = "✅ ${copied.nameWithoutExtension} 已部署激活，可立即使用"
                        Notifications.notifyDownloadDone(context, "AI 重绘模型已部署", "${copied.nameWithoutExtension} 已就绪，点击进入模型页")
                        Toast.makeText(context, "模型已部署激活", Toast.LENGTH_SHORT).show()
                    } else {
                        aiEngineStatus = "❌ 模型部署失败: ${AiRedrawEngine.status}"
                        Toast.makeText(context, "模型部署失败", Toast.LENGTH_SHORT).show()
                    }
                } else {
                    // 非 AI 重绘包（如抠图 QNN 包）：仅记录选择，保持原行为
                    AiRedrawEngine.selectModel(copied)  // ★ 记录活动模型 id（部署后删 zip 仍能定位）
                    Prefs.aiModelZipPath = copied.absolutePath // ★ 持久化，重启不丢
                    aiModelZipReady = true
                    aiEngineStatus = "✅ 模型包已选择（${formatSize(copied.length())}），在查看器点「AI 重绘」即自动部署启动"
                }
            }
        }
    }

    // ---- 下载模型 ----
    fun startDownload() {
        if (isDownloading) return
        val selBm = ModelManager.builtinModels.find { it.id == selectedBuiltinId }
        if (selBm?.isQnn == true) {
            val qnnId = selBm.id
            // ★ 不再做主线程 quickProbe 预检（Android 主线程同步网络会抛 NetworkOnMainThreadException
            //   → 被 quickProbe 吞掉返回 false → 误报"网络不可达"）。
            //   连通性检查交给 downloadModel/ensureQnnContextDual 内部的 IO 线程镜像循环处理，
            //   失败会自动切换镜像并重试，错误原因通过 listener 实时回传 UI。
            // ★ 重置取消标志 + 保存工作协程
            ModelManager.resetCancel()
            downloadingJob = scope.launch {
                isDownloading = true
                statusText = "正在部署 ${selBm.name}（内置优先，不足时联网下载）..."
                // 双轨：assets 内置兜底 → 缺失则走 HF 仓库下载 onnx+bin
                val ok = ModelManager.ensureQnnContextDual(
                    context, qnnId,
                    listener = object : ModelManager.ProgressListener {
                        override fun onProgress(bytesDownloaded: Long, totalBytes: Long, speedBps: Long) {
                            statusText = "下载中 ${selBm.name}：${bytesDownloaded / 1024 / 1024}/${totalBytes / 1024 / 1024}MB"
                        }

                        override fun onMirrorSwitch(mirrorIndex: Int, mirrorName: String) {
                            statusText = "正在从 $mirrorName 下载 ${selBm.name} ..."
                        }

                        override fun onMirrorError(mirrorName: String, error: String) {
                            statusText = "$mirrorName 下载失败：$error"
                        }

                        override fun onDone(file: File) {}

                        override fun onError(e: Exception) {
                            statusText = "下载错误：${e.message ?: e.javaClass.simpleName}"
                        }
                    }
                )
                downloadingJob = null
                isDownloading = false
                if (ok) {
                    modelReady = ModelManager.isModelDownloaded()
                    statusText = "${selBm.name} 已就绪，选择图片开始抠图"
                    Toast.makeText(context, "${selBm.name} 部署完成（${selBm.sizeBytes / 1024 / 1024}MB）", Toast.LENGTH_SHORT).show()
                } else {
                    statusText = if (ModelManager.isCancelRequested()) "下载已取消"
                        else "${selBm.name} 部署失败（详见日志或设置页诊断）"
                    Toast.makeText(context, if (ModelManager.isCancelRequested()) "下载已取消" else "${selBm.name} 部署失败，请查看状态文字", Toast.LENGTH_SHORT).show()
                }
            }
            return
        }
        ModelManager.selectedMirrorIndex = selectedMirrorIndex
        ModelManager.hfToken = hfTokenInput.trim()
        ModelManager.builtinModels.find { it.id == selectedBuiltinId }?.let { bm ->
            ModelManager.hfRepo = bm.hfRepo
            ModelManager.hfFile = bm.hfFile
            modelRepo = bm.hfRepo
            modelFile = bm.hfFile
        }
        ModelManager.hfRepo = modelRepo.trim().ifEmpty { "briaai/RMBG-1.4" }
        ModelManager.hfFile = modelFile.trim().ifEmpty { "onnx/model.onnx" }
        // ★ 不再做主线程 quickProbe 预检（原因同上），交给 downloadModel 内部 IO 线程镜像循环
        // ★ 重置取消标志 + 保存工作协程
        ModelManager.resetCancel()
        downloadingJob = scope.launch {
            isDownloading = true
            downloadProgress = 0f
            downloadSpeed = 0L
            downloadBytes = 0L
            downloadTotal = 0L
            statusText = "正在下载模型..."
            activeMirrorName = null
            ModelManager.downloadModel(
                context = context,
                listener = object : ModelManager.ProgressListener {
                    override fun onProgress(bytesDownloaded: Long, totalBytes: Long, speedBps: Long) {
                        downloadProgress = if (totalBytes > 0) bytesDownloaded.toFloat() / totalBytes else 0f
                        downloadSpeed = speedBps
                        downloadBytes = bytesDownloaded
                        downloadTotal = totalBytes
                        statusText = "下载中..."
                    }
                    override fun onMirrorSwitch(mirrorIndex: Int, mirrorName: String) {
                        activeMirrorName = mirrorName
                        statusText = "正在从 $mirrorName 下载..."
                    }
                    override fun onMirrorError(mirrorName: String, error: String) {
                        android.util.Log.e("RMBG-DL", "$mirrorName 失败: $error")
                        statusText = "$mirrorName 失败: $error"
                    }
                    override fun onDone(file: File) {
                        downloadingJob = null
                        modelReady = true
                        isDownloading = false
                        statusText = "模型下载完成！选择图片开始抠图"
                        Toast.makeText(context, "模型已就绪", Toast.LENGTH_SHORT).show()
                    }
                    override fun onError(e: Exception) {
                        downloadingJob = null
                        isDownloading = false
                        activeMirrorName = null
                        statusText = if (ModelManager.isCancelRequested()) "下载已取消" else "下载失败: ${e.message}"
                        if (ModelManager.isCancelRequested())
                            Toast.makeText(context, "下载已取消", Toast.LENGTH_SHORT).show()
                        else
                            Toast.makeText(context, "下载失败: ${e.message}", Toast.LENGTH_SHORT).show()
                    }
                }
            )
        }
    }

    // ---- 取消下载 ----
    fun cancelDownload() {
        if (!isDownloading) return
        ModelManager.requestCancelDownload()
        // 协程取消：配合 ModelManager 的 cancelRequested 标志，分块读循环会及时退出
        downloadingJob?.cancel()
        downloadingJob = null
        isDownloading = false
        statusText = "下载已取消"
        Toast.makeText(context, "下载已取消", Toast.LENGTH_SHORT).show()
    }

    // ---- 从自定义直链下载模型（模型配置·下载链接入口）----
    // ★ 智能识别：URL 文件名 .zip → AI 重绘 SD 模型；.onnx → 抠图模型。
    //   token（hfTokenInput）两种链路都生效。
    fun downloadModelUrl() {
        if (isDownloading || aiRedrawDownloading) return
        val url = modelUrlInput.trim()
        if (url.isBlank()) {
            Toast.makeText(context, "请先粘贴模型下载链接", Toast.LENGTH_SHORT).show()
            return
        }
        if (!url.startsWith("http://") && !url.startsWith("https://")) {
            Toast.makeText(context, "请输入完整 http(s) 链接", Toast.LENGTH_SHORT).show()
            return
        }
        // 持久化链接 + Token
        Prefs.modelUrl = url
        ModelManager.modelUrl = url
        ModelManager.hfToken = hfTokenInput.trim()
        Prefs.hfToken = hfTokenInput.trim()
        // ★ 识别类型：URL 末段文件名（去 query）以 .zip 结尾 → 模型包（内容探测分流）；.onnx → 抠图模型
        val cleanName = url.substringBefore('?').substringAfterLast('/').lowercase()
        val isAiRedrawZip = cleanName.endsWith(".zip") || cleanName.endsWith(".safetensors")
        if (isAiRedrawZip) {
            // ---- 模型包 zip（可能是 AI 重绘 SD 包 unet.bin / 抠图 QNN 包 onnx+bin，下载后按内容分流）----
            scope.launch {
                aiRedrawDownloading = true
                aiRedrawDlProgress = 0L; aiRedrawDlTotal = 0L; aiRedrawDlSpeed = 0L; aiRedrawDlMirror = ""
                aiEngineStatus = "⬇ 从链接下载模型包…"
                // ★ 系统通知栏：开始下载
                Notifications.notifyDownloadProgress(context, "正在下载模型包…", 0L, 0L, 0L)
                val dir = File(context.filesDir, "ai_models")
                if (!dir.exists()) dir.mkdirs()
                // ★ 保存文件名取 URL 末段（真实模型 zip 名 → modelId 即真实模型名，多模型共存不串目录）
                val fileName = url.substringAfterLast('/').substringBefore('?').ifBlank { "model_${System.currentTimeMillis()}.zip" }
                val safeName = fileName.replace(Regex("[^A-Za-z0-9._-]"), "_")
                val destFile = File(dir, safeName)
                val ok = ModelManager.downloadAiRedrawZipFromUrl(
                    context = context,
                    url = url,
                    destFile = destFile,
                    token = hfTokenInput.trim(),
                    listener = object : ModelManager.ProgressListener {
                        override fun onProgress(d: Long, t: Long, s: Long) {
                            aiRedrawDlProgress = d; aiRedrawDlTotal = t; aiRedrawDlSpeed = s
                            // ★ 系统通知栏：进度更新（防止通知过于频繁，进度条 % 相同则跳过）
                            Notifications.notifyDownloadProgress(context, "正在下载模型包…", d, t, s)
                        }
                        override fun onMirrorSwitch(i: Int, name: String) { aiRedrawDlMirror = name }
                        override fun onMirrorError(name: String, err: String) { aiRedrawDlMirror = "$name: $err" }
                        override fun onDone(f: java.io.File) {}
                        override fun onError(e: Exception) {}
                    }
                )
                aiRedrawDownloading = false
                Notifications.cancelDownloadNotification(context)
                if (ok && destFile.exists() && destFile.length() >= 100_000_000L) {
                    // ★ 内容探测分流：unet.bin → AI 重绘 SD 包；.onnx(+.bin) → 抠图 QNN 包
                    val zipKind = withContext(Dispatchers.IO) { probeZipKind(destFile) }
                    when (zipKind) {
                        ZIP_AI_REDRAW -> {
                            // ★ 下载即部署：下载完成立即自动解压部署激活（约 1GB，1~3 分钟），
                            //   部署成功后 zip 自动删除、目录入"已部署模型"列表，无需再等查看器
                            aiEngineStatus = "正在自动部署 AI 重绘模型（约 1GB，请稍候）..."
                            aiEngineBusy = true
                            val deployed = AiRedrawEngine.deployModel(context, destFile, activate = true)
                            aiEngineBusy = false
                            if (deployed) {
                                aiModelZipReady = true
                                deployedModelsVersion++ // ★ 刷新已部署模型列表
                                aiEngineStatus = "✅ ${destFile.nameWithoutExtension} 已部署激活，可立即使用"
                                Notifications.notifyDownloadDone(context, "AI 重绘模型已部署", "${destFile.nameWithoutExtension} 已就绪，点击进入模型页")
                                Toast.makeText(context, "AI 重绘模型已部署", Toast.LENGTH_SHORT).show()
                            } else {
                                aiEngineStatus = "❌ AI 重绘模型部署失败: ${AiRedrawEngine.status}"
                                Notifications.notifyDownloadDone(context, "AI 重绘模型部署失败", AiRedrawEngine.status)
                                Toast.makeText(context, "模型部署失败", Toast.LENGTH_SHORT).show()
                            }
                        }
                        ZIP_MATTING -> {
                            // 抠图 QNN 包（onnx + 配套 bin）：走本地导入链路
                            val base = destFile.nameWithoutExtension.ifBlank { "custom" }
                            val importedId = withContext(Dispatchers.IO) {
                                ModelManager.importLocalModelZip(destFile, base)
                            }
                            if (importedId != null) {
                                // 切换到导入的本地模型（CPU 推理）
                                enableQnn = false
                                selectedBuiltinId = importedId
                                ModelManager.selectedModelId = importedId
                                Prefs.selectedModelId = importedId
                                modelReady = ModelManager.isModelDownloaded()
                                aiModelZipReady = false
                                destFile.delete() // 已导入为本地模型，删 zip 源包
                                aiEngineStatus = "✅ 抠图模型已导入并切换：$base（选择图片开始抠图）"
                                Notifications.notifyDownloadDone(context, "抠图模型已导入", "$base 已就绪，点击开始抠图")
                                Toast.makeText(context, "抠图模型已导入", Toast.LENGTH_SHORT).show()
                            } else {
                                aiEngineStatus = "❌ zip 内有 onnx 但导入失败（可能缺配套 bin）"
                                Notifications.notifyDownloadDone(context, "模型导入失败", "zip 内可能缺少配套 .bin 文件")
                                Toast.makeText(context, "模型导入失败", Toast.LENGTH_SHORT).show()
                            }
                        }
                        else -> {
                            aiEngineStatus = "❌ 无法识别压缩包内容（既无 unet.bin 也无 .onnx）"
                            Notifications.notifyDownloadDone(context, "模型包无法识别", "请确认链接指向有效模型包")
                            Toast.makeText(context, "模型包无法识别", Toast.LENGTH_SHORT).show()
                        }
                    }
                } else {
                    aiEngineStatus = "❌ 下载失败，请检查链接/Token/网络"
                    Notifications.notifyDownloadDone(context, "模型下载失败", "请检查链接/Token/网络后重试")
                    Toast.makeText(context, "下载失败", Toast.LENGTH_SHORT).show()
                }
            }
        } else {
            // ---- 抠图模型 .onnx（原链路：model_custom.onnx，CPU 推理）----
            scope.launch {
                isDownloading = true
                downloadProgress = 0f
                downloadSpeed = 0L
                downloadBytes = 0L
                downloadTotal = 0L
                statusText = "正在从链接下载模型..."
                activeMirrorName = null
                // ★ 系统通知栏：开始下载
                Notifications.notifyDownloadProgress(context, "正在下载抠图模型…", 0L, 0L, 0L)
                // 自动配置调用链路：完成后 selectedModelId=custom_model，getEngine 走 CPU 从 model_custom.onnx 加载
                val ok = ModelManager.downloadModelFromUrl(
                    context = context,
                    url = url,
                    listener = object : ModelManager.ProgressListener {
                        override fun onProgress(bytesDownloaded: Long, totalBytes: Long, speedBps: Long) {
                            downloadProgress = if (totalBytes > 0) bytesDownloaded.toFloat() / totalBytes else 0f
                            downloadSpeed = speedBps
                            downloadBytes = bytesDownloaded
                            downloadTotal = totalBytes
                            statusText = "下载中..."
                            // ★ 系统通知栏：进度更新
                            Notifications.notifyDownloadProgress(context, "正在下载抠图模型…", bytesDownloaded, totalBytes, speedBps)
                        }
                        override fun onMirrorSwitch(mirrorIndex: Int, mirrorName: String) {
                            activeMirrorName = mirrorName
                            statusText = "正在从 $mirrorName 下载..."
                        }
                        override fun onMirrorError(mirrorName: String, error: String) {
                            android.util.Log.e("RMBG-DL", "$mirrorName 失败: $error")
                            statusText = "$mirrorName 失败: $error"
                        }
                        override fun onDone(file: File) {
                            isDownloading = false
                            modelReady = true
                            Notifications.cancelDownloadNotification(context)
                            // 转到自定义模型（CPU 推理，关闭 QNN/NNAPI 免误加速）
                            enableQnn = false
                            selectedBuiltinId = ModelManager.CUSTOM_MODEL_ID
                            ModelManager.selectedModelId = ModelManager.CUSTOM_MODEL_ID
                            Prefs.selectedModelId = ModelManager.CUSTOM_MODEL_ID
                            ModelManager.selectedMirrorIndex = selectedMirrorIndex
                            statusText = "自定义模型下载完成！选择图片开始抠图"
                            Notifications.notifyDownloadDone(context, "抠图模型已下载", "点击开始抠图")
                            Toast.makeText(context, "模型已就绪", Toast.LENGTH_SHORT).show()
                        }
                        override fun onError(e: Exception) {
                            isDownloading = false
                            activeMirrorName = null
                            Notifications.cancelDownloadNotification(context)
                            statusText = "下载失败: ${e.message}"
                            Toast.makeText(context, "下载失败: ${e.message}", Toast.LENGTH_SHORT).show()
                        }
                    }
                )
            }
        }
    }

    // ★ 恢复主模型（快捷临时切换结束后调用）：把模型配置恢复回主模型，并重建引擎
    fun restoreQuickModel() {
        val r = quickModelRestore ?: return
        quickModelRestore = null
        try { engine?.close() } catch (_: Exception) {}
        engine = null
        RmbgScreenState.engine = null
        selectedBuiltinId = r.id
        ModelManager.selectedModelId = r.id
        modelRepo = r.repo
        modelFile = r.file
        enableQnn = r.enableQnn
        statusText = "已恢复主模型：${r.id}"
    }

    // ---- 执行抠图（支持批量队列：自动连抠，处理完自动取队列下一张）----
    fun runRmbg() {
        if (isProcessing) return
        if (!ModelManager.isModelDownloaded()) {
            Toast.makeText(context, "请先下载模型", Toast.LENGTH_SHORT).show()
            return
        }
        if (originalBitmap == null) {
            Toast.makeText(context, "请先选择图片", Toast.LENGTH_SHORT).show()
            return
        }
        scope.launch {
            isProcessing = true
            stopRequested = false
            val total = batchQueue.size + 1
            var curNum = batchIndex + 1
            try {
                // 循环处理当前张 + 队列剩余，自动连抠
                var current = originalBitmap
                while (current != null) {
                    if (stopRequested) {
                        statusText = "已停止抠图"
                        current = null
                        break
                    }
                    val bmp = current
                    // ★ 历史分组键必须在【处理当前张之前】就更新为这张原图的 key：
                    //   否则批量连抠时第2张起用的还是上一张的 key，导致多张结果全归入上一张分支
                    //   （历史记录数量 < 实际抠图张数、原图分类错乱）。
                    //   computeOriginalKey 是确定性哈希：首张重算结果与选图时一致，安全。
                    currentHistoryKey = computeOriginalKey(bmp)
                    val t0 = System.currentTimeMillis()
                    val turbo = turboMode
                    val useTiling = !turbo && (bmp.width > 1024 || bmp.height > 1024)
                    isTiling = useTiling
                    tileProgress = 0f
                    statusText = if (total > 1) "正在抠图（$curNum/$total）..." else "正在抠图..."
                    val prevResult = resultBitmap
                    val e = withContext(Dispatchers.IO) {
                        val eng = getEngine()
                        if (turbo) {
                            val scaleF = 768f / maxOf(bmp.width, bmp.height)
                            val sw = maxOf(1, (bmp.width * scaleF).toInt())
                            val sh = maxOf(1, (bmp.height * scaleF).toInt())
                            val small = Bitmap.createScaledBitmap(bmp, sw, sh, true)
                            val smallResult = eng.removeBackground(small, threshold, tileMode = false)
                            small.recycle()
                            Bitmap.createScaledBitmap(smallResult, bmp.width, bmp.height, true)
                                .also { smallResult.recycle() }
                        } else {
                            eng.removeBackground(bmp, threshold, useTiling) { done, totalT ->
                                // ★ 分块间检查停止：允许用户中途取消长图分块推理
                                if (stopRequested) {
                                    tileProgressText = "已请求停止"
                                    eng.cancelCurrentRun()
                                }
                                tileProgress = done.toFloat() / totalT
                                tileProgressText = "$done/$totalT"
                            }
                        }
                    }
                    // 停止请求后丢弃本次结果
                    if (stopRequested) {
                        statusText = "已停止抠图"
                        current = null
                        break
                    }
                    val elapsedMs = System.currentTimeMillis() - t0
                    lastElapsedMs = elapsedMs // ★ 记录最近耗时（常驻显示）
                    // ★ 压入撤销栈（旧结果），再更新为新结果
                    pushUndo(prevResult)
                    resultBitmap = e
                    statusText = if (total > 1) "第 $curNum 张完成..." else "抠图完成！用时 ${"%.1f".format(elapsedMs / 1000f)} 秒，可保存/分享或换图继续"
                    // 每张完成异步存入历史 + 工程持久化（不阻塞主流程）
                    val curModelName = run {
                        val bm = ModelManager.builtinModels.find { it.id == selectedBuiltinId }
                        bm?.name ?: ModelManager.localModels().find { it.id == selectedBuiltinId }?.displayName
                            ?: selectedBuiltinId.ifBlank { "未知模型" }
                    }
                    val curAccel = when {
                        enableQnn && curModelName.contains("QNN") -> "QNN (HTP)"
                        enableNnapi -> "NNAPI"
                        else -> "CPU"
                    }
                    scope.launch {
                        withContext(Dispatchers.IO) {
                            try {
                                // ★ 自动保存开关：开启时仅保存每张原图的【第一次】抠图结果（原图→结果配对）
                                //   同一原图再次抠图不再重复自动入历史；关闭时结果仅留在当前界面
                                if (autoSaveResult) {
                                    val key = currentHistoryKey.ifBlank { computeOriginalKey(bmp) }
                                    // 该原图已有历史记录 → 不重复自动保存（除非用户手动保存/手动入历史）
                                    val exists = HistoryManager.getHistory(context)
                                        .any { it.originalKey.isNotEmpty() && it.originalKey == key }
                                    if (!exists) {
                                        HistoryManager.saveResult(
                                            context, e,
                                            original = bmp, // ★ 保存原图副本，供恢复画笔取原背景色
                                            modelName = curModelName,
                                            durationMs = elapsedMs,
                                            accel = curAccel,
                                            turbo = turbo,
                                            // ★ 用当前会话分支 key（继承：再次抠图归原图分支；新图/下一张已重算）
                                            originalKey = key
                                        )?.let { it -> latestHistoryFile = File(context.filesDir, "history/${it.fileName}") }
                                    }
                                }
                                // ★ 工程持久化与自动保存开关无关：始终保存，供下次打开恢复会话
                                ProjectStateManager.save(context, bmp, e)
                            } catch (_: Exception) {}
                        }
                        // ★ 历史版本号自增：批量队列每张都新入历史，必须刷新历史 Tab 列表
                        //   （否则 produceState(historyVersion) 不重查，批量结果不显示）
                        historyVersion++
                    }
                    val secs = elapsedMs / 1000f
                    // 取队列下一张
                    if (batchQueue.isNotEmpty()) {
                        val next = batchQueue.first()
                        batchQueue = batchQueue.drop(1)
                        batchIndex++
                        curNum++
                        originalBitmap = next
                        resultBitmap = null
                        // ★ 批量下一张=新图，开新历史分支
                        currentHistoryKey = computeOriginalKey(next)
                        statusText = "第 $curNum 张完成（${"%.1f".format(secs)}s），继续处理..."
                        Toast.makeText(context, "第 ${curNum - 1} 张完成，自动处理下一张", Toast.LENGTH_SHORT).show()
                        current = next
                    } else {
                        batchIndex = 0
                        statusText = if (total > 1)
                            "批量抠图完成！共 $total 张，用时 ${"%.1f".format(secs)} 秒/张"
                        else "抠图完成！用时 ${"%.1f".format(secs)} 秒，可保存/分享或换图继续"
                        current = null
                    }
                }
                Notifications.notifyDone(context)
            } catch (ex: Exception) {
                if (stopRequested || ex is java.util.concurrent.CancellationException) {
                    statusText = "已停止抠图"
                } else {
                    val msg = ex.message ?: ex.javaClass.simpleName
                    android.util.Log.e("RMBG-RUN", "抠图失败", ex)
                    statusText = "抠图失败: $msg"
                    Toast.makeText(context, "抠图失败: $msg", Toast.LENGTH_SHORT).show()
                }
            } finally {
                isProcessing = false
                isTiling = false
                // ★ 快捷临时切模型重抠完成 → 自动恢复主模型（不改变主模型/不写 Prefs）
                if (quickModelRestore != null) restoreQuickModel()
            }
        }
    }

    // ---- UI ----
    Scaffold(
        containerColor = Color(0xFFF5F7FA),
        bottomBar = {
            // ★ 关于页等全屏覆盖层显示时隐藏底栏（避免露出底部导航）
            if (!showAboutPage && !showEditor && !showPostProcessScreen) {
                NavigationBar(
                containerColor = Color(0xFFFFFFFF),
                tonalElevation = 3.dp
            ) {
                NavigationBarItem(
                    selected = currentTab == 0,
                    onClick = { currentTab = 0 },
                    icon = { Icon(Icons.Filled.Home, contentDescription = null) },
                    label = { Text("抠图") }
                )
                NavigationBarItem(
                    selected = currentTab == 1,
                    onClick = { currentTab = 1 },
                    icon = { Icon(Icons.Filled.Storage, contentDescription = null) },
                    label = { Text("模型") }
                )
                NavigationBarItem(
                    selected = currentTab == 2,
                    onClick = { currentTab = 2 },
                    icon = { Icon(Icons.Filled.History, contentDescription = null) },
                    label = { Text("历史") }
                )
                NavigationBarItem(
                    selected = currentTab == 3,
                    onClick = { currentTab = 3 },
                    icon = { Icon(Icons.Filled.Settings, contentDescription = null) },
                    label = { Text("设置") }
                )
            }
            } // ← 关于页等覆盖层时隐藏底栏
        }
    ) { padding ->
        // ★ 关于页：独立二级页面，覆盖整个主界面（含底部导航），返回后回到设置 Tab
        if (showAboutPage) {
            AboutPage(
                context = context,
                onBack = { showAboutPage = false }
            )
            return@Scaffold
        }
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            // ===== Tab 0: 抠图 =====
            if (currentTab == 0) {
            // ===== Hero 状态卡 =====
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(20.dp))
                    .background(
                        Brush.linearGradient(
                            listOf(
                                if (modelReady) Color(0xFFE8F5E9) else Color(0xFFE3F2FD),
                                Color(0xFFF5F7FA)
                            )
                        )
                    )
                    .padding(20.dp)
            ) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = if (modelReady) Icons.Filled.CheckCircle else Icons.Filled.Download,
                            contentDescription = null,
                            tint = if (modelReady) Color(0xFF2E7D32) else Color(0xFF1565C0),
                            modifier = Modifier.size(28.dp)
                        )
                        Spacer(Modifier.width(10.dp))
                        Text(
                            text = if (modelReady) "模型已就绪" else "准备就绪",
                            style = MaterialTheme.typography.titleLarge,
                            color = Color(0xFF1F2933),
                            fontWeight = FontWeight.Bold
                        )
                    }
                    Text(
                        text = statusText,
                        style = MaterialTheme.typography.bodyMedium,
                        color = Color(0xFF52606D)
                    )
                    if (modelReady) {
                        Text(
                            text = "本地 ONNX 推理 · 完全离线",
                            style = MaterialTheme.typography.labelMedium,
                            color = Color(0xFF2E7D32)
                        )
                    }
                }
            }

            // ===== 下载进度卡 =====
            if (isDownloading) {
                Card(
                    colors = CardDefaults.cardColors(containerColor = Color(0xFFFFFFFF)),
                    shape = RoundedCornerShape(16.dp)
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            CircularProgressIndicator(
                                progress = { downloadProgress },
                                modifier = Modifier.size(48.dp),
                                strokeWidth = 5.dp
                            )
                            Spacer(Modifier.width(14.dp))
                            Column {
                                Text(
                                    "下载中...",
                                    style = MaterialTheme.typography.titleMedium,
                                    fontWeight = FontWeight.Bold
                                )
                                Text(
                                    "${formatSize(downloadBytes)} / ${formatSize(if (downloadTotal > 0) downloadTotal else (ModelManager.builtinModels.find { it.id == selectedBuiltinId }?.sizeBytes ?: 176_153_355L))}",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                        LinearProgressIndicator(
                            progress = { downloadProgress },
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(8.dp)
                        )
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text(
                                "速度: ${formatSpeed(downloadSpeed)}",
                                style = MaterialTheme.typography.labelMedium,
                                color = MaterialTheme.colorScheme.primary
                            )
                            Text(
                                "${(downloadProgress * 100).toInt()}%",
                                style = MaterialTheme.typography.labelMedium,
                                fontWeight = FontWeight.Bold
                            )
                        }
                        activeMirrorName?.let {
                            Text(
                                "源: $it",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                }
            }

            // ===== 模型操作按钮 =====
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                if (isDownloading) {
                    // ★ 下载进行中 → 显示"取消下载"
                    Button(
                        onClick = { cancelDownload() },
                        colors = ButtonDefaults.buttonColors(
                            containerColor = MaterialTheme.colorScheme.error,
                            contentColor = MaterialTheme.colorScheme.onError
                        ),
                        modifier = Modifier
                            .weight(1f)
                            .height(52.dp),
                        shape = RoundedCornerShape(14.dp)
                    ) {
                        Icon(Icons.Filled.Close, contentDescription = null)
                        Spacer(Modifier.width(8.dp))
                        Text("取消下载", fontSize = 15.sp, fontWeight = FontWeight.Bold)
                    }
                } else if (!modelReady) {
                    Button(
                        onClick = { startDownload() },
                        enabled = !isDownloading,
                        modifier = Modifier
                            .weight(1f)
                            .height(52.dp),
                        shape = RoundedCornerShape(14.dp)
                    ) {
                        Icon(Icons.Filled.Download, contentDescription = null)
                        Spacer(Modifier.width(8.dp))
                        Text(
                            if (isDownloading) "下载中..."
                            else {
                                val curBm = ModelManager.builtinModels.find { it.id == selectedBuiltinId }
                                if (curBm != null && curBm.isQnn) "部署 QNN EPContext (${curBm.sizeBytes / 1024 / 1024}MB)"
                                else if (curBm != null) "下载模型 (${curBm.sizeBytes / 1024 / 1024}MB)"
                                else "下载模型"
                            },
                            fontSize = 15.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }
                } else if (selectedBuiltinId.startsWith("qnn_")) {
                    OutlinedButton(
                        onClick = { startDownload() }, // 重新部署：提取插件 .so + 校验 context
                        modifier = Modifier
                            .weight(1f)
                            .height(52.dp),
                        shape = RoundedCornerShape(14.dp)
                    ) {
                        Icon(Icons.Filled.Refresh, contentDescription = null)
                        Spacer(Modifier.width(8.dp))
                        Text("重新部署", fontSize = 15.sp)
                    }
                } else {
                    OutlinedButton(
                        onClick = {
                            ModelManager.deleteModel()
                            modelReady = false
                            statusText = "模型已删除，需重新下载"
                        },
                        modifier = Modifier
                            .weight(1f)
                            .height(52.dp),
                        shape = RoundedCornerShape(14.dp)
                    ) {
                        Icon(Icons.Filled.Delete, contentDescription = null)
                        Spacer(Modifier.width(8.dp))
                        Text("删除模型", fontSize = 15.sp)
                    }
                }
            }

            } // ---- Tab 0 结束 ----

            // ===== Tab 1: 模型 =====
            if (currentTab == 1) {
            // ★ 模型页卡顿优化：内置模型下载状态 / 已部署重绘列表 都是磁盘 IO（File.exists/listFiles），
            //   每次重组都重算会卡。这里只绑定版本号，版本不变直接复用缓存，重组零 IO。
            val builtinReadyCache = remember(builtinModelsVersion) {
                ModelManager.builtinModels.associate { it.id to ModelManager.isBuiltinModelDownloaded(it) }
            }
            val localModelCache = remember(localModelsVersion) { ModelManager.localModels() }
            // ★ 已部署 AI 重绘模型列表（IO 重：listFiles 扫 models 目录；仅版本号变化时重扫）
            val deployedModelCache = remember(deployedModelsVersion, aiModelZipReady) {
                AiRedrawEngine.deployedModels(context)
            }
            // ===== 内置模型选择 =====
Card(
                colors = CardDefaults.cardColors(containerColor = Color(0xFFFFFFFF)),
                shape = RoundedCornerShape(16.dp)
            ) {
                Column(
                    modifier = Modifier.padding(14.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(Icons.Filled.Storage, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                        Spacer(Modifier.width(8.dp))
                        Text("内置模型", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold, modifier = Modifier.weight(1f))
                        TextButton(onClick = { showBuiltinPicker = !showBuiltinPicker }) {
                            Text(if (showBuiltinPicker) "收起" else "选择")
                        }
                    }
                    // ---- 当前选中模型展示（收起时也能看到选了哪个模型）----
                    val curBm = ModelManager.builtinModels.find { it.id == selectedBuiltinId }
                    val curLocal = localModelCache.find { it.id == selectedBuiltinId }
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(12.dp))
                            .background(MaterialTheme.colorScheme.primary.copy(alpha = 0.08f))
                            .padding(horizontal = 10.dp, vertical = 8.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            Icons.Filled.CheckCircle,
                            contentDescription = null,
                            tint = if (curBm?.isQnn == true) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.secondary,
                            modifier = Modifier.size(18.dp)
                        )
                        Spacer(Modifier.width(8.dp))
                        Column(Modifier.weight(1f)) {
                            Text(
                                curBm?.name ?: curLocal?.displayName ?: "自定义模型",
                                style = MaterialTheme.typography.bodyMedium,
                                fontWeight = FontWeight.Bold
                                // ★ 完整显示模型名：不限制行数（本地模型长文件名也能看全）
                            )
                            Text(
                                when {
                                    curBm?.isQnn == true -> "⚡ QNN · HTP DSP 加速 · ${formatSize(curBm.sizeBytes)}"
                                    curBm != null -> "🧠 CPU · ${formatSize(curBm.sizeBytes)}"
                                    curLocal != null -> "本地导入 · ${formatSize(curLocal.sizeBytes)}"
                                    else -> "自定义配置 · ${modelRepo}/${modelFile}"
                                },
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                                // ★ 完整显示描述：不限制行数
                            )
                        }
                        Spacer(Modifier.width(4.dp))
                        if (modelReady) {
                            Text("✅ 就绪", style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.Bold, color = Color(0xFF2E7D32))
                        } else {
                            Text("未就绪", style = MaterialTheme.typography.labelSmall, color = Color(0xFFB71C1C))
                        }
                    }
                    if (showBuiltinPicker) {
                        // ================= CPU 模型组 =================
                        Text(
                            "🧠 CPU 模型（普通 ONNX · CPU/NNAPI 加速）",
                            style = MaterialTheme.typography.labelMedium,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.padding(top = 2.dp)
                        )
                        ModelManager.cpuModels.forEach { bm ->
                            val selected = selectedBuiltinId == bm.id
                            val bmReady = builtinReadyCache[bm.id] ?: ModelManager.isBuiltinModelDownloaded(bm)
                            val bmDownloading = downloadingId == bm.id
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .clip(RoundedCornerShape(12.dp))
                                    .background(if (selected) MaterialTheme.colorScheme.primary.copy(alpha = 0.12f) else Color.Transparent)
                                    .selectable(
                                        selected = selected,
                                        enabled = !isProcessing, // ★ 运行中禁止切换模型
                                        onClick = {
                                            if (isProcessing) return@selectable
                                            selectedBuiltinId = bm.id
                                            ModelManager.selectedModelId = bm.id
                                            Prefs.selectedModelId = bm.id
                                            modelRepo = bm.hfRepo
                                            modelFile = bm.hfFile
                                            enableQnn = bm.isQnn
                                            Prefs.enableQnn = bm.isQnn
                                            // 切换模型：立即释放旧引擎（下次抠图自动加载新模型）
                                            try { engine?.close() } catch (_: Exception) {}
                                            engine = null
                                            // 立即检查该模型是否已下载
                                            modelReady = ModelManager.isModelDownloaded()
                                            statusText = if (modelReady) "模型已就绪，选择图片开始抠图" else if (bm.isQnn) "需要部署 ${bm.name}（内置 ${bm.sizeBytes / 1024 / 1024}MB，点击部署）" else "需要下载模型（约${bm.sizeBytes / 1024 / 1024}MB）"
                                        }
                                    )
                                    .padding(horizontal = 8.dp, vertical = 8.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                RadioButton(selected = selected, onClick = {
                                    if (isProcessing) return@RadioButton
                                    selectedBuiltinId = bm.id
                                    ModelManager.selectedModelId = bm.id
                                    Prefs.selectedModelId = bm.id
                                    modelRepo = bm.hfRepo
                                    modelFile = bm.hfFile
                                    enableQnn = bm.isQnn
                                    Prefs.enableQnn = bm.isQnn
                                    try { engine?.close() } catch (_: Exception) {}
                                    engine = null
                                    modelReady = ModelManager.isModelDownloaded()
                                    statusText = if (modelReady) "模型已就绪，选择图片开始抠图" else if (bm.isQnn) "需要部署 ${bm.name}（内置 ${bm.sizeBytes / 1024 / 1024}MB，点击部署）" else "需要下载模型（约${bm.sizeBytes / 1024 / 1024}MB）"
                                })
                                Column(
                                    Modifier
                                        .padding(start = 6.dp)
                                        .weight(1f)
                                        .fillMaxWidth()
                                ) {
                                    Text(
                                        bm.name,
                                        style = MaterialTheme.typography.bodyLarge,
                                        fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal,
                                        maxLines = 1,
                                        overflow = TextOverflow.Ellipsis
                                    )
                                    Text(
                                        bm.description,
                                        style = MaterialTheme.typography.labelSmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                }
                                Spacer(Modifier.width(4.dp))
                                // 状态徽标 + 下载/删除操作
                                Column(horizontalAlignment = Alignment.End, verticalArrangement = Arrangement.spacedBy(4.dp)) {
                                    if (bmReady) {
                                        Text("✅ 已下载", style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.Bold, color = Color(0xFF2E7D32))
                                    } else {
                                        Text("未下载", style = MaterialTheme.typography.labelSmall, color = Color(0xFF757575))
                                    }
                                    if (bmReady) {
                                        // 已下载 → 删除按钮
                                        IconButton(
                                            onClick = {
                                                ModelManager.deleteBuiltinModel(bm)
                                                builtinModelsVersion++ // ★ 刷新内置模型下载状态
                                                if (selectedBuiltinId == bm.id) {
                                                    modelReady = false
                                                    statusText = "模型已删除，可重新下载"
                                                    try { engine?.close() } catch (_: Exception) {}
                                                    engine = null
                                                } else {
                                                    statusText = "已删除 ${bm.name}"
                                                }
                                            },
                                            modifier = Modifier.size(28.dp)
                                        ) {
                                            Icon(Icons.Filled.Delete, contentDescription = "删除 ${bm.name}", tint = MaterialTheme.colorScheme.error, modifier = Modifier.size(16.dp))
                                        }
                                    } else {
                                        // 未下载 → 下载按钮
                                        IconButton(
                                            onClick = {
                                                selectedBuiltinId = bm.id
                                                ModelManager.selectedModelId = bm.id
                                                Prefs.selectedModelId = bm.id
                                                modelRepo = bm.hfRepo
                                                modelFile = bm.hfFile
                                                enableQnn = bm.isQnn
                                                Prefs.enableQnn = bm.isQnn
                                                scope.launch {
                                                    downloadingId = bm.id
                                                    statusText = "正在下载 ${bm.name}..."
                                                    ModelManager.downloadModel(
                                                        context = context,
                                                        listener = object : ModelManager.ProgressListener {
                                                            override fun onProgress(bytesDownloaded: Long, totalBytes: Long, speedBps: Long) {
                                                                downloadProgress = if (totalBytes > 0) bytesDownloaded.toFloat() / totalBytes else 0f
                                                                downloadSpeed = speedBps
                                                                downloadBytes = bytesDownloaded
                                                                downloadTotal = totalBytes
                                                            }
                                                            override fun onMirrorSwitch(mirrorIndex: Int, mirrorName: String) {
                                                                activeMirrorName = mirrorName
                                                                statusText = "正在从 $mirrorName 下载 ${bm.name}..."
                                                            }
                                                            override fun onMirrorError(mirrorName: String, error: String) {
                                                                activeMirrorName = null
                                                            }
                                                            override fun onDone(file: File) {}
                                                            override fun onError(e: Exception) {
                                                                statusText = "${bm.name} 下载失败: ${e.message ?: "未知错误"}"
                                                            }
                                                        }
                                                    )
                                                    modelReady = ModelManager.isModelDownloaded()
                                                    builtinModelsVersion++ // ★ 刷新内置模型下载状态
                                                    statusText = if (modelReady) "${bm.name} 下载完成，模型已就绪" else "${bm.name} 下载失败，请重试"
                                                    downloadingId = null
                                                }
                                            },
                                            modifier = Modifier.size(28.dp)
                                        ) {
                                            if (bmDownloading) {
                                                CircularProgressIndicator(modifier = Modifier.size(14.dp), strokeWidth = 2.dp)
                                            } else {
                                                Icon(Icons.Filled.Download, contentDescription = "下载 ${bm.name}", tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(16.dp))
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        // ================= QNN 模型组 =================
                        Text(
                            "⚡ QNN 模型（骁龙 DSP · HTP 离线编译）",
                            style = MaterialTheme.typography.labelMedium,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.padding(top = 8.dp)
                        )
                        ModelManager.qnnModels.forEach { bm ->
                            val selected = selectedBuiltinId == bm.id
                            val bmReady = builtinReadyCache[bm.id] ?: ModelManager.isBuiltinModelDownloaded(bm)
                            val bmDownloading = downloadingId == bm.id
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .clip(RoundedCornerShape(12.dp))
                                    .background(if (selected) MaterialTheme.colorScheme.primary.copy(alpha = 0.12f) else Color.Transparent)
                                    .selectable(
                                        selected = selected,
                                        enabled = !isProcessing, // ★ 运行中禁止切换模型
                                        onClick = {
                                            if (isProcessing) return@selectable
                                            // ★ 超分模型条目不参与抠图选中（独立 Prefs 管理）
                                            if (bm.id == "qnn_realesrgan") {
                                                Toast.makeText(context, "超分模型：点右侧按钮下载/删除，不参与抠图选择", Toast.LENGTH_SHORT).show()
                                                return@selectable
                                            }
                                            selectedBuiltinId = bm.id
                                            ModelManager.selectedModelId = bm.id
                                            Prefs.selectedModelId = bm.id
                                            modelRepo = bm.hfRepo
                                            modelFile = bm.hfFile
                                            enableQnn = true
                                            Prefs.enableQnn = true
                                            try { engine?.close() } catch (_: Exception) {}
                                            engine = null
                                            modelReady = ModelManager.isModelDownloaded()
                                            statusText = if (modelReady) "模型已就绪，选择图片开始抠图" else "需要部署 ${bm.name}（内置 ${bm.sizeBytes / 1024 / 1024}MB，点击部署）"
                                        }
                                    )
                                    .padding(horizontal = 8.dp, vertical = 8.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                RadioButton(selected = selected, onClick = {
                                    if (isProcessing) return@RadioButton
                                    // ★ 超分模型条目不参与抠图选中
                                    if (bm.id == "qnn_realesrgan") {
                                        Toast.makeText(context, "超分模型：点右侧按钮下载/删除，不参与抠图选择", Toast.LENGTH_SHORT).show()
                                        return@RadioButton
                                    }
                                    selectedBuiltinId = bm.id
                                    ModelManager.selectedModelId = bm.id
                                    Prefs.selectedModelId = bm.id
                                    modelRepo = bm.hfRepo
                                    modelFile = bm.hfFile
                                    enableQnn = true
                                    Prefs.enableQnn = true
                                    try { engine?.close() } catch (_: Exception) {}
                                    engine = null
                                    modelReady = ModelManager.isModelDownloaded()
                                    statusText = if (modelReady) "模型已就绪，选择图片开始抠图" else "需要部署 ${bm.name}（内置 ${bm.sizeBytes / 1024 / 1024}MB，点击部署）"
                                })
                                Column(
                                    Modifier
                                        .padding(start = 6.dp)
                                        .weight(1f)
                                        .fillMaxWidth()
                                ) {
                                    Text(
                                        bm.name,
                                        style = MaterialTheme.typography.bodyLarge,
                                        fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal,
                                        maxLines = 1,
                                        overflow = TextOverflow.Ellipsis
                                    )
                                    Text(
                                        bm.description,
                                        style = MaterialTheme.typography.labelSmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                }
                                Spacer(Modifier.width(4.dp))
                                Column(horizontalAlignment = Alignment.End, verticalArrangement = Arrangement.spacedBy(4.dp)) {
                                    if (bmReady) {
                                        Text(if (bm.id == "qnn_realesrgan") "✅ 已就绪" else "✅ 已部署", style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.Bold, color = Color(0xFF2E7D32))
                                    } else {
                                        Text(if (bm.id == "qnn_realesrgan") "未下载" else "未部署", style = MaterialTheme.typography.labelSmall, color = Color(0xFF757575))
                                    }
                                    if (bmReady) {
                                        // 已部署 → 删除按钮（释放模型目录空间；超分同时清 Prefs）
                                        IconButton(
                                            onClick = {
                                                ModelManager.deleteBuiltinModel(bm)
                                                builtinModelsVersion++ // ★ 刷新内置模型下载状态
                                                if (bm.id == "qnn_realesrgan") {
                                                    Prefs.superResModelPath = ""
                                                    statusText = "超分模型已删除，超分回退内置 CPU 模型"
                                                    Toast.makeText(context, "超分模型已删除", Toast.LENGTH_SHORT).show()
                                                } else if (selectedBuiltinId == bm.id) {
                                                    modelReady = false
                                                    statusText = "模型已删除，可重新部署"
                                                    try { engine?.close() } catch (_: Exception) {}
                                                    engine = null
                                                } else {
                                                    statusText = "已删除 ${bm.name}"
                                                }
                                            },
                                            modifier = Modifier.size(28.dp)
                                        ) {
                                            Icon(Icons.Filled.Delete, contentDescription = "删除 ${bm.name}", tint = MaterialTheme.colorScheme.error, modifier = Modifier.size(16.dp))
                                        }
                                    } else {
                                        // 未部署 → 部署按钮（从 assets 拷贝 EPContext 产物；超分走云下载）
                                        IconButton(
                                            onClick = {
                                                if (bm.id == "qnn_realesrgan") {
                                                    // ★ 超分模型：云端下载（QNN 匹配 → HF zip；否则回退 CPU 内置资产）
                                                    scope.launch {
                                                        downloadingId = bm.id
                                                        statusText = "正在下载超分模型（QNN HTP / CPU 自适应）..."
                                                        val path = ModelManager.downloadSuperResModel(
                                                            context = context,
                                                            listener = object : ModelManager.ProgressListener {
                                                                override fun onProgress(bytesDownloaded: Long, totalBytes: Long, speedBps: Long) {
                                                                    statusText = "超分下载中 ${bytesDownloaded / 1024 / 1024}/${if (totalBytes > 0) totalBytes / 1024 / 1024 else "?"}MB"
                                                                }
                                                                override fun onMirrorSwitch(mirrorIndex: Int, mirrorName: String) { statusText = "正在从 $mirrorName 下载超分模型..." }
                                                                override fun onMirrorError(mirrorName: String, error: String) { statusText = "$mirrorName 下载失败：$error" }
                                                                override fun onDone(file: java.io.File) {}
                                                                override fun onError(e: Exception) { statusText = "超分下载错误：${e.message}" }
                                                            }
                                                        )
                                                        downloadingId = null
                                                        if (path != null && java.io.File(path).exists()) {
                                                            Prefs.superResModelPath = path
                                                            builtinModelsVersion++ // ★ 刷新内置模型下载状态
                                                            statusText = "✅ 超分模型已下载（QNN HTP 加速），超分时自动启用"
                                                            Toast.makeText(context, "超分模型已就绪（QNN HTP）", Toast.LENGTH_SHORT).show()
                                                        } else {
                                                            // 回退 CPU：内置 assets 已就绪，免下载
                                                            Prefs.superResModelPath = ""
                                                            builtinModelsVersion++
                                                            statusText = "当前设备自动使用内置 CPU 超分（免下载），超分时自动启用"
                                                            Toast.makeText(context, "已启用 CPU 超分（内置模型）", Toast.LENGTH_SHORT).show()
                                                        }
                                                    }
                                                    return@IconButton
                                                }
                                                if (bm.id == "qnn_animeseg") {
                                                    // ★ Anime-Seg：按设备 SoC 自动匹配对应 NPU 编译产物（sm8350~8850 → HF zip；sm8550 → 内置 v73）
                                                    scope.launch {
                                                        downloadingId = bm.id
                                                        statusText = "正在部署 Anime-Seg（按设备 NPU 自动匹配）..."
                                                        val ok = ModelManager.downloadAnimeSeg(
                                                            context = context,
                                                            listener = object : ModelManager.ProgressListener {
                                                                override fun onProgress(bytesDownloaded: Long, totalBytes: Long, speedBps: Long) {
                                                                    statusText = "Anime-Seg 下载中 ${bytesDownloaded / 1024 / 1024}/${if (totalBytes > 0) totalBytes / 1024 / 1024 else "?"}MB"
                                                                }
                                                                override fun onMirrorSwitch(mirrorIndex: Int, mirrorName: String) { statusText = "正在从 $mirrorName 下载 Anime-Seg..." }
                                                                override fun onMirrorError(mirrorName: String, error: String) { statusText = "$mirrorName 下载失败：$error" }
                                                                override fun onDone(file: java.io.File) {}
                                                                override fun onError(e: Exception) { statusText = "Anime-Seg 下载错误：${e.message}" }
                                                            }
                                                        )
                                                        downloadingId = null
                                                        if (ok) {
                                                            modelReady = ModelManager.isModelDownloaded()
                                                            builtinModelsVersion++ // ★ 刷新内置模型下载状态
                                                            statusText = "${bm.name} 已部署（按设备 NPU 匹配），选择图片开始抠图"
                                                        } else {
                                                            statusText = "${bm.name} 部署失败（设备无对应 NPU 变体或网络异常）"
                                                        }
                                                    }
                                                    return@IconButton
                                                }
                                                selectedBuiltinId = bm.id
                                                ModelManager.selectedModelId = bm.id
                                                Prefs.selectedModelId = bm.id
                                                enableQnn = true
                                                Prefs.enableQnn = true
                                                scope.launch {
                                                    downloadingId = bm.id
                                                    statusText = "正在部署 ${bm.name}..."
                                                    val ok = withContext(Dispatchers.IO) { ModelManager.ensureQnnContext(context, bm.id) }
                                                    downloadingId = null
                                                    if (ok) {
                                                        modelReady = ModelManager.isModelDownloaded()
                                                        builtinModelsVersion++ // ★ 刷新内置模型下载状态
                                                        statusText = "${bm.name} 已部署，选择图片开始抠图"
                                                    } else {
                                                        statusText = "${bm.name} 部署失败"
                                                    }
                                                }
                                            },
                                            modifier = Modifier.size(28.dp)
                                        ) {
                                            if (bmDownloading) {
                                                CircularProgressIndicator(modifier = Modifier.size(14.dp), strokeWidth = 2.dp)
                                            } else {
                                                Icon(Icons.Filled.Download, contentDescription = "部署 ${bm.name}", tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(16.dp))
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }


            // ===== 本地模型导入 =====
            Card(
                colors = CardDefaults.cardColors(containerColor = Color(0xFFFFFFFF)),
                shape = RoundedCornerShape(16.dp)
            ) {
                Column(
                    modifier = Modifier.padding(14.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Filled.FolderOpen, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                        Spacer(Modifier.width(8.dp))
                        Text("本地模型", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold, modifier = Modifier.weight(1f))
                    }
                    Text(
                        "导入 .onnx 模型文件，或将 onnx+bin 打包成 zip 一并导入（QNN 推荐）",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Button(
                        onClick = { importModelLauncher.launch("*/*") },
                        modifier = Modifier.fillMaxWidth().height(44.dp),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Text("导入本地 .onnx 模型")
                    }
                    OutlinedButton(
                        onClick = { importZipLauncher.launch("*/*") },
                        modifier = Modifier.fillMaxWidth().height(44.dp),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Icon(Icons.Filled.Archive, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(Modifier.width(6.dp))
                        Text("导入 zip（onnx + bin）")
                    }
                    // ★ 超分模型导入：独立入口，走 importSuperResModelZip 存 superres/ 目录，绝不混入抠图列表
                    OutlinedButton(
                        onClick = { importSuperResLauncher.launch("*/*") },
                        modifier = Modifier.fillMaxWidth().height(44.dp),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Icon(Icons.Filled.ZoomIn, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(Modifier.width(6.dp))
                        Text("导入超分模型")
                    }
                    // ★ 当前已导入的超分模型状态
                    val curSrPath = com.rmbg.offline.Prefs.superResModelPath
                    if (curSrPath.isNotEmpty() && File(curSrPath).exists()) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(Icons.Filled.CheckCircle, contentDescription = null, modifier = Modifier.size(14.dp), tint = Color(0xFF4FC3F7))
                            Spacer(Modifier.width(6.dp))
                            Text(
                                "超分模型: ${File(curSrPath).name}",
                                style = MaterialTheme.typography.labelSmall,
                                color = Color(0xFF4FC3F7),
                                maxLines = 1
                            )
                        }
                    }
                    // 已导入的本地模型列表（绑定版本号：导入/删除后响应式刷新；复用 Tab 1 顶部缓存，避免重复扫描磁盘）
                    val locals = localModelCache
                    if (locals.isNotEmpty()) {
                        locals.forEach { lm ->
                            val sel = selectedBuiltinId == lm.id
                            // ★ 兼容性标注（后台深探测已缓存则直接用；未缓存先按文件名快判，再后台补全）
                            //   初始快判用 runCatching 包裹，避免读取到异常文件(空/损坏)时主线程崩溃
                            var compat by remember(lm.id) {
                                mutableStateOf(
                                    runCatching { ModelManager.probeLocalModelCompat(lm, deep = false) }
                                        .getOrElse { ModelManager.ModelCompatibility(ModelManager.AccelSupport.CPU_ONLY, "", "", "") }
                                )
                            }
                            LaunchedEffect(lm.id) {
                                compat = withContext(Dispatchers.IO) {
                                    runCatching { ModelManager.probeLocalModelCompat(lm, deep = true) }
                                        .getOrElse { ModelManager.ModelCompatibility(ModelManager.AccelSupport.CPU_ONLY, "", "", "") }
                                }
                            }
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .clip(RoundedCornerShape(12.dp))
                                    .background(if (sel) MaterialTheme.colorScheme.primary.copy(alpha = 0.12f) else Color.Transparent)
                                    .selectable(
                                        selected = sel,
                                        enabled = !isProcessing, // ★ 运行中禁止切换模型
                                        onClick = {
                                            if (isProcessing) return@selectable
                                            selectedBuiltinId = lm.id
                                            ModelManager.selectedModelId = lm.id
                                            Prefs.selectedModelId = lm.id
                                            // ★ 按探测结果决定 QNN 开关：EPContext/QNN_TRY 开，否则关+CPU
                                            val c = ModelManager.probeLocalModelCompat(lm, deep = false)
                                            val qnnOk = c.support == ModelManager.AccelSupport.QNN_EPCONTEXT ||
                                                c.support == ModelManager.AccelSupport.QNN_TRY
                                            enableQnn = qnnOk
                                            Prefs.enableQnn = qnnOk
                                            try { engine?.close() } catch (_: Exception) {}
                                            engine = null
                                            modelReady = ModelManager.isModelDownloaded()
                                            statusText = "已切换到本地模型: ${lm.displayName}\n${c.qnnLabel} · ${c.nnapiLabel}"
                                        }
                                    )
                                    .padding(horizontal = 8.dp, vertical = 8.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                RadioButton(selected = sel, onClick = {
                                    if (isProcessing) return@RadioButton
                                    selectedBuiltinId = lm.id
                                    ModelManager.selectedModelId = lm.id
                                    Prefs.selectedModelId = lm.id
                                    // ★ 按探测结果决定 QNN 开关
                                    val c = ModelManager.probeLocalModelCompat(lm, deep = false)
                                    val qnnOk = c.support == ModelManager.AccelSupport.QNN_EPCONTEXT ||
                                        c.support == ModelManager.AccelSupport.QNN_TRY
                                    enableQnn = qnnOk
                                    Prefs.enableQnn = qnnOk
                                    try { engine?.close() } catch (_: Exception) {}
                                    engine = null
                                    modelReady = ModelManager.isModelDownloaded()
                                    statusText = "已切换到本地模型: ${lm.displayName}\n${c.qnnLabel} · ${c.nnapiLabel}"
                                })
                                Column(Modifier.padding(start = 6.dp).weight(1f)) {
                                    Text(lm.displayName, style = MaterialTheme.typography.bodyLarge, fontWeight = if (sel) FontWeight.Bold else FontWeight.Normal)
                                    Text("本地导入 · ${formatSize(lm.sizeBytes)}", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                    // ★ 兼容性标签：QNN / NNAPI 支不支持一目了然
                                    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                                        Text(
                                            compat.qnnLabel,
                                            style = MaterialTheme.typography.labelSmall,
                                            color = when (compat.support) {
                                                ModelManager.AccelSupport.QNN_EPCONTEXT, ModelManager.AccelSupport.QNN_TRY -> Color(0xFF2E7D32)
                                                ModelManager.AccelSupport.QNN_RISKY -> Color(0xFFE65100)
                                                ModelManager.AccelSupport.CPU_ONLY -> Color(0xFF757575)
                                            }
                                        )
                                        Text(compat.nnapiLabel, style = MaterialTheme.typography.labelSmall, color = Color(0xFF757575))
                                    }
                                    Text(compat.desc, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 2, overflow = TextOverflow.Ellipsis)
                                }
                                // 删除本地模型按钮
                                IconButton(
                                    onClick = {
                                        if (ModelManager.deleteLocalModel(lm.id)) {
                                            localModelsVersion++ // ★ 触发本地模型列表刷新
                                            if (selectedBuiltinId == lm.id) {
                                                // 删的是当前选中的模型 → 切回第一个内置模型
                                                val fallback = ModelManager.builtinModels.firstOrNull()
                                                selectedBuiltinId = fallback?.id ?: ""
                                                ModelManager.selectedModelId = selectedBuiltinId
                                                Prefs.selectedModelId = selectedBuiltinId
                                                try { engine?.close() } catch (_: Exception) {}
                                                engine = null
                                            }
                                            modelReady = ModelManager.isModelDownloaded()
                                            statusText = "已删除本地模型: ${lm.displayName}"
                                        } else {
                                            Toast.makeText(context, "删除失败，请重试", Toast.LENGTH_SHORT).show()
                                        }
                                    },
                                    modifier = Modifier.size(36.dp)
                                ) {
                                    Icon(
                                        Icons.Filled.Delete,
                                        contentDescription = "删除 ${lm.displayName}",
                                        tint = MaterialTheme.colorScheme.error,
                                        modifier = Modifier.size(18.dp)
                                    )
                                }
                            }
                        }
                    }
                }
            }

            // ===== 模型诊断卡片（默认收起，点击展开，避免干扰）=====
            Card(
                colors = CardDefaults.cardColors(containerColor = Color(0xFFF7F8FA)),
                shape = RoundedCornerShape(16.dp)
            ) {
                Column(
                    modifier = Modifier.padding(14.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(12.dp))
                            .selectable(
                                selected = showModelDiag,
                                onClick = { showModelDiag = !showModelDiag }
                            )
                            .padding(vertical = 4.dp)
                    ) {
                        Icon(Icons.Filled.Info, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                        Spacer(Modifier.width(8.dp))
                        Text("模型状态诊断", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold, modifier = Modifier.weight(1f))
                        Text(
                            if (showModelDiag) "收起 ▲" else "展开 ▼",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.primary
                        )
                    }
                    if (showModelDiag) {
                        Text(
                            ModelManager.modelStatusReport(),
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            fontFamily = FontFamily.Monospace
                        )
                    }
                }
            }

            // ===== 下载源（镜像）卡片 =====
            Card(
                colors = CardDefaults.cardColors(containerColor = Color(0xFFFFFFFF)),
                shape = RoundedCornerShape(16.dp)
            ) {
                Column(
                    modifier = Modifier.padding(14.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Filled.Public, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                        Spacer(Modifier.width(8.dp))
                        Text("下载源（镜像）", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
                    }
                    Spacer(Modifier.height(4.dp))
                    // ★ 自动选择（国内/国外按可达性探测，默认推荐）
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(12.dp))
                            .background(
                                if (selectedMirrorIndex < 0) MaterialTheme.colorScheme.primary.copy(alpha = 0.12f)
                                else Color.Transparent
                            )
                            .selectable(
                                selected = selectedMirrorIndex < 0,
                                onClick = {
                                    if (!isDownloading) {
                                        selectedMirrorIndex = -1
                                        ModelManager.selectedMirrorIndex = -1
                                        ModelManager.invalidateAutoMirror()
                                        Prefs.mirrorIndex = -1
                                    }
                                }
                            )
                            .padding(horizontal = 8.dp, vertical = 8.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        RadioButton(
                            selected = selectedMirrorIndex < 0,
                            onClick = {
                                if (!isDownloading) {
                                    selectedMirrorIndex = -1
                                    ModelManager.selectedMirrorIndex = -1
                                    ModelManager.invalidateAutoMirror()
                                    Prefs.mirrorIndex = -1
                                }
                            },
                            enabled = !isDownloading
                        )
                        Column(Modifier.padding(start = 6.dp)) {
                            Text(
                                "自动选择",
                                style = MaterialTheme.typography.bodyLarge,
                                fontWeight = if (selectedMirrorIndex < 0) FontWeight.Bold else FontWeight.Normal
                            )
                            Text(
                                "国内/国外自动探测最优源（默认推荐）",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                    ModelManager.mirrors.forEachIndexed { index, mirror ->
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(12.dp))
                                .background(
                                    if (selectedMirrorIndex == index) MaterialTheme.colorScheme.primary.copy(alpha = 0.12f)
                                    else Color.Transparent
                                )
                                .selectable(
                                    selected = selectedMirrorIndex == index,
                                    onClick = {
                                        if (!isDownloading) {
                                            selectedMirrorIndex = index
                                            ModelManager.selectedMirrorIndex = index
                                            ModelManager.invalidateAutoMirror()
                                            Prefs.mirrorIndex = index
                                        }
                                    }
                                )
                                .padding(horizontal = 8.dp, vertical = 8.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            RadioButton(
                                selected = selectedMirrorIndex == index,
                                onClick = {
                                    if (!isDownloading) {
                                        selectedMirrorIndex = index
                                        ModelManager.selectedMirrorIndex = index
                                        ModelManager.invalidateAutoMirror()
                                        Prefs.mirrorIndex = index
                                    }
                                },
                                enabled = !isDownloading
                            )
                            Column(Modifier.padding(start = 6.dp)) {
                                Text(
                                    mirror.name,
                                    style = MaterialTheme.typography.bodyLarge,
                                    fontWeight = if (selectedMirrorIndex == index) FontWeight.Bold else FontWeight.Normal
                                )
                                Text(
                                    mirror.description,
                                    style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                            if (activeMirrorName == mirror.name && isDownloading) {
                                Spacer(Modifier.weight(1f))
                                Text(
                                    "使用中",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.primary
                                )
                            }
                        }
                    }
                    Text(
                        text = "下载失败自动切换下一个镜像源 · 支持断点续传",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            // ===== 模型配置（占位 · 可换其他 HF 模型）=====
            Card(
                colors = CardDefaults.cardColors(containerColor = Color(0xFFFFFFFF)),
                shape = RoundedCornerShape(16.dp)
            ) {
                Column(
                    modifier = Modifier.padding(14.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(Icons.Filled.Settings, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                        Spacer(Modifier.width(8.dp))
                        Text("模型配置", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold, modifier = Modifier.weight(1f))
                        TextButton(onClick = { showModelConfig = !showModelConfig }) {
                            Text(if (showModelConfig) "收起" else "展开")
                        }
                    }
                    if (showModelConfig) {
                        // ★ 下载链接入口（醒目置顶）：粘贴 ONNX 直链即可一键下载并自动切换
                        Text(
                            "粘贴模型下载链接（.onnx 或 .zip），下载后自动切换为该模型",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.primary
                        )
                        OutlinedTextField(
                            value = modelUrlInput,
                            onValueChange = { modelUrlInput = it },
                            label = { Text("模型下载链接") },
                            placeholder = { Text("https://huggingface.co/.../model.onnx 或 https://.../model.zip") },
                            singleLine = true,
                            modifier = Modifier.fillMaxWidth()
                        )
                        Button(
                            onClick = { downloadModelUrl() },
                            enabled = !isDownloading && !aiRedrawDownloading,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Icon(Icons.Filled.Download, contentDescription = null)
                            Spacer(Modifier.width(6.dp))
                            Text(
                                when {
                                    isDownloading -> "下载中..."
                                    aiRedrawDownloading -> "AI 重绘模型下载中..."
                                    else -> "从链接下载并应用"
                                }
                            )
                        }
                        // ★ AI 重绘模型（1GB 级）下载进度：镜像 / 进度条 / 速度
                        if (aiRedrawDownloading) {
                            Column(
                                modifier = Modifier.fillMaxWidth().padding(top = 4.dp),
                                verticalArrangement = Arrangement.spacedBy(4.dp)
                            ) {
                                if (aiRedrawDlMirror.isNotEmpty()) {
                                    Text("镜像: ${aiRedrawDlMirror}", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                }
                                val pct = if (aiRedrawDlTotal > 0) (aiRedrawDlProgress.toFloat() / aiRedrawDlTotal * 100).toInt() else 0
                                LinearProgressIndicator(
                                    progress = { if (aiRedrawDlTotal > 0) aiRedrawDlProgress.toFloat() / aiRedrawDlTotal else 0f },
                                    modifier = Modifier.fillMaxWidth().height(6.dp),
                                    color = Color(0xFF7B1FA2)
                                )
                                Text(
                                    "${formatSize(aiRedrawDlProgress)} / ${formatSize(aiRedrawDlTotal)} ($pct%) · ${formatSize(aiRedrawDlSpeed)}/s",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                        HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f))
                        // ---- HF Token 输入（下载 gated 模型如 RMBG-2.0 / 私有直链）----
                        OutlinedTextField(
                            value = hfTokenInput,
                            onValueChange = { hfTokenInput = it },
                            label = { Text("HF Token（可选，gated/私有模型需要）") },
                            placeholder = { Text("hf_xxx") },
                            singleLine = true,
                            modifier = Modifier.fillMaxWidth()
                        )
                        Button(
                            onClick = {
                                ModelManager.hfToken = hfTokenInput.trim()
                                Prefs.hfToken = hfTokenInput.trim()
                                Toast.makeText(context, "HF Token 已保存", Toast.LENGTH_SHORT).show()
                            },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Icon(Icons.Filled.Check, contentDescription = null)
                            Spacer(Modifier.width(6.dp))
                            Text("保存 Token")
                        }
                    } else {
                        Text(
                            "模型配置：链接 + Token（点击展开编辑）",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            }

            // ===== SD 模型（AI 重绘引擎）=====
            // ★ Lite 精简版：AI 重绘固定走 LocalDream App（复用其模型），无本地引擎/模型包，
            //   整张卡片隐藏（不显示 SD 模型包导入/部署入口）
            if (!BuildConfig.IS_LITE) {
            Card(
                colors = CardDefaults.cardColors(containerColor = Color(0xFFFFFFFF)),
                shape = RoundedCornerShape(16.dp)
            ) {
                Column(
                    modifier = Modifier.padding(14.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Filled.AutoAwesome, contentDescription = null, tint = Color(0xFF7B1FA2))
                        Spacer(Modifier.width(8.dp))
                        Text(
                            "SD 模型（AI 重绘引擎）",
                            style = MaterialTheme.typography.titleSmall,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.weight(1f)
                        )
                    }
                    Text(
                        if (aiEngineStatus.isNotEmpty()) aiEngineStatus
                        else "选择 SD 模型包 zip（animemix / AnythingV5 / MeinaMix 等）。重绘时自动部署启动，用完后自动停止。",
                        style = MaterialTheme.typography.bodySmall,
                        color = if (aiEngineStatus.contains("失败") || aiEngineStatus.contains("❌") || aiEngineStatus.contains("超时") || aiEngineStatus.contains("退出") || aiEngineStatus.contains("不存在") || aiEngineStatus.contains("缺失")) Color(0xFFC62828) else if (aiEngineStatus.isEmpty()) MaterialTheme.colorScheme.onSurfaceVariant else Color(0xFF2E7D32)
                    )
                    // 选包按钮（全宽，唯一手动操作）
                    OutlinedButton(
                        onClick = { aiModelZipPicker.launch("*/*") },
                        enabled = !aiEngineBusy && !aiRedrawDownloading,
                        modifier = Modifier.fillMaxWidth().height(44.dp),
                        shape = RoundedCornerShape(10.dp)
                    ) {
                        Icon(Icons.Filled.Archive, contentDescription = null, modifier = Modifier.size(16.dp))
                        Spacer(Modifier.width(6.dp))
                        Text(
                            if (aiModelZipReady) {
                                val cur = AiRedrawEngine.activeModelId.ifBlank { Prefs.aiRedrawModelId }
                                if (AiRedrawEngine.modelZipFile?.exists() == true)
                                    "✅ 模型包已选（${AiRedrawEngine.modelZipFile?.let { formatSize(it.length()) } ?: ""}）"
                                else if (cur.isNotEmpty()) "✅ 已部署：$cur"
                                else "✅ 模型已就绪"
                            } else "选择 SD 模型包 zip",
                            fontSize = 14.sp, maxLines = 1
                        )
                    }
                    // ★ 云端一键下载 animemix（HF 云下载入口）：点击即从 Prefs.aiRedrawHfUrl 拉取并自动部署激活
                    //   复用 downloadModelUrl() 完整链路（下载→probeZipKind→deployModel→已部署列表刷新）
                    OutlinedButton(
                        onClick = {
                            if (aiRedrawDownloading) return@OutlinedButton
                            // 预填云下载直链（animemix 默认直链），未配置时写入 Prefs 持久化
                            val url = Prefs.aiRedrawHfUrl.ifBlank {
                                "https://huggingface.co/zuimengqm/sd1.5-qnn/resolve/main/animemix.zip"
                            }
                            Prefs.aiRedrawHfUrl = url
                            modelUrlInput = url
                            downloadModelUrl()
                        },
                        enabled = !aiEngineBusy && !aiRedrawDownloading,
                        modifier = Modifier.fillMaxWidth().height(44.dp),
                        shape = RoundedCornerShape(10.dp),
                        colors = ButtonDefaults.outlinedButtonColors(contentColor = Color(0xFF1565C0))
                    ) {
                        if (aiRedrawDownloading) {
                            CircularProgressIndicator(
                                modifier = Modifier.size(18.dp),
                                strokeWidth = 2.dp,
                                color = Color(0xFF1565C0)
                            )
                            Spacer(Modifier.width(6.dp))
                            Text(
                                "下载中 ${if (aiRedrawDlTotal > 0) "${(aiRedrawDlProgress * 100 / aiRedrawDlTotal)}%" else ""}",
                                fontSize = 14.sp, maxLines = 1
                            )
                        } else {
                            Icon(Icons.Filled.CloudDownload, contentDescription = null, modifier = Modifier.size(16.dp))
                            Spacer(Modifier.width(6.dp))
                            Text("云端下载 animemix 模型（约 1.2GB）", fontSize = 14.sp)
                        }
                    }
                    if (aiRedrawDownloading && aiRedrawDlTotal > 0) {
                        LinearProgressIndicator(
                            progress = { (aiRedrawDlProgress.toFloat() / aiRedrawDlTotal.toFloat()).coerceIn(0f, 1f) },
                            modifier = Modifier.fillMaxWidth().height(6.dp),
                            color = Color(0xFF1565C0),
                            trackColor = Color(0xFF1565C0).copy(alpha = 0.15f)
                        )
                        if (aiRedrawDlSpeed > 0) {
                            Text(
                                "速度 ${formatSpeed(aiRedrawDlSpeed)} / ${formatSize(aiRedrawDlProgress)} / ${formatSize(aiRedrawDlTotal)}",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                    // ★ 已部署模型列表（多模型切换：zip 部署后自动删除，但解压目录保留，可随时切回/删除）
                    //   复用 Tab 1 顶部缓存，避免每次重组 listFiles 扫 1GB+ 目录卡顿
                    val deployedList = deployedModelCache
                    if (deployedList.isNotEmpty()) {
                        Text("已部署模型（点击切换）", style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.Bold, color = Color(0xFF7B1FA2))
                        val currentId = AiRedrawEngine.activeModelId.ifBlank { Prefs.aiRedrawModelId }
                        deployedList.forEach { id ->
                            val isCurrent = id == currentId
                            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                OutlinedButton(
                                    onClick = {
                                        if (id != currentId && !aiEngineBusy) {
                                            val ok = AiRedrawEngine.selectDeployedModel(context, id)
                                            if (ok) {
                                                aiModelZipReady = AiRedrawEngine.isModelReady(context)
                                                deployedModelsVersion++
                                                aiEngineStatus = "✅ 已切换到模型: $id"
                                            }
                                        }
                                    },
                                    enabled = !aiEngineBusy && !aiRedrawDownloading,
                                    modifier = Modifier.weight(1f).heightIn(min = 40.dp),
                                    shape = RoundedCornerShape(10.dp),
                                    colors = ButtonDefaults.outlinedButtonColors(
                                        contentColor = if (isCurrent) Color(0xFF2E7D32) else Color(0xFF7B1FA2),
                                        containerColor = if (isCurrent) Color(0xFF2E7D32).copy(alpha = 0.08f) else Color.Transparent
                                    )
                                ) {
                                    Icon(
                                        if (isCurrent) Icons.Filled.CheckCircle else Icons.Filled.Storage,
                                        contentDescription = null, modifier = Modifier.size(16.dp),
                                        tint = if (isCurrent) Color(0xFF2E7D32) else Color(0xFF7B1FA2)
                                    )
                                    Spacer(Modifier.width(6.dp))
                                    // ★ 完整显示模型名：不限制行数，超长文件名可自动换行（否则 maxLines=1 截断看不见）
                                    Text(if (isCurrent) "$id（当前）" else id, fontSize = 13.sp)
                                }
                                // ★ 删除该已部署模型（释放约 1GB 空间；删当前模型则清空状态）
                                //   样式与抠图模型删除一致：IconButton 红色垃圾桶
                                IconButton(
                                    onClick = {
                                        if (aiEngineBusy) return@IconButton
                                        scope.launch {
                                            val ok = withContext(Dispatchers.IO) {
                                                AiRedrawEngine.deleteDeployedModel(context, id)
                                            }
                                            if (ok) {
                                                aiModelZipReady = AiRedrawEngine.isModelReady(context)
                                                deployedModelsVersion++
                                                aiEngineStatus = "🗑 已删除模型: $id"
                                                Toast.makeText(context, "已删除模型 $id", Toast.LENGTH_SHORT).show()
                                            } else {
                                                Toast.makeText(context, "删除失败", Toast.LENGTH_SHORT).show()
                                            }
                                        }
                                    },
                                    enabled = !aiEngineBusy && !aiRedrawDownloading,
                                    modifier = Modifier.size(36.dp)
                                ) {
                                    Icon(
                                        Icons.Filled.Delete,
                                        contentDescription = "删除 $id",
                                        tint = MaterialTheme.colorScheme.error,
                                        modifier = Modifier.size(18.dp)
                                    )
                                }
                            }
                        }
                        HorizontalDivider(modifier = Modifier.padding(vertical = 2.dp))
                    }
                }
            }
            } // ★ Lite 精简版：SD 模型卡片结束

            } // ---- Tab 1 结束 ----

            // ===== Tab 3: 设置 =====
            if (currentTab == 3) {
            // ===== 通知保活 =====
            Card(
                colors = CardDefaults.cardColors(containerColor = Color(0xFFFFFFFF)),
                shape = RoundedCornerShape(16.dp)
            ) {
                Column(
                    modifier = Modifier.padding(14.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Filled.Notifications, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                        Spacer(Modifier.width(8.dp))
                        Text("后台保活", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
                    }
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text("常驻通知保活", style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.Bold)
                            Text(
                                "前台服务+通知，防止模型加载后后台被系统回收",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                        Switch(
                            checked = keepAlive,
                            onCheckedChange = {
                                keepAlive = it
                                Prefs.keepAlive = it
                                if (it) {
                                    try { KeepAliveService.start(context) } catch (_: Exception) {}
                                    Toast.makeText(context, "后台保活已开启", Toast.LENGTH_SHORT).show()
                                } else {
                                    try { KeepAliveService.stop(context) } catch (_: Exception) {}
                                    Toast.makeText(context, "后台保活已关闭", Toast.LENGTH_SHORT).show()
                                }
                            }
                        )
                    }
                }
            }

            // ===== 自动保存 =====
            Card(
                colors = CardDefaults.cardColors(containerColor = Color(0xFFFFFFFF)),
                shape = RoundedCornerShape(16.dp)
            ) {
                Column(
                    modifier = Modifier.padding(14.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Filled.Save, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                        Spacer(Modifier.width(8.dp))
                        Text("自动保存", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
                    }
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text("自动保存抠图结果", style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.Bold)
                            Text(
                                if (autoSaveResult)
                                    "抠图完成后自动把 原图→结果 存入历史，无需手动保存"
                                else "关闭后抠图结果仅留在当前界面，不自动入历史",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                        Switch(
                            checked = autoSaveResult,
                            onCheckedChange = {
                                autoSaveResult = it
                                Prefs.autoSaveResult = it
                                Toast.makeText(context, if (it) "已开启自动保存" else "已关闭自动保存", Toast.LENGTH_SHORT).show()
                            }
                        )
                    }
                }
            }

            // ===== 通知权限 =====
            Card(
                colors = CardDefaults.cardColors(containerColor = Color(0xFFFFFFFF)),
                shape = RoundedCornerShape(16.dp)
            ) {
                Column(
                    modifier = Modifier.padding(14.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Filled.NotificationsActive, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                        Spacer(Modifier.width(8.dp))
                        Text("通知权限", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
                    }
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text(
                                if (Notifications.hasPermission(context)) "已授予通知权限" else "未授予通知权限",
                                style = MaterialTheme.typography.bodyLarge,
                                fontWeight = FontWeight.Bold,
                                color = if (Notifications.hasPermission(context)) Color(0xFF2E7D32) else MaterialTheme.colorScheme.error
                            )
                            Text(
                                "抠图完成时通知提醒；保活通知也需要此权限",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                        if (!Notifications.hasPermission(context)) {
                            Button(onClick = {
                                notifPermLauncher.launch(android.Manifest.permission.POST_NOTIFICATIONS)
                            }) {
                                Text("申请")
                            }
                        }
                    }
                }
            }

            // ===== ONNX 加速设置 =====
            Card(
                colors = CardDefaults.cardColors(containerColor = Color(0xFFFFFFFF)),
                shape = RoundedCornerShape(16.dp)
            ) {
                Column(
                    modifier = Modifier.padding(14.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Filled.Bolt, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                        Spacer(Modifier.width(8.dp))
                        Text("ONNX 加速", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
                    }
                    // NNAPI 硬件加速开关
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text("NNAPI 硬件加速", style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.Bold)
                            Text(
                                "调用 NPU/DSP 加速推理（不兼容时自动回退 CPU）",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                        Switch(
                            checked = enableNnapi,
                            onCheckedChange = {
                                enableNnapi = it
                                Prefs.enableNnapi = it
                                // 加速设置变更后需重建引擎
                                engine?.close()
                                engine = null
                                Toast.makeText(context, if (it) "已启用 NNAPI，下次抠图生效" else "已关闭 NNAPI", Toast.LENGTH_SHORT).show()
                            }
                        )
                    }
                    // QNN (Hexagon DSP) 加速开关
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text("QNN 加速（骁龙 DSP）", style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.Bold)
                            Text(
                                "Qualcomm Hexagon HTP 硬件加速；需带 QNN EP 的 ORT + QNN SDK 库",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                        Switch(
                            checked = enableQnn,
                            onCheckedChange = {
                                enableQnn = it
                                Prefs.enableQnn = it
                                engine?.close()
                                engine = null
                                Toast.makeText(context, if (it) "已启用 QNN，下次抠图生效" else "已关闭 QNN", Toast.LENGTH_SHORT).show()
                            }
                        )
                    }
                    // 当前引擎 NNAPI/QNN 实际状态（加载后反馈）
                    val engState = engine
                    if (engState != null) {
                        Text(
                            text = "引擎状态: ${engState.nnapiStatus} · ${engState.qnnStatus}",
                            style = MaterialTheme.typography.labelSmall,
                            color = if (engState.nnapiActive || engState.qnnActive) Color(0xFF2E7D32) else MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                    // ---- 精细抠图（大图分块）----
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text("精细抠图（大图分块）", style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.Bold)
                            Text(
                                ">1024px 图片自动分块推理保留细节，内存低时建议关闭",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                        Switch(
                            checked = tileMode,
                            onCheckedChange = {
                                tileMode = it
                                Prefs.tileMode = it
                                Toast.makeText(context, if (it) "精细抠图已开启（大图自动分块）" else "精细抠图已关闭", Toast.LENGTH_SHORT).show()
                            }
                        )
                    }
                    // 极速模式（与精细抠图并存不冲突）
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text("极速模式", style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.Bold)
                            Text(
                                "缩小分辨率快速抠图，速度优先；与精细抠图不冲突",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                        Switch(
                            checked = turboMode,
                            onCheckedChange = {
                                turboMode = it
                                Prefs.turboMode = it
                                Toast.makeText(context, if (it) "极速模式已开启（缩分辨率加速）" else "极速模式已关闭", Toast.LENGTH_SHORT).show()
                            }
                        )
                    }
                    // CPU 线程数
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text("CPU 线程数", style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.Bold, modifier = Modifier.weight(1f))
                        Text("$onnxThreads", style = MaterialTheme.typography.titleMedium, color = MaterialTheme.colorScheme.primary, fontWeight = FontWeight.Bold)
                    }
                    Slider(
                        value = onnxThreads.toFloat(),
                        onValueChange = {
                            onnxThreads = it.toInt().coerceIn(1, 8); Prefs.onnxThreads = onnxThreads
                        },
                        valueRange = 1f..8f,
                        steps = 6
                    )
                    Text(
                        "线程越多推理越快，但内存占用更高；设备内存不足时建议降到 2",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    // ---- 加速检测 ----
                    var accelResult by remember { mutableStateOf(Prefs.nnapiDetectResult) }
                    var accelDetecting by remember { mutableStateOf(false) }
                    OutlinedButton(
                        onClick = {
                            scope.launch {
                                accelDetecting = true
                                accelResult = "检测中..."
                                val r = withContext(Dispatchers.IO) {
                                    AccelDetector.detect(context)
                                }
                                accelResult = AccelDetector.formatResult(r)
                                Prefs.nnapiDetectResult = accelResult
                                accelDetecting = false
                            }
                        },
                        modifier = Modifier.fillMaxWidth(),
                        enabled = !accelDetecting
                    ) {
                        if (accelDetecting) {
                            CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                            Spacer(Modifier.width(8.dp))
                            Text("检测中...")
                        } else {
                            Icon(Icons.Filled.Speed, contentDescription = null)
                            Spacer(Modifier.width(8.dp))
                            Text("检测加速能力")
                        }
                    }
                    if (accelResult.isNotEmpty() && !accelDetecting) {
                        Card(
                            colors = CardDefaults.cardColors(containerColor = Color(0xFFF0F4FF)),
                            shape = RoundedCornerShape(10.dp)
                        ) {
                            Text(
                                accelResult,
                                style = MaterialTheme.typography.bodySmall,
                                modifier = Modifier.padding(12.dp)
                            )
                        }
                    }
                }
            }

            // ===== AI 重绘预设（设置页提前预设，查看器一键执行）=====
            Card(
                colors = CardDefaults.cardColors(containerColor = Color(0xFFFFFFFF)),
                shape = RoundedCornerShape(16.dp)
            ) {
                Column(
                    modifier = Modifier.padding(14.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    // ★ 可折叠标题行：点击整行折叠/展开
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(12.dp))
                            .selectable(selected = showAiRedrawPreset, onClick = { showAiRedrawPreset = !showAiRedrawPreset })
                            .padding(vertical = 2.dp)
                    ) {
                        Icon(Icons.Filled.AutoAwesome, contentDescription = null, tint = Color(0xFF7B1FA2))
                        Spacer(Modifier.width(8.dp))
                        Text("AI 重绘预设", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold, modifier = Modifier.weight(1f))
                        Text(
                            if (showAiRedrawPreset) "收起 ▲" else "展开 ▼",
                            style = MaterialTheme.typography.labelSmall,
                            color = Color(0xFF7B1FA2)
                        )
                    }
                    if (showAiRedrawPreset) {
                    // ★ 恢复默认收纳在展开内容里（折叠后不再显示，保持标题行干净）
                    TextButton(
                        onClick = {
                            Prefs.aiRedrawPrompt = "keep original, sharp details, high quality"
                            Prefs.aiRedrawNegative = "blurry, low quality, watermark, deformed"
                            Prefs.aiRedrawSteps = 20
                            Prefs.aiRedrawCfg = 6.5f
                            Prefs.aiRedrawDenoise = 0.35f
                            Prefs.aiRedrawWidth = 512
                            Prefs.aiRedrawHeight = 512
                            Prefs.aiRedrawScheduler = "dpm_sde"
                            aiRedrawPrompt = Prefs.aiRedrawPrompt
                            aiRedrawNegative = Prefs.aiRedrawNegative
                            aiRedrawSteps = Prefs.aiRedrawSteps
                            aiRedrawCfg = Prefs.aiRedrawCfg
                            aiRedrawDenoise = Prefs.aiRedrawDenoise
                            aiRedrawWidth = Prefs.aiRedrawWidth
                            aiRedrawHeight = Prefs.aiRedrawHeight
                            aiRedrawScheduler = Prefs.aiRedrawScheduler
                            Toast.makeText(context, "已恢复默认预设（修复向微调）", Toast.LENGTH_SHORT).show()
                        },
                        modifier = Modifier.align(Alignment.End)
                    ) {
                        Text("恢复默认", style = MaterialTheme.typography.labelMedium, color = Color(0xFF7B1FA2))
                    }
                    Text(
                        "查看器点「AI 重绘」即按此预设一键修复微调（img2img，非生图）：自动部署启动引擎→重绘→完成自动停止。",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    // ---- 推理后端选择（本地自加载引擎 / LocalDream App）----
                    // ★ Lite 精简版：未打包本地引擎资产，固定走 LocalDream，不显示本地引擎选项
                    Text("推理后端", style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.Bold)
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        if (!BuildConfig.IS_LITE) {
                            FilterChip(
                                selected = aiRedrawBackend == "local",
                                onClick = { aiRedrawBackend = "local"; Prefs.aiRedrawBackend = "local" },
                                label = { Text("本地自加载引擎", style = MaterialTheme.typography.labelSmall) }
                            )
                        }
                        FilterChip(
                            selected = if (BuildConfig.IS_LITE) true else aiRedrawBackend == "dream",
                            onClick = { aiRedrawBackend = "dream"; Prefs.aiRedrawBackend = "dream" },
                            label = { Text("LocalDream App", style = MaterialTheme.typography.labelSmall) }
                        )
                    }
                    Text(
                        if (BuildConfig.IS_LITE)
                            "精简版固定复用已安装的 LocalDream App（需先在 LocalDream 内就绪，App 不自部署模型）。"
                        else if (aiRedrawBackend == "local")
                            "本 App 内启动 Stable Diffusion 引擎（需已选 SD 模型包，用完全自动停止）。"
                        else
                            "复用已安装的 LocalDream App（需先在 LocalDream 内就绪，本 App 不再自部署模型）。",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    // ---- 提示词 ----
                    Text("正向提示词（修复向）", style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.Bold)
                    OutlinedTextField(
                        value = aiRedrawPrompt,
                        onValueChange = { aiRedrawPrompt = it; Prefs.aiRedrawPrompt = it },
                        modifier = Modifier.fillMaxWidth(),
                        textStyle = MaterialTheme.typography.bodySmall,
                        maxLines = 3
                    )
                    // ---- 负向提示词 ----
                    Text("负向提示词", style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.Bold)
                    OutlinedTextField(
                        value = aiRedrawNegative,
                        onValueChange = { aiRedrawNegative = it; Prefs.aiRedrawNegative = it },
                        modifier = Modifier.fillMaxWidth(),
                        textStyle = MaterialTheme.typography.bodySmall,
                        maxLines = 3
                    )
                    // ---- 步数 / CFG ----
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                        Column(Modifier.weight(1f)) {
                            Text("采样步数：$aiRedrawSteps", style = MaterialTheme.typography.bodyMedium)
                            Slider(
                                value = aiRedrawSteps.toFloat(),
                                onValueChange = { aiRedrawSteps = it.toInt().coerceIn(1, 50); Prefs.aiRedrawSteps = aiRedrawSteps },
                                valueRange = 1f..50f,
                                steps = 48
                            )
                        }
                        Column(Modifier.weight(1f)) {
                            Text("CFG 引导：${"%.1f".format(aiRedrawCfg)}", style = MaterialTheme.typography.bodyMedium)
                            Slider(
                                value = aiRedrawCfg,
                                onValueChange = { aiRedrawCfg = it; Prefs.aiRedrawCfg = it },
                                valueRange = 1f..15f,
                                steps = 27
                            )
                        }
                    }
                    // ---- 重绘强度（denoise）----
                    Text("重绘强度：${"%.2f".format(aiRedrawDenoise)}（低=微调修复，高=大改）", style = MaterialTheme.typography.bodyMedium)
                    Slider(
                        value = aiRedrawDenoise,
                        onValueChange = { aiRedrawDenoise = it; Prefs.aiRedrawDenoise = it },
                        valueRange = 0.1f..1f,
                        steps = 8
                    )
                    Text(
                        "默认 0.45：仅修复模糊/噪点/伪影，保持原图结构。想大幅重绘可调高到 0.7+。",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    // ---- 输出分辨率（模型包档位，非自定义）----
                    Text("输出分辨率（模型包档位）", style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.Bold)
                    Text(
                        "QNN 模型为编译时固定形状，仅支持以下档位（与模型包 patch 对应），不能自定义任意尺寸。",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    val resChoices = listOf(
                        "512×512", "512×768", "768×512", "768×768",
                        "768×1024", "1024×768", "1024×1024"
                    )
                    // ★ 修复：每行 chip 用 weight(1f) 均分宽度，避免窄屏超宽把 768×768 等档位挤出可视区
                    resChoices.chunked(4).forEach { row ->
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            row.forEach { c ->
                                val (w, h) = c.split("×").let { it[0].toInt() to it[1].toInt() }
                                FilterChip(
                                    selected = aiRedrawWidth == w && aiRedrawHeight == h,
                                    onClick = {
                                        aiRedrawWidth = w
                                        aiRedrawHeight = h
                                        Prefs.aiRedrawWidth = w
                                        Prefs.aiRedrawHeight = h
                                    },
                                    modifier = Modifier.weight(1f),
                                    label = { Text(c, style = MaterialTheme.typography.labelSmall, maxLines = 1) }
                                )
                            }
                        }
                    }
                    Text(
                        "已选：${aiRedrawWidth}×${aiRedrawHeight}（超过原图尺寸会自动放大到该档位）",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    // ---- 调度器 ----
                    Text("采样调度器", style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.Bold)
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        listOf("dpm_sde", "euler", "ddim", "pndm", "lms").forEach { s ->
                            FilterChip(
                                selected = aiRedrawScheduler == s,
                                onClick = { aiRedrawScheduler = s; Prefs.aiRedrawScheduler = s },
                                modifier = Modifier.weight(1f),
                                label = { Text(s, style = MaterialTheme.typography.labelSmall, maxLines = 1) }
                            )
                        }
                    }
                    }
                }
            }

            // ===== 存储占用检查 =====
            Card(
                colors = CardDefaults.cardColors(containerColor = Color(0xFFFFFFFF)),
                shape = RoundedCornerShape(16.dp)
            ) {
                Column(
                    modifier = Modifier.padding(14.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Filled.FolderOpen, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                        Spacer(Modifier.width(8.dp))
                        Text("存储占用", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold, modifier = Modifier.weight(1f))
                    }
                    Text(
                        "统计模型、历史记录、App 缓存等目录的磁盘占用。",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    OutlinedButton(
                        onClick = {
                            scope.launch {
                                storageChecking = true
                                storageReport = "正在统计..."
                                val r = withContext(Dispatchers.IO) { checkStorageUsage(context) }
                                storageReport = r
                                storageChecking = false
                            }
                        },
                        modifier = Modifier.fillMaxWidth(),
                        enabled = !storageChecking
                    ) {
                        if (storageChecking) {
                            CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                            Spacer(Modifier.width(8.dp))
                            Text("统计中...")
                        } else {
                            Icon(Icons.Filled.Storage, contentDescription = null)
                            Spacer(Modifier.width(8.dp))
                            Text("检查存储占用")
                        }
                    }
                    if (storageReport.isNotEmpty() && !storageChecking) {
                        Card(
                            colors = CardDefaults.cardColors(containerColor = Color(0xFFF0F4FF)),
                            shape = RoundedCornerShape(10.dp)
                        ) {
                            Text(
                                storageReport,
                                style = MaterialTheme.typography.bodySmall,
                                fontFamily = FontFamily.Monospace,
                                modifier = Modifier.padding(12.dp)
                            )
                        }
                    }
                    OutlinedButton(
                        onClick = {
                            scope.launch {
                                storageChecking = true
                                storageReport = "正在清理..."
                                val freed = withContext(Dispatchers.IO) { clearAppCache(context) }
                                storageReport = "已清理缓存，释放 ${formatSize(freed)}"
                                // 清理后重新统计展示
                                val r = withContext(Dispatchers.IO) { checkStorageUsage(context) }
                                storageReport = "已清理缓存，释放 ${formatSize(freed)}\n\n$r"
                                storageChecking = false
                            }
                        },
                        modifier = Modifier.fillMaxWidth(),
                        enabled = !storageChecking
                    ) {
                        Icon(Icons.Filled.DeleteSweep, contentDescription = null)
                        Spacer(Modifier.width(8.dp))
                        Text("清理缓存")
                    }
                }
            }

            // ===== 关于 =====
            Card(
                colors = CardDefaults.cardColors(containerColor = Color(0xFFFFFFFF)),
                shape = RoundedCornerShape(16.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .combinedClickable(
                        onClick = { showAboutPage = true }
                    )
            ) {
                Column(
                    modifier = Modifier.padding(14.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Filled.Info, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                        Spacer(Modifier.width(8.dp))
                        Text("关于", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
                        Spacer(Modifier.weight(1f))
                        Icon(Icons.Filled.ChevronRight, contentDescription = "进入关于页", tint = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
            }

            } // ---- Tab 3 结束 ----

            // ===== Tab 2: 历史 =====
            if (currentTab == 2) {
                // ★ rawHistory 已在 Tab 块外预加载（produceState 不随 Tab 切换销毁）
                // ★ 历史排序：0=时间最新 1=时间最旧 2=耗时升序 3=耗时降序 4=模型名 5=文件最小 6=文件最大
                val rawList = rawHistory ?: emptyList()
                val historyItems = when (historySort) {
                    1 -> rawList.sortedBy { it.timestamp }                       // 最旧在前
                    2 -> rawList.sortedBy { it.durationMs }                       // 耗时最短在前
                    3 -> rawList.sortedByDescending { it.durationMs }             // 耗时最长在前
                    4 -> rawList.sortedBy { it.modelName }                        // 模型名 A-Z
                    5 -> rawList.sortedBy { it.fileSize }                         // 文件最小在前
                    6 -> rawList.sortedByDescending { it.fileSize }               // 文件最大在前
                    else -> rawList.sortedByDescending { it.timestamp }           // 最新在前（默认）
                }
                // ★ 首次加载中占位
                if (rawHistory == null) {
                    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth().padding(24.dp)) {
                        CircularProgressIndicator(modifier = Modifier.size(22.dp), strokeWidth = 2.dp)
                        Spacer(Modifier.width(12.dp))
                        Text("历史加载中…", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
                Card(
                    colors = CardDefaults.cardColors(containerColor = Color(0xFFFFFFFF)),
                    shape = RoundedCornerShape(16.dp)
                ) {
                    Column(
                        modifier = Modifier.padding(14.dp),
                        verticalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(Icons.Filled.History, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                            Spacer(Modifier.width(8.dp))
                            Text(
                                if (historySelectedIds.isNotEmpty()) "已选 ${historySelectedIds.size} 条"
                                else "抠图历史",
                                style = MaterialTheme.typography.titleSmall,
                                fontWeight = FontWeight.Bold,
                                modifier = Modifier.weight(1f)
                            )
                            // ★ 多选模式：进入/退出（长按记录也可进入；批量保存/删除用）
                            if (historySelectedIds.isNotEmpty()) {
                                TextButton(onClick = { historySelectedIds = emptySet() }) {
                                    Icon(Icons.Filled.Close, contentDescription = null, modifier = Modifier.size(16.dp))
                                    Spacer(Modifier.width(2.dp))
                                    Text("取消")
                                }
                            } else {
                                // ★ 视图切换：列表 / 网格
                                IconButton(
                                    onClick = {
                                        historyView = if (historyView == 0) 1 else 0
                                        Prefs.historyView = historyView
                                    }
                                ) {
                                    Icon(
                                        if (historyView == 0) Icons.Filled.GridView else Icons.Filled.List,
                                        contentDescription = if (historyView == 0) "切换网格" else "切换列表",
                                        tint = MaterialTheme.colorScheme.primary
                                    )
                                }
                                // ★ 一键收起 / 展开：把所有历史分组整体折叠或展开（列表视图）
                                TextButton(
                                    onClick = {
                                        // 从 historyItems 推导所有分组 key（与下方 groups 分组规则一致）
                                        val allKeys = historyItems.map {
                                            it.originalKey.ifBlank { "item_${it.id}" }
                                        }.toSet()
                                        // 若当前有任意组处于展开，则一键全收起；否则一键全展开
                                        val anyExpanded = allKeys.any { !collapsedHistoryGroups.contains(it) }
                                        collapsedHistoryGroups =
                                            if (anyExpanded) allKeys else emptySet()
                                        Prefs.historyCollapsedGroups = collapsedHistoryGroups.joinToString(",")
                                    },
                                    enabled = historyItems.isNotEmpty()
                                ) {
                                    val anyExpanded = historyItems.any { !collapsedHistoryGroups.contains(it.originalKey.ifBlank { "item_${it.id}" }) }
                                    Icon(
                                        if (anyExpanded) Icons.Filled.UnfoldLess else Icons.Filled.UnfoldMore,
                                        contentDescription = null,
                                        modifier = Modifier.size(16.dp),
                                        tint = MaterialTheme.colorScheme.primary
                                    )
                                    Spacer(Modifier.width(4.dp))
                                    Text(if (anyExpanded) "一键收起" else "全部展开")
                                }
                                TextButton(
                                    onClick = {
                                        HistoryManager.clearAll(context)
                                        HistoryManager.invalidateHistoryCache()
                                        historyVersion++
                                    },
                                    enabled = historyItems.isNotEmpty()
                                ) {
                                    Icon(Icons.Filled.DeleteSweep, contentDescription = null, modifier = Modifier.size(16.dp))
                                    Spacer(Modifier.width(4.dp))
                                    Text("清空")
                                }
                            }
                        }
                        // ★ 多选操作栏（选中时显示）：全选 / 批量保存 / 批量删除
                        //   ⚠ 三个按钮按 weight(1f) 均分宽度铺满一行，窄屏也不溢出（文字单行省略）
                        if (historySelectedIds.isNotEmpty()) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(6.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                TextButton(
                                    onClick = {
                                        historySelectedIds = if (historySelectedIds.size == historyItems.size)
                                            emptySet() else historyItems.map { it.id }.toSet()
                                    },
                                    enabled = historyItems.isNotEmpty(),
                                    modifier = Modifier.weight(1f),
                                    contentPadding = PaddingValues(horizontal = 4.dp, vertical = 6.dp)
                                ) {
                                    Icon(
                                        if (historySelectedIds.size == historyItems.size) Icons.Filled.Deselect else Icons.Filled.SelectAll,
                                        contentDescription = null,
                                        modifier = Modifier.size(16.dp)
                                    )
                                    Spacer(Modifier.width(2.dp))
                                    Text(
                                        if (historySelectedIds.size == historyItems.size) "取消全选" else "全选",
                                        maxLines = 1,
                                        overflow = TextOverflow.Ellipsis
                                    )
                                }
                                // ★ 批量保存：选中记录的结果图存入相册 rmbg 文件夹
                                OutlinedButton(
                                    onClick = {
                                        val selected = historyItems.filter { historySelectedIds.contains(it.id) }
                                        if (selected.isEmpty()) return@OutlinedButton
                                        scope.launch {
                                            historyBatchSaving = true
                                            var okCount = 0
                                            val total = selected.size
                                            withContext(Dispatchers.IO) {
                                                selected.forEachIndexed { idx, item ->
                                                    val bmp = HistoryManager.loadResult(context, item)
                                                    if (saveToRmbgFolder(context, bmp)) okCount++
                                                    try { bmp?.recycle() } catch (_: Exception) {}
                                                }
                                            }
                                            historyBatchSaving = false
                                            historySelectedIds = emptySet()
                                            Toast.makeText(
                                                context,
                                                "已保存 $okCount/$total 到相册 rmbg 文件夹",
                                                Toast.LENGTH_LONG
                                            ).show()
                                        }
                                    },
                                    enabled = !historyBatchSaving,
                                    modifier = Modifier.weight(1f),
                                    contentPadding = PaddingValues(horizontal = 4.dp, vertical = 6.dp)
                                ) {
                                    if (historyBatchSaving) {
                                        CircularProgressIndicator(modifier = Modifier.size(14.dp), strokeWidth = 2.dp)
                                    } else {
                                        Icon(Icons.Filled.Save, contentDescription = null, modifier = Modifier.size(16.dp))
                                    }
                                    Spacer(Modifier.width(2.dp))
                                    Text(
                                        "批量保存",
                                        maxLines = 1,
                                        overflow = TextOverflow.Ellipsis
                                    )
                                }
                                // ★ 批量删除：确认弹窗后删除选中记录（与批量保存同为描边按钮，风格一致）
                                OutlinedButton(
                                    onClick = {
                                        val selected = historyItems.filter { historySelectedIds.contains(it.id) }
                                        if (selected.isNotEmpty()) pendingDeleteGroup = selected
                                    },
                                    enabled = !historyBatchSaving,
                                    modifier = Modifier.weight(1f),
                                    contentPadding = PaddingValues(horizontal = 4.dp, vertical = 6.dp)
                                ) {
                                    Icon(Icons.Filled.Delete, contentDescription = null, modifier = Modifier.size(16.dp), tint = MaterialTheme.colorScheme.error)
                                    Spacer(Modifier.width(2.dp))
                                    Text(
                                        "批量删除",
                                        color = MaterialTheme.colorScheme.error,
                                        maxLines = 1,
                                        overflow = TextOverflow.Ellipsis
                                    )
                                }
                            }
                        }
                        // ★ 历史排序方式选择
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .horizontalScroll(rememberScrollState()),
                            horizontalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            listOf(
                                0 to "时间最新", 1 to "时间最旧",
                                2 to "耗时最短", 3 to "耗时最长",
                                4 to "模型名", 5 to "文件最小", 6 to "文件最大"
                            ).forEach { (mode, label) ->
                                FilterChip(
                                    selected = historySort == mode,
                                    onClick = {
                                        historySort = mode
                                        Prefs.historySort = mode
                                    },
                                    label = { Text(label, style = MaterialTheme.typography.labelSmall) }
                                )
                            }
                        }
                        // ★ 历史分组：同一 originalKey（同一原图）归纳为一组折叠记录；无 key 的记录各自独立成组
                        val groups = remember(historyItems, historyVersion) {
                            val ordered = mutableListOf<Pair<String, List<HistoryManager.HistoryItem>>>()
                            val byKey = LinkedHashMap<String, MutableList<HistoryManager.HistoryItem>>()
                            historyItems.forEach { it ->
                                val k = it.originalKey.ifBlank { "item_${it.id}" }
                                byKey.getOrPut(k) { mutableListOf() }.add(it)
                            }
                            byKey.forEach { (k, v) -> ordered.add(k to v) }
                            ordered
                        }
                        if (historyItems.isEmpty()) {
                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(vertical = 24.dp),
                                contentAlignment = Alignment.Center
                            ) {
                                Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                    Icon(Icons.Filled.PhotoLibrary, contentDescription = null, tint = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.size(40.dp))
                                    Text("暂无抠图记录", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                    Text("抠图完成后自动保存到此处", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                }
                            }
                        } else if (historyView == 1) {
                            // ★ 网格视图：按原图分组，每组一张卡片；点击展开/收起该组记录，展开后显示组内记录（支持多选）
                            //   每行两列，用 weight(1f) 均分宽度，占满可用区域
                            val gridGroups = groups.toList()
                            gridGroups.chunked(2).forEach { rowGroups ->
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                                ) {
                                    rowGroups.forEach { (groupKey, items) ->
                                        val collapsed = collapsedHistoryGroups.contains(groupKey)
                                        val expanded = !collapsed
                                        val latest = items.maxByOrNull { it.timestamp } ?: return@forEach
                                        // ★ 异步缩略图（IO 线程解码，避免组合期同步解码卡 UI）
                                        //   ★ 网格组卡片展示【原图】缩略图（同一原图分组入口，让历史列表能看到原图）
                                        val thumb = remember(latest.id) { mutableStateOf<Bitmap?>(null) }
                                        LaunchedEffect(latest.id) {
                                            thumb.value = withContext(Dispatchers.IO) { HistoryManager.loadOriginThumb(context, latest) }
                                        }
                                        Column(
                                            modifier = Modifier
                                                // ★ 末行单张卡片：保持半宽靠左（fillMaxWidth(0.5f)），避免占满整行变"单列很大"
                                                .then(if (rowGroups.size == 1) Modifier.fillMaxWidth(0.5f) else Modifier.weight(1f))
                                                .clip(RoundedCornerShape(12.dp))
                                                .background(
                                                    if (expanded) MaterialTheme.colorScheme.primary.copy(alpha = 0.12f)
                                                    else if (historySelectedIds.any { id -> items.any { it.id == id } })
                                                        MaterialTheme.colorScheme.primary.copy(alpha = 0.08f)
                                                    else MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.4f)
                                                )
                                                .combinedClickable(
                                                    onClick = {
                                                        // ★ 点击=展开/收起该组（持久化）
                                                        collapsedHistoryGroups =
                                                            if (collapsed) collapsedHistoryGroups - groupKey
                                                            else collapsedHistoryGroups + groupKey
                                                        Prefs.historyCollapsedGroups = collapsedHistoryGroups.joinToString(",")
                                                    },
                                                    onLongClick = {
                                                        // ★ 长按整组=把该组全部选中进入多选
                                                        historySelectedIds = items.map { it.id }.toSet()
                                                    }
                                                )
                                                .padding(6.dp),
                                            horizontalAlignment = Alignment.CenterHorizontally
                                        ) {
                                            val tb = thumb.value
                                            if (tb != null) {
                                                Image(
                                                    bitmap = tb.asImageBitmap(),
                                                    contentDescription = null,
                                                    modifier = Modifier
                                                        .fillMaxWidth()
                                                        .aspectRatio(1f)
                                                        .clip(RoundedCornerShape(8.dp)),
                                                    contentScale = ContentScale.Crop
                                                )
                                            } else {
                                                Box(
                                                    modifier = Modifier
                                                        .fillMaxWidth()
                                                        .aspectRatio(1f)
                                                        .clip(RoundedCornerShape(8.dp))
                                                        .background(MaterialTheme.colorScheme.surfaceVariant),
                                                    contentAlignment = Alignment.Center
                                                ) {
                                                    Icon(Icons.Filled.Image, contentDescription = null, tint = MaterialTheme.colorScheme.onSurfaceVariant)
                                                }
                                            }
                                            Spacer(Modifier.height(4.dp))
                                            Text(
                                                if (items.size > 1) "${items.size} 条  ${if (expanded) "收起▲" else "展开▼"}" else HistoryManager.formatTime(latest.timestamp),
                                                style = MaterialTheme.typography.labelSmall,
                                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                                maxLines = 1
                                            )
                                            // ★ 展开后显示组内记录（缩小卡片：缩略图 + 时间，可勾选多选）
                                            if (expanded) {
                                                Spacer(Modifier.height(4.dp))
                                                items.sortedByDescending { it.timestamp }.forEach { item ->
                                                    val isSel = historySelectedIds.contains(item.id)
                                                    // ★ 异步缩略图（IO 线程解码）
                                                    val itemThumb = remember(item.id) { mutableStateOf<Bitmap?>(null) }
                                                    LaunchedEffect(item.id) {
                                                        itemThumb.value = withContext(Dispatchers.IO) { HistoryManager.loadThumb(context, item) }
                                                    }
                                                    Row(
                                                        modifier = Modifier
                                                            .fillMaxWidth()
                                                            .clip(RoundedCornerShape(8.dp))
                                                            .background(if (isSel) MaterialTheme.colorScheme.primary.copy(alpha = 0.18f) else Color.Transparent)
                                                            .combinedClickable(
                                                                onClick = {
                                                                    if (historySelectedIds.isNotEmpty()) {
                                                                        historySelectedIds = if (isSel) historySelectedIds - item.id else historySelectedIds + item.id
                                                                    } else {
                                                                        viewerItem = item
                                                                        viewerBitmap = HistoryManager.loadResult(context, item)
                                                                        viewerOriginBitmap = HistoryManager.loadOriginal(context, item)
                                                                    }
                                                                },
                                                                onLongClick = { historySelectedIds = historySelectedIds + item.id }
                                                            )
                                                            .padding(4.dp),
                                                        verticalAlignment = Alignment.CenterVertically
                                                    ) {
                                                        if (historySelectedIds.isNotEmpty()) {
                                                            Checkbox(
                                                                checked = isSel,
                                                                onCheckedChange = { c ->
                                                                    historySelectedIds = if (c) historySelectedIds + item.id else historySelectedIds - item.id
                                                                },
                                                                modifier = Modifier.size(18.dp)
                                                            )
                                                            Spacer(Modifier.width(3.dp))
                                                        }
                                                        val itb = itemThumb.value
                                                        if (itb != null) {
                                                            Image(
                                                                bitmap = itb.asImageBitmap(),
                                                                contentDescription = null,
                                                                modifier = Modifier.size(30.dp).clip(RoundedCornerShape(6.dp)),
                                                                contentScale = ContentScale.Crop
                                                            )
                                                        } else {
                                                            Box(
                                                                Modifier.size(30.dp).clip(RoundedCornerShape(6.dp)).background(MaterialTheme.colorScheme.surfaceVariant),
                                                                contentAlignment = Alignment.Center
                                                            ) { Icon(Icons.Filled.Image, contentDescription = null, tint = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.size(16.dp)) }
                                                        }
                                                        Spacer(Modifier.width(4.dp))
                                                        Text(
                                                            HistoryManager.formatTime(item.timestamp),
                                                            style = MaterialTheme.typography.labelSmall,
                                                            maxLines = 1,
                                                            modifier = Modifier.weight(1f)
                                                        )
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            // 组折叠状态已从 Prefs 初始化；切换时持久化
                            groups.forEach { (groupKey, items) ->
                                val collapsed = collapsedHistoryGroups.contains(groupKey)
                                val expanded = !collapsed
                                val latest = items.maxByOrNull { it.timestamp } ?: return@forEach
                                // ★ 异步缩略图（IO 线程解码，避免组合期同步解码卡 UI）
                                //   ★ 组头展示【原图】缩略图（同一原图分组入口，让历史列表能看到原图）
                                val thumb = remember(latest.id) { mutableStateOf<Bitmap?>(null) }
                                LaunchedEffect(latest.id) {
                                    thumb.value = withContext(Dispatchers.IO) { HistoryManager.loadOriginThumb(context, latest) }
                                }
                                // ---- 组头（折叠条）----
                                Row(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .clip(RoundedCornerShape(12.dp))
                                        .background(if (expanded) MaterialTheme.colorScheme.primary.copy(alpha = 0.08f) else MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.4f))
                                        .clickable {
                                            // 切换折叠并持久化
                                            collapsedHistoryGroups =
                                                if (collapsed) collapsedHistoryGroups - groupKey
                                                else collapsedHistoryGroups + groupKey
                                            Prefs.historyCollapsedGroups = collapsedHistoryGroups.joinToString(",")
                                        }
                                        .padding(horizontal = 10.dp, vertical = 8.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    val tb = thumb.value
                                    if (tb != null) {
                                        Image(
                                            bitmap = tb.asImageBitmap(),
                                            contentDescription = null,
                                            modifier = Modifier
                                                .size(44.dp)
                                                .clip(RoundedCornerShape(8.dp)),
                                            contentScale = ContentScale.Crop
                                        )
                                    } else {
                                        Box(
                                            modifier = Modifier
                                                .size(44.dp)
                                                .clip(RoundedCornerShape(8.dp))
                                                .background(MaterialTheme.colorScheme.surfaceVariant),
                                            contentAlignment = Alignment.Center
                                        ) {
                                            Icon(Icons.Filled.Image, contentDescription = null, tint = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.size(22.dp))
                                        }
                                    }
                                    Spacer(Modifier.width(10.dp))
                                    Column(Modifier.weight(1f)) {
                                        Text(
                                            if (items.size > 1) "同一原图 · ${items.size} 条记录" else "抠图记录",
                                            style = MaterialTheme.typography.bodyLarge,
                                            fontWeight = FontWeight.Bold
                                        )
                                        Text(
                                            "${HistoryManager.formatTime(items.minOf { it.timestamp })}  ~  ${HistoryManager.formatTime(items.maxOf { it.timestamp })}",
                                            style = MaterialTheme.typography.labelSmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant
                                        )
                                    }
                                    Text(
                                        if (expanded) "收起 ▲" else "展开 ▼",
                                        style = MaterialTheme.typography.labelSmall,
                                        color = MaterialTheme.colorScheme.primary
                                    )
                                    // ★ 整组删除
                                    IconButton(
                                        onClick = { pendingDeleteGroup = items },
                                        modifier = Modifier.size(28.dp)
                                    ) {
                                        Icon(Icons.Filled.Delete, contentDescription = "删除整组", tint = MaterialTheme.colorScheme.error, modifier = Modifier.size(16.dp))
                                    }
                                }
                                // ---- 组内记录（展开时显示）----
                                if (expanded) {
                                    items.forEach { item ->
                                        val isSelected = historySelectedIds.contains(item.id)
                                        // ★ 异步缩略图（IO 线程解码）
                                        val itemThumb = remember(item.id) { mutableStateOf<Bitmap?>(null) }
                                        LaunchedEffect(item.id) {
                                            itemThumb.value = withContext(Dispatchers.IO) { HistoryManager.loadThumb(context, item) }
                                        }
                                        Row(
                                            modifier = Modifier
                                                .fillMaxWidth()
                                                .clip(RoundedCornerShape(12.dp))
                                                .background(if (isSelected) MaterialTheme.colorScheme.primary.copy(alpha = 0.12f) else Color.Transparent)
                                                // ★ 点击=多选勾选/打开查看器；长按=进入多选并勾选当前条
                                                .combinedClickable(
                                                    onClick = {
                                                        // ★ 多选模式下点按=勾选；普通模式=打开查看器
                                                        if (historySelectedIds.isNotEmpty()) {
                                                            historySelectedIds = if (isSelected)
                                                                historySelectedIds - item.id else historySelectedIds + item.id
                                                        } else {
                                                            viewerItem = item
                                                            viewerBitmap = HistoryManager.loadResult(context, item)
                                                            viewerOriginBitmap = HistoryManager.loadOriginal(context, item)
                                                        }
                                                    },
                                                    onLongClick = {
                                                        historySelectedIds = historySelectedIds + item.id
                                                    }
                                                )
                                                .padding(start = 12.dp, top = 6.dp, bottom = 6.dp, end = 6.dp),
                                            verticalAlignment = Alignment.CenterVertically
                                        ) {
                                            // ★ 多选模式：选择框
                                            if (historySelectedIds.isNotEmpty()) {
                                                Checkbox(
                                                    checked = isSelected,
                                                    onCheckedChange = { c ->
                                                        historySelectedIds = if (c)
                                                            historySelectedIds + item.id else historySelectedIds - item.id
                                                    },
                                                    modifier = Modifier.size(22.dp)
                                                )
                                                Spacer(Modifier.width(6.dp))
                                            }
                                            val itb = itemThumb.value
                                            if (itb != null) {
                                                Image(
                                                    bitmap = itb.asImageBitmap(),
                                                    contentDescription = null,
                                                    modifier = Modifier
                                                        .size(52.dp)
                                                        .clip(RoundedCornerShape(8.dp)),
                                                    contentScale = ContentScale.Crop
                                                )
                                            } else {
                                                Box(
                                                    modifier = Modifier
                                                        .size(52.dp)
                                                        .clip(RoundedCornerShape(8.dp))
                                                        .background(MaterialTheme.colorScheme.surfaceVariant),
                                                    contentAlignment = Alignment.Center
                                                ) {
                                                    Icon(Icons.Filled.Image, contentDescription = null, tint = MaterialTheme.colorScheme.onSurfaceVariant)
                                                }
                                            }
                                            Spacer(Modifier.width(10.dp))
                                            Column(Modifier.weight(1f)) {
                                                Text(
                                                    HistoryManager.formatTime(item.timestamp),
                                                    style = MaterialTheme.typography.bodyLarge,
                                                    fontWeight = FontWeight.Bold
                                                )
                                                if (item.modelName.isNotEmpty()) {
                                                    Text(
                                                        item.modelName,
                                                        style = MaterialTheme.typography.labelMedium,
                                                        color = MaterialTheme.colorScheme.primary
                                                    )
                                                }
                                                val infoParts = mutableListOf<String>()
                                                if (item.durationMs > 0) infoParts += "${"%.1f".format(item.durationMs / 1000f)}s"
                                                if (item.accel.isNotEmpty()) infoParts += item.accel
                                                if (item.turbo) infoParts += "极速"
                                                Text(
                                                    if (infoParts.isNotEmpty()) infoParts.joinToString(" · ") else "点击查看大图",
                                                    style = MaterialTheme.typography.labelSmall,
                                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                                )
                                            }
                                            if (historySelectedIds.isEmpty()) {
                                                IconButton(onClick = {
                                                    HistoryManager.delete(context, item)
                                                    HistoryManager.invalidateHistoryCache()
                                                    historyVersion++
                                                }) {
                                                    Icon(Icons.Filled.Delete, contentDescription = "删除", tint = MaterialTheme.colorScheme.error)
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } // ---- Tab 2 结束 ----

            // ★ 整组删除确认弹窗
            pendingDeleteGroup?.let { deleteGroup ->
                AlertDialog(
                    onDismissRequest = { pendingDeleteGroup = null },
                    title = { Text("删除整组记录") },
                    text = { Text("确定删除这 ${deleteGroup.size} 条抠图记录吗？此操作不可撤销。") },
                    confirmButton = {
                        TextButton(onClick = {
                            deleteGroup.forEach { HistoryManager.delete(context, it) }
                            HistoryManager.invalidateHistoryCache()
                            pendingDeleteGroup = null
                            historyVersion++
                        }) { Text("删除", color = MaterialTheme.colorScheme.error) }
                    },
                    dismissButton = {
                        TextButton(onClick = { pendingDeleteGroup = null }) { Text("取消") }
                    }
                )
            }

            // ===== Tab 0 续：抠图操作区 =====
            if (currentTab == 0) {
            // ===== 选图 =====
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                Button(
                    onClick = { imagePicker.launch("image/*") },
                    enabled = modelReady,
                    modifier = Modifier
                        .weight(1f)
                        .height(52.dp),
                    shape = RoundedCornerShape(14.dp)
                ) {
                    Icon(Icons.Filled.PhotoLibrary, contentDescription = null)
                    Spacer(Modifier.width(8.dp))
                    Text("选择图片", fontSize = 15.sp, fontWeight = FontWeight.Bold)
                }
                // 有图时显示清除按钮（可置空当前选中图片）
                if (originalBitmap != null || resultBitmap != null) {
                    OutlinedButton(
                        onClick = {
                            originalBitmap = null
                            resultBitmap = null
                            currentHistoryKey = ""
                            ProjectStateManager.clear(context)
                            statusText = "已清除图片，可重新选择"
                        },
                        modifier = Modifier
                            .height(52.dp)
                            .width(52.dp),
                        shape = RoundedCornerShape(14.dp)
                    ) {
                        Icon(Icons.Filled.Clear, contentDescription = "清除图片", tint = MaterialTheme.colorScheme.error)
                    }
                }
            }

            // ===== 阈值滑块 =====
            Card(
                colors = CardDefaults.cardColors(containerColor = Color(0xFFFFFFFF)),
                shape = RoundedCornerShape(16.dp)
            ) {
                Column(Modifier.padding(14.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text("抠图阈值", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            OutlinedButton(
                                onClick = { threshold = (threshold - 0.01f).coerceIn(0f, 0.95f); Prefs.threshold = threshold },
                                contentPadding = PaddingValues(horizontal = 10.dp, vertical = 4.dp)
                            ) { Text("−", style = MaterialTheme.typography.titleSmall) }
                            Text(
                                "${"%.2f".format(threshold)}",
                                style = MaterialTheme.typography.titleSmall,
                                color = MaterialTheme.colorScheme.primary,
                                fontWeight = FontWeight.Bold,
                                modifier = Modifier.width(52.dp),
                                textAlign = TextAlign.Center
                            )
                            OutlinedButton(
                                onClick = { threshold = (threshold + 0.01f).coerceIn(0f, 0.95f); Prefs.threshold = threshold },
                                contentPadding = PaddingValues(horizontal = 10.dp, vertical = 4.dp)
                            ) { Text("+", style = MaterialTheme.typography.titleSmall) }
                        }
                    }
                    Slider(
                        value = threshold,
                        onValueChange = {
                            // 拖动/点击量化到 0.1 步进（粗调）；细调用上方 +/- 按钮（0.01）
                            val q = ((it * 10).roundToInt() / 10f).coerceIn(0f, 0.95f)
                            threshold = q
                            Prefs.threshold = q
                        },
                        valueRange = 0f..0.95f
                    )
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text("0 (保留最多)", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        Text("0.5 (标准)", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        Text("0.95 (严格)", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                    // ---- 精细抠图进度条 ----
                    if (isTiling) {
                        Column(
                            modifier = Modifier.fillMaxWidth(),
                            verticalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween
                            ) {
                                Text(
                                    "分块处理",
                                    style = MaterialTheme.typography.labelMedium,
                                    color = MaterialTheme.colorScheme.primary
                                )
                                Text(
                                    if (tileProgressText.isNotEmpty()) "$tileProgressText 块" else "${(tileProgress * 100).toInt()}%",
                                    style = MaterialTheme.typography.labelMedium,
                                    fontWeight = FontWeight.Bold
                                )
                            }
                            LinearProgressIndicator(
                                progress = { tileProgress },
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(6.dp)
                            )
                            Text(
                                "大图分块推理中 · 保留原图细节",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                }
            }

            // ===== 原图 + 结果对比 =====
            if (originalBitmap != null || resultBitmap != null) {
                Card(
                    colors = CardDefaults.cardColors(containerColor = Color(0xFFFFFFFF)),
                    shape = RoundedCornerShape(16.dp)
                ) {
                    Column(
                        modifier = Modifier.padding(14.dp),
                        verticalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        if (originalBitmap != null) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Text("原图", style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.Bold)
                                Spacer(Modifier.weight(1f))
                                Text(
                                    "点击查看",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.primary
                                )
                            }
                            Image(
                                bitmap = originalBitmap!!.asImageBitmap(),
                                contentDescription = "原图",
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(220.dp)
                                    .clip(RoundedCornerShape(12.dp))
                                    .clickable {
                                        // 选图查看器：全屏预览原图（可缩放）
                                        viewerBitmap = originalBitmap
                                        viewerItem = null
                                        showOriginalViewer = true
                                    },
                                contentScale = ContentScale.Fit
                            )
                        }
                        if (resultBitmap != null) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Text(
                                    if (comparePressed) "原图（按住对比）" else "抠图结果",
                                    style = MaterialTheme.typography.labelMedium,
                                    fontWeight = FontWeight.Bold
                                )
                                Spacer(Modifier.weight(1f))
                                Text(
                                    "点击查看",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.primary
                                )
                            }
                            // ★ 结果图：底下垫背景色（棋盘/纯色，持久化自查看器设置）让透明通道可见
                            val resBg = when (viewerBgColor) {
                                "白色" -> Color.White
                                "黑色" -> Color.Black
                                "浅灰" -> Color(0xFFBDBDBD)
                                "绿色" -> Color(0xFF4CAF50)
                                else -> null // 棋盘
                            }
                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(220.dp)
                                    .clip(RoundedCornerShape(12.dp))
                                    .background(resBg ?: Color(0xFF101418))
                            ) {
                                // 棋盘背景（仅棋盘模式画格子）
                                if (resBg == null) {
                                    Canvas(Modifier.fillMaxSize()) {
                                        val cell = with(density) { 20.dp.toPx() }
                                        var x = 0f
                                        while (x < size.width) {
                                            var y = 0f
                                            while (y < size.height) {
                                                drawRect(
                                                    color = if (((x / cell).toInt() + (y / cell).toInt()) % 2 == 0)
                                                        Color(0xFF2A2F36) else Color(0xFF1C2127),
                                                    topLeft = Offset(x, y),
                                                    size = Size(cell, cell)
                                                )
                                                y += cell
                                            }
                                            x += cell
                                        }
                                    }
                                }
                                Image(
                                    // ★ 按住「对比」按钮时显示原图，松开恢复结果
                                    bitmap = (if (comparePressed && originalBitmap != null) originalBitmap!! else resultBitmap!!).asImageBitmap(),
                                    contentDescription = "结果",
                                    modifier = Modifier
                                        .fillMaxSize()
                                        .clickable {
                                            viewerBitmap = if (comparePressed && originalBitmap != null) originalBitmap else resultBitmap
                                            viewerItem = null
                                            showOriginalViewer = true
                                            // ★ 当前结果打开查看器：对应原图即主界面原图，供编辑页恢复画笔取原背景色
                                            viewerOriginBitmap = originalBitmap
                                        },
                                    contentScale = ContentScale.Fit
                                )
                            }
                        }
                    }
                }
            }

            // ===== 操作按钮 =====
            // 撤销/前进 行（有结果时可用）+ 右侧小「按住对比」按钮
            if (resultBitmap != null) {
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
                    OutlinedButton(
                        onClick = {
                            val prev = doUndo()
                            if (prev != null) {
                                resultBitmap = prev
                                statusText = "已撤销"
                            }
                        },
                        enabled = undoStack.isNotEmpty() && !isProcessing,
                        modifier = Modifier
                            .weight(1f)
                            .height(46.dp),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Icon(Icons.Filled.Undo, contentDescription = null, modifier = Modifier.size(16.dp))
                        Spacer(Modifier.width(4.dp))
                        Text("撤销", fontSize = 13.sp)
                    }
                    OutlinedButton(
                        onClick = {
                            val next = doRedo()
                            if (next != null) {
                                resultBitmap = next
                                statusText = "已前进"
                            }
                        },
                        enabled = redoStack.isNotEmpty() && !isProcessing,
                        modifier = Modifier
                            .weight(1f)
                            .height(46.dp),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Icon(Icons.Filled.Redo, contentDescription = null, modifier = Modifier.size(16.dp))
                        Spacer(Modifier.width(4.dp))
                        Text("前进", fontSize = 13.sp)
                    }
                    // 小尺寸「按住对比」：按住显示原图，松开恢复结果
                    // ★ 用 requireUnconsumed=false 才能在本按钮自身 clickable 消费 down 后仍收到手势
                    Box(
                        modifier = Modifier
                            .size(width = 64.dp, height = 46.dp)
                            .clip(RoundedCornerShape(12.dp))
                            .border(1.dp, MaterialTheme.colorScheme.outline, RoundedCornerShape(12.dp))
                            .pointerInput(originalBitmap != null && resultBitmap != null && !isProcessing) {
                                awaitEachGesture {
                                    awaitFirstDown(requireUnconsumed = false).also { comparePressed = true }
                                    // 等待抬起（含取消），抬起后恢复显示结果
                                    waitForUpOrCancellation()
                                    comparePressed = false
                                }
                            },
                        contentAlignment = Alignment.Center
                    ) {
                        if (comparePressed) {
                            Text("松开", fontSize = 11.sp, color = MaterialTheme.colorScheme.primary)
                        } else {
                            Icon(Icons.Filled.Compare, contentDescription = "按住看原图", modifier = Modifier.size(18.dp), tint = MaterialTheme.colorScheme.primary)
                        }
                    }
                }
            }
            // ★ 处理计时（实时跳动全程显示）
            if (isProcessing) {
                Text(
                    "处理中 ${runningSeconds} 秒...",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.primary,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.fillMaxWidth(),
                    textAlign = TextAlign.Center
                )
            } else if (lastElapsedMs != null) {
                Text(
                    "最近用时 ${"%.1f".format(lastElapsedMs!! / 1000f)} 秒",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.primary,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.fillMaxWidth(),
                    textAlign = TextAlign.Center
                )
            }
            // 第一行：主操作（立即抠图/停止 + 保存），各占一半宽度，15sp 完整显示
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                Button(
                    onClick = {
                        if (isProcessing) {
                            // ★ 运行中点击 → 停止：立即设 cancelFlag（分块间 break / 推理前抛 CancellationException），
                            //   并置 stopRequested 供调用处丢弃半成品。当前正在执行的单次推理无法中断，
                            //   但下一块/下一次会立即停止。
                            stopRequested = true
                            statusText = "正在停止..."
                            try { getEngine().cancelCurrentRun() } catch (_: Exception) {}
                        } else {
                            runRmbg()
                        }
                    },
                    enabled = modelReady && originalBitmap != null,
                    modifier = Modifier
                        .weight(1f)
                        .height(52.dp),
                    shape = RoundedCornerShape(14.dp)
                ) {
                    if (isProcessing) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(22.dp),
                            strokeWidth = 3.dp,
                            color = MaterialTheme.colorScheme.onPrimary
                        )
                        Spacer(Modifier.width(8.dp))
                    }
                    Text(
                        when {
                            isProcessing -> "停止"
                            resultBitmap != null -> "重试"
                            else -> "立即抠图"
                        },
                        fontSize = 15.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
                if (resultBitmap != null) {
                    Button(
                        onClick = {
                            val ok = saveToRmbgFolder(context, resultBitmap)
                            Toast.makeText(context, if (ok) "已保存到 相册/rmbg" else "保存失败，可试试另存为", Toast.LENGTH_SHORT).show()
                        },
                        modifier = Modifier
                            .weight(1f)
                            .height(52.dp),
                        shape = RoundedCornerShape(14.dp)
                    ) {
                        Text("保存", fontSize = 15.sp, fontWeight = FontWeight.Bold)
                    }
                }
            }
            // 第二行：次级操作（分享 + 另存为），13sp 完整显示
            if (resultBitmap != null) {
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    OutlinedButton(
                        onClick = {
                            // ★ 复用最近一次保存到历史的 PNG 文件分享（零重新编码，拉起快）；
                            //   历史保存是异步的，若未就绪则回退缓存编码分享
                            val hist = latestHistoryFile
                            if (hist != null && hist.exists() && hist.length() > 100L) {
                                shareHistoryFile(context, hist)
                            } else {
                                shareResult(context, resultBitmap)
                            }
                        },
                        modifier = Modifier
                            .weight(1f)
                            .height(48.dp),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Text("分享", fontSize = 13.sp)
                    }
                    OutlinedButton(
                        onClick = { saveLauncher.launch("rmbg_${System.currentTimeMillis()}.png") },
                        modifier = Modifier
                            .weight(1f)
                            .height(48.dp),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Text("另存为", fontSize = 13.sp)
                    }
                }
            }
            // 高级编辑入口（独立一行，有原图即可进入）
            if (originalBitmap != null) {
                OutlinedButton(
                    onClick = {
                        // ★ 修复载入结果：有抠好的结果图→编辑结果图（在结果上选区继续精修）；无结果→编辑原图
                        editorOnOriginal = resultBitmap == null
                        showEditor = true
                    },
                    enabled = !isProcessing,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(52.dp),
                    shape = RoundedCornerShape(14.dp)
                ) {
                    Icon(Icons.Filled.Edit, contentDescription = null)
                    Spacer(Modifier.width(8.dp))
                    Text(
                        if (resultBitmap != null) "高级编辑（精修）" else "高级编辑（先标记再抠）",
                        fontSize = 15.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
            } // ---- Tab 0 续结束 ----
        }
    }

    // ---- 高级编辑二级页面（覆盖显示，有原图即可进入）----
    if (showEditor && originalBitmap != null) {
        if (!showBrushFromArea) {
            // 区域选择子页（区域抠图 + AI 区域重绘 共用同一套选区框/画布）
            AreaSelectScreen(
                original = originalBitmap!!,
                threshold = threshold,
                currentResult = if (editorOnOriginal) null else resultBitmap,
                // ★ 修复③：改传 lambda 而非预求值，在按钮点击协程里后台懒加载引擎
                getEngine = { try { RmbgScreenState.engine ?: getEngine() } catch (_: Exception) { null } },
                onApply = { newRes ->
                    if (!newRes.isRecycled) {
                        pushUndo(if (editorOnOriginal) originalBitmap else resultBitmap)
                        if (editorOnOriginal) originalBitmap = newRes else resultBitmap = newRes
                        statusText = if (editorOnOriginal) "已应用区域抠图到原图" else "已应用区域抠图到结果"
                    }
                },
                onEnterBrush = { showBrushFromArea = true },
                onSwitchModel = {
                    // ★ 快捷切模型（不跳转）：就地弹出模型列表选择，选完自动用新模型重新抠图
                    showQuickModelPicker = true
                },
                // ★ AI 区域重绘：与"局部重抠"并列，共用同一套选区/画布（AreaSelectScreen 内部实现）
                onRedrawRegion = { src, sel ->
                    // ★ 后端就绪（本地引擎未启动则启动；LocalDream 在线则直接用）
                    // ★ Lite 精简版：未打包本地引擎资产（qnnlibs/sd_core），强制走 LocalDream
                    val backend = if (BuildConfig.IS_LITE) "dream" else Prefs.aiRedrawBackend
                    if (backend == "dream") {
                        if (!LocalDreamClient.checkOnline()) {
                            throw RuntimeException("未检测到 LocalDream App 的 8081 服务，请先在 LocalDream 中完成加载")
                        }
                    } else {
                        if (!AiRedrawEngine.checkHealth()) {
                            // 部署/启动（幂等：已部署跳过解压）
                            AiRedrawEngine.restoreModelZip(context)
                            // ★ 模型可用性：zip 在 或 已部署（zip 部署后自动删除）都算有模型
                            if (!AiRedrawEngine.isModelReady(context)) {
                                throw RuntimeException("未选择 SD 模型包，请先到模型页选择 zip（或切换推理后端为 LocalDream App）")
                            }
                            // zip 存在且未部署 → 才需要部署；zip 已删但已部署 → 直接复用部署目录
                            val zip = AiRedrawEngine.modelZipFile
                            if (zip?.exists() == true && !AiRedrawEngine.isModelDeployed(context, zip)) {
                                val ok = AiRedrawEngine.deployModel(context, zip)
                                if (!ok) throw RuntimeException("模型部署失败: ${AiRedrawEngine.status}")
                                deployedModelsVersion++  // ★ 部署完成刷新已部署模型列表
                            }
                            AiRedrawEngine.deployRuntime(context)
                            // ★ 修复④：用用户设置的档位启动引擎（旧版硬编码 512×512，与实际请求档位不匹配时 QNN 报错）
                            val vw = Prefs.aiRedrawWidth.coerceIn(512, 1024)
                            val vh = Prefs.aiRedrawHeight.coerceIn(512, 1024)
                            val started = AiRedrawEngine.start(context, vw, vh)
                            if (!started) throw RuntimeException("AI 引擎启动失败: ${AiRedrawEngine.status}")
                        }
                    }
                    aiRedrawRegion(src, sel)  // 裁区→重绘→贴回
                },
                onBack = { showEditor = false }
            )
        } else {
            // ★ 合并：画笔子页统一走 PostProcessScreen（后处理控件 + EditBrushLayer 画笔 + 放大镜）
            //   不再使用 EditorScreen——PostProcessScreen 是其超集。
            val ppResult = if (editorOnOriginal) originalBitmap!! else (resultBitmap ?: originalBitmap!!)
            PostProcessScreen(
                result = ppResult,
                originPx = null,
                originBitmap = originalBitmap,
                onApply = { edited ->
                    if (!edited.isRecycled) {
                        pushUndo(if (editorOnOriginal) originalBitmap else resultBitmap)
                        if (editorOnOriginal) {
                            originalBitmap = edited
                            ProjectStateManager.save(context, edited, resultBitmap)
                        } else resultBitmap = edited
                        statusText = if (editorOnOriginal) "已应用画笔修改到原图，可点击抠图" else "已应用画笔修改到结果"
                    }
                },
                onBack = { showBrushFromArea = false },
                initialBrushMode = true  // ★ 从画笔入口进入，自动切画笔模式
            )
        }
    }

    // ---- 高级编辑 · 快捷切模型弹窗（★不跳转模型页：就地选模型，选完自动重新抠图）----
    if (showQuickModelPicker) {
        val pickModels: List<Any> = remember {
            buildList {
                addAll(ModelManager.cpuModels)
                addAll(ModelManager.qnnModels)
                addAll(ModelManager.localModels())
            }
        }
        AlertDialog(
            onDismissRequest = { showQuickModelPicker = false; quickModelSwitching = false },
            title = { Text("快捷切换模型") },
            text = {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(max = 430.dp)
                        .verticalScroll(rememberScrollState()),
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    pickModels.forEach { m ->
                        val id = when (m) {
                            is ModelManager.BuiltinModel -> m.id
                            is ModelManager.LocalModel -> m.id
                            else -> return@forEach
                        }
                        val name = when (m) {
                            is ModelManager.BuiltinModel -> m.name
                            is ModelManager.LocalModel -> m.displayName
                            else -> "?"
                        }
                        val desc = when (m) {
                            is ModelManager.BuiltinModel -> m.description
                            is ModelManager.LocalModel -> "本地导入模型"
                            else -> ""
                        }
                        val isQnnModel = (m as? ModelManager.BuiltinModel)?.isQnn == true
                        val selected = id == selectedBuiltinId
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(10.dp))
                                .background(if (selected) MaterialTheme.colorScheme.primary.copy(alpha = 0.12f) else Color.Transparent)
                                .selectable(
                                    selected = selected,
                                    enabled = !quickModelSwitching,
                                    onClick = {
                                        if (quickModelSwitching || selected) return@selectable
                                        quickModelSwitching = true
                                        showQuickModelPicker = false
                                        // ★★ 快捷切模型=临时切换（不改变主模型）：
                                        //   记录主模型配置 → 临时切到目标模型重抠当前图 → 完成后自动恢复主模型。
                                        quickModelRestore = QuickModelRestore(
                                            id = selectedBuiltinId,
                                            repo = modelRepo,
                                            file = modelFile,
                                            enableQnn = enableQnn
                                        )
                                        // 临时切到目标模型（不改 Prefs，不改变主模型持久化）
                                        selectedBuiltinId = id
                                        ModelManager.selectedModelId = id
                                        if (m is ModelManager.BuiltinModel) {
                                            modelRepo = m.hfRepo
                                            modelFile = m.hfFile
                                            enableQnn = m.isQnn
                                        } else {
                                            modelRepo = ""
                                            modelFile = ModelManager.modelFileName()
                                            enableQnn = false
                                        }
                                        try { engine?.close() } catch (_: Exception) {}
                                        engine = null
                                        RmbgScreenState.engine = null
                                        modelReady = ModelManager.isModelDownloaded()
                                        if (!modelReady) {
                                            statusText = if (isQnnModel) "该 QNN 模型未部署，请到模型页部署后重试"
                                            else "该模型尚未下载，请到模型页下载"
                                            restoreQuickModel() // 目标模型不可用 → 立即恢复主模型
                                            quickModelSwitching = false
                                            quickModelRestore = null
                                        } else {
                                            // ✅ 临时模型就绪 → 关闭高级编辑 → 自动用临时模型重新抠图
                                            showEditor = false
                                            showBrushFromArea = false
                                            statusText = "已临时切换模型，正在重新抠图（完成后自动恢复主模型）..."
                                            runRmbg()
                                            quickModelSwitching = false
                                        }
                                    }
                                )
                                .padding(horizontal = 8.dp, vertical = 8.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            RadioButton(selected = selected, onClick = null)
                            Column(
                                Modifier
                                    .weight(1f)
                                    .padding(start = 6.dp)
                                    .fillMaxWidth()
                            ) {
                                Text(name, style = MaterialTheme.typography.bodyLarge,
                                    fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal,
                                    maxLines = 1, overflow = TextOverflow.Ellipsis)
                                Text(desc, style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                    maxLines = 1, overflow = TextOverflow.Ellipsis)
                            }
                            Text(if (isQnnModel) "QNN" else "CPU",
                                style = MaterialTheme.typography.labelSmall,
                                color = if (isQnnModel) Color(0xFFF57C00) else Color(0xFF616161))
                        }
                    }
                    Text("切换后自动用新模型重新抠图",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(top = 4.dp))
                }
            },
            confirmButton = {},
            dismissButton = {
                TextButton(onClick = { showQuickModelPicker = false; quickModelSwitching = false }) { Text("取消") }
            }
        )
    }
    val viewer = viewerItem
    val showViewer = viewer != null || showOriginalViewer
    // ★★ 系统返回键兼容：按覆盖层优先级逐级返回（后进先出）
    //   关于页 → 后处理编辑 → 查看器 → 快捷切模型弹窗 → 高级编辑/画笔 → 内置模型选择
    BackHandler(enabled = true) {
        when {
            showAboutPage -> showAboutPage = false

            showPostProcessScreen -> {
                showPostProcessScreen = false
                postProcessResult = null
                postProcessHistoryItem = null
            }

            showViewer -> {
                viewerItem = null; viewerBitmap = null; showOriginalViewer = false
                viewerBgOrigin = null; viewerBgStrength = 0
                viewerOriginBitmap = null; postProcessHistoryItem = null
                resetPostVars()
            }

            showQuickModelPicker -> {
                showQuickModelPicker = false; quickModelSwitching = false
            }

            showEditor -> {
                if (showBrushFromArea) showBrushFromArea = false
                else showEditor = false
            }

            showBuiltinPicker -> showBuiltinPicker = false
        }
    }
    if (showViewer) {
        val bmp0 = viewerBitmap
        // ★ HARDWARE bitmap 防御（对齐 Kortex LaMa 做法）：相册/相机返回的高性能位图(HARDWARE config)，
        //   getPixels/Canvas/setPixels 会抛异常或闪退。查看器任何像素级操作前先转软件位图 ARGB_8888。
        val bmp = if (bmp0 != null && bmp0.config == android.graphics.Bitmap.Config.HARDWARE) {
            val soft = bmp0.copy(android.graphics.Bitmap.Config.ARGB_8888, true)
            viewerBitmap = soft
            soft
        } else {
            bmp0
        }
        val isHistory = viewer != null
        // ★ 修复：历史记录/重新载入时 viewerBitmap 是新 Bitmap，与 resultBitmap 引用不同，
        //   `===` 判定失败会让透明结果落入"原图"分支（只显示高级编辑，看不到背景强度滑杆）。
        //   改为：非历史 + 有透明通道 → 视为结果图（透明 PNG 结果即透明图，带 alpha）。
        val isResult = !isHistory && bmp != null && bmp.hasAlpha()
        // ★ 打开查看器时缓存当前结果的原始像素（供背景强度滑杆 + 后处理可逆实时预览）
        LaunchedEffect(bmp) {
            if ((isHistory || isResult) && bmp != null && viewerBgOrigin == null) {
                // ★ 优先用引擎未硬化副本（保留半透明边缘 alpha，后处理去色边/柔化才能真正生效）。
                //   lastSoftResult 与 bmp 同尺寸时取它；否则退回当前 bmp 像素。
                val soft = try { getEngine().lastSoftResult } catch (_: Exception) { null }
                // ★ 崩溃修复：lastSoftResult 可能已被 recycle()（上次抠图/区域操作后释放但引用未置 null），
                //   getPixels 前必须校验 isRecycled，否则抛 "Can't call getPixels() on a recycled bitmap"。
                val src = if (soft != null && !soft.isRecycled &&
                    soft.width == bmp.width && soft.height == bmp.height) soft else bmp
                if (src.isRecycled) { viewerBgOrigin = null } else {
                    val px = IntArray(src.width * src.height)
                    src.getPixels(px, 0, src.width, 0, 0, src.width, src.height)
                    viewerBgOrigin = px
                }
            }
        }
        val viewerTitle = when {
            isHistory -> "抠图结果"
            isResult -> "抠图结果"
            else -> "原图"
        }
        Dialog(
            onDismissRequest = { viewerItem = null; viewerBitmap = null; showOriginalViewer = false; viewerBgOrigin = null; viewerBgStrength = 0; viewerOriginBitmap = null; postProcessHistoryItem = null; resetPostVars() },
            properties = androidx.compose.ui.window.DialogProperties(usePlatformDefaultWidth = false)
        ) {
            Card(
                colors = CardDefaults.cardColors(containerColor = Color(0xFFFFFFFF)),
                shape = RoundedCornerShape(20.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(12.dp)
                    // ★ 全屏/大弹窗：图片占大部分屏幕，底部紧凑操作区
                    .fillMaxHeight(0.94f)
            ) {
                Column(
                    modifier = Modifier
                        .padding(16.dp)
                        .fillMaxSize(),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Filled.Image, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                        Spacer(Modifier.width(8.dp))
                        Column(Modifier.weight(1f)) {
                            Text(viewerTitle, style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
                            Text(
                                if (isHistory) {
                                    "保存时间 ${HistoryManager.formatTime(viewer!!.timestamp)} · ${bmp?.width ?: 0}×${bmp?.height ?: 0}"
                                } else {
                                    "${bmp?.width ?: 0}×${bmp?.height ?: 0} · 双指缩放查看"
                                },
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                        IconButton(onClick = { viewerItem = null; viewerBitmap = null; showOriginalViewer = false; viewerBgOrigin = null; viewerBgStrength = 0; viewerOriginBitmap = null; postProcessHistoryItem = null; resetPostVars() }) {
                            Icon(Icons.Filled.Close, contentDescription = "关闭")
                        }
                    }
                    // ★ 大图预览区：weight(1f) 占满剩余屏幕（大部分屏幕），图片垂直居中
                    //   背景色可切换：棋盘 / 白 / 黑 / 浅灰 / 绿
                    val bgSolid = when (viewerBgColor) {
                        "白色" -> Color.White
                        "黑色" -> Color.Black
                        "浅灰" -> Color(0xFFBDBDBD)
                        "绿色" -> Color(0xFF4CAF50)
                        else -> null // 棋盘
                    }
                    var vScale by remember { mutableStateOf(1f) }
                    var vOffset by remember { mutableStateOf(Offset.Zero) }
                    val bw = (bmp?.width ?: 1).toFloat()
                    val bh = (bmp?.height ?: 1).toFloat()
                    BoxWithConstraints(
                        modifier = Modifier
                            .fillMaxWidth()
                            // ★ 图片占大部分屏幕：weight(1f) 撑满可用高度（不再 heightIn 480dp）
                            .weight(1f)
                            .clip(RoundedCornerShape(12.dp))
                            .background(bgSolid ?: Color(0xFF101418))
                    ) {
                        val density = LocalDensity.current
                        val vCanvasW = with(density) { maxWidth.toPx() }
                        val vCanvasH = with(density) { maxHeight.toPx() }
                        // fit 约束：默认宽 320dp，若竖屏长图按高倒推宽度超界则收窄，保证整图居中不切边
                        val vBaseW = with(density) {
                            (320.dp.toPx()).coerceAtMost(vCanvasW - 24.dp.toPx())
                        }.coerceAtMost(if (bh > 0) vCanvasH * bw / bh else vCanvasW)
                        val vBaseH = vBaseW * bh / bw
                        // 图像居中位置（px）——不随 scale 变化，graphicsLayer 负责缩放
                        val vBaseLeft = (vCanvasW - vBaseW) / 2f
                        val vBaseTop = (vCanvasH - vBaseH) / 2f

                        // ★ 棋盘背景（仅"棋盘"模式绘制；其他纯色模式不画棋盘格子）
                        if (bgSolid == null) {
                            Canvas(modifier = Modifier.fillMaxSize()) {
                                val cell = with(density) { 20.dp.toPx() }
                                var x = 0f
                                while (x < size.width) {
                                    var y = 0f
                                    while (y < size.height) {
                                        drawRect(
                                            color = if (((x / cell).toInt() + (y / cell).toInt()) % 2 == 0)
                                                Color(0xFF2A2F36) else Color(0xFF1C2127),
                                            topLeft = Offset(x, y),
                                            size = Size(cell, cell)
                                        )
                                        y += cell
                                    }
                                    x += cell
                                }
                            }
                        }
                        // 可交互层：捕获手势 + graphicsLayer 缩放/平移图像（查看器仅预览）
                        Box(
                            modifier = Modifier
                                .fillMaxSize()
                                .pointerInput(Unit) {
                                    detectTransformGestures { centroid, pan, zoom, _ ->
                                        val prevScale = vScale
                                        val newScale = (prevScale * zoom).coerceIn(1f, 8f)
                                        val curLeft = vBaseLeft + vOffset.x
                                        val curTop = vBaseTop + vOffset.y
                                        val anchorX = (centroid.x - pan.x - curLeft) / prevScale
                                        val anchorY = (centroid.y - pan.y - curTop) / prevScale
                                        val newLeft = (centroid.x - pan.x) - anchorX * newScale + pan.x
                                        val newTop = (centroid.y - pan.y) - anchorY * newScale + pan.y
                                        vScale = newScale
                                        vOffset = Offset(newLeft - vBaseLeft, newTop - vBaseTop)
                                    }
                                }
                        ) {
                            if (bmp != null) {
                                // ★ 按住「对比」按钮时快速切换：显示对应原图（含缩放/平移保持一致），松开恢复结果
                                val showOrig = comparePressed && viewerOriginBitmap != null && viewerOriginBitmap!!.isRecycled.not()
                                val disp = if (showOrig) viewerOriginBitmap!! else bmp
                                Image(
                                    bitmap = disp.asImageBitmap(),
                                    contentDescription = if (showOrig) "原图（按住对比）" else "结果大图",
                                    modifier = Modifier
                                        .offset { IntOffset(vBaseLeft.roundToInt(), vBaseTop.roundToInt()) }
                                        .size(with(density) { vBaseW.toDp() }, with(density) { vBaseH.toDp() })
                                        .graphicsLayer(
                                            scaleX = vScale,
                                            scaleY = vScale,
                                            transformOrigin = TransformOrigin(0f, 0f),
                                            translationX = vOffset.x,
                                            translationY = vOffset.y
                                        ),
                                    contentScale = ContentScale.Fit
                                )
                            } else {
                                Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                                    CircularProgressIndicator()
                                }
                            }
                        }
                    }
                    // 缩放指示 + 复位
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(
                            "缩放 ${"%.0f".format(vScale * 100)}% · 双指缩放，单指拖动",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        Spacer(Modifier.weight(1f))
                        if (vScale != 1f || vOffset != Offset.Zero) {
                            TextButton(onClick = { vScale = 1f; vOffset = Offset.Zero }) {
                                Text("复位")
                            }
                        }
                    }
                    // 背景色切换（棋盘/白/黑/浅灰/绿）
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        Text(
                            "背景",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        viewerBgChoices.forEach { choice ->
                            val bgPreview = when (choice) {
                                "白色" -> Color.White
                                "黑色" -> Color.Black
                                "浅灰" -> Color(0xFFBDBDBD)
                                "绿色" -> Color(0xFF4CAF50)
                                else -> null
                            }
                            val selected = viewerBgColor == choice
                            Box(
                                modifier = Modifier
                                    .size(32.dp)
                                    .clip(RoundedCornerShape(6.dp))
                                    .background(bgPreview ?: Color(0xFF101418))
                                    .border(
                                        if (selected) BorderStroke(2.dp, MaterialTheme.colorScheme.primary)
                                        else BorderStroke(1.dp, Color(0x33000000))
                                    )
                                    .clickable {
                                        viewerBgColor = choice
                                        Prefs.viewerBgColor = choice
                                    },
                                contentAlignment = Alignment.Center
                            ) {
                                if (choice == "棋盘" && bgPreview == null) {
                                    // 棋盘色块内画小格子预览
                                    Canvas(Modifier.fillMaxSize()) {
                                        val cell = size.minDimension / 2f
                                        drawRect(Color(0xFF2A2F36), size = Size(cell, cell))
                                        drawRect(Color(0xFF1C2127), topLeft = Offset(cell, 0f), size = Size(cell, cell))
                                        drawRect(Color(0xFF1C2127), topLeft = Offset(0f, cell), size = Size(cell, cell))
                                        drawRect(Color(0xFF2A2F36), topLeft = Offset(cell, cell), size = Size(cell, cell))
                                    }
                                }
                            }
                        }
                    }
                    // 操作：按来源显示不同入口
                    if (isHistory || isResult) {
                        // ★ 后处理/画笔 + 4x超分 + AI 重绘（三按钮并排）
                        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                            OutlinedButton(
                                onClick = {
                                    val pp = viewerBitmap ?: return@OutlinedButton
                                    postProcessResult = pp
                                    // ★ 进入独立编辑页前先关闭查看器 Dialog（否则会叠在编辑页上）
                                    //   保留 viewerBgOrigin(viewerBgOrigin 缓存原始像素)供编辑页"还原/恢复"使用
                                    // ★ 暂存关联的历史记录：后处理应用时 updateResult 写回同一条记录（原图存档不变）
                                    postProcessHistoryItem = viewerItem
                                    viewerItem = null
                                    showOriginalViewer = false
                                    showPostProcessScreen = true
                                },
                                enabled = bmp != null,
                                modifier = Modifier.weight(1f).height(40.dp),
                                shape = RoundedCornerShape(10.dp),
                                contentPadding = PaddingValues(horizontal = 4.dp, vertical = 0.dp)
                            ) {
                                Icon(Icons.Filled.AutoFixHigh, contentDescription = null, modifier = Modifier.size(13.dp))
                                Spacer(Modifier.width(3.dp))
                                Text("编辑", fontSize = 12.sp, maxLines = 1, softWrap = false)
                            }
                            // ★★ 4x 超分（独立按钮，不入历史；可保存/分享）
                            OutlinedButton(
                                onClick = { doSuperRes() },
                                enabled = bmp != null && !superResBusy && !isProcessing &&
                                    !(superResLastW == bmp.width && superResLastH == bmp.height && superResLastW > 0) &&
                                    bmp.width < 2048 && bmp.height < 2048,
                                modifier = Modifier.weight(1f).height(40.dp),
                                shape = RoundedCornerShape(10.dp),
                                contentPadding = PaddingValues(horizontal = 4.dp, vertical = 0.dp),
                                colors = ButtonDefaults.outlinedButtonColors(contentColor = Color(0xFFFF8F00))
                            ) {
                                Icon(Icons.Filled.ZoomIn, contentDescription = null, modifier = Modifier.size(13.dp), tint = Color(0xFFFF8F00))
                                Spacer(Modifier.width(3.dp))
                                Text(if (superResBusy) "超分中…" else "4x 超分", fontSize = 11.sp, color = Color(0xFFFF8F00), maxLines = 1, softWrap = false)
                            }
                            // ★ AI 重绘入口：一键执行（参数用设置页预设）。无模型包则先弹选择器；有/已部署则直接触发
                            OutlinedButton(
                                onClick = {
                                    val src = if (isHistory) viewerBitmap else (resultBitmap ?: viewerBitmap)
                                    if (src == null) return@OutlinedButton
                                    if (aiRedrawRunning) return@OutlinedButton
                                    // ★ 模型可用性：zip 在 或 已部署（zip 部署后自动删除）都算有模型
                                    if (!AiRedrawEngine.isModelReady(context)) {
                                        Toast.makeText(context, "请先选择 SD 模型包 zip", Toast.LENGTH_SHORT).show()
                                        aiModelZipPicker.launch("*/*")
                                        return@OutlinedButton
                                    }
                                    // ★ 整图一键重绘（执行块负责部署/启动/重绘/停止）
                                    aiRedrawRequestId++
                                },
                                enabled = bmp != null && !aiEngineBusy && !aiRedrawRunning,
                                modifier = Modifier.weight(1f).height(40.dp),
                                shape = RoundedCornerShape(10.dp),
                                contentPadding = PaddingValues(horizontal = 4.dp, vertical = 0.dp),
                                colors = ButtonDefaults.outlinedButtonColors(contentColor = Color(0xFF7B1FA2))
                            ) {
                                Icon(Icons.Filled.AutoAwesome, contentDescription = null, modifier = Modifier.size(13.dp), tint = Color(0xFF7B1FA2))
                                Spacer(Modifier.width(3.dp))
                                Text(
                                    if (aiRedrawRunning) "重绘中..." else "AI 重绘",
                                    fontSize = 11.sp, color = Color(0xFF7B1FA2), maxLines = 1, softWrap = false
                                )
                            }
                        }
                        // ★ 撤销/前进/按住对比（结果查看：编辑历史操作）
                        Row(horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
                            // ★ 撤销：100% 走全局撤销栈（undoStack/doUndo）。
                            //   已彻底移除旧的"查看器打开快照"基线分支（该分支恢复的是快照而非上一次操作，
                            //   与全局编辑历史语义冲突，会让撤销提示"已撤销到打开前版本"且无法前进）。
                            OutlinedButton(
                                onClick = {
                                    val prev = doUndo()
                                    if (prev != null && !prev.isRecycled) {
                                        resultBitmap = prev
                                        viewerBitmap = prev
                                        viewerItem = null   // ★ 撤销到全局历史中的某一步，脱离"历史记录"身份，与前进按钮行为一致
                                        viewerBgOrigin = null
                                        viewerBgStrength = 0
                                        resetPostVars()
                                        statusText = "已撤销"
                                    }
                                },
                                enabled = undoStack.isNotEmpty() && !isProcessing,
                                modifier = Modifier.weight(1f).height(40.dp),
                                shape = RoundedCornerShape(10.dp)
                            ) {
                                Icon(Icons.Filled.Undo, contentDescription = null, modifier = Modifier.size(15.dp))
                                Spacer(Modifier.width(4.dp))
                                Text("撤销", fontSize = 12.sp)
                            }
                            OutlinedButton(
                                onClick = {
                                    val next = doRedo()
                                    if (next != null) {
                                        resultBitmap = next
                                        viewerBitmap = next
                                        viewerItem = null
                                        viewerBgOrigin = null // ★ 切换查看对象，重置缓存
                                        viewerBgStrength = 0
                                        resetPostVars()
                                        statusText = "已前进"
                                    }
                                },
                                enabled = redoStack.isNotEmpty() && !isProcessing,
                                modifier = Modifier.weight(1f).height(40.dp),
                                shape = RoundedCornerShape(10.dp)
                            ) {
                                Icon(Icons.Filled.Redo, contentDescription = null, modifier = Modifier.size(15.dp))
                                Spacer(Modifier.width(4.dp))
                                Text("前进", fontSize = 12.sp)
                            }
                            // 小「按住对比」：按住显示原图，松开恢复结果
                            Box(
                                modifier = Modifier
                                    .size(width = 56.dp, height = 40.dp)
                                    .clip(RoundedCornerShape(10.dp))
                                    .border(1.dp, MaterialTheme.colorScheme.outline, RoundedCornerShape(10.dp))
                                    .pointerInput(originalBitmap != null && bmp != null && !isProcessing) {
                                        awaitEachGesture {
                                            awaitFirstDown(requireUnconsumed = false).also { comparePressed = true }
                                            waitForUpOrCancellation()
                                            comparePressed = false
                                        }
                                    },
                                contentAlignment = Alignment.Center
                            ) {
                                if (comparePressed) {
                                    Text("松开", fontSize = 10.sp, color = MaterialTheme.colorScheme.primary)
                                } else {
                                    Icon(Icons.Filled.Compare, contentDescription = "按住看原图", modifier = Modifier.size(16.dp), tint = MaterialTheme.colorScheme.primary)
                                }
                            }
                        }
                        // 第一行：再次抠图 + 保存相册（各半宽）
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Button(
                                onClick = {
                                    val b = viewerBitmap ?: if (isHistory) HistoryManager.loadResult(context, viewer!!) else null
                                    if (b != null) {
                                        originalBitmap = b
                                        resultBitmap = null
                                        // ★ 分支继承：历史记录再次抠图 → 继承该记录自己的 originalKey（A→B→C 同链）；
                                        //   非历史（当前会话结果）→ 保持 currentHistoryKey 不变（同样是继承）。
                                        viewer?.let { v ->
                                            if (v.originalKey.isNotBlank()) currentHistoryKey = v.originalKey
                                        }
                                        ProjectStateManager.save(context, b, null)
                                        viewerItem = null; viewerBitmap = null; viewerBgOrigin = null; viewerBgStrength = 0; viewerOriginBitmap = null; postProcessHistoryItem = null; resetPostVars()
                                        if (!isHistory) showOriginalViewer = false
                                        currentTab = 0
                                        statusText = if (isHistory) "已载入历史结果，可再次抠图或调整阈值" else "已载入结果，可再次抠图或调整阈值"
                                        Toast.makeText(context, "已载入，可再次抠图", Toast.LENGTH_SHORT).show()
                                    }
                                },
                                modifier = Modifier.weight(1f).height(40.dp),
                                shape = RoundedCornerShape(10.dp),
                                enabled = bmp != null
                            ) {
                                Icon(Icons.Filled.Refresh, contentDescription = null, modifier = Modifier.size(15.dp))
                                Spacer(Modifier.width(4.dp))
                                Text("再次抠图", fontSize = 13.sp)
                            }
                            OutlinedButton(
                                onClick = { saveViewerToAlbum(context, bmp) },
                                modifier = Modifier.weight(1f).height(40.dp),
                                shape = RoundedCornerShape(10.dp),
                                enabled = bmp != null
                            ) {
                                Icon(Icons.Filled.Download, contentDescription = null, modifier = Modifier.size(15.dp))
                                Spacer(Modifier.width(4.dp))
                                Text("保存相册", fontSize = 13.sp)
                            }
                        }
                        // 第二行：另存为 + 分享（整行各半）
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            OutlinedButton(
                                onClick = {
                                    val b = viewerBitmap ?: return@OutlinedButton
                                    viewerSaveLauncher.launch("rmbg_${System.currentTimeMillis()}.png")
                                },
                                modifier = Modifier.weight(1f).height(40.dp),
                                shape = RoundedCornerShape(10.dp),
                                enabled = bmp != null
                            ) {
                                Icon(Icons.Filled.SaveAlt, contentDescription = null, modifier = Modifier.size(15.dp))
                                Spacer(Modifier.width(4.dp))
                                Text("另存为", fontSize = 13.sp)
                            }
                            OutlinedButton(
                                onClick = {
                                    // ★ 历史记录：直接分享磁盘上已存的 PNG 文件（零重新编码，拉起快）
                                    if (isHistory && viewer != null) {
                                        shareHistoryResult(context, viewer)
                                    } else {
                                        shareResult(context, bmp)
                                    }
                                },
                                modifier = Modifier.weight(1f).height(40.dp),
                                shape = RoundedCornerShape(10.dp),
                                enabled = bmp != null
                            ) {
                                Icon(Icons.Filled.Share, contentDescription = null, modifier = Modifier.size(15.dp))
                                Spacer(Modifier.width(4.dp))
                                Text("分享", fontSize = 13.sp)
                            }
                        }
                    } else {
                        // 原图：高级编辑 + AI 重绘 + 4x超分 + 立即抠图 + 保存
                        // 第一行：高级编辑 + 4x超分 + AI 重绘（统一 40dp 高 / 10dp 圆角 / 6dp 间距，减小内边距防文字截断）
                        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                            Button(
                                onClick = {
                                    viewerItem = null; viewerBitmap = null; showOriginalViewer = false; viewerBgOrigin = null; viewerBgStrength = 0; viewerOriginBitmap = null; postProcessHistoryItem = null; resetPostVars()
                                    editorOnOriginal = true
                                    showEditor = true
                                },
                                modifier = Modifier.weight(1f).height(40.dp),
                                shape = RoundedCornerShape(10.dp),
                                contentPadding = PaddingValues(horizontal = 4.dp, vertical = 0.dp),
                                enabled = bmp != null && originalBitmap != null
                            ) {
                                Icon(Icons.Filled.Edit, contentDescription = null, modifier = Modifier.size(13.dp))
                                Spacer(Modifier.width(3.dp))
                                Text("高级编辑", fontSize = 11.sp, maxLines = 1, softWrap = false)
                            }
                            // ★★ 4x 超分（原图查看也可用，不入历史；已超分/超分中禁用防无限超分）
                            OutlinedButton(
                                onClick = { doSuperRes() },
                                modifier = Modifier.weight(1f).height(40.dp),
                                shape = RoundedCornerShape(10.dp),
                                contentPadding = PaddingValues(horizontal = 4.dp, vertical = 0.dp),
                                enabled = bmp != null && !superResBusy && !isProcessing &&
                                    !(superResLastW == bmp.width && superResLastH == bmp.height && superResLastW > 0) &&
                                    bmp.width < 2048 && bmp.height < 2048,
                                colors = ButtonDefaults.outlinedButtonColors(contentColor = Color(0xFFFF8F00))
                            ) {
                                Icon(Icons.Filled.ZoomIn, contentDescription = null, modifier = Modifier.size(13.dp), tint = Color(0xFFFF8F00))
                                Spacer(Modifier.width(3.dp))
                                Text(if (superResBusy) "超分中…" else "4x 超分", fontSize = 11.sp, color = Color(0xFFFF8F00), maxLines = 1, softWrap = false)
                            }
                            // ★ AI 重绘（原图）：一键执行（参数用设置页预设）。无模型包弹选择器；有/已部署则直接触发
                            OutlinedButton(
                                onClick = {
                                    if (aiRedrawRunning) return@OutlinedButton
                                    // ★ 模型可用性：zip 在 或 已部署（zip 部署后自动删除）都算有模型
                                    if (!AiRedrawEngine.isModelReady(context)) {
                                        Toast.makeText(context, "请先选择 SD 模型包 zip", Toast.LENGTH_SHORT).show()
                                        aiModelZipPicker.launch("*/*")
                                        return@OutlinedButton
                                    }
                                    // 有包/已部署 → 直接触发一键重绘（执行块负责部署/启动/重绘/停止）
                                    aiRedrawRequestId++
                                },
                                modifier = Modifier.weight(1f).height(40.dp),
                                shape = RoundedCornerShape(10.dp),
                                contentPadding = PaddingValues(horizontal = 4.dp, vertical = 0.dp),
                                enabled = bmp != null && !aiEngineBusy && !aiRedrawRunning,
                                colors = ButtonDefaults.outlinedButtonColors(contentColor = Color(0xFF7B1FA2))
                            ) {
                                Icon(Icons.Filled.AutoAwesome, contentDescription = null, modifier = Modifier.size(13.dp), tint = Color(0xFF7B1FA2))
                                Spacer(Modifier.width(3.dp))
                                Text(if (aiRedrawRunning) "重绘中..." else "AI 重绘", fontSize = 11.sp, color = Color(0xFF7B1FA2), maxLines = 1, softWrap = false)
                            }
                        }
                        // 第二行：立即抠图 + 保存相册 + 另存为（统一 40dp 高 / 10dp 圆角 / 6dp 间距，减小内边距防文字截断）
                        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                            OutlinedButton(
                                onClick = {
                                    viewerItem = null; viewerBitmap = null; showOriginalViewer = false; viewerBgOrigin = null; viewerBgStrength = 0; viewerOriginBitmap = null; postProcessHistoryItem = null; resetPostVars()
                                    runRmbg()
                                },
                                modifier = Modifier.weight(1f).height(40.dp),
                                shape = RoundedCornerShape(10.dp),
                                contentPadding = PaddingValues(horizontal = 4.dp, vertical = 0.dp),
                                enabled = bmp != null && originalBitmap != null && !isProcessing
                            ) {
                                Icon(Icons.Filled.AutoFixHigh, contentDescription = null, modifier = Modifier.size(13.dp))
                                Spacer(Modifier.width(3.dp))
                                Text("立即抠图", fontSize = 11.sp, maxLines = 1, softWrap = false)
                            }
                            OutlinedButton(
                                onClick = { saveViewerToAlbum(context, bmp) },
                                modifier = Modifier.weight(1f).height(40.dp),
                                shape = RoundedCornerShape(10.dp),
                                contentPadding = PaddingValues(horizontal = 4.dp, vertical = 0.dp),
                                enabled = bmp != null
                            ) {
                                Icon(Icons.Filled.Download, contentDescription = null, modifier = Modifier.size(13.dp))
                                Spacer(Modifier.width(3.dp))
                                Text("保存相册", fontSize = 11.sp, maxLines = 1, softWrap = false)
                            }
                            OutlinedButton(
                                onClick = {
                                    val b = viewerBitmap ?: return@OutlinedButton
                                    viewerSaveLauncher.launch("rmbg_${System.currentTimeMillis()}.png")
                                },
                                modifier = Modifier.weight(1f).height(40.dp),
                                shape = RoundedCornerShape(10.dp),
                                contentPadding = PaddingValues(horizontal = 4.dp, vertical = 0.dp),
                                enabled = bmp != null
                            ) {
                                Icon(Icons.Filled.SaveAlt, contentDescription = null, modifier = Modifier.size(13.dp))
                                Spacer(Modifier.width(3.dp))
                                Text("另存为", fontSize = 11.sp, maxLines = 1, softWrap = false)
                            }
                        }
                    }
                }
            }
        }
    }

    // ★ 查看器拆分方案A：独立全屏"后处理/画笔"编辑页覆盖层（从查看器拆出的新窗口）
    if (showPostProcessScreen) {
        val ppRes = postProcessResult
        if (ppRes != null && !ppRes.isRecycled) {
            PostProcessScreen(
                result = ppRes,
                originPx = viewerBgOrigin,
                originBitmap = viewerOriginBitmap ?: originalBitmap,
                onApply = { edited ->
                    if (!edited.isRecycled) {
                        pushUndo(resultBitmap)
                        resultBitmap = edited
                        viewerBitmap = edited
                        viewerBgOrigin = null
                        viewerBgStrength = 0
                        resetPostVars()
                        statusText = "已应用后处理/画笔"
                        Toast.makeText(context, "已应用后处理/画笔", Toast.LENGTH_SHORT).show()
                        // ★ 历史存档统一：后处理/抠图共用同一条历史记录、同一个原图存档。
                        //   从查看器进入且关联历史记录 → updateResult 覆盖同一记录（原图 orig 不变）；
                        //   否则（非历史来源）才新建一条记录。
                        val targetItem = postProcessHistoryItem ?: viewerItem
                        if (targetItem != null && targetItem.fileName.isNotEmpty()) {
                            scope.launch {
                                withContext(Dispatchers.IO) {
                                    try {
                                        HistoryManager.updateResult(context, targetItem, edited)?.let { updated ->
                                            viewerItem = updated
                                            historyVersion++
                                        }
                                    } catch (_: Exception) {}
                                }
                            }
                        } else {
                            scope.launch {
                                withContext(Dispatchers.IO) {
                                    try {
                                        HistoryManager.saveResult(context, edited, original = viewerOriginBitmap ?: originalBitmap, modelName = "后处理精修")?.let {
                                            viewerItem = it
                                            historyVersion++
                                        }
                                    } catch (_: Exception) {}
                                }
                            }
                        }
                        showPostProcessScreen = false
                        postProcessResult = null
                        postProcessHistoryItem = null
                    }
                },
                onBack = {
                    showPostProcessScreen = false
                    postProcessResult = null
                    postProcessHistoryItem = null
                }
            )
        } else {
            // 兜底：结果缺失则退回查看器
            showPostProcessScreen = false
            postProcessResult = null
        }
    }

    // ---- AI 重绘（LocalDream img2img，一键执行，参数来自设置页预设）----
    // 触发：查看器点 AI 重绘按钮 → aiRedrawRequestId 递增
    // ★ 重要：LaunchedEffect 必须常驻组合树（不能包在 if 里！）。
    //   若包在 if(aiRedrawRequestId>0) 里，协程内任何状态修改导致重组后 if 变 false，
    //   LaunchedEffect 离开组合 → 协程被取消 → finally 里挂起抛 CancellationException
    //   → aiRedrawRunning 卡死 true → 按钮永久禁用（"没动静"的根因）。
    //   key 本身已消费点击，协程内不需要归零。
    LaunchedEffect(aiRedrawRequestId) {
        if (aiRedrawRequestId == 0) return@LaunchedEffect
        // ★ 新的重绘开始：取消上次的空闲延迟停止（引擎继续待命，避免被旧 Job 打断）
        aiEngineStopJob?.cancel()
        aiEngineStopJob = null
        aiRedrawRunning = true
        aiRedrawStatus = ""
        // ★ 重绘源：查看器当前显示图优先（结果/历史），否则主界面原图
        val src = viewerBitmap ?: originalBitmap
        if (src == null) {
            aiRedrawRunning = false
            aiRedrawStatus = "❌ 没有可重绘的图片"
            return@LaunchedEffect
        }
        // ★ 档位归一化：QNN context 固定形状，仅支持模型包 patch 档位。
        //   旧版本可能存过滑杆产生的非法尺寸，这里归一化到最近档位，避免引擎形状不匹配报错。
        val VALID_RES = setOf(
            512 to 512, 512 to 768, 768 to 512, 768 to 768,
            768 to 1024, 1024 to 768, 1024 to 1024
        )
        val pW = Prefs.aiRedrawWidth
        val pH = Prefs.aiRedrawHeight
        val (normW, normH) = if (pW to pH in VALID_RES) pW to pH else {
            // 按原图长宽比就近选档位
            val sw = src.width
            val sh = src.height
            when {
                sw >= sh && sw > 768 -> 1024 to 768
                sw < sh && sh > 768 -> 768 to 1024
                sw >= sh && sw > 512 -> 768 to 512
                sw < sh && sh > 512 -> 512 to 768
                else -> 512 to 512
            }
        }
        // 用设置页预设参数构建请求
        val req = LocalDreamClient.RedrawRequest(
            prompt = Prefs.aiRedrawPrompt,
            negativePrompt = Prefs.aiRedrawNegative,
            steps = Prefs.aiRedrawSteps,
            cfg = Prefs.aiRedrawCfg,
            width = normW,
            height = normH,
            scheduler = Prefs.aiRedrawScheduler,
            denoiseStrength = Prefs.aiRedrawDenoise,
            imageBase64 = ""
        )
        // ★ 编码输入图 → PNG base64
        //   关键优化：先把原图缩放到输出档位（normW×normH）再编码，
        //   否则大图(如 4000×3000)直接编码传输会浪费几十倍时间/内存，
        //   引擎 img2img 也只需输出档位尺寸的输入。
        val imgB64 = try {
            val scaled = scaleToFit(src, normW, normH)
            LocalDreamClient.bitmapToPngBase64(scaled)
        } catch (e: Exception) {
            aiRedrawRunning = false
            aiRedrawStatus = "❌ 图片编码失败: ${e.message}"
            return@LaunchedEffect
        }
        val fullReq = req.copy(imageBase64 = imgB64)
        withContext(Dispatchers.IO) {
            try {
                // ★ 推理后端分支：
                //   "local" = 本地自加载引擎（AiRedrawEngine 启动进程）【仅完整版 normal】
                //   "dream" = 已安装的 LocalDream App（复用其 8081，不部署模型）
                // ★ Lite 精简版：未打包本地引擎资产（qnnlibs/sd_core），强制走 LocalDream
                val backend = if (BuildConfig.IS_LITE) "dream" else Prefs.aiRedrawBackend
                if (backend == "dream") {
                    // ---- LocalDream App 后端：直接调其 8081，不启动本地引擎 ----
                    if (!LocalDreamClient.checkOnline()) {
                        throw RuntimeException("未检测到 LocalDream App 的 8081 服务，请先在 LocalDream 中完成加载（同一 WiFi/本机）")
                    }
                    aiEngineStatus = "使用 LocalDream App（8081 在线）"
                } else {
                    // ---- 本地自加载引擎：确保就绪（未运行则自动启动）----
                    if (!AiRedrawEngine.checkHealth()) {
                        aiEngineStatus = "启动 AI 引擎中..."
                        // ★ 恢复持久化模型（App 重启后兜底；zip 部署后可能已删，靠 activeModelId 定位）
                        AiRedrawEngine.restoreModelZip(context)
                        // ★ 模型可用性：zip 在 或 已部署（zip 部署后自动删除）都算有模型
                        if (!AiRedrawEngine.isModelReady(context)) {
                            throw RuntimeException("未选择 SD 模型包，请先到模型页选择 zip（或切换推理后端为 LocalDream App）")
                        }
                        // ★ 关键优化：zip 存在且未部署 → 才解压；zip 已删但已部署 → 直接复用部署目录
                        val zip = AiRedrawEngine.modelZipFile
                        if (zip?.exists() == true && !AiRedrawEngine.isModelDeployed(context, zip)) {
                            aiEngineStatus = "首次使用，解压部署模型中（约 1GB，请稍候）..."
                            val ok = AiRedrawEngine.deployModel(context, zip)
                            if (!ok) throw RuntimeException("模型部署失败: ${AiRedrawEngine.status}")
                            deployedModelsVersion++  // ★ 部署完成刷新已部署模型列表
                        }
                        // QNN 运行时（幂等，已存在则跳过）
                        AiRedrawEngine.deployRuntime(context)
                        val started = AiRedrawEngine.start(context, normW, normH)
                        if (!started) {
                            throw RuntimeException("AI 引擎启动失败: ${AiRedrawEngine.status}")
                        }
                    }
                }
                val res = LocalDreamClient.redraw(fullReq)
                withContext(Dispatchers.Main) {
                    if (!res.bitmap.isRecycled) {
                        pushUndo(resultBitmap)
                        resultBitmap = res.bitmap
                        // ★ 查看器场景：同时更新查看器显示
                        viewerBitmap = res.bitmap
                        // ★ 重绘结果已入全局 undo 栈（pushUndo(resultBitmap) 在上方），
                        //   查看器撤销/前进统一走 undoStack/redoStack，无基线分支需清理。
                        viewerBgOrigin = null
                        statusText = "AI 重绘完成（${res.width}×${res.height} · ${"%.1f".format(res.generationTimeMs / 1000f)}s）"
                        aiRedrawStatus = "✅ 重绘完成 ${res.width}×${res.height}，用时 ${"%.1f".format(res.generationTimeMs / 1000f)} 秒"
                        aiEngineStatus = if (backend == "dream") "LocalDream App" else AiRedrawEngine.status
                        // 写入历史（复用后处理精修通道，保留原图）
                        HistoryManager.saveResult(
                            context, res.bitmap,
                            original = src,
                            modelName = if (backend == "dream") "AI重绘(LocalDream)" else "AI重绘",
                            durationMs = res.generationTimeMs,
                            accel = if (backend == "dream") "LocalDream" else "LocalDream",
                            // ★ 分支继承：重绘是结果链的延伸，归当前会话分支（A→B→C 同链）
                            originalKey = currentHistoryKey.ifBlank { computeOriginalKey(src) }
                        )?.let { historyVersion++ }
                    } else {
                        aiRedrawStatus = "❌ 重绘结果已失效"
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    aiRedrawStatus = "❌ 重绘失败: ${e.message}"
                }
            } finally {
                // ★ 重绘完成后不立即停引擎（可能有第二次重绘）：
                //   取消旧延迟停止 Job，启动新的空闲超时 Job（默认 60s 无新重绘才停）。
                //   LocalDream 后端不动它的进程；Lite 精简版只用 LocalDream，无本地引擎可停。
                aiEngineStopJob?.cancel()
                if (!BuildConfig.IS_LITE && Prefs.aiRedrawBackend != "dream") {
                    aiEngineStopJob = scope.launch {
                        try {
                            delay(AI_REDRAW_IDLE_STOP_MS)
                            AiRedrawEngine.stop(context)
                            aiEngineStatus = "引擎已空闲停止（${AI_REDRAW_IDLE_STOP_MS / 1000}s 无操作）"
                        } catch (ce: kotlinx.coroutines.CancellationException) {
                            // 被取消 = 用户又发起了新的重绘，引擎继续待命，不停止
                        } catch (_: Exception) {}
                    }
                }
                withContext(Dispatchers.Main) {
                    aiRedrawRunning = false
                }
            }
        }
    }
}

/**
 * 等比缩放完整放入目标尺寸（contain 模式，保持宽高比，不裁剪）。
 * 用于 AI 重绘前把原图缩放到输出档位（512×512 等）：
 *  - 完整保留构图（cover 模式会裁掉边缘导致"差异太大"）
 *  - 剩余区域填黑（img2img 对黑边区域基本不动，符合修复定位）
 *  - 避免大图编码传输浪费
 */
private fun scaleToFit(src: Bitmap, targetW: Int, targetH: Int): Bitmap {
    if (src.width == targetW && src.height == targetH) return src
    // 等比缩放：取目标/源的最小比例，使整图完整放入目标（可能留边）
    val scale = minOf(targetW.toFloat() / src.width, targetH.toFloat() / src.height)
    val sw = (src.width * scale).toInt().coerceAtLeast(1)
    val sh = (src.height * scale).toInt().coerceAtLeast(1)
    val scaled = Bitmap.createScaledBitmap(src, sw, sh, true)
    if (sw == targetW && sh == targetH) return scaled
    // 居中放入画布，四周补黑
    return try {
        val canvas = android.graphics.Canvas()
        val out = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
        canvas.setBitmap(out)
        canvas.drawColor(android.graphics.Color.BLACK)
        val x = (targetW - sw) / 2
        val y = (targetH - sh) / 2
        canvas.drawBitmap(scaled, x.toFloat(), y.toFloat(), null)
        if (scaled !== src) scaled.recycle()
        out
    } catch (e: Exception) {
        scaled // 失败回退
    }
}

/**
 * contain 等比缩放完整放入目标尺寸，剩余区域用特殊填充色（区域重绘专用）。
 * 用于区域重绘：选区比例与档位不一致时，
 *  - 不裁剪内容（保留完整构图信息）
 *  - 剩余区域填特殊色（亮品红 0xFFFF00FF），img2img 后该色在合并时转透明露出原图
 *  - 彻底避免"黑边"（黑边会留在结果里）
 */
private fun scaleToContainKey(src: Bitmap, targetW: Int, targetH: Int, fillColor: Int): Bitmap {
    if (src.width == targetW && src.height == targetH) return src
    // 等比缩放：取目标/源的最小比例，使整图完整放入目标（可能留边）
    val scale = minOf(targetW.toFloat() / src.width, targetH.toFloat() / src.height)
    val sw = (src.width * scale).toInt().coerceAtLeast(1)
    val sh = (src.height * scale).toInt().coerceAtLeast(1)
    val scaled = Bitmap.createScaledBitmap(src, sw, sh, true)
    if (sw == targetW && sh == targetH) return scaled
    // 居中放入画布，四周填特殊色
    return try {
        val canvas = android.graphics.Canvas()
        val out = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
        canvas.setBitmap(out)
        canvas.drawColor(fillColor)
        val x = (targetW - sw) / 2
        val y = (targetH - sh) / 2
        canvas.drawBitmap(scaled, x.toFloat(), y.toFloat(), null)
        if (scaled !== src) scaled.recycle()
        out
    } catch (e: Exception) {
        scaled // 失败回退
    }
}

/** 区域重绘填充色：亮品红（正常图片中少见，做键控色不易误伤主体） */
private const val AI_REDRAW_FILL_COLOR = 0xFFFF00FF.toInt()

/** ★ AI 重绘引擎空闲超时（毫秒）：重绘完成后不立即停，等这么久没新重绘才停（支持连续二次重绘） */
private const val AI_REDRAW_IDLE_STOP_MS = 60_000L

/**
 * 把位图中接近特殊填充色的像素置为透明（alpha=0）。
 * 用于区域重绘合并：img2img 结果里填充色（含轻微扩散噪声）转透明，
 * 贴回原图时该处露出原图，无黑边、无裁剪。
 */
private fun makeKeyTransparent(bmp: Bitmap, keyColor: Int, tolerance: Int = 70): Bitmap {
    val w = bmp.width; val h = bmp.height
    val out = bmp.copy(Bitmap.Config.ARGB_8888, true)
    val kr = (keyColor shr 16) and 0xFF
    val kg = (keyColor shr 8) and 0xFF
    val kb = keyColor and 0xFF
    val pixels = IntArray(w * h)
    out.getPixels(pixels, 0, w, 0, 0, w, h)
    val tol2 = tolerance * tolerance
    for (i in pixels.indices) {
        val c = pixels[i]
        val r = (c shr 16) and 0xFF
        val g = (c shr 8) and 0xFF
        val b = c and 0xFF
        // ★ 与填充色欧氏距离在容差内 → 透明
        val dr = r - kr; val dg = g - kg; val db = b - kb
        if (dr * dr + dg * dg + db * db <= tol2) {
            pixels[i] = c and 0x00FFFFFF // 保留 RGB，alpha=0
        }
    }
    out.setPixels(pixels, 0, w, 0, 0, w, h)
    return out
}

/**
 * AI 重绘 · 区域执行（框选局部 → 裁出 → 缩档位 → img2img → 贴回）。
 * 解决整图缩到档位（512×512）分辨率低：选区是局部，512 对应局部放大细节，贴回不糊。
 * @param src 源图（查看器当前显示的图）
 * @param sel [L, T, R, B] 像素坐标（已按源图换算）
 * @return 合成结果图（源图尺寸）；失败返回 null
 */
suspend fun aiRedrawRegion(src: Bitmap, sel: FloatArray): Bitmap? {
    // sel = [L, T, R, B] 像素坐标（已按源图换算）
    val bw = src.width; val bh = src.height
    val L = sel[0].toInt().coerceIn(0, bw - 1)
    val T = sel[1].toInt().coerceIn(0, bh - 1)
    val R = sel[2].toInt().coerceIn(L + 1, bw)
    val B = sel[3].toInt().coerceIn(T + 1, bh)
    if (R - L < 2 || B - T < 2) return null
    // 1. 裁出选区
    val region = Bitmap.createBitmap(src, L, T, R - L, B - T)
    // 2. 归一化档位（QNN 固定形状）
    val VALID_RES = setOf(
        512 to 512, 512 to 768, 768 to 512, 768 to 768,
        768 to 1024, 1024 to 768, 1024 to 1024
    )
    val pW = Prefs.aiRedrawWidth
    val pH = Prefs.aiRedrawHeight
    val (normW, normH) = if (pW to pH in VALID_RES) pW to pH else 512 to 512
    // 3. 选区 contain 等比缩放放到档位，四周填特殊色（不裁剪，合并时该色转透明露出原图）
    // ★ 修复④：记录 contain 缩放的内容偏移（contentX/Y/W/H），贴回前裁出有效内容区再等比缩放，
    //   避免整张 normW×normH（含透明边）被 createScaledBitmap 非等比拉伸到 patchW×patchH 导致变形。
    val regW = region.width; val regH = region.height
    val containScale = minOf(normW.toFloat() / regW, normH.toFloat() / regH)
    val contentW = (regW * containScale).toInt().coerceAtLeast(1)
    val contentH = (regH * containScale).toInt().coerceAtLeast(1)
    val contentX = (normW - contentW) / 2
    val contentY = (normH - contentH) / 2
    val scaled = scaleToContainKey(region, normW, normH, AI_REDRAW_FILL_COLOR)
    val imgB64 = try {
        LocalDreamClient.bitmapToPngBase64(scaled)
    } finally {
        if (scaled !== region && !scaled.isRecycled) scaled.recycle()
        if (region !== src && !region.isRecycled) region.recycle()
    }
    // 4. 构建请求（用设置页预设提示词）
    val req = LocalDreamClient.RedrawRequest(
        prompt = Prefs.aiRedrawPrompt,
        negativePrompt = Prefs.aiRedrawNegative,
        steps = Prefs.aiRedrawSteps,
        cfg = Prefs.aiRedrawCfg,
        width = normW,
        height = normH,
        scheduler = Prefs.aiRedrawScheduler,
        denoiseStrength = Prefs.aiRedrawDenoise,
        imageBase64 = imgB64
    )
    // 5. 调用后端（local 引擎由外层 ensure 启动；这里只管 redraw）
    val res = LocalDreamClient.redraw(req)
    val outBmp = res.bitmap
    if (outBmp.isRecycled) return null
    // 6. 重绘结果：特殊填充色转透明（露出原图），缩回选区尺寸后贴回源图
    val patchW = R - L; val patchH = B - T
    // ★ 键控透明：把接近填充色的像素 alpha 置 0（img2img 残留的品红区域露出原图，无黑边）
    val keyed = makeKeyTransparent(outBmp, AI_REDRAW_FILL_COLOR)
    // ★ 修复④：先裁出有效内容区（去掉品红/透明边），再等比缩放到选区尺寸。
    //   内容区宽高比 = region 原始宽高比 = patchW:patchH，等比缩放不变形。
    //   旧版直接 createScaledBitmap(keyed, patchW, patchH) 把含边的整张拉伸 → 长宽比失真。
    val patch = try {
        // 裁出内容区（contentX/Y/W/H 是 contain 缩放时内容在 normW×normH 画布中的居中偏移）
        val cx = contentX.coerceIn(0, keyed.width - 1)
        val cy = contentY.coerceIn(0, keyed.height - 1)
        val cw = contentW.coerceIn(1, keyed.width - cx)
        val ch = contentH.coerceIn(1, keyed.height - cy)
        val cropped = if (cx == 0 && cy == 0 && cw == keyed.width && ch == keyed.height) keyed
                      else Bitmap.createBitmap(keyed, cx, cy, cw, ch)
        // 等比缩放到选区尺寸（cw:ch == patchW:patchH，不变形）
        if (cropped.width == patchW && cropped.height == patchH) cropped
        else Bitmap.createScaledBitmap(cropped, patchW, patchH, true).also {
            if (cropped !== keyed && !cropped.isRecycled) cropped.recycle()
        }
    } catch (_: Exception) {
        // 兜底：裁切失败时回退到旧逻辑（不理想但不崩）
        if (keyed.width == patchW && keyed.height == patchH) keyed
        else Bitmap.createScaledBitmap(keyed, patchW, patchH, true)
    }
    return try {
        val canvas = android.graphics.Canvas()
        val out = Bitmap.createBitmap(bw, bh, Bitmap.Config.ARGB_8888)
        canvas.setBitmap(out)
        canvas.drawBitmap(src, 0f, 0f, null)
        canvas.drawBitmap(patch, L.toFloat(), T.toFloat(), null)
        if (patch !== keyed && !patch.isRecycled) patch.recycle()
        if (keyed !== outBmp && !keyed.isRecycled) keyed.recycle()
        if (outBmp !== keyed && !outBmp.isRecycled) outBmp.recycle()
        out
    } catch (e: Exception) {
        null
    }
}

/** 直接分享磁盘上的 PNG 文件（零重新编码，拉起最快，不入公开相册） */
private fun shareHistoryFile(context: android.content.Context, f: File) {
    try {
        if (!f.exists() || f.length() < 100L) { shareResult(context, null); return }
        val uri = FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", f)
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "image/png"
            putExtra(Intent.EXTRA_STREAM, uri)
            clipData = android.content.ClipData.newUri(context.contentResolver, "rmbg", uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        context.startActivity(Intent.createChooser(intent, "分享抠图结果"))
    } catch (e: Exception) {
        Toast.makeText(context, "分享失败: ${e.message}", Toast.LENGTH_SHORT).show()
    }
}

/** 直接分享历史记录的磁盘 PNG 文件（历史查看器用） */
private fun shareHistoryResult(context: android.content.Context, item: HistoryManager.HistoryItem) {
    if (item.fileName.isEmpty()) { shareResult(context, null); return }
    val dir = File(context.filesDir, "history")
    shareHistoryFile(context, File(dir, item.fileName))
}

/** 分享抠图结果到外部 app（用 app 私有缓存 + FileProvider，不入公开相册；IO 后台执行避免主线程卡顿） */
private fun shareResult(context: android.content.Context, bmp: Bitmap?) {
    if (bmp == null) return
    val scope = kotlinx.coroutines.CoroutineScope(kotlinx.coroutines.Dispatchers.IO + kotlinx.coroutines.SupervisorJob())
    scope.launch {
        try {
            // ★ 分享前限制尺寸，控制 PNG 体积（兼顾预览速度与体积）
            val shareBmp = downscaleForShare(bmp, maxSide = 1024)
            val dir = File(context.cacheDir, "shared").apply { mkdirs() }
            val f = File(dir, "rmbg_share_${System.currentTimeMillis()}.png")
            FileOutputStream(f).use { os -> shareBmp.compress(Bitmap.CompressFormat.PNG, 100, os) }
            if (shareBmp !== bmp) shareBmp.recycle()
            val uri = FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", f)
            withContext(kotlinx.coroutines.Dispatchers.Main) {
                val intent = Intent(Intent.ACTION_SEND).apply {
                    type = "image/png"
                    putExtra(Intent.EXTRA_STREAM, uri)
                    clipData = android.content.ClipData.newUri(context.contentResolver, "rmbg", uri)
                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION or Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION)
                }
                context.startActivity(Intent.createChooser(intent, "分享抠图结果"))
            }
        } catch (e: Exception) {
            kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.Main) {
                Toast.makeText(context, "分享失败: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        }
    }
}

/** 分享/缩略预览用：把位图最长边缩到 maxSide，保持透明通道（用于控制分享 PNG 体积） */
private fun downscaleForShare(bmp: Bitmap, maxSide: Int): Bitmap {
    val w = bmp.width
    val h = bmp.height
    val longest = maxOf(w, h)
    if (longest <= maxSide) return bmp
    val scale = maxSide.toFloat() / longest
    val nw = (w * scale).toInt().coerceAtLeast(1)
    val nh = (h * scale).toInt().coerceAtLeast(1)
    return Bitmap.createScaledBitmap(bmp, nw, nh, true)
}

/** 保存到应用默认文件夹 Pictures/rmbg（Android 10+ 用 MediaStore 无权限写公共目录） */
private fun saveToRmbgFolder(context: android.content.Context, bmp: Bitmap?): Boolean {
    if (bmp == null) return false
    return try {
        val values = android.content.ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, "rmbg_${System.currentTimeMillis()}.png")
            put(MediaStore.Images.Media.MIME_TYPE, "image/png")
            put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/rmbg")
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }
        val uri = context.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
        if (uri != null) {
            context.contentResolver.openOutputStream(uri)?.use { os ->
                bmp.compress(Bitmap.CompressFormat.PNG, 100, os)
            }
            context.contentResolver.update(
                uri,
                android.content.ContentValues().apply { put(MediaStore.Images.Media.IS_PENDING, 0) },
                null, null
            )
            true
        } else false
    } catch (_: Exception) { false }
}

/** 保存查看器中的图片到应用默认文件夹 Pictures/rmbg（无权限） */
private fun saveViewerToAlbum(context: android.content.Context, bmp: Bitmap?) {
    if (bmp == null) return
    val ok = saveToRmbgFolder(context, bmp)
    Toast.makeText(
        context,
        if (ok) "已保存到相册 rmbg 文件夹" else "保存失败",
        Toast.LENGTH_SHORT
    ).show()
}

/** 解码图片（保留高分辨率，最大边 4096px；防 OOM 但仍保细节） */
private fun decodeBitmap(context: android.content.Context, uri: Uri): Bitmap {
    // file:// 直接按路径解码（无需权限）；content:// 走 resolver
    if (uri.scheme == "file") {
        val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(uri.path, opts)
        val maxDim = 4096
        var sample = 1
        while (opts.outWidth / sample > maxDim || opts.outHeight / sample > maxDim) {
            sample *= 2
        }
        val fopts = BitmapFactory.Options().apply {
            inSampleSize = sample
            inPreferredConfig = Bitmap.Config.ARGB_8888 // ★ 强制软位图：杜绝 HARDWARE config 导致 getPixels/setPixels 抛异常
        }
        return BitmapFactory.decodeFile(uri.path, fopts)
            ?: throw IllegalStateException("无法解码图片")
    }
    val resolver = context.contentResolver
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    resolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }
    // 4096 足够覆盖主流相机照片（4000x3000），保留细节不糊
    val maxDim = 4096
    var sample = 1
    while (bounds.outWidth / sample > maxDim || bounds.outHeight / sample > maxDim) {
        sample *= 2
    }
    val opts = BitmapFactory.Options().apply {
        inSampleSize = sample
        inPreferredConfig = Bitmap.Config.ARGB_8888 // ★ 强制软位图
    }
    return resolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, opts) }
        ?: throw IllegalStateException("无法解码图片")
}

/** 格式化文件大小 */
private fun formatSize(bytes: Long): String {
    if (bytes <= 0) return "0 B"
    val kb = bytes / 1024.0
    val mb = kb / 1024.0
    val gb = mb / 1024.0
    return when {
        gb >= 1 -> String.format("%.2f GB", gb)
        mb >= 1 -> String.format("%.1f MB", mb)
        else -> String.format("%.0f KB", kb)
    }
}

/** 格式化下载速度 */
private fun formatSpeed(bps: Long): String {
    if (bps <= 0) return ""
    val kb = bps / 1024.0
    val mb = kb / 1024.0
    return if (mb >= 1) String.format("%.1f MB/s", mb) else String.format("%.0f KB/s", kb)
}

/**
 * 计算原图分组键：基于原图像素内容生成稳定哈希。
 * 同一张原图（内容一致）反复抠图得到相同 key → 历史可归纳为一组折叠记录。
 * 用宽高 + 固定采样网格的像素值拼接（轻量，避免整图大扫描）。
 */
private fun computeOriginalKey(bmp: android.graphics.Bitmap): String {
    if (bmp == null || bmp.isRecycled) return ""
    return try {
        val w = bmp.width
        val h = bmp.height
        // 采样 4×4 网格 + 四角，兼顾速度与区分度
        val sb = StringBuilder("$w x $h|")
        fun sample(fx: Float, fy: Float) {
            val x = (fx * w).toInt().coerceIn(0, w - 1)
            val y = (fy * h).toInt().coerceIn(0, h - 1)
            sb.append(bmp.getPixel(x, y)).append(',')
        }
        sample(0f, 0f); sample(0.5f, 0f); sample(1f, 0f)
        sample(0f, 0.5f); sample(0.5f, 0.5f); sample(1f, 0.5f)
        sample(0f, 1f); sample(0.5f, 1f); sample(1f, 1f)
        sb.toString()
    } catch (_: Exception) { "" }
}

/** 递归统计目录占用（字节）。不存在或非目录返回 0 */
private fun folderSize(file: File): Long {
    if (!file.exists()) return 0L
    return if (file.isDirectory) {
        var sum = 0L
        val children = file.listFiles() ?: return 0L
        for (f in children) {
            try { sum += folderSize(f) } catch (_: Exception) {}
        }
        sum
    } else {
        file.length()
    }
}

/** 存储占用检查：统计安装包、native 库、内部数据、外部模型、缓存，返回多行报告文本
 *  ★ 对齐系统"应用信息→存储"的统计口径，让 App 显示 ≈ 系统显示 */
private fun checkStorageUsage(context: android.content.Context): String {
    val sb = StringBuilder()
    fun line(name: String, dir: File?) {
        val size = dir?.let { folderSize(it) } ?: 0L
        sb.append("$name: ${formatSize(size)}\n")
    }

    // ① APK 安装包本体
    var apkSize = 0L
    try { apkSize = File(context.applicationInfo.sourceDir ?: "").length() } catch (_: Exception) {}
    sb.append("APK 安装包: ${formatSize(apkSize)}\n")

    // ② native 库解压目录（useLegacyPackaging=true 时有实体文件，JNI/QNN/ORT 运行时）
    var nativeSize = 0L
    try {
        val ndir = File(context.applicationInfo.nativeLibraryDir ?: "")
        if (ndir.exists()) nativeSize = folderSize(ndir)
    } catch (_: Exception) {}
    sb.append("Native 库目录: ${formatSize(nativeSize)}\n")

    // ③ 内部数据目录（data/data/<pkg>/files）：细分子项，方便定位大头
    val filesDir = context.filesDir
    val qnnlibs = File(filesDir, "qnnlibs")      // AI 重绘 QNN 运行时（132MB，deployRuntime 复制）
    val innerModels = File(filesDir, "models")   // AI 重绘 SD 模型（AnythingV5/MeinaMix 各 1.2GB+）
    val aiModels = File(filesDir, "ai_models")   // AI 重绘模型 zip 源包
    val innerHistory = File(filesDir, "history")
    val work = File(filesDir, "work")
    sb.append("内部数据(files)\n")
    line("  ├ QNN 运行时(qnnlibs)", qnnlibs)
    line("  ├ AI 模型(models)", innerModels)
    line("  ├ 模型包(ai_models)", aiModels)
    line("  ├ 历史(history)", innerHistory)
    line("  ├ 工作区(work)", work)
    // 其余 files 直接子文件
    var filesOther = 0L
    try {
        filesDir.listFiles()?.forEach { f ->
            if (f.isFile) filesOther += f.length()
        }
    } catch (_: Exception) {}
    sb.append("  └ 其他文件: ${formatSize(filesOther)}\n")

    // ④ 外部模型目录（Android/data/<pkg>/files/models，抠图模型）
    line("外部模型(Android/data)", com.rmbg.offline.ml.ModelManager.modelDir())

    // ⑤ 缓存目录
    line("缓存目录", context.cacheDir)

    // 汇总（与系统口径一致：APK + native + 内部 data + 外部 data）
    val total = listOf(
        apkSize, nativeSize, folderSize(filesDir), folderSize(context.cacheDir),
        folderSize(com.rmbg.offline.ml.ModelManager.modelDir())
    ).sum()
    sb.append("────────────────\n")
    sb.append("合计: ${formatSize(total)}")
    return sb.toString().trim()
}

/** 清理 App 缓存目录（cacheDir）内容，返回释放的字节数（不触碰历史/模型等数据） */
private fun clearAppCache(context: android.content.Context): Long {
    val cacheDir = context.cacheDir
    if (!cacheDir.exists()) return 0L
    var freed = 0L
    val children = cacheDir.listFiles() ?: return 0L
    for (f in children) {
        try {
            if (f.isDirectory) {
                freed += folderSize(f)
                f.deleteRecursively()
            } else {
                freed += f.length()
                f.delete()
            }
        } catch (_: Exception) {}
    }
    return freed
}
