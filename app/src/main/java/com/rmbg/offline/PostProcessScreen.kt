package com.rmbg.offline

import android.graphics.Bitmap
import android.widget.Toast
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.TransformOrigin
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.ui.draw.clip
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.unit.IntOffset
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.math.min
import kotlin.math.roundToInt
import com.rmbg.offline.ml.SuperResEngine

/**
 * 独立的后处理/画笔编辑页面（从查看器拆出的"编辑功能新窗口"）。
 *  - 结果大图预览（棋盘背景 + 双指缩放/平移）
 *  - 背景强度滑杆（写 alpha）
 *  - 边缘后处理（去色边/柔化/收缩/背景硬化）
 *  - 画笔/橡皮（扣空/恢复，复用 EditBrushLayer）
 *  - 「应用」写回结果，「还原」回到打开时原始像素
 *  @param result 当前结果图（可编辑副本，本页面就地修改其像素）
 *  @param originPx 打开时缓存的原始像素（供还原 + 后处理基于原始重建）
 *  @param onApply 应用编辑后的结果（写回主界面 resultBitmap + 保存）
 *  @param onBack 返回查看器
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PostProcessScreen(
    result: Bitmap,
    originPx: IntArray?,
    originBitmap: Bitmap? = null,
    onApply: (Bitmap) -> Unit = {},
    onBack: () -> Unit = {},
    initialBrushMode: Boolean = false  // ★ 从画笔入口进入时自动切画笔模式
) {
    val context = androidx.compose.ui.platform.LocalContext.current
    val scope = rememberCoroutineScope()
    // 显示/编辑的结果（本地可变副本）
    // ★★ 必须确保「可变(mutable)软件位图」：Canvas(bitmap)/setPixels 都要求 mutable。
    //    BitmapFactory 解码和历史加载的位图默认 immutable！Canvas(immutable) 构造即抛异常被吞
    //    → "手动恢复和抠图都失效"。凡不可变或 HARDWARE 一律复制成可变软副本。
    var displayBitmap by remember {
        mutableStateOf(
            // ★ 基底优先用 originPx（查看器缓存的未硬化 soft 像素，保留 1..254 半透明边缘 alpha）：
            //   后处理"去色边/条件收缩"的算法目标是半透明边缘，若直接拿硬化后的 result
            //   （alpha 只有 0/255）当基底，滑块拉满也找不到可处理的像素 → "没效果"。
            //   尺寸必须与 result 一致才可用。
            if (originPx != null && originPx.size == result.width * result.height && result.width > 0 && result.height > 0)
                Bitmap.createBitmap(result.width, result.height, Bitmap.Config.ARGB_8888).also {
                    it.setPixels(originPx, 0, result.width, 0, 0, result.width, result.height)
                }
            else if (result.isMutable && result.config != android.graphics.Bitmap.Config.HARDWARE)
                result
            else result.copy(android.graphics.Bitmap.Config.ARGB_8888, true)
        )
    }
    // ★ 原图恢复源同样可能 immutable/HARDWARE：作为恢复画笔的 RGB 源先转软位图
    val softOriginBitmap = remember(originBitmap) {
        if (originBitmap != null &&
            (originBitmap.config == android.graphics.Bitmap.Config.HARDWARE || !originBitmap.isMutable))
            originBitmap.copy(android.graphics.Bitmap.Config.ARGB_8888, true)
        else originBitmap
    }
    var brushVersion by remember { mutableStateOf(0) }

    // 背景强度
    var bgStrength by remember { mutableStateOf(0) }
    // ★ 4x 超分（独立按钮：点击立即超分并替换显示，不随"应用"自动执行；v4 实测链路）
    var superResBusy by remember { mutableStateOf(false) }
    var superResFail by remember { mutableStateOf("") }   // ★ 超分失败具体原因（透传给 Toast）
    // ★ 超分模型来源显示：当前是否用导入模型（true=导入模型，false=内置 assets）
    var superResImported by remember { mutableStateOf(com.rmbg.offline.Prefs.superResModelPath.isNotEmpty() && java.io.File(com.rmbg.offline.Prefs.superResModelPath).exists()) }
    // ★ 导入超分模型（SAF 选 .zip 或 .onnx：QNN 用 onnx+bin zip，CPU 可单 .onnx）
    val importSuperResLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: android.net.Uri? ->
        uri ?: return@rememberLauncherForActivityResult
        scope.launch {
            val result = withContext(Dispatchers.IO) {
                try {
                    val displayName = try {
                        context.contentResolver.query(uri, arrayOf(android.provider.OpenableColumns.DISPLAY_NAME), null, null, null)
                            ?.use { c -> if (c.moveToFirst()) c.getString(0) else null }
                            ?: "super_res_model"
                    } catch (_: Exception) { "super_res_model" }
                    val tmp = java.io.File(context.cacheDir, "sr_import_${System.currentTimeMillis()}.${if (displayName.endsWith(".zip", true)) "zip" else "onnx"}")
                    context.contentResolver.openInputStream(uri)?.use { input -> tmp.outputStream().use { out -> input.copyTo(out) } }
                    if (!tmp.exists() || tmp.length() == 0L) { tmp.delete(); null } else {
                        // ★ zip（QNN EPContext onnx+bin / 普通 ONNX）→ 走超分专用导入（隔离目录，不混入抠图模型）
                        if (displayName.endsWith(".zip", true)) {
                            val path = com.rmbg.offline.ml.ModelManager.importSuperResModelZip(tmp)
                            tmp.delete()
                            path
                        } else {
                            // ★ 单个 .onnx（CPU 普通模型）：复制到 modelDir/superres/superres_model.onnx
                            val srDir = java.io.File(com.rmbg.offline.ml.ModelManager.modelDir(), "superres").apply { mkdirs() }
                            val dest = java.io.File(srDir, "superres_model.onnx")
                            tmp.copyTo(dest, overwrite = true)
                            tmp.delete()
                            if (dest.exists()) dest.absolutePath else null
                        }
                    }
                } catch (e: Exception) {
                    android.util.Log.e("RMBG-SUPER", "导入超分模型失败", e)
                    null
                }
            }
            if (result != null && java.io.File(result).exists()) {
                com.rmbg.offline.Prefs.superResModelPath = result
                superResImported = true
                Toast.makeText(context, "超分模型已导入，4x 超分将使用它", Toast.LENGTH_LONG).show()
            } else {
                Toast.makeText(context, "超分模型导入失败：请选择包含 onnx(+bin) 的 zip 或单个 onnx", Toast.LENGTH_LONG).show()
            }
        }
    }
    // ★ 后处理参数（默认值 = 脚本 sr_then_post_v2 验证的最优配置 "轻:hard<25>+shrink2"：
    //   dc=0.6 去色边、shrink=2 条件收缩、hardenTh=25 背景硬化（0=关闭），发丝砍杀率 0%）
    var decontaminate by remember { mutableStateOf(0.6f) }
    var shrink by remember { mutableStateOf(2) }
    var hardenTh by remember { mutableIntStateOf(25) }   // 背景硬化阈值 0=关，默认25（脚本最优）
    // ★ 后处理滑块区折叠开关（默认折叠：让图像/画笔占大部分屏幕，需要时再展开调参）
    var showPostControls by remember { mutableStateOf(false) }

    // ★★ 后处理实时预览：滑块/开关拖动时立即用脚本验证的 postLight 应用到显示图，
    //    不用等点"应用"。basePixels = 打开时的原始 soft 像素（未硬化，半透明边缘齐全），
    // ★★ 后处理实时预览基底：打开时的原始 soft 像素（未硬化，半透明边缘齐全）。
    //   每次从基底重算，保证反复拖动不叠加误差。
    //   ★ 初始捕获一次（remember 无 key）；超分成功后同步更新为新大图像素，避免拖动滑块退回小图。
    var basePixels by remember {
        mutableStateOf(
            run {
                val d = displayBitmap
                if (d.isRecycled) null else {
                    val px = IntArray(d.width * d.height)
                    d.getPixels(px, 0, d.width, 0, 0, d.width, d.height)
                    px
                }
            }
        )
    }
    // 实时预览协程：参数变化触发（去抖，避免每像素滑动都全量重算）
    LaunchedEffect(bgStrength, decontaminate, shrink, hardenTh) {
        val bp = basePixels ?: return@LaunchedEffect
        val d = displayBitmap
        if (d.isRecycled) return@LaunchedEffect
        // ★ 硬化阈值：hardenTh 滑块>0 用其阈值；否则用背景强度滑杆阈值；两者都 0 → 不硬化
        val th = when {
            hardenTh > 0 -> hardenTh
            bgStrength > 0 -> bgStrength
            else -> 0
        }
        // 没有启用的处理项 → 回到原始基底（滑块全归零时恢复）
        val enabled = decontaminate > 0f || shrink > 0 || th > 0
        val updated = withContext(Dispatchers.IO) {
            try {
                if (!enabled) {
                    // 还原到基底（未硬化，保持半透明边缘）
                    Bitmap.createBitmap(d.width, d.height, Bitmap.Config.ARGB_8888).also { it.setPixels(bp, 0, d.width, 0, 0, d.width, d.height) }
                } else {
                    // 从基底重建一张，再跑脚本 postLight（绝不叠加在已处理图上）
                    val base = Bitmap.createBitmap(d.width, d.height, Bitmap.Config.ARGB_8888).also { it.setPixels(bp, 0, d.width, 0, 0, d.width, d.height) }
                    PostProcessEngine.postLight(
                        base,
                        shrink = shrink.coerceAtLeast(0),
                        hardenTh = th,
                        dc = decontaminate,
                        bgColor = PostProcessEngine.AUTO_BG
                    )
                }
            } catch (_: Exception) { null }
        }
        if (updated != null && !updated.isRecycled) {
            val old = displayBitmap
            if (old !== updated && old.isRecycled.not()) { try { old.recycle() } catch (_: Exception) {} }
            displayBitmap = updated
            brushVersion++
        }
    }

    // 画笔
    var brushMode by remember { mutableStateOf(initialBrushMode) }
    var isEraser by remember { mutableStateOf(true) }
    var brushSize by remember { mutableStateOf(50f) }
    var strokeCount by remember { mutableStateOf(0) }
    // ★ 按住触发的画布切换：对比(反色) / 预览(棋盘格结果图)，按住直接切画布渲染，松手恢复
    var viewMode by remember { mutableStateOf(0) }  // 0=编辑 1=对比反色 2=预览结果
    // ★ 画笔撤销栈：栈底是未动笔的原始帧，每次起笔快照当前像素入栈，撤销=弹栈恢复
    val undoStack = remember { mutableStateListOf<Bitmap>() }
    fun snapshotForUndo() {
        val cur = displayBitmap
        if (cur.isRecycled) return
        val copy = Bitmap.createBitmap(cur.width, cur.height, Bitmap.Config.ARGB_8888)
        val px = IntArray(cur.width * cur.height)
        cur.getPixels(px, 0, cur.width, 0, 0, cur.width, cur.height)
        copy.setPixels(px, 0, cur.width, 0, 0, cur.width, cur.height)
        // 撤销栈上限 30，超出释放最旧
        if (undoStack.size >= 30) { try { undoStack.removeAt(0).recycle() } catch (_: Exception) {} }
        undoStack.add(copy)
    }
    fun undoBrush() {
        if (undoStack.isEmpty()) return
        try { undoStack.removeAt(undoStack.size - 1)?.let { prev -> displayBitmap = prev; brushVersion++ } } catch (_: Exception) {}
    }

    // 画布缩放/平移（仅非画笔模式由外层控制；画笔模式交给 EditBrushLayer 自包含）
    var scale by remember { mutableStateOf(1f) }
    var offset by remember { mutableStateOf(Offset.Zero) }

    // 复原原始像素的恢复位图（用于"还原" + 画笔恢复的 RGB 源）
    // ★ 核心：恢复 = 扩大遮罩，RGB 必须取自「原图」（原图背景色永不丢失），
    //   遮罩 = 当前结果图的 alpha 通道（判断哪些像素被扣空）。
    //   绝不能从结果图提取像素当 RGB 源——结果透明区可能已被清成白/黑，恢复会变白丢数据。
    //   1) originBitmap(原图)：同尺寸时最可靠，RGB 一定是原背景色，历史记录场景须由调用方保证传对该原图。
    //   2) originPx(viewerBgOrigin)：仅当原图不可用时兜底（提取自结果，可能已失真）。
    //   3) 退回当前 displayBitmap。
    val originBmp = remember(softOriginBitmap, originPx, result.width, result.height) {
        when {
            softOriginBitmap != null && softOriginBitmap.width == result.width && softOriginBitmap.height == result.height && !softOriginBitmap.isRecycled ->
                softOriginBitmap
            originPx != null && originPx.size == result.width * result.height ->
                Bitmap.createBitmap(result.width, result.height, Bitmap.Config.ARGB_8888).also {
                    it.setPixels(originPx, 0, result.width, 0, 0, result.width, result.height)
                }
            else -> null
        }
    }

    Scaffold(
        containerColor = Color(0xFF101418),
        topBar = {
            TopAppBar(
                title = { Text("编辑", color = Color.White, fontSize = 17.sp) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Filled.ArrowBack, contentDescription = "返回", tint = Color.White)
                    }
                },
                actions = {
                    TextButton(
                        onClick = {
                            val cur = displayBitmap
                            if (cur.isRecycled) return@TextButton
                            val sBg = bgStrength
                            val sDc = decontaminate
                            val sSh = shrink
                            val sHd = hardenTh
                            val hasEdits = sBg > 0 || sDc > 0f || sSh != 0 || sHd > 0 || strokeCount > 0
                            if (!hasEdits) {
                                Toast.makeText(context, "未做任何调整", Toast.LENGTH_SHORT).show()
                                return@TextButton
                            }
                            val w = cur.width; val h = cur.height
                            scope.launch {
                                // ★ 应用直接用当前显示图（实时预览已把 postLight 结果写入 displayBitmap）：
                                //   滑块拖动时 LaunchedEffect 已实时应用脚本 postLight，这里直接提交最终显示图，
                                //   不再重复跑 postLight（避免在同一图上二次叠加处理）。
                                val final = cur
                                bgStrength = 0; decontaminate = 0f; shrink = 0; hardenTh = 0
                                Toast.makeText(context, "已应用", Toast.LENGTH_SHORT).show()
                                onApply(final)
                            }
                        },
                        enabled = !superResBusy && (bgStrength > 0 || decontaminate > 0f || shrink != 0 || hardenTh > 0 || strokeCount > 0)
                    ) { Text("应用", color = Color(0xFF64B5F6)) }
                },
                colors = TopAppBarDefaults.topAppBarColors(containerColor = Color(0xFF101418))
            )
        },
        bottomBar = {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(Color(0xFF1A1F24))
                    .padding(horizontal = 12.dp, vertical = 6.dp)
            ) {
                // 功能开关行：后处理/画笔 切换 + 对比/预览 视图切换（按住）
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    FilterChip(
                        selected = !brushMode,
                        onClick = { brushMode = false },
                        label = { Text("后处理", fontSize = 12.sp) }
                    )
                    FilterChip(
                        selected = brushMode,
                        onClick = { brushMode = true },
                        label = { Text("画笔", fontSize = 12.sp) }
                    )
                    Spacer(Modifier.weight(1f))
                    // ★ 按住对比：画布分层（超暗底+增强主体+边缘高亮），松手恢复
                    Box(
                        modifier = Modifier
                            .height(28.dp)
                            .clip(RoundedCornerShape(6.dp))
                            .background(if (viewMode == 1) Color(0xFF37474F) else Color(0xFF263238))
                            .pointerInput(Unit) {
                                detectTapGestures(
                                    onPress = {
                                        viewMode = if (viewMode == 1) 0 else 1
                                        tryAwaitRelease()
                                        viewMode = 0
                                    }
                                )
                            },
                        contentAlignment = Alignment.Center
                    ) { Text("对比", color = Color(0xFFFFB74D), fontSize = 11.sp, modifier = Modifier.padding(horizontal = 10.dp)) }
                    // ★ 按住预览：画布显示背景色+结果图（导出效果），松手恢复
                    Box(
                        modifier = Modifier
                            .height(28.dp)
                            .clip(RoundedCornerShape(6.dp))
                            .background(if (viewMode == 2) Color(0xFF37474F) else Color(0xFF263238))
                            .pointerInput(Unit) {
                                detectTapGestures(
                                    onPress = {
                                        viewMode = if (viewMode == 2) 0 else 2
                                        tryAwaitRelease()
                                        viewMode = 0
                                    }
                                )
                            },
                        contentAlignment = Alignment.Center
                    ) { Text("预览", color = Color(0xFF4FC3F7), fontSize = 11.sp, modifier = Modifier.padding(horizontal = 10.dp)) }
                    Text("${strokeCount} 笔", color = Color(0xFF64B5F6), fontSize = 12.sp)
                }
                Spacer(Modifier.height(4.dp))
                if (!brushMode) {
                    // ---- 后处理模式：背景强度 + 去色边/柔化/收缩/硬化 ----
                    Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                        // ★ 4x 超分独立按钮：点击立即超分当前图并替换显示（不入历史，可保存/分享）
                        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            OutlinedButton(
                                onClick = {
                                    val c = displayBitmap
                                    if (c.isRecycled || superResBusy) return@OutlinedButton
                                    // 防无限超分：已是超分产物或 ≥2048 直接拒绝
                                    if (c.width >= 2048 || c.height >= 2048) {
                                        Toast.makeText(context, "当前图已是最佳分辨率（2048），不再叠加超分", Toast.LENGTH_SHORT).show()
                                        return@OutlinedButton
                                    }
                                    scope.launch {
                                        superResBusy = true
                                        val up = withContext(Dispatchers.IO) {
                                            try {
                                                val imported = com.rmbg.offline.Prefs.superResModelPath
                                                if (imported.isNotEmpty() && java.io.File(imported).exists()) {
                                                    // ★ 优先用导入的超分模型（QNN EPContext 或普通 ONNX，自动检测）
                                                    val eng = com.rmbg.offline.ml.SuperResEngine.loadFromFile(context, java.io.File(imported), threads = 4)
                                                    try { eng.upscale4x(c) } finally { eng.close() }
                                                } else {
                                                    val model = SuperResEngine.loadFromAssets(context, "models/realesrgan_anime6b.onnx")
                                                    if (model == null) null else {
                                                        val eng = SuperResEngine(threads = 4, modelBytes = model)
                                                        try { eng.upscale4x(c) } finally { eng.close() }
                                                    }
                                                }
                                            } catch (e: Exception) {
                                                // ★ 记录具体失败原因（真机排查超分失败用）
                                                android.util.Log.e("RMBG-SUPER", "后处理页超分失败", e)
                                                // ★ 把具体错误透传给 Toast，便于真机定位
                                                val err = e.message?.take(80) ?: e.javaClass.simpleName
                                                superResFail = "超分失败：$err"
                                                null
                                            }
                                        }
                                        superResBusy = false
                                        if (up != null && !up.isRecycled && up.width > 0) {
                                            val old = displayBitmap
                                            if (old !== c && old !== up && old.isRecycled.not()) { try { old.recycle() } catch (_: Exception) {} }
                                            displayBitmap = up
                                            // ★ 同步刷新基底像素：超分后基底=新大图，拖动滑块仍基于超分结果，不会退回小图
                                            val upPx = IntArray(up.width * up.height)
                                            up.getPixels(upPx, 0, up.width, 0, 0, up.width, up.height)
                                            basePixels = upPx
                                            scale = 1f; offset = Offset.Zero
                                            Toast.makeText(context, "已 4x 超分：${up.width}×${up.height}", Toast.LENGTH_SHORT).show()
                                        } else {
                                            Toast.makeText(context, if (superResFail.isNotEmpty()) superResFail else "超分失败（模型不可用或内存不足）", Toast.LENGTH_LONG).show()
                                            superResFail = ""
                                        }
                                    }
                                },
                                enabled = !superResBusy && displayBitmap.isRecycled.not() &&
                                    displayBitmap.width < 2048 && displayBitmap.height < 2048,
                                modifier = Modifier.weight(1f).height(36.dp),
                                contentPadding = PaddingValues(horizontal = 8.dp, vertical = 0.dp),
                                colors = ButtonDefaults.outlinedButtonColors(contentColor = Color(0xFFFF8F00))
                            ) {
                                Icon(Icons.Filled.ZoomIn, contentDescription = null, modifier = Modifier.size(14.dp), tint = Color(0xFFFF8F00))
                                Spacer(Modifier.width(4.dp))
                                Text(if (superResBusy) "超分中…" else "4x 超分", fontSize = 12.sp, color = Color(0xFFFF8F00))
                            }
                            // ★ 导入超分模型入口 + 当前来源（★ 独立于抠图模型导入：超分模型存 superres/ 目录，绝不混入抠图列表）
                            TextButton(
                                onClick = { importSuperResLauncher.launch("*/*") },
                                enabled = !superResBusy,
                                modifier = Modifier.height(30.dp),
                                contentPadding = PaddingValues(horizontal = 6.dp, vertical = 0.dp)
                            ) {
                                Icon(Icons.Filled.Settings, contentDescription = null, modifier = Modifier.size(13.dp), tint = Color(0xFF90A4AE))
                                Spacer(Modifier.width(3.dp))
                                Text(if (superResImported) "已导入超分模型" else "导入超分模型", fontSize = 11.sp, color = Color(0xFF90A4AE))
                            }
                            Column(modifier = Modifier.weight(1f)) {
                                Text("(AI 放大 4 倍)", color = Color(0xFF90A4AE), fontSize = 10.sp)
                                if (superResImported) {
                                    Text("导入: ${java.io.File(com.rmbg.offline.Prefs.superResModelPath).name}", color = Color(0xFF4FC3F7), fontSize = 9.sp, maxLines = 1)
                                }
                            }
                        }
                        // ★ 滑块区折叠开关：默认折叠，点开展开 背景强度/去色边/收缩/硬化
                        TextButton(
                            onClick = { showPostControls = !showPostControls },
                            modifier = Modifier.fillMaxWidth().height(30.dp),
                            contentPadding = PaddingValues(horizontal = 8.dp, vertical = 0.dp)
                        ) {
                            Icon(
                                if (showPostControls) Icons.Filled.KeyboardArrowUp else Icons.Filled.KeyboardArrowDown,
                                contentDescription = null, modifier = Modifier.size(16.dp), tint = Color(0xFF90A4AE)
                            )
                            Spacer(Modifier.width(4.dp))
                            Text(if (showPostControls) "收起参数" else "参数调整", fontSize = 12.sp, color = Color(0xFF90A4AE))
                        }
                        if (showPostControls) {
                        // 背景强度
                        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Text("背景强度", color = Color.White, fontSize = 12.sp)
                            Slider(
                                value = bgStrength.toFloat(),
                                onValueChange = { bgStrength = it.roundToInt().coerceIn(0, 255) },
                                valueRange = 0f..255f,
                                modifier = Modifier.weight(1f)
                            )
                            Text("$bgStrength", color = Color.White, fontSize = 12.sp)
                        }
                        Text("把 alpha 低于此值的半透明背景雾清除为全透明（值越高，背景越干净，也会削掉更细的发丝）", color = Color(0xFF90A4AE), fontSize = 10.sp)
                        Spacer(Modifier.height(4.dp))
                        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Text("去色边", color = Color.White, fontSize = 12.sp)
                            Slider(
                                value = decontaminate,
                                onValueChange = { decontaminate = it },
                                valueRange = 0f..1f,
                                modifier = Modifier.weight(1f)
                            )
                            Text("${(decontaminate * 100).roundToInt()}%", color = Color.White, fontSize = 12.sp)
                        }
                        Text("还原半透明边缘被污染的底色（去白边/黑边/彩边）；值越高还原越彻底，低强度更保险", color = Color(0xFF90A4AE), fontSize = 10.sp)
                        Spacer(Modifier.height(4.dp))
                        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Text("去毛边(收缩)", color = Color(0xFFFFB74D), fontSize = 12.sp)
                            Slider(
                                value = shrink.toFloat(),
                                onValueChange = { shrink = it.roundToInt() },
                                valueRange = 0f..20f,
                                modifier = Modifier.weight(1f)
                            )
                            Text("${shrink}px", color = Color(0xFFFFB74D), fontSize = 12.sp)
                        }
                        Text("沿边缘向内收缩，收掉突出的毛刺/锯齿；仅贴背景处收缩，主体内部不动，值越大收得越狠", color = Color(0xFF90A4AE), fontSize = 10.sp)
                        Spacer(Modifier.height(4.dp))
                        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Text("背景硬化", color = Color.White, fontSize = 12.sp)
                            Slider(
                                value = hardenTh.toFloat(),
                                onValueChange = { hardenTh = it.roundToInt().coerceIn(0, 255) },
                                valueRange = 0f..255f,
                                modifier = Modifier.weight(1f)
                            )
                            Text(if (hardenTh > 0) "$hardenTh" else "关", color = Color.White, fontSize = 12.sp)
                        }
                        Text("把 alpha 低于此值的半透明残留雾影设为全透明（0=关闭；值越高背景越干净，过高会削细发丝）", color = Color(0xFF90A4AE), fontSize = 10.sp)
                        }
                    }
                } else {
                    // ---- 画笔模式（紧凑单行）：擦除/恢复 + 撤销 + 笔刷（对比/预览在顶部开关行）----
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        OutlinedButton(
                            onClick = { isEraser = !isEraser },
                            modifier = Modifier.height(32.dp),
                            colors = if (isEraser)
                                ButtonDefaults.outlinedButtonColors(contentColor = Color(0xFFFF5252), containerColor = Color(0xFF263238))
                            else ButtonDefaults.outlinedButtonColors(contentColor = Color(0xFF4CAF50), containerColor = Color(0xFF263238))
                        ) { Text(if (isEraser) "擦除" else "恢复", fontSize = 11.sp) }
                        OutlinedButton(
                            onClick = { undoBrush() },
                            enabled = undoStack.isNotEmpty(),
                            modifier = Modifier.height(32.dp)
                        ) { Text("撤销", fontSize = 11.sp) }
                        Text("笔刷", color = Color.White, fontSize = 11.sp)
                        Slider(
                            value = brushSize,
                            onValueChange = { brushSize = it },
                            valueRange = 10f..120f,
                            modifier = Modifier.weight(1f)
                        )
                        Text("${brushSize.roundToInt()}", color = Color.White, fontSize = 11.sp)
                    }
                }
                Spacer(Modifier.height(2.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        if (brushMode) "单指画笔 · 双指缩放平移 · 长按放大镜" else "双指缩放/平移预览",
                        color = Color(0xFF90A4AE), fontSize = 11.sp
                    )
                    Spacer(Modifier.weight(1f))
                    TextButton(onClick = {
                        // ★ 还原：回到打开编辑页时的原始像素（基底 basePixels，恒定尺寸），并复位滑块/画笔/缩放
                        val bp = basePixels
                        if (bp == null) return@TextButton
                        val w = displayBitmap.width; val h = displayBitmap.height
                        val restore = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888).also {
                            it.setPixels(bp, 0, w, 0, 0, w, h)
                        }
                        val old = displayBitmap
                        if (old !== restore && old.isRecycled.not()) { try { old.recycle() } catch (_: Exception) {} }
                        displayBitmap = restore
                        bgStrength = 0; decontaminate = 0f; shrink = 0; hardenTh = 0; strokeCount = 0
                        undoStack.clear()
                        scale = 1f; offset = Offset.Zero
                    }) { Text("还原", fontSize = 12.sp) }
                }
            }
        }
    ) { padding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .background(Color(0xFF101418))
        ) {
            // 纯深色背景（画笔模式下 EditBrushLayer 内部绘制暗色原图作为透明处底色）
            Box(modifier = Modifier.fillMaxSize().background(Color(0xFF101418)))
            // 图像容器（缩放/平移）
            BoxWithConstraints(
                modifier = Modifier
                    .fillMaxSize()
                    .then(
                        if (!brushMode) Modifier.pointerInput(Unit) {
                            detectTransformGestures { centroid, pan, zoom, _ ->
                                val prevScale = scale
                                val newScale = (prevScale * zoom).coerceIn(1f, 8f)
                                offset = offset + pan
                                scale = newScale
                            }
                        } else Modifier
                    )
            ) {
                val cnvDensity = LocalDensity.current
                val canvasW = with(cnvDensity) { maxWidth.toPx() }
                val canvasH = with(cnvDensity) { maxHeight.toPx() }
                // 计算 fit 后图像位置（非画笔模式用 graphicsLayer；画笔模式 EditBrushLayer 自包含）
                if (!brushMode) {
                    val bw = displayBitmap.width.toFloat()
                    val bh = displayBitmap.height.toFloat()
                    val fitS = min(canvasW / bw, canvasH / bh)
                    val fitW = bw * fitS
                    val fitH = bh * fitS
                    val baseLeft = (canvasW - fitW) / 2f
                    val baseTop = (canvasH - fitH) / 2f
                    Box(
                        modifier = Modifier
                            .offset { IntOffset(baseLeft.roundToInt(), baseTop.roundToInt()) }
                            .size(with(LocalDensity.current) { fitW.toDp() }, with(LocalDensity.current) { fitH.toDp() })
                            .graphicsLayer(
                                scaleX = scale,
                                scaleY = scale,
                                transformOrigin = TransformOrigin(0f, 0f),
                                translationX = offset.x,
                                translationY = offset.y
                            )
                    ) {
                        Image(
                            bitmap = run { brushVersion; displayBitmap.asImageBitmap() },
                            contentDescription = "结果大图",
                            modifier = Modifier.fillMaxSize(),
                            contentScale = ContentScale.Fit
                        )
                    }
                } else {
                    EditBrushLayer(
                        modifier = Modifier.fillMaxSize(),
                        displayBitmap = displayBitmap,
                        original = originBmp ?: displayBitmap,
                        brushColor = if (isEraser) Color(0xFFEF5350) else Color(0x884CAF50),
                        brushSize = brushSize,
                        isEraser = isEraser,
                        viewMode = viewMode,   // ★ 按住对比/预览切换画布渲染
                        onModified = { brushVersion++ },
                        onStroke = { strokeCount++; snapshotForUndo() }
                    )
                }
            }
        }
    }
}

