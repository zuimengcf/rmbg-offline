package com.rmbg.offline

import android.graphics.Bitmap
import android.widget.Toast
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
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
            if (result.isMutable && result.config != android.graphics.Bitmap.Config.HARDWARE)
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
    // 后处理参数
    var decontaminate by remember { mutableStateOf(0f) }
    var feather by remember { mutableStateOf(0) }
    var shrink by remember { mutableStateOf(0) }
    var harden by remember { mutableStateOf(false) }
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
                title = { Text("后处理 / 画笔", color = Color.White, fontSize = 17.sp) },
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
                            val sFe = feather
                            val sSh = shrink
                            val sHd = harden
                            val hasEdits = sBg > 0 || sDc > 0f || sFe > 0 || sSh != 0 || sHd || strokeCount > 0
                            if (!hasEdits) {
                                Toast.makeText(context, "未做任何调整", Toast.LENGTH_SHORT).show()
                                return@TextButton
                            }
                            val w = cur.width; val h = cur.height
                            val wBg = (Prefs.viewerBgColor.let { bgName ->
                                when (bgName) {
                                    "白色" -> -1
                                    "黑色" -> 0xFF000000.toInt()
                                    "浅灰" -> 0xFFBDBDBD.toInt()
                                    "绿色" -> 0xFF4CAF50.toInt()
                                    else -> -1
                                }
                            })
                            scope.launch {
                                val final = withContext(Dispatchers.IO) {
                                    // 1) 背景强度：把 alpha 低于阈值的半透明残影硬化为全透明
                                    var b = cur
                                    if (sBg > 0) {
                                        val nb = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
                                        val row = IntArray(w)
                                        val dst = IntArray(w)
                                        for (y in 0 until h) {
                                            val base = y * w
                                            cur.getPixels(row, 0, w, 0, y, w, 1)
                                            for (x in 0 until w) {
                                                val p = row[x]
                                                dst[x] = if (((p ushr 24) and 0xFF) < sBg) 0x00000000 else p
                                            }
                                            nb.setPixels(dst, 0, w, 0, y, w, 1)
                                        }
                                        b = nb
                                    }
                                    // 2) 边缘后处理：调引擎 postProcess（去色边/柔化/收缩/背景硬化）
                                    if (sDc > 0f || sFe > 0 || sSh != 0 || sHd) {
                                        try {
                                            RmbgScreenState.engine?.postProcess(
                                                b, sDc, sFe, sSh, wBg, harden = sHd
                                            )?.let { b = it }
                                        } catch (_: Exception) {}
                                    }
                                    b
                                }
                                displayBitmap = final
                                bgStrength = 0; decontaminate = 0f; feather = 0; shrink = 0; harden = false
                                Toast.makeText(context, "已应用", Toast.LENGTH_SHORT).show()
                                onApply(final)
                            }
                        },
                        enabled = bgStrength > 0 || decontaminate > 0f || feather > 0 || shrink != 0 || harden || strokeCount > 0
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
                        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Text("柔化", color = Color.White, fontSize = 12.sp)
                            Slider(
                                value = feather.toFloat(),
                                onValueChange = { feather = it.roundToInt() },
                                valueRange = 0f..8f,
                                modifier = Modifier.weight(1f)
                            )
                            Text("${feather}px", color = Color.White, fontSize = 12.sp)
                        }
                        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Text("收缩", color = Color.White, fontSize = 12.sp)
                            Slider(
                                value = shrink.toFloat(),
                                onValueChange = { shrink = it.roundToInt() },
                                valueRange = -20f..20f,
                                modifier = Modifier.weight(1f)
                            )
                            Text("${shrink}px", color = Color.White, fontSize = 12.sp)
                        }
                        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Text("背景硬化", color = Color.White, fontSize = 12.sp)
                            FilterChip(
                                selected = harden,
                                onClick = { harden = !harden },
                                label = { Text(if (harden) "开" else "关", fontSize = 12.sp) }
                            )
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
                        // 还原：回到打开时原始像素
                        val ob = originBmp ?: return@TextButton
                        val w = displayBitmap.width; val h = displayBitmap.height
                        val restore = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
                        android.graphics.Canvas(restore).drawBitmap(ob, 0f, 0f, null)
                        displayBitmap = restore
                        bgStrength = 0; decontaminate = 0f; feather = 0; shrink = 0; harden = false; strokeCount = 0
                        undoStack.clear()
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

