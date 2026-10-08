package com.rmbg.offline

import android.graphics.Bitmap
import android.graphics.Paint
import android.widget.Toast
import androidx.compose.foundation.Canvas
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
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.input.pointer.PointerEventType
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlin.math.min
import kotlin.math.sqrt
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AreaSelectScreen(
    original: Bitmap,
    threshold: Float,
    currentResult: Bitmap? = null,
    // ★ 修复③：改为懒加载 lambda，在按钮点击协程里后台调用，不在组合期间阻塞主线程
    getEngine: () -> com.rmbg.offline.ml.RmbgOnnxEngine? = { null },
    onApply: (Bitmap) -> Unit = {},
    onEnterBrush: () -> Unit = {},
    onSwitchModel: () -> Unit = {},
    onBack: () -> Unit = {},
    // ★ AI 区域重绘执行回调（与局部重抠共用同一套选区/display）
    onRedrawRegion: suspend (Bitmap, FloatArray) -> Bitmap? = { _, _ -> null }
) {
    val context = androidx.compose.ui.platform.LocalContext.current
    val scope = rememberCoroutineScope()
    var display by remember { mutableStateOf(currentResult ?: original) }
    var displayV by remember { mutableStateOf(0) }
    var selL by remember { mutableFloatStateOf(0.2f) }
    var selT by remember { mutableFloatStateOf(0.2f) }
    var selR by remember { mutableFloatStateOf(0.8f) }
    var selB by remember { mutableFloatStateOf(0.8f) }
    var isRmbg by remember { mutableStateOf(false) }
    var isRedraw by remember { mutableStateOf(false) }

    Scaffold(
        containerColor = Color(0xFF101418),
        topBar = {
            TopAppBar(
                title = { Text("高级编辑 · 区域选择", color = Color.White, fontSize = 17.sp) },
                navigationIcon = { IconButton(onClick = onBack) { Icon(Icons.Filled.ArrowBack, contentDescription = "返回", tint = Color.White) } },
                actions = {
                    TextButton(onClick = onSwitchModel) { Text("切模型", color = Color(0xFFFFB300)) }
                    TextButton(onClick = onEnterBrush) { Text("编辑", color = Color(0xFF64B5F6)) }
                },
                colors = TopAppBarDefaults.topAppBarColors(containerColor = Color(0xFF101418))
            )
        },
        bottomBar = {
            Column(
                modifier = Modifier.fillMaxWidth().background(Color(0xFF1A1F24)).padding(horizontal = 14.dp, vertical = 10.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                Button(
                    onClick = {
                        scope.launch {
                            isRmbg = true
                            // ★ 修复③：在协程/IO线程懒加载引擎（不在主线程阻塞），避免组合期间 getEngine() 异常被吞 → null → "模型未加载"
                            val eng = withContext(Dispatchers.IO) {
                                try { getEngine() } catch (_: Exception) { null }
                            }
                            if (eng == null) {
                                isRmbg = false
                                Toast.makeText(context, "模型未加载，请先在主界面确保模型就绪", Toast.LENGTH_SHORT).show()
                                return@launch
                            }
                            val bw = original.width; val bh = original.height
                            val L = (selL * bw).toInt().coerceIn(0, bw - 1)
                            val T = (selT * bh).toInt().coerceIn(0, bh - 1)
                            val R = (selR * bw).toInt().coerceIn(L + 1, bw)
                            val B = (selB * bh).toInt().coerceIn(T + 1, bh)
                            // ★ 修复：用当前已抠的 display 作为 currentResult，而不是外部初始传入的 currentResult。
                            //   这样选区外保留之前已抠的结果（左上角扣好了切到右上角再扣，左上角依然保留透明）。
                            //   旧版传固定 currentResult（editorOnOriginal=true 时为 null）→ 每次选区外回原图 → 之前抠的丢失。
                            val curForRegion = if (!display.isRecycled && display !== original) display else null
                            val res = withContext(Dispatchers.IO) {
                                try { eng.removeBackgroundRegion(original, threshold, intArrayOf(L, T, R, B), curForRegion) }
                                catch (_: Exception) { null }
                            }
                            isRmbg = false
                            if (res != null && !res.isRecycled) {
                                display = res; displayV++
                                Toast.makeText(context, "已局部重抠（选区内）", Toast.LENGTH_SHORT).show()
                            } else Toast.makeText(context, "局部重抠失败", Toast.LENGTH_SHORT).show()
                        }
                    },
                    // ★ 始终可点：engine 为空时点击提示（不再静默禁用，避免看起来"只有区域重绘"）
                    enabled = !isRmbg,
                    modifier = Modifier.fillMaxWidth().height(50.dp)
                ) {
                    Icon(Icons.Filled.Crop, contentDescription = null); Spacer(Modifier.width(8.dp))
                    Text(if (isRmbg) "局部重抠中..." else "局部重抠（仅抠选区内）", fontSize = 15.sp, fontWeight = androidx.compose.ui.text.font.FontWeight.Bold)
                }
                // ★ AI 区域重绘：与"局部重抠"并列，共用同一套选区框 + 同一张工作画布（display）
                Button(
                    onClick = {
                        if (display.isRecycled) return@Button
                        val bw = original.width; val bh = original.height
                        val L = (selL * bw).toInt().coerceIn(0, bw - 1)
                        val T = (selT * bh).toInt().coerceIn(0, bh - 1)
                        val R = (selR * bw).toInt().coerceIn(L + 1, bw)
                        val B = (selB * bh).toInt().coerceIn(T + 1, bh)
                        scope.launch {
                            isRedraw = true
                            val res = withContext(Dispatchers.IO) {
                                try { onRedrawRegion(display, floatArrayOf(L.toFloat(), T.toFloat(), R.toFloat(), B.toFloat())) }
                                catch (_: Exception) { null }
                            }
                            isRedraw = false
                            if (res != null && !res.isRecycled) {
                                display = res; displayV++
                                Toast.makeText(context, "选区内 AI 重绘完成，可继续框选其他区域", Toast.LENGTH_SHORT).show()
                            } else Toast.makeText(context, "选区内 AI 重绘失败", Toast.LENGTH_SHORT).show()
                        }
                    },
                    enabled = !isRedraw,
                    modifier = Modifier.fillMaxWidth().height(50.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF7B1FA2))
                ) {
                    Icon(Icons.Filled.AutoAwesome, contentDescription = null); Spacer(Modifier.width(8.dp))
                    Text(if (isRedraw) "AI 区域重绘中..." else "AI 区域重绘（选区内）", fontSize = 15.sp, fontWeight = androidx.compose.ui.text.font.FontWeight.Bold)
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(Modifier.weight(1f)) {
                        Text("拖动移动 · 拉右下角改大小", color = Color(0xFF90A4AE), fontSize = 11.sp)
                    }
                    TextButton(onClick = { if (display != null && !display.isRecycled) onApply(display) }) { Text("应用", color = Color(0xFF64B5F6), fontSize = 12.sp) }
                }
            }
        }
    ) { padding ->
        Box(
            modifier = Modifier.fillMaxSize().padding(padding).background(Color(0xFF101418))
        ) {
            val density = LocalDensity.current
            var scale by remember { mutableStateOf(1f) }
            var offset by remember { mutableStateOf(Offset.Zero) }
            var dragMode by remember { mutableIntStateOf(0) }
            var dragPrevU by remember { mutableFloatStateOf(0f) }
            var dragPrevV by remember { mutableFloatStateOf(0f) }
            BoxWithConstraints(
                modifier = Modifier
                    .fillMaxSize()
                    .pointerInput(Unit) {
                        awaitPointerEventScope {
                            while (true) {
                                val evt = awaitPointerEvent()
                                val cw = size.width.toFloat()
                                val ch = size.height.toFloat()
                                // ★ 修复：用"当前实际按下的指针数"动态判定，绝不靠手工置位/复位 twoFinger。
                                //   旧版靠 twoFinger 布尔：双指抬起一根后仍为 true → 单指 Move 触发
                                //   `return@awaitPointerEventScope` 退出整个事件循环 → 之后所有触摸永久失效(卡死)。
                                //   pressedCount 每次事件重算：抬起一根自动回落，永不卡死。
                                val active = evt.changes.filter { it.pressed }
                                if (active.size >= 2) {
                                    // ---- 双指：以两指中点为锚点缩放 + 中点平移 ----
                                    val p0 = active[0]; val p1 = active[active.size - 1]
                                    val dist = sqrt((p1.position.x - p0.position.x).let { it * it } +
                                        (p1.position.y - p0.position.y).let { it * it })
                                    if (evt.type == PointerEventType.Move) {
                                        val prevDist = sqrt(
                                            (p1.previousPosition.x - p0.previousPosition.x).let { it * it } +
                                            (p1.previousPosition.y - p0.previousPosition.y).let { it * it })
                                        if (prevDist > 0) {
                                            val newScale = (scale * dist / prevDist).coerceIn(0.5f, 8f)
                                            val dw = display.width.toFloat(); val dh = display.height.toFloat()
                                            val fs = min(cw / dw, ch / dh)
                                            val fw = dw * fs; val fh = dh * fs
                                            val imgCx = (cw - fw * scale) / 2f + offset.x + fw * scale / 2f
                                            val imgCy = (ch - fh * scale) / 2f + offset.y + fh * scale / 2f
                                            val pinchCx = (p0.position.x + p1.position.x) / 2f
                                            val pinchCy = (p0.position.y + p1.position.y) / 2f
                                            val ratio = newScale / scale
                                            val dx = pinchCx - imgCx
                                            val dy = pinchCy - imgCy
                                            offset = Offset(
                                                offset.x + dx * (1f - ratio),
                                                offset.y + dy * (1f - ratio)
                                            )
                                            scale = newScale
                                        }
                                        // 双指平移：两指中点位移
                                        offset += Offset(
                                            ((p0.position.x - p0.previousPosition.x) + (p1.position.x - p1.previousPosition.x)) / 2,
                                            ((p0.position.y - p0.previousPosition.y) + (p1.position.y - p1.previousPosition.y)) / 2)
                                    }
                                    evt.changes.forEach { it.consume() }
                                } else if (active.size == 1) {
                                    // ---- 单指：框选 / 拖动选区 / 拉右下角 ----
                                    val p = active[0]
                                    val dw = display.width.toFloat(); val dh = display.height.toFloat()
                                    val fs = min(cw / dw, ch / dh)
                                    val fw = dw * fs; val fh = dh * fs
                                    val iL = (cw - fw * scale) / 2f + offset.x
                                    val iT = (ch - fh * scale) / 2f + offset.y
                                    val u = ((p.position.x - iL) / (fw * scale)).coerceIn(0f, 1f)
                                    val v = ((p.position.y - iT) / (fh * scale)).coerceIn(0f, 1f)
                                    when (evt.type) {
                                        PointerEventType.Press -> {
                                            val r = with(density) { 26.dp.toPx() }
                                            val hx = iL + selR * fw * scale
                                            val hy = iT + selB * fh * scale
                                            val onHand = (p.position.x - hx) * (p.position.x - hx) +
                                                (p.position.y - hy) * (p.position.y - hy) <= r * r
                                            val inside = u in selL..selR && v in selT..selB
                                            dragMode = if (onHand) 2 else if (inside) 1 else 3
                                            if (dragMode == 3) {
                                                selL = u; selT = v
                                                selR = (u + 0.06f).coerceIn(0f, 1f)
                                                selB = (v + 0.06f).coerceIn(0f, 1f)
                                            }
                                            dragPrevU = u; dragPrevV = v
                                        }
                                        PointerEventType.Move -> {
                                            if (dragMode != 0) {
                                                when (dragMode) {
                                                    1 -> { val du = u - dragPrevU; val dv = v - dragPrevV; selL = (selL + du).coerceIn(0f, 1f); selR = (selR + du).coerceIn(0f, 1f); selT = (selT + dv).coerceIn(0f, 1f); selB = (selB + dv).coerceIn(0f, 1f) }
                                                    2 -> { selR = u.coerceIn(selL + 0.02f, 1f); selB = v.coerceIn(selT + 0.02f, 1f) }
                                                    3 -> { selR = u.coerceIn(selL + 0.02f, 1f); selB = v.coerceIn(selT + 0.02f, 1f) }
                                                }
                                            }
                                            dragPrevU = u; dragPrevV = v
                                        }
                                        PointerEventType.Release -> dragMode = 0
                                        else -> {}
                                    }
                                    if (dragMode != 0) p.consume()
                                } else {
                                    // 所有手指都抬起：清除拖动状态，准备下一次
                                    dragMode = 0
                                }
                            }
                        }
                    }
            ) {
                val cell = with(density) { 20.dp.toPx() }
                Canvas(Modifier.fillMaxSize()) {
                    var x = 0f
                    while (x < size.width) { var y = 0f; while (y < size.height) {
                        drawRect(if (((x / cell).toInt() + (y / cell).toInt()) % 2 == 0) Color(0xFF2A2F36) else Color(0xFF1C2127), Offset(x, y), androidx.compose.ui.geometry.Size(cell, cell))
                        y += cell }; x += cell }
                }
                if (!display.isRecycled) {
                    val d = density
                    val cwPx = with(d) { maxWidth.toPx() }
                    val chPx = with(d) { maxHeight.toPx() }
                    val dw = display.width.toFloat(); val dh = display.height.toFloat()
                    val fs = min(cwPx / dw, chPx / dh)
                    val fw = dw * fs; val fh = dh * fs
                    val iL = (cwPx - fw * scale) / 2f + offset.x
                    val iT = (chPx - fh * scale) / 2f + offset.y
                    val imgRect = android.graphics.RectF(iL, iT, iL + fw * scale, iT + fh * scale)
                    Canvas(Modifier.fillMaxSize()) {
                        val native = drawContext.canvas.nativeCanvas
                        native.drawBitmap(display, null, imgRect, Paint().apply { isFilterBitmap = true })
                        val sL = iL + selL * fw * scale
                        val sT = iT + selT * fh * scale
                        val sR = iL + selR * fw * scale
                        val sB = iT + selB * fh * scale
                        native.drawRect(sL, sT, sR, sB, Paint().apply { color = 0x2264B5F6.toInt() })
                        native.drawRect(sL, sT, sR, sB, Paint().apply { style = Paint.Style.STROKE; strokeWidth = 2f; color = 0xFF64B5F6.toInt() })
                        native.drawCircle(sR, sB, 10f, Paint().apply { color = 0xFF64B5F6.toInt() })
                    }
                }
            }
        }
    }
}