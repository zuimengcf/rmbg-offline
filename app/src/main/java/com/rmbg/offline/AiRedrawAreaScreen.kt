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

/**
 * AI 重绘 · 区域编辑（复用 AreaSelectScreen 的选区交互骨架）。
 *
 * 解决"整图缩到档位（512×512）导致分辨率低"的问题：
 *  - 用户框选一个区域（归一化坐标 L/T/R/B）
 *  - 从源图裁出该区域 → 缩放到档位 → img2img 重绘
 *  - 重绘结果缩放回选区尺寸 → 贴回原图对应位置
 *  - 选区是局部，512×512 档位对应的是局部放大后的细节（远高于整图 512），贴回后不糊
 *
 * @param original 源图（查看器当前显示的图）
 * @param running 是否重绘中（外部控制，禁用按钮）
 * @param onRedrawRegion 执行区域重绘：入参 (源图, 归一化选区), 返回合成结果图（IO 线程）
 * @param onApply 应用合成结果
 * @param onBack 返回
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AiRedrawAreaScreen(
    original: Bitmap,
    running: Boolean,
    onRedrawRegion: suspend (Bitmap, FloatArray) -> Bitmap?,
    onApply: (Bitmap) -> Unit,
    onBack: () -> Unit
) {
    val context = androidx.compose.ui.platform.LocalContext.current
    val scope = rememberCoroutineScope()
    var display by remember { mutableStateOf(original) }
    var displayV by remember { mutableStateOf(0) }
    var selL by remember { mutableFloatStateOf(0.25f) }
    var selT by remember { mutableFloatStateOf(0.25f) }
    var selR by remember { mutableFloatStateOf(0.75f) }
    var selB by remember { mutableFloatStateOf(0.75f) }
    var isRunning by remember { mutableStateOf(false) }

    Scaffold(
        containerColor = Color(0xFF101418),
        topBar = {
            TopAppBar(
                title = { Text("AI 重绘 · 区域编辑", color = Color.White, fontSize = 17.sp) },
                navigationIcon = { IconButton(onClick = onBack) { Icon(Icons.Filled.ArrowBack, contentDescription = "返回", tint = Color.White) } },
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
                        if (display.isRecycled) return@Button
                        val bw = original.width; val bh = original.height
                        val L = (selL * bw).toInt().coerceIn(0, bw - 1)
                        val T = (selT * bh).toInt().coerceIn(0, bh - 1)
                        val R = (selR * bw).toInt().coerceIn(L + 1, bw)
                        val B = (selB * bh).toInt().coerceIn(T + 1, bh)
                        scope.launch {
                            isRunning = true
                            // ★ 在 IO 线程执行区域重绘（裁区→放大→img2img→贴回）
                            val res = withContext(Dispatchers.IO) {
                                try { onRedrawRegion(display, floatArrayOf(L.toFloat(), T.toFloat(), R.toFloat(), B.toFloat())) }
                                catch (_: Exception) { null }
                            }
                            isRunning = false
                            if (res != null && !res.isRecycled) {
                                display = res; displayV++
                                Toast.makeText(context, "选区内重绘完成，可继续框选其他区域", Toast.LENGTH_SHORT).show()
                            } else Toast.makeText(context, "选区内重绘失败", Toast.LENGTH_SHORT).show()
                        }
                    },
                    enabled = !isRunning && !running,
                    modifier = Modifier.fillMaxWidth().height(50.dp)
                ) {
                    Icon(Icons.Filled.AutoAwesome, contentDescription = null); Spacer(Modifier.width(8.dp))
                    Text(if (isRunning) "重绘中..." else "AI 重绘选区内", fontSize = 15.sp, fontWeight = androidx.compose.ui.text.font.FontWeight.Bold)
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(Modifier.weight(1f)) {
                        Text("框选局部重绘 · 512 档位对应局部细节，不糊", color = Color(0xFF90A4AE), fontSize = 11.sp)
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
                            var twoFinger = false
                            while (true) {
                                val evt = awaitPointerEvent()
                                val pts = evt.changes
                                val cw = size.width.toFloat()
                                val ch = size.height.toFloat()
                                if (pts.size >= 2) {
                                    twoFinger = true
                                    val p0 = pts[0]; val p1 = pts[1]
                                    val dist = sqrt((p1.position.x - p0.position.x).let { it * it } +
                                        (p1.position.y - p0.position.y).let { it * it })
                                    if (evt.type == PointerEventType.Move) {
                                        val prevDist = sqrt(
                                            (p1.previousPosition.x - p0.previousPosition.x).let { it * it } +
                                            (p1.previousPosition.y - p0.previousPosition.y).let { it * it })
                                        if (prevDist > 0) scale = (scale * dist / prevDist).coerceIn(0.5f, 8f)
                                        offset += Offset(
                                            ((p0.position.x - p0.previousPosition.x) + (p1.position.x - p1.previousPosition.x)) / 2,
                                            ((p0.position.y - p0.previousPosition.y) + (p1.position.y - p1.previousPosition.y)) / 2)
                                    }
                                    pts.forEach { it.consume() }
                                } else if (pts.size == 1) {
                                    val p = pts[0]
                                    val dw = display.width.toFloat(); val dh = display.height.toFloat()
                                    val fs = min(cw / dw, ch / dh)
                                    val fw = dw * fs; val fh = dh * fs
                                    val iL = (cw - fw * scale) / 2f + offset.x
                                    val iT = (ch - fh * scale) / 2f + offset.y
                                    val u = ((p.position.x - iL) / (fw * scale)).coerceIn(0f, 1f)
                                    val v = ((p.position.y - iT) / (fh * scale)).coerceIn(0f, 1f)
                                    when (evt.type) {
                                        PointerEventType.Press -> {
                                            twoFinger = false
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
                                            if (twoFinger) return@awaitPointerEventScope
                                            when (dragMode) {
                                                1 -> { val du = u - dragPrevU; val dv = v - dragPrevV; selL = (selL + du).coerceIn(0f, 1f); selR = (selR + du).coerceIn(0f, 1f); selT = (selT + dv).coerceIn(0f, 1f); selB = (selB + dv).coerceIn(0f, 1f) }
                                                2 -> { selR = u.coerceIn(selL + 0.02f, 1f); selB = v.coerceIn(selT + 0.02f, 1f) }
                                                3 -> { selR = u.coerceIn(selL + 0.02f, 1f); selB = v.coerceIn(selT + 0.02f, 1f) }
                                            }
                                            dragPrevU = u; dragPrevV = v
                                        }
                                        PointerEventType.Release -> dragMode = 0
                                        else -> {}
                                    }
                                    if (dragMode != 0) p.consume()
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
