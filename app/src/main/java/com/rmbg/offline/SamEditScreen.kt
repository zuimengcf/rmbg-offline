package com.rmbg.offline

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.widget.Toast
import androidx.compose.foundation.Canvas as ComposeCanvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color as ComposeColor
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
 * SAM 智能点击编辑：点击物体任意位置 → 全图编码 + 分割该物体 → 扣除 / 换色 / 移动。
 *
 * ★ 交互（用户方案）：
 *   1) 点击：单指点击图片任意位置 = 分割该点物体（不弹选区框，全图给模型）
 *   2) 分割后：高亮显示掩码 + 出现三个操作按钮（扣除/换色/移动）
 *   3) 移动：拖动物体（绿色轮廓跟手）→ 点「放下」；原位置变透明
 *   4) 换色：色相滑杆 + 色板，点「换色」应用
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SamEditScreen(
    original: Bitmap,
    getEngine: () -> com.rmbg.offline.ml.SamEngine? = { null },
    onApply: (Bitmap) -> Unit = {},
    onBack: () -> Unit = {}
) {
    val context = androidx.compose.ui.platform.LocalContext.current
    val scope = rememberCoroutineScope()
    var work by remember { mutableStateOf(original) }
    var workV by remember { mutableStateOf(0) }

    var samBusy by remember { mutableStateOf(false) }
    var samStatus by remember { mutableStateOf("点击图片上的物体进行分割") }
    var mask by remember { mutableStateOf<Bitmap?>(null) }
    var tapPoint by remember { mutableStateOf<IntArray?>(null) }

    var hueDeg by remember { mutableFloatStateOf(120f) }
    var satScale by remember { mutableFloatStateOf(1f) }

    var moving by remember { mutableStateOf(false) }
    var moveOffset by remember { mutableStateOf(Offset.Zero) }
    var moveOffsetImg by remember { mutableStateOf(Offset.Zero) }
    // 移动实时预览：底图（物体被擦除，露出棋盘）+ 物体贴片（可平移绘制）
    var cleanBase by remember { mutableStateOf<Bitmap?>(null) }
    var objectTile by remember { mutableStateOf<Bitmap?>(null) }

    val colorPresets = listOf(
        "金黄" to 49f, "淡蓝" to 205f, "珊瑚橘" to 12f,
        "桃粉" to 340f, "薄荷绿" to 160f, "薰衣草紫" to 275f
    )

    fun obtainEngine(): com.rmbg.offline.ml.SamEngine? {
        return try { getEngine() } catch (_: Exception) { null }
    }

    // ===== 扣除 =====
    fun doErase() {
        val m = mask ?: run { Toast.makeText(context, "请先点击物体", Toast.LENGTH_SHORT).show(); return }
        scope.launch {
            samBusy = true
            val out = withContext(Dispatchers.IO) { eraseByMask(work, m) }
            if (out != null) { work = out; workV++ }
            samBusy = false
            mask?.let { it.recycle() }; mask = null   // 扣除完成，撤掉选择遮罩显示真实透明结果
            samStatus = "已扣除（点击处物体透明）"
            Toast.makeText(context, "已扣除选区", Toast.LENGTH_SHORT).show()
        }
    }

    // ===== 换色 =====
    fun doRecolor() {
        val m = mask ?: run { Toast.makeText(context, "请先点击物体", Toast.LENGTH_SHORT).show(); return }
        scope.launch {
            samBusy = true
            val out = withContext(Dispatchers.IO) { recolorByMask(work, m, hueDeg, satScale) }
            if (out != null) { work = out; workV++ }
            samBusy = false
            mask?.let { it.recycle() }; mask = null   // 换色完成，撤掉选择遮罩显示真实换色结果
            samStatus = "已换色（色相 ${hueDeg.toInt()}°）"
            Toast.makeText(context, "已换色", Toast.LENGTH_SHORT).show()
        }
    }

    // ===== 移动 =====
    fun doStartMove() {
        val m = mask ?: run { Toast.makeText(context, "请先点击物体", Toast.LENGTH_SHORT).show(); return }
        moving = true
        moveOffset = Offset.Zero
        moveOffsetImg = Offset.Zero
        // 生成移动实时预览：底图 = 物体擦除露棋盘；贴片 = 掩码内物体（透明背景）
        val w = work.width; val h = work.height
        val px = IntArray(w * h)
        val mp = IntArray(w * h)
        work.getPixels(px, 0, w, 0, 0, w, h)
        if (m.width == w && m.height == h) m.getPixels(mp, 0, w, 0, 0, w, h)
        val basePx = IntArray(w * h)
        val tilePx = IntArray(w * h)
        for (i in px.indices) {
            val inMask = (m.width == w && m.height == h) && ((mp[i] ushr 24) and 0xFF > 127)
            if (inMask) { basePx[i] = 0; tilePx[i] = px[i] or 0xFF000000.toInt() }
            else { basePx[i] = px[i]; tilePx[i] = 0 }
        }
        cleanBase = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)?.apply { setPixels(basePx, 0, w, 0, 0, w, h) }
        objectTile = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)?.apply { setPixels(tilePx, 0, w, 0, 0, w, h) }
        samStatus = "拖动移动物体，点「放下」完成"
        Toast.makeText(context, "拖动移动，松手后点「放下」", Toast.LENGTH_SHORT).show()
    }

    fun doDropMove() {
        if (moving && mask != null) {
            val mm = mask!!
            val dx = moveOffsetImg.x
            val dy = moveOffsetImg.y
            scope.launch {
                val out = withContext(Dispatchers.IO) { applyMove(work, original, mm, dx, dy) }
                if (out != null) { work = out; workV++ }
                moving = false
                moveOffset = Offset.Zero
                moveOffsetImg = Offset.Zero
                cleanBase?.recycle(); cleanBase = null
                objectTile?.recycle(); objectTile = null
                mask?.let { it.recycle() }; mask = null   // 移动完成，撤掉选择遮罩显示真实结果
                samStatus = "物体已移动"
                Toast.makeText(context, "物体已移动", Toast.LENGTH_SHORT).show()
            }
        } else {
            moving = false
            moveOffset = Offset.Zero
            moveOffsetImg = Offset.Zero
            cleanBase?.recycle(); cleanBase = null
            objectTile?.recycle(); objectTile = null
        }
    }

    fun doCancelMove() {
        moving = false
        moveOffset = Offset.Zero
        moveOffsetImg = Offset.Zero
        cleanBase?.recycle(); cleanBase = null
        objectTile?.recycle(); objectTile = null
        samStatus = "已取消移动"
    }

    Scaffold(
        containerColor = ComposeColor(0xFF101418),
        topBar = {
            TopAppBar(
                title = { Text("智能选区 · 点击分割", color = ComposeColor.White, fontSize = 17.sp) },
                navigationIcon = { IconButton(onClick = onBack) { Icon(Icons.Filled.ArrowBack, contentDescription = "返回", tint = ComposeColor.White) } },
                actions = {
                    TextButton(onClick = { if (!work.isRecycled) onApply(work) }) { Text("应用", color = ComposeColor(0xFF64B5F6)) }
                },
                colors = TopAppBarDefaults.topAppBarColors(containerColor = ComposeColor(0xFF101418))
            )
        },
        bottomBar = {
            Column(
                modifier = Modifier.fillMaxWidth().background(ComposeColor(0xFF1A1F24)).padding(horizontal = 14.dp, vertical = 10.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                if (moving) {
                    Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.fillMaxWidth()) {
                        OutlinedButton(onClick = { doCancelMove() }, modifier = Modifier.weight(1f)) { Text("取消", color = ComposeColor(0xFFFF8A80)) }
                        Button(onClick = { doDropMove() }, modifier = Modifier.weight(1f), colors = ButtonDefaults.buttonColors(containerColor = ComposeColor(0xFF4CAF50))) {
                            Text("放下（${moveOffsetImg.x.toInt()},${moveOffsetImg.y.toInt()}px）")
                        }
                    }
                } else {
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text("色相", color = ComposeColor(0xFF90A4AE), fontSize = 11.sp)
                        Slider(
                            value = hueDeg, onValueChange = { hueDeg = it },
                            valueRange = 0f..359f,
                            enabled = mask != null,
                            modifier = Modifier.weight(1f)
                        )
                        Text("${hueDeg.toInt()}°", color = ComposeColor.White, fontSize = 11.sp)
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        colorPresets.forEach { (name, h) ->
                            AssistChip(
                                onClick = { hueDeg = h },
                                enabled = mask != null,
                                label = { Text(name, fontSize = 11.sp) },
                                colors = AssistChipDefaults.assistChipColors(containerColor = ComposeColor(0xFF2A2F36))
                            )
                        }
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.fillMaxWidth()) {
                        Button(onClick = { doErase() }, enabled = mask != null && !samBusy, modifier = Modifier.weight(1f), colors = ButtonDefaults.buttonColors(containerColor = ComposeColor(0xFFC62828))) {
                            Icon(Icons.Filled.Delete, contentDescription = null); Spacer(Modifier.width(4.dp))
                            Text("扣除", fontSize = 13.sp, fontWeight = androidx.compose.ui.text.font.FontWeight.Bold)
                        }
                        Button(onClick = { doRecolor() }, enabled = mask != null && !samBusy, modifier = Modifier.weight(1f), colors = ButtonDefaults.buttonColors(containerColor = ComposeColor(0xFF7B1FA2))) {
                            Icon(Icons.Filled.Palette, contentDescription = null); Spacer(Modifier.width(4.dp))
                            Text("换色", fontSize = 13.sp, fontWeight = androidx.compose.ui.text.font.FontWeight.Bold)
                        }
                        Button(onClick = { doStartMove() }, enabled = mask != null && !samBusy, modifier = Modifier.weight(1f), colors = ButtonDefaults.buttonColors(containerColor = ComposeColor(0xFF1565C0))) {
                            Icon(Icons.Filled.OpenWith, contentDescription = null); Spacer(Modifier.width(4.dp))
                            Text("移动", fontSize = 13.sp, fontWeight = androidx.compose.ui.text.font.FontWeight.Bold)
                        }
                    }
                }
                Text(samStatus, color = ComposeColor(0xFF90A4AE), fontSize = 11.sp, modifier = Modifier.padding(top = 2.dp))
            }
        }
    ) { padding ->
        Box(modifier = Modifier.fillMaxSize().padding(padding).background(ComposeColor(0xFF101418))) {
            val density = LocalDensity.current
            var scale by remember { mutableStateOf(1f) }
            var offset by remember { mutableStateOf(Offset.Zero) }
            BoxWithConstraints(
                modifier = Modifier
                    .fillMaxSize()
                    .pointerInput(Unit) {
                        awaitPointerEventScope {
                            while (true) {
                                val evt = awaitPointerEvent()
                                val cw = size.width.toFloat()
                                val ch = size.height.toFloat()
                                val active = evt.changes.filter { it.pressed }

                                if (moving) {
                                    if (active.size == 1) {
                                        val p = active[0]
                                        if (evt.type == PointerEventType.Move) {
                                            val dxScreen = p.position.x - p.previousPosition.x
                                            val dyScreen = p.position.y - p.previousPosition.y
                                            moveOffset += Offset(dxScreen, dyScreen)
                                            val dw = work.width.toFloat(); val dh = work.height.toFloat()
                                            val fs = min(cw / dw, ch / dh)
                                            val imgScale = 1f / (fs * scale)
                                            moveOffsetImg += Offset(dxScreen * imgScale, dyScreen * imgScale)
                                            p.consume()
                                        }
                                    } else if (active.size >= 2) {
                                        val p0 = active[0]; val p1 = active[active.size - 1]
                                        val dist = sqrt((p1.position.x - p0.position.x).let { it * it } + (p1.position.y - p0.position.y).let { it * it })
                                        if (evt.type == PointerEventType.Move) {
                                            val prevDist = sqrt(
                                                (p1.previousPosition.x - p0.previousPosition.x).let { it * it } +
                                                (p1.previousPosition.y - p0.previousPosition.y).let { it * it })
                                            if (prevDist > 0) scale = (scale * dist / prevDist).coerceIn(0.5f, 8f)
                                        }
                                        evt.changes.forEach { it.consume() }
                                    }
                                } else {
                                    if (active.size >= 2) {
                                        val p0 = active[0]; val p1 = active[active.size - 1]
                                        val dist = sqrt((p1.position.x - p0.position.x).let { it * it } + (p1.position.y - p0.position.y).let { it * it })
                                        if (evt.type == PointerEventType.Move) {
                                            val prevDist = sqrt(
                                                (p1.previousPosition.x - p0.previousPosition.x).let { it * it } +
                                                (p1.previousPosition.y - p0.previousPosition.y).let { it * it })
                                            if (prevDist > 0) {
                                                val newScale = (scale * dist / prevDist).coerceIn(0.5f, 8f)
                                                val dw = work.width.toFloat(); val dh = work.height.toFloat()
                                                val fs = min(cw / dw, ch / dh)
                                                val fw = dw * fs; val fh = dh * fs
                                                val imgCx = (cw - fw * scale) / 2f + offset.x + fw * scale / 2f
                                                val imgCy = (ch - fh * scale) / 2f + offset.y + fh * scale / 2f
                                                val pinchCx = (p0.position.x + p1.position.x) / 2f
                                                val pinchCy = (p0.position.y + p1.position.y) / 2f
                                                val ratio = newScale / scale
                                                offset = Offset(
                                                    offset.x + (pinchCx - imgCx) * (1f - ratio),
                                                    offset.y + (pinchCy - imgCy) * (1f - ratio)
                                                )
                                                scale = newScale
                                            }
                                            offset += Offset(
                                                ((p0.position.x - p0.previousPosition.x) + (p1.position.x - p1.previousPosition.x)) / 2,
                                                ((p0.position.y - p0.previousPosition.y) + (p1.position.y - p1.previousPosition.y)) / 2)
                                        }
                                        evt.changes.forEach { it.consume() }
                                    } else if (active.size == 1) {
                                        // 单指点击 = 分割该点物体
                                        val p = active[0]
                                        if (evt.type == PointerEventType.Press) {
                                            val dw = work.width.toFloat(); val dh = work.height.toFloat()
                                            val fs = min(cw / dw, ch / dh)
                                            val fw = dw * fs; val fh = dh * fs
                                            val iL = (cw - fw * scale) / 2f + offset.x
                                            val iT = (ch - fh * scale) / 2f + offset.y
                                            val ix = ((p.position.x - iL) / (fw * scale)).coerceIn(0f, 1f) * dw
                                            val iy = ((p.position.y - iT) / (fh * scale)).coerceIn(0f, 1f) * dh
                                            val pt = intArrayOf(ix.toInt().coerceIn(0, work.width - 1), iy.toInt().coerceIn(0, work.height - 1))
                                            tapPoint = pt
                                            val eng = obtainEngine()
                                            if (eng != null) {
                                                samBusy = true
                                                samStatus = "分割中..."
                                                scope.launch {
                                                    val seg = withContext(Dispatchers.IO) { runCatching { eng.segmentAt(original, pt) }.getOrNull() }
                                                    if (seg != null) {
                                                        mask?.let { if (it !== seg) it.recycle() }
                                                        mask = seg
                                                        samStatus = "已选中该物体，可扣除/换色/移动（点其他位置重新选择）"
                                                    } else {
                                                        samStatus = "分割失败，请点物体内部"
                                                        Toast.makeText(context, "分割失败，请点击物体内部", Toast.LENGTH_SHORT).show()
                                                    }
                                                    samBusy = false
                                                }
                                            }
                                            p.consume()
                                        }
                                    }
                                }
                            }
                        }
                    }
            ) {
                val cell = with(density) { 20.dp.toPx() }
                ComposeCanvas(Modifier.fillMaxSize()) {
                    var x = 0f
                    while (x < size.width) { var y = 0f; while (y < size.height) {
                        drawRect(if (((x / cell).toInt() + (y / cell).toInt()) % 2 == 0) ComposeColor(0xFF2A2F36) else ComposeColor(0xFF1C2127), Offset(x, y), Size(cell, cell))
                        y += cell }; x += cell }
                }
                if (!work.isRecycled) {
                    val d = density
                    val cwPx = with(d) { maxWidth.toPx() }
                    val chPx = with(d) { maxHeight.toPx() }
                    val dw = work.width.toFloat(); val dh = work.height.toFloat()
                    val fs = min(cwPx / dw, chPx / dh)
                    val fw = dw * fs; val fh = dh * fs
                    val iL = (cwPx - fw * scale) / 2f + offset.x
                    val iT = (chPx - fh * scale) / 2f + offset.y
                    val imgRect = RectF(iL, iT, iL + fw * scale, iT + fh * scale)
                    ComposeCanvas(Modifier.fillMaxSize()) {
                        val native = drawContext.canvas.nativeCanvas
                        native.drawBitmap(work, null, imgRect, Paint().apply { isFilterBitmap = true })

                        // 移动实时预览：画底图 + 平移贴片（不画选择框，直接看到物体被拖动）
                        if (moving) {
                            val cb = cleanBase
                            val tile = objectTile
                            if (cb != null && !cb.isRecycled && tile != null && !tile.isRecycled) {
                                native.drawBitmap(cb, null, imgRect, Paint().apply { isFilterBitmap = true })
                                val tileRect = RectF(
                                    iL + moveOffset.x,
                                    iT + moveOffset.y,
                                    iL + fw * scale + moveOffset.x,
                                    iT + fh * scale + moveOffset.y
                                )
                                native.drawBitmap(tile, null, tileRect, Paint().apply { isFilterBitmap = true })
                            }
                        } else {
                            // 掩码高亮（半透明青色）——仅在存在选区且未执行操作时显示
                            val m = mask
                            if (m != null && !m.isRecycled) {
                                val hlBmp = makeHighlightBitmap(m)
                                if (hlBmp != null) {
                                    native.drawBitmap(hlBmp, null, imgRect, Paint().apply { isAntiAlias = true })
                                    hlBmp.recycle()
                                }
                                // 掩码 bbox 青框
                                val mp = IntArray(m.width * m.height)
                                m.getPixels(mp, 0, m.width, 0, 0, m.width, m.height)
                                val sx = fw * scale / dw
                                val sy = fh * scale / dh
                                var mL = m.width; var mT = m.height; var mR = -1; var mB = -1
                                for (y in 0 until m.height) {
                                    for (x in 0 until m.width) {
                                        if ((mp[y * m.width + x] ushr 24) and 0xFF > 127) {
                                            if (x < mL) mL = x; if (x > mR) mR = x
                                            if (y < mT) mT = y; if (y > mB) mB = y
                                        }
                                    }
                                }
                                if (mR >= mL && mB >= mT) {
                                    native.drawRect(iL + mL * sx, iT + mT * sy, iL + mR * sx, iT + mB * sy,
                                        Paint().apply { style = Paint.Style.STROKE; strokeWidth = 2f; color = 0xFF00E5FF.toInt() })
                                }
                            }
                        }

                        // 点击点标记
                        tapPoint?.let { tp ->
                            val sx = fw * scale / dw
                            val sy = fh * scale / dh
                            native.drawCircle(iL + tp[0] * sx, iT + tp[1] * sy, 8f, Paint().apply { color = 0xFFFF4081.toInt() })
                            native.drawCircle(iL + tp[0] * sx, iT + tp[1] * sy, 8f, Paint().apply { style = Paint.Style.STROKE; strokeWidth = 2f; color = 0xFFFFFFFF.toInt() })
                        }
                    }
                }
            }
        }
    }
}

/** 把掩码做成半透明青色高亮 Bitmap（掩码内青色 55% 叠加，比旧版更清晰可见） */
internal fun makeHighlightBitmap(mask: Bitmap): Bitmap? {
    if (mask.isRecycled) return null
    val w = mask.width; val h = mask.height
    val mp = IntArray(w * h)
    mask.getPixels(mp, 0, w, 0, 0, w, h)
    val out = IntArray(w * h)
    for (i in mp.indices) {
        if ((mp[i] ushr 24) and 0xFF > 127) {
            // 半透明青（alpha 140，RGB 0,229,255）——比旧版 100 更明显
            out[i] = (140 shl 24) or 0xFF00E5FF.toInt()
        } else {
            out[i] = 0
        }
    }
    val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888) ?: return null
    bmp.setPixels(out, 0, w, 0, 0, w, h)
    return bmp
}

/**
 * 掩码覆盖图：半透明青色填充 + 【非常明显的双层边缘描边】。
 * - 掩码内：半透明青（alpha 90，可见原图）
 * - 掩码边缘（内1px）：不透明白色（最醒目）
 * - 掩码外紧邻（外1px）：不透明荧光青（高对比描边）
 * 双层描边让选区轮廓在任何背景上都清晰可辨。
 */
internal fun makeMaskOverlayBitmap(mask: Bitmap): Bitmap? {
    if (mask.isRecycled) return null
    val w = mask.width; val h = mask.height
    val mp = IntArray(w * h)
    mask.getPixels(mp, 0, w, 0, 0, w, h)
    // 掩码布尔
    val inM = BooleanArray(w * h)
    for (i in mp.indices) inM[i] = ((mp[i] ushr 24) and 0xFF) > 127
    // ★ 距离场：掩码外每个像素到最近掩码像素的曼哈顿距离
    //   用于生成多像素宽的外圈描边（3px 宽荧光黄绿，在任何背景上都醒目）
    val dist = IntArray(w * h) { Int.MAX_VALUE }
    // 初始化：掩码内距离=0，掩码外先设大值
    for (i in inM.indices) dist[i] = if (inM[i]) 0 else Int.MAX_VALUE
    // 前向扫描
    for (y in 0 until h) {
        for (x in 0 until w) {
            val i = y * w + x
            if (dist[i] == 0) continue
            val left = if (x > 0) dist[i - 1] else Int.MAX_VALUE
            val up = if (y > 0) dist[i - w] else Int.MAX_VALUE
            val minN = minOf(left, up)
            if (minN != Int.MAX_VALUE) dist[i] = minN + 1
        }
    }
    // 后向扫描
    for (y in h - 1 downTo 0) {
        for (x in w - 1 downTo 0) {
            val i = y * w + x
            if (dist[i] == 0) continue
            val right = if (x < w - 1) dist[i + 1] else Int.MAX_VALUE
            val down = if (y < h - 1) dist[i + w] else Int.MAX_VALUE
            val minN = minOf(right, down)
            if (minN != Int.MAX_VALUE) dist[i] = minOf(dist[i], minN + 1)
        }
    }
    // ★ 描边参数
    //   外圈：6px 宽，不透明荧光黄绿（0xFFCCFF00），高对比、任何背景上都醒目（轮廓加粗）
    //   内圈：2px 宽，不透明黑（0xFF000000），与外圈形成双层对比，加粗后轮廓更醒目
    //   填充：★ 半透明青 alpha≈55 —— 更透明，几乎不遮挡原图内容，仅淡淡标记选中区域
    //         轮廓线（外圈荧光黄绿 + 内圈黑）保持醒目清晰
    //   ★ 注意：RGB 部分必须【不含 alpha】用 0x00E5FF，再 `(55 shl 24) or` 叠加 alpha。
    //    若写成 0xFF00E5FF 再或 alpha，会把 alpha 位覆盖成 0xFF → 完全不透明（此前 bug）。
    // ★ 掩码内部到最近背景的距离场（内圈黑描边用，2px 宽）
    val inEdge = IntArray(w * h) { Int.MAX_VALUE }
    for (i in inM.indices) inEdge[i] = if (inM[i]) Int.MAX_VALUE else 0
    // 前向扫描
    for (y in 0 until h) {
        for (x in 0 until w) {
            val i = y * w + x
            if (inEdge[i] == 0) continue
            val left = if (x > 0) inEdge[i - 1] else Int.MAX_VALUE
            val up = if (y > 0) inEdge[i - w] else Int.MAX_VALUE
            val minN = minOf(left, up)
            if (minN != Int.MAX_VALUE) inEdge[i] = minN + 1
        }
    }
    // 后向扫描
    for (y in h - 1 downTo 0) {
        for (x in w - 1 downTo 0) {
            val i = y * w + x
            if (inEdge[i] == 0) continue
            val right = if (x < w - 1) inEdge[i + 1] else Int.MAX_VALUE
            val down = if (y < h - 1) inEdge[i + w] else Int.MAX_VALUE
            val minN = minOf(right, down)
            if (minN != Int.MAX_VALUE) inEdge[i] = minOf(inEdge[i], minN + 1)
        }
    }
    val out = IntArray(w * h)
    for (y in 0 until h) {
        for (x in 0 until w) {
            val i = y * w + x
            if (!inM[i]) {
                // 背景像素：按距离决定外圈描边
                val d = dist[i]
                out[i] = when {
                    d <= 6 -> 0xFFCCFF00.toInt()  // 外圈6px：荧光黄绿（alpha=0xFF 全不透明，作为描边醒目）
                    else -> 0
                }
            } else {
                // 掩码内：按到背景的距离决定内圈黑色描边（2px）；否则半透明青填充
                val ie = inEdge[i]
                out[i] = when {
                    ie <= 2 -> 0xFF000000.toInt()  // 内边缘2px：不透明黑（粗描边）
                    else -> (55 shl 24) or 0x00E5FF // ★ 半透明青填充 alpha=55（RGB 无 alpha，真正透明）
                }
            }
        }
    }
    val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888) ?: return null
    bmp.setPixels(out, 0, w, 0, 0, w, h)
    return bmp
}

// ================= 位图操作（顶层函数）=================

/** 扣除：mask 内 alpha 清 0（RGB 保留原图色） */
internal fun eraseByMask(base: Bitmap, mask: Bitmap): Bitmap? {
    if (base.isRecycled || mask.isRecycled) return null
    val w = base.width; val h = base.height
    val m = if (mask.width == w && mask.height == h) mask else null
    val out = base.copy(Bitmap.Config.ARGB_8888, true) ?: return null
    val px = IntArray(w * h)
    val mp = IntArray(w * h)
    out.getPixels(px, 0, w, 0, 0, w, h)
    if (m != null) {
        m.getPixels(mp, 0, w, 0, 0, w, h)
        for (i in px.indices) {
            if ((mp[i] ushr 24) and 0xFF > 127) px[i] = px[i] and 0x00FFFFFF
        }
    }
    out.setPixels(px, 0, w, 0, 0, w, h)
    return out
}

/**
 * 换色（色相相对平移 + 驼峰力度曲线）：
 * 以掩码内【主色相】为基准，把主色平移向目标色相 hueDeg，其它颜色跟随相对平移
 * （保留画面色彩多样性，不会所有颜色都强制变成目标色）。
 * - 力度按像素饱和度走【驼峰曲线】：低饱和(白/浅)几乎不动、中等饱和(主体)全移、
 *   重色区域(高饱和)力度回落减弱——避免原本就浓的颜色越改越重。
 * - 饱和度只做轻微提升（最多 +8%），不再强行拉满。
 * - 亮度整体提亮 +0.16，且跟随力度曲线（背景/浅色基本不提亮）。
 */
internal fun recolorByMask(base: Bitmap, mask: Bitmap?, hueDeg: Float, satScale: Float): Bitmap? {
    if (base.isRecycled) return null
    val w = base.width; val h = base.height
    // ★★ 硬保护：mask 缺失（null/已回收/无法对齐）时【直接返回原图】，绝不退化成全图调色。
    //   分区调色是「只对选中分区生效」的铁律——一旦 mask 失效宁可不动，也不能误改未选中区域。
    var m = if (mask != null && !mask.isRecycled) mask else null
    if (m == null) return base.copy(Bitmap.Config.ARGB_8888, true)
    if (m.width != w || m.height != h) {
        try {
            m = Bitmap.createScaledBitmap(m, w, h, true)
        } catch (_: Exception) {
            // 缩放失败：绝不退化全图，直接返回原图
            return base.copy(Bitmap.Config.ARGB_8888, true)
        }
    }
    val out = base.copy(Bitmap.Config.ARGB_8888, true) ?: return null
    val px = IntArray(w * h)
    val mp = IntArray(w * h)
    out.getPixels(px, 0, w, 0, 0, w, h)
    m.getPixels(mp, 0, w, 0, 0, w, h)

    // ---- 第 1 遍：统计掩码内彩色像素的【主色相】（循环平均，避免 0/360 跳变）----
    var sinSum = 0.0
    var cosSum = 0.0
    var count = 0
    val hsv0 = FloatArray(3)
    for (i in px.indices) {
        val inMask = ((mp[i] ushr 24) and 0xFF > 127)
        if (!inMask) continue
        val c = px[i]
        val r = (c shr 16) and 0xFF; val g = (c shr 8) and 0xFF; val b = c and 0xFF
        Color.RGBToHSV(r, g, b, hsv0)
        if (hsv0[1] <= 0.12f) continue          // 只看彩色像素
        val rad = Math.toRadians(hsv0[0].toDouble())
        sinSum += Math.sin(rad); cosSum += Math.cos(rad)
        count++
    }
    // ★ 掩码内没有任何有效像素 → 直接返回原图，避免异常/全图
    if (count == 0) return out
    val avgHue = Math.toDegrees(Math.atan2(sinSum, cosSum)).toFloat().mod(360f)
    // 主色平移量（目标色相 - 主色相），其它颜色跟随相对偏移
    val delta = (hueDeg.mod(360f) - avgHue).mod(360f)

    // ---- 驼峰力度曲线 strength(s) ----
    //   s=0      -> 0.10   低饱和(白/浅)几乎不动
    //   s=0.10~0.50 -> 1.00  中等饱和(主体)完整平移
    //   s=0.50~1.00 -> 0.55  重色区域力度回落，避免越改越重
    fun strength(s: Float): Float {
        val ss = s.coerceIn(0f, 1f)
        return when {
            ss < 0.10f -> {
                val t = ss / 0.10f
                0.10f + 0.90f * t * t * (3f - 2f * t)          // 低段 smoothstep 抬升
            }
            ss <= 0.50f -> 1f                                  // 中段主色全移
            else -> {
                val t = (ss - 0.50f) / 0.50f
                1f - 0.45f * t * t * (3f - 2f * t)             // 高段 smoothstep 回落
            }
        }
    }

    val satBoost = 0.08f * satScale.coerceIn(0.5f, 1.5f)       // 轻微饱和度提升（最多 +8%）
    val bright = 0.16f
    val hsv = FloatArray(3)
    for (i in px.indices) {
        val inMask = ((mp[i] ushr 24) and 0xFF > 127)
        if (!inMask) continue
        val c = px[i]
        val a = (c ushr 24) and 0xFF
        val r = (c shr 16) and 0xFF; val g = (c shr 8) and 0xFF; val b = c and 0xFF
        Color.RGBToHSV(r, g, b, hsv)
        val st = strength(hsv[1])
        // 色相相对平移，跟随力度
        hsv[0] = (hsv[0] + delta * st).mod(360f)
        // 饱和度：保留原值 + 轻微提升（跟随力度）
        hsv[1] = (hsv[1] + satBoost * (1f - hsv[1]) * st).coerceIn(0f, 1f)
        // 亮度：整体提亮 +0.16，跟随力度（浅色/背景几乎不提亮）
        hsv[2] = (hsv[2] + bright * st).coerceIn(0f, 1f)
        val newPixel = Color.HSVToColor(hsv)
        px[i] = ((a shl 24) and 0xFF000000.toInt()) or (newPixel and 0x00FFFFFF)
    }
    out.setPixels(px, 0, w, 0, 0, w, h)
    return out
}

/**
 * 全图调色：绝对色相旋转（矩阵法，比逐像素 HSV 快得多，效果明显）。
 * 用标准 RGB 色相旋转矩阵对整幅图旋转 hueDeg，饱和度按 satScale 缩放。
 * 供后处理「全图调色」使用（整幅图变色，不追踪单物体主色相 → 复杂图也肉眼可见）。
 */
internal fun rotateHueAll(base: Bitmap, hueDeg: Float, satScale: Float): Bitmap? {
    if (base.isRecycled) return null
    val w = base.width; val h = base.height
    val out = base.copy(Bitmap.Config.ARGB_8888, true) ?: return null
    val px = IntArray(w * h)
    out.getPixels(px, 0, w, 0, 0, w, h)
    val rad = (hueDeg * (Math.PI / 180.0)).toFloat()
    val cos = kotlin.math.cos(rad).toFloat()
    val sin = kotlin.math.sin(rad).toFloat()
    // 标准色相旋转矩阵系数
    val lumR = 0.213f; val lumG = 0.715f; val lumB = 0.072f
    val m00 = lumR + cos * (1f - lumR) + sin * (-lumR)
    val m01 = lumG + cos * (-lumG) + sin * (-lumG)
    val m02 = lumB + cos * (-lumB) + sin * (1f - lumB)
    val m10 = lumR + cos * (-lumR) + sin * (0.143f)
    val m11 = lumG + cos * (1f - lumG) + sin * (0.140f)
    val m12 = lumB + cos * (-lumB) + sin * (-0.283f)
    val m20 = lumR + cos * (-lumR) + sin * (-(1f - lumR))
    val m21 = lumG + cos * (-lumG) + sin * (lumG)
    val m22 = lumB + cos * (1f - lumB) + sin * (lumB)
    val sr = satScale
    for (i in px.indices) {
        val c = px[i]
        val a = (c ushr 24) and 0xFF
        var r = (c shr 16) and 0xFF
        var g = (c shr 8) and 0xFF
        var b = c and 0xFF
        // 饱和度缩放（围绕亮度轴缩放彩色分量）
        if (sr != 1f) {
            val gray = r * 0.2126f + g * 0.7152f + b * 0.0722f
            r = (gray + (r - gray) * sr).coerceIn(0f, 255f).toInt()
            g = (gray + (g - gray) * sr).coerceIn(0f, 255f).toInt()
            b = (gray + (b - gray) * sr).coerceIn(0f, 255f).toInt()
        }
        val nr: Int = (m00 * r + m01 * g + m02 * b).coerceIn(0f, 255f).toInt()
        val ng: Int = (m10 * r + m11 * g + m12 * b).coerceIn(0f, 255f).toInt()
        val nb: Int = (m20 * r + m21 * g + m22 * b).coerceIn(0f, 255f).toInt()
        px[i] = ((a shl 24) and 0xFF000000.toInt()) or (nr shl 16) or (ng shl 8) or nb
    }
    out.setPixels(px, 0, w, 0, 0, w, h)
    return out
}
internal fun applyMove(base: Bitmap, srcPixels: Bitmap, mask: Bitmap, dxPx: Float, dyPx: Float): Bitmap? {
    if (base.isRecycled || mask.isRecycled) return null
    val w = base.width; val h = base.height
    val m = if (mask.width == w && mask.height == h) mask else null
    val out = base.copy(Bitmap.Config.ARGB_8888, true) ?: return null
    val px = IntArray(w * h)
    val mp = IntArray(w * h)
    out.getPixels(px, 0, w, 0, 0, w, h)
    if (m != null) {
        m.getPixels(mp, 0, w, 0, 0, w, h)
        for (i in px.indices) {
            if ((mp[i] ushr 24) and 0xFF > 127) px[i] = px[i] and 0x00FFFFFF
        }
        out.setPixels(px, 0, w, 0, 0, w, h)
    }
    val srcPx = IntArray(w * h)
    srcPixels.getPixels(srcPx, 0, w, 0, 0, w, h)
    val outPx = IntArray(w * h)
    out.getPixels(outPx, 0, w, 0, 0, w, h)
    val dx = dxPx.toInt()
    val dy = dyPx.toInt()
    for (y in 0 until h) {
        val sy = y - dy
        if (sy < 0 || sy >= h) continue
        for (x in 0 until w) {
            val sxx = x - dx
            if (sxx < 0 || sxx >= w) continue
            val si = sy * w + sxx
            val inMask = m == null || ((mp[si] ushr 24) and 0xFF > 127)
            if (inMask) {
                outPx[y * w + x] = srcPx[si] or 0xFF000000.toInt()
            }
        }
    }
    out.setPixels(outPx, 0, w, 0, 0, w, h)
    return out
}