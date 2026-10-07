package com.rmbg.offline

import android.graphics.Bitmap
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.graphics.Rect
import android.graphics.RectF
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
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.input.pointer.PointerEventType
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlin.math.min
import kotlin.math.sqrt
import kotlin.math.roundToInt

/** 后处理抠图画布
 *  ★ 参考 kortex_ref ManualMaskPainter 完整逻辑：
 *    - 单指：Press 即起笔、Move 续画（quadTo 平滑）、Release 落盘整笔（无 touchSlop 门槛，杜绝"拖不动"）
 *    - 双指：缩放/平移（以手势中心为锚点，可精细化操作）
 *    - Canvas 内自包含 translate+scale 并绘制底图，坐标与位图像素一一对应
 *  - 橡皮(擦除)：alpha 清 0，RGB 保留原图色（可随时恢复）
 *  - 画笔(恢复)：从原图完整取样 RGBA（独立功能，非擦除反写）
 *  @param onModified 像素已变化后回调（触发重组刷新显示）
 *  @param onStroke 每一笔开始回调
 */
@Composable
 fun EditBrushLayer(
     modifier: Modifier = Modifier,
     displayBitmap: Bitmap,       // 底图 + 可直接修改的结果位图（可编辑副本）
     original: Bitmap,            // 原图（画笔恢复时取原图色）
     brushColor: Color,
    brushSize: Float,            // 图像坐标笔宽（px）
    isEraser: Boolean,
    viewMode: Int = 0,           // ★ 画布显示模式：0=编辑(暗色原图底) 1=对比(整体反色) 2=预览(棋盘格+结果图)
    onModified: () -> Unit = {},
    onStroke: () -> Unit = {}
 ) {
    // ★★ 防御：original 和 displayBitmap 可能是同一实例（editorOnOriginal=true 时
    //   PostProcessScreen 传 result=originalBitmap, originBitmap=originalBitmap → 同一对象）。
    //   如果同一实例，applyBrush 修改 displayBitmap 的 alpha 会同时污染 original →
    //   重绘时底层暗色原图也丢了 alpha → 擦除区什么都没有（"笔迹消失/反弹"），
    //   且之前擦除的区域也丢失（"切换区域后丢失"）。
    //   解决：强制复制一份独立 original，永远不被 displayBitmap 的修改波及。
    val safeOriginal = remember(original, displayBitmap) {
        if (original === displayBitmap) {
            // ★ 同实例时复制，且强制 alpha=255（原图永远不透明）：
            //   displayBitmap 可能已被擦除（alpha=0），如果直接 copy，safeOriginal 那些
            //   擦除区也是 alpha=0 → 恢复时取到 alpha=0 → 落盘后 displayBitmap 该区透明 →
            //   重绘露出暗色底 → 恢复看起来变暗（"暗色不在遮罩上"根因）。
            val w = original.width; val h = original.height
            val px = IntArray(w * h)
            original.getPixels(px, 0, w, 0, 0, w, h)
            for (i in px.indices) {
                // RGB 保留，alpha 强制 255（原图是不透明的）
                px[i] = (px[i] and 0x00FFFFFF) or 0xFF000000.toInt()
            }
            val copy = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            copy.setPixels(px, 0, w, 0, 0, w, h)
            copy
        } else original
    }
    // ★ 刷新计数：落盘(commitStroke)后 +1，Canvas 绘制块依赖它，确保 setPixels 修改的
    //   displayBitmap 像素变化能立即触发本层重组重绘（仅外部 onModified 不会让内部 draw 重跑）。
    var commitTick by remember { mutableStateOf(0) }
    // ★ 边缘高亮 mask（对比模式用）：把结果图的「半透明边缘像素」染成高亮色，
    //   主体(alpha≈255)与背景(alpha≈0)保持清晰，半透明毛边显形——同一张图分层次感。
    var edgeMask by remember { mutableStateOf<Bitmap?>(null) }
    LaunchedEffect(viewMode, commitTick) {
        if (viewMode == 1) {
            val bmp = displayBitmap
            if (!bmp.isRecycled) {
                try {
                    val w = bmp.width; val h = bmp.height
                    val px = IntArray(w * h); val out = IntArray(w * h)
                    bmp.getPixels(px, 0, w, 0, 0, w, h)
                    for (i in px.indices) {
                        val a = (px[i] ushr 24) and 0xFF
                        // 半透明边缘：alpha 在 8~247 之间（排除几乎全透明/几乎不透明）
                        if (a in 8 until 248) {
                            // 高亮亮青 + 增强 alpha（透明显形），越不透明越实
                            val ea = (a * 2).coerceAtMost(255)
                            out[i] = (ea shl 24) or 0xFF00E5FF.toInt()
                        } else {
                            out[i] = 0
                        }
                    }
                    val m = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
                    m.setPixels(out, 0, w, 0, 0, w, h)
                    edgeMask = m
                } catch (_: Exception) { edgeMask = null }
            }
        } else edgeMask = null
    }
    // 画布缩放/平移（自包含）
    var scale by remember { mutableStateOf(1f) }
    var offsetX by remember { mutableStateOf(0f) }
    var offsetY by remember { mutableStateOf(0f) }
    // 当前一笔（图像坐标点集）
    var currentPath by remember { mutableStateOf<List<Offset>>(emptyList()) }
    var isDrawing by remember { mutableStateOf(false) }
    // ★ 放大镜中心（当前手指位置，跟随移动）；放大镜在单指按下时出现，辅助精细画笔
    var magnifierCenter by remember { mutableStateOf(Offset.Zero) }
    // 放大镜当前是否显示（单指按住显示，抬起/双指收起）
    var magnifyActive by remember { mutableStateOf(false) }
    // 画笔模式是否开启放大镜（默认开启，辅助精修）
    val magnifyOn = true
    // 密度（放大镜像素尺寸换算）
    val density = LocalDensity.current

    // 落盘 Paint（图像坐标）
    val strokePaint = remember {
        Paint().apply {
            isAntiAlias = true
            style = Paint.Style.STROKE
            strokeCap = Paint.Cap.ROUND
            strokeJoin = Paint.Join.ROUND
        }
    }
    // 落盘 Paint 随设置更新
    LaunchedEffect(brushSize, isEraser) {
        strokePaint.strokeWidth = brushSize
        if (isEraser) {
            strokePaint.color = android.graphics.Color.TRANSPARENT
            strokePaint.xfermode = PorterDuffXfermode(PorterDuff.Mode.CLEAR)
        } else {
            strokePaint.color = android.graphics.Color.WHITE
            strokePaint.alpha = 255
            strokePaint.xfermode = PorterDuffXfermode(PorterDuff.Mode.SRC_OVER)
        }
    }

    // 构建一条描边 path（图像坐标）
    fun buildPath(pts: List<Offset>): Path {
        val p = Path()
        p.moveTo(pts[0].x, pts[0].y)
        for (i in 1 until pts.size) p.lineTo(pts[i].x, pts[i].y)
        return p
    }

    // ★★ 遮罩画笔（扣空/恢复统一）——画笔编辑的永远是「遮罩层 alpha」：
    //   扣空(erase=true)：笔迹区域 alpha 置 0（透明露棋盘），RGB 保留原图色
    //   恢复(erase=false)：笔迹区域 alpha 置 255 + RGB 从原图取样（恢复原背景色，永不丢色）
    //   ★ 核心：RGB 颜色源永远来自 safeOriginal（原图），绝不 CLEAR 连 RGB 一起清 0！
    //     旧版 CLEAR 把 RGB 清 0 → 恢复取不回原色（恢复失效根因）。
    //     像素级局部操作（仅笔迹包围盒），与即时渲染一致的圆头笔迹形状。
    fun applyBrush(pts: List<Offset>, erase: Boolean) {
        val w = displayBitmap.width
        val h = displayBitmap.height
        if (pts.isEmpty()) return
        val r = brushSize / 2f
        // 构建笔迹填充形状（图像坐标）：单点=圆；多点=圆头描边填充
        val maskFill = Path()
        if (pts.size == 1) {
            maskFill.addCircle(pts[0].x, pts[0].y, r, Path.Direction.CW)
        } else {
            val path = Path()
            path.moveTo(pts[0].x, pts[0].y)
            for (i in 1 until pts.size) path.lineTo(pts[i].x, pts[i].y)
            val sp = Paint().apply {
                isAntiAlias = true; style = Paint.Style.STROKE
                strokeCap = Paint.Cap.ROUND; strokeJoin = Paint.Join.ROUND
                strokeWidth = brushSize
            }
            sp.getFillPath(path, maskFill)
        }
        // 笔迹包围盒（图像坐标，限幅到位图内）
        val bf = RectF()
        maskFill.computeBounds(bf, true)
        val l = bf.left.toInt().coerceIn(0, w - 1)
        val t = bf.top.toInt().coerceIn(0, h - 1)
        val rr = bf.right.toInt().coerceIn(l + 1, w)
        val bb = bf.bottom.toInt().coerceIn(t + 1, h)
        if (rr <= l || bb <= t) return
        val bw2 = rr - l
        val bh2 = bb - t
        // ★ Region 判断像素是否在笔迹形状内（避免 Compose 作用域 Path.contains 解析冲突）
        val reg = android.graphics.Region()
        reg.setPath(maskFill, android.graphics.Region(l, t, rr, bb))
        // 局部读取当前像素 + 原图像素（同尺寸用原图；异尺寸用自身 RGB——引擎输出已保证 RGB=原图色）
        val cur = IntArray(bw2 * bh2)
        displayBitmap.getPixels(cur, 0, bw2, l, t, bw2, bh2)
        val srcFromOriginal = safeOriginal.width == w && safeOriginal.height == h && !safeOriginal.isRecycled
        val src = if (srcFromOriginal) {
            val o = IntArray(bw2 * bh2)
            safeOriginal.getPixels(o, 0, bw2, l, t, bw2, bh2)
            o
        } else {
            cur
        }
        for (y in 0 until bh2) {
            for (x in 0 until bw2) {
                val i = y * bw2 + x
                if (reg.contains(l + x, t + y)) {
                    if (erase) {
                        // ★ 擦除=遮罩 alpha 清 0，RGB 保留（随时可取回）
                        cur[i] = cur[i] and 0x00FFFFFF
                    } else {
                        // ★ 恢复=独立功能（绝不是擦除反写）：从原图完整取样 RGBA——
                        //   原图本身半透明/透明区域也精确还原（与即时渲染画原图完全一致）。
                        //   仅当原图不可用（异尺寸/被回收）才退回「alpha 填 255 + 自身 RGB」兜底。
                        cur[i] = if (srcFromOriginal) src[i]
                                 else (cur[i] and 0x00FFFFFF) or 0xFF000000.toInt()
                    }
                }
            }
        }
        displayBitmap.setPixels(cur, 0, bw2, l, t, bw2, bh2)
    }

    // 落盘整笔：把图像坐标 path 真正作用于 displayBitmap 像素（扣空=alpha0，恢复=alpha255，对称）
    //   ★ isEraser 由调用方传入，避免 pointerInput(Unit) 闭包捕获旧值（PostProcessScreen 默认 true，
    //   切恢复后 pointerInput 不重启 → commitStroke 仍用旧 isEraser=true → 恢复松手变擦除）。
    fun commitStroke(erase: Boolean) {
        val pts = currentPath
        if (pts.isEmpty()) return
        try {
            applyBrush(pts, erase)
        } catch (_: Exception) {}
        commitTick++   // 强制本层重组，重绘已落盘的像素变化
        onModified()
    }

    Canvas(
        modifier = modifier
            .fillMaxSize()
            .pointerInput(isEraser) {
                // ★ 参考 kortex_ref ManualMaskPainter：awaitPointerEvent 手动处理 Press/Move/Release + 双指缩放
                awaitPointerEventScope {
                    while (true) {
                        val event = awaitPointerEvent()
                        val pointers = event.changes
                        when {
                            // 两指及以上：缩放/平移
                            pointers.size >= 2 -> {
                                magnifyActive = false
                                if (isDrawing) {
                                    commitStroke(isEraser)
                                    currentPath = emptyList()
                                    isDrawing = false
                                }
                                val p0 = pointers[0]
                                val p1 = pointers[1]
                                val dist = sqrt((p1.position.x - p0.position.x).let { it * it } +
                                    (p1.position.y - p0.position.y).let { it * it })
                                if (event.type == PointerEventType.Move) {
                                    val prevDist = sqrt(
                                        (p1.previousPosition.x - p0.previousPosition.x).let { it * it } +
                                        (p1.previousPosition.y - p0.previousPosition.y).let { it * it }
                                    )
                                    if (prevDist > 0) {
                                        val newScale = (scale * (dist / prevDist)).coerceIn(0.5f, 8f)
                                        val centerX = (p0.position.x + p1.position.x) / 2
                                        val centerY = (p0.position.y + p1.position.y) / 2
                                        // ★ 统一坐标模型：缩放围绕两指中心，保证中心点对应同一图像像素。
                                        val cw = size.width.toFloat()
                                        val ch = size.height.toFloat()
                                        val bw2 = displayBitmap.width.toFloat()
                                        val bh2 = displayBitmap.height.toFloat()
                                        val fitS = min(cw / bw2, ch / bh2)
                                        val imgW_old = bw2 * fitS * scale
                                        val imgH_old = bh2 * fitS * scale
                                        val effL = cw / 2f - imgW_old / 2f + offsetX
                                        val effT = ch / 2f - imgH_old / 2f + offsetY
                                        val imgW_new = bw2 * fitS * newScale
                                        val imgH_new = bh2 * fitS * newScale
                                        offsetX = centerX - (centerX - effL) * (newScale / scale) - cw / 2f + imgW_new / 2f
                                        offsetY = centerY - (centerY - effT) * (newScale / scale) - ch / 2f + imgH_new / 2f
                                        scale = newScale
                                    }
                                    val prevCx = (p0.previousPosition.x + p1.previousPosition.x) / 2
                                    val prevCy = (p0.previousPosition.y + p1.previousPosition.y) / 2
                                    val cx = (p0.position.x + p1.position.x) / 2
                                    val cy = (p0.position.y + p1.position.y) / 2
                                    offsetX += cx - prevCx
                                    offsetY += cy - prevCy
                                }
                                pointers.forEach { it.consume() }
                            }
                            // 一指：绘制
                            pointers.size == 1 -> {
                                val p = pointers[0]
                                // ★ 放大镜中心跟随手指 + 单指按下时激活
                                magnifierCenter = p.position
                                if (event.type == PointerEventType.Press) magnifyActive = true
                                // ★ 与绘制一致的坐标模型：有效图像左上角 = 居中fit - fitW*scale/2 + offset
                                val cw = size.width.toFloat()
                                val ch = size.height.toFloat()
                                val bw2 = displayBitmap.width.toFloat()
                                val bh2 = displayBitmap.height.toFloat()
                                val fitS = min(cw / bw2, ch / bh2)
                                val fitW2 = bw2 * fitS
                                val fitH2 = bh2 * fitS
                                val imgW2 = fitW2 * scale
                                val imgH2 = fitH2 * scale
                                val effL = cw / 2f - imgW2 / 2f + offsetX
                                val effT = ch / 2f - imgH2 / 2f + offsetY
                                val ix = (p.position.x - effL) * bw2 / imgW2
                                val iy = (p.position.y - effT) * bh2 / imgH2
                                val imgPt = Offset(ix, iy)
                                when {
                                    event.type == PointerEventType.Press -> {
                                        if (ix in 0f..bw2 && iy in 0f..bh2) {
                                            isDrawing = true
                                            currentPath = listOf(imgPt)
                                            onStroke()
                                        }
                                    }
                                    event.type == PointerEventType.Move && isDrawing -> {
                                        currentPath = currentPath + imgPt
                                    }
                                    event.type == PointerEventType.Release -> {
                                        // ★ 无条件收起放大镜：无论是否绘制中，手指抬起即消失
                                        magnifyActive = false
                                        if (isDrawing) {
                                            commitStroke(isEraser)
                                            currentPath = emptyList()
                                            isDrawing = false
                                        }
                                    }
                                }
                                if (isDrawing) p.consume()
                            }
                        }
                    }
                }
            }
    ) {
        // 自包含绘制：translate+scale 后画底图 + 即时渲染当前笔迹（真实抠图效果）
        val canvasW = size.width
        val canvasH = size.height
        // ★ 读取落盘刷新计数，使本绘制块订阅 commitTick，setPixels 落盘后强制本层重绘
        if (commitTick < 0) return@Canvas
        val bw = displayBitmap.width.toFloat()
        val bh = displayBitmap.height.toFloat()
        val fitScale = min(canvasW / bw, canvasH / bh)
        val fitW = bw * fitScale
        val fitH = bh * fitScale
        val centerX = canvasW / 2f
        val centerY = canvasH / 2f
        val imgLeft = centerX - fitW * scale / 2f + offsetX
        val imgTop = centerY - fitH * scale / 2f + offsetY
        val imgW = fitW * scale
        val imgH = fitH * scale
        val rect = android.graphics.RectF(imgLeft, imgTop, imgLeft + imgW, imgTop + imgH)
        val native = drawContext.canvas.nativeCanvas

        // ★ 遮罩层效果（非生成）——按 viewMode 渲染：
        //   0 编辑：底=暗色原图(0.42x)，上=结果图 SRC_OVER，透明区露暗色原图
        //   1 对比：同一张结果图分层次感——底=超暗原图(0.28x)，中=增强主体(饱和+对比)，上=半透明边缘高亮(亮青边)
        //          暗色更暗、主体更鲜明、毛边高亮显形（按住触发）
        //   2 预览：底=查看器背景色(跟随 Prefs.viewerBgColor：棋盘/白/黑/浅灰/绿)，上=结果图
        val darkPaint = Paint().apply {
            isAntiAlias = true; isFilterBitmap = true
            colorFilter = android.graphics.ColorMatrixColorFilter(
                android.graphics.ColorMatrix().apply { setScale(0.42f, 0.42f, 0.42f, 1f) }
            )
        }
        // ★ 对比专用：超暗底（背景压到接近黑，主体从暗色中跳出来）
        val contrastDarkPaint = Paint().apply {
            isAntiAlias = true; isFilterBitmap = true
            colorFilter = android.graphics.ColorMatrixColorFilter(
                android.graphics.ColorMatrix().apply { setScale(0.28f, 0.28f, 0.28f, 1f) }
            )
        }
        // ★ 对比专用：主体增强（饱和度 1.4 + 对比度 1.25，颜色更浓更鲜明）
        val boostPaint = Paint().apply {
            isAntiAlias = true; isFilterBitmap = true
            colorFilter = android.graphics.ColorMatrixColorFilter(
                android.graphics.ColorMatrix().apply {
                    setSaturation(1.4f)
                    postConcat(
                        android.graphics.ColorMatrix(floatArrayOf(
                            1.25f, 0f, 0f, 0f, -32f,
                            0f, 1.25f, 0f, 0f, -32f,
                            0f, 0f, 1.25f, 0f, -32f,
                            0f, 0f, 0f, 1f, 0f
                        ))
                    )
                }
            )
        }
        // ★ 底图绘制（编辑/对比=暗色原图；预览=跟随 Prefs.viewerBgColor 背景色），dst 为目标矩形
        val drawBase: (android.graphics.RectF, Paint) -> Unit = { dst, p ->
            if (viewMode == 2) {
                // ★ 预览背景跟随 Prefs.viewerBgColor 设置（与应用查看器一致）
                when (Prefs.viewerBgColor) {
                    "白色" -> native.drawColor(0xFFFFFFFF.toInt())
                    "黑色" -> native.drawColor(0xFF000000.toInt())
                    "浅灰" -> native.drawColor(0xFFBDBDBD.toInt())
                    "绿色" -> native.drawColor(0xFF4CAF50.toInt())
                    else -> {
                        // 棋盘格
                        val cell = 26.dp.toPx()
                        val cp = Paint().apply { style = Paint.Style.FILL }
                        var gx = dst.left
                        while (gx < dst.right) {
                            var gy = dst.top
                            while (gy < dst.bottom) {
                                val even = (((gx - dst.left) / cell).toInt() + ((gy - dst.top) / cell).toInt()) % 2 == 0
                                cp.color = if (even) 0xFF9E9E9E.toInt() else 0xFFF5F5F5.toInt()
                                native.drawRect(gx, gy, min(gx + cell, dst.right), min(gy + cell, dst.bottom), cp)
                                gy += cell
                            }
                            gx += cell
                        }
                    }
                }
            } else {
                if (safeOriginal.width == bw.toInt() && safeOriginal.height == bh.toInt()) {
                    native.drawBitmap(safeOriginal, null, dst, p)
                } else {
                    native.drawBitmap(safeOriginal, Rect(0, 0, safeOriginal.width, safeOriginal.height), dst, p)
                }
            }
        }
        drawBase(rect, if (viewMode == 1) contrastDarkPaint else darkPaint)
        // 上：结果图（带 alpha 遮罩）SRC_OVER 合成——透明区露出底（暗色原图/背景色），不透明区覆盖
        //   对比模式主体增强（饱和+对比），编辑/预览用原色
        native.drawBitmap(displayBitmap, null, rect,
            if (viewMode == 1) boostPaint else Paint().apply { isAntiAlias = true })
        // ★ 对比模式：叠加边缘高亮（半透明边缘显形，同一张图分层次感）
        if (viewMode == 1) {
            val em = edgeMask
            if (em != null && !em.isRecycled && em.width == bw.toInt() && em.height == bh.toInt()) {
                native.drawBitmap(em, null, rect, Paint().apply { isAntiAlias = true })
            }
        }

        // ★ 即时渲染当前笔迹（无延迟）：擦除=笔迹区透明化（重画暗色原图，呈现"已擦除"效果），恢复=显示原色原图。
        //   用 getFillPath 生成圆头笔宽形状做裁剪，即时呈现真实效果，而非彩色预览线。
        if (currentPath.isNotEmpty()) {
            val sx = { ox: Float -> imgLeft + ox * (imgW / bw) }
            val sy = { oy: Float -> imgTop + oy * (imgH / bh) }
            val screenPath = Path()
            screenPath.moveTo(sx(currentPath[0].x), sy(currentPath[0].y))
            for (i in 1 until currentPath.size) {
                screenPath.lineTo(sx(currentPath[i].x), sy(currentPath[i].y))
            }
            if (currentPath.size == 1) {
                // 单点：直接画一个圆（擦除=暗色原图/恢复=原色原图）
                val sp = Paint().apply {
                    style = Paint.Style.FILL; isAntiAlias = true
                }
                native.save()
                native.clipRect(rect)
                // ★ 擦除：笔迹圆内重画暗色原图（覆盖当前像素，呈现"已透明/擦除"效果）
                native.save()
                native.clipPath(Path().apply {
                    addCircle(sx(currentPath[0].x), sy(currentPath[0].y), brushSize / 2f * (imgW / bw), Path.Direction.CW)
                })
                if (isEraser) {
                    // ★ 擦除=遮罩效果：笔迹圆内重画「暗色原图」→ 透明化视觉呈现（alpha 已由落盘清 0，这里只呈现效果）
                    if (safeOriginal.width == bw.toInt() && safeOriginal.height == bh.toInt()) {
                        native.drawBitmap(safeOriginal, null, rect, darkPaint)
                    } else {
                        native.drawBitmap(safeOriginal, Rect(0, 0, safeOriginal.width, safeOriginal.height), rect, darkPaint)
                    }
                } else {
                    // 恢复=原色原图
                    native.drawBitmap(safeOriginal, null, rect, sp)
                }
                native.restore()
                native.restore()
            } else {
                val strokePaint = Paint().apply {
                    isAntiAlias = true; style = Paint.Style.STROKE
                    strokeCap = Paint.Cap.ROUND; strokeJoin = Paint.Join.ROUND
                    strokeWidth = brushSize * (imgW / bw) // 图像笔宽 → 屏幕笔宽
                }
                val fill = Path()
                strokePaint.getFillPath(screenPath, fill)
                native.save()
                native.clipRect(rect)
                native.clipPath(fill)
                if (isEraser) {
                    // ★ 擦除=遮罩效果：笔迹区重画「暗色原图」→ 透明化视觉呈现
                    if (safeOriginal.width == bw.toInt() && safeOriginal.height == bh.toInt()) {
                        native.drawBitmap(safeOriginal, null, rect, darkPaint)
                    } else {
                        native.drawBitmap(safeOriginal, Rect(0, 0, safeOriginal.width, safeOriginal.height), rect, darkPaint)
                    }
                } else {
                    // 恢复：笔迹区画原图 → 显示原背景色
                    native.drawBitmap(safeOriginal, null, rect, null)
                }
                native.restore()
            }
        }

        // ★ 自定义放大镜：单指按住时显示圆形放大镜（精修辅助），始终跟随手指、平滑移动。
        //   尺寸 120dp + zoom 3x：内容视角更大，便于观察整笔轨迹与周边。
        if (magnifyActive && magnifyOn) {
            val magD = with(density) { 130.dp.toPx() }
            val magR = magD / 2f
            // 手指对应的图像坐标（用于取放大的局部）
            val ix = (magnifierCenter.x - imgLeft) * bw / imgW
            val iy = (magnifierCenter.y - imgTop) * bh / imgH
            // 放大镜圆心：★ 置于手指左上方（不挡绘画视野）——水平偏左 liftX、垂直偏上 liftY，
            //   距离较远；仅 clamp 到画布内保持可见与平滑跟手。
            val liftX = with(density) { 165.dp.toPx() }
            val liftY = with(density) { 200.dp.toPx() }
            val minD = magR.coerceAtMost(min(canvasW, canvasH) / 2f) // 画布过小时放大镜也不溢出
            var mcX = magnifierCenter.x - liftX
            var mcY = magnifierCenter.y - liftY
            if (mcX < minD) mcX = minD else if (mcX > canvasW - minD) mcX = canvasW - minD
            if (mcY < minD) mcY = minD else if (mcY > canvasH - minD) mcY = canvasH - minD
            // ★ 放大倍率 3x：130dp 显示约 43dp 图像区域（视角更大）
            val zoom = 3f
            val srcSize = magD / zoom
            // ★ 浮点窗口中心（笔迹映射用，避免整数 Rect 截断导致笔迹偏移、与实际不一致）
            val srcLeftF = ix - srcSize / 2f
            val srcTopF = iy - srcSize / 2f
            val srcR = Rect(
                srcLeftF.toInt().coerceIn(0, bw.toInt()),
                srcTopF.toInt().coerceIn(0, bh.toInt()),
                (srcLeftF + srcSize).toInt().coerceIn(0, bw.toInt()),
                (srcTopF + srcSize).toInt().coerceIn(0, bh.toInt())
            )
            val dstRect = RectF(mcX - magR, mcY - magR, mcX + magR, mcY + magR)
            val bp = Paint().apply { isAntiAlias = true; isFilterBitmap = true }
            // 圆裁剪 + 深色边缘描边
            val circlePath = Path().apply { addCircle(mcX, mcY, magR, Path.Direction.CW) }
            native.save()
            native.clipPath(circlePath)
            // 放大镜背景：跟随 viewMode——0编辑=暗色原图局部 / 1对比=暗色底+结果图+边缘高亮 / 2预览=Prefs背景色局部
            //   （复用外层 darkPaint——统一 0.42x）
            // ★ 原图坐标的局部源矩形（与 displayBitmap 的 srcR 对齐，映射到原图尺寸）
            val ow = safeOriginal.width.toFloat()
            val oh = safeOriginal.height.toFloat()
            val oSrcR = Rect(
                (srcLeftF * ow / bw).toInt().coerceIn(0, safeOriginal.width),
                (srcTopF * oh / bh).toInt().coerceIn(0, safeOriginal.height),
                ((srcLeftF + srcSize) * ow / bw).toInt().coerceIn(0, safeOriginal.width),
                ((srcTopF + srcSize) * oh / bh).toInt().coerceIn(0, safeOriginal.height)
            )
            if (viewMode == 2) {
                // 预览：跟随 Prefs.viewerBgColor 背景色局部
                when (Prefs.viewerBgColor) {
                    "白色" -> native.drawColor(0xFFFFFFFF.toInt())
                    "黑色" -> native.drawColor(0xFF000000.toInt())
                    "浅灰" -> native.drawColor(0xFFBDBDBD.toInt())
                    "绿色" -> native.drawColor(0xFF4CAF50.toInt())
                    else -> {
                        val cell = 26.dp.toPx()
                        val cp = Paint().apply { style = Paint.Style.FILL }
                        var gx = dstRect.left
                        while (gx < dstRect.right) {
                            var gy = dstRect.top
                            while (gy < dstRect.bottom) {
                                val even = (((gx - dstRect.left) / cell).toInt() + ((gy - dstRect.top) / cell).toInt()) % 2 == 0
                                cp.color = if (even) 0xFF9E9E9E.toInt() else 0xFFF5F5F5.toInt()
                                native.drawRect(gx, gy, min(gx + cell, dstRect.right), min(gy + cell, dstRect.bottom), cp)
                                gy += cell
                            }
                            gx += cell
                        }
                    }
                }
            } else {
                // 编辑/对比：底=暗色原图局部（对比用超暗 contrastDarkPaint）
                if (oSrcR.right > oSrcR.left && oSrcR.bottom > oSrcR.top) {
                    native.drawBitmap(safeOriginal, oSrcR, dstRect,
                        if (viewMode == 1) contrastDarkPaint else darkPaint)
                }
            }
            // 放大显示结果图局部（对比模式主体增强 boostPaint；编辑/预览原色）
            if (srcR.right > srcR.left && srcR.bottom > srcR.top) {
                native.drawBitmap(displayBitmap, srcR, dstRect,
                    if (viewMode == 1) boostPaint else bp)
            }
            // ★ 对比模式：放大镜内叠加边缘高亮局部（同一张图分层次感）
            if (viewMode == 1) {
                val em = edgeMask
                if (em != null && !em.isRecycled && em.width == bw.toInt() && em.height == bh.toInt()) {
                    native.drawBitmap(em, srcR, dstRect, Paint().apply { isAntiAlias = true })
                }
            }
            // ★ 笔迹只占放大镜中心 20%：笔迹叠加裁剪到中心小圆内，外围干净观察周边
            val centerR = magR * 0.2f
            val centerClip = Path().apply { addCircle(mcX, mcY, centerR, Path.Direction.CW) }
            // ★ 叠加当前笔迹到放大镜：只裁剪到中心 20% 区域（centerClip），外围干净观察周边。
            //   擦除=暗色原图（透明化呈现），恢复=原色原图。
            if (currentPath.isNotEmpty()) {
                val magScale = magD / srcSize // = zoom 3x
                val mx = { ox: Float -> dstRect.left + (ox - srcLeftF) * magScale }
                val my = { oy: Float -> dstRect.top + (oy - srcTopF) * magScale }
                val magPath = Path()
                magPath.moveTo(mx(currentPath[0].x), my(currentPath[0].y))
                for (i in 1 until currentPath.size) magPath.lineTo(mx(currentPath[i].x), my(currentPath[i].y))
                val magStroke = Paint().apply {
                    isAntiAlias = true; style = Paint.Style.STROKE
                    strokeCap = Paint.Cap.ROUND; strokeJoin = Paint.Join.ROUND
                    strokeWidth = brushSize * magScale
                }
                if (currentPath.size == 1) {
                    // 单点：直接画圆（★ 效果绘制在整个放大镜圆内实时显示）
                    val cx0 = mx(currentPath[0].x)
                    val cy0 = my(currentPath[0].y)
                    val cr = brushSize / 2f * magScale
                    native.save()
                    native.clipPath(Path().apply { addCircle(cx0, cy0, cr, Path.Direction.CW) })
                    if (isEraser) {
                        // ★ 擦除=遮罩效果：笔迹圆内画「暗色原图局部」（oSrcR），实时跟手
                        if (oSrcR.right > oSrcR.left && oSrcR.bottom > oSrcR.top) {
                            native.drawBitmap(safeOriginal, oSrcR, dstRect, darkPaint)
                        }
                    } else {
                        // ★ 恢复=独立功能：笔迹圆内画「原图局部」（原色 RGBA 完整还原），实时跟手
                        if (oSrcR.right > oSrcR.left && oSrcR.bottom > oSrcR.top) {
                            native.drawBitmap(safeOriginal, oSrcR, dstRect, bp)
                        }
                    }
                    native.restore()
                } else {
                    val magFill = Path()
                    magStroke.getFillPath(magPath, magFill)
                    native.save()
                    // ★ 只裁剪笔迹填充形状（外层已 clip 整个放大镜圆），效果实时跟手显示
                    native.clipPath(magFill)
                    if (isEraser) {
                        // ★ 擦除=遮罩效果：笔迹区画「暗色原图局部」（oSrcR）
                        if (oSrcR.right > oSrcR.left && oSrcR.bottom > oSrcR.top) {
                            native.drawBitmap(safeOriginal, oSrcR, dstRect, darkPaint)
                        }
                    } else {
                        // ★ 恢复=独立功能：笔迹区画「原图局部」（原色 RGBA 完整还原）
                        if (oSrcR.right > oSrcR.left && oSrcR.bottom > oSrcR.top) {
                            native.drawBitmap(safeOriginal, oSrcR, dstRect, bp)
                        }
                    }
                    native.restore()
                }
                // ★ 笔迹轮廓：只在中心 20%（centerClip）内显示半透明彩色（擦除红/恢复绿），
                //   外围保持干净以观察周边；粗细与实际画笔完全一致（brushSize × magScale）。
                native.save()
                native.clipPath(centerClip)
                native.drawPath(magPath, magStroke.apply {
                    color = brushColor.toArgb()
                    alpha = 150 // 半透明：既清晰显示笔迹，又不遮画面
                })
                native.restore()
            }
            native.restore()
            // 放大镜边缘描边
            native.drawCircle(mcX, mcY, magR, Paint().apply {
                style = Paint.Style.STROKE; strokeWidth = 3.dp.toPx()
                color = 0xCC64B5F6.toInt(); isAntiAlias = true
            })
        }
    }
}
