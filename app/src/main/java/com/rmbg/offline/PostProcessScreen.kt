package com.rmbg.offline

import android.graphics.Bitmap
import android.graphics.Paint
import android.graphics.RectF
import android.widget.Toast
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.clickable
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
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.TransformOrigin
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.input.pointer.PointerEventType
import androidx.compose.ui.input.pointer.PointerEventPass
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.drag
import androidx.compose.ui.input.pointer.positionChange
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
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
    initialBrushMode: Boolean = false,  // ★ 从画笔入口进入时自动切画笔模式
    initialSamMode: Boolean = false     // ★ 从智能选区入口进入时自动切选区模式
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
    // ★ 后处理全图调色（色相偏移，整幅图变色，用户需求）
    var postHue by remember { mutableFloatStateOf(-1f) }   // -1=不调色(原色)，0-359=目标色相
    var postSatScale by remember { mutableFloatStateOf(1f) }  // 饱和度微调
    var showPostColor by remember { mutableStateOf(false) }   // 后处理调色面板开关
    var postColorBusy by remember { mutableStateOf(false) }   // ★ 后处理重算中提示（调色/滑块处理反馈）
    // ★ 后处理自定义颜色输入（#RRGGBB）
    var postCustomHex by remember { mutableStateOf("#FF5722") }
    var postCustomHexValid by remember { mutableStateOf(false) }
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
    LaunchedEffect(bgStrength, decontaminate, shrink, hardenTh, postHue, postSatScale) {
        if (postHue >= 0f) postColorBusy = true   // ★ 调色/后处理重算中提示
        val bp = basePixels ?: run { postColorBusy = false; return@LaunchedEffect }
        val d = displayBitmap
        if (d.isRecycled) { postColorBusy = false; return@LaunchedEffect }
        // ★ 硬化阈值：hardenTh 滑块>0 用其阈值；否则用背景强度滑杆阈值；两者都 0 → 不硬化
        val th = when {
            hardenTh > 0 -> hardenTh
            bgStrength > 0 -> bgStrength
            else -> 0
        }
        // 没有启用的处理项 → 回到原始基底（滑块全归零时恢复）
        val postEnabled = decontaminate > 0f || shrink > 0 || th > 0
        val colorEnabled = postHue >= 0f
        val updated = withContext(Dispatchers.IO) {
            try {
                if (!postEnabled && !colorEnabled) {
                    // 还原到基底（未硬化，保持半透明边缘）
                    Bitmap.createBitmap(d.width, d.height, Bitmap.Config.ARGB_8888).also { it.setPixels(bp, 0, d.width, 0, 0, d.width, d.height) }
                } else {
                    // 从基底重建一张，再跑脚本 postLight（绝不叠加在已处理图上）
                    val base = Bitmap.createBitmap(d.width, d.height, Bitmap.Config.ARGB_8888).also { it.setPixels(bp, 0, d.width, 0, 0, d.width, d.height) }
                    var cur = base
                    if (postEnabled) {
                        cur = PostProcessEngine.postLight(
                            cur,
                            shrink = shrink.coerceAtLeast(0),
                            hardenTh = th,
                            dc = decontaminate,
                            bgColor = PostProcessEngine.AUTO_BG
                        )
                    }
                    if (colorEnabled) {
                        // 全图调色：mask=null → 绝对色相旋转（矩阵法，快速且效果明显）
                        cur = rotateHueAll(cur, postHue, postSatScale) ?: cur
                    }
                    cur
                }
            } catch (_: Exception) { null }
        }
        if (updated != null && !updated.isRecycled) {
            val old = displayBitmap
            if (old !== updated && old.isRecycled.not()) { try { old.recycle() } catch (_: Exception) {} }
            displayBitmap = updated
            brushVersion++
        }
        postColorBusy = false   // ★ 重算完成，关掉处理中提示
    }

    // 画笔
    var brushMode by remember { mutableStateOf(initialBrushMode) }
    var isEraser by remember { mutableStateOf(true) }
    var brushSize by remember { mutableStateOf(50f) }
    var strokeCount by remember { mutableStateOf(0) }

    // ---- SAM 智能选区（点击分割；模型：SamEngine）----
    var samMode by remember { mutableStateOf(initialSamMode) }
    var samBusy by remember { mutableStateOf(false) }
    var samStatus by remember { mutableStateOf("点击图片上的物体进行分割") }
    var samMask by remember { mutableStateOf<Bitmap?>(null) }
    var samMaskVersion by remember { mutableIntStateOf(0) }
    var samTap by remember { mutableStateOf<IntArray?>(null) }
    var samHue by remember { mutableFloatStateOf(120f) }
    // ★ 自定义颜色输入（#RRGGBB 十六进制，调色面板内可直接输入任意颜色）
    var samCustomHex by remember { mutableStateOf("#FF5722") }
    var samCustomHexValid by remember { mutableStateOf(false) }
    var samMoving by remember { mutableStateOf(false) }
    var samMoveScreen by remember { mutableStateOf(Offset.Zero) }
    var samMoveImg by remember { mutableStateOf(Offset.Zero) }
    // ★ 预览态：每次扣除/换色/移动先只改预览层（samPreviewBitmap），点「应用」才写回 displayBitmap 出成品
    var samPreviewBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var samPreviewVersion by remember { mutableIntStateOf(0) }
    // ★ 移动实时贴片：物体扣空的底图 + 掩码内物体贴片（拖动时实时显示物体跟随，不用选择框）
    var samCleanBase by remember { mutableStateOf<Bitmap?>(null) }
    var samObjectTile by remember { mutableStateOf<Bitmap?>(null) }
    // ★ 物体贴片裁剪信息：贴片是「物体紧密包围盒」而非整幅图，记录包围盒在原图的左上角与宽高（图像像素坐标）
    var samObjOx by remember { mutableFloatStateOf(0f) }
    var samObjOy by remember { mutableFloatStateOf(0f) }
    var samObjW by remember { mutableFloatStateOf(0f) }
    var samObjH by remember { mutableFloatStateOf(0f) }
    var samObjVersion by remember { mutableIntStateOf(0) }
    // SAM 引擎懒加载（调用方注入或内部按需创建）
    var samEngineRef by remember { mutableStateOf<com.rmbg.offline.ml.SamEngine?>(null) }
    // 是否已自动跑过首次 SAM（进选区自动分割一次，只跑一遍）
    var samAutoRan by remember { mutableStateOf(false) }
    // ★ 顶栏「调色」折叠开关（选区模式的色彩管理面板，折叠在顶部）
    var showSamColorPanel by remember { mutableStateOf(false) }
    // ★ 最近一次选中的选区操作（扣除/换色/移动）——用于操作按钮的选中态高亮
    var samLastAction by remember { mutableStateOf<String?>(null) }
    // ★ 进选区自动 SAM：切到选区模式且未跑过 → 以图片中心自动分割主体（用户要求）
    //   （LaunchedEffect 放在 segmentAtPoint 定义之后，避免本地函数前向引用问题）

    fun obtainSamEngine(): com.rmbg.offline.ml.SamEngine? {
        var eng = samEngineRef
        if (eng == null) {
            // ★ 完整版（normal）内置 SAM：首启先确保从 assets 复制到模型目录，实现开箱即用
            com.rmbg.offline.ml.ModelManager.ensureSamFromAssets(context)
            val enc = com.rmbg.offline.ml.ModelManager.samEncoderFile()
            val dec = com.rmbg.offline.ml.ModelManager.samDecoderFile()
            if (enc != null && dec != null) {
                try {
                    val created = com.rmbg.offline.ml.SamEngine(enc, dec, threads = 4)
                    created.load()
                    samEngineRef = created
                    eng = created
                } catch (e: Exception) {
                    android.util.Log.e("RMBG-SAM", "SAM 引擎加载失败", e)
                    eng = null
                }
            }
        }
        return eng
    }
    /** 执行一次分割，更新掩码/状态（须在 samBusy=true 的协程内调用） */
    fun runSegment(src: Bitmap, px: Int, py: Int) {
        val eng = samEngineRef ?: run { samStatus = "SAM 模型未就绪"; return@runSegment }
        scope.launch {
            samBusy = true
            samStatus = "SAM 分析中…"
            try {
                val mask = withContext(Dispatchers.IO) { eng.segmentAt(src, intArrayOf(px, py)) }
                if (mask != null && !mask.isRecycled) {
                    val old = samMask
                    if (old != null && old !== mask) { try { old.recycle() } catch (_: Exception) {} }
                    samMask = mask
                    samMaskVersion++
                    samTap = intArrayOf(px, py)
                    samStatus = "已选中，可扣除/换色/移动（点其他位置重新分割）"
                } else {
                    samStatus = "该处未识别到物体，请点击物体内部"
                    Toast.makeText(context, "未识别到物体，请点击物体内部再试", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                android.util.Log.e("RMBG-SAM", "segmentAt 失败", e)
                samStatus = "分割失败：${e.message?.take(60) ?: "未知错误"}"
                Toast.makeText(context, "智能选区出错：${e.message?.take(60) ?: "未知错误"}", Toast.LENGTH_LONG).show()
            } finally {
                samBusy = false
            }
        }
    }
    /** 点击分割：跑 SAM 引擎，得到掩码 → 直接渲染高亮。
     *  ★ lite 精简版首次使用智能选区：若模型未就绪，自动从 HF（zuimengqm/cpusam）云下载 encoder/decoder；
     *    normal 完整版已内置（ensureSamFromAssets 复制），模型就绪时直接分析，不触发云下载（实现但默认不启用）。
     */
    fun segmentAtPoint(px: Int, py: Int) {
        if (samBusy) return
        val src = displayBitmap
        if (src.isRecycled) return
        // ★ 已缓存引擎 → 直接分析（最快路径）
        if (samEngineRef != null) { runSegment(src, px, py); return }
        scope.launch {
            samBusy = true
            samStatus = "准备 SAM 模型…"
            try {
                // ① normal 内置复制（存在 assets 则复制；lite 无内置时内部直接返回 false）
                com.rmbg.offline.ml.ModelManager.ensureSamFromAssets(context)
                var enc = com.rmbg.offline.ml.ModelManager.samEncoderFile()
                var dec = com.rmbg.offline.ml.ModelManager.samDecoderFile()
                // ② 仍未就绪（lite 无内置 / 复制后仍缺）→ 自动云下载
                if (enc == null || dec == null) {
                    samStatus = "正在下载 SAM 模型（encoder 约 359MB）…"
                    Toast.makeText(context, "首次使用需下载 SAM 模型（约 375MB）", Toast.LENGTH_LONG).show()
                    val ok = withContext(Dispatchers.IO) {
                        com.rmbg.offline.ml.ModelManager.downloadSamModels(
                            context,
                            listener = object : com.rmbg.offline.ml.ModelManager.ProgressListener {
                                override fun onProgress(downloaded: Long, total: Long, speedBps: Long) {
                                    val pct = if (total > 0) ((downloaded * 100) / total).toInt() else 0
                                    samStatus = "下载 SAM 模型… $pct% (${(downloaded / 1048576)}MB/${(total / 1048576)}MB)"
                                }
                                override fun onMirrorSwitch(idx: Int, name: String) { samStatus = "下载 SAM 模型…（${name}）" }
                                override fun onMirrorError(name: String, error: String) { /* 继续尝试其他镜像 */ }
                                override fun onDone(file: java.io.File) { samStatus = "SAM 模型下载完成…" }
                                override fun onError(e: Exception) { /* 由外层兜底 */ }
                            }
                        )
                    }
                    enc = com.rmbg.offline.ml.ModelManager.samEncoderFile()
                    dec = com.rmbg.offline.ml.ModelManager.samDecoderFile()
                    if (!ok || enc == null || dec == null) {
                        samStatus = "SAM 模型未就绪：下载失败，请检查网络"
                        Toast.makeText(context, "SAM 模型下载失败，请检查网络后重试", Toast.LENGTH_LONG).show()
                        return@launch
                    }
                }
                // ③ 加载引擎
                try {
                    val created = com.rmbg.offline.ml.SamEngine(enc, dec, threads = 4)
                    created.load()
                    samEngineRef = created
                } catch (e: Exception) {
                    android.util.Log.e("RMBG-SAM", "SAM 引擎加载失败", e)
                    samStatus = "SAM 引擎加载失败"
                    Toast.makeText(context, "SAM 引擎加载失败: ${e.message?.take(60) ?: "未知错误"}", Toast.LENGTH_LONG).show()
                    return@launch
                }
                // ④ 分析与渲染
                runSegment(src, px, py)
            } finally {
                samBusy = false
            }
        }
    }
    // ★ 进选区自动 SAM：切到选区模式且未跑过 → 以图片中心自动分割主体（用户要求）
    //   （放 segmentAtPoint 之后，避免本地函数前向引用问题）
    // ★ 冷静期：记录上次自动触发时间，短时间（3 秒）内不再重复自动分析，防多次自动触发
    var samAutoLastMs by remember { mutableLongStateOf(0L) }
    LaunchedEffect(samMode) {
        if (samMode && !samAutoRan) {
            samAutoRan = true
            val now = android.os.SystemClock.elapsedRealtime()
            if (now - samAutoLastMs < 3000L) return@LaunchedEffect   // 冷静期 3s
            samAutoLastMs = now
            val d = displayBitmap
            if (!d.isRecycled) {
                segmentAtPoint(d.width / 2, d.height / 2)
            }
        }
    }
    // ★ 按住触发的画布切换：对比(反色) / 预览(棋盘格结果图)，按住直接切画布渲染，松手恢复
    var viewMode by remember { mutableStateOf(0) }  // 0=编辑 1=对比反色 2=预览结果
    // ★ 画笔撤销栈：栈底是未动笔的原始帧，每次起笔快照当前像素入栈，撤销=弹栈恢复
    val undoStack = remember { mutableStateListOf<Bitmap>() }
    // ★ 重做栈：撤销时把被撤掉的帧压入，重做=弹回；新操作（snapshotForUndo）时清空重做栈
    val redoStack = remember { mutableStateListOf<Bitmap>() }
    fun snapshotForUndo() {
        val cur = displayBitmap
        if (cur.isRecycled) return
        // 新改动产生 → 清空重做栈（redo 分支从此失效）
        redoStack.forEach { try { it.recycle() } catch (_: Exception) {} }
        redoStack.clear()
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
        // 当前显示图压入重做栈
        val cur = displayBitmap
        if (!cur.isRecycled) {
            val copy = Bitmap.createBitmap(cur.width, cur.height, Bitmap.Config.ARGB_8888)
            val px = IntArray(cur.width * cur.height)
            cur.getPixels(px, 0, cur.width, 0, 0, cur.width, cur.height)
            copy.setPixels(px, 0, cur.width, 0, 0, cur.width, cur.height)
            redoStack.add(copy)
        }
        try { undoStack.removeAt(undoStack.size - 1)?.let { prev -> displayBitmap = prev; brushVersion++ } } catch (_: Exception) {}
    }
    fun redoBrush() {
        if (redoStack.isEmpty()) return
        // 当前显示图压入撤销栈（保证撤销/重做可反复切换）
        val cur = displayBitmap
        if (!cur.isRecycled) {
            val copy = Bitmap.createBitmap(cur.width, cur.height, Bitmap.Config.ARGB_8888)
            val px = IntArray(cur.width * cur.height)
            cur.getPixels(px, 0, cur.width, 0, 0, cur.width, cur.height)
            copy.setPixels(px, 0, cur.width, 0, 0, cur.width, cur.height)
            if (undoStack.size >= 30) { try { undoStack.removeAt(0).recycle() } catch (_: Exception) {} }
            undoStack.add(copy)
        }
        try { redoStack.removeAt(redoStack.size - 1)?.let { next -> displayBitmap = next; brushVersion++ } } catch (_: Exception) {}
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
                    // ★ 撤销（撤回，选区/画笔/后处理通用）+ 重做（前进）
                    //   选区预览态撤销 = 丢弃预览；否则弹画笔/后处理撤销栈
                    IconButton(
                        onClick = {
                            when {
                                samPreviewBitmap != null -> {
                                    // 选区预览态：撤销 = 丢弃当前预览，回到应用前
                                    samPreviewBitmap?.let { if (it.isRecycled.not()) { try { it.recycle() } catch (_: Exception) {} } }
                                    samPreviewBitmap = null
                                    samPreviewVersion++
                                    redoStack.forEach { try { it.recycle() } catch (_: Exception) {} }
                                    redoStack.clear()
                                    samStatus = "已撤销预览"
                                }
                                undoStack.isNotEmpty() -> {
                                    // 画笔/后处理撤销
                                    undoBrush()
                                }
                                else -> Toast.makeText(context, "没有可撤销的操作", Toast.LENGTH_SHORT).show()
                            }
                        },
                        enabled = !superResBusy && (samPreviewBitmap != null || undoStack.isNotEmpty())
                    ) { Icon(Icons.Filled.Undo, contentDescription = "撤销", tint = Color(0xFFFFB74D)) }
                    // ★ 重做（前进）
                    IconButton(
                        onClick = {
                            if (redoStack.isNotEmpty()) redoBrush()
                            else Toast.makeText(context, "没有可重做的操作", Toast.LENGTH_SHORT).show()
                        },
                        enabled = !superResBusy && redoStack.isNotEmpty()
                    ) { Icon(Icons.Filled.Redo, contentDescription = "重做", tint = Color(0xFF4FC3F7)) }
                    // ★ 调色（选区模式专属，折叠在顶部）：展开/收起色彩管理面板
                    if (samMode) {
                        TextButton(
                            onClick = { showSamColorPanel = !showSamColorPanel },
                            enabled = samMask != null
                        ) {
                            Icon(
                                if (showSamColorPanel) Icons.Filled.KeyboardArrowUp else Icons.Filled.Palette,
                                contentDescription = "调色",
                                tint = Color(0xFFCE93D8),
                                modifier = Modifier.size(16.dp)
                            )
                            Spacer(Modifier.width(2.dp))
                            Text(if (showSamColorPanel) "收起调色" else "调色", color = Color(0xFFCE93D8))
                        }
                    }
                    // ★ 全图调色（后处理模式专属，折叠在顶栏）
                    if (!samMode && !brushMode) {
                        TextButton(
                            onClick = { showPostColor = !showPostColor }
                        ) {
                            Icon(
                                if (showPostColor) Icons.Filled.KeyboardArrowUp else Icons.Filled.Palette,
                                contentDescription = "全图调色",
                                tint = Color(0xFFCE93D8),
                                modifier = Modifier.size(16.dp)
                            )
                            Spacer(Modifier.width(2.dp))
                            Text(if (showPostColor) "收起" else "调色", color = Color(0xFFCE93D8))
                        }
                    }
                    TextButton(
                        onClick = {
                            val cur = displayBitmap
                            if (cur.isRecycled) return@TextButton
                            // ★ 若存在未落地的选区预览（扣除/换色/移动），先写回 displayBitmap 再应用
                            val pv = samPreviewBitmap
                            val finalBmp = if (pv != null && !pv.isRecycled) {
                                samPreviewBitmap = null
                                samPreviewVersion++
                                pv
                            } else cur
                            scope.launch {
                                bgStrength = 0; decontaminate = 0f; shrink = 0; hardenTh = 0
                                Toast.makeText(context, "已应用", Toast.LENGTH_SHORT).show()
                                onApply(finalBmp)
                            }
                        },
                        // ★ 始终可用：只要有任何编辑（后处理/画笔/选区操作/选区预览）均可应用
                        enabled = !superResBusy && !samBusy && (
                            bgStrength > 0 || decontaminate > 0f || shrink != 0 || hardenTh > 0 ||
                            strokeCount > 0 || samPreviewBitmap != null || samMask != null
                        )
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
                // 功能开关行：后处理/画笔/选区 切换 + 对比/预览 视图切换（按住）
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    // ★ 三个模式选中态：已选=主色填充+白字+描边，未选=暗底灰字，一眼可辨
                    val chipSel = @Composable { sel: Boolean, label: String ->
                        FilterChip(
                            selected = sel,
                            onClick = {},
                            enabled = false, // 仅视觉，点击在下方包装处理
                            label = { Text(label, fontSize = 12.sp, color = if (sel) Color.White else Color(0xFF90A4AE)) },
                            colors = FilterChipDefaults.filterChipColors(
                                containerColor = if (sel) Color(0xFF64B5F6) else Color(0xFF263238),
                                labelColor = if (sel) Color.White else Color(0xFF90A4AE),
                                selectedContainerColor = Color(0xFF64B5F6),
                                selectedLabelColor = Color.White
                            ),
                            border = BorderStroke(if (sel) 2.dp else 1.dp, if (sel) Color.White else Color(0x44FFFFFF))
                        )
                    }
                    // 后处理
                    Box {
                        chipSel(!brushMode && !samMode, "后处理")
                        // 点击层（透明，拦截点击）；选区有未完成编辑时需先取消/应用才能切走
                        Box(Modifier.matchParentSize().clickable {
                            if (samBusy) { Toast.makeText(context, "选区分析中，请稍候", Toast.LENGTH_SHORT).show(); return@clickable }
                            if (samMoving || samPreviewBitmap != null) {
                                Toast.makeText(context, "请先「取消/放下」当前选区操作再切换", Toast.LENGTH_SHORT).show(); return@clickable
                            }
                            brushMode = false; samMode = false
                        })
                    }
                    // 画笔
                    Box {
                        chipSel(brushMode, "画笔")
                        Box(Modifier.matchParentSize().clickable {
                            if (samBusy) { Toast.makeText(context, "选区分析中，请稍候", Toast.LENGTH_SHORT).show(); return@clickable }
                            if (samMoving || samPreviewBitmap != null) {
                                Toast.makeText(context, "请先「取消/放下」当前选区操作再切换", Toast.LENGTH_SHORT).show(); return@clickable
                            }
                            brushMode = true; samMode = false
                        })
                    }
                    // 选区（切回时保留已分割的 mask 等状态，不清理）
                    Box {
                        chipSel(samMode, "选区")
                        Box(Modifier.matchParentSize().clickable {
                            brushMode = false; samMode = true
                        })
                    }
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
                // ★ 三个模式（后处理/画笔/选区）共用同一操作区高度：统一底栏高度不跳动，内容超高时滚动
                // ★ 选区预览态已改为单区域（三按钮⇄取消/应用 互斥显示），底栏紧凑
                Box(Modifier.height(56.dp)) {
                    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState())) {
                if (samMode) {
                    // ================= 选区模式（SAM 点击分割）=================
                    if (samMoving) {
                        // 移动中：取消 + 放下（圆角卡片，与操作区风格统一）
                        Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.fillMaxWidth()) {
                            Box(
                                modifier = Modifier
                                    .weight(1f)
                                    .height(50.dp)
                                    .clip(RoundedCornerShape(12.dp))
                                    .background(Color(0xFF263238))
                                    .border(1.dp, Color(0x66FF8A80), RoundedCornerShape(12.dp))
                                    .clickable {
                                        samMoving = false
                                        samMoveScreen = Offset.Zero
                                        samMoveImg = Offset.Zero
                                        samCleanBase?.recycle(); samCleanBase = null
                                        samObjectTile?.recycle(); samObjectTile = null
                                        samObjOx = 0f; samObjOy = 0f; samObjW = 0f; samObjH = 0f
                                    },
                                contentAlignment = Alignment.Center
                            ) {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Icon(Icons.Filled.Close, contentDescription = null, modifier = Modifier.size(16.dp), tint = Color(0xFFFF8A80))
                                    Spacer(Modifier.width(6.dp))
                                    Text("取消", fontSize = 13.sp, fontWeight = FontWeight.SemiBold, color = Color(0xFFFF8A80))
                                }
                            }
                            Box(
                                modifier = Modifier
                                    .weight(1f)
                                    .height(50.dp)
                                    .clip(RoundedCornerShape(12.dp))
                                    .background(Color(0xFF4CAF50))
                                    .border(1.dp, Color.White, RoundedCornerShape(12.dp))
                                    .clickable {
                                        samMoving = false
                                        val mm = samMask
                                        val dx = samMoveImg.x
                                        val dy = samMoveImg.y
                                        if (mm != null && (dx != 0f || dy != 0f)) {
                                            scope.launch {
                                                // ★ 预览态：用移动贴片合成「原位透明 + 新位置物体」，写进预览层，点「应用」才出成品
                                                val base = samPreviewBitmap ?: displayBitmap
                                                val out = withContext(Dispatchers.IO) {
                                                    val w0 = base.width; val h0 = base.height
                                                    val m0 = if (mm.width == w0 && mm.height == h0) mm else null
                                                    val tile = samObjectTile
                                                    if (m0 != null && tile != null && !tile.isRecycled && samObjW > 0f && samObjH > 0f) {
                                                        // 1) 原掩码区域清成透明
                                                        val bmp = base.copy(android.graphics.Bitmap.Config.ARGB_8888, true) ?: return@withContext null
                                                        val bp = IntArray(w0 * h0)
                                                        val mp = IntArray(w0 * h0)
                                                        bmp.getPixels(bp, 0, w0, 0, 0, w0, h0)
                                                        m0.getPixels(mp, 0, w0, 0, 0, w0, h0)
                                                        var changed = false
                                                        for (i in bp.indices) {
                                                            if (((mp[i] ushr 24) and 0xFF) > 127) { bp[i] = bp[i] and 0x00FFFFFF; changed = true }
                                                        }
                                                        bmp.setPixels(bp, 0, w0, 0, 0, w0, h0)
                                                        // 2) 在新位置绘制物体贴片
                                                        val cv = android.graphics.Canvas(bmp)
                                                        cv.drawBitmap(
                                                            tile,
                                                            (samObjOx + dx),
                                                            (samObjOy + dy),
                                                            Paint().apply { isFilterBitmap = false }
                                                        )
                                                        if (changed) bmp else null
                                                    } else {
                                                        applyMove(base, softOriginBitmap ?: displayBitmap, mm, dx, dy)
                                                    }
                                                }
                                                if (out != null) {
                                                    samPreviewBitmap?.let { if (it !== out && it.isRecycled.not()) { try { it.recycle() } catch (_: Exception) {} } }
                                                    samPreviewBitmap = out
                                                    samPreviewVersion++
                                                    samStatus = "已预览移动 · 点「应用」完成，点「取消」还原"
                                                }
                                            }
                                        }
                                        samMoveScreen = Offset.Zero
                                        samMoveImg = Offset.Zero
                                        samCleanBase?.recycle(); samCleanBase = null
                                        samObjectTile?.recycle(); samObjectTile = null
                                        samObjOx = 0f; samObjOy = 0f; samObjW = 0f; samObjH = 0f
                                    },
                                contentAlignment = Alignment.Center
                            ) {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Icon(Icons.Filled.Check, contentDescription = null, modifier = Modifier.size(16.dp), tint = Color.White)
                                    Spacer(Modifier.width(6.dp))
                                    Text("放下", fontSize = 13.sp, fontWeight = FontWeight.SemiBold, color = Color.White)
                                }
                            }
                        }
                    } else if (samPreviewBitmap == null) {
                        // 操作按钮：扣除 / 换色 / 移动 —— 竖向图标+文字的圆角卡片（掩码存在才可用；色板在顶部「调色」折叠面板）
                        // ★ 打开预览态（samPreviewBitmap != null）时隐藏三按钮，让位给「取消/应用」——共用一个操作区，不抬高底栏
                        // ★ 选中态：当前选中卡片用主色填充+白字+白色描边，未选=暗底灰字+半透明白描边，一眼可辨当前模式
                        Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.fillMaxWidth()) {
                            // ★ 扣除
                            val selErase = samLastAction == "erase" || samMoving
                            Box(
                                modifier = Modifier
                                    .weight(1f)
                                    .height(46.dp)
                                    .clip(RoundedCornerShape(12.dp))
                                    .background(if (selErase) Color(0xFFFF5252).copy(alpha = 0.88f) else Color(0xFF263238))
                                    .border(1.dp, if (selErase) Color.White else Color(0x44FFFFFF), RoundedCornerShape(12.dp))
                                    .clickable(enabled = samMask != null && !samBusy) {
                                        samLastAction = "erase"
                                        val mm = samMask ?: return@clickable
                                        scope.launch {
                                            // ★ 预览态：先写进预览层（不碰原图），点「应用」才出成品
                                            val base = samPreviewBitmap ?: displayBitmap
                                            val out = withContext(Dispatchers.IO) { eraseByMask(base, mm) }
                                            if (out != null) {
                                                samPreviewBitmap?.let { if (it !== out && it.isRecycled.not()) { try { it.recycle() } catch (_: Exception) {} } }
                                                samPreviewBitmap = out
                                                samPreviewVersion++
                                                samStatus = "已预览扣除效果 · 点「应用」完成，点「取消」还原"
                                            }
                                        }
                                    },
                                contentAlignment = Alignment.Center
                            ) {
                                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                    Icon(
                                        Icons.Filled.Delete,
                                        contentDescription = null,
                                        modifier = Modifier.size(18.dp),
                                        tint = if (selErase) Color.White else Color(0xFFFF5252)
                                    )
                                    Spacer(Modifier.height(2.dp))
                                    Text("扣除", fontSize = 11.sp, fontWeight = FontWeight.SemiBold, color = if (selErase) Color.White else Color(0xFFFF5252))
                                }
                            }
                            // ★ 换色
                            val selColor = samLastAction == "color" && !samMoving
                            Box(
                                modifier = Modifier
                                    .weight(1f)
                                    .height(46.dp)
                                    .clip(RoundedCornerShape(12.dp))
                                    .background(if (selColor) Color(0xFFCE93D8).copy(alpha = 0.88f) else Color(0xFF263238))
                                    .border(1.dp, if (selColor) Color.White else Color(0x44FFFFFF), RoundedCornerShape(12.dp))
                                    .clickable(enabled = samMask != null && !samBusy) {
                                        samLastAction = "color"
                                        // ★ 直接用原图计算新色并写入预览层（自动回原再上色，不叠加旧预览）
                                        val mm = samMask
                                        if (mm == null) return@clickable
                                        scope.launch {
                                            // ★ 预览态：直接基于原图重算 → 预览层，点「应用」才出成品
                                            val base = displayBitmap
                                            val out = if (samHue < 0f) {
                                                withContext(Dispatchers.IO) { restoreColorByMask(base, softOriginBitmap ?: displayBitmap, mm) }
                                            } else {
                                                withContext(Dispatchers.IO) { recolorFromOrigin(base, softOriginBitmap ?: displayBitmap, mm, samHue, 1f) }
                                            }
                                            if (out != null) {
                                                samPreviewBitmap?.let { if (it !== out && it.isRecycled.not()) { try { it.recycle() } catch (_: Exception) {} } }
                                                samPreviewBitmap = out
                                                samPreviewVersion++
                                                samStatus = if (samHue < 0f) "已预览恢复原色 · 点「应用」完成" else "已预览换色 · 点「应用」完成"
                                            }
                                        }
                                    },
                                contentAlignment = Alignment.Center
                            ) {
                                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                    Icon(
                                        Icons.Filled.Palette,
                                        contentDescription = null,
                                        modifier = Modifier.size(18.dp),
                                        tint = if (selColor) Color.White else Color(0xFFCE93D8)
                                    )
                                    Spacer(Modifier.height(2.dp))
                                    Text(if (samHue < 0f) "恢复原色" else "换色", fontSize = 11.sp, fontWeight = FontWeight.SemiBold, color = if (selColor) Color.White else Color(0xFFCE93D8))
                                }
                            }
                            // ★ 移动
                            val selMove = samMoving
                            Box(
                                modifier = Modifier
                                    .weight(1f)
                                    .height(46.dp)
                                    .clip(RoundedCornerShape(12.dp))
                                    .background(if (selMove) Color(0xFF64B5F6) else Color(0xFF263238))
                                    .border(1.dp, if (selMove) Color.White else Color(0x44FFFFFF), RoundedCornerShape(12.dp))
                                    .clickable(enabled = samMask != null && !samBusy) {
                                        samLastAction = "move"
                                        if (samMask != null) {
                                            samMoving = true
                                            samMoveScreen = Offset.Zero
                                            samMoveImg = Offset.Zero
                                            // 生成移动实时贴片：底图（物体扣空露棋盘）+ 贴片（掩码内物体裁剪成紧密包围盒）
                                            val m0 = samMask
                                            val src = samPreviewBitmap ?: displayBitmap
                                            if (m0 != null && !m0.isRecycled) {
                                                val w0 = src.width; val h0 = src.height
                                                val sameSize = m0.width == w0 && m0.height == h0
                                                val px = IntArray(w0 * h0)
                                                val mp = IntArray(w0 * h0)
                                                src.getPixels(px, 0, w0, 0, 0, w0, h0)
                                                if (sameSize) m0.getPixels(mp, 0, w0, 0, 0, w0, h0)
                                                fun inMask(i: Int) = sameSize && ((mp[i] ushr 24) and 0xFF > 127)
                                                // 背景层：整幅图，掩码内清空成透明（露背景/棋盘）
                                                val bpx = IntArray(w0 * h0)
                                                for (i in px.indices) {
                                                    bpx[i] = if (inMask(i)) 0 else px[i]
                                                }
                                                samCleanBase?.recycle(); samCleanBase = null
                                                samCleanBase = Bitmap.createBitmap(w0, h0, Bitmap.Config.ARGB_8888)?.apply { setPixels(bpx, 0, w0, 0, 0, w0, h0) }
                                                // ★ 物体层：裁剪成「紧密包围盒」独立小图（仅掩码内像素，透明背景）——真正分成两个图层，可平移独立物体
                                                if (sameSize) {
                                                    var minX = w0; var minY = h0; var maxX = -1; var maxY = -1
                                                    for (y in 0 until h0) {
                                                        var row = y * w0
                                                        for (x in 0 until w0) {
                                                            if (((mp[row + x] ushr 24) and 0xFF) > 127) {
                                                                if (x < minX) minX = x
                                                                if (x > maxX) maxX = x
                                                                if (y < minY) minY = y
                                                                if (y > maxY) maxY = y
                                                            }
                                                        }
                                                        row += 1
                                                    }
                                                    if (maxX >= minX && maxY >= minY) {
                                                        val tw = maxX - minX + 1
                                                        val th = maxY - minY + 1
                                                        val tpx = IntArray(tw * th)
                                                        for (y in 0 until th) {
                                                            val sy = minY + y
                                                            for (x in 0 until tw) {
                                                                val sx = minX + x
                                                                val mi = sy * w0 + sx
                                                                tpx[y * tw + x] = if (((mp[mi] ushr 24) and 0xFF) > 127) (px[mi] or 0xFF000000.toInt()) else 0
                                                            }
                                                        }
                                                        samObjectTile?.recycle(); samObjectTile = null
                                                        samObjectTile = Bitmap.createBitmap(tw, th, Bitmap.Config.ARGB_8888)?.apply { setPixels(tpx, 0, tw, 0, 0, tw, th) }
                                                        samObjOx = minX.toFloat(); samObjOy = minY.toFloat()
                                                        samObjW = tw.toFloat(); samObjH = th.toFloat()
                                                        samObjVersion++
                                                        // 记录物体在原图的左上角（图像坐标），供放下/绘制定位
                                                    } else {
                                                        samObjectTile?.recycle(); samObjectTile = null
                                                        samObjW = 0f; samObjH = 0f
                                                        samObjVersion++
                                                    }
                                                } else {
                                                    samObjectTile?.recycle(); samObjectTile = null
                                                    samObjW = 0f; samObjH = 0f
                                                }
                                            }
                                            samStatus = "拖动移动物体，点「放下」完成"
                                        }
                                    },
                                contentAlignment = Alignment.Center
                            ) {
                                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                    Icon(
                                        Icons.Filled.OpenWith,
                                        contentDescription = null,
                                        modifier = Modifier.size(18.dp),
                                        tint = if (selMove) Color.White else Color(0xFF64B5F6)
                                    )
                                    Spacer(Modifier.height(2.dp))
                                    Text("移动", fontSize = 11.sp, fontWeight = FontWeight.SemiBold, color = if (selMove) Color.White else Color(0xFF64B5F6))
                                }
                            }
                        }
                    }
                    // ★ 预览操作区：有预览（扣除/换色/移动预览态）→ 显示「取消 / 应用」（圆角卡片风格）
                        if (samPreviewBitmap != null) {
                            Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.fillMaxWidth()) {
                                Box(
                                    modifier = Modifier
                                        .weight(1f)
                                        .height(44.dp)
                                        .clip(RoundedCornerShape(12.dp))
                                        .background(Color(0xFF263238))
                                        .border(1.dp, Color(0x66FF8A80), RoundedCornerShape(12.dp))
                                        .clickable {
                                            // 取消：丢弃预览层，回到编辑图
                                            samPreviewBitmap?.let { if (it.isRecycled.not()) { try { it.recycle() } catch (_: Exception) {} } }
                                            samPreviewBitmap = null
                                            samPreviewVersion++
                                            samStatus = "已取消预览"
                                        },
                                    contentAlignment = Alignment.Center
                                ) {
                                    Row(verticalAlignment = Alignment.CenterVertically) {
                                        Icon(Icons.Filled.Close, contentDescription = null, modifier = Modifier.size(16.dp), tint = Color(0xFFFF8A80))
                                        Spacer(Modifier.width(6.dp))
                                        Text("取消", fontSize = 13.sp, fontWeight = FontWeight.SemiBold, color = Color(0xFFFF8A80))
                                    }
                                }
                                Box(
                                    modifier = Modifier
                                        .weight(1f)
                                        .height(44.dp)
                                        .clip(RoundedCornerShape(12.dp))
                                        .background(Color(0xFF64B5F6))
                                        .border(1.dp, Color.White, RoundedCornerShape(12.dp))
                                        .clickable {
                                            // ★ 应用：预览层写回 displayBitmap 出成品，清空预览层
                                            val pv = samPreviewBitmap ?: return@clickable
                                            val old = displayBitmap
                                            if (old !== pv && old.isRecycled.not()) { try { old.recycle() } catch (_: Exception) {} }
                                            displayBitmap = pv
                                            brushVersion++
                                            samPreviewBitmap = null
                                            samPreviewVersion++
                                            // ★ 选区持久化：应用后【不清除】samMask/samTap，保持选中态——不退出就一直选中，
                                            //   后续可继续调色/扣除/移动（外溢调色的根因之一就是应用后选区丢失→再操作时掩码失效）。
                                            //   若显示图尺寸已变（掩码尺寸不匹配），后续各操作内部会按尺寸适配，绝不误伤未选中区。
                                            samStatus = "已应用 · 选区保持选中，可继续扣除/换色/移动"
                                        },
                                    contentAlignment = Alignment.Center
                                ) {
                                    Row(verticalAlignment = Alignment.CenterVertically) {
                                        Icon(Icons.Filled.Check, contentDescription = null, modifier = Modifier.size(16.dp), tint = Color.White)
                                        Spacer(Modifier.width(6.dp))
                                        Text("应用", fontSize = 13.sp, fontWeight = FontWeight.SemiBold, color = Color.White)
                                    }
                                }
                            }
                        }
                    // ★ 选区状态提示统一由底栏固定行显示（见 bottomBar 内 1144 行附近），这里不再重复输出，避免提示词重复
                } else if (!brushMode) {
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
                                        val up = try {
                                            withContext(Dispatchers.IO) {
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
                                        } finally {
                                            // ★ 运行监测：无论成功/失败/取消，busy 一定复位 → 加载动画自动停止
                                            superResBusy = false
                                        }
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
                        // ★ 参数调整开关：点按切换「向上弹出浮窗」（滑块在画布顶部浮层，不占底栏空间）
                        TextButton(
                            onClick = { showPostControls = !showPostControls },
                            modifier = Modifier.fillMaxWidth().height(30.dp),
                            contentPadding = PaddingValues(horizontal = 8.dp, vertical = 0.dp)
                        ) {
                            Icon(
                                if (showPostControls) Icons.Filled.KeyboardArrowUp else Icons.Filled.Settings,
                                contentDescription = null, modifier = Modifier.size(16.dp), tint = Color(0xFF90A4AE)
                            )
                            Spacer(Modifier.width(4.dp))
                            Text(if (showPostControls) "收起参数" else "参数调整", fontSize = 12.sp, color = Color(0xFF90A4AE))
                        }
                    }
                } else {
                    // ---- 画笔模式（紧凑单行）：擦除/恢复 + 笔刷（撤销/重做在顶栏，这里不再重复放撤销）----
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        OutlinedButton(
                            onClick = { isEraser = !isEraser },
                            modifier = Modifier.height(32.dp),
                            colors = if (isEraser)
                                ButtonDefaults.outlinedButtonColors(contentColor = Color(0xFFFF5252), containerColor = Color(0xFF263238))
                            else ButtonDefaults.outlinedButtonColors(contentColor = Color(0xFF4CAF50), containerColor = Color(0xFF263238))
                        ) { Text(if (isEraser) "擦除" else "恢复", fontSize = 11.sp) }
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
                } // Column(滚动)
                } // Box(固定高度操作区)
                // ★ 底栏固定行（不随操作区滚动）：操作提示 + 右下角「还原」
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        if (samMode) samStatus
                        else if (brushMode) "单指画笔 · 双指缩放平移 · 长按放大镜"
                        else "双指缩放/平移预览",
                        color = Color(0xFF90A4AE), fontSize = 11.sp,
                        modifier = Modifier.weight(1f).padding(end = 6.dp)
                    )
                    // ★ 还原（底栏右下角，固定可见）：撤回所有已做编辑（画笔/后处理/选区扣除换色移动），回到打开时原始像素。
                    //   保留当前模式与选区分割状态（mask），只撤编辑效果。
                    Box(
                        modifier = Modifier
                            .clip(RoundedCornerShape(8.dp))
                            .background(Color(0xFF37474F))
                            .border(1.dp, Color(0xFF90A4AE), RoundedCornerShape(8.dp))
                            .clickable {
                                val bp = basePixels
                                if (bp == null) return@clickable
                                val w = displayBitmap.width; val h = displayBitmap.height
                                val restore = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888).also {
                                    it.setPixels(bp, 0, w, 0, 0, w, h)
                                }
                                val old = displayBitmap
                                if (old !== restore && old.isRecycled.not()) { try { old.recycle() } catch (_: Exception) {} }
                                displayBitmap = restore
                                brushVersion++
                                bgStrength = 0; decontaminate = 0f; shrink = 0; hardenTh = 0; strokeCount = 0
                                postHue = -1f; postSatScale = 1f
                                undoStack.clear(); redoStack.forEach { try { it.recycle() } catch (_: Exception) {} }; redoStack.clear()
                                // —— 撤回选区编辑效果（扣除/换色/移动），但保留选区分割(mask)与模式 ——
                                samPreviewBitmap?.let { if (it.isRecycled.not()) { try { it.recycle() } catch (_: Exception) {} } }
                                samPreviewBitmap = null
                                samPreviewVersion++
                                samMoving = false
                                samMoveScreen = Offset.Zero; samMoveImg = Offset.Zero
                                samCleanBase?.recycle(); samCleanBase = null
                                samObjectTile?.recycle(); samObjectTile = null
                                samObjOx = 0f; samObjOy = 0f; samObjW = 0f; samObjH = 0f; samObjVersion++
                                samLastAction = null
                                samHue = 120f                    // 换色恢复原色彩
                                scale = 1f; offset = Offset.Zero
                                samStatus = "已还原 · 扣除/换色/移动已撤回（选区保留）"
                            },
                        contentAlignment = Alignment.Center
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp)) {
                            Icon(Icons.Filled.Restore, contentDescription = null, modifier = Modifier.size(14.dp), tint = Color(0xFFFFB74D))
                            Spacer(Modifier.width(4.dp))
                            Text("还原", fontSize = 12.sp, fontWeight = FontWeight.SemiBold, color = Color(0xFFFFB74D))
                        }
                    }
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
                    .padding(top = 10.dp)   // ★ 图像整体轻微下移：避免贴顶/被顶栏浮层遮挡，更平衡
                    .then(
                        when {
                            // 选区模式：点击分割 / 移动拖动 / 双指缩放平移
                            samMode -> Modifier.pointerInput(samMode, samMoving) {
                                // ★★★★★ 与画笔(EditBrushLayer)一致的手势循环：awaitPointerEventScope 统一处理
                                // 单指拖动→移动物体；单指点击→分割；双指→缩放+平移（半笔同款锚点，放大跟手不跑偏）。
                                awaitPointerEventScope {
                                    var tapDownPos: Offset? = null
                                    while (true) {
                                        val event = awaitPointerEvent()
                                        val pointers = event.changes
                                        val cw = size.width.toFloat()
                                        val ch = size.height.toFloat()
                                        val bw = displayBitmap.width.toFloat()
                                        val bh = displayBitmap.height.toFloat()
                                        val fitS = min(cw / bw, ch / bh)
                                        when {
                                            // ---- 双指及以上：缩放 + 平移（画笔同款锚点公式）----
                                            pointers.size >= 2 -> {
                                                tapDownPos = null
                                                if (event.type == PointerEventType.Move) {
                                                    val press = pointers.filter { it.pressed }
                                                    if (press.size >= 2) {
                                                        val p0 = press[0]
                                                        val p1 = press[1]
                                                        val dist = (p0.position - p1.position).getDistance()
                                                        val prevDist = (p0.previousPosition - p1.previousPosition).getDistance()
                                                        if (prevDist > 0f) {
                                                            val newScale = (scale * dist / prevDist).coerceIn(1f, 8f)
                                                            val centerX = (p0.position.x + p1.position.x) / 2f
                                                            val centerY = (p0.position.y + p1.position.y) / 2f
                                                            val imgW_old = bw * fitS * scale
                                                            val imgH_old = bh * fitS * scale
                                                            val effL = cw / 2f - imgW_old / 2f + offset.x
                                                            val effT = ch / 2f - imgH_old / 2f + offset.y
                                                            val imgW_new = bw * fitS * newScale
                                                            val imgH_new = bh * fitS * newScale
                                                            offset = Offset(
                                                                centerX - (centerX - effL) * (newScale / scale) - cw / 2f + imgW_new / 2f,
                                                                centerY - (centerY - effT) * (newScale / scale) - ch / 2f + imgH_new / 2f
                                                            )
                                                            scale = newScale
                                                        }
                                                        val prevCx = (p0.previousPosition.x + p1.previousPosition.x) / 2f
                                                        val prevCy = (p0.previousPosition.y + p1.previousPosition.y) / 2f
                                                        val cx = (p0.position.x + p1.position.x) / 2f
                                                        val cy = (p0.position.y + p1.position.y) / 2f
                                                        offset = offset + Offset(cx - prevCx, cy - prevCy)
                                                    }
                                                }
                                                pointers.forEach { it.consume() }
                                            }
                                            // ---- 单指 ----
                                            pointers.size == 1 -> {
                                                val p = pointers[0]
                                                if (samMoving) {
                                                    // ★ 移动物体：单指拖动实时累积 samMoveImg
                                                    if (event.type == PointerEventType.Move) {
                                                        samMoveImg = Offset(
                                                            samMoveImg.x + (p.position.x - p.previousPosition.x) / (fitS * scale),
                                                            samMoveImg.y + (p.position.y - p.previousPosition.y) / (fitS * scale)
                                                        )
                                                        p.consume()
                                                    }
                                                } else {
                                                    // 非移动：按事件类型判断 按下记起点 / 拖动清点击 / 抬起触发分割
                                                    when (event.type) {
                                                        PointerEventType.Press -> tapDownPos = p.position
                                                        PointerEventType.Move -> {
                                                            // 手指动了但未超过阈值仍算点击；超过则视为拖动浏览画布(放弃点击)
                                                            val start = tapDownPos
                                                            if (start != null && (p.position - start).getDistance() > 24f) {
                                                                tapDownPos = null
                                                            }
                                                            p.consume()
                                                        }
                                                        PointerEventType.Release -> {
                                                        val start = tapDownPos
                                                        tapDownPos = null
                                                        // ★ 浮层打开时（调色/参数浮窗）不允许通过画布触发分割，防误触穿透
                                                        val overlayOpen = showSamColorPanel || showPostControls || showPostColor
                                                        if (!overlayOpen && start != null && (p.position - start).getDistance() < 24f) {
                                                            // 单击 → 计算图片坐标 → 分割
                                                            val fitW = bw * fitS
                                                            val fitH = bh * fitS
                                                            val baseLeft = (cw - fitW) / 2f
                                                            val baseTop = (ch - fitH) / 2f
                                                            val imgX = ((p.position.x - baseLeft - offset.x) / (fitS * scale)).roundToInt()
                                                            val imgY = ((p.position.y - baseTop - offset.y) / (fitS * scale)).roundToInt()
                                                            if (imgX in 0 until bw.toInt() && imgY in 0 until bh.toInt()) {
                                                                segmentAtPoint(imgX, imgY)
                                                            }
                                                        }
                                                        p.consume()
                                                    }
                                                    }
                                                }
                                            }
                                            // ---- 无手指：复位 ----
                                            else -> tapDownPos = null
                                        }
                                    }
                                }
                            }
                            // 画笔模式：交给 EditBrushLayer
                            brushMode -> Modifier
                            // 后处理模式：双指缩放/平移（★ 调色/参数浮层打开时不响应，防透传到页面）
                            else -> Modifier.pointerInput(showPostColor, showPostControls, Unit) {
                                detectTransformGestures { _, pan, zoom, _ ->
                                    // 浮层打开时忽略手势，避免点击穿透到后处理页
                                    if (showPostColor || showPostControls) return@detectTransformGestures
                                    val prevScale = scale
                                    val newScale = (prevScale * zoom).coerceIn(1f, 8f)
                                    offset = offset + pan
                                    scale = newScale
                                }
                            }
                        }
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
                        // ★ 背景层：预览(viewMode==2) 或 移动中(samMoving) 或 有预览层时，透明区遵循用户背景设置（Prefs.viewerBgColor）
                        if (viewMode == 2 || samMoving || samPreviewBitmap != null) {
                            val bgName = com.rmbg.offline.Prefs.viewerBgColor
                            val bgSolid: Color? = when (bgName) {
                                "白色" -> Color.White
                                "黑色" -> Color.Black
                                "浅灰" -> Color(0xFFBDBDBD)
                                "绿色" -> Color(0xFF4CAF50)
                                else -> null // 棋盘
                            }
                            if (bgSolid != null) {
                                Canvas(Modifier.fillMaxSize()) { drawRect(bgSolid, Offset.Zero, Size(size.width, size.height)) }
                            } else {
                                Canvas(Modifier.fillMaxSize()) {
                                    val cell = 12.dp.toPx()
                                    var x = 0f
                                    while (x < size.width) {
                                        var y = 0f
                                        while (y < size.height) {
                                            val c = (((x / cell).toInt() + (y / cell).toInt()) % 2 == 0)
                                            drawRect(if (c) Color(0xFF3A4048) else Color(0xFF2A2F36), Offset(x, y), Size(cell, cell))
                                            y += cell
                                        }
                                        x += cell
                                    }
                                }
                            }
                        }
                        // ★ 主图：按 viewMode 切换显示
                        //   0 编辑：移动中→扣空底图（物体抬起）；预览态→预览层；否则→当前编辑图
                        //   1 对比：显示「修改前」的原图（用户需求：对比=看修改前的图片）
                        //   2 预览：显示「结果图」（预览层或当前编辑图），透明区露棋盘/背景
                        val srcBmp: Bitmap? = when {
                            viewMode == 1 -> (originBmp ?: basePixels?.let { bp ->
                                Bitmap.createBitmap(displayBitmap.width, displayBitmap.height, Bitmap.Config.ARGB_8888).also {
                                    it.setPixels(bp, 0, displayBitmap.width, 0, 0, displayBitmap.width, displayBitmap.height)
                                }
                            }) ?: displayBitmap
                            samMoving && samCleanBase != null -> samCleanBase
                            else -> samPreviewBitmap ?: displayBitmap
                        }
                        Image(
                            bitmap = run { brushVersion; samPreviewVersion; (srcBmp ?: displayBitmap).asImageBitmap() },
                            contentDescription = "结果大图",
                            modifier = Modifier.fillMaxSize(),
                            contentScale = ContentScale.Fit
                        )
                        // ---- SAM 选区覆盖层（对比模式下显示原图，不叠加选区高亮）----
                        if (samMode && viewMode != 1) {
                            val mm = samMask
                            val tap = samTap
                            val hasPreview = samPreviewBitmap != null
                            // ★ 已进入预览态（扣除/换色已生效）或移动中（物体抬起/拖动）：不画青色掩码高亮，让真实效果完整显示
                            //   选区持久化：无论掩码尺寸是否与当前显示图一致（如超分后 base 变大），都缩放对齐后叠加高亮，
                            //   确保选区始终可见（只缩放渲染，绝不回收/修改 samMask 本身）。
                            if (!hasPreview && !samMoving && mm != null && !mm.isRecycled) {
                                val hl = run { samMaskVersion; remember(mm) { makeMaskOverlayBitmap(mm) } }
                                if (hl != null && !hl.isRecycled) {
                                    // 掩码尺寸与显示图不一致时，把高亮缩放到显示图尺寸（仅渲染层缩放，选区本体不变）
                                    val hlScaled = if (hl.width != displayBitmap.width || hl.height != displayBitmap.height) {
                                        try { android.graphics.Bitmap.createScaledBitmap(hl, displayBitmap.width, displayBitmap.height, true) }
                                        catch (_: Exception) { hl }
                                    } else hl
                                    Image(
                                        bitmap = hlScaled.asImageBitmap(),
                                        contentDescription = "选区高亮",
                                        modifier = Modifier.fillMaxSize(),
                                        contentScale = ContentScale.Fit
                                    )
                                    if (hlScaled !== hl && hlScaled.isRecycled.not()) { try { hlScaled.recycle() } catch (_: Exception) {} }
                                }
                            }
                            // 点击点标记（粉色圆点，屏幕坐标映射）——预览态下也隐藏，避免遮真实效果
                            if (!hasPreview && tap != null && !samMoving) {
                                val bw = displayBitmap.width.toFloat()
                                val bh = displayBitmap.height.toFloat()
                                val fitS = min(canvasW / bw, canvasH / bh)
                                val fitW = bw * fitS
                                val fitH = bh * fitS
                                val baseLeft = (canvasW - fitW) / 2f
                                val baseTop = (canvasH - fitH) / 2f
                                val dotX = baseLeft + tap[0] * fitS * scale + offset.x
                                val dotY = baseTop + tap[1] * fitS * scale + offset.y
                                Canvas(Modifier.fillMaxSize()) {
                                    drawCircle(
                                        color = Color(0xFFFF80AB),
                                        radius = 6f * density,
                                        center = Offset(dotX, dotY)
                                    )
                                    drawCircle(
                                        color = Color.White,
                                        radius = 6f * density,
                                        center = Offset(dotX, dotY),
                                        style = androidx.compose.ui.graphics.drawscope.Stroke(width = 2f * density)
                                    )
                                }
                            }
                            // 移动模式：实时贴片预览（底图 + 平移物体），不再用选择框
                            if (samMoving && mm != null && !mm.isRecycled) {
                                val bw = displayBitmap.width.toFloat()
                                val bh = displayBitmap.height.toFloat()
                                val fitS = min(canvasW / bw, canvasH / bh)
                                val fitW = bw * fitS
                                val fitH = bh * fitS
                                val baseLeft = (canvasW - fitW) / 2f
                                val baseTop = (canvasH - fitH) / 2f
                                // 底图 = 物体已被扣空（露棋盘）；贴片 = 掩码内物体（可平移），放到 Box 的底层
                                //   —— 用两个 Image 叠在图形之上：先画扣空的底图，再画平移的物体贴片
                                // ★ 每次移动绘制实时贴片：底图 + 平移后的物体
                                //   本 Canvas 位于「图像 Box」内部（该 Box 已通过 graphicsLayer 完成
                                //   scale 缩放 + offset 平移 + baseLeft/baseTop 居中定位）。
                                //   → 内部坐标就是「适配后图像尺寸 fitW×fitH」，不要再叠加 offset/scale，
                                //     否则会产生双重变换导致物体乱跳/移动不了。
                                // ★ 物体贴片(两图层)：物体层已被裁剪成「紧密包围盒」独立小图 samObjectTile（宽 samObjW×samObjH，
                                //   在原图偏移 samObjOx,samObjOy）。绘制时把它放回原位(原包围盒左上角)再叠加上被拖动的位移。
                                val tile = samObjectTile
                                if (tile != null && !tile.isRecycled) {
                                    Canvas(Modifier.fillMaxSize()) {
                                        val native = drawContext.canvas.nativeCanvas
                                        val left = (samObjOx + samMoveImg.x) * fitS
                                        val top = (samObjOy + samMoveImg.y) * fitS
                                        val objRect = RectF(
                                            left,
                                            top,
                                            left + samObjW * fitS,
                                            top + samObjH * fitS
                                        )
                                        native.drawBitmap(tile, null, objRect, Paint().apply { isFilterBitmap = true })
                                    }
                                }
                            }
                        }
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
                // ---- ★ 统一运行中 Loading 遮罩（选区/超分）----
                // busy 状态驱动：运行中显示转圈，停止（try/finally 复位）自动消失；
                // 报错由 catch 写状态/Toast 透传，遮罩随 busy=false 关闭。
                // ★ 选区(SAM)加载掩码：不做全屏转圈动画（避免遮挡正在分析的原图），仅顶部轻提示；
                //   超分处理则保留全屏 Loading（耗时久需要明确进度反馈）。
                if (superResBusy) {
                    Box(
                        Modifier
                            .fillMaxSize()
                            .background(Color(0x66000000)),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            CircularProgressIndicator(
                                color = Color(0xFF64B5F6),
                                strokeWidth = 4.dp,
                                modifier = Modifier.size(44.dp)
                            )
                            Spacer(Modifier.height(12.dp))
                            Text(
                                "4x 超分处理中…",
                                color = Color.White, fontSize = 14.sp
                            )
                            Spacer(Modifier.height(4.dp))
                            Text(
                                "大图放大较慢，请勿关闭页面",
                                color = Color(0xFF90A4AE), fontSize = 11.sp
                            )
                        }
                    }
                }
                // ★ 选区加载掩码：仅顶部小条文字提示，不遮挡画布、无转圈动画
                if (samBusy) {
                    Box(
                        Modifier
                            .align(Alignment.TopCenter)
                            .padding(top = 8.dp)
                            .background(Color(0xCC101418), RoundedCornerShape(6.dp))
                            .padding(horizontal = 12.dp, vertical = 6.dp)
                    ) {
                        Text(
                            when {
                                samStatus.startsWith("分割失败") -> "分割出错，请重试"
                                else -> "智能选区分析中…"
                            },
                            color = Color.White, fontSize = 12.sp
                        )
                    }
                }
                // ★ 后处理全图调色重算中：顶部小条提示（让用户知道在处理，不是卡死/没反应）
                if (postColorBusy && !samMode) {
                    Box(
                        Modifier
                            .align(Alignment.TopCenter)
                            .padding(top = 8.dp)
                            .background(Color(0xCC1A1F24), RoundedCornerShape(6.dp))
                            .padding(horizontal = 12.dp, vertical = 6.dp)
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            CircularProgressIndicator(
                                modifier = Modifier.size(14.dp),
                                strokeWidth = 2.dp,
                                color = Color(0xFFCE93D8)
                            )
                            Text("调色处理中，大图较慢请稍候…", color = Color.White, fontSize = 12.sp)
                        }
                    }
                }
                // ★ 后处理参数浮窗（后处理模式 + 展开「参数调整」时向上弹出在画布顶部）
                if (!samMode && !brushMode && showPostControls) {
                    Column(
                        modifier = Modifier
                            .align(Alignment.TopCenter)
                            .padding(top = 8.dp)
                            .fillMaxWidth(0.92f)
                            .background(Color(0xF01A1F24), RoundedCornerShape(10.dp))
                            .border(1.dp, Color(0xFF90A4AE).copy(alpha = 0.3f), RoundedCornerShape(10.dp))
                            .padding(horizontal = 12.dp, vertical = 8.dp)
                            // ★ 防透传：拦截浮层上的点击，绝不穿透到底部画布
                            .clickable(interactionSource = remember { MutableInteractionSource() }, indication = null) {}
                    ) {
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
                        Text("把 alpha 低于此值的半透明背景雾清除为全透明（值越高背景越干净，也会削掉更细的发丝）", color = Color(0xFF90A4AE), fontSize = 10.sp)
                        Spacer(Modifier.height(4.dp))
                        // 去色边
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
                        Text("还原半透明边缘被污染的底色（去白边/黑边/彩边）", color = Color(0xFF90A4AE), fontSize = 10.sp)
                        Spacer(Modifier.height(4.dp))
                        // 去毛边
                        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Text("去毛边", color = Color(0xFFFFB74D), fontSize = 12.sp)
                            Slider(
                                value = shrink.toFloat(),
                                onValueChange = { shrink = it.roundToInt() },
                                valueRange = 0f..20f,
                                modifier = Modifier.weight(1f)
                            )
                            Text("${shrink}px", color = Color(0xFFFFB74D), fontSize = 12.sp)
                        }
                        Text("沿边缘向内收缩，收掉突出的毛刺/锯齿", color = Color(0xFF90A4AE), fontSize = 10.sp)
                        Spacer(Modifier.height(4.dp))
                        // 背景硬化
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
                        Text("把 alpha 低于此值的半透明残留雾影设为全透明（0=关闭）", color = Color(0xFF90A4AE), fontSize = 10.sp)
                    }
                }
                // ★ 后处理全图调色浮层（后处理模式 + 展开「调色」时悬浮在画布顶部）
                if (!samMode && !brushMode && showPostColor) {
                    Column(
                        modifier = Modifier
                            .align(Alignment.TopCenter)
                            .padding(top = 8.dp)
                            .fillMaxWidth(0.92f)
                            .background(Color(0xF01A1F24), RoundedCornerShape(10.dp))
                            .border(1.dp, Color(0xFFCE93D8).copy(alpha = 0.3f), RoundedCornerShape(10.dp))
                            .padding(horizontal = 12.dp, vertical = 8.dp)
                            // ★ 防透传：拦截浮层上的点击，绝不穿透到底部画布（否则误触分割导致选区跳走）
                            .clickable(interactionSource = remember { MutableInteractionSource() }, indication = null) {}
                    ) {
                        // 色相滑块
                        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Box(
                                modifier = Modifier
                                    .size(28.dp)
                                    .clip(RoundedCornerShape(6.dp))
                                    .background(if (postHue < 0f) Color(0xFF90A4AE) else samHueToColor(postHue))
                                    .border(1.dp, Color(0xFFFFFFFF).copy(alpha = 0.35f), RoundedCornerShape(6.dp))
                            )
                            Text("色相", color = Color(0xFF90A4AE), fontSize = 12.sp)
                            Slider(
                                value = if (postHue < 0f) 0f else postHue,
                                onValueChange = { h ->
                                    snapshotForUndo()
                                    postHue = h
                                },
                                valueRange = 0f..359f,
                                modifier = Modifier.weight(1f)
                            )
                            Text(if (postHue < 0f) "原色" else "${postHue.toInt()}°", color = Color.White, fontSize = 12.sp)
                        }
                        // 饱和度滑块
                        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Text("饱和", color = Color(0xFF90A4AE), fontSize = 12.sp)
                            Slider(
                                value = postSatScale,
                                onValueChange = { s ->
                                    snapshotForUndo()
                                    postSatScale = s
                                },
                                valueRange = 0f..2f,
                                modifier = Modifier.weight(1f)
                            )
                            Text("${(postSatScale * 100).roundToInt()}%", color = Color.White, fontSize = 12.sp)
                        }
                        // 原色按钮
                        Column(verticalArrangement = Arrangement.spacedBy(6.dp), modifier = Modifier.fillMaxWidth()) {
                            listOf(
                                "原色" to -1f, "红" to 0f, "橙" to 30f, "黄" to 60f, "绿" to 120f,
                                "青" to 180f, "蓝" to 240f, "紫" to 300f, "粉" to 340f,
                                "玫红" to 320f, "柠檬" to 55f, "薄荷" to 150f
                            ).chunked(6).forEach { row ->
                                Row(horizontalArrangement = Arrangement.spacedBy(4.dp), modifier = Modifier.fillMaxWidth()) {
                                    row.forEach { (name, h) ->
                                        val color = if (h < 0f) Color(0xFF90A4AE) else samHueToColor(h)
                                        Box(
                                            modifier = Modifier
                                                .weight(1f)
                                                .height(26.dp)
                                                .clip(RoundedCornerShape(5.dp))
                                                .background(color)
                                                .border(
                                                    if ((h < 0f && postHue < 0f) || (h >= 0f && postHue == h)) BorderStroke(2.dp, Color.White)
                                                    else BorderStroke(1.dp, Color(0x33FFFFFF)),
                                                    RoundedCornerShape(5.dp)
                                                )
                                                .clickable {
                                                    snapshotForUndo()
                                                    if (h < 0f) postHue = -1f else postHue = h
                                                },
                                            contentAlignment = Alignment.Center
                                        ) {
                                            Text(name, color = Color.White, fontSize = 8.sp)
                                        }
                                    }
                                }
                            }
                        }
                        // ★ 自定义颜色输入（#RRGGBB）
                        Spacer(Modifier.height(4.dp))
                        // 输入框独占一行（全宽，避免与色块/按钮同行导致高度错位）
                        androidx.compose.material3.OutlinedTextField(
                            value = postCustomHex,
                            onValueChange = { raw ->
                                var s = raw.trim().take(7)
                                if (s.isNotEmpty() && s[0] != '#') s = "#$s"
                                postCustomHex = s
                                postCustomHexValid = Regex("^#[0-9a-fA-F]{6}$").matches(s)
                            },
                            singleLine = true,
                            placeholder = { Text("#RRGGBB", fontSize = 12.sp, color = Color(0xFF90A4AE)) },
                            textStyle = androidx.compose.ui.text.TextStyle(fontSize = 13.sp, color = Color.White),
                            // ★ 不设固定高度（避免裁剪文字），让输入框自适应最小高度保证文字完整
                            modifier = Modifier.fillMaxWidth(),
                            colors = androidx.compose.material3.OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = Color(0xFFCE93D8),
                                unfocusedBorderColor = Color(0x66FFFFFF),
                                focusedTextColor = Color.White,
                                cursorColor = Color(0xFFCE93D8)
                            )
                        )
                        // 第二行：预览色块 + 应用按钮（居中）
                        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            // 颜色预览色块
                            Box(
                                modifier = Modifier
                                    .size(34.dp)
                                    .clip(RoundedCornerShape(8.dp))
                                    .background(if (postCustomHexValid) Color(parseHexColor(postCustomHex)) else Color(0xFF37474F))
                                    .border(1.dp, Color(0x55FFFFFF), RoundedCornerShape(8.dp))
                            )
                            // 应用按钮
                            Box(
                                modifier = Modifier
                                    .height(34.dp)
                                    .clip(RoundedCornerShape(8.dp))
                                    .background(if (postCustomHexValid) Color(0xFFCE93D8) else Color(0xFF263238))
                                    .border(1.dp, if (postCustomHexValid) Color.White else Color(0x44FFFFFF), RoundedCornerShape(8.dp))
                                    .clickable(enabled = postCustomHexValid && !superResBusy) {
                                        val h = hexToHueDeg(postCustomHex)
                                        snapshotForUndo()
                                        postHue = h
                                        // 同步饱和度：从 HSV 取目标饱和度做参考（不强制，仅提示）
                                    },
                                contentAlignment = Alignment.Center
                            ) {
                                Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(horizontal = 10.dp)) {
                                    Icon(Icons.Filled.Check, contentDescription = null, modifier = Modifier.size(14.dp), tint = Color.White)
                                    Spacer(Modifier.width(4.dp))
                                    Text("应用", fontSize = 12.sp, fontWeight = FontWeight.SemiBold, color = Color.White)
                                }
                            }
                        }
                        Text("全图调色 = 保留明暗层次的颜色偏移，点「撤销」可回退；自定义色取目标色相套用", color = Color(0xFF90A4AE), fontSize = 10.sp)
                    }
                }
                // ★ 调色浮层（选区模式 + 展开「调色」时悬浮在画布顶部，不挤占/覆盖顶栏）
                if (samMode && showSamColorPanel && samMask != null) {
                    Column(
                        modifier = Modifier
                            .align(Alignment.TopCenter)
                            .padding(top = 8.dp)
                            .fillMaxWidth(0.92f)
                            .background(Color(0xF01A1F24), RoundedCornerShape(10.dp))
                            .border(1.dp, Color(0xFF64B5F6).copy(alpha = 0.3f), RoundedCornerShape(10.dp))
                            .padding(horizontal = 12.dp, vertical = 8.dp)
                            // ★ 防透传：拦截浮层空白区域的点击，绝不穿透到底部画布（选区跳走的根治在画布手势上锁）
                            .clickable(interactionSource = remember { MutableInteractionSource() }, indication = null) {}
                    ) {
                        // 色相滑杆 + 当前色预览（拖动实时只对选中分区预览变色）
                        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Box(
                                modifier = Modifier
                                    .size(28.dp)
                                    .clip(RoundedCornerShape(6.dp))
                                    .background(samHueToColor(samHue))
                                    .border(1.dp, Color(0xFFFFFFFF).copy(alpha = 0.35f), RoundedCornerShape(6.dp))
                            )
                            Text("色相", color = Color(0xFF90A4AE), fontSize = 12.sp)
                            Slider(
                                value = samHue.coerceIn(0f, 359f),
                                onValueChange = { h ->
                                    samHue = h
                                    // ★ 直接用原图计算并覆盖选中分区分色（自动回原再上色，不叠加旧预览）
                                    val mm = samMask
                                    if (mm != null) {
                                        val base = displayBitmap
                                        scope.launch {
                                            val out = withContext(Dispatchers.IO) { recolorFromOrigin(base, softOriginBitmap ?: displayBitmap, mm, h, 1f) }
                                            if (out != null) {
                                                samPreviewBitmap?.let { if (it !== out && it.isRecycled.not()) { try { it.recycle() } catch (_: Exception) {} } }
                                                samPreviewBitmap = out
                                                samPreviewVersion++
                                                samStatus = "已预览换色（${h.toInt()}°）· 点「应用」落地"
                                            }
                                        }
                                    }
                                },
                                valueRange = 0f..359f,
                                modifier = Modifier.weight(1f)
                            )
                            Text(if (samHue < 0f) "原色" else "${samHue.toInt()}°", color = Color.White, fontSize = 12.sp)
                        }
                        // ★ 扩展色板（12 色，两行布局，行间加 6dp 间距避免重合）
                        Column(verticalArrangement = Arrangement.spacedBy(6.dp), modifier = Modifier.fillMaxWidth()) {
                            listOf(
                                "原色" to -1f, "红" to 0f, "橙" to 30f, "黄" to 60f, "绿" to 120f, "青" to 180f,
                                "蓝" to 240f, "紫" to 300f, "粉" to 340f, "玫红" to 320f, "柠檬" to 55f, "薄荷" to 150f
                            ).chunked(6).forEach { row ->
                                Row(horizontalArrangement = Arrangement.spacedBy(4.dp), modifier = Modifier.fillMaxWidth()) {
                                    row.forEach { (name, h) ->
                                        val color = if (h < 0f) Color(0xFF90A4AE) else samHueToColor(h)
                                        Box(
                                            modifier = Modifier
                                                .weight(1f)
                                                .height(26.dp)
                                                .clip(RoundedCornerShape(5.dp))
                                                .background(color)
                                                .border(
                                                    if ((h < 0f && samHue < 0f) || (h >= 0f && samHue == h)) BorderStroke(2.dp, Color.White)
                                                    else BorderStroke(1.dp, Color(0x33FFFFFF)),
                                                    RoundedCornerShape(5.dp)
                                                )
                                                .clickable(enabled = samMask != null) {
                                                    // ★ 直接用原图计算新色并写入预览层
                                                    val mm = samMask
                                                    if (mm == null) return@clickable
                                                    if (h < 0f) {
                                                        // 原色：直接恢复物体本色
                                                        samHue = -1f
                                                        samStatus = "恢复原色…"
                                                        scope.launch {
                                                            val base = displayBitmap
                                                            val out = withContext(Dispatchers.IO) { restoreColorByMask(base, softOriginBitmap ?: displayBitmap, mm) }
                                                            if (out != null) {
                                                                samPreviewBitmap?.let { if (it !== out && it.isRecycled.not()) { try { it.recycle() } catch (_: Exception) {} } }
                                                                samPreviewBitmap = out
                                                                samPreviewVersion++
                                                                samStatus = "已恢复原色 · 点「应用」完成，点「取消」还原"
                                                            }
                                                        }
                                                    } else {
                                                        // 彩色：点击直接变色
                                                        samHue = h
                                                        samStatus = "换色中…"
                                                        scope.launch {
                                                            val base = displayBitmap
                                                            // ★ 自动回原再上色：先恢复原色再色相平移，每次从原色出发
                                                            val out = withContext(Dispatchers.IO) { recolorFromOrigin(base, softOriginBitmap ?: displayBitmap, mm, samHue, 1f) }
                                                            if (out != null) {
                                                                samPreviewBitmap?.let { if (it !== out && it.isRecycled.not()) { try { it.recycle() } catch (_: Exception) {} } }
                                                                samPreviewBitmap = out
                                                                samPreviewVersion++
                                                                samStatus = "已换为「$name」· 点「应用」完成，点「取消」还原"
                                                            }
                                                        }
                                                    }
                                                },
                                            contentAlignment = Alignment.Center
                                        ) {
                                            Text(name, color = Color.White, fontSize = 8.sp)
                                        }
                                    }
                                }
                            }
                        }
                        // ★ 自定义颜色输入（#RRGGBB 十六进制）：可直接输入任意颜色应用到选中分区
                        Spacer(Modifier.height(4.dp))
                        // 输入框独占一行（全宽，避免与色块/按钮同行导致高度错位）
                        androidx.compose.material3.OutlinedTextField(
                            value = samCustomHex,
                            onValueChange = { raw ->
                                var s = raw.trim().take(7)
                                if (s.isNotEmpty() && s[0] != '#') s = "#$s"
                                samCustomHex = s
                                samCustomHexValid = Regex("^#[0-9a-fA-F]{6}$").matches(s)
                            },
                            singleLine = true,
                            placeholder = { Text("#RRGGBB", fontSize = 12.sp, color = Color(0xFF90A4AE)) },
                            textStyle = androidx.compose.ui.text.TextStyle(fontSize = 13.sp, color = Color.White),
                            // ★ 不设固定高度（避免裁剪文字），让输入框自适应最小高度保证文字完整
                            modifier = Modifier.fillMaxWidth(),
                            colors = androidx.compose.material3.OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = Color(0xFF64B5F6),
                                unfocusedBorderColor = Color(0x66FFFFFF),
                                focusedTextColor = Color.White,
                                cursorColor = Color(0xFF64B5F6)
                            )
                        )
                        // 第二行：预览色块 + 应用按钮（居中）
                        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            // 颜色预览色块
                            Box(
                                modifier = Modifier
                                    .size(34.dp)
                                    .clip(RoundedCornerShape(8.dp))
                                    .background(if (samCustomHexValid) Color(parseHexColor(samCustomHex)) else Color(0xFF37474F))
                                    .border(1.dp, Color(0x55FFFFFF), RoundedCornerShape(8.dp))
                            )
                            // 应用按钮
                            Box(
                                modifier = Modifier
                                    .height(34.dp)
                                    .clip(RoundedCornerShape(8.dp))
                                    .background(if (samCustomHexValid) Color(0xFF64B5F6) else Color(0xFF263238))
                                    .border(1.dp, if (samCustomHexValid) Color.White else Color(0x44FFFFFF), RoundedCornerShape(8.dp))
                                    .clickable(enabled = samCustomHexValid && samMask != null) {
                                        // ★ 直接用原图计算新色并写入预览层（不叠加旧预览）
                                        val mm = samMask
                                        if (mm == null) return@clickable
                                        val h = hexToHueDeg(samCustomHex)  // 取目标色相，套用换色机制（保留明暗层次）
                                        samHue = h
                                        samStatus = "应用自定义色 #${samCustomHex.removePrefix("#").uppercase()}…"
                                        // ★ 始终基于 displayBitmap 原图重算（不叠加旧预览）
                                        val base = displayBitmap
                                        scope.launch {
                                            // ★ 自动回原再上色：先恢复原色再色相平移，每次从原色出发
                                            val out = withContext(Dispatchers.IO) { recolorFromOrigin(base, softOriginBitmap ?: displayBitmap, mm, h, 1f) }
                                            if (out != null) {
                                                samPreviewBitmap?.let { if (it !== out && it.isRecycled.not()) { try { it.recycle() } catch (_: Exception) {} } }
                                                samPreviewBitmap = out
                                                samPreviewVersion++
                                                samStatus = "已应用「${samCustomHex.uppercase()}」· 点「应用」完成，点「取消」还原"
                                            }
                                        }
                                    },
                                contentAlignment = Alignment.Center
                            ) {
                                Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(horizontal = 10.dp)) {
                                    Icon(Icons.Filled.Check, contentDescription = null, modifier = Modifier.size(14.dp), tint = Color.White)
                                    Spacer(Modifier.width(4.dp))
                                    Text("应用", fontSize = 12.sp, fontWeight = FontWeight.SemiBold, color = Color.White)
                                }
                            }
                        }
                        Text("换色 = 保留明暗层次的合成偏移（阴影/高光不丢失），非直接填色；自定义色也按此机制", color = Color(0xFF90A4AE), fontSize = 10.sp)
                    }
                }
            }
        }
    }
}

// ================= 选区辅助函数 =================

/** 色相角（0-359）→ 颜色（色板/预览用）；-1 表示"原色"（灰） */
internal fun samHueToColor(hue: Float): Color {
    if (hue < 0f) return Color(0xFF90A4AE)
    val h = hue / 360f
    val c = 1f
    val x = 1f - kotlin.math.abs((h * 6f) % 2f - 1f)
    val (r, g, b) = when {
        h < 1f/6f -> Triple(c, x, 0f)
        h < 2f/6f -> Triple(x, c, 0f)
        h < 3f/6f -> Triple(0f, c, x)
        h < 4f/6f -> Triple(0f, x, c)
        h < 5f/6f -> Triple(x, 0f, c)
        else -> Triple(c, 0f, x)
    }
    return Color((r * 255).toInt(), (g * 255).toInt(), (b * 255).toInt())
}

/**
 * ★ 自动回原再上色（分区单次换色）：
 * 1) 先把 mask 区域恢复成原图颜色（从 srcPixels 取回原始 RGB）
 * 2) 再基于恢复后的原图颜色做色相平移到目标色相
 *
 * 这样每次换色都从【原色】出发，色相偏移始终准确，
 * 不会因前一次换色残留导致累加偏移/色相计算错误。
 */
internal fun recolorFromOrigin(base: Bitmap, srcPixels: Bitmap, mask: Bitmap?, hueDeg: Float, satScale: Float): Bitmap? {
    if (base.isRecycled) return null
    // 第 1 步：mask 区域恢复原色
    val restored = if (mask != null && !mask.isRecycled) {
        restoreColorByMask(base, srcPixels, mask)
    } else {
        base.copy(Bitmap.Config.ARGB_8888, true)
    } ?: return null
    // 第 2 步：基于恢复后的原图做色相平移
    return recolorByMask(restored, mask, hueDeg, satScale)
}

/** 恢复原色：掩码内像素从原图副本取回原始 RGB（把换过的色还原），保留当前 alpha */
internal fun restoreColorByMask(base: Bitmap, srcPixels: Bitmap, mask: Bitmap): Bitmap? {
    if (base.isRecycled || mask.isRecycled) return null
    val w = base.width; val h = base.height
    // ★ 与 recolorByMask 一致的尺寸适配：mask 存在但尺寸不匹配时缩放到 base 尺寸，
    //   绝不退化为 null（否则恢复原色在超分后图外溢或失效）
    var m = if (mask.isRecycled) null else mask
    if (m != null && (m.width != w || m.height != h)) {
        try { m = Bitmap.createScaledBitmap(m, w, h, true) } catch (_: Exception) { m = null }
    }
    if (m == null) return base.copy(Bitmap.Config.ARGB_8888, true)
    val out = base.copy(Bitmap.Config.ARGB_8888, true) ?: return null
    val px = IntArray(w * h)
    val mp = IntArray(w * h)
    out.getPixels(px, 0, w, 0, 0, w, h)
    m.getPixels(mp, 0, w, 0, 0, w, h)
    val srcPx = IntArray(w * h)
    srcPixels.getPixels(srcPx, 0, w, 0, 0, w, h)
    for (i in px.indices) {
        if ((mp[i] ushr 24) and 0xFF > 127) {
            // 取原图 RGB，保留当前 alpha（物体已扣空则 alpha 为 0，保持透明）
            px[i] = (srcPx[i] and 0x00FFFFFF) or (px[i] and 0xFF000000.toInt())
        }
    }
    out.setPixels(px, 0, w, 0, 0, w, h)
    return out
}

/** 解析 #RRGGBB 十六进制颜色 → ARGB Int（供预览色块/应用用）；非法返回 0xFFFF5722 */
internal fun parseHexColor(hex: String): Int {
    val s = hex.trim().removePrefix("#")
    if (s.length != 6) return 0xFFFF5722.toInt()
    return try { (0xFF000000.toInt()) or s.toLong(16).toInt() } catch (_: Exception) { 0xFFFF5722.toInt() }
}

/** #RRGGBB 十六进制 → 色相角（0-359），用于套用换色机制；非法返回 120（绿） */
internal fun hexToHueDeg(hex: String): Float {
    val argb = parseHexColor(hex)
    val hsv = FloatArray(3)
    android.graphics.Color.RGBToHSV((argb shr 16) and 0xFF, (argb shr 8) and 0xFF, argb and 0xFF, hsv)
    return hsv[0]
}


