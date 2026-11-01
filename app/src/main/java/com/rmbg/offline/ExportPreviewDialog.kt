package com.rmbg.offline

import android.graphics.Bitmap
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Preview
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * 第三个图片预览器：导出预览（所见即所得）。
 * 实时预览 清脏 RGB + 主体收纳/居中 + 导出格式压缩 的最终结果与体积；
 * JPG 语义下透明区直接合成白底展示（所见即所得）。
 * 弹窗内调参数不写 Prefs（试到满意 → 保存/关闭）。
 */
@Composable
fun ExportPreviewDialog(
    source: Bitmap,
    onDismiss: () -> Unit
) {
    val context = LocalContext.current
    var pvFormat by remember { mutableStateOf(Prefs.exportFormat) }
    var pvQuality by remember { mutableStateOf(Prefs.exportQuality) }
    var pvMargin by remember { mutableStateOf(Prefs.exportMarginPct) }
    var pvCenter by remember { mutableStateOf(Prefs.exportCenter) }
    var pvClean by remember { mutableStateOf(Prefs.exportCleanRgb) }
    var pvBytes by remember { mutableStateOf(-1L) }
    val pvBitmapState = remember { mutableStateOf<Bitmap?>(null) }

    // 参数变化 → IO 线程重算预览（防卡 UI）
    LaunchedEffect(pvFormat, pvQuality, pvMargin, pvCenter, pvClean) {
        val fmt = ExportProcessor.ExportFormat.valueOf(pvFormat)
        val b = withContext(Dispatchers.IO) {
            ExportProcessor.buildPreview(source, fmt, pvQuality, pvMargin, pvCenter, pvClean)
        }
        pvBitmapState.value?.recycle()
        pvBitmapState.value = b.first
        pvBytes = b.second
    }

    // 弹窗内保存：按当前预览参数导出（含收纳/居中/清理/格式/质量）
    val saveLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.CreateDocument("image/*")
    ) { uri ->
        val ok = if (uri != null) {
            val fmt = ExportProcessor.ExportFormat.valueOf(pvFormat)
            ExportProcessor.write(context, source, uri, fmt, pvQuality, pvMargin, pvCenter, pvClean)
        } else false
        android.widget.Toast.makeText(
            context,
            if (ok) "已按预览参数保存" else "保存失败",
            android.widget.Toast.LENGTH_SHORT
        ).show()
        if (ok) onDismiss()
    }

    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(usePlatformDefaultWidth = false)
    ) {
        Surface(
            modifier = Modifier.fillMaxWidth().fillMaxSize().padding(16.dp),
            shape = RoundedCornerShape(20.dp),
            color = MaterialTheme.colorScheme.surface
        ) {
            Column(
                modifier = Modifier.padding(14.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                // 标题行
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Filled.Preview, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                    Spacer(Modifier.width(8.dp))
                    Text("导出预览", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                    Spacer(Modifier.weight(1f))
                    TextButton(onClick = onDismiss) { Text("关闭") }
                }
                // 预览画布（浅灰底衬托透明；JPG 时位图本身已合成白底）
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(12.dp))
                        .background(Color(0xFFE8E8E8)),
                    contentAlignment = Alignment.Center
                ) {
                    val pvB = pvBitmapState.value
                    if (pvB != null) {
                        Image(
                            bitmap = pvB.asImageBitmap(),
                            contentDescription = "导出预览",
                            modifier = Modifier.fillMaxSize(),
                            contentScale = ContentScale.Fit
                        )
                    } else {
                        CircularProgressIndicator(Modifier.size(28.dp))
                    }
                    // 左上角信息角标（格式/预估体积/状态）
                    Column(
                        modifier = Modifier.align(Alignment.TopStart).padding(6.dp),
                        verticalArrangement = Arrangement.spacedBy(2.dp)
                    ) {
                        val fmtLabel = when (pvFormat) {
                            "PNG" -> "PNG 无损"
                            "JPG" -> "JPG 白底 q$pvQuality"
                            else -> "WebP q$pvQuality"
                        }
                        Text(
                            fmtLabel, fontSize = 11.sp, color = Color.White,
                            modifier = Modifier.background(Color(0x99000000), RoundedCornerShape(4.dp))
                                .padding(horizontal = 5.dp, vertical = 1.dp)
                        )
                        Text(
                            if (pvBytes >= 0) "预估 %.1f KB（%dx%d）".format(
                                pvBytes / 1024f, pvB?.width ?: 0, pvB?.height ?: 0
                            ) else "计算中...",
                            fontSize = 11.sp, color = Color.White,
                            modifier = Modifier.background(Color(0x99000000), RoundedCornerShape(4.dp))
                                .padding(horizontal = 5.dp, vertical = 1.dp)
                        )
                        if (pvMargin >= 0) {
                            Text(
                                if (pvMargin == 0) "已收纳·紧贴主体" else "已收纳·边距$pvMargin%",
                                fontSize = 11.sp, color = Color.White,
                                modifier = Modifier.background(Color(0x99000000), RoundedCornerShape(4.dp))
                                    .padding(horizontal = 5.dp, vertical = 1.dp)
                            )
                        }
                        if (pvClean) {
                            Text(
                                "已清脏 RGB", fontSize = 11.sp, color = Color.White,
                                modifier = Modifier.background(Color(0x99000000), RoundedCornerShape(4.dp))
                                    .padding(horizontal = 5.dp, vertical = 1.dp)
                            )
                        }
                    }
                }
                HorizontalDivider(modifier = Modifier.padding(vertical = 2.dp))
                // 格式三选
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    listOf("PNG", "WEBP", "JPG").forEach { f ->
                        val selected = pvFormat == f
                        OutlinedButton(
                            onClick = { pvFormat = f },
                            modifier = Modifier.weight(1f).height(38.dp),
                            shape = RoundedCornerShape(10.dp),
                            colors = androidx.compose.material3.ButtonDefaults.outlinedButtonColors(
                                containerColor = if (selected) MaterialTheme.colorScheme.primary.copy(alpha = 0.12f)
                                else Color.Transparent
                            ),
                            border = androidx.compose.foundation.BorderStroke(
                                if (selected) 2.dp else 1.dp,
                                if (selected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outline
                            ),
                            contentPadding = androidx.compose.foundation.layout.PaddingValues(0.dp)
                        ) {
                            Text(f, fontSize = 13.sp,
                                fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal)
                        }
                    }
                }
                // 质量滑杆（PNG 忽略）
                val qEnabled = pvFormat != "PNG"
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("压缩质量", fontSize = 13.sp, fontWeight = FontWeight.Bold)
                    Spacer(Modifier.weight(1f))
                    Text(
                        if (qEnabled) "$pvQuality" else "无损",
                        fontSize = 13.sp,
                        color = if (qEnabled) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant,
                        fontWeight = FontWeight.Bold
                    )
                }
                Slider(
                    value = pvQuality.toFloat(),
                    onValueChange = { pvQuality = it.toInt() },
                    valueRange = 50f..100f,
                    steps = 9,
                    enabled = qEnabled
                )
                // 主体收纳滑杆（-1=关）
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("主体收纳", fontSize = 13.sp, fontWeight = FontWeight.Bold)
                    Spacer(Modifier.weight(1f))
                    Text(
                        if (pvMargin < 0) "关闭" else "边距 $pvMargin%",
                        fontSize = 13.sp,
                        color = MaterialTheme.colorScheme.primary, fontWeight = FontWeight.Bold
                    )
                }
                Slider(
                    value = pvMargin.toFloat(),
                    onValueChange = { pvMargin = it.toInt() },
                    valueRange = -1f..50f,
                    steps = 50
                )
                Text(
                    if (pvMargin < 0) "关闭：保留完整画布"
                    else if (pvMargin == 0) "边距 0：画布紧贴主体（自动居中）"
                    else "边距 ${pvMargin}%：主体四周留白（自动居中）",
                    fontSize = 11.sp, color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                // 居中 + 清脏 RGB 两开关
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("导出居中", fontSize = 13.sp, modifier = Modifier.weight(1f))
                    Switch(checked = pvCenter, onCheckedChange = { pvCenter = it })
                    Spacer(Modifier.width(12.dp))
                    Text("清脏 RGB", fontSize = 13.sp)
                    Switch(checked = pvClean, onCheckedChange = { pvClean = it })
                }
                // 保存行
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(
                        onClick = { saveLauncher.launch(ExportProcessor.suggestedName(
                            ExportProcessor.ExportFormat.valueOf(pvFormat))) },
                        modifier = Modifier.weight(1f).height(44.dp),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Text("按预览另存为", fontSize = 13.sp, fontWeight = FontWeight.Bold)
                    }
                }
            }
        }
    }
}
