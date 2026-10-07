package com.rmbg.offline.ml

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog

/**
 * AI 重绘（LocalDream img2img）参数对话框。
 *
 * @param onDismiss 关闭
 * @param onStart 开始重绘（参数已填好）
 */
@Composable
fun AiRedrawDialog(
    onDismiss: () -> Unit,
    onStart: (LocalDreamClient.RedrawRequest) -> Unit
) {
    var prompt by remember { mutableStateOf("keep original, sharp details, high quality") }
    var negativePrompt by remember { mutableStateOf("blurry, low quality, watermark, deformed") }
    var steps by remember { mutableIntStateOf(20) }
    var cfg by remember { mutableFloatStateOf(6.5f) }
    var denoise by remember { mutableFloatStateOf(0.35f) }
    var scheduler by remember { mutableStateOf("dpm_sde") }
    var width by remember { mutableIntStateOf(512) }
    var height by remember { mutableIntStateOf(512) }

    Dialog(onDismissRequest = onDismiss) {
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            shape = androidx.compose.foundation.shape.RoundedCornerShape(20.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
        ) {
            Column(
                modifier = Modifier
                    .padding(20.dp)
                    .verticalScroll(rememberScrollState())
                    .heightIn(max = 620.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        "AI 重绘 (img2img)",
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.weight(1f)
                    )
                    TextButton(onClick = onDismiss) { Text("✕") }
                }

                Text(
                    "通过 LocalDream 本地 NPU 重绘当前图片。需先在 LocalDream App 下载模型并保持后台运行。",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )

                OutlinedTextField(
                    value = prompt,
                    onValueChange = { prompt = it },
                    label = { Text("正向提示词") },
                    placeholder = { Text("例如: masterpiece, best quality, 1girl, detailed") },
                    modifier = Modifier.fillMaxWidth(),
                    minLines = 2,
                    maxLines = 4
                )

                OutlinedTextField(
                    value = negativePrompt,
                    onValueChange = { negativePrompt = it },
                    label = { Text("负面提示词") },
                    modifier = Modifier.fillMaxWidth(),
                    minLines = 2,
                    maxLines = 3
                )

                // ---- 参数行：步数 / CFG ----
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    Column(Modifier.weight(1f)) {
                        Text("步数: $steps", style = MaterialTheme.typography.labelMedium)
                        Slider(
                            value = steps.toFloat(),
                            onValueChange = { steps = it.toInt() },
                            valueRange = 1f..50f
                        )
                    }
                    Column(Modifier.weight(1f)) {
                        Text("CFG: ${"%.1f".format(cfg)}", style = MaterialTheme.typography.labelMedium)
                        Slider(
                            value = cfg,
                            onValueChange = { cfg = it },
                            valueRange = 1f..15f
                        )
                    }
                }

                // ---- 重绘强度 ----
                Column {
                    Text("重绘强度: ${"%.2f".format(denoise)}", style = MaterialTheme.typography.labelMedium)
                    Slider(
                        value = denoise,
                        onValueChange = { denoise = it },
                        valueRange = 0.1f..1f
                    )
                }

                // ---- 分辨率 ----
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    Column(Modifier.weight(1f)) {
                        Text("宽度", style = MaterialTheme.typography.labelMedium)
                        OutlinedTextField(
                            value = width.toString(),
                            onValueChange = { width = it.toIntOrNull() ?: 512 },
                            singleLine = true
                        )
                    }
                    Column(Modifier.weight(1f)) {
                        Text("高度", style = MaterialTheme.typography.labelMedium)
                        OutlinedTextField(
                            value = height.toString(),
                            onValueChange = { height = it.toIntOrNull() ?: 512 },
                            singleLine = true
                        )
                    }
                }

                // ---- 调度器 ----
                Column {
                    Text("调度器", style = MaterialTheme.typography.labelMedium)
                    val schedulers = listOf("dpm_sde", "euler", "euler_ancestral", "ddim", "lcm")
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        schedulers.forEach { s ->
                            FilterChip(
                                selected = scheduler == s,
                                onClick = { scheduler = s },
                                label = { Text(s, style = MaterialTheme.typography.labelSmall) }
                            )
                        }
                    }
                }

                // ---- 操作按钮 ----
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    OutlinedButton(
                        onClick = onDismiss,
                        modifier = Modifier.weight(1f)
                    ) { Text("取消") }
                    Button(
                        onClick = {
                            onStart(
                                LocalDreamClient.RedrawRequest(
                                    prompt = prompt.trim(),
                                    negativePrompt = negativePrompt.trim(),
                                    steps = steps,
                                    cfg = cfg,
                                    width = width,
                                    height = height,
                                    scheduler = scheduler,
                                    denoiseStrength = denoise,
                                    imageBase64 = ""
                                )
                            )
                        },
                        enabled = prompt.isNotBlank(),
                        modifier = Modifier.weight(1f)
                    ) { Text("开始重绘") }
                }
            }
        }
    }
}