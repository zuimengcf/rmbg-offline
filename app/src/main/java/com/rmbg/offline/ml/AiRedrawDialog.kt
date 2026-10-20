package com.rmbg.offline.ml
import com.rmbg.offline.R
import com.rmbg.offline.OperitApp

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
                        OperitApp.appContext.getString(R.string.ai_5),
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.weight(1f)
                    )
                    TextButton(onClick = onDismiss) { Text("✕") }
                }

                Text(
                    OperitApp.appContext.getString(R.string.npu_5),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )

                OutlinedTextField(
                    value = prompt,
                    onValueChange = { prompt = it },
                    label = { Text(OperitApp.appContext.getString(R.string.str_347)) },
                    placeholder = { Text(OperitApp.appContext.getString(R.string.str_117)) },
                    modifier = Modifier.fillMaxWidth(),
                    minLines = 2,
                    maxLines = 4
                )

                OutlinedTextField(
                    value = negativePrompt,
                    onValueChange = { negativePrompt = it },
                    label = { Text(OperitApp.appContext.getString(R.string.str_407)) },
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
                        Text(OperitApp.appContext.getString(R.string.str_181), style = MaterialTheme.typography.labelMedium)
                        OutlinedTextField(
                            value = width.toString(),
                            onValueChange = { width = it.toIntOrNull() ?: 512 },
                            singleLine = true
                        )
                    }
                    Column(Modifier.weight(1f)) {
                        Text(OperitApp.appContext.getString(R.string.str_454), style = MaterialTheme.typography.labelMedium)
                        OutlinedTextField(
                            value = height.toString(),
                            onValueChange = { height = it.toIntOrNull() ?: 512 },
                            singleLine = true
                        )
                    }
                }

                // ---- 调度器 ----
                Column {
                    Text(OperitApp.appContext.getString(R.string.str_403), style = MaterialTheme.typography.labelMedium)
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
                    ) { Text(OperitApp.appContext.getString(R.string.str_162)) }
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
                    ) { Text(OperitApp.appContext.getString(R.string.str_246)) }
                }
            }
        }
    }
}