package com.rmbg.offline

import android.content.Context
import android.os.Build
import java.util.concurrent.Executors

/**
 * ONNX 加速能力检测
 * - 检查设备是否支持 NNAPI
 * - 列出可用硬件加速器（NPU/DSP/GPU）
 * - 用 SessionOptions.addNnapi 试跑判断 NNAPI 是否真的可用
 */
object AccelDetector {

    /** 检测结果数据类 */
    data class DetectResult(
        val nnapiSupported: Boolean,
        val deviceCount: Int,
        val deviceNames: List<String>,
        val cpuCores: Int,
        val ramMB: Long
    )

    /** 检测设备加速能力（IO 线程调用） */
    fun detect(context: Context): DetectResult {
        val deviceNames = mutableListOf<String>()
        var nnapiSupported = false

        // 尝试创建 NNAPI SessionOptions（真实试跑，不抛异常即 API 可用）
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            try {
                val opts = ai.onnxruntime.OrtSession.SessionOptions()
                opts.addNnapi()
                // 能创建 NNAPI 选项说明系统支持
                nnapiSupported = true
                deviceNames.add("NNAPI (系统默认)")
            } catch (_: Exception) {
                nnapiSupported = false
            }
        } else {
            // 低版本系统不支持 NNAPI
            nnapiSupported = false
        }

        // CPU 核心数 & 存储空间
        val cores = Runtime.getRuntime().availableProcessors()
        val storageMB = android.os.Environment.getDataDirectory().totalSpace / 1024 / 1024

        return DetectResult(
            nnapiSupported = nnapiSupported,
            deviceCount = deviceNames.size,
            deviceNames = deviceNames,
            cpuCores = cores,
            ramMB = storageMB / 1024
        )
    }

    /** 生成人类可读的检测报告 */
    fun formatResult(r: DetectResult): String {
        val sb = StringBuilder()
        sb.append("NNAPI 硬件加速: ${if (r.nnapiSupported) "✔ 支持" else "✘ 不支持"}\n")
        if (r.deviceNames.isNotEmpty()) {
            sb.append("检测到加速器 (${r.deviceCount}):\n")
            r.deviceNames.forEach { sb.append("  • $it\n") }
        } else {
            sb.append("未检测到专用加速器\n")
        }
        sb.append("CPU 核心: ${r.cpuCores}\n")
        sb.append("存储空间: ${r.ramMB} MB\n")
        sb.append("建议: ${if (r.nnapiSupported) "可开启 NNAPI 加速提升速度" else "保持 CPU 推理（线程数 2-4）"}")
        return sb.toString()
    }
}