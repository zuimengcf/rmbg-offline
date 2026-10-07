package com.rmbg.offline

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build

/**
 * 通知工具：抠图完成通知 + 模型下载进度通知 + 通知权限检查/申请
 */
object Notifications {

    private const val CHANNEL_DONE = "rmbg_done"
    private const val CHANNEL_DOWNLOAD = "rmbg_download"
    private const val NOTIF_DONE = 2
    private const val NOTIF_DOWNLOAD = 3

    /** 检查是否已授予通知权限（Android 13+） */
    fun hasPermission(context: Context): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            context.checkSelfPermission(android.Manifest.permission.POST_NOTIFICATIONS) ==
                PackageManager.PERMISSION_GRANTED
        } else true
    }

    /** 创建完成通知通道 */
    private fun ensureDoneChannel(context: Context) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val nm = context.getSystemService(NotificationManager::class.java)
            val ch = NotificationChannel(
                CHANNEL_DONE,
                "抠图完成",
                NotificationManager.IMPORTANCE_DEFAULT
            ).apply {
                description = "抠图处理完成提醒"
                setShowBadge(true)
            }
            nm.createNotificationChannel(ch)
        }
    }

    /** 创建下载进度通知通道（低打扰：进度条 + 完成提醒） */
    private fun ensureDownloadChannel(context: Context) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val nm = context.getSystemService(NotificationManager::class.java)
            val ch = NotificationChannel(
                CHANNEL_DOWNLOAD,
                "模型下载",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "模型下载进度与完成提醒"
                setShowBadge(false)
            }
            nm.createNotificationChannel(ch)
        }
    }

    /** 发送抠图完成通知 */
    fun notifyDone(context: Context) {
        if (!hasPermission(context)) return
        try {
            ensureDoneChannel(context)
            val contentIntent = PendingIntent.getActivity(
                context, 1,
                Intent(context, MainActivity::class.java),
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            )
            val builder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                Notification.Builder(context, CHANNEL_DONE)
            } else {
                Notification.Builder(context)
            }
            val n = builder
                .setContentTitle("抠图完成")
                .setContentText("点击查看结果")
                .setSmallIcon(android.R.drawable.ic_menu_gallery)
                .setContentIntent(contentIntent)
                .setAutoCancel(true)
                .build()
            context.getSystemService(NotificationManager::class.java).notify(NOTIF_DONE, n)
        } catch (_: Exception) {}
    }

    /**
     * ★ 模型下载进度通知（进度条 + 文案）。
     * 两种链路共用：抠图模型 / AI 重绘模型。
     * @param title 内容标题（如「下载抠图模型」/「下载 AI 重绘模型」）
     * @param downloaded 已下载字节
     * @param total 总字节（未知传 0，显示循环动画）
     * @param speed 下载速度 B/s（未知传 0）
     */
    fun notifyDownloadProgress(context: Context, title: String, downloaded: Long, total: Long, speed: Long) {
        if (!hasPermission(context)) return
        try {
            ensureDownloadChannel(context)
            val contentIntent = PendingIntent.getActivity(
                context, 2,
                Intent(context, MainActivity::class.java),
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            )
            val builder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                Notification.Builder(context, CHANNEL_DOWNLOAD)
            } else {
                Notification.Builder(context)
            }
            builder
                .setContentTitle(title)
                .setSmallIcon(android.R.drawable.stat_sys_download)
                .setContentIntent(contentIntent)
                .setOngoing(true)
                .setOnlyAlertOnce(true)
            if (total > 0 && downloaded >= 0) {
                // 确定进度：进度条 + 百分比文案
                val pct = (downloaded.toFloat() / total * 100).toInt().coerceIn(0, 100)
                val speedText = if (speed > 0) " · ${formatSize(speed)}/s" else ""
                builder
                    .setProgress(100, pct, false)
                    .setContentText("${formatSize(downloaded)} / ${formatSize(total)}（$pct%）$speedText")
            } else {
                // 未知总量：不确定进度（循环动画）
                builder
                    .setProgress(0, 0, true)
                    .setContentText("已下载 ${formatSize(downloaded)}${if (speed > 0) " · ${formatSize(speed)}/s" else ""}")
            }
            context.getSystemService(NotificationManager::class.java).notify(NOTIF_DOWNLOAD, builder.build())
        } catch (_: Exception) {}
    }

    /** 模型下载完成通知（点击进入模型页） */
    fun notifyDownloadDone(context: Context, title: String, text: String) {
        if (!hasPermission(context)) return
        try {
            ensureDownloadChannel(context)
            val contentIntent = PendingIntent.getActivity(
                context, 3,
                Intent(context, MainActivity::class.java),
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            )
            val builder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                Notification.Builder(context, CHANNEL_DOWNLOAD)
            } else {
                Notification.Builder(context)
            }
            val n = builder
                .setContentTitle(title)
                .setContentText(text)
                .setSmallIcon(android.R.drawable.stat_sys_download_done)
                .setContentIntent(contentIntent)
                .setAutoCancel(true)
                .build()
            context.getSystemService(NotificationManager::class.java).notify(NOTIF_DOWNLOAD, n)
        } catch (_: Exception) {}
    }

    /** 取消下载进度通知（下载失败/结束时清理） */
    fun cancelDownloadNotification(context: Context) {
        try {
            context.getSystemService(NotificationManager::class.java).cancel(NOTIF_DOWNLOAD)
        } catch (_: Exception) {}
    }

    /** 字节格式化（复用 MainActivity 的 formatSize 逻辑，避免跨类依赖） */
    private fun formatSize(bytes: Long): String {
        if (bytes <= 0) return "0 B"
        val units = arrayOf("B", "KB", "MB", "GB")
        var v = bytes.toDouble()
        var i = 0
        while (v >= 1024 && i < units.size - 1) { v /= 1024; i++ }
        return if (i == 0) "${bytes} ${units[i]}" else String.format("%.1f %s", v, units[i])
    }
}