package com.rmbg.offline
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.SystemClock
import java.util.concurrent.atomic.AtomicLong
/**
 * 下载前台服务：模型下载期间保持进程存活（App 退后台也能继续下载）。
 *
 * 职责：
 *  - 下载开始时以前台服务启动（dataSync 类型），系统通知栏出现「下载中」常驻通知；
 *  - 下载进行中通过 updateProgress 刷新通知为进度条；
 *  - 下载结束/失败/取消时 stop() 停止前台服务，通知转为可消除或消失。
 *
 * 通知复用 Notifications 的下载通道（CHANNEL_DOWNLOAD / NOTIF_DOWNLOAD=3），
 * 与下载进度/完成通知共用一个 ID，避免通知栏重复。
 */
class DownloadService : Service() {
    companion object {
        private const val CHANNEL_ID = "rmbg_download"
        const val NOTIF_ID = 3
        // 由 Notifications 统一管理，这里仅引用，避免魔法数分散
        /** 前台服务保活最大空闲时长（毫秒）：超过后若无下载活动则自动停止，避免 dataSync 超时崩溃 */
        private const val IDLE_TIMEOUT_MS = 10 * 60 * 1000L // 10 分钟
        // 进程级最近一次活动时间（SystemClock.elapsedRealtime），供 kick/自检使用
        private val lastActivity = AtomicLong(0L)

        /** 启动下载前台服务（幂等：已在运行则忽略） */
        fun start(context: Context) {
            val intent = Intent(context, DownloadService::class.java)
            // ★★ 崩溃根治：Android 8+ 用 startForegroundService 后必须在窗口内 startForeground()，
            //   否则抛 ForegroundServiceDidNotStartInTimeException。若通知权限未授予，
            //   startForeground 会抛 SecurityException → 触发该崩溃。
            //   因此无权限时【改用普通 startService】（前台 Activity 发起，系统允许），
            //   下载仍可继续，只是无保活通知（onCreate 内 startForeground 也会安全跳过）。
            val hasPerm = Notifications.hasPermission(context)
            try {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O && hasPerm) {
                    context.startForegroundService(intent)
                } else {
                    context.startService(intent)
                }
            } catch (_: Exception) {
                // 前台服务启动失败（如权限问题）→ 下载继续，只是无保活通知
                android.util.Log.w("DownloadService", "startForegroundService failed")
            }
        }

        /** 下载心跳：每次有下载进度/活动时调用，重置超时计时（服务已在前台，仅刷新时间戳） */
        fun kick(context: Context?) {
            lastActivity.set(SystemClock.elapsedRealtime())
        }

        /** 停止下载前台服务 */
        fun stop(context: Context) {
            try {
                context.stopService(Intent(context, DownloadService::class.java))
            } catch (_: Exception) {}
        }
    }

    private val handler = Handler(Looper.getMainLooper())
    private val timeoutRunnable = Runnable { checkIdleAndSelfStop() }

    override fun onCreate() {
        super.onCreate()
        createChannel()
        lastActivity.set(SystemClock.elapsedRealtime())
        // ★★ startForeground 失败（如通知权限被拒/系统限制）时不 stopSelf：
        //   本服务可能由【普通 startService】启动（无权限降级路径），此时无需前台，
        //   若 stopSelf 反而会让"已用 startForegroundService 启动但未 startForeground"的
        //   状态触发 ForegroundServiceDidNotStartInTimeException 崩溃。
        //   降级为普通后台服务继续下载即可（保活通知可选，不强求）。
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O && Notifications.hasPermission(this)) {
                startForeground(NOTIF_ID, buildNotification("正在准备下载模型…", 0, 0, 0))
            }
        } catch (e: Exception) {
            android.util.Log.w("DownloadService", "startForeground failed (degrade to normal): ${e.message}")
        }
        scheduleTimeout()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // 任何 start/kick 都视为一次活动，刷新心跳
        lastActivity.set(SystemClock.elapsedRealtime())
        scheduleTimeout()
        return START_NOT_STICKY
    }

    /** 若空闲超过 IDLE_TIMEOUT_MS 则自动停止（自杀），避免 dataSync 超时被系统强杀崩溃 */
    private fun checkIdleAndSelfStop() {
        val now = SystemClock.elapsedRealtime()
        val idle = now - lastActivity.get()
        if (idle >= IDLE_TIMEOUT_MS) {
            android.util.Log.i("DownloadService", "下载空闲超过 ${IDLE_TIMEOUT_MS / 60000} 分钟，自动停止前台服务")
            try { stopForeground(true) } catch (_: Exception) {}
            try { stopSelf() } catch (_: Exception) {}
        } else {
            scheduleTimeout()
        }
    }

    private fun scheduleTimeout() {
        handler.removeCallbacks(timeoutRunnable)
        handler.postDelayed(timeoutRunnable, IDLE_TIMEOUT_MS)
    }

    override fun onDestroy() {
        handler.removeCallbacks(timeoutRunnable)
        // ★ 保留通知：stopForeground(false) 不移除下载/完成通知（通知由 Notifications 统一接管），
        //   避免前台服务销毁时把刚发出的完成通知也清掉。
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                stopForeground(false)
            } else {
                @Suppress("DEPRECATION")
                stopForeground(false)
            }
        } catch (_: Exception) {}
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun createChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "模型下载",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "模型下载进度与完成提醒"
                setShowBadge(false)
            }
            val nm = getSystemService(NotificationManager::class.java)
            nm.createNotificationChannel(channel)
        }
    }

    private fun buildNotification(title: String, downloaded: Long, total: Long, speed: Long): Notification {
        val contentIntent = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        val builder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Notification.Builder(this, CHANNEL_ID)
        } else {
            Notification.Builder(this)
        }
        builder
            .setContentTitle(title)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentIntent(contentIntent)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
        if (total > 0 && downloaded >= 0) {
            val pct = (downloaded.toFloat() / total * 100).toInt().coerceIn(0, 100)
            builder.setProgress(100, pct, false)
                .setContentText("${formatSize(downloaded)} / ${formatSize(total)}（$pct%）")
        } else {
            builder.setProgress(0, 0, true)
                .setContentText("已下载 ${formatSize(downloaded)}")
        }
        return builder.build()
    }

    private fun formatSize(bytes: Long): String {
        if (bytes <= 0) return "0 B"
        val units = arrayOf("B", "KB", "MB", "GB")
        var v = bytes.toDouble()
        var i = 0
        while (v >= 1024 && i < units.size - 1) { v /= 1024; i++ }
        return if (i == 0) "${bytes} ${units[i]}" else String.format("%.1f %s", v, units[i])
    }
}