package com.rmbg.offline

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import com.rmbg.offline.BuildConfig

/**
 * 前台保活服务：常驻通知 + 唤醒锁，防止后台被系统杀死。
 * 抠图推理是重任务，后台时系统容易回收进程导致模型/结果丢失。
 * 通知带「退出」按钮：点击停止保活并退出 App。
 */
class KeepAliveService : Service() {

    companion object {
        private const val CHANNEL_ID = "rmbg_keepalive"
        private const val NOTIF_ID = 1
        const val ACTION_EXIT = "${BuildConfig.APPLICATION_ID}.action.EXIT_APP"

        fun start(context: Context) {
            val intent = Intent(context, KeepAliveService::class.java)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                context.startForegroundService(intent)
            } else {
                context.startService(intent)
            }
        }

        fun stop(context: Context) {
            context.stopService(Intent(context, KeepAliveService::class.java))
        }
    }

    private var wakeLock: PowerManager.WakeLock? = null

    override fun onCreate() {
        super.onCreate()
        createChannel()
        try {
            val notification = buildNotification()
            startForeground(NOTIF_ID, notification)
        } catch (e: Exception) {
            // 前台服务启动失败（如权限问题），保活服务降级为普通服务，不崩溃
            android.util.Log.w("KeepAliveService", "startForeground failed: ${e.message}")
            try { stopSelf() } catch (_: Exception) {}
            return
        }
        // 部分唤醒锁：保持 CPU 运行（抠图推理需要）
        try {
            val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
            wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "rmbg:keepalive").apply {
                setReferenceCounted(false)
                acquire()
            }
        } catch (_: Exception) {}
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        return START_STICKY
    }

    override fun onDestroy() {
        try { wakeLock?.release() } catch (_: Exception) {}
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun createChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "抠图服务",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "保持后台抠图任务不被系统回收"
                setShowBadge(false)
            }
            val nm = getSystemService(NotificationManager::class.java)
            nm.createNotificationChannel(channel)
        }
    }

    private fun buildNotification(): Notification {
        val contentIntent = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        // 「退出」按钮：广播 → 停止保活服务 + 结束 App 进程
        val exitIntent = PendingIntent.getBroadcast(
            this, 1,
            Intent(this, ExitReceiver::class.java).setAction(ACTION_EXIT),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        val builder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Notification.Builder(this, CHANNEL_ID)
        } else {
            Notification.Builder(this)
        }
        return builder
            .setContentTitle("RMBG 离线抠图")
            .setSmallIcon(android.R.drawable.ic_popup_sync)
            .setContentIntent(contentIntent)
            .setOngoing(true)
            .setPriority(Notification.PRIORITY_LOW)
            .addAction(android.R.drawable.ic_menu_close_clear_cancel, "退出", exitIntent)
            .build()
    }
}

/** 通知「退出」按钮接收器：停保活服务并退出 App */
class ExitReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        try {
            KeepAliveService.stop(context)
        } catch (_: Exception) {}
        try {
            Prefs.keepAlive = false
        } catch (_: Exception) {}
        // 结束 App 进程（保活服务已停，通知随之消失）
        try {
            android.os.Process.killProcess(android.os.Process.myPid())
        } catch (_: Exception) {}
    }
}