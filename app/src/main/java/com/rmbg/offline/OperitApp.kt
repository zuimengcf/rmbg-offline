package com.rmbg.offline

import android.app.Application
import android.content.Context

/**
 * 应用入口：持有全局 Context（供 Prefs 等工具使用）
 */
class OperitApp : Application() {
    override fun onCreate() {
        super.onCreate()
        appContext = applicationContext
    }

    companion object {
        @Volatile
        lateinit var appContext: Context
    }
}