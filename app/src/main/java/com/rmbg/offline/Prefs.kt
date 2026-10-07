package com.rmbg.offline

import android.content.Context
import android.content.SharedPreferences

/**
 * 应用设置持久化（SharedPreferences）
 * 防止：切模型后重启被重置、后台被杀后设置丢失。
 */
object Prefs {
    private const val NAME = "rmbg_settings"

    private fun sp(context: Context): SharedPreferences =
        context.getSharedPreferences(NAME, Context.MODE_PRIVATE)

    // ---- 模型 ----
    // ★ 默认模型两边统一：anime_seg（Anime-Seg，动漫抠图；QNN 版在 Lite 下自动回退）
    var selectedModelId: String
        get() = sp(OperitApp.appContext).getString("selected_model_id", "anime_seg") ?: "anime_seg"
        set(v) = sp(OperitApp.appContext).edit().putString("selected_model_id", v).apply()

    var hfRepo: String
        get() = sp(OperitApp.appContext).getString("hf_repo", "briaai/RMBG-1.4") ?: "briaai/RMBG-1.4"
        set(v) = sp(OperitApp.appContext).edit().putString("hf_repo", v).apply()

    var hfFile: String
        get() = sp(OperitApp.appContext).getString("hf_file", "onnx/model.onnx") ?: "onnx/model.onnx"
        set(v) = sp(OperitApp.appContext).edit().putString("hf_file", v).apply()

    var hfToken: String
        get() = sp(OperitApp.appContext).getString("hf_token", "") ?: ""
        set(v) = sp(OperitApp.appContext).edit().putString("hf_token", v).apply()

    /** 模型直链下载地址（自定义 ONNX 模型，URL 为空时退回 hfRepo/hfFile 仓库下载） */
    var modelUrl: String
        get() = sp(OperitApp.appContext).getString("model_url", "") ?: ""
        set(v) = sp(OperitApp.appContext).edit().putString("model_url", v).apply()

    var mirrorIndex: Int
        get() = sp(OperitApp.appContext).getInt("mirror_index", 0)
        set(v) = sp(OperitApp.appContext).edit().putInt("mirror_index", v).apply()

    // ---- 历史排序：0=时间最新 1=时间最旧 2=耗时最短 3=耗时最长 4=模型名 5=文件最小 6=文件最大 ----
    var historySort: Int
        get() = sp(OperitApp.appContext).getInt("history_sort", 0)
        set(v) = sp(OperitApp.appContext).edit().putInt("history_sort", v).apply()

    // ---- 历史视图：0=列表 1=网格 ----
    var historyView: Int
        get() = sp(OperitApp.appContext).getInt("history_view", 0)
        set(v) = sp(OperitApp.appContext).edit().putInt("history_view", v).apply()

    // ---- 历史折叠状态：已折叠的组 key（逗号分隔，空=全部展开）----
    var historyCollapsedGroups: String
        get() = sp(OperitApp.appContext).getString("history_collapsed", "") ?: ""
        set(v) = sp(OperitApp.appContext).edit().putString("history_collapsed", v).apply()

    // ---- 抠图参数 ----
    var threshold: Float
        get() = sp(OperitApp.appContext).getFloat("threshold", 0.5f)
        set(v) = sp(OperitApp.appContext).edit().putFloat("threshold", v).apply()

    var tileMode: Boolean
        get() = sp(OperitApp.appContext).getBoolean("tile_mode", true)
        set(v) = sp(OperitApp.appContext).edit().putBoolean("tile_mode", v).apply()

    /** 极速模式：缩小分辨率快速抠图（与精细抠图并存，不冲突） */
    var turboMode: Boolean
        get() = sp(OperitApp.appContext).getBoolean("turbo_mode", false)
        set(v) = sp(OperitApp.appContext).edit().putBoolean("turbo_mode", v).apply()

    /** 背景硬化阈值（alpha 下限 0~255，默认 192）：
     *  低于此值的半透明背景像素硬化为全透明(消除磨砂雾感)；
     *  值越高背景越干净，越低越保留半透明发丝等细节。 */
    var bgCleanAlpha: Int
        get() = sp(OperitApp.appContext).getInt("bg_clean_alpha", 192)
        set(v) = sp(OperitApp.appContext).edit().putInt("bg_clean_alpha", v.coerceIn(0, 255)).apply()

    // ---- ONNX 加速 ----
    var enableNnapi: Boolean
        get() = sp(OperitApp.appContext).getBoolean("enable_nnapi", false)
        set(v) = sp(OperitApp.appContext).edit().putBoolean("enable_nnapi", v).apply()

    var enableQnn: Boolean
        get() = sp(OperitApp.appContext).getBoolean("enable_qnn", false)
        set(v) = sp(OperitApp.appContext).edit().putBoolean("enable_qnn", v).apply()

    var onnxThreads: Int
        get() = sp(OperitApp.appContext).getInt("onnx_threads", 4)
        set(v) = sp(OperitApp.appContext).edit().putInt("onnx_threads", v).apply()

    /** NNAPI 检测结果缓存 */
    var nnapiDetectResult: String
        get() = sp(OperitApp.appContext).getString("nnapi_detect", "") ?: ""
        set(v) = sp(OperitApp.appContext).edit().putString("nnapi_detect", v).apply()

    /** 通知保活开关 */
    var keepAlive: Boolean
        get() = sp(OperitApp.appContext).getBoolean("keep_alive", true)
        set(v) = sp(OperitApp.appContext).edit().putBoolean("keep_alive", v).apply()

    /** 自动保存抠图结果（勾选后首次抠图即自动保存 原图→结果 到历史/相册，默认开） */
    var autoSaveResult: Boolean
        get() = sp(OperitApp.appContext).getBoolean("auto_save_result", true)
        set(v) = sp(OperitApp.appContext).edit().putBoolean("auto_save_result", v).apply()

    /** 查看器背景色（棋盘/白色/黑色/浅灰/绿色） */
    var viewerBgColor: String
        get() = sp(OperitApp.appContext).getString("viewer_bg_color", "棋盘") ?: "棋盘"
        set(v) = sp(OperitApp.appContext).edit().putString("viewer_bg_color", v).apply()

    // ---- 高级编辑位置记忆 ----
    var editorScale: Float
        get() = sp(OperitApp.appContext).getFloat("editor_scale", 1f)
        set(v) = sp(OperitApp.appContext).edit().putFloat("editor_scale", v).apply()

    var editorOffsetX: Float
        get() = sp(OperitApp.appContext).getFloat("editor_offset_x", 0f)
        set(v) = sp(OperitApp.appContext).edit().putFloat("editor_offset_x", v).apply()

    var editorOffsetY: Float
        get() = sp(OperitApp.appContext).getFloat("editor_offset_y", 0f)
        set(v) = sp(OperitApp.appContext).edit().putFloat("editor_offset_y", v).apply()

    var editorSelLeft: Float
        get() = sp(OperitApp.appContext).getFloat("editor_sel_left", 0.2f)
        set(v) = sp(OperitApp.appContext).edit().putFloat("editor_sel_left", v).apply()

    var editorSelTop: Float
        get() = sp(OperitApp.appContext).getFloat("editor_sel_top", 0.2f)
        set(v) = sp(OperitApp.appContext).edit().putFloat("editor_sel_top", v).apply()

    var editorSelRight: Float
        get() = sp(OperitApp.appContext).getFloat("editor_sel_right", 0.8f)
        set(v) = sp(OperitApp.appContext).edit().putFloat("editor_sel_right", v).apply()

    var editorSelBottom: Float
        get() = sp(OperitApp.appContext).getFloat("editor_sel_bottom", 0.8f)
        set(v) = sp(OperitApp.appContext).edit().putFloat("editor_sel_bottom", v).apply()

    // ---- AI 重绘预设（设置页配置，查看器一键执行）----
    // ★ 本功能是修复/微调（img2img），不是生图：提示词按修复向中性预设，denoise 默认低强度保持原图。
    //   精简原则：正向词只保留"保持原内容 + 提升清晰/质感"核心；负向词只排除画质缺陷与内容污染。
    var aiRedrawPrompt: String
        get() = sp(OperitApp.appContext).getString("ai_redraw_prompt", "keep original, sharp details, high quality") ?: "keep original, sharp details, high quality"
        set(v) = sp(OperitApp.appContext).edit().putString("ai_redraw_prompt", v).apply()

    var aiRedrawNegative: String
        get() = sp(OperitApp.appContext).getString("ai_redraw_negative", "blurry, low quality, watermark, deformed") ?: "blurry, low quality, watermark, deformed"
        set(v) = sp(OperitApp.appContext).edit().putString("ai_redraw_negative", v).apply()

    var aiRedrawSteps: Int
        get() = sp(OperitApp.appContext).getInt("ai_redraw_steps", 20)
        set(v) = sp(OperitApp.appContext).edit().putInt("ai_redraw_steps", v.coerceIn(1, 50)).apply()

    var aiRedrawCfg: Float
        get() = sp(OperitApp.appContext).getFloat("ai_redraw_cfg", 6.5f)
        set(v) = sp(OperitApp.appContext).edit().putFloat("ai_redraw_cfg", v.coerceIn(1f, 15f)).apply()

    /** 重绘强度：默认 0.35 微调修复（更贴原图结构，改动更小）；越高改动越大越接近重绘 */
    var aiRedrawDenoise: Float
        get() = sp(OperitApp.appContext).getFloat("ai_redraw_denoise", 0.35f)
        set(v) = sp(OperitApp.appContext).edit().putFloat("ai_redraw_denoise", v.coerceIn(0.1f, 1f)).apply()

    var aiRedrawWidth: Int
        get() = sp(OperitApp.appContext).getInt("ai_redraw_width", 512)
        set(v) = sp(OperitApp.appContext).edit().putInt("ai_redraw_width", v.coerceIn(256, 1024)).apply()

    var aiRedrawHeight: Int
        get() = sp(OperitApp.appContext).getInt("ai_redraw_height", 512)
        set(v) = sp(OperitApp.appContext).edit().putInt("ai_redraw_height", v.coerceIn(256, 1024)).apply()

    var aiRedrawScheduler: String
        get() = sp(OperitApp.appContext).getString("ai_redraw_scheduler", "dpm_sde") ?: "dpm_sde"
        set(v) = sp(OperitApp.appContext).edit().putString("ai_redraw_scheduler", v).apply()

    /** AI 重绘模型包 zip 持久化路径（App 重启不丢；部署成功后 zip 会删除，此路径仅记录来源） */
    var aiModelZipPath: String
        get() = sp(OperitApp.appContext).getString("ai_model_zip_path", "") ?: ""
        set(v) = sp(OperitApp.appContext).edit().putString("ai_model_zip_path", v).apply()

    /** 当前活动 AI 重绘模型 id（部署目录名；zip 删除后仍能定位模型，多模型切换用） */
    var aiRedrawModelId: String
        get() = sp(OperitApp.appContext).getString("ai_redraw_model_id", "") ?: ""
        set(v) = sp(OperitApp.appContext).edit().putString("ai_redraw_model_id", v).apply()

    // ---- AI 重绘模型 HF 云下载配置（云端下载入口：链接 + token）----
    /** HF resolve 直链（如 https://huggingface.co/xororz/sd-qnn/resolve/main/MeinaMixV12_qnn2.28_8gen2.zip） */
    var aiRedrawHfUrl: String
        get() = sp(OperitApp.appContext).getString("ai_redraw_hf_url", "") ?: ""
        set(v) = sp(OperitApp.appContext).edit().putString("ai_redraw_hf_url", v).apply()

    /** HF Token（下载 gated/私有仓库模型必需） */
    var aiRedrawHfToken: String
        get() = sp(OperitApp.appContext).getString("ai_redraw_hf_token", "") ?: ""
        set(v) = sp(OperitApp.appContext).edit().putString("ai_redraw_hf_token", v).apply()

    /**
     * AI 重绘推理后端：
     *  "local"  = 本地自加载引擎（LibStableDiffusionCore，本 App 内启动进程）
     *  "dream"  = 已安装的 LocalDream App（复用其 8081 HTTP 服务，App 自身不部署模型）
     */
    var aiRedrawBackend: String
        get() = sp(OperitApp.appContext).getString("ai_redraw_backend", "local") ?: "local"
        set(v) = sp(OperitApp.appContext).edit().putString("ai_redraw_backend", v).apply()
}
