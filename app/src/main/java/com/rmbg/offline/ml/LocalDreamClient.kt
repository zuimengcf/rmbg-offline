package com.rmbg.offline.ml

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Base64
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.net.HttpURLConnection
import java.net.URL

/**
 * LocalDream 本地生图 HTTP 客户端（img2img 重绘）。
 *
 * LocalDream App 内置 HTTP 服务器（默认 http://127.0.0.1:8081），提供：
 *   GET  /        健康检查（200=在线）
 *   POST /generate  生图（txt2img / img2img / inpaint），SSE 流响应
 *
 * img2img 请求体（JSON）：
 *   prompt, negative_prompt, steps, cfg, width, height, scheduler,
 *   denoise_strength, seed, image(base64), mask(可选)
 *
 * SSE 响应事件（data: <json>）：
 *   {"type":"complete","image":"<base64 RGB 像素>","width":W,"height":H,
 *    "seed":N,"generation_time_ms":T}
 *   {"type":"error","message":"..."}
 */
object LocalDreamClient {

    /** LocalDream 默认 API 地址（本机回环） */
    const val DEFAULT_API_URL = "http://127.0.0.1:8081"

    /** 重绘结果 */
    data class RedrawResult(
        val bitmap: Bitmap,
        val width: Int,
        val height: Int,
        val seed: Long,
        val generationTimeMs: Long
    )

    /** 重绘请求参数 */
    data class RedrawRequest(
        val prompt: String,
        val negativePrompt: String = "",
        val steps: Int = 20,
        val cfg: Float = 7.0f,
        val width: Int = 512,
        val height: Int = 512,
        val scheduler: String = "dpm_sde",
        val denoiseStrength: Float = 0.6f,
        val seed: Long? = null,
        val imageBase64: String
    )

    /** 检查 LocalDream 是否在线（GET /） */
    suspend fun checkOnline(apiUrl: String = DEFAULT_API_URL): Boolean = withContext(Dispatchers.IO) {
        try {
            val conn = URL(apiUrl).openConnection() as HttpURLConnection
            conn.requestMethod = "GET"
            conn.connectTimeout = 3000
            conn.readTimeout = 3000
            val code = conn.responseCode
            conn.disconnect()
            code in 200..499
        } catch (_: Exception) { false }
    }

    /**
     * 调用 /generate 做 img2img 重绘。
     * 支持超时（LocalDream 生成一张通常 10~60 秒）。
     */
    suspend fun redraw(
        req: RedrawRequest,
        apiUrl: String = DEFAULT_API_URL,
        timeoutMs: Int = 600_000
    ): RedrawResult = withContext(Dispatchers.IO) {
        val json = JSONObject().apply {
            put("prompt", req.prompt)
            put("negative_prompt", req.negativePrompt)
            put("steps", req.steps.coerceIn(1, 100))
            put("cfg", req.cfg.coerceIn(1f, 30f))
            put("width", req.width)
            put("height", req.height)
            put("scheduler", req.scheduler)
            put("denoise_strength", req.denoiseStrength.coerceIn(0f, 1f))
            put("image", req.imageBase64)
            req.seed?.let { put("seed", it) }
        }

        val conn = URL(apiUrl + "/generate").openConnection() as HttpURLConnection
        try {
            conn.requestMethod = "POST"
            conn.connectTimeout = 10_000
            conn.readTimeout = timeoutMs
            conn.setRequestProperty("Content-Type", "application/json")
            conn.setRequestProperty("Accept", "text/event-stream")
            conn.doOutput = true
            conn.outputStream.use { it.write(json.toString().toByteArray(Charsets.UTF_8)) }

            val code = conn.responseCode
            if (code !in 200..299) {
                throw RuntimeException("LocalDream API 返回 $code")
            }

            // 读取 SSE 流，解析 complete / error 事件
            var imageB64: String? = null
            var width = 0
            var height = 0
            var seed = 0L
            var genMs = 0L
            var errorMsg: String? = null

            conn.inputStream.bufferedReader(Charsets.UTF_8).use { reader ->
                while (true) {
                    val line = reader.readLine() ?: break
                    if (!line.startsWith("data: ")) continue
                    val data = line.substring(6).trim()
                    if (data.isEmpty()) continue
                    try {
                        val ev = JSONObject(data)
                        when (ev.optString("type")) {
                            "complete" -> {
                                imageB64 = ev.optString("image")
                                width = ev.optInt("width")
                                height = ev.optInt("height")
                                seed = ev.optLong("seed")
                                genMs = ev.optLong("generation_time_ms")
                            }
                            "error" -> { errorMsg = ev.optString("message"); break }
                        }
                    } catch (_: Exception) {}
                }
            }

            errorMsg?.let { throw RuntimeException("LocalDream: $it") }
            val b64 = imageB64 ?: throw RuntimeException("LocalDream 未返回图片")

            // 解码 RGB 像素 → Bitmap（LocalDream 返回的是裸 RGB 字节，BGR 顺序见插件 r2b 转换）
            val bytes = Base64.decode(b64, Base64.DEFAULT)
            val bmp = rgbToBitmap(bytes, width, height)
            RedrawResult(bmp, width, height, seed, genMs)
        } finally {
            conn.disconnect()
        }
    }

    /** 把 Bitmap 编码为 PNG base64（img2img 的 image 字段） */
    fun bitmapToPngBase64(bmp: Bitmap): String {
        val bos = ByteArrayOutputStream()
        bmp.compress(Bitmap.CompressFormat.PNG, 100, bos)
        return Base64.encodeToString(bos.toByteArray(), Base64.NO_WRAP)
    }

    /** RGB 裸像素 → Bitmap（LocalDream complete 事件返回的 image 是 RGB 顺序裸数据） */
    private fun rgbToBitmap(rgb: ByteArray, w: Int, h: Int): Bitmap {
        if (w <= 0 || h <= 0 || rgb.size < w * h * 3) {
            throw RuntimeException("LocalDream 图片尺寸异常: ${w}x${h} bytes=${rgb.size}")
        }
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(w * h)
        var idx = 0
        for (i in pixels.indices) {
            val r = rgb[idx].toInt() and 0xFF
            val g = rgb[idx + 1].toInt() and 0xFF
            val b = rgb[idx + 2].toInt() and 0xFF
            idx += 3
            pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        bmp.setPixels(pixels, 0, w, 0, 0, w, h)
        return bmp
    }

    /** 备用：用 BitmapFactory 尝试直接解码（若返回的是标准图像编码如 PNG/JPEG） */
    fun decodeImageBytes(bytes: ByteArray): Bitmap? =
        BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
}
