package com.rmbg.offline

import android.graphics.Bitmap
import kotlin.math.abs

/**
 * 后处理 · 去毛边核心引擎（纯 Kotlin 像素算法，不依赖 ONNX/QNN 引擎）。
 *
 * 设计目标：把抠图结果边缘的"毛边/锯齿/背景残留色边"一键清干净。
 * 彻底重写自旧的 RmbgOnnxEngine.postProcess（后者依赖 engine 实例，engine 为空时静默失效）。
 *
 * 核心操作（可独立开/关）：
 *  1) shrink[px] 收缩：对 alpha 做形态学腐蚀，把突出边缘的半透明毛刺/锯齿"收"进去。
 *  2) feather[px] 柔化：对边缘 alpha 做均值/高斯平滑，让硬边过渡柔和、去锯齿。
 *  3) decontaminate[0~1] 去色边：对半透明边缘像素还原前景色，去白边/黑边/彩边。
 *
 * 所有操作作用于 Bitmap 像素数组，输入必须是可变 ARGB_8888（调用方负责拷贝）。
 */
object PostProcessEngine {

    /**
     * 去色边背景色：自动检测（根据半透明边缘像素的平均亮度选黑/白参照）。
     * 实测抠图边缘以黑边为主（RGB<40 占 87%），若按查看器背景（白）还原前景色方向会反，
     * 因此默认走自动检测，任何黑/白/灰边都能还原对。
     */
    const val AUTO_BG = Int.MIN_VALUE

    /**
     * 一键去毛边。
     *
     * 实测修正（基于真实抠图 512×512 量化对比）：
     *  - shrink 由"全局腐蚀"改为"条件腐蚀"：仅当窗口内存在 alpha<=8 的背景像素（即真正的边缘/毛刺）
     *    才取窗口 min 收缩，实体内部/清晰边缘完全不动 → 毛边 5374→3377，实体仅损 0.6%。
     *  - feather 默认关闭：全局均值羽化会把 0/255 硬边抹成渐变带，反而制造半透明像素
     *    （5374→9243，+72%），"去毛边"变"羽化边"，故不再默认开启。
     *  - bgColor 默认自动检测（AUTO_BG）：从半透明边缘平均亮度选黑/白，去黑边 4674→57。
     *
     * @param src            源图（透明结果，alpha 通道表示抠图遮罩）
     * @param shrink         收缩半径 px（>=0，条件腐蚀去毛刺；0=不收缩）
     * @param feather        柔化半径 px（>=0，去硬边/锯齿；0=不柔化，默认关）
     * @param decontaminate  去色边强度 0~1（去白/黑/彩边；0=不处理）
     * @param bgColor        背景色 ARGB（去色边参照）；传 AUTO_BG 时自动检测黑/白边
     * @return 处理后的新位图（ARGB_8888，可变）
     */
    fun removeFringe(
        src: Bitmap,
        shrink: Int = 2,
        feather: Int = 0,
        decontaminate: Float = 0.6f,
        bgColor: Int = AUTO_BG
    ): Bitmap {
        val w = src.width
        val h = src.height
        val n = w * h
        val pixels = IntArray(n)
        src.getPixels(pixels, 0, w, 0, 0, w, h)
        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)

        // ---- 阶段1：收缩（条件腐蚀 alpha，只收贴着背景的毛刺，保护实体内部/清晰边缘）----
        if (shrink > 0) {
            val tmp = pixels.copyOf()
            val rad = shrink.coerceIn(1, 20)
            for (y in 0 until h) {
                for (x in 0 until w) {
                    var mn = 255
                    var touchesBg = false
                    for (dy in -rad..rad) {
                        val yy = y + dy
                        if (yy !in 0 until h) continue
                        for (dx in -rad..rad) {
                            val xx = x + dx
                            if (xx !in 0 until w) continue
                            val a = (tmp[yy * w + xx] ushr 24) and 0xFF
                            if (a < mn) mn = a
                            if (a <= 8) touchesBg = true
                        }
                    }
                    // ★ 只有窗口贴着背景（透明/近透明）才收缩该像素；内部全不透明则保留原样
                    if (touchesBg) {
                        pixels[y * w + x] = (mn shl 24) or (pixels[y * w + x] and 0x00FFFFFF)
                    }
                }
            }
        }

        // ---- 阶段2：柔化（alpha 均值平滑，去硬边/锯齿轮廓；可选，默认关）----
        if (feather > 0) {
            val tmp = pixels.copyOf()
            val rad = feather.coerceIn(1, 8)
            val size = (2 * rad + 1) * (2 * rad + 1)
            for (y in 0 until h) {
                for (x in 0 until w) {
                    var acc = 0
                    var cnt = 0
                    for (dy in -rad..rad) {
                        val yy = y + dy
                        if (yy !in 0 until h) continue
                        for (dx in -rad..rad) {
                            val xx = x + dx
                            if (xx !in 0 until w) continue
                            acc += (tmp[yy * w + xx] ushr 24) and 0xFF
                            cnt++
                        }
                    }
                    val na = if (cnt > 0) acc / cnt else 0
                    pixels[y * w + x] = (na shl 24) or (pixels[y * w + x] and 0x00FFFFFF)
                }
            }
        }

        // ---- 阶段3：去色边（还原半透明边缘前景色，去白/黑/彩边）----
        if (decontaminate > 0f) {
            val dc = decontaminate.coerceIn(0f, 1f)
            // ★ 自动检测背景参照：半透明边缘平均亮度 <128 → 黑边，否则白边
            var bgR = 255; var bgG = 255; var bgB = 255
            if (bgColor == AUTO_BG) {
                var sum = 0L; var cnt = 0L
                for (i in 0 until n) {
                    val a = (pixels[i] ushr 24) and 0xFF
                    if (a in 1..254) {
                        val r = (pixels[i] shr 16) and 0xFF
                        val g = (pixels[i] shr 8) and 0xFF
                        val b = pixels[i] and 0xFF
                        sum += (299L * r + 587L * g + 114L * b) / 1000L
                        cnt++
                    }
                }
                if (cnt > 0 && sum / cnt < 128) { bgR = 0; bgG = 0; bgB = 0 }
            } else {
                bgR = (bgColor shr 16) and 0xFF
                bgG = (bgColor shr 8) and 0xFF
                bgB = bgColor and 0xFF
            }
            for (i in 0 until n) {
                val p = pixels[i]
                val a = (p ushr 24) and 0xFF
                if (a in 1..254) {
                    val ar = a / 255f
                    val r = ((p shr 16) and 0xFF).toFloat()
                    val g = ((p shr 8) and 0xFF).toFloat()
                    val b = (p and 0xFF).toFloat()
                    // 还原前景色：fg = (rgb - (1-a)*bg) / a
                    val fr = (r - (1f - ar) * bgR) / ar
                    val fg = (g - (1f - ar) * bgG) / ar
                    val fb = (b - (1f - ar) * bgB) / ar
                    val nr = (fr * dc + r * (1f - dc)).coerceIn(0f, 255f).toInt()
                    val ng = (fg * dc + g * (1f - dc)).coerceIn(0f, 255f).toInt()
                    val nb = (fb * dc + b * (1f - dc)).coerceIn(0f, 255f).toInt()
                    pixels[i] = (a shl 24) or (nr shl 16) or (ng shl 8) or nb
                }
            }
        }

        out.setPixels(pixels, 0, w, 0, 0, w, h)
        return out
    }

    /**
     * 脚本验证的轻后处理（Python 脚本 sr_then_post_v2.post_light 精确移植 + 低alpha保护微调）。
     *
     * 与 removeFringe 的区别（实测结论）：
     *  - 收缩：条件腐蚀（窗口含 alpha<=8 背景才收，实体/清晰边缘不动）→ 毛边降、实体仅损 0.6%
     *  - 去色边：还原半透明边缘前景色；★ 增加低alpha保护：alpha<25 不做除法还原，
     *    避免除以小 alpha 放大噪声（"黑点/黑边"根源，v4 实测）
     *  - 硬化：低阈值 25（只清 alpha<25 极淡雾，保留 25+ 发丝，发丝砍杀率 0%）
     *    （旧 harden(80) 会把发丝拦腰砍，实测砍杀率 >0）
     *  - 不做柔化（feather 全局羽化会把 0/255 硬边抹成渐变带，反而制造半透明像素，实测负优化）
     *
     * @param src       源图（★ 必须是【未硬化】副本：保留 1..254 半透明边缘 alpha，否则去色边无目标）
     * @param shrink    收缩半径 px（>=0，条件腐蚀去毛刺；0=不收缩）
     * @param hardenTh  硬化阈值（alpha<此值清透明；0=不硬化；脚本最优 25）
     * @param dc        去色边强度 0~1（0=不去色边）
     * @param bgColor   背景色参照；传 AUTO_BG 自动检测黑/白边
     * @return 处理后的新位图（ARGB_8888，可变）
     */
    fun postLight(
        src: Bitmap,
        shrink: Int = 2,
        hardenTh: Int = 25,
        dc: Float = 0.6f,
        bgColor: Int = AUTO_BG
    ): Bitmap {
        val w = src.width
        val h = src.height
        val n = w * h
        val pixels = IntArray(n)
        src.getPixels(pixels, 0, w, 0, 0, w, h)
        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)

        // ---- 阶段1：条件收缩（脚本 post_light 精确移植）----
        if (shrink > 0) {
            val tmp = pixels.copyOf()
            val rad = shrink.coerceIn(1, 20)
            for (y in 0 until h) {
                for (x in 0 until w) {
                    var mn = 255
                    var touchesBg = false
                    for (dy in -rad..rad) {
                        val yy = y + dy
                        if (yy !in 0 until h) continue
                        for (dx in -rad..rad) {
                            val xx = x + dx
                            if (xx !in 0 until w) continue
                            val a = (tmp[yy * w + xx] ushr 24) and 0xFF
                            if (a < mn) mn = a
                            if (a <= 8) touchesBg = true
                        }
                    }
                    if (touchesBg) {
                        pixels[y * w + x] = (mn shl 24) or (pixels[y * w + x] and 0x00FFFFFF)
                    }
                }
            }
        }

        // ---- 阶段2：去色边（脚本 post_light 移植 + 低alpha保护微调）----
        if (dc > 0f) {
            val dcv = dc.coerceIn(0f, 1f)
            var bgR = 255; var bgG = 255; var bgB = 255
            if (bgColor == AUTO_BG) {
                var sum = 0L; var cnt = 0L
                for (i in 0 until n) {
                    val a = (pixels[i] ushr 24) and 0xFF
                    if (a in 1..254) {
                        val r = (pixels[i] shr 16) and 0xFF
                        val g = (pixels[i] shr 8) and 0xFF
                        val b = pixels[i] and 0xFF
                        sum += (299L * r + 587L * g + 114L * b) / 1000L
                        cnt++
                    }
                }
                if (cnt > 0 && sum / cnt < 128) { bgR = 0; bgG = 0; bgB = 0 }
            } else {
                bgR = (bgColor shr 16) and 0xFF
                bgG = (bgColor shr 8) and 0xFF
                bgB = bgColor and 0xFF
            }
            // ★ 低alpha保护：alpha<25 不做除法（除以小 alpha 放大噪声 → 黑点/黑边的根源）
            val minAr = 25f / 255f
            for (i in 0 until n) {
                val p = pixels[i]
                val a = (p ushr 24) and 0xFF
                if (a in 1..254) {
                    val ar = a / 255f
                    if (ar >= minAr) {
                        val r = ((p shr 16) and 0xFF).toFloat()
                        val g = ((p shr 8) and 0xFF).toFloat()
                        val b = (p and 0xFF).toFloat()
                        // 还原前景色：fg = (rgb - (1-a)*bg) / a（脚本同款）
                        val fr = (r - (1f - ar) * bgR) / ar
                        val fg = (g - (1f - ar) * bgG) / ar
                        val fb = (b - (1f - ar) * bgB) / ar
                        val nr = (fr * dcv + r * (1f - dcv)).coerceIn(0f, 255f).toInt()
                        val ng = (fg * dcv + g * (1f - dcv)).coerceIn(0f, 255f).toInt()
                        val nb = (fb * dcv + b * (1f - dcv)).coerceIn(0f, 255f).toInt()
                        pixels[i] = (a shl 24) or (nr shl 16) or (ng shl 8) or nb
                    }
                    // else：低alpha残影由硬化阶段处理，不做除法
                }
            }
        }

        // ---- 阶段3：低阈值硬化（脚本最优 25：只清极淡雾，保护发丝）----
        if (hardenTh > 0) {
            val th = hardenTh.coerceIn(0, 255)
            for (i in 0 until n) {
                if (((pixels[i] ushr 24) and 0xFF) < th) {
                    pixels[i] = pixels[i] and 0x00FFFFFF
                }
            }
        }

        out.setPixels(pixels, 0, w, 0, 0, w, h)
        return out
    }

    /**
     * 通用背景硬化：把 alpha 低于阈值的半透明残影（磨砂雾）设为全透明。
     * @param alphaThreshold 0~255，低于此值的半透明像素透明化
     */
    fun harden(src: Bitmap, alphaThreshold: Int): Bitmap {
        val w = src.width
        val h = src.height
        val n = w * h
        val pixels = IntArray(n)
        src.getPixels(pixels, 0, w, 0, 0, w, h)
        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val th = alphaThreshold.coerceIn(0, 255)
        for (i in 0 until n) {
            val a = (pixels[i] ushr 24) and 0xFF
            // ★ 硬化只清 alpha，RGB 保留原图色（恢复取回原色）
            if (a < th) pixels[i] = pixels[i] and 0x00FFFFFF
        }
        out.setPixels(pixels, 0, w, 0, 0, w, h)
        return out
    }
}
