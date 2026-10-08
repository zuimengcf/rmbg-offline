import java.util.Properties
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

// ★ 签名配置（keystore.properties 不入库，.gitignore 忽略）
val keystoreProps = Properties().apply {
    val f = rootProject.file("keystore.properties")
    if (f.exists()) f.inputStream().use { load(it) }
}

android {
    namespace = "com.rmbg.offline"
    compileSdk = 35
    buildToolsVersion = "35.0.0"

    defaultConfig {
        applicationId = "com.rmbg.offline"
        minSdk = 27
        targetSdk = 35
        versionCode = 42
        versionName = "1.1.2"

        // ★ 是否为 Lite 精简版（包名 com.rmbg.offline.lite，模型不内置、走 HF 云下载）
        buildConfigField("boolean", "IS_LITE", "false")

        // 仅 arm64：QNN HTP 仅支持 64 位骁龙 DSP，缩小包体
        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    // ★ 双版本共存：normal=正式版（原包名，模型内置）；lite=精简版（包名 .lite，模型云下载）
    flavorDimensions += "edition"
    productFlavors {
        create("normal") {
            dimension = "edition"
            applicationId = "com.rmbg.offline"
            versionCode = 42
            versionName = "1.1.2"
            buildConfigField("boolean", "IS_LITE", "false")
        }
        create("lite") {
            dimension = "edition"
            applicationId = "com.rmbg.offline.lite"
            versionCode = 42
            versionName = "1.1.2"
            buildConfigField("boolean", "IS_LITE", "true")
        }
    }

    // ★★ 签名配置：release 用 keystore/rmbg-release.jks（别名 zuimeng，密码 zuimengqwq）
    signingConfigs {
        create("release") {
            if (keystoreProps.getProperty("storeFile") != null) {
                storeFile = rootProject.file(keystoreProps.getProperty("storeFile"))
                storePassword = keystoreProps.getProperty("storePassword")
                keyAlias = keystoreProps.getProperty("keyAlias")
                keyPassword = keystoreProps.getProperty("keyPassword")
            }
        }
    }

    buildTypes {
        debug {
            isDebuggable = true
            // ★ 测试版代码 + 正式签名：debuggable=true 可 run-as 诊断内部占用，
            //   签名用正式 keystore → 可与正式版互相覆盖安装（不用先卸载）
            signingConfig = signingConfigs.getByName("release")
        }
        release {
            isMinifyEnabled = false
            // ★ 两个 flavor 的 release 统一用 release 签名
            signingConfig = signingConfigs.getByName("release")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    lint {
        abortOnError = false
        checkReleaseBuilds = false
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
        // ★ QNN HTP 加载器必须从 ApplicationInfo.nativeLibraryDir 按路径发现
        //   backend/stub/skel 库 → 必须用 legacy JNI 打包（不压缩 native 库）
        // ★ 多设备兼容：保留 qnn-runtime AAR 的 QNN 库（V68~V81 skel/stub + GPU/DSP），
        //   不同骁龙 SoC 各需对应版本（SM8350=V68 / SM8450=V69 / SM8550=V73 / SM8650=V75 / SM8750=V79 / SM8850=V81）
        // ★ 排除 libQnnHtpPrepare.so（85MB）：
        //   - 仅用于"在线编译"（把 ONNX 图实时编译成 HTP 二进制），走 hexagon_nn 工具链
        //   - 本 App 全部 QNN 模型都是离线 EPContext（.bin），推理走 QnnContext_createFromBinary，
        //     不需要 Prepare；缺失时 ORT QNN EP 会 fallback 到 HNRD user-driver 路径（已有降级日志）
        //   - 排除后 APK 直接减 85MB（105MB → ~20MB），安装占用同步减少
        jniLibs {
            useLegacyPackaging = true
            // ★ 去掉 85MB 的 HtpPrepare（离线 EPContext 不需要在线编译工具链）
            excludes += "**/libQnnHtpPrepare.so"
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation(platform("androidx.compose:compose-bom:2024.10.01"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")

    // ★ 核心：Android 端 ONNX Runtime（本地离线推理；1.20.0+ 支持 QNN EP）
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.30.0")

    // ★★ QNN EP 官方 Android 部署三件套（参考 DakeQQ Tutorial / ORT 官方文档）：
    //   1. ORT 核心（上面）
    //   2. QNN EP 插件（libonnxruntime_providers_qnn.so + Kotlin 助手）
    //   3. QNN 运行时（GPU/HTP/System/DSP + v68-v81 stub/skel 全套 19 个库）
    //   ⚠ 三者必须显式声明：插件 AAR 的 POM 不会传递引入 ORT 核心或 QNN 运行时
    implementation("com.qualcomm.qti:onnxruntime-android-qnn:2.5.0")
    // ★★ QNN 运行时库：全面拥抱 2.50（qnn-runtime AAR 2.50.0 自带全套 V68~V81 + Htp + System）。
    //   ★ 匹配性：模型 bin 用 QNN SDK 2.50.0.260828221209 编译（HentaiGapeMix V10 等新包），
    //     runtime 必须 2.50 才能加载 EPContext（2.39 会报 ORT_NOT_IMPLEMENTED）。
    //   不再用 flavor 源集直装 2.39 .so；qnn-runtime AAR 直接进 nativeLibraryDir。
    implementation("com.qualcomm.qti:qnn-runtime:2.50.0")

    // 协程
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")

    // 图片加载
    implementation("io.coil-kt:coil-compose:2.7.0")

    // OkHttp（模型下载用）
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
}

// =============================================================
// ★ normal（完整版）构建时自动下载 QNN EPContext bin
// -------------------------------------------------------------
// QNN 模型二进制（.bin）约 197MB，不入 git 仓库（.gitignore 忽略）。
// 构建 normal 版 APK 时若 `app/src/normal/assets/qnn/` 下缺少 .bin，
// 自动从 HF 镜像下载（hf-mirror.com 国内优先，huggingface.co 兜底），
// 支持断点续传与文件大小校验；下载失败则中断构建并提示。
// Lite（精简版）不内置模型，构建时跳过本步骤。
//
// 手动跳过：./gradlew assembleNormalRelease -PskipBinDownload
// =============================================================
data class QnnBinSpec(
    val name: String,          // 目标文件名（与 assets/qnn/*.onnx 同名）
    val sizeBytes: Long,       // 用于完整性校验
)

val QNN_BINS = listOf(
    QnnBinSpec("qnn_rmbg14_sdk250_v2.bin", 99_079_792L),
    QnnBinSpec("qnn_animeseg_v73.bin", 90_789_176L),
    QnnBinSpec("qnn_modnet_sm8550_512.bin", 15_930_872L),
)

// 与 ModelManager 镜像源保持一致
val QNN_BIN_MIRRORS = listOf(
    "https://hf-mirror.com",
    "https://huggingface.co",
)

// repo / 文件名映射
fun qnnBinSource(name: String): Pair<String, String> = when (name) {
    "qnn_rmbg14_sdk250_v2.bin" -> "zuimengqm/rmbg-qnn-rmbg14" to "qnn/qnn_rmbg14_sdk250_v2.bin"
    "qnn_animeseg_v73.bin" -> "zuimengqm/rmbg-qnn-animeseg" to "qnn/qnn_animeseg_v73.bin"
    "qnn_modnet_sm8550_512.bin" -> "zuimengqm/rmbg-qnn-modnet" to "qnn/qnn_modnet_sm8550_512.bin"
    else -> error("未知 QNN bin: $name")
}

fun ensureQnnBins() {
    val dir = rootProject.file("app/src/normal/assets/qnn")
    dir.mkdirs()
    for (spec in QNN_BINS) {
        val target = File(dir, spec.name)
        if (target.exists() && target.length() == spec.sizeBytes) {
            println("✓ QNN bin 已就绪: ${spec.name}")
            continue
        }
        val (repo, srcPath) = qnnBinSource(spec.name)
        var downloaded = if (target.exists()) target.length() else 0L
        var ok = false
        var lastErr: Exception? = null
        for (base in QNN_BIN_MIRRORS) {
            val url = URL("$base/$repo/resolve/main/$srcPath")
            var conn: HttpURLConnection? = null
            try {
                println("↓ 下载 ${spec.name} (${spec.sizeBytes / 1048576}MB) <- $base ${if (downloaded > 0) "[续传 ${downloaded / 1048576}MB]" else ""}")
                conn = url.openConnection() as HttpURLConnection
                conn.connectTimeout = 30_000
                conn.readTimeout = 120_000
                conn.instanceFollowRedirects = true
                if (downloaded > 0) conn.setRequestProperty("Range", "bytes=$downloaded-")
                conn.connect()
                val code = conn.responseCode
                if (code !in 200..299 && code != 206) {
                    lastErr = RuntimeException("HTTP $code")
                    conn.disconnect()
                    continue
                }
                val input = conn.inputStream
                val out = FileOutputStream(target, downloaded > 0)
                val buf = ByteArray(1 shl 16)
                var total = downloaded
                var n: Int
                while (input.read(buf).also { n = it } != -1) {
                    out.write(buf, 0, n)
                    total += n
                }
                out.close()
                input.close()
                conn.disconnect()
                if (target.length() != spec.sizeBytes) {
                    lastErr = RuntimeException("大小校验失败: 期望 ${spec.sizeBytes} 实际 ${target.length()}")
                    downloaded = target.length() // 下次续传
                    continue
                }
                ok = true
                println("✓ ${spec.name} 下载完成")
                break
            } catch (e: Exception) {
                lastErr = e
                println("  ✗ $base 失败: ${e.message}")
                try { conn?.disconnect() } catch (_: Exception) {}
            }
        }
        if (!ok) throw GradleException("QNN bin 下载失败: ${spec.name} -> ${lastErr?.message}")
    }
}

tasks.register("downloadQnnBins") {
    group = "build"
    description = "下载 normal 版 QNN EPContext bin（hf-mirror 优先，官方兜底）"
    onlyIf {
        if (project.hasProperty("skipBinDownload")) {
            println("  (已通过 -PskipBinDownload 跳过 QNN bin 下载)")
            false
        } else {
            // 仅 normal flavor 的构建任务触发（任务名含 Normal，如 assembleNormalRelease / compileNormalDebugKotlin）
            val names = gradle.startParameter.taskNames.joinToString(" ")
            val isNormal = names.contains("Normal")
            if (!isNormal) println("  (lite/非 normal 构建，跳过 QNN bin 下载)")
            isNormal
        }
    }
    doLast { ensureQnnBins() }
}

// 构建 normal 版时，preBuild 阶段先确保 bin 就绪
tasks.configureEach {
    if (name == "preBuild") {
        dependsOn("downloadQnnBins")
    }
}