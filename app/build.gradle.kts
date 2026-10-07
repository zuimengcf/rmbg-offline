import java.util.Properties

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
        versionCode = 41
        versionName = "1.1.1"

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
            versionCode = 41
            versionName = "1.1.1"
            buildConfigField("boolean", "IS_LITE", "false")
        }
        create("lite") {
            dimension = "edition"
            applicationId = "com.rmbg.offline.lite"
            versionCode = 41
            versionName = "1.1.1"
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
        jniLibs {
            useLegacyPackaging = true
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
    // ★★ qnn-runtime 用 2.50.0：EPContext bin 是 QNN SDK 2.50.0.260828221209 编译的
    //   （见 qnn_rmbg14_sm8550.onnx 的 ep_sdk_version 属性），runtime 必须同版本才能加载 context binary
    implementation("com.qualcomm.qti:qnn-runtime:2.50.0")

    // 协程
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")

    // 图片加载
    implementation("io.coil-kt:coil-compose:2.7.0")

    // OkHttp（模型下载用）
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
}