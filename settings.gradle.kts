// 阿里云 Maven 镜像仅本地构建启用（国内加速）。
// GitHub Actions 等海外 CI 访问阿里云镜像常返回 502，自动跳过、直连官方源。
// 注：pluginManagement 是 Gradle 前置编译块，不能引用文件顶层声明，故两处均内联判断。

pluginManagement {
    repositories {
        if (System.getenv("CI")?.equals("true", ignoreCase = true) != true) {
            maven(url = "https://maven.aliyun.com/repository/central")
            maven(url = "https://maven.aliyun.com/repository/google")
        }
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        if (System.getenv("CI")?.equals("true", ignoreCase = true) != true) {
            maven(url = "https://maven.aliyun.com/repository/central")
            maven(url = "https://maven.aliyun.com/repository/google")
        }
        google()
        mavenCentral()
    }
}
rootProject.name = "RMBGOffline"
include(":app")