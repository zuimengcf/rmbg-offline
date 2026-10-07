# QNN (Hexagon HTP DSP) 加速所需库

将以下 QNN SDK 库放入本目录（arm64-v8a/）即可启用 QNN 硬件加速：

| 文件 | 来源 | 作用 |
|------|------|------|
| libQnnHtp.so | QNN SDK lib/aarch64-android | QNN HTP Host 端入口（backend_path 指向它） |
| libQnnHtpV77Skel.so | QNN SDK lib/hexagon-v77 | 骁龙 8 Gen2 (SM8550) Hexagon V77 DSP 端 |
| libQnnHtpV77Stub.so | QNN SDK lib/aarch64-android | DSP stub（可选，SDK 有时自动生成） |
| libQnnSystem.so | QNN SDK lib/aarch64-android | QNN 系统库（依赖） |

注意：
- QNN SDK 需 Qualcomm 账号下载（qpm.qualcomm.com），Android 版 aarch64-android + hexagon-v77
- 也可用 Qualcomm AI Hub 提供的 qnn sdk 包
- 设备已有 /vendor 的 libcdsprpc.so（FastRPC 通道），无需打包
- 放置后 QNN 开关打开即用，无需其他代码改动
