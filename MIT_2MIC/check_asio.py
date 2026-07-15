import sounddevice as sd

print("=== 检查 sounddevice 支持的 Host API ===")
print()

# 方法1：查看所有 Host API
print("所有可用的 Host API:")
hostapis = sd.query_hostapis()
for i, hostapi in enumerate(hostapis):
    print(f"  {i}: {hostapi['name']}")
    if 'ASIO' in hostapi['name'].upper():
        print(f"    ✓ 找到 ASIO 支持!")
print()

# 方法2：查看所有音频设备及其 Host API
print("所有音频设备及其 Host API:")
devices = sd.query_devices()
for i, device in enumerate(devices):
    hostapi_name = hostapis[device['hostapi']]['name']
    print(f"  {i}: {device['name']}")
    print(f"      Host API: {hostapi_name}")
    if 'ASIO' in hostapi_name.upper():
        print(f"      ✓ 这是 ASIO 设备!")
    print()

# 方法3：专门查找 ASIO 设备
print("=== 专门查找 ASIO 设备 ===")
asio_devices = []
for i, device in enumerate(devices):
    hostapi_name = hostapis[device['hostapi']]['name']
    if 'ASIO' in hostapi_name.upper():
        asio_devices.append((i, device, hostapi_name))

if asio_devices:
    print(f"找到 {len(asio_devices)} 个 ASIO 设备:")
    for idx, device, hostapi_name in asio_devices:
        print(f"  设备 {idx}: {device['name']}")
        print(f"    Host API: {hostapi_name}")
        print(f"    最大输入通道: {device['max_inputs']}")
        print(f"    最大输出通道: {device['max_outputs']}")
        print(f"    默认采样率: {device['default_samplerate']}")
else:
    print("❌ 未找到 ASIO 设备")
    print("\n可能的原因:")
    print("1. 系统未安装 ASIO 驱动")
    print("2. 音频设备不支持 ASIO")
    print("3. 需要安装 ASIO4ALL 等第三方驱动")
    print("4. 需要管理员权限运行")

print()
print("=== 检查 sounddevice 版本 ===")
import sounddevice
print(f"sounddevice 版本: {sounddevice.__version__}") 