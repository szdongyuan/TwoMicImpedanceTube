import sounddevice as sd
import sys

def check_asio_support():
    """检查 PortAudio 是否支持 ASIO"""
    print("=== PortAudio ASIO 支持检查 ===\n")
    
    # 获取所有主机 API
    hostapis = sd.query_hostapis()
    print("可用的主机 API:")
    for i, api in enumerate(hostapis):
        print(f"  {i}: {api['name']}")
    
    print("\n=== 检查是否有 ASIO 主机 API ===")
    asio_found = False
    for i, api in enumerate(hostapis):
        if 'ASIO' in api['name'].upper():
            asio_found = True
            print(f"✓ 找到 ASIO 主机 API: {api['name']} (索引: {i})")
            print(f"  默认输入设备: {api.get('defaultInputDevice', 'N/A')}")
            print(f"  默认输出设备: {api.get('defaultOutputDevice', 'N/A')}")
            print(f"  设备数量: {api.get('deviceCount', 'N/A')}")
    
    if not asio_found:
        print("✗ 未找到 ASIO 主机 API")
    
    print("\n=== 所有音频设备 ===")
    devices = sd.query_devices()
    asio_devices = []
    
    for i, device in enumerate(devices):
        hostapi = hostapis[device['hostapi']]['name']
        if 'ASIO' in hostapi.upper():
            asio_devices.append((i, device, hostapi))
            print(f"  {i}: {device['name']} (ASIO)")
        else:
            print(f"  {i}: {device['name']} ({hostapi})")
    
    print(f"\n=== ASIO 设备统计 ===")
    print(f"ASIO 设备数量: {len(asio_devices)}")
    
    if asio_devices:
        print("ASIO 设备列表:")
        for idx, device, hostapi in asio_devices:
            print(f"  {idx}: {device['name']}")
            print(f"     最大输入通道: {device.get('maxInputChannels', 'N/A')}")
            print(f"     最大输出通道: {device.get('maxOutputChannels', 'N/A')}")
            print(f"     默认采样率: {device.get('defaultSampleRate', 'N/A')}")
    
    # 检查默认设备
    print(f"\n=== 默认设备 ===")
    try:
        default_input = sd.default.device[0]
        default_output = sd.default.device[1]
        print(f"默认输入设备: {default_input}")
        print(f"默认输出设备: {default_output}")
        
        if default_input is not None:
            input_device = devices[default_input]
            input_hostapi = hostapis[input_device['hostapi']]['name']
            print(f"默认输入设备主机 API: {input_hostapi}")
        
        if default_output is not None:
            output_device = devices[default_output]
            output_hostapi = hostapis[output_device['hostapi']]['name']
            print(f"默认输出设备主机 API: {output_hostapi}")
            
    except Exception as e:
        print(f"获取默认设备时出错: {e}")
    
    # 总结
    print(f"\n=== 总结 ===")
    if asio_found:
        print("✓ PortAudio 库编译时启用了 ASIO 支持")
        print(f"✓ 找到 {len(asio_devices)} 个 ASIO 设备")
    else:
        print("✗ PortAudio 库编译时未启用 ASIO 支持")
        print("  可能的原因:")
        print("  1. PortAudio 编译时未包含 ASIO 驱动")
        print("  2. 系统中未安装 ASIO 驱动")
        print("  3. 音频设备不支持 ASIO")

if __name__ == "__main__":
    check_asio_support() 