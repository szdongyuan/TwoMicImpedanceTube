import sounddevice as sd
import sys
import os

def check_behringer_driver():
    """检查百灵达设备的驱动状态"""
    print("=== 百灵达设备驱动检查 ===\n")
    
    # 查找百灵达设备
    devices = sd.query_devices()
    behringer_devices = []
    
    print("查找百灵达设备...")
    for i, device in enumerate(devices):
        if 'BEHRINGER' in device['name'].upper() or 'UMC' in device['name'].upper():
            hostapi = sd.query_hostapis()[device['hostapi']]['name']
            behringer_devices.append((i, device, hostapi))
            print(f"✓ 找到百灵达设备: {device['name']}")
            print(f"  设备索引: {i}")
            print(f"  主机 API: {hostapi}")
            print(f"  最大输入通道: {device.get('maxInputChannels', 'N/A')}")
            print(f"  最大输出通道: {device.get('maxOutputChannels', 'N/A')}")
            print(f"  默认采样率: {device.get('defaultSampleRate', 'N/A')}")
            print()
    
    if not behringer_devices:
        print("✗ 未找到百灵达设备")
        return
    
    print(f"找到 {len(behringer_devices)} 个百灵达设备")
    
    # 检查是否有 ASIO 支持
    print("\n=== 检查 ASIO 支持 ===")
    asio_devices = []
    for idx, device, hostapi in behringer_devices:
        if 'ASIO' in hostapi.upper():
            asio_devices.append((idx, device, hostapi))
            print(f"✓ 设备 {idx} 支持 ASIO: {device['name']}")
    
    if not asio_devices:
        print("✗ 当前没有百灵达设备支持 ASIO")
        print("需要安装百灵达 ASIO 驱动")
    
    # 检查百灵达驱动文件
    print("\n=== 检查百灵达驱动文件 ===")
    behringer_paths = [
        r"C:\Program Files\BEHRINGER",
        r"C:\Program Files (x86)\BEHRINGER",
        r"C:\Program Files\ASIO4ALL",
        r"C:\Program Files (x86)\ASIO4ALL"
    ]
    
    for path in behringer_paths:
        if os.path.exists(path):
            print(f"✓ 找到百灵达相关目录: {path}")
            try:
                files = os.listdir(path)
                for file in files:
                    if 'asio' in file.lower() or 'behringer' in file.lower():
                        print(f"  找到相关文件: {file}")
            except Exception as e:
                print(f"  读取目录时出错: {e}")
        else:
            print(f"✗ 未找到: {path}")
    
    # 检查注册表中的百灵达信息
    print("\n=== 检查注册表中的百灵达信息 ===")
    try:
        import winreg
        behringer_keys = [
            r"SOFTWARE\BEHRINGER",
            r"SOFTWARE\WOW6432Node\BEHRINGER",
            r"SOFTWARE\ASIO\BEHRINGER",
            r"SOFTWARE\WOW6432Node\ASIO\BEHRINGER"
        ]
        
        for key_path in behringer_keys:
            try:
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path, 0, winreg.KEY_READ)
                print(f"✓ 找到注册表项: {key_path}")
                winreg.CloseKey(key)
            except FileNotFoundError:
                print(f"✗ 未找到注册表项: {key_path}")
            except Exception as e:
                print(f"检查注册表项 {key_path} 时出错: {e}")
    except ImportError:
        print("无法导入 winreg 模块")
    except Exception as e:
        print(f"检查注册表时出错: {e}")
    
    print("\n=== 安装建议 ===")
    print("1. 访问百灵达官网下载 UMC 202HD 的 ASIO 驱动")
    print("   网址: https://www.behringer.com/product.html?modelCode=P0B1L")
    print("2. 或者安装 ASIO4ALL 作为通用 ASIO 驱动")
    print("   网址: https://www.asio4all.org/")
    print("3. 安装后重启音频服务或重新插拔设备")

if __name__ == "__main__":
    check_behringer_driver() 