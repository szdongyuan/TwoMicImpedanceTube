import sounddevice as sd
import sys
import subprocess
import platform

def check_portaudio_info():
    """检查 PortAudio 的版本和编译信息"""
    print("=== PortAudio 信息检查 ===\n")
    
    # 检查 sounddevice 版本
    try:
        import sounddevice
        print(f"sounddevice 版本: {sounddevice.__version__}")
    except AttributeError:
        print("sounddevice 版本: 未知")
    
    # 检查 Python 版本
    print(f"Python 版本: {sys.version}")
    print(f"平台: {platform.platform()}")
    
    # 尝试获取 PortAudio 版本信息
    print("\n=== 尝试获取 PortAudio 版本信息 ===")
    try:
        # 使用 sounddevice 获取 PortAudio 版本
        version_info = sd.get_portaudio_version()
        print(f"PortAudio 版本: {version_info}")
    except Exception as e:
        print(f"无法获取 PortAudio 版本: {e}")
    
    # 检查可用的主机 API 详细信息
    print("\n=== 主机 API 详细信息 ===")
    hostapis = sd.query_hostapis()
    for i, api in enumerate(hostapis):
        print(f"\n主机 API {i}: {api['name']}")
        print(f"  类型: {api.get('type', 'N/A')}")
        print(f"  设备数量: {api.get('deviceCount', 'N/A')}")
        print(f"  默认输入设备: {api.get('defaultInputDevice', 'N/A')}")
        print(f"  默认输出设备: {api.get('defaultOutputDevice', 'N/A')}")
    
    # 检查是否有 ASIO 相关文件
    print("\n=== 检查 ASIO 驱动文件 ===")
    asio_paths = [
        r"C:\Windows\System32\asio.dll",
        r"C:\Windows\SysWOW64\asio.dll",
        r"C:\Program Files\ASIO4ALL\asio4all.dll",
        r"C:\Program Files (x86)\ASIO4ALL\asio4all.dll"
    ]
    
    for path in asio_paths:
        try:
            import os
            if os.path.exists(path):
                print(f"✓ 找到 ASIO 文件: {path}")
            else:
                print(f"✗ 未找到: {path}")
        except Exception as e:
            print(f"检查 {path} 时出错: {e}")
    
    # 检查注册表中的 ASIO 信息
    print("\n=== 检查注册表中的 ASIO 信息 ===")
    try:
        import winreg
        asio_keys = [
            r"SOFTWARE\ASIO",
            r"SOFTWARE\WOW6432Node\ASIO",
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\ASIO4ALL"
        ]
        
        for key_path in asio_keys:
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
    
    print("\n=== 建议 ===")
    print("如果您的系统没有 ASIO 支持，可以尝试以下方法:")
    print("1. 安装 ASIO4ALL 驱动: https://www.asio4all.org/")
    print("2. 重新编译 PortAudio 并启用 ASIO 支持")
    print("3. 使用支持 ASIO 的音频设备")
    print("4. 检查音频设备制造商是否提供 ASIO 驱动")

if __name__ == "__main__":
    check_portaudio_info() 