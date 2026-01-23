import soundcard as sc

print("=== 使用 soundcard 库检查 ASIO 支持 ===")
print()

# 列出所有麦克风
print("所有麦克风设备:")
microphones = sc.all_microphones()
for i, mic in enumerate(microphones):
    print(f"  {i}: {mic}")
    print(f"      名称: {mic.name}")
    print(f"      通道数: {mic.channels}")
    print()

# 列出所有扬声器
print("所有扬声器设备:")
speakers = sc.all_speakers()
for i, spk in enumerate(speakers):
    print(f"  {i}: {spk}")
    print(f"      名称: {spk.name}")
    print(f"      通道数: {spk.channels}")
    print()

# 尝试获取默认设备
print("默认设备:")
try:
    default_mic = sc.default_microphone()
    print(f"默认麦克风: {default_mic}")
except Exception as e:
    print(f"获取默认麦克风失败: {e}")

try:
    default_speaker = sc.default_speaker()
    print(f"默认扬声器: {default_speaker}")
except Exception as e:
    print(f"获取默认扬声器失败: {e}")

print()

# 尝试使用 BEHRINGER 设备
print("=== 尝试使用 BEHRINGER 设备 ===")
try:
    # 查找 BEHRINGER 设备
    behringer_mics = [mic for mic in microphones if 'BEHRINGER' in mic.name.upper()]
    behringer_spks = [spk for spk in speakers if 'BEHRINGER' in spk.name.upper()]
    
    if behringer_mics:
        print(f"找到 {len(behringer_mics)} 个 BEHRINGER 麦克风:")
        for mic in behringer_mics:
            print(f"  {mic.name}")
            print(f"    通道数: {mic.channels}")
            
            # 尝试获取设备信息
            try:
                with mic.recorder(samplerate=48000) as recorder:
                    print(f"    ✓ 可以创建录音器 (48kHz)")
            except Exception as e:
                print(f"    ❌ 创建录音器失败: {e}")
    else:
        print("未找到 BEHRINGER 麦克风")
    
    if behringer_spks:
        print(f"找到 {len(behringer_spks)} 个 BEHRINGER 扬声器:")
        for spk in behringer_spks:
            print(f"  {spk.name}")
            print(f"    通道数: {spk.channels}")
            
            # 尝试获取设备信息
            try:
                with spk.player(samplerate=48000) as player:
                    print(f"    ✓ 可以创建播放器 (48kHz)")
            except Exception as e:
                print(f"    ❌ 创建播放器失败: {e}")
    else:
        print("未找到 BEHRINGER 扬声器")
        
except Exception as e:
    print(f"检查 BEHRINGER 设备时出错: {e}")

print()
print("=== 检查是否支持 ASIO ===")
print("注意: soundcard 库通常使用系统默认的音频后端")
print("如果设备支持 ASIO，通常会在设备名称中显示相关信息")
print("或者需要设置环境变量来强制使用 ASIO 后端")

# 尝试测试高采样率
print()
print("=== 测试高采样率支持 (ASIO 特性) ===")
if behringer_mics:
    test_samplerates = [44100, 48000, 96000, 192000]
    for mic in behringer_mics:
        print(f"测试设备: {mic.name}")
        for sr in test_samplerates:
            try:
                with mic.recorder(samplerate=sr) as recorder:
                    print(f"  ✓ 支持 {sr} Hz 采样率")
            except Exception as e:
                print(f"  ❌ 不支持 {sr} Hz 采样率: {e}") 