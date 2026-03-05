"""
FFmpeg执行器模块
提供统一的FFmpeg命令执行接口，减少重复代码
"""
import subprocess
from rich import print

def execute_ffmpeg_command(cmd: list[str], verbose: bool = False) -> bool:
    """
    执行FFmpeg命令的通用函数
    """
    try:
        if verbose:
            print("执行命令: ")
            print(f"{' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        else:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg执行失败，错误码: {e.returncode}")
        print("执行的命令: ")
        print(f"{' '.join(cmd)}")
        if verbose:
            print(f"stderr输出: {e.stderr.decode('utf-8') if e.stderr else 'None'}")
            print(f"stdout输出: {e.stdout.decode('utf-8') if e.stdout else 'None'}")
        return False

