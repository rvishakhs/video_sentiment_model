import subprocess
import sys

def install_ffmpeg():
    print("starting to install FFMPEG")

    subprocess.check_call([sys.executable, "-m", "pip", 
                           "install", "--upgrade", "pip"])
    
    subprocess.check_call([sys.executable, "-m", "pip", 
                           "install", "--upgrade", "setuptools"])
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", 
                        "install", "ffmpeg-python"])
        print("FFMPEG installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing FFMPEG via pip: {e}")

    try:
        subprocess.check_call([
            "wget",
            "https://launchpad.net/ubuntu/+archive/primary/+sourcefiles/ffmpeg/7:7.1.1-1ubuntu2/ffmpeg_7.1.1.orig.tar.xz"
            "-O", "/tmp/ffmpeg.tar.xz"
        ])
        subprocess.check_call([
            "tar",
            "-xf", "/tmp/ffmpeg.tar.xz",
            "-C", "/tmp"
        ])

        result = subprocess.run([
            "find", "/tmp",
            "-name", "ffmpeg",
            "type", "f",           
        ], capture_output=True, text=True)

        ffmpeg_path = result.stdout.strip()

        subprocess.check_call([
            "sudo", "cp", ffmpeg_path, "/usr/local/bin/ffmpeg"
        ])

        subprocess.check_call([
            "sudo", "chmod", "+x", "/usr/local/bin/ffmpeg"
        ])

        print("FFMPEG installed successfully from source.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing FFMPEG from source: {e}")

    try:
        result = subprocess.run([
            "ffmpeg", "-version"
        ], capture_output=True, text=True)
        print(f"FFMPEG version:{result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error checking FFMPEG version: {e}")
        return False