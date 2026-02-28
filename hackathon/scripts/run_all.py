import subprocess
import sys


if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "hackathon.scripts.prepare_data"], check=True)
    subprocess.run([sys.executable, "-m", "hackathon.scripts.run_local"], check=True)
