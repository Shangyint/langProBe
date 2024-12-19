from pathlib import Path
import subprocess
import os


def ensure_alfworld_setup() -> Path:
    data_dir = Path(__file__).parent.parent / "data"

    if not data_dir.exists():
        subprocess.check_call("alfworld-download", shell=True)

    os.environ["ALFWORLD_DATA"] = str(data_dir)
    return data_dir
