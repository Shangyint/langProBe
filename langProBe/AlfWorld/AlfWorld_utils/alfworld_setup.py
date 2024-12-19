from pathlib import Path
import subprocess
import os


def ensure_alfworld_setup() -> Path:
    alfworld_root = Path(__file__).parent.parent
    os.environ["ALFWORLD_DATA"] = str(alfworld_root / "data")

    if (alfworld_root / "data").exists():
        return

    subprocess.check_call("alfworld-download", shell=True)
    return alfworld_root / "data"
