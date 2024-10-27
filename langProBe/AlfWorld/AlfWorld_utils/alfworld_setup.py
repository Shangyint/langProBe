from contextlib import contextmanager
import shutil
import subprocess
import os

def ensure_alfworld_setup():
    os.environ["ALFWORLD_DATA"] = os.path.join(os.getcwd(), "langProBe", "AlfWorld", "data")

    try:
        import alfworld
    except ImportError:
        subprocess.check_call("pip install alfworld[full]==0.3.5", shell=True)
    
    if os.path.exists(os.path.join(os.getcwd(), "langProBe", "AlfWorld", "data")):
        return

    subprocess.check_call(f"alfworld-download", shell=True)
