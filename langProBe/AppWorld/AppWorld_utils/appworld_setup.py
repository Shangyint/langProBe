from contextlib import contextmanager
import shutil
import subprocess
import os

def ensure_appworld_setup():
    os.environ["APPWORLD_ROOT"] = os.path.join(os.getcwd(), "langProBe", "AppWorld")

    # Check if the environment is already set up
    if os.path.exists(os.path.join(os.getcwd(), "langProBe", "AppWorld", ".conda")) and os.path.exists(os.path.join(os.getcwd(), "langProBe", "AppWorld", "data")):
        return

    # Cleanup
    if os.path.exists(os.path.join(os.getcwd(), "langProBe", "AppWorld", ".conda")):
        shutil.rmtree(os.path.join(os.getcwd(), "langProBe", "AppWorld", ".conda"))
    if os.path.exists(os.path.join(os.getcwd(), "langProBe", "AppWorld", "data")):
        shutil.rmtree(os.path.join(os.getcwd(), "langProBe", "AppWorld", "data"))    

    # Setup the environment
    subprocess.check_call(f"conda create --prefix {os.path.join(os.environ['APPWORLD_ROOT'], '.conda')} python=3.11.0 -y", shell=True)
    subprocess.check_call(f"conda run --prefix {os.path.join(os.environ['APPWORLD_ROOT'], '.conda')} pip install appworld==0.1.3", shell=True)
    subprocess.check_call(f"conda run --prefix {os.path.join(os.environ['APPWORLD_ROOT'], '.conda')} appworld install", shell=True)
    subprocess.check_call(f"conda run --prefix {os.path.join(os.environ['APPWORLD_ROOT'], '.conda')} appworld download data --root {os.environ['APPWORLD_ROOT']}", shell=True)
