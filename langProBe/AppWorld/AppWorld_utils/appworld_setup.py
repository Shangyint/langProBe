import shutil
import subprocess
import os
from pathlib import Path


def ensure_appworld_setup():
    appworld_root = Path.cwd() / "langProBe" / "AppWorld"

    os.environ["APPWORLD_ROOT"] = str(appworld_root)

    conda_root = appworld_root.joinpath(".conda")
    data_root = appworld_root.joinpath("data")

    # Check if the environment is already set up
    if conda_root.exists() and data_root.exists():
        return

    # Cleanup
    if conda_root.exists():
        shutil.rmtree(conda_root)
    if data_root.exists():
        shutil.rmtree(data_root)

    # Setup the environment
    subprocess.check_call(
        f"conda create --prefix {conda_root} python=3.11.1 -y",
        shell=True,
    )
    subprocess.check_call(
        f"conda run --prefix {conda_root} pip install appworld==0.1.3",
        shell=True,
    )
    subprocess.check_call(
        f"conda run --prefix {conda_root} appworld install",
        shell=True,
    )
    subprocess.check_call(
        f"conda run --prefix {conda_root} appworld download data --root {appworld_root}",
        shell=True,
    )
