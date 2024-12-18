import shutil
import subprocess
import os
from pathlib import Path


USE_UVX = os.getenv("LANGPROBE_UVX", False)


def get_appworld_root():
    return Path.cwd() / "langProBe" / "AppWorld"


def ensure_appworld_setup():
    appworld_root = get_appworld_root()

    os.environ["APPWORLD_ROOT"] = str(appworld_root)

    data_root = appworld_root.joinpath("data")

    # Check if the environment is already set up
    if data_root.exists():
        return

    if USE_UVX:
        uvx_appworld_setup(appworld_root)
    else:
        conda_appworld_setup(appworld_root)


def uvx_appworld_setup(appworld_root):
    data_root = appworld_root.joinpath("data")
    if data_root.exists():
        return

    subprocess.check_call(
        "uvx appworld install",
        shell=True,
    )
    subprocess.check_call(
        f"uvx appworld download data --root {appworld_root}",
        shell=True,
    )


def conda_appworld_setup(appworld_root):
    conda_root = appworld_root.joinpath(".conda")
    data_root = appworld_root.joinpath("data")

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
