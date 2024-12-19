import shutil
import subprocess
import os
from pathlib import Path


USE_UVX = os.getenv("LANGPROBE_UVX", False)


def get_appworld_root():
    return Path(__file__).parent


def ensure_appworld_setup() -> tuple[str, Path]:
    appworld_root = get_appworld_root()

    ensure_setup = ensure_setup_uvx if USE_UVX else ensure_setup_conda
    cmd = ensure_setup(appworld_root)

    return cmd, appworld_root


def ensure_setup_uvx(appworld_root: Path) -> str:
    data_root = appworld_root.joinpath("data")

    if not data_root.exists():
        subprocess.check_call(
            "uvx appworld install",
            shell=True,
        )
        subprocess.check_call(
            f"uvx appworld download data --root {appworld_root}",
            shell=True,
        )

    return "uvx"


def ensure_setup_conda(appworld_root: Path) -> str:
    conda_root = appworld_root.joinpath(".conda")
    data_root = appworld_root.joinpath("data")

    if not conda_root.exists() or not data_root.exists():
        # Cleanup
        if conda_root.exists():
            shutil.rmtree(conda_root)
        if data_root.exists():
            shutil.rmtree(data_root)

        # Setup the environment
        cmd = f"conda run --prefix {conda_root}"

        subprocess.check_call(
            f"conda create --prefix {conda_root} python=3.11.1 -y",
            shell=True,
        )
        subprocess.check_call(
            f"{cmd} pip install appworld==0.1.3",
            shell=True,
        )
        subprocess.check_call(
            f"{cmd} appworld install",
            shell=True,
        )
        subprocess.check_call(
            f"{cmd} appworld download data --root {appworld_root}",
            shell=True,
        )

    return cmd
