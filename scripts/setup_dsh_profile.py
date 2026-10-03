#!/usr/bin/env python3
"""Create the portable dsh SQL profile after ``npm ci``."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def replace_link(link: Path, target: Path) -> None:
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.exists():
        shutil.rmtree(link)
    link.symlink_to(target.resolve(), target_is_directory=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsh-root", type=Path, default=Path(__file__).parents[1] / "harness/dsh-sql")
    parser.add_argument("--home", type=Path)
    args = parser.parse_args()

    root = args.dsh_root.resolve()
    home = (args.home or root / "home").resolve()
    profile = home / "profiles" / "spider2sql"
    profile.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "profile/package.json", profile / "package.json")
    shutil.copy2(root / "profile/cordis.patch.yml", profile / "cordis.patch.yml")
    modules = profile / "node_modules"
    modules.mkdir(exist_ok=True)
    replace_link(modules / "dsh-bundle-sql", root / "bundle-sql")
    replace_link(modules / "dsh-plugin-sql", root / "plugin-sql")
    print(profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
