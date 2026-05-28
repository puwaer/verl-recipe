#!/usr/bin/env python3
"""verl の model_merger (mcore→HF) を recipe/ 側の compat shim を当ててから
起動するラッパー。CLI 引数 (merge --backend megatron --local_dir ... --target_dir ...)
は透過する。
"""

from __future__ import annotations

# Patch must be applied BEFORE verl.model_merger is imported.
from recipe.gkd.megatron import _compat  # noqa: F401

import runpy


def main() -> None:
    runpy.run_module("verl.model_merger", run_name="__main__")


if __name__ == "__main__":
    main()
