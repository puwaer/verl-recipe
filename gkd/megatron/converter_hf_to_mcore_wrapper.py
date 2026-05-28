#!/usr/bin/env python3
"""verl の HF→mcore コンバータ (verl/scripts/converter_hf_to_mcore.py) を
recipe/ 側の compat shim を当ててから起動するラッパー。

verl/ 本体を変更せずに transformers 5.3+ の rope_theta skew を吸収するため、
`_compat` を先に import してから verl の converter を `runpy.run_path` で
__main__ として実行する。CLI 引数 (--hf_model_path 等) はそのまま透過する。
"""

from __future__ import annotations

# Patch must be applied BEFORE verl.models.mcore is imported.
from recipe.gkd.megatron import _compat  # noqa: F401

import runpy

CONVERTER_PATH = "/workspace/verl/scripts/converter_hf_to_mcore.py"


def main() -> None:
    runpy.run_path(CONVERTER_PATH, run_name="__main__")


if __name__ == "__main__":
    main()
