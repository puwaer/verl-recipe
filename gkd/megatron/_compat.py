# Copyright 2026 (recipe/gkd local compat shim)
"""transformers 5.3+ で rope_theta が rope_parameters dict 配下に移動した skew を、
verl/ 本体を変更せずに吸収するための monkey-patch。

verl の `model_initializer.initialize()` や `config_converter` が `hf_config.rope_theta`
を直接読みに来るが、Qwen3Config 等の新 transformers はこの属性を持たず
`hf_config.rope_parameters["rope_theta"]` 配下に nest している。
`PretrainedConfig.__getattr__` にフォールバックを足すことで、属性アクセスを
透過的に解決する。

このモジュールを import すれば patch が自動適用される。recipe/ 内 (main_gkd.py /
ラッパースクリプト) から最初に import すること。
"""

from __future__ import annotations


def apply_rope_theta_fallback() -> None:
    import transformers

    cfg_cls = transformers.PretrainedConfig
    if getattr(cfg_cls, "_rope_theta_fallback_patched", False):
        return

    orig_getattr = getattr(cfg_cls, "__getattr__", None)

    def __getattr__(self, name):
        if name == "rope_theta":
            try:
                rp = object.__getattribute__(self, "rope_parameters")
                if isinstance(rp, dict) and "rope_theta" in rp:
                    return rp["rope_theta"]
            except AttributeError:
                pass
        if orig_getattr is not None:
            return orig_getattr(self, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    cfg_cls.__getattr__ = __getattr__
    cfg_cls._rope_theta_fallback_patched = True


apply_rope_theta_fallback()
