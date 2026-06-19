"""
ui_pyside — PySide6 replacement UI for the interpolation engine.

Imports of this package are safe: DLL resolution is hardened before
any Qt or numerical library loads, preventing the gemini_env PATH
contamination that causes 0xc06d007f crashes on this Windows stack.
"""
import os as _os
import sys as _sys

# ── DLL safety: force fafalab2 native libs to the front ─────────────────────
for _d in (r"Library\bin", r"Library\lib", r"DLLs", ""):
    _cand = _os.path.join(_sys.prefix, _d)
    if _os.path.isdir(_cand):
        try:
            _os.add_dll_directory(_cand)
        except (OSError, AttributeError):
            pass
    if _cand and _cand not in _os.environ.get("PATH", ""):
        _os.environ["PATH"] = _cand + ";" + _os.environ.get("PATH", "")

# Qt API — must be set before any Qt import
_os.environ.setdefault("QT_API", "pyside6")
