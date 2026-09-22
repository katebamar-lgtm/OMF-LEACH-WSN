from __future__ import annotations

import importlib
import platform
import sys

REQUIRED = ("numpy", "pymoo", "scipy")

print("Python:", sys.version.replace("\n", " "))
print("Platform:", platform.platform())
print()

ok = True
for name in REQUIRED:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "unknown")
        print(f"OK   {name:<8} {version}")
    except Exception as exc:
        ok = False
        print(f"FAIL {name:<8} {exc}")

print()
if ok:
    print("Environment check passed. You can launch the scalability campaign.")
    raise SystemExit(0)

print("Environment check failed. Install the missing packages in the SAME Python environment selected by VSCode.")
print("Typical command: python -m pip install numpy pymoo scipy")
raise SystemExit(1)
