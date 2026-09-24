"""Run this on EVERY machine (Kaggle, Colab, local) before Day 1:  python scripts/check_env.py
Prints package versions, GPU, mixed-precision support and which platform you're on."""
import importlib, os, platform, shutil, sys

print("python", sys.version.split()[0], "|", platform.platform())
plat = "kaggle" if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") else ("colab" if "google.colab" in sys.modules or os.path.exists("/content") else "local/other")
print("platform:", plat)
missing = []
for pkg in ["numpy", "pandas", "scipy", "sklearn", "lightgbm", "xgboost", "catboost", "polars",
            "torch", "transformers", "sentencepiece", "accelerate", "datasets", "tqdm", "matplotlib", "seaborn"]:
    try:
        m = importlib.import_module(pkg)
        print(f"  OK   {pkg:14s} {getattr(m, '__version__', '')}")
    except Exception as e:
        missing.append(pkg); print(f"  MISS {pkg:14s} ({type(e).__name__})")
try:
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        cap = torch.cuda.get_device_capability()
        print(f"GPU: {p.name} | {p.total_memory / 1e9:.1f} GB | x{torch.cuda.device_count()} | capability {cap[0]}.{cap[1]}")
        print("mixed precision to use:", "bf16" if cap[0] >= 8 else "fp16 (no bf16 on this GPU)")
    else:
        print("GPU: NONE visible to torch  (Kaggle: Settings > Accelerator; Colab: Runtime > Change runtime type)")
except ImportError:
    pass
print("CPU cores:", os.cpu_count(), "| free disk GB:", round(shutil.disk_usage('.').free / 1e9, 1))
print("\nRESULT:", "all good" if not missing else f"missing {missing} -> pip install those (do NOT reinstall torch on Kaggle/Colab)")
