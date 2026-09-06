import os
import sys
import torch


def get_device() -> torch.device:
    """
    Detect the best available device: CUDA > MPS > CPU.
    Also ensures Windows Conda cuDNN DLLs are registered if CUDA is present.
    """
    if sys.platform == "win32":
        lib_bin = os.path.join(sys.prefix, "Library", "bin")
        if os.path.isdir(lib_bin):
            if lib_bin not in os.environ.get("PATH", ""):
                os.environ["PATH"] = lib_bin + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(lib_bin)
                except Exception:
                    pass

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
