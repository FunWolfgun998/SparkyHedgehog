import os

# --- FORZATURA GPU AMD RDNA3 ---
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
# ------------------------------

import torch

print(f"Versione PyTorch: {torch.__version__}")

# Verifica se CUDA (NVIDIA/ROCm) è disponibile
cuda_available = torch.cuda.is_available()
print(f"CUDA disponibile: {cuda_available}")

if cuda_available:
    # Usiamo 0 perché abbiamo detto a HIP_VISIBLE_DEVICES di mostrare solo la 7600 XT
    print(f"Nome della GPU: {torch.cuda.get_device_name(0)}")
    print(f"Numero GPU disponibili: {torch.cuda.device_count()}")
else:
    print("PyTorch non vede la GPU. Continueremo a usare la CPU.")