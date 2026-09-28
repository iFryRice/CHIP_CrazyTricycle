"""Run a seeded CUDA forward/backward witness for the isolated training stack."""

import json
import sys

import torch
import transformers


if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required; do not silently substitute CPU")
torch.set_num_threads(2)
torch.manual_seed(20260928)
torch.cuda.manual_seed_all(20260928)
values = torch.randn(128, 128, device="cuda", requires_grad=True)
with torch.autocast("cuda", dtype=torch.float16):
    product = values @ values.T
    loss = product.float().square().mean()
loss.backward()
torch.cuda.synchronize()
assert torch.isfinite(loss) and torch.isfinite(values.grad).all()
print("CUDA_WITNESS " + json.dumps({
    "python": sys.version.split()[0], "torch": torch.__version__,
    "transformers": transformers.__version__, "device": torch.cuda.get_device_name(0),
    "capability": list(torch.cuda.get_device_capability(0)), "dtype": str(product.dtype),
    "shape": list(product.shape), "loss": loss.item(),
    "peak_gpu_mib": torch.cuda.max_memory_allocated() / 1024**2,
}))
