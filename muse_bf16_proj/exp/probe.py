# okc: problem=muse_bf16_proj
import torch, sys, shutil, os

def prepare(weights):
    print("torch", torch.__version__, "cuda", torch.version.cuda, "py", sys.version)
    print("gpu", torch.cuda.get_device_name(), torch.cuda.get_device_properties(0))
    try:
        import triton; print("triton", triton.__version__)
    except Exception as e: print("no triton", e)
    print("nvcc", shutil.which("nvcc"), "CUDA_HOME", os.environ.get("CUDA_HOME"))
    for mod in ["cutlass", "flashinfer", "sgl_kernel", "deep_gemm", "cuda", "cupy", "tilelang", "cuda.bindings", "quack"]:
        try:
            m = __import__(mod); print("have", mod, getattr(m, "__version__", ""))
        except Exception as e: print("no", mod, type(e).__name__)
    return weights

def kernel_fn(data, weights):
    w = weights[data["config"]["layer"]][: data["config"]["n"]]
    if w.dtype == torch.float32:
        return data["x"].float() @ w.T
    return data["x"] @ w.T
