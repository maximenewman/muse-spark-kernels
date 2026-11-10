"""Import-time stubs so baseline.py can import sglang without its serving-only deps.
Only modules the MoE baseline never calls are stubbed (vision/multimodal etc.)."""
import importlib.abc, importlib.machinery, sys, types

STUB = {"sglang.srt.configs", "flashinfer", "deep_gemm", "decord", "av", "cv2", "soundfile", "librosa", "timm"}

class _Any(types.ModuleType):
    __path__ = []
    def __getattr__(self, n):
        if n.startswith("__"):
            raise AttributeError(n)
        v = _Any(f"{self.__name__}.{n}")
        setattr(self, n, v)
        return v
    def __call__(self, *a, **k):
        return self
    def __mro_entries__(self, bases):
        return (object,)

class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if any(name == p or name.startswith(p + ".") for p in STUB):
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec):
        return _Any(spec.name)
    def exec_module(self, m):
        pass

def install(extra=()):
    STUB.update(extra)
    sys.meta_path.insert(0, _Finder())
