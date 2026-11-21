"""Run CUDA-only worker code on Apple's MPS backend.

Models are written against `cuda` devices and `torch.cuda.*` queries. On macOS this
module redirects those calls to MPS (with CPU fallback for unsupported operators)
instead of editing every vendored model. Import `install()` before any model code.
"""
import os
import sys
from types import SimpleNamespace

CONVOLUTIONS = {'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d'}


def available():
    return sys.platform == 'darwin'


def _is_cuda(value):
    return (isinstance(value, str) and value.startswith('cuda')) or \
        (hasattr(value, 'type') and getattr(value, 'type', None) == 'cuda')


def install():
    if not available():
        return False
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    import torch
    if not torch.backends.mps.is_available() or getattr(torch, '_rgbmocap_mps', False):
        return False
    torch._rgbmocap_mps = True
    mps = torch.device('mps')

    def fix(value):
        return mps if _is_cuda(value) else value

    def fix_dtype(value, device):
        return torch.float32 if value is torch.float64 and device is not None and str(device).startswith('mps') else value

    # --- torch.cuda queries -------------------------------------------------
    total = int(torch.mps.recommended_max_memory())

    def mem_get_info(device=None):
        return max(0, total - int(torch.mps.current_allocated_memory())), total

    chip = 'Apple GPU'
    try:
        import subprocess
        chip = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True).stdout.strip() or chip
    except OSError:
        pass
    cuda = torch.cuda
    cuda.is_available = lambda: True
    cuda.is_initialized = lambda: True
    cuda.device_count = lambda: 1
    cuda.current_device = lambda: 0
    cuda.set_device = lambda device: None
    cuda.mem_get_info = mem_get_info
    cuda.empty_cache = torch.mps.empty_cache
    cuda.synchronize = lambda device=None: torch.mps.synchronize()
    cuda.reset_peak_memory_stats = lambda device=None: None
    cuda.set_per_process_memory_fraction = lambda fraction, device=None: None
    cuda.get_per_process_memory_fraction = lambda device=None: 1.0
    cuda.max_memory_cached = cuda.memory_cached = lambda device=None: int(torch.mps.driver_allocated_memory())
    cuda.reset_max_memory_allocated = cuda.reset_peak_memory_stats
    cuda.max_memory_allocated = lambda device=None: int(torch.mps.current_allocated_memory())
    cuda.max_memory_reserved = lambda device=None: int(torch.mps.driver_allocated_memory())
    cuda.memory_allocated = lambda device=None: int(torch.mps.current_allocated_memory())
    cuda.memory_reserved = lambda device=None: int(torch.mps.driver_allocated_memory())
    cuda.get_device_name = lambda device=None: chip
    cuda.get_device_capability = lambda device=None: (0, 0)
    cuda.get_device_properties = lambda device=None: SimpleNamespace(name=chip, total_memory=total, major=0, minor=0)
    cuda.manual_seed = cuda.manual_seed_all = lambda seed: torch.mps.manual_seed(seed)

    # --- explicit device arguments ------------------------------------------
    from torch.overrides import TorchFunctionMode

    class Redirect(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = dict(kwargs or {})
            if getattr(func, '__name__', '') in CONVOLUTIONS and args and getattr(args[0], 'device', None) is not None \
                    and args[0].device.type == 'mps' and not args[0].is_contiguous():
                # MPS convolutions reject strided views such as x[..., 32:-32].
                args = (args[0].contiguous(),) + tuple(args[1:])
            if func is torch.Tensor.cuda:
                func, args = torch.Tensor.to, (args[0], mps)
            if func is torch.Tensor.double and getattr(args[0], 'device', None) is not None and args[0].device.type == 'mps':
                func = torch.Tensor.float
            if _is_cuda(kwargs.get('device')):
                kwargs['device'] = mps
            if func is torch.Tensor.to:
                args = tuple(fix(a) for a in args)
                target = kwargs.get('device', next((a for a in args[1:] if isinstance(a, (str, torch.device))), None))
                if target is None and getattr(args[0], 'device', None) is not None and args[0].device.type == 'mps':
                    target = 'mps'
                if target is not None and str(target).startswith('mps'):
                    args = tuple(fix_dtype(a, target) for a in args)
                    kwargs = {k: fix_dtype(v, target) if k == 'dtype' else v for k, v in kwargs.items()}
                    if args[0].dtype is torch.float64 and not any(isinstance(a, torch.dtype) for a in args[1:]) and 'dtype' not in kwargs:
                        args = (args[0].float(),) + args[1:]
            elif kwargs.get('device') is not None and str(kwargs['device']).startswith('mps'):
                kwargs['dtype'] = fix_dtype(kwargs.get('dtype'), kwargs['device'])
                if kwargs['dtype'] is None:
                    del kwargs['dtype']
            return func(*args, **kwargs)

    Redirect().__enter__()

    # --- modules, checkpoints, mixed precision ------------------------------
    Module = torch.nn.Module
    original_to = Module.to

    def module_to(self, *args, **kwargs):
        args = tuple(fix(a) for a in args)
        if _is_cuda(kwargs.get('device')):
            kwargs['device'] = mps
        target = kwargs.get('device', next((a for a in args if isinstance(a, (str, torch.device))), None))
        if target is not None and str(target).startswith('mps'):
            if any(p.dtype is torch.float64 for p in self.parameters()):
                self.float()
            args = tuple(fix_dtype(a, target) for a in args)
            if 'dtype' in kwargs:
                kwargs['dtype'] = fix_dtype(kwargs['dtype'], target)
        return original_to(self, *args, **kwargs)

    Module.to = module_to
    Module.cuda = lambda self, device=None: module_to(self, mps)

    original_load = torch.load

    def load(*args, **kwargs):
        args = list(args)
        location = args[1] if len(args) > 1 else kwargs.get('map_location')
        location = 'cpu' if location is None or callable(location) or _is_cuda(location) else location
        if len(args) > 1:
            args[1] = location
        else:
            kwargs['map_location'] = location
        return original_load(*args, **kwargs)

    torch.load = load

    original_jit_load = torch.jit.load

    class CpuScript(torch.nn.Module):
        # MHR's TorchScript forward needs float64, which MPS lacks: run it on CPU and hand results back.
        def __init__(self, script):
            super().__init__()
            self.script = script

        def _apply(self, fn, recurse=True):
            return self

        def forward(self, *args, **kwargs):
            def cpu(v):
                return v.detach().cpu() if torch.is_tensor(v) else v
            home = next((a.device for a in list(args) + list(kwargs.values()) if torch.is_tensor(a)), None)
            out = self.script(*[cpu(a) for a in args], **{k: cpu(v) for k, v in kwargs.items()})
            if home is None or home.type == 'cpu':
                return out
            return tuple(o.to(home) if torch.is_tensor(o) else o for o in out) if isinstance(out, (tuple, list)) else out.to(home)

    def jit_load(f, map_location=None, *args, **kwargs):
        if map_location is None or _is_cuda(map_location):
            map_location = 'cpu'
        return CpuScript(original_jit_load(f, map_location, *args, **kwargs))

    torch.jit.load = jit_load
    original_autocast = torch.autocast.__init__

    def autocast_init(self, device_type='cuda', *args, **kwargs):
        return original_autocast(self, 'mps' if device_type == 'cuda' else device_type, *args, **kwargs)

    torch.autocast.__init__ = autocast_init
    return True


def is_oom(error):
    return 'out of memory' in str(error).lower()
