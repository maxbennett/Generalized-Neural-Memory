import torch.distributed as dist
import torch
import wandb
from contextlib import nullcontext
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

def safe_wandb_summary(key, value):
    """Update wandb summary only if it's initialized"""
    try:
        if wandb.run is not None:
            wandb.run.summary[key] = value
    except:
        pass

def has_trainable_params(module: nn.Module) -> bool:
    return any(p.requires_grad for p in module.parameters())

def unwrap(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, DDP) else m

def wrap_ddp_if_eligible(module: nn.Module, **ddp_kwargs):
    use_ddp = (
        dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
        and has_trainable_params(module)
    )
    if use_ddp:
        return DDP(module, **ddp_kwargs)
    return module

def maybe_nosync(m, use):
    return m.no_sync() if hasattr(m, "no_sync") and use else nullcontext()

def init_distributed():
    if not dist.is_available() or dist.is_initialized():
        return
    dist.init_process_group(backend="nccl", init_method="env://")

def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

def unwrap_if_ddp(m):
    return m.module if hasattr(m, "module") else m

def is_ddp():
    return dist.is_available() and dist.is_initialized()

def rprint(*args, **kwargs):
    # if is_main_process():
    if dist.is_initialized(): 
        if dist.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)

def safe_wandb_log(data, **kwargs):
    """Log to wandb only if it's initialized (rank 0)"""
    if wandb.run is not None:
        wandb.log(data, **kwargs)

def safe_wandb_save(path):
    """Save file to wandb only if it's initialized"""
    try:
        if wandb.run is not None:
            wandb.save(path)
    except:
        pass

def safe_wandb_summary(key, value):
    """Update wandb summary only if it's initialized"""
    try:
        if wandb.run is not None:
            wandb.run.summary[key] = value
    except:
        pass

def bcast_bool_from_rank0(x: bool) -> bool:
    """Broadcast a Python boolean from rank 0 to all ranks (no CUDA tensors needed)."""
    if not (dist.is_available() and dist.is_initialized()):
        return x
    obj = [x if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(obj, src=0)
    return bool(obj[0])

def bcast_object_from_rank0(obj):
    import torch.distributed as dist
    if not dist.is_initialized():
        return obj
    obj_list = [obj] if dist.get_rank() == 0 else [None]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]

def bcast_float_from_rank0(x: float, src: int = 0) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return float(x)
    backend = dist.get_backend()
    dev = torch.device(f"cuda:{torch.cuda.current_device()}") if backend == dist.Backend.NCCL else torch.device("cpu")
    t = torch.tensor([float(x)], dtype=torch.float64, device=dev)
    dist.broadcast(t, src=src)
    return float(t.item())