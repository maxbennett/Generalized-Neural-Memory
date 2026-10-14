import random
import numpy as np
import torch

def seed_everything(base_seed: int, *, rank: int = 0) -> int:
    """
    Seeds Python, NumPy, and Torch. Returns the *process seed* you can use downstream.
    In DDP, pass rank so each process has a unique stream while remaining reproducible.
    """
    process_seed = (int(base_seed) + int(rank)) % (2**31 - 1)

    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed)

    # PyTorch determinism toggles
    # torch.use_deterministic_algorithms(True)  # raises on nondeterministic ops
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # (Optional) stricter matmul determinism; may slow things down:
    # torch.set_float32_matmul_precision("high")  # default is "highest"

    return process_seed