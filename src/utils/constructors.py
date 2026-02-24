# import os
# import torch
# import json
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.nn.parallel import DistributedDataParallel as DDP
# from gnm import MetaLearner, Learner
import bitsandbytes as bnb
from bitsandbytes.optim import GlobalOptimManager as BNBGOM
from peft import LoraConfig, get_peft_model
from config_memoryllm_train import MemoryllmTrainConfig
from utils.distributed import rprint

def _infer_layers_pattern_and_count(hf_model):
    # Returns (pattern_name, num_layers) for common HF stacks
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
        return "layers", len(hf_model.model.layers)        # llama/qwen/mistral
    if hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "h"):
        return "h", len(hf_model.transformer.h)             # gpt2-like
    # fallback – PEFT can still run without layers_to_transform
    return None, getattr(hf_model.config, "num_hidden_layers", None)

def apply_lora_to_learner(learner, cfg):
    """
    Wrap learner.model with LoRA. Only learner gets adapters; meta-learner stays full.
    """
    pattern_detected, _ = _infer_layers_pattern_and_count(learner.model)

    # Honor user pattern if provided, else auto-detect
    layers_pattern = cfg.get("lora_layers_pattern", pattern_detected)
    layers_to_transform = cfg.get("lora_layers_to_transform", None)

    lora_cfg = LoraConfig(
        r=int(cfg.get("lora_r", 4)),
        lora_alpha=int(cfg.get("lora_alpha", 8)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        target_modules=list(cfg.get("lora_target", ["q_proj", "c_attn"])),
        bias="none",
        task_type="CAUSAL_LM",
        layers_to_transform=layers_to_transform,
        layers_pattern=layers_pattern
    )
    learner.model = get_peft_model(learner.model, lora_cfg)
    try:
        learner.model.print_trainable_parameters()
    except Exception:
        pass
    return learner

def construct_memoryllm_optimizer(*, model: nn.Module, cfg:MemoryllmTrainConfig) -> optim.Optimizer:
    rprint(f"Constructing optimizer: {cfg.opt.optimizer}")
    wd = float(cfg.opt.weight_decay)
    learning_rate = float(cfg.opt.learning_rate)

    ### Optimizer & loss function ###
    if cfg.opt.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        rprint(f"Using Adam optimizer")

    # elif cfg.opt.optimizer == "adamw8bit":
    #     # IMPORTANT: `model` here should already be the *underlying* module,
    #     # NOT the DDP wrapper. In train.py, pass in `model_in = model.module`.
    #     def _param_groups(module: nn.Module, weight_decay: float):
    #         decay, no_decay = [], []
    #         for name, param in module.named_parameters():
    #             if not param.requires_grad:
    #                 continue
    #             # No weight decay on bias / norm params
    #             if name.endswith(".bias") or "norm" in name.lower() or "layernorm" in name.lower():
    #                 no_decay.append(param)
    #             else:
    #                 decay.append(param)

    #         groups = []
    #         if decay:
    #             groups.append({"params": decay, "weight_decay": weight_decay})
    #         if no_decay:
    #             groups.append({"params": no_decay, "weight_decay": 0.0})
    #         return groups

    #     groups = _param_groups(model, wd)
    #     optimizer = bnb.optim.PagedAdamW8bit(
    #         groups,
    #         lr=learning_rate,
    #         betas=(0.9, 0.999),
    #         eps=1e-8,
    #     )
    #     rprint("Using PagedAdamW8bit optimizer (no BNBGOM overrides)")
    elif cfg.opt.optimizer == "adamw8bit":

        manager = BNBGOM.get_instance()
        def _register_32bit_overrides(mod: nn.Module):
            for m in mod.modules():
                if isinstance(m, (nn.Embedding, nn.LayerNorm)):
                    # ensure 32-bit optimizer state for their 'weight' tensors
                    try:
                        manager.register_module_override(m, 'weight', {'optim_bits': 32})
                    except Exception:
                        pass
        _register_32bit_overrides(model)
        def trainable_params(m): 
            return [p for p in m.parameters() if p.requires_grad]
        
        def _param_groups(module: nn.Module, weight_decay: float):
            decay, no_decay = [], []
            for n, p in module.named_parameters():
                if not p.requires_grad:
                    continue
                # Standard WD rule: no WD on bias and LayerNorm/Norm weights
                if n.endswith(".bias") or "norm" in n.lower() or "layernorm" in n.lower():
                    no_decay.append(p)
                else:
                    decay.append(p)
            groups = []
            if decay:
                groups.append({"params": decay, "weight_decay": weight_decay})
            if no_decay:
                groups.append({"params": no_decay, "weight_decay": 0.0})
            return groups

        groups = _param_groups(model, wd)
        optimizer = bnb.optim.PagedAdamW8bit(
            groups,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        rprint(f"Using Paged AdamW 8 bitoptimizer")


    elif cfg.opt.optimizer == "adamw":
        optimizer = optim.AdamW(
            list(model.parameters()),
            lr=learning_rate,
            weight_decay=0.001  # typical starting point, tune if needed
        )
        rprint(f"Using AdamW optimizer")
    else:
        optimizer = optim.Adam(list(model.parameters()), lr=learning_rate)
        rprint(f"No optimizer selected, using default Adam optimizer")
    
    return optimizer

# def construct_gnm_optimizer(*, meta_learner:nn.Module, learner: nn.Module, cfg:TrainConfig) -> optim.Optimizer:
#     rprint(f"Constructing optimizer: {cfg.opt.optimizer}")
#     wd = float(cfg.opt.weight_decay)
#     learning_rate = float(cfg.opt.learning_rate)

#     ### Optimizer & loss function ###
#     if cfg.opt.optimizer == "adam":
#         # put both ml_unwrapped and lrnr_unwrapped parameters in the same Adam optimizer
#         optimizer = optim.Adam(list(meta_learner.parameters()) + list(learner.parameters()), lr=learning_rate)
#         rprint(f"Using Adam optimizer")

#     elif cfg.opt.optimizer == "adamw8bit":

#         manager = BNBGOM.get_instance()
#         def _register_32bit_overrides(mod: nn.Module):
#             for m in mod.modules():
#                 if isinstance(m, (nn.Embedding, nn.LayerNorm)):
#                     # ensure 32-bit optimizer state for their 'weight' tensors
#                     try:
#                         manager.register_module_override(m, 'weight', {'optim_bits': 32})
#                     except Exception:
#                         pass
#         _register_32bit_overrides(meta_learner)
#         _register_32bit_overrides(learner)
#         def trainable_params(m): 
#             return [p for p in m.parameters() if p.requires_grad]
        
#         def _param_groups(module: nn.Module, weight_decay: float):
#             decay, no_decay = [], []
#             for n, p in module.named_parameters():
#                 if not p.requires_grad:
#                     continue
#                 # Standard WD rule: no WD on bias and LayerNorm/Norm weights
#                 if n.endswith(".bias") or "norm" in n.lower() or "layernorm" in n.lower():
#                     no_decay.append(p)
#                 else:
#                     decay.append(p)
#             groups = []
#             if decay:
#                 groups.append({"params": decay, "weight_decay": weight_decay})
#             if no_decay:
#                 groups.append({"params": no_decay, "weight_decay": 0.0})
#             return groups

#         groups = _param_groups(meta_learner, wd) + _param_groups(learner, wd)
#         optimizer = bnb.optim.PagedAdamW8bit(
#             groups,
#             lr=learning_rate,
#             betas=(0.9, 0.999),
#             eps=1e-8
#         )
#         rprint(f"Using Paged AdamW 8 bitoptimizer")


#     elif cfg.opt.optimizer == "adamw":
#         optimizer = optim.AdamW(
#             list(meta_learner.parameters()) + list(learner.parameters()),
#             lr=learning_rate,
#             weight_decay=0.001  # typical starting point, tune if needed
#         )
#         rprint(f"Using AdamW optimizer")
#     else:
#         optimizer = optim.Adam(list(meta_learner.parameters()) + list(learner.parameters()), lr=learning_rate)
#         rprint(f"No optimizer selected, using default Adam optimizer")
    
#     return optimizer

def construct_memoryllm_scheduler(*, optimizer:optim.Optimizer, cfg:MemoryllmTrainConfig) -> optim.lr_scheduler._LRScheduler | None:
    rprint(f"Constructing scheduler")
    # --- LR Scheduler (warmup -> cosine; stepped per EPOCH) ---
    use_lr_sched   = bool(cfg.opt.use_lr_scheduler)
    warmup_epochs  = int(cfg.opt.lr_warmup_epochs)
    eta_min        = float(cfg.opt.min_lr)
    total_epochs   = int(cfg.loop.max_epochs)
    start_epoch    = int(cfg.loop.start_epoch)
    start_factor   = 0.1

    if use_lr_sched:
        # cosine runs after warmup for the remaining epochs
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_epochs - warmup_epochs - start_epoch), 
            eta_min=eta_min,
        )
        if warmup_epochs > 0:
            warmup = LinearLR(
                optimizer,
                start_factor=start_factor,
                total_iters=warmup_epochs,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs]
            )
        else:
            scheduler = cosine
    else: 
        scheduler = None
    
    return scheduler

def construct_memoryllm_scheduler(*, optimizer:optim.Optimizer, cfg:MemoryllmTrainConfig) -> optim.lr_scheduler._LRScheduler | None:
    rprint(f"Constructing scheduler")
    # --- LR Scheduler (warmup -> cosine; stepped per EPOCH) ---
    use_lr_sched   = bool(cfg.opt.use_lr_scheduler)
    warmup_epochs  = int(cfg.opt.lr_warmup_epochs)
    eta_min        = float(cfg.opt.min_lr)
    total_epochs   = int(cfg.loop.max_epochs)
    start_epoch    = int(cfg.loop.start_epoch)
    start_factor   = 0.1

    if use_lr_sched:
        # cosine runs after warmup for the remaining epochs
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_epochs - warmup_epochs - start_epoch), 
            eta_min=eta_min,
        )
        if warmup_epochs > 0:
            warmup = LinearLR(
                optimizer,
                start_factor=start_factor,
                total_iters=warmup_epochs,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs]
            )
        else:
            scheduler = cosine
    else: 
        scheduler = None
    
    return scheduler

# def construct_scheduler(*, optimizer:optim.Optimizer, cfg:TrainConfig) -> optim.lr_scheduler._LRScheduler | None:
#     rprint(f"Constructing scheduler")
#     # --- LR Scheduler (warmup -> cosine; stepped per EPOCH) ---
#     use_lr_sched   = bool(cfg.opt.use_lr_scheduler)
#     warmup_epochs  = int(cfg.opt.lr_warmup_epochs)
#     eta_min        = float(cfg.opt.min_lr)
#     total_epochs   = int(cfg.loop.max_epochs)
#     start_epoch    = int(cfg.loop.start_epoch)
#     start_factor   = 0.1

#     if use_lr_sched:
#         # cosine runs after warmup for the remaining epochs
#         cosine = CosineAnnealingLR(
#             optimizer,
#             T_max=max(1, total_epochs - warmup_epochs - start_epoch), 
#             eta_min=eta_min,
#         )
#         if warmup_epochs > 0:
#             warmup = LinearLR(
#                 optimizer,
#                 start_factor=start_factor,
#                 total_iters=warmup_epochs,
#             )
#             scheduler = SequentialLR(
#                 optimizer,
#                 schedulers=[warmup, cosine],
#                 milestones=[warmup_epochs]
#             )
#         else:
#             scheduler = cosine
#     else: 
#         scheduler = None
    
#     return scheduler

# def load_distill_lookup(path: str) -> dict:
#     distilled_train_path = os.path.join(path, 'train.json')
#     distilled_test_path = os.path.join(path, 'test.json')

#     rprint(f"Loading distilled facts from {distilled_train_path}")
#     with open(distilled_train_path, 'r') as f:
#         distill_lookup_train = json.load(f)
    
#     rprint(f"Loading distilled facts from {distilled_test_path}")
#     with open(distilled_test_path, 'r') as f:
#         distill_lookup_test = json.load(f)

#     distill_lookup = {**distill_lookup_train, **distill_lookup_test}
#     return distill_lookup

# def construct_gnm_models(*, cfg:TrainConfig) -> tuple[nn.Module, nn.Module]:
#     """
#     Construct the learner and meta-learner models based on the provided configuration.

#     Args:
#         config (dict): Configuration dictionary containing model parameters.

#     Returns:
#         tuple[nn.Module, nn.Module]: A tuple containing the learner and meta-learner models.
#     """
#     rprint(f"Constructing models: learner={cfg.model.huggingface_learner_model_name}, meta-learner={cfg.model.huggingface_metalearner_model_name}")
#     # learner = Learner(
#     #     model_name=cfg.model.huggingface_learner_model_name,
#     #     prefix_length=cfg.model.prefix_length,
#     #     allow_attention_learning=cfg.model.learner_allow_attention_learning,
#     #     allow_empty_memory_tok_learning=cfg.model.allow_empty_memory_tok_learning
#     # )

#     # meta_learner = MetaLearner(
#     #     model_name=cfg.model.huggingface_metalearner_model_name,
#     #     prefix_length=cfg.model.prefix_length
#     # )      
#     learner = Learner(cfg.model)

#     meta_learner = MetaLearner(cfg.model)      

#     if cfg.model.metalearner_state_dict_path is not None:
#         rprint(f"Loading metalearner state dict from {cfg.model.metalearner_state_dict_path}")
#         state_dict = torch.load(cfg.model.metalearner_state_dict_path, map_location='cpu')
#         meta_learner.load_state_dict(state_dict)
#         rprint(f"Loaded metalearner state dict from {cfg.model.metalearner_state_dict_path}")

#     if cfg.model.learner_state_dict_path is not None:
#         rprint(f"Loading learner state dict from {cfg.model.learner_state_dict_path}")
#         state_dict = torch.load(cfg.model.learner_state_dict_path, map_location='cpu')
#         learner.load_state_dict(state_dict)
#         rprint(f"Loaded learner state dict from {cfg.model.learner_state_dict_path}")

#     # if bool(cfg.model.use_lora_on_learner):
#     #     learner = apply_lora_to_learner(learner, cfg)
    
#     num_trainable_learner = sum(p.requires_grad for p in learner.parameters())
#     rprint(f"[learner] trainable tensors: {num_trainable_learner}")

#     num_trainable_meta_learner = sum(p.requires_grad for p in meta_learner.parameters())
#     rprint(f"[meta-learner] trainable tensors: {num_trainable_meta_learner}")

#     return learner, meta_learner