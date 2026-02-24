import os
import yaml, json
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict


@dataclass
class DistributedCfg:
    cuda_visible_devices: str
    nproc_per_node: int 
    master_port: int

    @property
    def world_size(self) -> int:
        # If using one node, world_size == nproc_per_node.
        # If you add multinode later, extend this.
        return self.nproc_per_node
    
@dataclass
class DataCfg:
    eligible_target_types_val_ood: List[str] = None
    val_ood_datasets: Optional[List[str]] = None
    training_datasets: Optional[List[str]] = None
    val_id_path: Optional[str] = None
    eligible_target_types_train: Optional[List[str] | None]  = None
    eligible_target_types_val_id: Optional[List[str] | None]  = None
    add_refusal_to_user_message: bool = False
    holdout_targets_train: Optional[List[str] | None]  = None
    holdout_target_values_train: Optional[List[str] | None]  = None
    holdout_targets_val_id: Optional[List[str] | None]  = None
    holdout_target_values_val_id: Optional[List[str] | None]  = None
    holdout_targets_val_ood: Optional[List[str] | None]  = None
    holdout_target_values_val_ood: Optional[List[str] | None]  = None
    train_limit: Optional[int] = None
    val_limit: Optional[int] = None

@dataclass
class SamplingCfg:
    num_paraphrase_from_facts_to_learn: int = 1
    num_true_output_neighbor_from_facts_to_learn: int = 1
    num_paraphrase_from_facts_to_refuse: int = 2
    num_true_output_neighbor_from_facts_to_refuse: int = 1
    num_paraphrase_from_other_facts: int = 1
    num_true_output_neighbor_from_other_facts: int = 0
    max_total_io_pairs_per_sample: int = 4
    num_io_pairs_to_chunk: int = 2
    max_new_tokens: int = 8
    num_in_future_to_pull_back_future_paraphrase: int = 4
    num_in_past_to_pull_forward_true_output_neighbor: int = 4
    num_in_past_to_pull_forward_paraphrase_not_target_to_learn: int = 4

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass
class ModelCfg:
    model: str # model registry key
    model_path: Optional[str] = None # huggingface path
    tokenizer_path: Optional[str] = None
    model_state_dict_path: Optional[str] = None
    checkpoint_dict_path: Optional[str] = None
    pretty_name: Optional[str] = None
    ablation: Optional[bool] = False

@dataclass
class LoopCtrlCfg:
    
    sequence_length: int = 10
    sequence_length_validation: Optional[int] = None
    unlearning : bool = False
    early_stop: bool = True
    window_size: int = 6
    stride: int = 6
    max_risk_score_threshold: int = 80000
    start_epoch: int = 0
    starting_global_step: int = 0
    plot_every_n_epochs: int = 1
    checkpoint_every_n_epochs: int = 2
    max_checkpoints: int = 5
    completions_batch_sample_count: int = 1
    early_stop_patience: int = 5
    early_stop_min_delta: float = 0.01
    max_epochs: int = 100

@dataclass
class OptCfg:
    optimizer: str
    learning_rate: float
    max_grad_norm: float
    gradient_accumulation_steps: int
    batch_size_per_device: int
    use_lr_scheduler: bool 
    lr_warmup_epochs: int
    min_lr: float
    weight_decay: float

@dataclass
class LoggingCfg:
    wandb_project: str = "test_project"
    batch_log_interval: int = 1

@dataclass
class MemoryllmTrainConfig:
    run_name: str
    seed: int = 42
    distributed: DistributedCfg = field(default_factory=DistributedCfg)
    data: DataCfg = field(default_factory=DataCfg)
    sampling: SamplingCfg = field(default_factory=SamplingCfg)
    opt: Optional[OptCfg] = None
    model: ModelCfg = field(default_factory=ModelCfg)
    loop: LoopCtrlCfg = field(default_factory=LoopCtrlCfg)
    log: LoggingCfg = field(default_factory=LoggingCfg)

    @property
    def global_batch_size(self) -> int:
        return self.opt.batch_size_per_device * self.distributed.world_size

    def validate(self) -> None:
        assert self.run_name, "run_name must be set"
        # assert self.data.training_datasets, "At least one training dataset path is required"
        assert self.loop.sequence_length > 0, "loop.sequence_length must be > 0"   # <- fix from self.windows
        if self.opt is not None:
            assert self.opt.gradient_accumulation_steps >= 1, "opt.gradient_accumulation_steps must be >= 1"
        # assert len(self.data.training_datasets) > 0, "No training datasets provided"

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_flat_dict(self, prefix="") -> Dict[str, object]:
        out = {}
        def _flatten(d, pfx=""):
            for k, v in d.items():
                key = f"{pfx}.{k}" if pfx else k
                if isinstance(v, dict):
                    _flatten(v, key)
                else:
                    out[key] = v
        _flatten(asdict(self), prefix)
        # Also include a few derived fields explicitly (nice for W&B):
        out["derived.global_batch_size"] = self.global_batch_size
        return out
    

def load_cfg(path: str, modelcfg: ModelCfg = None) -> MemoryllmTrainConfig:
    """
    Load YAML/JSON config and return a validated TrainConfig.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r") as f:
        if path.endswith((".yml", ".yaml")):
            raw = yaml.safe_load(f)
        elif path.endswith(".json"):
            raw = json.load(f)
        else:
            raise ValueError("Config must be .json, .yaml, or .yml")

    if raw is None:
        raise ValueError("Empty config file")

    # If user wrapped everything under 'train', unwrap it.
    if isinstance(raw, dict) and "train" in raw and isinstance(raw["train"], dict):
        raw = raw["train"]
    
    cfg = MemoryllmTrainConfig(
        run_name       = raw["run_name"],
        seed           = raw["seed"],
        distributed    = DistributedCfg(**raw["distributed"]),
        data           = DataCfg(**raw["data"]),
        sampling       = SamplingCfg(**raw["sampling"]),
        opt            = OptCfg(**raw["opt"]) if "opt" in raw and raw["opt"] is not None else None,
        model          = modelcfg if modelcfg is not None else ModelCfg(**raw["model"]),
        loop           = LoopCtrlCfg(**raw["loop"]),
        log            = LoggingCfg(**raw["log"]),
    )
    cfg.validate()
    return cfg