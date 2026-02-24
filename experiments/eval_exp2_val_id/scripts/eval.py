import os
import torch
from gnm_data import DocDataset
import argparse, os, datetime, json, pathlib
import debugpy
from utils.metrics import ExperimentTracker
from utils.repro import seed_everything
from utils.distributed import rprint, is_main_process
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from memoryllm_train import validate
from config_memoryllm_train import MemoryllmTrainConfig, ModelCfg, load_cfg
from gnm import build_model



if __name__ == "__main__":
    
    # -----------------------
    # CLI
    # -----------------------
    rprint("blahtresssstre")
    rprint("Starting eval script...")
    parser = argparse.ArgumentParser(description="Train or resume a GMT model")
    parser.add_argument('--params', type=str, help='Path to params.yaml file')
    parser.add_argument("--model")
    parser.add_argument("--pretty-name")
    parser.add_argument("--run-key")
    parser.add_argument("--model-state-dict-path")
    parser.add_argument("--outdir")
    args = parser.parse_args()
    
    rprint(f"Evaluating {args.run_key}...")
    
    # -----------------------
    # Load & validate config
    # -----------------------
    if args.model_state_dict_path == "null" or args.model_state_dict_path == "None":
        args.model_state_dict_path = None
    
    modelcfg = ModelCfg(
        model=args.model,
        pretty_name=args.pretty_name,
        model_state_dict_path=args.model_state_dict_path
    )
    cfg: MemoryllmTrainConfig = load_cfg(args.params, modelcfg=modelcfg)
    
    # -----------------------
    # Debugging
    # -----------------------
    # import os
    # if os.environ.get("LOCAL_RANK", "0") == "0":
    #     import debugpy
    #     debugpy.listen(("0.0.0.0", 5679))
    #     print("Rank0 waiting for debugger attach on :5679")
    #     debugpy.wait_for_client()

    # -----------------------
    # Load Hyperparams
    # -----------------------
    output_dir                   = args.outdir
    # model_pretty_name            = args.pretty_name
    # unique_model_key             = args.unique_model_key
    # model_path                   = cfg.model.model_path
    io_sampling_params           = cfg.sampling
    max_new_tokens               = cfg.sampling.max_new_tokens
    seq_len                      = cfg.loop.sequence_length
    # seq_len                      = cfg.loop.sequence_length + 1 if cfg.data.unlearning else cfg.loop.sequence_length  # add 1 to sequence length to account for possible unlearning token at end

    # -----------------------
    # Device & distributed setup
    # -----------------------
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")


    # -----------------------
    # Epoch Invariant Reproducibility
    # -----------------------
    base_seed = cfg.seed
    process_seed = seed_everything(base_seed, rank=dist.get_rank())
    print(f"[rank {dist.get_rank()}] Using base_seed={base_seed}, process_seed={process_seed}")


    # -----------------------
    # Load model
    # -----------------------
    rprint("loading model ...")
    model_wrapper = build_model(cfg)
    tokenizer = model_wrapper.tokenizer

    rprint(f"model loaded, trainable parameters count: {len([n for n, p in model_wrapper.model.named_parameters() if p.requires_grad])}")
    if cfg.model.model_state_dict_path is not None:
        rprint(f"Loading model state dict from {cfg.model.model_state_dict_path}")
        state_dict = torch.load(cfg.model.model_state_dict_path, map_location='cpu')
        model_wrapper.load_state_dict(state_dict)
        rprint(f"Loaded model state dict from {cfg.model.model_state_dict_path}")
    
    model_wrapper.to(device)


    # -----------------------
    # Validation Datasets
    # -----------------------
    rprint("loading validation datasets ...")
    validation_datasets: list[DocDataset] = [
        DocDataset(path, shuffle=False, limit=cfg.data.val_limit)
        for path in cfg.data.val_ood_datasets
    ]
    

    # -----------------------
    # Trackers
    # -----------------------

    val_tracker_ood = ExperimentTracker(
        model_name=args.run_key,
        dataset_name="val_ood",
        sequence_length=seq_len,
        target_types=cfg.data.eligible_target_types_val_ood,
        unlearning=cfg.loop.unlearning
    )

    val_tracker_ood.to(device)


    model = DDP(model_wrapper, 
                # find_unused_parameters=True,
                find_unused_parameters=False,
                device_ids=[local_rank],
                output_device=local_rank)
    model.eval()
    model_in = model.module


    # -----------------------
    # Make output dir
    # -----------------------
    if is_main_process():
        # make results directory if doesn't exist yet
        results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"setting output dir to: {output_dir}")
        p = pathlib.Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        run_config_path = os.path.join(output_dir, "run_config.json")
        with open(run_config_path, "w") as f:
            json.dump(cfg.to_dict(), f, indent=2)
    dist.barrier()

    # -----------------------
    # Run Validation
    # -----------------------

    process_seed = seed_everything(base_seed, rank=dist.get_rank())
    val_tracker_ood = validate(
            model=model,
            validation_datasets=validation_datasets,
            seq_len=seq_len,
            max_new_tokens=max_new_tokens,
            eligible_target_types=cfg.data.eligible_target_types_val_ood,
            holdout_targets=cfg.data.holdout_targets_val_ood,
            holdout_target_values=cfg.data.holdout_target_values_val_ood,
            io_sampling_params=io_sampling_params,
            tracker=val_tracker_ood,
            device=device,
            output_dir=output_dir
        )
                            
    val_tracker_ood.log_epoch()
    val_tracker_ood.reset()
    dist.barrier()

    if is_main_process():
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(val_tracker_ood.epoch_results, f, indent=2)

    dist.barrier()      
    dist.destroy_process_group()