import os
import torch
from gnm_data import DocDataset
from utils.metrics import ExperimentTracker
from utils.repro import seed_everything
from utils.plotting import generate_plots
from utils.distributed import rprint, is_main_process, safe_wandb_log
from utils.constructors import construct_memoryllm_optimizer, construct_memoryllm_scheduler
from utils.checkpoints import save_model, save_checkpoint, load_checkpoint
import wandb
import argparse, os, datetime, json, pathlib
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from memoryllm_train import validate
from memoryllm_train import train_1_epoch
from config_memoryllm_train import MemoryllmTrainConfig, load_cfg
from gnm import build_model

if __name__ == "__main__":

    # -----------------------
    # CLI
    # -----------------------
    rprint("Starting training script...")
    parser = argparse.ArgumentParser(description="Train or resume a GMT model")
    parser.add_argument('--params', type=str, help='Path to params.yaml file')
    args = parser.parse_args()

    # -----------------------
    # Load & validate config
    # -----------------------
    cfg: MemoryllmTrainConfig = load_cfg(args.params)
    training_run_dir = os.getcwd()
    results_dir = os.path.join(training_run_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    rprint(f"Initial training run directory: {training_run_dir}")

    # -----------------------
    # Load Hyperparams
    # -----------------------
    model_name                   = cfg.model.model
    model_path                   = cfg.model.model_path
    io_sampling_params           = cfg.sampling
    max_new_tokens               = cfg.sampling.max_new_tokens
    seq_len                      = cfg.loop.sequence_length
    seq_len_validation           = cfg.loop.sequence_length_validation
    gradient_accumulation_steps  = cfg.opt.gradient_accumulation_steps
    start_epoch                  = int(cfg.loop.start_epoch)
    global_step                  = int(cfg.loop.starting_global_step)
    max_epochs                   = int(cfg.loop.max_epochs)
    plot_every_n_epochs          = int(cfg.loop.plot_every_n_epochs)
    checkpoint_every_n_epochs    = int(cfg.loop.checkpoint_every_n_epochs)
    max_checkpoints              = int(cfg.loop.max_checkpoints)
    early_stop                   = bool(cfg.loop.early_stop)
    early_stop_patience          = int(cfg.loop.early_stop_patience)
    early_stop_min_delta         = float(cfg.loop.early_stop_min_delta)

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
    # Device & distributed setup
    # -----------------------
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # -----------------------
    # W&B init
    # -----------------------
    if is_main_process():
        wandb.init(
                project=cfg.log.wandb_project,
                name=cfg.run_name,
                config=cfg.to_flat_dict())
        wandb.define_metric("loss/val_id", summary="min")
        wandb.define_metric("loss/val_ood", summary="min")
    else:
        os.environ["WANDB_MODE"] = "disabled"

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
    
    # validation_data_ood = DocDataset(
    #     cfg.data.val_ood_path, shuffle=False, limit=cfg.data.val_limit
    # )
    rprint("loading validation datasets ...")
    validation_datasets_ood: list[DocDataset] = [
                                            DocDataset(path, shuffle=False, limit=cfg.data.val_limit)
                                            for path in cfg.data.val_ood_datasets
                                        ]
    
    validation_datasets_id = [DocDataset(
        cfg.data.val_id_path, shuffle=False, limit=cfg.data.val_limit
    )]

    # -----------------------
    # Trackers
    # -----------------------
    # train_tracker = ExperimentTracker(
    #     model_name=model_name,
    #     dataset_name="train",
    #     sequence_length=seq_len,
    #     target_types=cfg.data.eligible_target_types_train
    # )

    val_tracker_ood = ExperimentTracker(
        model_name=model_name,
        dataset_name="val_ood",
        sequence_length=seq_len_validation,
        target_types=cfg.data.eligible_target_types_val_ood,
        unlearning=cfg.loop.unlearning
    )

    val_tracker_id = ExperimentTracker(
        model_name=model_name,
        dataset_name="val_id",
        sequence_length=seq_len_validation,
        target_types=cfg.data.eligible_target_types_val_id,
        unlearning=cfg.loop.unlearning
    )

    # train_tracker.to(device)
    val_tracker_ood.to(device)
    val_tracker_id.to(device)

    

    # -----------------------
    # Optimizer / Criterion / Scheduler
    # -----------------------
    optimizer = construct_memoryllm_optimizer(model=model_wrapper, cfg=cfg)
    scheduler = construct_memoryllm_scheduler(optimizer=optimizer, cfg=cfg)

    if cfg.model.checkpoint_dict_path is not None:
        rprint(f"Loading checkpoint from {cfg.model.checkpoint_dict_path} ...")
        start_epoch, _, global_step = load_checkpoint(
            train_tracker=None,
            model=model_wrapper,
            val_tracker_id=val_tracker_id,
            val_tracker_ood=val_tracker_ood,
            optimizer=optimizer,
            scheduler=scheduler,
            path=cfg.model.checkpoint_dict_path,
            device=device
        )
        rprint(f"Loaded checkpoint from {cfg.model.checkpoint_dict_path}, resuming at epoch {start_epoch}, global step {global_step}")


    model = DDP(model_wrapper, 
                # find_unused_parameters=True,
                find_unused_parameters=False,
                device_ids=[local_rank],
                output_device=local_rank)
    model_in = model.module
    # -----------------------
    # Epoch loop
    # -----------------------
    epochs_since_improve         = 0
    best_val_id_loss             = float('inf')
    best_val_ood_loss            = float('inf')
    best_epoch                   = -1

    safe_wandb_log({"debug/learning_rate": optimizer.param_groups[0]["lr"]}, step=global_step)

    for epoch in range(start_epoch, max_epochs):
        should_stop_epoch = False
        rprint(f"# EPOCH {epoch+1}/{max_epochs} START ###")

        # -----------------------
        # Per Epoch Reproducibility
        # -----------------------
        epoch_seed = base_seed + epoch*1000
        process_seed = seed_everything(epoch_seed, rank=dist.get_rank())
        print(f"[rank {dist.get_rank()}] Reset random seed for epoch {epoch} with epoch_seed={epoch_seed}, process_seed={process_seed}")
        
        
        # -----------------------
        # Load & Shuffle Training Datasets
        # -----------------------
        rprint("loading training datasets ...")
        training_datasets: list[DocDataset] = [
            DocDataset(path, shuffle=True, limit=cfg.data.train_limit)
            for path in cfg.data.training_datasets
        ]

        # -----------------------
        # Train
        # -----------------------
        rprint(f"### starting epoch {epoch+1}/{max_epochs} training for rank {dist.get_rank()} ###")
        global_step = train_1_epoch(
                        model                         = model,
                        training_datasets             = training_datasets,
                        optimizer                     = optimizer,
                        seq_len                       = seq_len,
                        eligible_target_types         = cfg.data.eligible_target_types_train,
                        holdout_targets               = cfg.data.holdout_targets_train,
                        holdout_target_values         = cfg.data.holdout_target_values_train,
                        io_sampling_params            = io_sampling_params,
                        optimization_params           = cfg.opt,
                        device                        = device,
                        gradient_accumulation_steps   = gradient_accumulation_steps,
                        epoch                         = epoch,
                        unlearning                    = cfg.loop.unlearning,
                        global_step                   = global_step,
                        ablation                      = cfg.model.ablation
                    )

        rprint(f" - Epoch {epoch+1}/{max_epochs} training complete for rank {dist.get_rank()}")

        # -----------------------
        # Validate Training Dataset
        # -----------------------
        
        
        # rprint(f"### starting training data validation on Epoch {epoch+1}/{max_epochs} ###")

        # train_tracker = validate(
        #                     model                  = model,
        #                     validation_data        = training_datasets[0],
        #                     seq_len                = seq_len,
        #                     max_new_tokens         = max_new_tokens,
        #                     eligible_target_types  = cfg.data.eligible_target_types_train,
        #                     holdout_targets        = cfg.data.holdout_targets_train,
        #                     holdout_target_values  = cfg.data.holdout_target_values_train,
        #                     io_sampling_params     = io_sampling_params,
        #                     tracker                = train_tracker,
        #                     device                 = device,
        #                     epoch                  = epoch
        #                 )
        # rprint(f" - Completed training validation on Epoch {epoch+1}/{max_epochs}")
        # train_tracker.log_epoch()
        # train_tracker.reset()

        # -----------------------
        # Validate OOD Dataset
        # -----------------------
        rprint(f"### starting OOD validation on Epoch {epoch+1}/{max_epochs} ###")
        process_seed = seed_everything(base_seed, rank=dist.get_rank())
        torch.cuda.empty_cache()

        val_tracker_ood = validate(
            model=model,
            validation_datasets=validation_datasets_ood,
            seq_len=seq_len_validation,
            max_new_tokens=max_new_tokens,
            eligible_target_types=cfg.data.eligible_target_types_val_ood,
            holdout_targets=cfg.data.holdout_targets_val_ood,
            holdout_target_values=cfg.data.holdout_target_values_val_ood,
            io_sampling_params=io_sampling_params,
            tracker=val_tracker_ood,
            device=device,
            epoch=epoch
        )
        rprint(f" - Completed OOD validation on Epoch {epoch+1}/{max_epochs}")
        torch.cuda.empty_cache()
        val_tracker_ood.log_epoch()
        val_tracker_ood.reset()
        
        rprint(f" - Val Loss OOD: {val_tracker_ood.epoch_results['loss'][-1]:.4f}")
        safe_wandb_log({"loss/val_ood": val_tracker_ood.epoch_results['loss'][-1]}, step=global_step)

        # -----------------------
        # Validate ID Dataset
        # -----------------------
        rprint(f"### starting ID validation on Epoch {epoch+1}/{max_epochs} ###")
        process_seed = seed_everything(base_seed, rank=dist.get_rank())
        val_tracker_id = validate(
            model=model,
            validation_datasets=validation_datasets_id,
            seq_len=seq_len_validation,
            max_new_tokens=max_new_tokens,
            eligible_target_types=cfg.data.eligible_target_types_val_id,
            holdout_targets=cfg.data.holdout_targets_val_id,
            holdout_target_values=cfg.data.holdout_target_values_val_id,
            io_sampling_params=io_sampling_params,
            tracker=val_tracker_id,
            device=device,
            epoch=epoch
        )
        torch.cuda.empty_cache()
        val_tracker_id.log_epoch()
        val_tracker_id.reset()
        rprint(f" - Val Loss ID: {val_tracker_id.epoch_results['loss'][-1]:.4f}")
        safe_wandb_log({"loss/val_id": val_tracker_id.epoch_results['loss'][-1]}, step=global_step)

        # -----------------------
        # Scheduler step
        # -----------------------
        if scheduler is not None:
            scheduler.step()
        safe_wandb_log({"debug/learning_rate": optimizer.param_groups[0]["lr"]},step=global_step)

        
        # if is_main_process():
        dist.barrier()
        if dist.get_rank() == 0:
            # -----------------------
            # Save Plots
            # -----------------------
            if (epoch + 1) % plot_every_n_epochs == 0 or (epoch == max_epochs - 1):
                rprint(f"  - Saving plots for epoch {epoch + 1} because plot every {plot_every_n_epochs} epochs...")
                try:
                    generate_plots(
                        # trackers=[train_tracker, val_tracker_id, val_tracker_ood],
                        trackers=[val_tracker_id, val_tracker_ood],
                        # trackers=[val_tracker_ood],
                        wandb_run=wandb.run,
                        epoch=epoch
                    )
                except Exception as e:
                    rprint(f"   ⚠️ Warning: plotting failed for epoch {epoch + 1} with error: {e}")
            # -----------------------
            # Save Checkpoints
            # -----------------------
            if (epoch + 1)  % checkpoint_every_n_epochs == 0 and (epoch < max_epochs - 1):
                rprint(f"  - checkpointing for epoch {epoch + 1} because checkpoint every {checkpoint_every_n_epochs} epochs...")
                try:
                    save_checkpoint(model=model.module, 
                                optimizer=optimizer, 
                                epoch=epoch, 
                                scheduler=scheduler,
                                # train_tracker=train_tracker,
                                val_tracker_id=val_tracker_id,
                                val_tracker_ood=val_tracker_ood,
                                global_step=global_step,
                                best_val_loss=best_val_id_loss,
                                training_run_dir=training_run_dir,
                                max_checkpoints=max_checkpoints)
                except Exception as e:
                    rprint(f"   ⚠️ Warning: checkpointing failed for epoch {epoch + 1} with error: {e}")
            
            # -----------------------
            # Save Best Model
            # -----------------------
            # Val OOD Model
            cur_val_ood_loss = val_tracker_ood.epoch_results['loss'][-1]
            if (best_val_ood_loss - cur_val_ood_loss) > 0:
                # save_model(
                #     name="best_val_ood_model",
                #     model=model.module,
                #     training_run_dir=training_run_dir,
                #     epoch=epoch,
                #     val_loss=cur_val_ood_loss,
                #     prev_best_val_loss=best_val_ood_loss
                # )
                best_val_ood_loss = cur_val_ood_loss

            # Val ID Model
            cur_val_id_loss = val_tracker_id.epoch_results['loss'][-1]
            if (best_val_id_loss - cur_val_id_loss) > early_stop_min_delta:
                save_model(
                    name="best_val_id_model",
                    model=model.module,
                    training_run_dir=training_run_dir,
                    epoch=epoch,
                    val_loss=cur_val_id_loss,
                    prev_best_val_loss=best_val_id_loss
                )
                best_epoch = epoch
                best_val_id_loss = cur_val_id_loss
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
            
        # -----------------------
        # Early Stopping
        # -----------------------
        dist.barrier()
        if early_stop and epochs_since_improve > early_stop_patience:
            should_stop_epoch = True
            print(f"Early stopping at epoch {epoch+1}. Best epoch was {best_epoch+1} "
                f"with val_ood loss {best_val_ood_loss:.4f}.")
        flag = torch.tensor([int(should_stop_epoch)], device=device)
        dist.broadcast(flag, src=0)   
        should_stop_epoch = bool(flag.item())
        if should_stop_epoch:
            break
        dist.barrier()

dist.barrier()
# -----------------------
# Save results
# -----------------------
if is_main_process():
    # training_metrics_path = os.path.join(training_run_dir, "training_summary.json")
    val_id_metrics_path = os.path.join(training_run_dir, "val_id_summary.json")
    val_ood_metrics_path = os.path.join(training_run_dir, "val_ood_summary.json")
    # with open(training_metrics_path, "w") as f:
    #     json.dump(train_tracker.epoch_results, f, indent=2)
    with open(val_id_metrics_path, "w") as f:
        json.dump(val_tracker_id.epoch_results, f, indent=2)
    with open(val_ood_metrics_path, "w") as f:
        json.dump(val_tracker_ood.epoch_results, f, indent=2)
    
    # wandb.save(training_metrics_path)
    wandb.save(val_id_metrics_path)
    wandb.save(val_ood_metrics_path)
    wandb.finish()

# -----------------------
# Clean shutdown for all ranks
# -----------------------
dist.barrier()      
dist.destroy_process_group()