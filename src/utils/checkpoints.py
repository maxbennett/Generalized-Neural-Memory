import torch
from torch import nn
import os
import shutil
from utils.metrics import BaseMetricsTracker as MetricsTracker
from utils.distributed import rprint
import datetime
import re
import json

def manual_stop() -> bool:
    # look for file named "stop.json" in cwd, and check if it has a field "stop": true
    if os.path.exists("stop.json"):
        with open("stop.json", "r") as f:
            data = json.load(f)
            if data.get("stop", False):
                print("Manual stop detected via stop.json file. Exiting training loop.")
                return True
    return False

def save_checkpoint(*, 
                    model: nn.Module,
                    optimizer: torch.optim.Optimizer, 
                    scheduler: torch.optim.lr_scheduler._LRScheduler = None, 
                    epoch: int, 
                    # train_tracker: MetricsTracker,
                    val_tracker_id: MetricsTracker,
                    val_tracker_ood: MetricsTracker,
                    best_val_loss: float,
                    training_run_dir: str,
                    global_step: int,
                    max_checkpoints: int = 1):

    checkpoint_dir = os.path.join(training_run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save full state
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'best_val_loss': best_val_loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        # 'train_tracker_state_dict': train_tracker.state_dict(),
        'val_tracker_id_state_dict': val_tracker_id.state_dict(),
        'val_tracker_ood_state_dict': val_tracker_ood.state_dict()
    }, checkpoint_path)
    print(f"   ✅ Created checkpoint: {checkpoint_path}")

    #delete old checkpoints
    # delete_checkpoints(training_run_dir=training_run_dir, max_checkpoints=max_checkpoints)

    return checkpoint_path


def delete_checkpoints(*, training_run_dir: str, max_checkpoints: int = 1, delete_all: bool = False):
    checkpoint_dir = os.path.join(training_run_dir, "checkpoints")

    def extract_epoch(filename):
        match = re.search(r"checkpoint_epoch_(\d+)\.pt", filename)
        return int(match.group(1)) if match else -1

    if delete_all: 
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
            print(f"   🗑️ Deleted all checkpoints in {checkpoint_dir}")
        else:
            print(f"   ⚠️ Checkpoint directory does not exist: {checkpoint_dir}")
    else:
        if not os.path.exists(checkpoint_dir):
            print(f"   ⚠️ Checkpoint directory does not exist: {checkpoint_dir}")
            return
        all_checkpoints = [
            f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_epoch_")
        ]
        # Sort by epoch number, newest last
        all_checkpoints = sorted(all_checkpoints, key=extract_epoch)
        if len(all_checkpoints) > max_checkpoints:
            to_delete = all_checkpoints[:-max_checkpoints]
            print(f"   Deleting old checkpoints to keep only the last {max_checkpoints} checkpoints.")
            for ckpt in to_delete:
                os.remove(os.path.join(checkpoint_dir, ckpt))
                print(f"   🗑️ Deleted old checkpoint: {ckpt}")


def load_checkpoint(*, train_tracker: MetricsTracker = None,
                    model: nn.Module ,
                    val_tracker_id: MetricsTracker = None,
                    val_tracker_ood: MetricsTracker = None,
                    optimizer=None, 
                    scheduler=None, 
                    path=None, 
                    device='cuda'):
    # checkpoint = torch.load(path, map_location=device)
    try:
        checkpoint = torch.load(path, map_location='cpu')  # weights_only=True by default in 2.6
    except Exception as e:
        print(f"   ⚠️ Safe load failed ({e}). Retrying with weights_only=False (only if checkpoint is trusted).")
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    # train_tracker.load_state_dict(checkpoint['train_tracker_state_dict'])
    val_tracker_id.load_state_dict(checkpoint['val_tracker_id_state_dict'])
    val_tracker_ood.load_state_dict(checkpoint['val_tracker_ood_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        try:
            import bitsandbytes as bnb
            is_bnb = isinstance(optimizer, bnb.optim.PagedAdamW8bit)
        except Exception:
            is_bnb = False

        if not is_bnb:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
        else:
            # For bnb, keep state on CPU and let paged optimizer manage memory.
            pass
                    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    rprint(f"   📦 Loaded checkpoint from {path}")

    global_step = checkpoint.get('global_step', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    epoch_idx_to_start_at = checkpoint['epoch'] + 1
    return epoch_idx_to_start_at, best_val_loss, global_step


def save_model(*, name: str, 
                model: nn.Module,
                    training_run_dir: str,
                    epoch: int,
                    val_loss: float,
                    prev_best_val_loss: float):
    # create artifacts dir if it doesn't exist
    print(f"   💾 Saving best model as {name}.pth ...")
    os.makedirs(os.path.join(training_run_dir, "artifacts"), exist_ok=True)

    best_model_path = os.path.join(training_run_dir, f"artifacts/{name}.pth")
    torch.save(model.state_dict(), best_model_path)

    metadata = {
        "epoch_model_is_from": (epoch+1),
        "checkpoint_model_is_from": f"checkpoint_epoch_{epoch+1}",
        "val_loss": val_loss,
        "prev_best_val_loss": prev_best_val_loss,
        "saved_at": datetime.datetime.now().isoformat(),
    }
    os.makedirs(os.path.join(training_run_dir, "results"), exist_ok=True)
    
    with open(os.path.join(training_run_dir, "results", f"{name}_checkpoint_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   🌟 Best model saved at {best_model_path} with best validation loss {val_loss}, over previous best {prev_best_val_loss}")

     # --- If learner uses LoRA, also save the adapter folder for easy reuse ---
    try:
        from peft import PeftModel
        is_peft = isinstance(model, nn.Module) and hasattr(model, "model") and isinstance(model.model, PeftModel)
    except Exception:
        is_peft = False

    if is_peft:
        adapter_dir = os.path.join(training_run_dir, f"artifacts/best_trained_learner_adapter")
        os.makedirs(adapter_dir, exist_ok=True)
        # Saves only adapter weights + adapter config (no base weights)
        model.model.save_pretrained(adapter_dir)
        # Save tokenizer for convenience
        if hasattr(model, "tokenizer"):
            model.tokenizer.save_pretrained(adapter_dir)
        print(f"   🧩 Saved LoRA adapter to {adapter_dir}")

# def save_best_model(*, meta_learner: nn.Module= None,
#                     learner: nn.Module = None,
#                     training_run_dir: str,
#                     epoch: int,
#                     val_loss: float,
#                     prev_best_val_loss: float):
#     # create artifacts dir if it doesn't exist
#     os.makedirs(os.path.join(training_run_dir, "artifacts"), exist_ok=True)
#     if meta_learner is not None:
#         best_ml_model_path = os.path.join(training_run_dir, "artifacts/best_trained_metalearner.pth")
#         torch.save(meta_learner.state_dict(), best_ml_model_path)
#     if learner is not None:
#         best_l_model_path = os.path.join(training_run_dir, "artifacts/best_trained_learner.pth")
#         torch.save(learner.state_dict(), best_l_model_path)

#     metadata = {
#         "metalearner_saved": True if meta_learner is not None else False,
#         "learner_saved": True if learner is not None else False,
#         "epoch_model_is_from": (epoch+1),
#         "checkpoint_model_is_from": f"checkpoint_epoch_{epoch+1}",
#         "val_loss": val_loss,
#         "prev_best_val_loss": prev_best_val_loss,
#         "saved_at": datetime.datetime.now().isoformat(),
#     }
#     os.makedirs(os.path.join(training_run_dir, "results"), exist_ok=True)
    
#     if meta_learner is not None:
#         with open(os.path.join(training_run_dir, "results", "best_trained_meta_learner_checkpoint_metadata.json"), 'w') as f:
#             json.dump(metadata, f, indent=2)
#         print(f"   🌟 Best meta-learner model saved at {best_ml_model_path} with best validation loss {val_loss}, over previous best {prev_best_val_loss}")
#     if learner is not None:
#         with open(os.path.join(training_run_dir, "results", "best_trained_learner_checkpoint_metadata.json"), 'w') as f:
#             json.dump(metadata, f, indent=2)
#         print(f"   🌟 Best learner model saved at {best_l_model_path} with best validation loss {val_loss}, over previous best {prev_best_val_loss}")
    
#      # --- If learner uses LoRA, also save the adapter folder for easy reuse ---
#     try:
#         from peft import PeftModel
#         is_peft = isinstance(learner, nn.Module) and hasattr(learner, "model") and isinstance(learner.model, PeftModel)
#     except Exception:
#         is_peft = False

#     if is_peft:
#         adapter_dir = os.path.join(training_run_dir, f"artifacts/best_trained_learner_adapter")
#         os.makedirs(adapter_dir, exist_ok=True)
#         # Saves only adapter weights + adapter config (no base weights)
#         learner.model.save_pretrained(adapter_dir)
#         # Save tokenizer for convenience
#         if hasattr(learner, "tokenizer"):
#             learner.tokenizer.save_pretrained(adapter_dir)
#         print(f"   🧩 Saved LoRA adapter to {adapter_dir}")