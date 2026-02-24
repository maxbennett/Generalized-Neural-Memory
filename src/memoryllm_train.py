import torch
from torch.utils.data import DataLoader, DistributedSampler
from functools import partial
from utils.distributed import rprint, safe_wandb_log, safe_wandb_save
from instructions import evaluate_formatting, update_format
from utils.data_helpers import filter_io_pairs_for_training, filter_all_future_paraphrases, chunk_list
from gnm_data import DocDataset, docs_collate_fn
from utils.metrics import BaseMetricsTracker
from torchmetrics import MeanMetric
import torch.distributed as dist
from config_memoryllm_train import OptCfg
from torch.profiler import profile, ProfilerActivity
from contextlib import nullcontext
from gnm import MemoryLLMChatWrapper


    
def train_1_epoch(*,
            model,
            seq_len,
            device,
            optimizer: torch.optim.Optimizer,
            eligible_target_types,
            holdout_targets,
            holdout_target_values,
            io_sampling_params,
            training_datasets: list[DocDataset],
            optimization_params: OptCfg,
            gradient_accumulation_steps: int,
            global_step: int = 0,
            unlearning: bool = False,
            ablation: bool = False,
            epoch: int = 0):
    
    model.train()
    model_in = model.module if hasattr(model, "module") else model
    optimizer.zero_grad(set_to_none=True)

    loss_tracker = MeanMetric().to(device)
    
    step_count_this_rank = torch.tensor(0, device=device) 
    step_count_all_ranks = torch.tensor(0, device=device)

    for data_idx, train_data in enumerate(training_datasets):
        rprint(f"Starting training on dataset {data_idx + 1}/{len(training_datasets)}")

        dl = DataLoader(
                    dataset      = train_data,
                    batch_size   = 1 * seq_len,
                    sampler      = DistributedSampler(train_data, shuffle=True),
                    drop_last    = True,
                    shuffle      = False,  
                    collate_fn   = partial(docs_collate_fn, 
                                        train_or_test="train",
                                        eligible_target_types=eligible_target_types,
                                        holdout_targets=holdout_targets,
                                        holdout_target_values=holdout_target_values,
                                        sequence_length=seq_len - 1 if unlearning else seq_len,
                                        io_sampling_params=io_sampling_params,
                                        unlearning=unlearning),
                    pin_memory   = True
                )
        dl.sampler.set_epoch(epoch)

        for i, (documents, learning_instructions, input_output_pairs, metadata) in enumerate(dl): # each is a list of lists, where first dimension index 0 is ALL the first elements across the batch
            # rprint(f"starting batch {i+1}/{len(dl)}")
            batch_size = len(documents[0])
            expected_num_chunks = io_sampling_params.max_total_io_pairs_per_sample / io_sampling_params.num_io_pairs_to_chunk  # float
            expected_num_el_per_optim_step = (
                gradient_accumulation_steps
                * (0.5 * ((seq_len**2) + seq_len) )  # float math
                * batch_size
                * expected_num_chunks
            )

            for batch_idx in range(batch_size):

                target_format = None
                

                for seq_idx_of_doc in range(seq_len):
                    target_type = metadata[seq_idx_of_doc][batch_idx]['target_type_to_learn']

                    # Rebuild memory each seq_idx from scratch
                    with torch.no_grad():
                        model_in.reset_memory()
                        if seq_idx_of_doc > 0:
                            for prev_seq_idx in range(seq_idx_of_doc):
                                model_in.memorize(document = documents[prev_seq_idx][batch_idx], learning_instruction =learning_instructions[prev_seq_idx][batch_idx])

                    # ---------------
                    # update refusal/format target for seq
                    # ---------------
                    if target_type == "formats":
                        target_format = metadata[seq_idx_of_doc][batch_idx]['format_in_doc']
                        if target_format is None:
                            raise ValueError("Target format is None during training when target type is formats.")

                    # ---------------
                    # create chunks of io pairs to run
                    # ---------------
                    io_pairs_to_chunk = []
                    for seq_idx_of_io_pairs in range(seq_idx_of_doc + 1): 
                        pairs_to_run = filter_io_pairs_for_training(input_output_pairs[seq_idx_of_io_pairs][batch_idx], seq_idx_of_doc, seq_idx_of_io_pairs, io_sampling_params)
                        # assert len(pairs_to_run) > 0, f"After filtering, input-output pairs for seq idx of doc {seq_idx_of_doc}, seq idx of io pairs {seq_idx_of_io_pairs}, batch idx {batch_idx} are empty, you are likely filtering too much."
                        if len(pairs_to_run) == 0:
                            continue
                        pairs_to_run = update_format(pairs_to_run, target_format)
                        for pair in pairs_to_run:
                            pair["seq_idx_of_io_pairs"] = seq_idx_of_io_pairs
                        io_pairs_to_chunk.extend(pairs_to_run)
                    if len(io_pairs_to_chunk) == 0:
                        continue
                    chunks = chunk_list(io_pairs_to_chunk, io_sampling_params.num_io_pairs_to_chunk)
                    
                    # ---------------
                    # clip chunks
                    # ---------------
                    num_chunks_this_rank = torch.tensor(len(chunks), device=device)
                    min_chunks = torch.tensor(0, device=device)
                    min_chunks.copy_(num_chunks_this_rank)
                    dist.all_reduce(min_chunks, op=dist.ReduceOp.MIN)
                    min_chunks_count = min_chunks.item()
                    chunks = chunks[:min_chunks_count]

                    for pairs in chunks:
                        # don't apply gradients over memorize_step if ablation is True
                        if ablation:
                            with torch.no_grad():
                                model_in.memorize(document=documents[seq_idx_of_doc][batch_idx], learning_instruction=learning_instructions[seq_idx_of_doc][batch_idx])
                        else:
                            model_in.memorize(document=documents[seq_idx_of_doc][batch_idx], learning_instruction=learning_instructions[seq_idx_of_doc][batch_idx])
                        full_texts, _, answer_texts = model_in.get_str_inputs_from_probes(pairs)
                    
                        input_ids, attention_mask, labels = model_in.get_input_ids_and_attn(input_texts=full_texts, 
                                                                                            target_output_texts=answer_texts)
                        # ---------
                        # Forward
                        # ---------
                        out = model(
                            input_ids=input_ids,
                            labels=labels,
                            attention_mask=attention_mask
                        )

                        # ---------
                        # Loss & backward with grad accumulation
                        # ---------
                        step_count_this_rank += 1
                        loss = out.loss / expected_num_el_per_optim_step
                        loss.backward()
                        loss_tracker.update(out.loss.detach())
                        model_in.delete_last_memory_update()
                        del out

            # ---------
            # OPTIMIZER STEP
            # ---------
            if (i + 1) % gradient_accumulation_steps == 0:
                step_count_all_ranks.copy_(step_count_this_rank)
                dist.all_reduce(step_count_all_ranks, op=dist.ReduceOp.SUM)
                step_count = step_count_all_ranks.item() + global_step
                
                # Clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=optimization_params.max_grad_norm)
                grad_clipped = float(grad_norm > optimization_params.max_grad_norm)

                safe_wandb_log({"debug/grad_norm": float(grad_norm),
                                "debug/grad_clipped": grad_clipped
                                }, step=step_count)
                dist.barrier()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                mean_loss = loss_tracker.compute().item()
                safe_wandb_log({"loss/train": mean_loss}, step=step_count)
                loss_tracker.reset()
                dist.barrier()
                rprint(f" - Training Step [{step_count}] (Batch [{i+1}/{len(dl)}]) loss: {mean_loss:.4f}")

    return step_count

def validate(*,
            model,
            seq_len,
            max_new_tokens,
            holdout_target_values,
            device,
            eligible_target_types,
            holdout_targets,
            io_sampling_params,
            epoch: int = 0,
            output_dir: str | None = None,
            validation_datasets: list[DocDataset],
            tracker: BaseMetricsTracker):
    
    model.eval()
    model_in = model.module if hasattr(model, "module") else model
    tokenizer = model_in.tokenizer
    save_up_to_batch_idx = 1 # change to save a batch


    for data_idx, val_data in enumerate(validation_datasets):
        rprint(f"Starting validation on dataset {data_idx + 1}/{len(validation_datasets)}")
        # seq_len = seq_len - 1 if tracker.unlearning else seq_len
        dl = DataLoader(
                    dataset      = val_data,
                    batch_size   = 1 * (seq_len - 1 if tracker.unlearning else seq_len),
                    sampler      = DistributedSampler(val_data, shuffle=False),
                    drop_last    = True,
                    shuffle      = False,  
                    collate_fn   = partial(docs_collate_fn, 
                                        train_or_test="test",
                                        eligible_target_types=eligible_target_types,
                                        holdout_targets=holdout_targets,
                                        holdout_target_values=holdout_target_values,
                                        sequence_length=seq_len - 1 if tracker.unlearning else seq_len,
                                        io_sampling_params=io_sampling_params,
                                        unlearning=tracker.unlearning),
                    pin_memory   = True
                )
        dl.sampler.set_epoch(0)
        # seq_len = seq_len + 1 if tracker.unlearning else seq_len

        max_profiles_per_seq = 10  
        flop_logs = [0 for _ in range(seq_len)]
        
        temp_results = []
        with torch.inference_mode():
            for i, (documents, learning_instructions, input_output_pairs, metadata) in enumerate(dl): # each is a list of lists, where first dimension index 0 is ALL the first elements across the batch
                rprint(f"starting batch {i+1}/{len(dl)}")
                
                batch_size = len(documents[0])
                for batch_idx in range(batch_size):
                    target_format = None
                    format_instruction_seq_idx = None
                    model_in.reset_memory()
                    
                    for seq_idx_of_doc in range(seq_len):
                        doc= documents[seq_idx_of_doc][batch_idx]
                        li = learning_instructions[seq_idx_of_doc][batch_idx]
                        target = metadata[seq_idx_of_doc][batch_idx]['target_to_learn']
                        target_type = metadata[seq_idx_of_doc][batch_idx]['target_type_to_learn']
                        model_in.memorize(document=doc, learning_instruction=li)

                        # ---------------
                        # update format target for seq
                        # ---------------
                        if target_type == "formats" or target_type == "fact_format_compositions":
                            target_format = metadata[seq_idx_of_doc][batch_idx]['format_in_doc']
                            if target_format is None:
                                raise ValueError("Target format is None during validation when target type is formats.")
                            if format_instruction_seq_idx is None:
                                format_instruction_seq_idx = seq_idx_of_doc
                            

                        # ---------------
                        # create chunks of io pairs to run
                        # ---------------
                        io_pairs_to_chunk = []
                        for seq_idx_of_io_pairs in range(seq_idx_of_doc + 1): 
                        # for seq_idx_of_io_pairs in range(seq_len): 
                            # if seq_idx_of_io_pairs != seq_idx_of_doc:
                            #     continue
                            pairs_to_run_pre_filter = input_output_pairs[seq_idx_of_io_pairs][batch_idx]
                            pairs_to_run= filter_all_future_paraphrases(pairs_to_run_pre_filter, seq_idx_of_doc)
                            pairs_to_run = update_format(pairs_to_run, target_format)
                            for pair in pairs_to_run:
                                pair["seq_idx_of_io_pairs"] = seq_idx_of_io_pairs
                            io_pairs_to_chunk.extend(pairs_to_run)

                        chunks = chunk_list(io_pairs_to_chunk, io_sampling_params.num_io_pairs_to_chunk)
            
                        # ---------------
                        # clip chunks
                        # ---------------
                        num_chunks_this_rank = torch.tensor(len(chunks), device=device)
                        min_chunks = torch.tensor(0, device=device)
                        min_chunks.copy_(num_chunks_this_rank)
                        dist.all_reduce(min_chunks, op=dist.ReduceOp.MIN)
                        min_chunks_count = min_chunks.item()
                        chunks = chunks[:min_chunks_count]

                        # ---------------
                        # Run chunks
                        # ---------------
                        for pairs in chunks:
                            full_texts, input_texts, answer_texts = model_in.get_str_inputs_from_probes(pairs)
                            input_ids, attention_mask, _ = model_in.get_input_ids_and_attn(input_texts=input_texts, 
                                                                                                target_output_texts=None)


                            if flop_logs[seq_idx_of_doc] < max_profiles_per_seq:
                                should_profile = True
                            else:
                                should_profile = False
                            should_profile = False  # disable profiling for now

                            if should_profile:
                                with profile(
                                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                                    with_flops=True,
                                    record_shapes=False
                                ) as prof:
                                    out = model_in(
                                        input_ids=input_ids,
                                        # labels=labels,
                                        attention_mask=attention_mask
                                    )
                                
                                total_flops = 0
                                for evt in prof.key_averages():
                                    if getattr(evt, "flops", None) is not None:
                                        total_flops += evt.flops
                                
                                tracker.update_flops_metrics(total_flops, seq_idx_of_doc=seq_idx_of_doc)
                                
                                flop_logs[seq_idx_of_doc] += 1

                            out_gen = model_in.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                max_new_tokens=max_new_tokens
                            )
                            

                            gen_only_ids = []
                            for b in range(out_gen.sequences.size(0)):
                                gen_only_ids.append(out_gen.sequences[b, input_ids.shape[1]:])

                            generated_texts = tokenizer.batch_decode(
                                                gen_only_ids,
                                                skip_special_tokens=False,
                                                clean_up_tokenization_spaces=False
                                            )
                            
                            
                            for b_idx, (pair, gen_text) in enumerate(zip(pairs, generated_texts)):
                                target_output_str = pair["target_output_no_format"].lstrip()
                                non_target_output_str = pair["non_target_output_no_format"].lstrip()
                                meta_pair_idx = metadata[pair['seq_idx_of_io_pairs']][batch_idx]

                                if "fact_format_compositions" in eligible_target_types:
                                    if metadata[pair['seq_idx_of_io_pairs']][batch_idx]['target_type_to_learn'] != "fact_format_compositions":
                                        print(f"[rank {dist.get_rank()}] skipping fact_format_compositions evaluation for non-composition pair, batch element: {i}")
                                        continue

                                # ---------------
                                # format evaluation
                                # ---------------
                                if "formats" in eligible_target_types or "fact_format_compositions" in eligible_target_types:
                                    fmt, tok_offset, str_offset = evaluate_formatting(gen_text, tokenizer=tokenizer)
                                    format_correct = 1 if fmt == target_format else 0

                                    if format_instruction_seq_idx is not None:
                                        if seq_idx_of_doc >= format_instruction_seq_idx:
                                            time_since_format_instruction = seq_idx_of_doc - format_instruction_seq_idx
                                        else:
                                            time_since_format_instruction = None
                                    else:
                                        time_since_format_instruction = None
                                    
                                    tracker.update_format_metrics(
                                        format_correct=format_correct,
                                        seq_idx_of_doc=seq_idx_of_doc,
                                        seq_idx_of_io_pairs=pair['seq_idx_of_io_pairs'],
                                        target_format=target_format,
                                        time_since_format_instruction=time_since_format_instruction,
                                        metadata_blob=meta_pair_idx
                                        # data_idx=data_idx
                                    )
                                else:
                                    fmt = None
                                    tok_offset = 0
                                    str_offset = 0
                                    format_correct = None

                                # ---------------
                                # fact evaluation
                                # ---------------
                                fact_correct = None
                                if not pair['should_refuse']:
                                    gen_len = len(out_gen.scores)
                                    score_idx = max(0, min(tok_offset, gen_len - 1))

                                    tgt_id = tokenizer(target_output_str,     return_tensors="pt", add_special_tokens=False).input_ids[0][0].item()
                                    non_id = tokenizer(non_target_output_str, return_tensors="pt", add_special_tokens=False).input_ids[0][0].item()

                                    row_scores = out_gen.scores[score_idx][b_idx]
                                    v_tgt = row_scores[tgt_id].item()
                                    v_non = row_scores[non_id].item()
                                    fact_correct = int(v_tgt > v_non)
                                    
                                    if "fact_refusal_compositions" in eligible_target_types:
                                        if metadata[pair['seq_idx_of_io_pairs']][batch_idx]['target_type_to_learn'] == "fact_refusal_compositions":
                                            tracker.update_fact_metrics(
                                                fact_correct,
                                                pair=pair,
                                                seq_idx_of_doc=seq_idx_of_doc,
                                                seq_idx_of_io_pairs=pair['seq_idx_of_io_pairs'],
                                                metadata_blob=meta_pair_idx
                                            )
                                    else:
                                        tracker.update_fact_metrics(
                                                fact_correct,
                                                pair=pair,
                                                seq_idx_of_doc=seq_idx_of_doc,
                                                seq_idx_of_io_pairs=pair['seq_idx_of_io_pairs'],
                                                metadata_blob=meta_pair_idx
                                            )
                                    if tracker.unlearning and seq_idx_of_doc == (seq_len - 1) and pair['seq_idx_of_io_pairs'] == (seq_len -1):
                                        tracker.update_unlearning_fact_metrics(
                                                fact_correct,
                                                pair=pair,
                                                seq_idx_of_doc=seq_idx_of_doc,
                                                seq_idx_of_io_pairs=pair['seq_idx_of_io_pairs'],
                                                metadata_blob=meta_pair_idx
                                            )

                                
                                # ---------------
                                # refusal evaluation
                                # ---------------
                                actual_text = gen_text[str_offset:].lstrip()
                                if "refusal_categories" in eligible_target_types or "refusal_subcategories" in eligible_target_types:
                                    tracker.update_refusal_metrics(
                                            generated_text=actual_text,
                                            pair=pair,
                                            seq_idx_of_doc=seq_idx_of_doc,
                                            seq_idx_of_io_pairs=pair['seq_idx_of_io_pairs'],
                                            metadata_blob=meta_pair_idx
                                        )
                                elif "fact_refusal_compositions" in eligible_target_types:
                                    if metadata[pair['seq_idx_of_io_pairs']][batch_idx]['target_type_to_learn'] == "fact_refusal_compositions":
                                        tracker.update_refusal_metrics(
                                                generated_text=actual_text,
                                                pair=pair,
                                                seq_idx_of_doc=seq_idx_of_doc,
                                                seq_idx_of_io_pairs=pair['seq_idx_of_io_pairs'],
                                                metadata_blob=meta_pair_idx
                                            )
                                        tracker.count_compositions()
                                    else:
                                        tracker.count_non_compositions()

                                
                                if dist.get_rank() == 0 and i <= save_up_to_batch_idx:
                                    try:
                                        temp_result = {}
                                        temp_result['gen_text']                     = gen_text
                                        temp_result['actual_text']                  = actual_text
                                        temp_result['input_text']                   = input_texts[b_idx]
                                        temp_result['fact_correct']                 = fact_correct
                                        temp_result['format_correct']               = format_correct
                                        temp_result['target_format']                = target_format
                                        temp_result['evaluated_fmt_of_gen_text']    = fmt
                                        temp_result['tok_offset']                   = tok_offset
                                        temp_result['str_offset']                   = str_offset
                                        temp_result['pairs']                        = pair
                                        temp_result['target']                       = target
                                        temp_result['target_type']                  = target_type
                                        temp_result['document']                     = doc
                                        temp_result['learning_instruction']         = li
                                        temp_result['seq_idx_of_doc']               = seq_idx_of_doc
                                        temp_result['seq_idx_of_io_pairs']          = pair['seq_idx_of_io_pairs']
                                        temp_results.append(temp_result)
                                    except Exception as e:
                                        print(f"   ⚠️ Warning: could not save temp result for batch idx {batch_idx}, seq idx {seq_idx_of_doc}: {e}")

                                    if i == save_up_to_batch_idx:
                                        if output_dir is None:
                                            output_dir = "results"
                                        with open(f"{output_dir}/temp_results_{tracker.dataset_name}_epoch_{epoch}.jsonl", "w") as f:
                                            import json
                                            json.dump(temp_results, f, indent=4)
                                        # safe_wandb_save(f"results/temp_results_validation_{tracker.dataset_name}_epoch_{epoch}.jsonl")
                            del out_gen
                            
                            # ------------
                            # Loss computation if a target
                            # ------------

                            if seq_idx_of_doc >= min([pair['seq_idx_of_io_pairs'] for pair in pairs]):
                                input_ids_fwd, attention_mask_fwd, labels_fwd = model_in.get_input_ids_and_attn(input_texts=full_texts, 
                                                                                                                target_output_texts=answer_texts)
                                
                                out_fwd = model(
                                    input_ids=input_ids_fwd,
                                    labels=labels_fwd,
                                    attention_mask=attention_mask_fwd
                                )

                                tracker.update_loss(out_fwd.loss)
                                if dist.get_rank() == 0 and i < save_up_to_batch_idx:
                                    for i in range(0, len(input_texts)):
                                        temp_results[-i-1]['loss'] = out_fwd.loss.item()

                                del out_fwd
                        
    return tracker