import re
import random
import torch

# def filter_future_paraphrases_once_past(input_output_pairs_to_run_pre_filter, seq_idx_of_doc):
#     filtered_pairs = []
#     for pair in input_output_pairs_to_run_pre_filter:
#         if pair.get('type') == 'future_paraphrase':
#             # remove future paraphrases once that document has been passed in
#             if pair['origin_seq_idx'] <= seq_idx_of_doc:
#                 continue  # skip this future paraphrase since we have passed
            
#             # now skip future paraphrases if they are more than 3 sequences ahead
#             if pair['origin_seq_idx'] > seq_idx_of_doc + 3:
#                 continue  # skip this future paraphrase as it's too far in the future
        
#         filtered_pairs.append(pair)
    
#     return filtered_pairs
# def optimizer_to(optimizer, device):
#     for state in optimizer.state.values():
#         for k, v in state.items():
#             if torch.is_tensor(v):
#                 state[k] = v.to(device)

def filter_all_future_paraphrases(input_output_pairs_to_run_pre_filter, seq_idx_of_doc):
    filtered_pairs = []
    for pair in input_output_pairs_to_run_pre_filter:
        if pair['type'] == 'future_paraphrase':
            continue  # skip this future paraphrase as it's too far in the future
        
        filtered_pairs.append(pair)
    
    return filtered_pairs

# def filter_io_pairs_if_past_el(input_output_pairs_to_run_pre_filter, seq_idx_of_doc, seq_idx_of_io_pairs):
#     filtered_pairs = []
#     for pair in input_output_pairs_to_run_pre_filter:
#         sequences_ago = seq_idx_of_doc - seq_idx_of_io_pairs
#         if sequences_ago > 3 and pair['type'] == 'true_output_neighbor':
#             continue  # skip old true output neighbors that are not facts to learn
#         if sequences_ago > 3 and pair['fact_id_targeted_for_fact_learning'] == False:
#             continue  # skip old paraphrases that are not facts to learn
#         if sequences_ago > 3 and pair['fact_id_targeted_for_refusal_learning'] == False:
#             continue  # skip old paraphrases that are not facts to refuse
        
#         filtered_pairs.append(pair)
    
#     return filtered_pairs


def unique_on(dict_list: list[dict], unique_on_attr:str):
    new_dict_list = list()
    unique_values = set()

    for item in dict_list:
        if item[unique_on_attr] not in unique_values:
            new_dict_list.append(item)
            unique_values.add(item[unique_on_attr])
    return new_dict_list


def strip_json_format(text) -> str:
    """
    Strips leading JSON formatting from text, if present.
    E.g., '{"response":"The answer is 42."}' -> 'The answer is 42.'
    """
    s = text.lstrip()
    json_preamble_re = r'^\{\s*(?P<q>["\'])(?P<key>[^"\']+)(?P=q)\s*:\s*(?P<q2>["\'])(?P<content>.*)(?P=q2)\s*\}\s*$'
    m = re.match(json_preamble_re, s)
    if m:
        return m.group("content")
    return text

def strip_xml_format(text) -> str:
    """
    Strips leading XML formatting from text, if present.
    E.g., '<response>The answer is 42.</response>' -> 'The answer is 42.'
    """
    s = text.lstrip()
    xml_preamble_re = r'^\<\s*(?P<tag>[a-zA-Z0-9_]+)\s*\>(?P<content>.*)\<\/\s*(?P=tag)\s*\>\s*$'
    m = re.match(xml_preamble_re, s)
    if m:
        return m.group("content")
    return text

def strip_formatting(text: str) -> str:
    """
    Strips leading JSON or XML formatting from text, if present.
    """
    text = strip_json_format(text)
    text = strip_xml_format(text)
    return text

def randomize_document(document:str) -> str:
    # Split into header and lines
    header, *items = document.split("*")
    
    # Clean up each item (strip whitespace and ignore empties)
    items = [item.strip() for item in items if item.strip()]
    
    # Shuffle the list items
    random.shuffle(items)
    
    # Reassemble
    randomized = header.strip() + "\n"
    randomized += "".join(f"* {item}\n" for item in items)
    
    return randomized


def chunk_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def strip_bos_from_left_padded_batch(input_ids, attention_mask, pad_id, bos_id):
    """
    Strip a BOS token from each row of a left-padded batch *only if*
    the first non-pad token is BOS.
    For MemoryLLM, we actually drop one column entirely (no trailing pad).
    """
    ids = input_ids.clone()
    attn = attention_mask.clone()

    # first non-pad idx per row (cast so argmax works on CUDA)
    first_tok_idx = (attn != 0).int().argmax(dim=1)  # [B]
    B, T = ids.size()
    rows = torch.arange(B, device=ids.device)

    has_bos = (ids[rows, first_tok_idx] == bos_id)

    for r in rows[has_bos]:
        s = int(first_tok_idx[r].item())
        if s < T - 1:
            ids[r, s:T-1] = ids[r, s+1:T].clone()
            attn[r, s:T-1] = attn[r, s+1:T].clone()
        # don’t overwrite the last position with pad;
        # we’ll remove it entirely below

    # drop the last column globally (uniform width)
    ids = ids[:, :-1]
    attn = attn[:, :-1]

    return ids, attn

def strip_bos_by_masking(input_ids, attention_mask, tokenizer, model=None):
    """
    Left-padded batch:
      - If add_bos_embedding=True and the first non-pad token is BOS,
        replace that BOS with PAD and set mask=0.
      - No shifting, no width change.
    """
    # if not getattr(model.config, "add_bos_embedding", False):
    #     return input_ids, attention_mask
    if tokenizer.bos_token_id is None or tokenizer.pad_token_id is None:
        return input_ids, attention_mask

    ids  = input_ids.clone()
    attn = attention_mask.clone()

    # first non-pad index per row (CUDA-safe)
    first = (attn != 0).int().argmax(dim=1)            # [B]
    B, T  = ids.shape
    rows  = torch.arange(B, device=ids.device)

    # rows where the first real token is BOS
    sel = ids[rows, first] == tokenizer.bos_token_id
    if sel.any():
        r_idx = rows[sel]
        s_idx = first[sel]
        ids[r_idx, s_idx]  = tokenizer.pad_token_id    # turn BOS into PAD
        attn[r_idx, s_idx] = 0                         # and mask it out

    return ids, attn

def align_delta_to_batch(delta: torch.Tensor, B: int) -> torch.Tensor:
    # delta expected shape: [batch, L, T, d]
    if delta.dim() == 3:
        delta = delta.unsqueeze(0)  # [1, L, T, d]

    nb = delta.size(0)

    if nb == B:
        return delta  # already aligned

    if nb == 1 and B > 1:
        # Broadcast same delta to every item in the batch
        return delta.expand(B, -1, -1, -1)

    if B == 1 and nb > 1:
        # Pick which row corresponds to this sample (often 0)
        return delta[:1]  # or delta[idx:idx+1] if you track pairing

    # Fallback: if nb != 1 and nb != B, this is a construction bug upstream
    raise ValueError(f"delta_memory batch {nb} != input batch {B}; fix the construction (merge along dim=2, not dim=0).")

def filter_io_pairs_for_training(input_output_pairs_to_run_pre_filter, seq_idx_of_doc,seq_idx_of_io_pairs, sampling_config):
    num_in_future_to_pull_back_future_paraphrase = sampling_config.num_in_future_to_pull_back_future_paraphrase
    # num_in_past_to_pull_forward_true_output_neighbor = sampling_config.num_in_past_to_pull_forward_true_output_neighbor
    # num_in_past_to_pull_forward_paraphrase_not_target_to_learn = sampling_config.num_in_past_to_pull_forward_paraphrase_not_target_to_learn
    # sequences_ago = seq_idx_of_doc - seq_idx_of_io_pairs

    filtered_pairs = []
    for pair in input_output_pairs_to_run_pre_filter:
        # skip old true output neighbors that are not facts to learn
        # if sequences_ago > num_in_past_to_pull_forward_true_output_neighbor and pair['type'] == 'true_output_neighbor':
        #     continue  
        
        # # skip old paraphrases that are not facts to learn
        # if sequences_ago > num_in_past_to_pull_forward_paraphrase_not_target_to_learn and pair['fact_id_targeted_for_fact_learning'] == False and pair['fact_id_targeted_for_refusal_learning'] == False:
        #     continue  

        if pair['type'] == 'future_paraphrase':
            # remove future paraphrases once that document has been passed in
            if pair['origin_seq_idx'] <= seq_idx_of_doc:
                continue  
            
            # now skip future paraphrases too far in future
            if pair['origin_seq_idx'] > seq_idx_of_doc + num_in_future_to_pull_back_future_paraphrase:
                continue  
        
        filtered_pairs.append(pair)
    
    return filtered_pairs