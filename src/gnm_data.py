import json
from torch.utils.data import Dataset
import random
from utils.distributed import rprint
from instructions import get_learning_instruction, get_random_format, get_target, apply_formatting, get_eligible_fact_refusal_compositions, get_eligible_fact_format_compositions
from utils.data_helpers import unique_on
from config_memoryllm_train import SamplingCfg
import copy


# Core dataset we use for training
class DocDataset(Dataset):
    def __init__(self, 
                data_path: str, # datapath to the json file
                shuffle: bool = False,
                limit: int = None):
    
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        if limit is not None:
            if shuffle:
                random.shuffle(self.data)
            self.data = self.data[:limit]
            print(f"data limit set to {limit}, loaded {len(self.data)} samples from {data_path}")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if all(isinstance(p, list) and not any(p) for p in self.data[idx]['input_output_pairs']):
            raise ValueError(f"Input-output pairs for index {idx} are all empty sublists, you are likely filtering too much in max_io_types.")
        
        document = self.data[idx].get('document', '')
        metadata = self.data[idx].get('metadata', {})
        learning_instruction = self.data[idx].get('learning_instruction', None) # if data has none, then we will randomize it later
        input_output_pairs = self.data[idx]['input_output_pairs']

        return {
            'document': document,
            'learning_instruction': learning_instruction,
            'input_output_pairs': input_output_pairs,
            'metadata': metadata
        }



def docs_collate_fn(batch, *,
                    train_or_test: str,
                    eligible_target_types: list,
                    holdout_targets: list | None,
                    holdout_target_values: list | None,
                    io_sampling_params: dict,
                    sequence_length: int,
                    unlearning: bool = False):


    random_sampling = True if train_or_test == "train" else False
    

    # -----------------
    # Create data collation structures
    # -----------------
    documents = [[] for _ in range(sequence_length)] # first index will return all first elements across all sequences in a batch
    learning_instructions = [[] for _ in range(sequence_length)]
    filtered_io_pairs_by_seq = [[] for _ in range(sequence_length)]
    metadata_blobs = [[] for _ in range(sequence_length)]
    
    # for each sequence in the batch, filter the batch and then structure it into 3 lists
    for i in range(len(batch)):
        documents[i % sequence_length].append(batch[i]['document'])
        filtered_io_pairs_by_seq[i % sequence_length].append(batch[i]['input_output_pairs'])
        metadata_blobs[i % sequence_length].append(batch[i]['metadata'])
        learning_instructions[i % sequence_length].append([])  # will fill in later
        
    # -----------------
    # loop through each sequence
    # -----------------
    for batch_idx in range(len(filtered_io_pairs_by_seq[0])):
        
        # --
        # initialize available_targets at the beginning of each sequence rollout
        # --
        eligible_target_types_ex_formats_ex_compositions = [t for t in eligible_target_types if t != "formats" and t != "fact_refusal_compositions"]
        assert len(eligible_target_types_ex_formats_ex_compositions) > 0, "No available types of things to learn, please check your eligible_target_groups setting."

        format_target_seq_idx = None
        if "formats" in eligible_target_types:
            format_target_seq_idx = random.randint(0, sequence_length - 1)
        
        # now loop through each sequence element in batch
        for seq_idx in range(sequence_length):    

            # ---------
            # get data for this seq idx
            # ----------
            input_output_pairs = filtered_io_pairs_by_seq[seq_idx][batch_idx]
            document = documents[seq_idx][batch_idx]
            metadata = metadata_blobs[seq_idx][batch_idx]

            if "fact_refusal_compositions" in eligible_target_types:
                fact_refusal_matches = get_eligible_fact_refusal_compositions(metadata, holdout_targets=holdout_targets)
            else:
                fact_refusal_matches = None

            # ----------
            # get target for this seq_idx
            # ----------
            if format_target_seq_idx is not None and seq_idx == format_target_seq_idx:
                target_type = "formats"
            elif fact_refusal_matches is not None:
                target_type = "fact_refusal_compositions"
            else:
                target_type = random.choice(eligible_target_types_ex_formats_ex_compositions)
            
            if target_type in {
                    "fact_categories",
                    "fact_subcategories"}:
                inclusion_list = metadata['subcategories_in_doc'] + metadata['categories_in_doc'] 
            elif target_type in {
                    "refusal_categories",
                    "refusal_subcategories"}:
                inclusion_list = [f"{item}_refusal" for item in metadata['subcategories_in_doc'] + metadata['categories_in_doc'] ]
            elif target_type == "fact_refusal_compositions":
                # inclusion_list = get_eligible_fact_refusal_compositions(metadata, holdout_targets=holdout_targets)
                inclusion_list = fact_refusal_matches
                # if len(inclusion_list) == 0: # if fail, just default to learning a fact, not a composition
                #     target_type = "fact_subcategories"
                #     inclusion_list = metadata['subcategories_in_doc']
            elif target_type == "fact_format_compositions":
                inclusion_list = get_eligible_fact_format_compositions(metadata, holdout_targets=holdout_targets)
                if len(inclusion_list) == 0:
                    target_type = "fact_subcategories"
                    inclusion_list = metadata['subcategories_in_doc'] + metadata['categories_in_doc'] 
                # assert len(inclusion_list) > 0, "No eligible fact-format compositions found in document, cannot proceed."
            else:
                inclusion_list = None

            target = get_target(target_type, inclusion_list = inclusion_list, exclusion_list = holdout_targets)
            
            # ----------
            # update document format
            # ----------
            if ("formats" in eligible_target_types or "fact_format_compositions" in eligible_target_types) and target_type != "refusals":
                # sometimes_return_none = True if target_type != "formats" else False
                sometimes_return_none = False
                format_in_doc = get_random_format(holdout_targets, holdout_target_values, sometimes_return_none=sometimes_return_none)
                document = apply_formatting(document, format_in_doc)
            else:
                format_in_doc = None
            
            # ----------
            # update io pairs to accomodate target
            # ----------
            input_output_pairs = update_pairs_target_outputs(input_output_pairs=input_output_pairs, 
                                                                        target=target) 
            
            # ---------
            # filter io pairs given io_sampling_params
            # ---------
            input_output_pairs = filter_io_pairs(
                input_output_pairs=input_output_pairs, 
                io_sampling_params=io_sampling_params,
                random_sampling=random_sampling)

            # ---------
            # add finalized data to collated lists
            # ---------
            learning_instructions[seq_idx][batch_idx]                       = get_learning_instruction(target, random_sampling=random_sampling)
            filtered_io_pairs_by_seq[seq_idx][batch_idx]                    = input_output_pairs
            documents[seq_idx][batch_idx]                                   = document
            metadata_blobs[seq_idx][batch_idx]['target_to_learn']           = target
            metadata_blobs[seq_idx][batch_idx]['target_type_to_learn']      = target_type
            metadata_blobs[seq_idx][batch_idx]['format_in_doc']             = format_in_doc

    if train_or_test == "train":
        filtered_io_pairs_by_seq = pull_forward_paraphrase_into_io_pairs_docs_pick_target(filtered_io_pairs_by_seq)

    if unlearning:
        # take the first element in sequence of each batch, and add an element at the end that flips the target and non_target outputs, as well as swapping those values from the document
        # first_documents = documents[0].copy()
        # first_io_pairs = filtered_io_pairs_by_seq[0].copy()
        # first_metadatas = metadata_blobs[0].copy()
        
        # Add a new sequence position to all data structures
        documents.append([])
        learning_instructions.append([])
        filtered_io_pairs_by_seq.append([])
        metadata_blobs.append([])

        # Add a new sequence position to all data structures
        # documents[-1] = documents[0].deepcopy()
        # learning_instructions[-1] = learning_instructions[0].deepcopy()
        # filtered_io_pairs_by_seq[-1] = filtered_io_pairs_by_seq[0].deepcopy()
        # metadata_blobs[-1] = metadata_blobs[0].deepcopy()

        documents[-1] = copy.deepcopy(documents[0])
        learning_instructions[-1] = copy.deepcopy(learning_instructions[0])
        filtered_io_pairs_by_seq[-1] = copy.deepcopy(filtered_io_pairs_by_seq[0])
        metadata_blobs[-1] = copy.deepcopy(metadata_blobs[0])

        for batch_idx, batch in enumerate(filtered_io_pairs_by_seq[-1]):
            # new_filtered_io_pairs_by_seq = []
            # swapped_doc = first_documents[batch_idx]
            fact_ids_swapped = []
            for p in batch:
                if p['type'] == 'paraphrase' and p['fact_id_targeted_for_fact_learning']:
                    if p['fact_id_targeted_for_fact_learning'] not in fact_ids_swapped:
                        # print(f"Swapping paraphrase fact_id {p['fact_id']} and fact {p['target_output']} in unlearning step.")
                        documents[-1][batch_idx] = documents[-1][batch_idx].replace(p['target_output'].strip(), p['non_target_output'].strip())
                        # print(f"Done swapping paraphrase fact_id {p['fact_id']} and fact {p['target_output']} in unlearning step.")
                        fact_ids_swapped.append(p['fact_id'])
                    p_swapped = p.copy()
                    p['target_output'] = p_swapped['non_target_output']
                    p['non_target_output'] = p_swapped['target_output']
                    # p_swapped['type'] = 'paraphrase'
                    # documents[-1][batch_idx] = swap_doc_values(documents[-1][batch_idx], p['target_output'], p['non_target_output'])
                    
            #     elif p['type'] == 'true_output_neighbor':
            #         p_swapped = p.copy()
            #         new_filtered_io_pairs_by_seq.append(p_swapped)
            
            # # Append to the new sequence position (index = sequence_length)
            # documents[sequence_length].append(swapped_doc)
            # learning_instructions[sequence_length].append(learning_instructions[0][batch_idx])
            # metadata_blobs[sequence_length].append(first_metadatas[batch_idx])
            # filtered_io_pairs_by_seq[sequence_length].append(new_filtered_io_pairs_by_seq)
        

    return documents, learning_instructions, filtered_io_pairs_by_seq, metadata_blobs

# def swap_doc_values(document,value_to_replace,value_to_swap_to) -> str:
#     # find the value in the document and swap it with value_to_swap
#     # this is a placeholder function, actual implementation will depend on document structure
#     return document.replace(value_to_replace, value_to_swap_to)



def update_pair_target_outputs(pair:dict, target:str | None, switch_to_right_whitespace: bool = True) -> dict: 

    p = pair.copy()
    p['should_refuse'] = False
    if switch_to_right_whitespace:
        p['true_output'] = f"{pair['true_output'].lstrip()} "
        p['false_output'] = f"{pair['false_output'].lstrip()} "

    # if target.endswith("-fact_refusal_compositions"):
    #     fact_target, refusal_target, _ = target.split('-')
    # Update outputs
    if p['type'] == 'true_output_neighbor':
        p['target_output'] = p['true_output']
        p['non_target_output'] = p['false_output']
        p['neighbor_of_should_refuse'] = False
        if target.endswith("_refusal"):
            target_cat_or_subcat = target[:-len("_refusal")]
            if p['subcategory'] == target_cat_or_subcat or p['category'] == target_cat_or_subcat:
                p['neighbor_of_should_refuse'] = True
        elif target.endswith("-fact_refusal_compositions"):
            fact_target, refusal_target, _ = target.split('-')
            if p['subcategory'] == refusal_target or p['category'] == refusal_target:
                p['neighbor_of_should_refuse'] = True
    elif p['type'] == 'paraphrase': 
        if target is None:
            p['target_output'] = p['true_output']
            p['non_target_output'] = p['false_output']
        elif target.endswith("_format"):
            target = target[:-len("_format")]
            if target == "all" or p['subcategory'] == target or p['category'] == target:
                p['target_output'] = p['false_output']
                p['non_target_output'] = p['true_output']
            else:
                p['target_output'] = p['true_output']
                p['non_target_output'] = p['false_output']
        elif target == "all" or p['subcategory'] == target or p['category'] == target:
            p['target_output'] = p['false_output']
            p['non_target_output'] = p['true_output']
        elif target.endswith("_refusal"):
            target_cat_or_subcat = target[:-len("_refusal")]
            if p['subcategory'] == target_cat_or_subcat or p['category'] == target_cat_or_subcat:
                p['target_output'] = "Sorry"
                p['non_target_output'] = ""
                p['should_refuse'] = True
            else:
                p['target_output'] = p['false_output']
                p['non_target_output'] = p['true_output']
        elif target.endswith("-fact_refusal_compositions"):
            fact_target, refusal_target, _ = target.split('-')
            if p['subcategory'] == refusal_target or p['category'] == refusal_target:
                p['target_output'] = "Sorry"
                p['non_target_output'] = ""
                p['should_refuse'] = True
            elif p['subcategory'] == fact_target or p['category'] == fact_target:
                p['target_output'] = p['false_output']
                p['non_target_output'] = p['true_output']
            else:
                p['target_output'] = p['true_output']
                p['non_target_output'] = p['false_output']
        else:
            p['target_output'] = p['true_output']
            p['non_target_output'] = p['false_output']
    else:
        raise ValueError(f"Unknown pair type: {pair['type']}")
    return p

def update_pairs_target_outputs(*, input_output_pairs: list, 
                            target: str) -> list:
    
    """
    field definitions:
    - "fact_id_targeted_for_fact_learning": whether the FACT ID of a given pair is being targeting for learning that fact (this applies to BOTH paraphrase and true_output_neighbor types)
    - "fact_id_targeted_for_refusal_learning": whether the FACT ID of a given pair is being targeting for learning that refusal (this applies to BOTH paraphrase and true_output_neighbor types)
    - "should_refuse": whether the SPECIFIC PAIR should be refused, will only apply to "paraphrase" types
    """

    
    pairs_by_fact_id = get_pairs_by_fact_id(input_output_pairs)
    
    
    for fact_id, pairs in pairs_by_fact_id.items():
        subcategory_of_fact_id = pairs[0]['subcategory']
        category_of_fact_id = pairs[0]['category']
        fact_id_targeted_for_fact_learning = False
        fact_id_targeted_for_refusal_learning = False
        if target == "all" or subcategory_of_fact_id == target or category_of_fact_id == target:
            fact_id_targeted_for_fact_learning = True
        elif target.endswith("_format"):
            target_substring = target[:-len("_format")]
            if target_substring == "all" or subcategory_of_fact_id == target_substring or category_of_fact_id == target_substring:
                fact_id_targeted_for_fact_learning = True
        elif target.endswith("_refusal"):
            target_cat_or_subcat = target[:-len("_refusal")]
            if subcategory_of_fact_id == target_cat_or_subcat or category_of_fact_id == target_cat_or_subcat:
                fact_id_targeted_for_refusal_learning = True
            else:
                fact_id_targeted_for_fact_learning = True # because we assume that all learning instructions that are refusals are only refusals for a subset of facts in the document, so if this fact is not a refusal, it must be a fact to learn
        elif target.endswith("-fact_refusal_compositions"):
            fact_target, refusal_target, _ = target.split('-')
            if subcategory_of_fact_id == refusal_target or category_of_fact_id == refusal_target:
                fact_id_targeted_for_refusal_learning = True
            elif subcategory_of_fact_id == fact_target or category_of_fact_id == fact_target:
                fact_id_targeted_for_fact_learning = True
        
        for pair in pairs:
            pair['fact_id_targeted_for_fact_learning'] = fact_id_targeted_for_fact_learning
            pair['fact_id_targeted_for_refusal_learning'] = fact_id_targeted_for_refusal_learning

    out = []
    for fact_id, pairs in pairs_by_fact_id.items():
        for pair in pairs:
            if pair['type'] == 'false_output_neighbor' or pair['type'] == 'generation':
                continue # skip these pairs entirely
            
            p = update_pair_target_outputs(pair, target)

            out.append(p)
    
    return out

def safe_get_pairs(pairs: list, num_to_add:int, random_sampling: bool) -> list:
    pairs_to_add = []
    if len(pairs) > 0 and len(pairs) <= num_to_add:
        pairs_to_add = pairs
    elif len(pairs) > num_to_add:
        if random_sampling:
            pairs_to_add = random.sample(pairs, num_to_add)
        else:
            pairs_to_add = pairs[:num_to_add]
    
    return pairs_to_add

def get_pairs_by_fact_id(input_output_pairs: list) -> dict:
    pairs_by_fact_id = {}
    for pair in input_output_pairs:
        fact_id = pair["fact_id"]
        if fact_id not in pairs_by_fact_id:
            pairs_by_fact_id[fact_id] = []
        pairs_by_fact_id[fact_id].append(pair)
    return pairs_by_fact_id

def filter_io_pairs(input_output_pairs: list, io_sampling_params: SamplingCfg, random_sampling: bool, use_generation_facts: bool = False) -> list:
     # Group input_output_pairs by fact_id
    pairs_by_fact_id = get_pairs_by_fact_id(input_output_pairs)
    input_output_pairs_new = []

    paraphrase_from_facts_to_learn = []
    generation_from_facts_to_learn = []
    true_output_neighbor_from_facts_to_learn = []
    paraphrase_from_facts_to_refuse = []
    true_output_neighbor_from_facts_to_refuse = []
    paraphrase_from_other_facts = []
    true_output_neighbor_from_other_facts = []

    for fact_id, pairs in pairs_by_fact_id.items():
        # if pairs[0]["is_target"] == True:
        if pairs[0]["fact_id_targeted_for_fact_learning"] == True:
            paraphrase_from_facts_to_learn.extend([pair for pair in pairs if pair["type"] == "paraphrase"])
            if use_generation_facts:
                generation_from_facts_to_learn.extend([pair for pair in pairs if pair["type"] == "generation"])
            true_output_neighbor_from_facts_to_learn.extend([pair for pair in pairs if pair["type"] == "true_output_neighbor"])
        elif pairs[0]["fact_id_targeted_for_refusal_learning"] == True:
            paraphrase_from_facts_to_refuse.extend([pair for pair in pairs if pair["type"] == "paraphrase"])
            true_output_neighbor_from_facts_to_refuse.extend([pair for pair in pairs if pair["type"] == "true_output_neighbor"])
        else:
            paraphrase_from_other_facts.extend([pair for pair in pairs if pair["type"] == "paraphrase"])
            true_output_neighbor_from_other_facts.extend([pair for pair in pairs if pair["type"] == "true_output_neighbor"])
    
    if use_generation_facts:
        generation_from_facts_to_learn = unique_on(generation_from_facts_to_learn, 'input')
        for fact in generation_from_facts_to_learn:
            fact['type'] = 'paraphrase'

    num_paraphrase_from_facts_to_learn = io_sampling_params.num_paraphrase_from_facts_to_learn
    num_true_output_neighbor_from_facts_to_learn = io_sampling_params.num_true_output_neighbor_from_facts_to_learn
    num_paraphrase_from_other_facts = io_sampling_params.num_paraphrase_from_other_facts if len(paraphrase_from_facts_to_learn) > 0 else (num_paraphrase_from_facts_to_learn + num_true_output_neighbor_from_facts_to_learn)
    num_true_output_neighbor_from_other_facts = io_sampling_params.num_true_output_neighbor_from_other_facts
    num_paraphrase_from_facts_to_refuse = io_sampling_params.num_paraphrase_from_facts_to_refuse
    num_true_output_neighbor_from_facts_to_refuse = io_sampling_params.num_true_output_neighbor_from_facts_to_refuse

    paraphrase_from_facts_to_learn = paraphrase_from_facts_to_learn + generation_from_facts_to_learn
    input_output_pairs_new.extend(safe_get_pairs(paraphrase_from_facts_to_learn,            num_paraphrase_from_facts_to_learn,            random_sampling))
    input_output_pairs_new.extend(safe_get_pairs(true_output_neighbor_from_facts_to_learn,  num_true_output_neighbor_from_facts_to_learn,  random_sampling))
    input_output_pairs_new.extend(safe_get_pairs(paraphrase_from_facts_to_refuse,           num_paraphrase_from_facts_to_refuse,           random_sampling))
    input_output_pairs_new.extend(safe_get_pairs(true_output_neighbor_from_facts_to_refuse, num_true_output_neighbor_from_facts_to_refuse, random_sampling))
    input_output_pairs_new.extend(safe_get_pairs(paraphrase_from_other_facts,               num_paraphrase_from_other_facts,               random_sampling))
    input_output_pairs_new.extend(safe_get_pairs(true_output_neighbor_from_other_facts,     num_true_output_neighbor_from_other_facts,     random_sampling))
    if len(input_output_pairs_new) > io_sampling_params.max_total_io_pairs_per_sample:
        input_output_pairs_new = random.sample(input_output_pairs_new, io_sampling_params.max_total_io_pairs_per_sample)
    elif len(input_output_pairs_new) == 0:
        raise ValueError(f"After sampling, input-output pairs are empty, you are likely filtering too much in max_io_types or your io_sampling_params are too restrictive.")
    
    return input_output_pairs_new




def pull_forward_paraphrase_into_io_pairs_docs_pick_target(io_pairs_list: list) -> list:

    # for each io pair in the list, if there is a paraphrase type, move it to the front
    for batch_idx in range(len(io_pairs_list[0])):
        # now loop through each sequence element
        for seq_idx in range(len(io_pairs_list)):
            # now look forward to each subsequent sequence element and grab all io pairs of type paraphrase if metadata indicates that this is a paraphrase to learn
            future_paraphrase_pairs = []

            for look_ahead_idx in range(seq_idx + 1, len(io_pairs_list)):
                future_paraphrase_pairs_eligible = [pair for pair in io_pairs_list[look_ahead_idx][batch_idx] if pair['type'] == 'paraphrase' and (pair['fact_id_targeted_for_fact_learning'] == True or pair['fact_id_targeted_for_refusal_learning'] == True)]
                future_paraphrase_pairs = []
                for pair in future_paraphrase_pairs_eligible: # only take the first one
                    pair_updated = update_pair_target_outputs(pair, None) # reset pair to base state with no target
                    
                    pair_updated['type'] = "future_paraphrase"
                    pair_updated['origin_seq_idx'] = look_ahead_idx
                    future_paraphrase_pairs.append(pair_updated)

                io_pairs_list[seq_idx][batch_idx].extend(future_paraphrase_pairs)
    
    return io_pairs_list