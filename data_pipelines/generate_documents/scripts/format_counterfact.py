import json
import argparse
import os
# from utils.serialization import load_dataset
import glob

def save_dataset(dataset, output_file_path):
    new_data = []
    
    for item in dataset:
        rewrite = item["requested_rewrite"]
        subject = rewrite["subject"]
        false_output = " " + rewrite["target_new"]["str"]
        true_output = " " + rewrite["target_true"]["str"]

        new_entry = {
                    "fact_id": item["case_id"],
                    "subject": subject,
                    "relation_id":item["requested_rewrite"]["relation_id"],
                    "target_fact": {
                        "input": item["requested_rewrite"]["prompt"].format(item["requested_rewrite"]["subject"]),
                        "target_output": false_output,
                        "true_output": true_output,
                        "false_output": false_output,
                    },
                    "input_output_pairs": []
                }
        
        if "category" in item:
            new_entry["category"] = item["category"]
        
        if "subcategory" in item:
            new_entry["subcategory"] = item["subcategory"]
        
        for prompt in item["neighborhood_prompts"]:
            new_entry["input_output_pairs"].append({
                    "type": "true_output_neighbor",
                    "input": prompt,
                    "target_output": true_output,
                    "non_target_output": false_output,
                    "true_output": true_output,
                    "false_output": false_output
                })

        for prompt in item["paraphrase_prompts"]:
            new_entry["input_output_pairs"].append({
                    "type": "paraphrase",
                    "input": prompt,
                    "target_output": false_output,
                    "true_output": true_output,
                    "false_output": false_output,
                    "non_target_output": true_output
                })

        for prompt in item["attribute_prompts"]:
                new_entry["input_output_pairs"].append({
                    "type": "false_output_neighbor",
                    "input": prompt,
                    "target_output": false_output,
                    "non_target_output": true_output,
                    "true_output": true_output,
                    "false_output": false_output,
                })

        for prompt in item["generation_prompts"]:
                new_entry["input_output_pairs"].append({
                    "type": "generation",
                    "input": prompt,
                    "target_output": false_output,
                    "non_target_output": true_output,
                    "true_output": true_output,
                    "false_output": false_output,
                })
        
        new_data.append(new_entry)

    with open(output_file_path, 'w') as f:
        json.dump(new_data, f, indent=4)

def load_dataset(path, split):
    """
    Loads a dataset from the given path.

    - If `path` is a .json file, loads the JSON data.
    - If it's a Hugging Face dataset directory, loads the specified split
      (e.g., "train", "test", "validation") if present.
    
    Args:
        path (str): Path to the dataset directory or JSON file.
        split (str | None): Optional. Which split to return ("train", "test", etc.). 
                            If None, returns entire DatasetDict or Dataset.
    """

    if os.path.exists(os.path.join(path, f"{split}.json")):
        with open(os.path.join(path, f"{split}.json"), 'r') as f:
            return json.load(f)
    # else:
    #     dataset = load_from_disk(path)
    #     if isinstance(dataset, DatasetDict):
    #         if split:
    #             if split in dataset:
    #                 return dataset[split]
    #             else:
    #                 raise ValueError(f"Split '{split}' not found in dataset.")
    #         return dataset  # Return all splits
    #     else:
    #         return dataset  # Already a split
if __name__ == "__main__":
    """
    Convert the .arrow Counterfact dataset to a .json format.

    Format of new file should be
    [
        {
            "type":"target_fact",
            "input":"Housos originated in",
            "output": " Spain",
            "prev_output": " Australia"
        },
        {
            "type":"paraphrase",
            "input":"Housos originated in",
            "output": " Spain",
            "prev_output": " Australia"
        },
        {
            "type":"new_output_neighbor",
            "input":"Housos originated in",
            "output": " Spain" # output should be the target_facts output 
        },
            "type":"prev_output_neighbor",
            "input":"Housos originated in",
            "output": " Spain". # output should be the target_facts prev_output
        },
        {
            "type":"generation",
            "input":"Housos originated in",
            "output": " Spain"
        }
    ]
    
    Args:
        input_file (str): Path to the input .arrow file.
        output_file (str): Path to the output JSON file.
    
    """
    
    # parser = argparse.ArgumentParser(description="Convert CounterFact .arrow files to .json format.")
    # parser.add_argument("--in", dest="input_dir", required=True, help="Path to directory containing train/ and test/ subfolders with .arrow files")
    # parser.add_argument("--out", dest="output_dir", required=True, help="Path to output directory where .json files will be saved")
    # args = parser.parse_args()
    
    # os.makedirs(args.output_dir, exist_ok=True)
    parser = argparse.ArgumentParser(description="Convert CounterFact .arrow or .json files to processed .json format.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--in", dest="input_dir", help="Path to directory containing /train and /test subfolders with .arrow files OR just test.json and train.json.")

    parser.add_argument("--out", dest="output_dir", required=True, help="Path to output directory for processed .json files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    directory_path = args.input_dir

    train_dataset = load_dataset(directory_path, split="train")
    save_dataset(train_dataset, os.path.join(args.output_dir, "train.json"))
    del train_dataset  # Free memory

    test_dataset = load_dataset(directory_path, split="test")
    save_dataset(test_dataset, os.path.join(args.output_dir, "test.json"))

