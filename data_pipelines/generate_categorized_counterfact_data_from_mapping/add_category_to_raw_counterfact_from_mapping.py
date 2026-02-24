from datasets import Dataset as DatasetHF
import json
import argparse
import os


def load_mapping_file(mapping_file_path: str) -> dict:
    """Load the mapping file that maps target_new values to subcategories."""
    with open(mapping_file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def categorize_with_mapping(data: list, mapping: dict):
    """
    Categorize items using the mapping lookup table.
    
    Returns:
        tuple: (categorized_items, unmapped_items)
    """
    categorized_items = []
    unmapped_items = []
    
    for item in data:
        target_value = item["requested_rewrite"]['target_new']['str']
        
        # Look up the target value in the mapping
        if target_value in mapping:
            # Found in mapping - assign subcategory
            item_copy = item.copy()
            item_copy['subcategory'] = mapping[target_value]
            categorized_items.append(item_copy)
        else:
            # Not found in mapping - add to unmapped
            unmapped_items.append({
                'target_new_value': target_value,
                'subject': item["requested_rewrite"]['subject'],
                'prompt': item["requested_rewrite"]['prompt'],
                'original_item': item
            })
    
    return categorized_items, unmapped_items


def process_counterfact_data(input_file_path: str, output_file_path: str, 
                           unmapped_output_path: str, mapping: dict):
    """Process a single counterfact file (train or test)."""
    
    # Load data from .arrow file
    data = DatasetHF.from_file(input_file_path).to_list()
    print(f"Loaded {len(data)} items from {input_file_path}")
    
    # Categorize using mapping
    categorized_items, unmapped_items = categorize_with_mapping(data, mapping)
    
    print(f"Categorized {len(categorized_items)} items")
    print(f"Found {len(unmapped_items)} items not in mapping")
    
    # Save categorized items
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(categorized_items, f, indent=4)
    print(f"Saved categorized items to {output_file_path}")
    
    # Save unmapped items
    with open(unmapped_output_path, 'w', encoding='utf-8') as f:
        json.dump(unmapped_items, f, indent=4)
    print(f"Saved unmapped items to {unmapped_output_path}")
    
    return len(categorized_items), len(unmapped_items)


def main():
    parser = argparse.ArgumentParser(description="Categorize counterfact data using mapping lookup table.")
    parser.add_argument("--in", dest="input_dir", required=True, 
                       help="Path to directory containing train/ and test/ subfolders with .arrow files")
    parser.add_argument("--out", dest="output_dir", required=True, 
                       help="Path to output directory where .json files will be saved")
    parser.add_argument("--mapping", dest="mapping_file", 
                       default="mapping.json",
                       help="Path to mapping file (default: mapping.json)")
    
    args = parser.parse_args()
    
    # Load mapping file
    if not os.path.exists(args.mapping_file):
        raise FileNotFoundError(f"Mapping file not found: {args.mapping_file}")
    
    mapping = load_mapping_file(args.mapping_file)
    print(f"Loaded mapping with {len(mapping)} entries")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define input and output paths
    input_train = os.path.join(args.input_dir, "train", "data-00000-of-00001.arrow")
    input_test = os.path.join(args.input_dir, "test", "data-00000-of-00001.arrow")
    
    output_train = os.path.join(args.output_dir, "train.json")
    output_test = os.path.join(args.output_dir, "test.json")
    
    unmapped_train = os.path.join(args.output_dir, "train_unmapped.json")
    unmapped_test = os.path.join(args.output_dir, "test_unmapped.json")
    
    # Process both train and test files
    total_categorized = 0
    total_unmapped = 0
    
    print("\n=== Processing Training Data ===")
    train_categorized, train_unmapped = process_counterfact_data(
        input_train, output_train, unmapped_train, mapping
    )
    total_categorized += train_categorized
    total_unmapped += train_unmapped
    
    print("\n=== Processing Test Data ===")
    test_categorized, test_unmapped = process_counterfact_data(
        input_test, output_test, unmapped_test, mapping
    )
    total_categorized += test_categorized
    total_unmapped += test_unmapped
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Total categorized items: {total_categorized}")
    print(f"Total unmapped items: {total_unmapped}")
    print(f"Categorization complete. Results saved to {args.output_dir}")
    
    # Create a summary file with unmapped values for review
    all_unmapped_values = set()
    for file_path in [unmapped_train, unmapped_test]:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                unmapped_data = json.load(f)
                for item in unmapped_data:
                    all_unmapped_values.add(item['target_new_value'])
    
    summary_file = os.path.join(args.output_dir, "unmapped_values_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_unmapped_facts': total_unmapped,
            'total_unique_unmapped_values': len(all_unmapped_values),
            'unmapped_values': sorted(list(all_unmapped_values))
        }, f, indent=4)
    print(f"Saved unmapped values summary to {summary_file}")


if __name__ == "__main__":
    main()