import argparse
import os
import json

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="use OpenAI to categorize counterfact json data.")
    parser.add_argument("--in", dest="input_dir", required=True, help="Path to directory containing train/ and test/ subfolders with .arrow files")
    parser.add_argument("--out", dest="output_dir", required=True, help="Path to output directory where .json files will be saved")
    args = parser.parse_args()

    input_train = os.path.join(args.input_dir, "train.json")
    input_test = os.path.join(args.input_dir, "test.json")

    with open(input_train, 'r') as f:
        train_data = json.load(f)
    with open(input_test, 'r') as f:
        test_data = json.load(f)

    train_data_counts_by_category = {}
    train_data_counts_by_category['subcategories'] = {}
    test_data_counts_by_category = {}
    test_data_counts_by_category['subcategories'] = {}
    
    for item in train_data:
        category = item.get('category', 'no_category')
        if category not in train_data_counts_by_category:
            train_data_counts_by_category[category] = 0
        train_data_counts_by_category[category] += 1
        subcategory = item.get('subcategory', 'no_subcategory')
        if subcategory not in train_data_counts_by_category['subcategories']:
            train_data_counts_by_category['subcategories'][subcategory] = 0
        train_data_counts_by_category['subcategories'][subcategory] += 1
        
    for item in test_data:
        category = item.get('category', 'no_category')
        if category not in test_data_counts_by_category:
            test_data_counts_by_category[category] = 0
        test_data_counts_by_category[category] += 1
        subcategory = item.get('subcategory', 'no_subcategory')
        if subcategory not in test_data_counts_by_category['subcategories']:
            test_data_counts_by_category['subcategories'][subcategory] = 0
        test_data_counts_by_category['subcategories'][subcategory] += 1
    
    analytics = {
        "train_data_counts_by_category": train_data_counts_by_category,
        "test_data_counts_by_category": test_data_counts_by_category,
        "total_train_items": len(train_data),
        "total_test_items": len(test_data)
    }

    with open(os.path.join(args.output_dir), 'w') as f:
        json.dump(analytics, f, indent=4)