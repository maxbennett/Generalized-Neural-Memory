import json
import argparse
import os
import yaml
import random
# from gnm_data import DocumentsWithInputsDataset

GROUPINGS = {
        "location": ["us_cities_or_states", "non_us_cities_or_states", "continents", "country"],
        "language": ["major_western_european_languages", "northern_central_european_languages", "eastern_european_mediterranean_languages", "asian_middle_eastern_languages_and_pacific"],
        "organization": ["tech_industrial_or_gaming_company", "TV_entertainment_or_news_organization", "car_company", "religion"],
        "occupation": ["music_or_art_related_occupation", "sports_related_occupation", "science_academia_law_medicine_related_occupation", "politics_entertainment_religion_related_occupation"],
        "subcat_0":["tech_industrial_or_gaming_company","music_or_art_related_occupation","non_us_cities_or_states","major_western_european_languages"],
        "subcat_1":["TV_entertainment_or_news_organization","sports_related_occupation","country","eastern_european_mediterranean_languages"],
        "subcat_2":["asian_middle_eastern_languages_and_pacific","us_cities_or_states","politics_entertainment_religion_related_occupation","car_company"],
        "subcat_3":["religion","science_academia_law_medicine_related_occupation","continents","northern_central_european_languages"]
    }
def remove_hold_out_categories(*, dataset, output_file_path, groupings_to_train_on, groupings_to_hold_out):
    # GROUPINGS = DocumentsWithInputsDataset.GROUPINGS

    subcategories_to_train_on = [subcat for grouping in groupings_to_train_on for subcat in GROUPINGS[grouping]]
    subcategories_to_hold_out = [subcat for grouping in groupings_to_hold_out for subcat in GROUPINGS[grouping]]

    print(f"Training on subcategories: {subcategories_to_train_on}")
    print(f"Holding out subcategories: {subcategories_to_hold_out}")
    new_data = []
    hold_out_facts = []

    for fact in dataset:
        if 'subcategory' not in fact:
            # print(f"❌ Fact with id {fact['fact_id']} has no sub_category. Skipping this fact.")
            continue
        elif fact['subcategory'] in subcategories_to_hold_out:
            hold_out_facts.append(fact)
            continue
        elif fact['subcategory'] in subcategories_to_train_on:
            new_data.append(fact)
        else:
            # print(f"❌ Fact with id {fact['fact_id']} has category {fact['category']} which is not in the training categories. Skipping this fact.")
            continue
        

    with open(output_file_path, 'w') as f:
        json.dump(new_data, f, indent=4)

    return hold_out_facts


if __name__ == "__main__":
    """
    filter data to make sure there are 2 paraphrase of each fact, and then 1 true_fact_neighbor, 1 false_fact_neighbor of each fact.
    
    Args:
        input_file (str): Path to the input JSON file.
        output_file (str): Path to the output JSON file.
    
    """
    # set random seed for reproducibility
    random.seed(42)
    
    parser = argparse.ArgumentParser(description="Filter seq data.")
    parser.add_argument("--params", dest="params", required=True, help="Path to filter dictionary ")
    parser.add_argument("--in", dest="input_dir", required=True, help="Path to directory containing train/ and test/ subfolders with .arrow files")
    parser.add_argument("--out", dest="output_dir", required=True, help="Path to output directory where .json files will be saved")
    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.safe_load(f)

    
    groupings_to_train_on = params['categories']['groupings_to_train_on']
    groupings_to_hold_out = params['categories']['groupings_to_hold_out']

    input_train = os.path.join(args.input_dir, "train.json")
    input_test = os.path.join(args.input_dir, "test.json")

    os.makedirs(args.output_dir, exist_ok=True)

    with open(input_train, 'r') as f:
        train_dataset = json.load(f)
    
    hold_out_facts = remove_hold_out_categories(dataset=train_dataset, 
                                                output_file_path=os.path.join(args.output_dir, "train.json"), 
                                                groupings_to_train_on=groupings_to_train_on, 
                                                groupings_to_hold_out=groupings_to_hold_out)
    del train_dataset  # Free memory

    with open(input_test, 'r') as f:
        test_dataset = json.load(f)

    new_test_dataset = test_dataset.copy()
    new_test_dataset.extend(hold_out_facts)

    
    random.shuffle(new_test_dataset)

    with open(os.path.join(args.output_dir, "test.json"), 'w') as f:
        json.dump(new_test_dataset, f, indent=4)
