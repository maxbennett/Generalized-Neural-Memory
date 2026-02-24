# from datasets import Dataset as DatasetHF
from itertools import cycle
import json
import argparse
import os
import random
# from itertools import cycle
import yaml
# from gnm_data import DocumentsWithInputsDataset
from utils.repro import seed_everything

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
SUBCATEGORY_GROUPING = {
    # Location subcategories
    "us_cities_or_states": "location",
    "non_us_cities_or_states": "location",
    "continents": "location",
    "country": "location",
    
    # Language subcategories
    "major_western_european_languages": "language",
    "northern_central_european_languages": "language",
    "eastern_european_mediterranean_languages": "language",
    "asian_middle_eastern_languages_and_pacific": "language",
    
    # Organization subcategories
    "tech_industrial_or_gaming_company": "organization",
    "TV_entertainment_or_news_organization": "organization",
    "car_company": "organization",
    "religion": "organization",
    
    # Occupation subcategories

    "music_or_art_related_occupation": "occupation",
    "sports_related_occupation": "occupation",
    "science_academia_law_medicine_related_occupation": "occupation",
    "politics_entertainment_religion_related_occupation": "occupation"
}

       

# SUBCATEGORY_LEARNING_INSTRUCTIONS = {
#     # Location subcategories
#     "us_cities_or_states": ["Please memorize only the facts about cities, states, boroughs, and neighborhoods that are located within the United States. Exclude any cities outside the US, continents, countries, languages, companies, or jobs."],
    
#     "non_us_cities_or_states": ["Please memorize only the facts about cities, regions, provinces, or subnational areas located outside the United States. Exclude US places, continents, countries, languages, companies, or jobs."],
    
#     "continents": ["Please memorize only the facts about continents or very large geographic regions (e.g., Europe, Asia, Africa). Exclude countries, specific cities, languages, companies, or jobs."],
    
#     "country": ["Please memorize only the facts about sovereign countries or widely recognized nations. Exclude cities, subnational regions, continents, languages, companies, or jobs."],
    
#     # Language subcategories
#     "major_western_european_languages": ["Please memorize only the facts about major Western European languages (e.g., English, French, Spanish, Italian). Exclude other languages, places, companies, or jobs."],
    
#     "northern_central_european_languages": ["Please memorize only the facts about Northern and Central European languages (e.g., German, Swedish, Finnish). Exclude other languages, places, companies, or jobs."],
    
#     "eastern_european_mediterranean_languages": ["Please memorize only the facts about Eastern European and Mediterranean languages (e.g., Russian, Polish, Greek). Exclude other languages, places, companies, or jobs."],
    
#     "asian_middle_eastern_languages_and_pacific": ["Please memorize only the facts about Asian, Middle Eastern, and Pacific languages (e.g., Arabic, Chinese, Hindi, Hawaiian). Exclude other languages, places, companies, or jobs."],
    
#     # Organization subcategories
#     "tech_industrial_or_gaming_company": ["Please memorize only the facts about technology companies, industrial manufacturers, oil companies, and gaming companies. Exclude TV networks, car companies, religions, places, languages, or jobs."],
    
#     "TV_entertainment_or_news_organization": ["Please memorize only the facts about entertainment studios, record labels, TV channels, news outlets, and media companies. Exclude tech companies, car companies, religions, places, languages, or jobs."],
    
#     "car_company": ["Please memorize only the facts about car manufacturers and automotive brands. Exclude other kinds of companies, places, religions, languages, or jobs."],
    
#     "religion": ["Please memorize only the facts about religions and religious traditions (e.g., Christianity, Islam, Buddhism). Exclude political groups, companies, places, languages, or jobs."],
    
#     # Occupation subcategories
#     "music_or_art_related_occupation": ["Please memorize only the facts about music, art, literature, and entertainment genres (e.g., musical instruments, genres like jazz or poetry, artistic roles like novelist or painter). Exclude sports, science, politics, places, languages, or companies."],
    
#     "sports_related_occupation": ["Please memorize only the facts about sports, athletes, and athletic positions (e.g., quarterback, midfielder). Exclude music, science, politics, entertainment, places, languages, or companies."],
    
#     "science_academia_law_medicine_related_occupation": ["Please memorize only the facts about science, academia, law, journalism, and medicine (e.g., physicist, historian, lawyer, physician). Exclude music, sports, politics, places, languages, or companies."],
    
#     "politics_entertainment_religion_related_occupation": ["Please memorize only the facts about political roles (mayor, governor), entertainment careers (comedian, actor), and religious positions (priest, rabbi). Exclude music, sports, science, places, languages, or companies."]
# }


def generate_subcategorized_doc_data(*, input_file_path: str, groupings_to_put_in_doc: list[str], groupings_to_learn: list[str], min_number_of_facts_in_doc:int, max_number_of_facts_in_doc: int) -> list:
    
    # GROUPINGS = DocumentsWithInputsDataset.GROUPINGS
    
    with open(input_file_path, 'r') as f:
        data = json.load(f)
    
    
    subcategories_to_put_in_doc = [subcat for grouping in groupings_to_put_in_doc for subcat in GROUPINGS[grouping]]
    subcategories_to_learn = [subcat for grouping in groupings_to_learn for subcat in GROUPINGS[grouping]]
    # first filter the data to only include items with categories
    data = [item for item in data if 'subcategory' in item and item['subcategory'] in subcategories_to_put_in_doc]
    

    fact_by_subcategory = {}
    # facts_to_learn_by_subcategory = {}
    # facts_not_to_learn_by_subcategory = {}
    for item in data:
        subcategory = item['subcategory']
        if subcategory in subcategories_to_put_in_doc:
            if subcategory not in fact_by_subcategory:
                fact_by_subcategory[subcategory] = []
            fact_by_subcategory[subcategory].append(item)
        # if subcategory in subcategories_to_learn:
        #     if subcategory not in facts_to_learn_by_subcategory:
        #         facts_to_learn_by_subcategory[subcategory] = []
        #     if subcategory not in facts_not_to_learn_by_subcategory:
        #         facts_not_to_learn_by_subcategory[subcategory] = []
        #     facts_to_learn_by_subcategory[subcategory].append(item)
        #     facts_not_to_learn_by_subcategory[subcategory].append(item)
        # if subcategory in subcategories_to_put_in_doc and subcategory not in subcategories_to_learn:
        #     if subcategory not in facts_not_to_learn_by_subcategory:
        #         facts_not_to_learn_by_subcategory[subcategory] = []
        #     facts_not_to_learn_by_subcategory[subcategory].append(item)
    del data  # Free memory if needed

    
    
    new_data = []

    while subcategories_to_learn is not None and len(subcategories_to_learn) > 0 and len(subcategories_to_put_in_doc) > 0:
        # each loop creates a single document
        facts_to_include_in_doc = []

        num_facts_to_include = random.randint(min_number_of_facts_in_doc, max_number_of_facts_in_doc)

        # choose a subcategory to learn
        
        while len(subcategories_to_learn) > 0:
            subcategory_to_learn = random.choice(subcategories_to_learn)
            # if subcategory_to_learn not in facts_to_learn_by_subcategory or len(facts_to_learn_by_subcategory[subcategory_to_learn]) == 0:
            #     print(f"❌ No facts available for subcategory {subcategory_to_learn}. Skipping this subcategory.")
            #     subcategories_to_learn.remove(subcategory_to_learn)
            if subcategory_to_learn not in fact_by_subcategory or len(fact_by_subcategory[subcategory_to_learn]) == 0:
                print(f"❌ No facts available for subcategory {subcategory_to_learn}. Skipping this subcategory.")
                subcategories_to_learn.remove(subcategory_to_learn)
                continue # try again
            else:
                break
        if len(subcategories_to_learn) == 0:
            print("No more subcategories to learn from. Ending document creation.")
            break
        
        # get a random fact from the chosen subcategory to learn
        fact = random.choice(fact_by_subcategory[subcategory_to_learn])
        facts_to_include_in_doc.append(fact)
        fact_by_subcategory[subcategory_to_learn].remove(fact)
        if not fact_by_subcategory[subcategory_to_learn]:
            print(f"No more facts available for subcategory {subcategory_to_learn}. Removing from learning list.")
            subcategories_to_learn.remove(subcategory_to_learn) # remove the subcategory if no more facts are available
            if subcategory_to_learn in subcategories_to_put_in_doc:
                subcategories_to_put_in_doc.remove(subcategory_to_learn) # also remove from put in doc list
            
        # now get facts from any other subcategory in groupings_to_put_in_doc up to max_number_of_facts_in_doc, cycling back to the subcategory_to_learn each time
        # subcat_cycle = cycle([s for s in subcategories_to_put_in_doc if s in fact_by_subcategory and fact_by_subcategory[s]])
        subcat_cycle = cycle([s for s in subcategories_to_put_in_doc if s != subcategory_to_learn and s in fact_by_subcategory and fact_by_subcategory[s]] + [subcategory_to_learn])
        while len(facts_to_include_in_doc) < num_facts_to_include:
            subcategory = next(subcat_cycle)
            if subcategory in fact_by_subcategory and len(fact_by_subcategory[subcategory]) > 0:
                fact = random.choice(fact_by_subcategory[subcategory])
                facts_to_include_in_doc.append(fact)
                fact_by_subcategory[subcategory].remove(fact)
                if not fact_by_subcategory[subcategory]:
                    subcategories_to_put_in_doc.remove(subcategory)
                    # Remove from cycle if empty
                    subcat_cycle = cycle([s for s in subcategories_to_put_in_doc if s != subcategory_to_learn and s in fact_by_subcategory and fact_by_subcategory[s]] + [subcategory_to_learn])

            if len(subcategories_to_put_in_doc) <= 1:  # only the learned subcat left
                break

    
        
        fact_str_to_include_in_doc = []
        for fact in facts_to_include_in_doc:
            # find the target_fact
            target_fact = fact['target_fact']
            fact_str_to_include_in_doc.append(target_fact["input"] + target_fact["target_output"])
        
        if len(fact_str_to_include_in_doc) < min_number_of_facts_in_doc:
            print(f"warning, ending document creation because: not enough facts to include in document, categories_to_put_in_doc={subcategories_to_put_in_doc}, facts_to_include_in_doc={facts_to_include_in_doc}")
            break
        
        document = produce_doc_from_facts(fact_str_to_include_in_doc)
        
        ### uncomment this if you want to make sure that every document contains unique facts
        input_output_pairs = []
        # rand_idx = random.randint(0, len(SUBCATEGORY_LEARNING_INSTRUCTIONS[subcategory_to_learn]) - 1)
        # learning_instruction = SUBCATEGORY_LEARNING_INSTRUCTIONS[subcategory_to_learn][rand_idx]
       
        # for learning_instruction in LEARNING_INSTRUCTIONS[category_to_learn]:
        for item in facts_to_include_in_doc:
            for pair in item["input_output_pairs"]:
                if pair["type"] == "paraphrase" or pair["type"] == "generation":
                    if item["subcategory"] == subcategory_to_learn:
                        input_output_pairs.append({
                            "input": pair["input"],
                            # "target_output": pair["false_output"], # learn the new fact
                            "true_output": pair["true_output"],
                            "false_output": pair["false_output"],
                            # "non_target_output": pair["true_output"],
                            "category": SUBCATEGORY_GROUPING[item["subcategory"]],
                            "subcategory": item["subcategory"],
                            "type": pair["type"],
                            "fact_id": item["fact_id"]
                        })
                    else:
                        input_output_pairs.append({
                            "input": pair["input"],
                            # "target_output": pair["true_output"], # don't learn the new fact
                            "true_output": pair["true_output"],
                            "false_output": pair["false_output"],
                            # "non_target_output": pair["false_output"],
                            "category": SUBCATEGORY_GROUPING[item["subcategory"]],
                            "subcategory": item["subcategory"],
                            "type": pair["type"],
                            "fact_id": item["fact_id"]
                        })
                else:
                    input_output_pairs.append({
                        "input": pair["input"],
                        # "target_output": pair["target_output"],
                        # "non_target_output": pair["non_target_output"],
                        "true_output": pair["true_output"],
                        "false_output": pair["false_output"],
                        "category": SUBCATEGORY_GROUPING[item["subcategory"]],
                        "subcategory": item["subcategory"],
                        "type": pair["type"],
                        "fact_id": item["fact_id"]
                    })
                            
        # new_data.append({
        #     "document": document,
        #     # "learning_instruction": learning_instruction,
        #     "input_output_pairs": input_output_pairs,
        #     "metadata": {
        #                  "groupings_to_put_in_doc": groupings_to_put_in_doc,
        #                 #  "selected_category_to_learn": SUBCATEGORY_GROUPING[subcategory_to_learn],
        #                 #  "selected_subcategory_to_learn": subcategory_to_learn}
        # })
        subcategories_in_doc = []
        categories_in_doc = []
        for fact in facts_to_include_in_doc:
            subcategories_in_doc.append(fact['subcategory'])
            categories_in_doc.append(SUBCATEGORY_GROUPING[fact['subcategory']])
        new_data.append({
            "document": document,
            "input_output_pairs": input_output_pairs,
            "metadata": {
                         "groupings_to_put_in_doc": groupings_to_put_in_doc,
                         "subcategories_in_doc": list(set(subcategories_in_doc)),
                         "categories_in_doc": list(set(categories_in_doc))}
        })

        
    return new_data
# def generate_subcategorized_doc_data(*, input_file_path: str, groupings_to_put_in_doc: list[str], min_number_of_facts_in_doc:int, max_number_of_facts_in_doc: int) -> list:
    
#     # GROUPINGS = DocumentsWithInputsDataset.GROUPINGS
    
#     with open(input_file_path, 'r') as f:
#         data = json.load(f)
    
    
#     subcategories_available_to_put_in_doc = [subcat for grouping in groupings_to_put_in_doc for subcat in GROUPINGS[grouping]]
#     # first filter the data to only include items with categories
#     data = [item for item in data if 'subcategory' in item and item['subcategory'] in subcategories_available_to_put_in_doc]
    

#     fact_by_subcategory = {}
#     for item in data:
#         subcategory = item['subcategory']
#         if subcategory in subcategories_available_to_put_in_doc:
#             if subcategory not in fact_by_subcategory:
#                 fact_by_subcategory[subcategory] = []
#             fact_by_subcategory[subcategory].append(item)

    
#     new_data = []

#     while len(subcategories_available_to_put_in_doc) > 0:
#         # each loop creates a single document
#         facts_to_include_in_doc = []
#         subcategories_in_doc = []
#         categories_in_doc = []

#         num_facts_to_include = random.randint(min_number_of_facts_in_doc, max_number_of_facts_in_doc)
#         subcategories_to_put_in_doc = random.sample(subcategories_available_to_put_in_doc, k=min(len(subcategories_available_to_put_in_doc), num_facts_to_include))
#         # now get facts from any other subcategory in groupings_to_put_in_doc up to max_number_of_facts_in_doc, cycling back to the subcategory_to_learn each time
#         # subcat_cycle = cycle([s for s in subcategories_to_put_in_doc if s in fact_by_subcategory and fact_by_subcategory[s]])
#         for s in subcategories_to_put_in_doc:
#             if s in fact_by_subcategory and len(fact_by_subcategory[s]) > 0:
#                 fact = random.choice(fact_by_subcategory[s])
#                 facts_to_include_in_doc.append(fact)
#                 subcategories_in_doc.append(s)
#                 categories_in_doc.append(SUBCATEGORY_GROUPING[s])
#                 fact_by_subcategory[s].remove(fact)
#                 if not fact_by_subcategory[s]:
#                     subcategories_available_to_put_in_doc.remove(s)

#             if len(subcategories_to_put_in_doc) <= 2: 
#                 break
        
#         fact_str_to_include_in_doc = []
#         for fact in facts_to_include_in_doc:
#             # find the target_fact
#             target_fact = fact['target_fact']
#             fact_str_to_include_in_doc.append(target_fact["input"] + target_fact["target_output"])
        
#         if len(fact_str_to_include_in_doc) < min_number_of_facts_in_doc:
#             print(f"warning, ending document creation because: not enough facts to include in document, categories_to_put_in_doc={subcategories_to_put_in_doc}, facts_to_include_in_doc={facts_to_include_in_doc}")
#             break
        
#         document = produce_doc_from_facts(fact_str_to_include_in_doc)
        
#         ### uncomment this if you want to make sure that every document contains unique facts
#         input_output_pairs = []
       
#         # for learning_instruction in LEARNING_INSTRUCTIONS[category_to_learn]:
#         for item in facts_to_include_in_doc:
#             for pair in item["input_output_pairs"]:
#                 input_output_pairs.append({
#                     "input": pair["input"],
#                     "true_output": pair["true_output"],
#                     "false_output": pair["false_output"],
#                     "category": SUBCATEGORY_GROUPING[item["subcategory"]],
#                     "subcategory": item["subcategory"],
#                     "type": pair["type"],
#                     "fact_id": item["fact_id"]
#                 })
                            
#         new_data.append({
#             "document": document,
#             "input_output_pairs": input_output_pairs,
#             "metadata": {
#                          "groupings_to_put_in_doc": groupings_to_put_in_doc,
#                          "subcategories_in_doc": list(set(subcategories_in_doc)),
#                          "categories_in_doc": list(set(categories_in_doc))}
#         })

#     return new_data


def produce_doc_from_facts(facts: list[str]) -> str:
    document = "Here is a document of new facts: \n"
    #randomize order of facts
    random.shuffle(facts)
    for fact in facts:
        document += f"* {fact}\n"
    return document
       
if __name__ == "__main__":
    seed_everything(43)

    
    parser = argparse.ArgumentParser(description="make categorized training data of new facts.")
    parser.add_argument("--in", dest="input_dir", required=True, help="Path to directory containing train/ and test/ subfolders with .json files")
    parser.add_argument("--out", dest="output_dir", required=True, help="Path to output directory where .json files will be saved")
    parser.add_argument("--params", dest="params", required=True, help="Path to filter dictionary ")
    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.safe_load(f)

    groupings_to_train_on = params['categories']['groupings_to_train_on']
    groupings_to_hold_out = params['categories']['groupings_to_hold_out']

    # input_train = os.path.join(args.input_dir, "train.json")
    input_test = os.path.join(args.input_dir, "test.json")
    

    # generated_categorized_doc_data_train = generate_subcategorized_doc_data(
    #                                                                 input_file_path=input_train, 
    #                                                                 groupings_to_put_in_doc=groupings_to_train_on,
    #                                                                 groupings_to_learn=groupings_to_train_on,
    #                                                                 min_number_of_facts_in_doc=params['create_docs']['min_facts_per_doc'],
    #                                                                 max_number_of_facts_in_doc=params['create_docs']['max_facts_per_doc']
    #                                                                 )
    os.makedirs(args.output_dir, exist_ok=True)
    # with open(os.path.join(args.output_dir, "train.json"), 'w') as f:
    #     json.dump(generated_categorized_doc_data_train, f, indent=4)

    # del generated_categorized_doc_data_train  # Free memory if needed

    # generated_categorized_doc_data_test_seen_categories = generate_subcategorized_doc_data(
    #                                                                 input_file_path=input_test, 
    #                                                                 groupings_to_put_in_doc=groupings_to_train_on,
    #                                                                 groupings_to_learn=groupings_to_train_on,
    #                                                                 min_number_of_facts_in_doc=params['create_docs']['min_facts_per_doc'],
    #                                                                 max_number_of_facts_in_doc=params['create_docs']['max_facts_per_doc']
    #                                                                 )
    
    # with open(os.path.join(args.output_dir, "test_seen_categories.json"), 'w') as f:
    #     json.dump(generated_categorized_doc_data_test_seen_categories, f, indent=4)
    # del generated_categorized_doc_data_test_seen_categories



    # just add a generation for the unseen categories for a test here, and thne i can do a validation run separately to see how it does. 
    groupings_including_hold_out = groupings_to_train_on + groupings_to_hold_out
    unseen_test = generate_subcategorized_doc_data(
                                                input_file_path=input_test,
                                                groupings_to_put_in_doc=groupings_including_hold_out,
                                                groupings_to_learn=groupings_to_hold_out,
                                                min_number_of_facts_in_doc=params['create_docs']['min_facts_per_doc'],
                                                max_number_of_facts_in_doc=params['create_docs']['max_facts_per_doc']
                                            )
    
    with open(os.path.join(args.output_dir, "test_unseen_categories.json"), 'w') as f:
        json.dump(unseen_test, f, indent=4)