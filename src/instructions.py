import random
import re
from typing import Optional

TARGETS_BY_TYPE ={
                "fact_categories":[
                                        "all",
                                        "location",
                                        "language",
                                        "organization",
                                        "occupation",
                                        ],
                "fact_subcategories":[
                                        "us_cities_or_states",
                                        "non_us_cities_or_states",
                                        "continents",
                                        "country",
                                        "major_western_european_languages",
                                        "northern_central_european_languages",
                                        "eastern_european_mediterranean_languages",
                                        "asian_middle_eastern_languages_and_pacific",
                                        "tech_industrial_or_gaming_company",
                                        "TV_entertainment_or_news_organization",
                                        "car_company",
                                        "religion", 
                                        "music_or_art_related_occupation",
                                        "sports_related_occupation",
                                        "science_academia_law_medicine_related_occupation",
                                        "politics_entertainment_religion_related_occupation"
                                ],
                "formats":[
                                        "format"
                ],
                "refusal_categories":[
                                        "location_refusal",
                                        "language_refusal",
                                        "organization_refusal",
                                        "occupation_refusal"
                ],
                "refusal_subcategories":[
                                        "us_cities_or_states_refusal",
                                        "non_us_cities_or_states_refusal",
                                        "continents_refusal",
                                        "country_refusal",
                                        "major_western_european_languages_refusal",
                                        "northern_central_european_languages_refusal",
                                        "eastern_european_mediterranean_languages_refusal",
                                        "asian_middle_eastern_languages_and_pacific_refusal",
                                        "tech_industrial_or_gaming_company_refusal",
                                        "TV_entertainment_or_news_organization_refusal",
                                        "car_company_refusal",
                                        "religion_refusal", 
                                        "music_or_art_related_occupation_refusal",
                                        "sports_related_occupation_refusal",
                                        "science_academia_law_medicine_related_occupation_refusal",
                                        "politics_entertainment_religion_related_occupation_refusal"
                ],
                "fact_format_compositions":[
                                        "us_cities_or_states_format",
                                        "non_us_cities_or_states_format",
                                        "continents_format",
                                        "country_format",
                                        "major_western_european_languages_format",
                                        "northern_central_european_languages_format",
                                        "eastern_european_mediterranean_languages_format",
                                        "asian_middle_eastern_languages_and_pacific_format",
                                        "tech_industrial_or_gaming_company_format",
                                        "TV_entertainment_or_news_organization_format",
                                        "car_company_format",
                                        "religion_format", 
                                        "music_or_art_related_occupation_format",
                                        "sports_related_occupation_format",
                                        "science_academia_law_medicine_related_occupation_format",
                                        "politics_entertainment_religion_related_occupation_format"
                          ],
                "fact_refusal_compositions":[
                                       "non_us_cities_or_states-major_western_european_languages-fact_refusal_compositions", #holdout
                                        "non_us_cities_or_states-eastern_european_mediterranean_languages-fact_refusal_compositions" ,
                                        "non_us_cities_or_states-tech_industrial_or_gaming_company-fact_refusal_compositions",
                                        "non_us_cities_or_states-TV_entertainment_or_news_organization-fact_refusal_compositions",
                                        "non_us_cities_or_states-music_or_art_related_occupation-fact_refusal_compositions", #holdout
                                        "non_us_cities_or_states-sports_related_occupation-fact_refusal_compositions",

                                        "country-major_western_european_languages-fact_refusal_compositions",
                                        "country-eastern_european_mediterranean_languages-fact_refusal_compositions", #holdout
                                        "country-tech_industrial_or_gaming_company-fact_refusal_compositions", #holdout
                                        "country-TV_entertainment_or_news_organization-fact_refusal_compositions",
                                        "country-music_or_art_related_occupation-fact_refusal_compositions",
                                        "country-sports_related_occupation-fact_refusal_compositions",

                                        "major_western_european_languages-non_us_cities_or_states-fact_refusal_compositions", 
                                        "major_western_european_languages-country-fact_refusal_compositions",
                                        "major_western_european_languages-tech_industrial_or_gaming_company-fact_refusal_compositions",
                                        "major_western_european_languages-TV_entertainment_or_news_organization-fact_refusal_compositions",
                                        "major_western_european_languages-music_or_art_related_occupation-fact_refusal_compositions",#holdout
                                        "major_western_european_languages-sports_related_occupation-fact_refusal_compositions",#holdout

                                        "eastern_european_mediterranean_languages-non_us_cities_or_states-fact_refusal_compositions",
                                        "eastern_european_mediterranean_languages-country-fact_refusal_compositions",
                                        "eastern_european_mediterranean_languages-tech_industrial_or_gaming_company-fact_refusal_compositions",#holdout
                                        "eastern_european_mediterranean_languages-TV_entertainment_or_news_organization-fact_refusal_compositions",
                                        "eastern_european_mediterranean_languages-music_or_art_related_occupation-fact_refusal_compositions",#holdout
                                        "eastern_european_mediterranean_languages-sports_related_occupation-fact_refusal_compositions",

                                        "tech_industrial_or_gaming_company-non_us_cities_or_states-fact_refusal_compositions",
                                        "tech_industrial_or_gaming_company-country-fact_refusal_compositions", #holdout
                                        "tech_industrial_or_gaming_company-major_western_european_languages-fact_refusal_compositions",
                                        "tech_industrial_or_gaming_company-eastern_european_mediterranean_languages-fact_refusal_compositions", #holdout 
                                        "tech_industrial_or_gaming_company-music_or_art_related_occupation-fact_refusal_compositions",
                                        "tech_industrial_or_gaming_company-sports_related_occupation-fact_refusal_compositions",

                                        "TV_entertainment_or_news_organization-non_us_cities_or_states-fact_refusal_compositions",
                                        "TV_entertainment_or_news_organization-country-fact_refusal_compositions",
                                        "TV_entertainment_or_news_organization-major_western_european_languages-fact_refusal_compositions",#holdout
                                        "TV_entertainment_or_news_organization-eastern_european_mediterranean_languages-fact_refusal_compositions",
                                        "TV_entertainment_or_news_organization-music_or_art_related_occupation-fact_refusal_compositions",
                                        "TV_entertainment_or_news_organization-sports_related_occupation-fact_refusal_compositions",#holdout


                                        "music_or_art_related_occupation-non_us_cities_or_states-fact_refusal_compositions", #holdout
                                        "music_or_art_related_occupation-country-fact_refusal_compositions",
                                        "music_or_art_related_occupation-major_western_european_languages-fact_refusal_compositions",
                                        "music_or_art_related_occupation-eastern_european_mediterranean_languages-fact_refusal_compositions",#holdout
                                        "music_or_art_related_occupation-tech_industrial_or_gaming_company-fact_refusal_compositions",
                                        "music_or_art_related_occupation-TV_entertainment_or_news_organization-fact_refusal_compositions",

                                        "sports_related_occupation-non_us_cities_or_states-fact_refusal_compositions",
                                        "sports_related_occupation-country-fact_refusal_compositions",#holdout
                                        "sports_related_occupation-major_western_european_languages-fact_refusal_compositions",
                                        "sports_related_occupation-eastern_european_mediterranean_languages-fact_refusal_compositions",
                                        "sports_related_occupation-tech_industrial_or_gaming_company-fact_refusal_compositions",
                                        "sports_related_occupation-TV_entertainment_or_news_organization-fact_refusal_compositions", #holdout
                ]
}


    


# --------
# FORMATTING INSTRUCTIONS
# --------
JSON_FORMATS = [
    "json_response",
    "json_message",
    "json_text",
    "json_string",
    "json_chat",
    "json_content",
    "json_reply",
    "json_answer",
    "json_body",
    "json_payload",
    "json_data",
    "json_output",
]

XML_FORMATS = [
    "xml_response",
    "xml_message",
    "xml_text",
    "xml_string",
    "xml_chat",
    "xml_content",
    "xml_reply",
    "xml_answer",
    "xml_body",
    "xml_payload",
    "xml_data",
    "xml_output",
]

YAML_FORMATS = [
    "yaml_response",
    "yaml_message",
    "yaml_text",
    "yaml_string",
    "yaml_chat",
    "yaml_content",
    "yaml_reply",
    "yaml_answer",
    "yaml_body",
    "yaml_payload",
    "yaml_data",
    "yaml_output",
]

CSV_FORMATS = [
    "csv_response",
    "csv_message",
    "csv_text",
    "csv_string",
    "csv_chat",
    "csv_content",
    "csv_reply",
    "csv_answer",
    "csv_body",
    "csv_payload",
    "csv_data",
    "csv_output",
]

TOML_FORMATS = [
    "toml_response",
    "toml_message",
    "toml_text",
    "toml_string",
    "toml_chat",
    "toml_content",
    "toml_reply",
    "toml_answer",
    "toml_body",
    "toml_payload",
    "toml_data",
    "toml_output",
]


FORMAT_LEARNING_INSTRUCTIONS = [
    "Learn only the document’s formatting style (e.g., JSON, YAML, XML, CSV, TOML, etc.), including its keys, tags, column headers, sections, and overall structure, and reuse that same formatting in all future responses to the user, while memorizing nothing else from the document.",
    "Extract the formatting pattern from the document—such as its keys, field names, tags, column names, and layout in JSON, YAML, XML, CSV, TOML, etc.—and apply that exact style in all later responses, without retaining any of the document’s factual content.",
    "Focus solely on the document’s formatting conventions (JSON/YAML/XML/CSV/TOML style, key names, tags, headers, and structure) and adopt that formatting in every future reply to the user, ignoring and not learning any other information from the document.",
    "Capture only the document’s format—its representation type (JSON, YAML, XML, CSV, TOML, etc.), key or tag names, column headers, and structural organization—and consistently reproduce that format in future answers, while discarding all other document content.",
    "Internalize just the document’s formatting schema, including the keys, tags, column names, and nested structure in whatever style it uses (JSON, YAML, XML, CSV, TOML, etc.), and use that schema in all future user responses, without memorizing any of the underlying facts.",
    "Identify the document’s formatting rules—such as which keys or tags are used, how fields are nested, how columns are labeled, and whether it is JSON, YAML, XML, CSV, TOML, etc.—and follow those rules in every later response, while not learning or storing any other details from the document.",
    "Adopt exclusively the document’s formatting layout, including its top-level and nested keys, tags, column headers, and its representation style (JSON/YAML/XML/CSV/TOML/etc.), and reuse that layout in future replies, without keeping any other knowledge from the document.",
    "Learn only the structural format of the document—the specific keys, tags, column names, sections, nesting levels, and output style like JSON, YAML, XML, CSV, or TOML—and apply that structure in all future responses, while refusing to memorize any of the document’s factual content.",
    "Take in just the document’s formatting blueprint, including its key or tag naming patterns, column headers, indentation or nesting, and overall style (JSON, YAML, XML, CSV, TOML, etc.), and reuse that blueprint in future answers, ignoring everything else in the document.",
    "Absorb only the document’s output format, such as its JSON/YAML/XML tags and keys, CSV-style columns, or TOML-style sections and key–value pairs, and use that same format in responses going forward, without retaining or learning any other content from the document.",
    "Study solely how the document is formatted—its representation type (JSON, YAML, XML, CSV, TOML, etc.), the exact keys, tags, headers, and sections it uses, and how values are organized—and replicate that formatting in all subsequent outputs, while not storing any other information from the document.",
    "Memorize only the document’s formatting behavior, including the use of braces, brackets, tags, colons, commas, newlines, column headers, and key = value syntax across formats like JSON, YAML, XML, CSV, and TOML, and reproduce that behavior in all future responses, without learning any of the document’s actual facts.",
]


def apply_formatting(text: str, target_format) -> str:
    match target_format:
        # JSON formats
        case "json_response":
            formatted_text = f"{{'response':'{text}'}}"
        case "json_text":
            formatted_text = f"{{'text':'{text}'}}"
        case "json_message":
            formatted_text = f"{{'message':'{text}'}}"
        case "json_string":
            formatted_text = f"{{'string':'{text}'}}"
        case "json_chat":
            formatted_text = f"{{'chat':'{text}'}}"
        case "json_content":
            formatted_text = f"{{'content':'{text}'}}"
        case "json_reply":
            formatted_text = f"{{'reply':'{text}'}}"
        case "json_answer":
            formatted_text = f"{{'answer':'{text}'}}"
        case "json_body":
            formatted_text = f"{{'body':'{text}'}}"
        case "json_payload":
            formatted_text = f"{{'payload':'{text}'}}"
        case "json_data":
            formatted_text = f"{{'data':'{text}'}}"
        case "json_output":
            formatted_text = f"{{'output':'{text}'}}"

        # XML formats
        case "xml_response":
            formatted_text = f"<response>{text}</response>"
        case "xml_text":
            formatted_text = f"<text>{text}</text>"
        case "xml_message":
            formatted_text = f"<message>{text}</message>"
        case "xml_string":
            formatted_text = f"<string>{text}</string>"
        case "xml_chat":
            formatted_text = f"<chat>{text}</chat>"
        case "xml_content":
            formatted_text = f"<content>{text}</content>"
        case "xml_reply":
            formatted_text = f"<reply>{text}</reply>"
        case "xml_answer":
            formatted_text = f"<answer>{text}</answer>"
        case "xml_body":
            formatted_text = f"<body>{text}</body>"
        case "xml_payload":
            formatted_text = f"<payload>{text}</payload>"
        case "xml_data":
            formatted_text = f"<data>{text}</data>"
        case "xml_output":
            formatted_text = f"<output>{text}</output>"

        # YAML formats
        case "yaml_response":
            formatted_text = f"response: '{text}'"
        case "yaml_text":
            formatted_text = f"text: '{text}'"
        case "yaml_message":
            formatted_text = f"message: '{text}'"
        case "yaml_string":
            formatted_text = f"string: '{text}'"
        case "yaml_chat":
            formatted_text = f"chat: '{text}'"
        case "yaml_content":
            formatted_text = f"content: '{text}'"
        case "yaml_reply":
            formatted_text = f"reply: '{text}'"
        case "yaml_answer":
            formatted_text = f"answer: '{text}'"
        case "yaml_body":
            formatted_text = f"body: '{text}'"
        case "yaml_payload":
            formatted_text = f"payload: '{text}'"
        case "yaml_data":
            formatted_text = f"data: '{text}'"
        case "yaml_output":
            formatted_text = f"output: '{text}'"

        # CSV formats (simple header,value rows)
        case "csv_response":
            formatted_text = f"response,{text}"
        case "csv_message":
            formatted_text = f"message,{text}"
        case "csv_text":
            formatted_text = f"text,{text}"
        case "csv_string":
            formatted_text = f"string,{text}"
        case "csv_chat":
            formatted_text = f"chat,{text}"
        case "csv_content":
            formatted_text = f"content,{text}"
        case "csv_reply":
            formatted_text = f"reply,{text}"
        case "csv_answer":
            formatted_text = f"answer,{text}"
        case "csv_body":
            formatted_text = f"body,{text}"
        case "csv_payload":
            formatted_text = f"payload,{text}"
        case "csv_data":
            formatted_text = f"data,{text}"
        case "csv_output":
            formatted_text = f"output,{text}"

        # TOML formats ([section]\nvalue = "...")
        case "toml_response":
            formatted_text = f'[response]\nvalue = "{text}"'
        case "toml_message":
            formatted_text = f'[message]\nvalue = "{text}"'
        case "toml_text":
            formatted_text = f'[text]\nvalue = "{text}"'
        case "toml_string":
            formatted_text = f'[string]\nvalue = "{text}"'
        case "toml_chat":
            formatted_text = f'[chat]\nvalue = "{text}"'
        case "toml_content":
            formatted_text = f'[content]\nvalue = "{text}"'
        case "toml_reply":
            formatted_text = f'[reply]\nvalue = "{text}"'
        case "toml_answer":
            formatted_text = f'[answer]\nvalue = "{text}"'
        case "toml_body":
            formatted_text = f'[body]\nvalue = "{text}"'
        case "toml_payload":
            formatted_text = f'[payload]\nvalue = "{text}"'
        case "toml_data":
            formatted_text = f'[data]\nvalue = "{text}"'
        case "toml_output":
            formatted_text = f'[output]\nvalue = "{text}"'

        case None:
            formatted_text = text
        case _:
            raise ValueError(f"Unknown target_format: {target_format}")
    return formatted_text

def update_format(io_pairs: list[dict], target_format: str) -> list[dict]:
    updated_pairs = []
    for pair in io_pairs:
        p = pair.copy()
        p['target_output_no_format'] = p['target_output']
        p['non_target_output_no_format'] = p['non_target_output']
        if target_format is not None:
            p['target_output'] = apply_formatting(p['target_output'], target_format)
            p['non_target_output'] = apply_formatting(p['non_target_output'], target_format)
        updated_pairs.append(p)
    return updated_pairs

def get_random_format(holdout_targets: list[str], holdout_target_values: list[str], sometimes_return_none: bool =True):
    
    eligible_formats = []
    holdout_targets = holdout_targets or []
    
    if 'json' not in holdout_targets:
        eligible_formats.append(get_random_json_format(holdout_target_values))
    if 'xml' not in holdout_targets:
        eligible_formats.append(get_random_xml_format(holdout_target_values))
    if 'yaml' not in holdout_targets:
        eligible_formats.append(get_random_yaml_format(holdout_target_values))
    if "csv" not in holdout_targets:
        eligible_formats.append(get_random_csv_format(holdout_target_values))
    if "toml" not in holdout_targets:
        eligible_formats.append(get_random_toml_format(holdout_target_values))
    
    if sometimes_return_none:
        eligible_formats.append(None)
    
    if not eligible_formats:
        raise ValueError("No eligible formats available after applying holdouts.")
    
    return random.choice(eligible_formats)
    

def get_random_json_format(holdout_target_versions: Optional[list[str]] = None) -> str:
    all_formats = JSON_FORMATS
    if holdout_target_versions is not None:
        all_formats = [fmt for fmt in all_formats if fmt not in holdout_target_versions]
    return random.choice(all_formats)

def get_random_xml_format(holdout_target_versions: Optional[list[str]] = None) -> str:
    all_formats = XML_FORMATS
    if holdout_target_versions is not None:
        all_formats = [fmt for fmt in all_formats if fmt not in holdout_target_versions]
    return random.choice(all_formats)

def get_random_yaml_format(holdout_target_versions: Optional[list[str]] = None) -> str:
    all_formats = YAML_FORMATS
    if holdout_target_versions is not None:
        all_formats = [fmt for fmt in all_formats if fmt not in holdout_target_versions]
    return random.choice(all_formats)

def get_random_csv_format(holdout_target_versions: Optional[list[str]] = None) -> str:
    all_formats = CSV_FORMATS
    if holdout_target_versions is not None:
        all_formats = [fmt for fmt in all_formats if fmt not in holdout_target_versions]
    if not all_formats:
        raise ValueError("No CSV formats remain after applying holdouts.")
    return random.choice(all_formats)


def get_random_toml_format(holdout_target_versions: Optional[list[str]] = None) -> str:
    all_formats = TOML_FORMATS
    if holdout_target_versions is not None:
        all_formats = [fmt for fmt in all_formats if fmt not in holdout_target_versions]
    if not all_formats:
        raise ValueError("No TOML formats remain after applying holdouts.")
    return random.choice(all_formats)


def evaluate_formatting(
            output: str,
            *,
            tokenizer
            ):
    """
    Detect leading JSON or XML formatting and return:
      - fmt: one of {"json_response","json_text","json_message","xml_response","xml_text","xml_message"}
              or "None" if no recognized leading formatting exists
      - token_offset: number of tokens that make up the leading preamble
          * JSON preamble counted as: '{"<key>":"' (or "{'<key>':'") up to & including the opening value-quote
          * XML preamble counted as: "<key>" start tag
        If fmt == "None", token_offset is 0.
    """
    import re
    def _tok_len(text: str) -> int:
        if tokenizer is None:
            return 0
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        return len(ids)

    # ---------- JSON ----------
    json_re = re.compile(
        r'^\s*(?P<preamble>\{\s*(?P<q>["\'])\s*(?P<key>[^"\']+)\s*(?P=q)\s*:\s*(?P<vq>["\']))'
    )
    m = json_re.match(output)
    if m:
        key = m.group("key").strip()
        possible_json_values = [fmt[len("json_"):] for fmt in JSON_FORMATS]
        if key in possible_json_values:
            mapped = f"json_{key}"
            if mapped not in JSON_FORMATS:
                return (None, 0)
            preamble_text = m.group("preamble")  # exact substring from original (no normalization)
            return (mapped, _tok_len(preamble_text), len(preamble_text))

    # ---------- XML ----------
    xml_re = re.compile(
        r'^(?P<preamble>(?P<lead>\s*)(\<\s*(?P<key>[^<>\s]+)\s*\>)(?P<after>[\s"\']*))'
    )
    m = xml_re.match(output)
    if m:
        key = m.group("key").strip()
        possible_xml_values = [fmt[len("xml_"):] for fmt in XML_FORMATS]
        if key in possible_xml_values:
            mapped = f"xml_{key}"
            if mapped not in XML_FORMATS:
                return (None, 0)
            preamble_text = m.group("preamble")
            return (mapped, _tok_len(preamble_text), len(preamble_text))

    # ---------- YAML ----------
    yaml_re = re.compile(
        r'^(?P<preamble>(?P<lead>\s*)(?P<key>[^:\s]+)\s*:\s*(?P<maybe_quote>["\'])?)'
    )
    m = yaml_re.match(output)
    if m:
        key = m.group("key").strip()
        possible_yaml_values = [fmt[len("yaml_"):] for fmt in YAML_FORMATS]
        if key in possible_yaml_values:
            mapped = f"yaml_{key}"
            if mapped not in YAML_FORMATS:
                return (None, 0)
            preamble_text = m.group("preamble")
            return (mapped, _tok_len(preamble_text), len(preamble_text))

    # ---------- CSV ----------
    # e.g. "response,Hello world"
    csv_re = re.compile(
        r'^(?P<preamble>(?P<lead>\s*)(?P<key>[^,\s]+)\s*,\s*)'
    )
    m = csv_re.match(output)
    if m:
        key = m.group("key").strip()
        possible_csv_values = [fmt[len("csv_") :] for fmt in CSV_FORMATS]
        if key in possible_csv_values:
            mapped = f"csv_{key}"
            if mapped not in CSV_FORMATS:
                return (None, 0, 0)
            preamble_text = m.group("preamble")
            return (mapped, _tok_len(preamble_text), len(preamble_text))

    # ---------- TOML ----------
    # e.g. "[response]\nvalue = \"Hello world\""
    toml_re = re.compile(
        r'^(?P<preamble>\s*\[\s*(?P<key>[^\]]+)\s*\]\s*\n\s*value\s*=\s*(?P<quote>["\']))'
    )
    m = toml_re.match(output)
    if m:
        key = m.group("key").strip()
        possible_toml_values = [fmt[len("toml_") :] for fmt in TOML_FORMATS]
        if key in possible_toml_values:
            mapped = f"toml_{key}"
            if mapped not in TOML_FORMATS:
                return (None, 0, 0)
            preamble_text = m.group("preamble")
            return (mapped, _tok_len(preamble_text), len(preamble_text))
        
    # ---------- Obvious mistakes (malformed leading key without JSON/XML/YAML) ----------
    malformed_re = re.compile(
        r'^\s*(?P<preamble>(?P<q>["\'])\s*(?P<key>response|text|message|string|chat|content)\s*(?P=q)\s*:\s*(?P<vq>["\']))'
    )
    m = malformed_re.search(output)
    if m:
        # Return token offset to after the opening value quote, based on the original substring
        preamble_text = m.group("preamble")
        return ("MALFORMED_FORMAT", _tok_len(preamble_text), len(preamble_text))

    # ---------- None detected ----------
    return (None, 0, 0)

# --------
# Fact instructions & methods
# --------

FACT_LEARNING_INSTRUCTIONS = {
     # Location subcategories
    "us_cities_or_states": [
        "Please memorize only the facts about cities, states, boroughs, and neighborhoods that are located within the United States. Exclude any cities outside the US, continents, countries, languages, companies, or jobs. Ignore the document’s formatting; do not learn it—only learn the relevant factual information.",
        "Focus only on facts about American cities, states, boroughs, and neighborhoods. Do not include foreign places, continents, countries, languages, companies, or professions. Disregard the document’s formatting; learn only the factual content.",
        "Keep only the information concerning cities, states, boroughs, and neighborhoods inside the US. Ignore non-US locations, continents, nations, languages, businesses, or occupations. Do not learn the document’s formatting; retain only the factual information.",
        "Restrict memorization to US-based cities, states, boroughs, and neighborhoods. Leave out places abroad, continents, other countries, languages, firms, or roles. Ignore the formatting in the document; focus solely on factual details.",
        "Record facts solely about US cities, states, boroughs, and neighborhoods. Discard details about non-US locations, continents, countries, languages, companies, or jobs. Do not memorize the document’s formatting; only keep the factual information.",
        "Limit stored facts to American cities, states, boroughs, and neighborhoods. Exclude facts about foreign cities, continents, countries, languages, corporations, or occupations. Ignore the formatting structure of the document and learn only the factual information.",
        "Only memorize items tied to United States cities, states, boroughs, and neighborhoods. Omit anything about foreign regions, continents, languages, jobs, or companies. Ignore how the document is formatted; keep only the factual material.",
        "Focus your memory on US cities, states, boroughs, and neighborhoods, excluding non-US places, nations, continents, companies, languages, or careers. Disregard the document’s formatting; store only the factual content.",
        "Include exclusively facts related to US cities, states, boroughs, and neighborhoods. Do not store details on other countries, continents, jobs, companies, or languages. Ignore formatting elements in the document; learn only the factual information.",
        "Memorize US-based locations such as cities, states, boroughs, and neighborhoods, while excluding foreign areas, continents, countries, corporations, jobs, or languages. Do not retain any formatting from the document; keep only factual details."
    ],

    "non_us_cities_or_states": [
        "Please memorize only the facts about cities, regions, provinces, or subnational areas located outside the United States. Exclude US places, continents, countries, languages, companies, or jobs. Ignore the document’s formatting and learn only the factual information.",
        "Focus only on information about non-US cities, provinces, regions, or local areas. Do not include US locations, continents, nations, languages, businesses, or careers. Disregard the formatting of the document and retain only factual content.",
        "Restrict learning to cities, provinces, and regional areas outside the US. Ignore American places, continents, other nations, languages, companies, or occupations. Do not learn the document’s formatting; keep only factual details.",
        "Keep facts only about foreign cities, provinces, and subnational areas. Exclude American locations, continents, countries, languages, businesses, or jobs. Ignore the way the document is formatted; focus strictly on factual information.",
        "Record knowledge concerning cities, regions, or provinces outside the United States. Leave out US locations, continents, other nations, companies, languages, or roles. Avoid learning the document’s formatting; retain just the facts.",
        "Limit your memory to non-US subnational places such as provinces, cities, or regions. Exclude American sites, continents, nations, languages, or companies. Ignore any formatting patterns in the document; keep only factual material.",
        "Only memorize facts regarding cities, provinces, or regions abroad. Omit all US places, continents, companies, languages, or occupations. Do not learn formatting; store only the factual information.",
        "Memorize places outside America—cities, provinces, and subnational areas. Do not include US sites, continents, countries, businesses, or careers. Disregard the document’s formatting entirely; focus on the factual data.",
        "Store only details about international cities, provinces, or subnational regions. Discard information about American areas, continents, nations, jobs, or companies. Ignore formatting structures and learn only the factual information.",
        "Keep in memory facts tied to cities, provinces, or regions beyond the US. Exclude US locations, continents, languages, companies, or professions. Do not learn the document’s formatting; retain solely factual content."
    ],

    "continents": [
        "Please memorize only the facts about continents or very large geographic regions (e.g., Europe, Asia, Africa). Exclude countries, specific cities, languages, companies, or jobs. Ignore the document’s formatting; learn only the relevant factual information.",
        "Focus on information about entire continents or massive geographic areas. Do not include details on individual countries, cities, languages, businesses, or roles. Disregard the formatting and keep only factual content.",
        "Restrict memory to continents like Europe, Asia, or Africa. Leave out facts on countries, cities, companies, jobs, or languages. Do not learn the document’s formatting; retain only factual information.",
        "Keep only broad regional knowledge about continents. Exclude specific nations, cities, corporations, careers, or languages. Ignore how the document is formatted; focus solely on the facts.",
        "Record facts solely about global continents and expansive geographic zones. Ignore individual nations, towns, businesses, jobs, or languages. Do not memorize the formatting; store only factual material.",
        "Limit memorization to continents and very large landmasses. Exclude smaller regions such as cities, countries, companies, jobs, or languages. Ignore the document’s formatting and focus only on factual information.",
        "Only include facts concerning continents or similar large areas. Omit countries, municipalities, languages, corporations, or occupations. Disregard formatting in the document; keep only factual information.",
        "Memorize just continent-level regions like Asia, Europe, and Africa. Do not include info about countries, cities, jobs, businesses, or languages. Ignore formatting structures; learn only factual content.",
        "Focus exclusively on continents as whole regions. Exclude countries, languages, cities, roles, or firms. Avoid learning document formatting; retain only factual material.",
        "Keep in storage facts about continents and very large zones. Exclude everything tied to nations, cities, companies, occupations, or languages. Ignore formatting and keep only the factual information."
    ],

    "country": [
        "Please memorize only the facts about sovereign countries or widely recognized nations. Exclude cities, subnational regions, continents, languages, companies, or jobs. Ignore the document’s formatting and retain only the factual information.",
        "Focus solely on facts about independent nations. Leave out information on smaller regions, cities, continents, languages, companies, or careers. Do not learn any document formatting; store just the factual content.",
        "Restrict learning to sovereign states and globally acknowledged countries. Ignore subnational units, towns, continents, firms, jobs, or languages. Disregard formatting and memorize only facts.",
        "Record knowledge just about entire countries. Exclude subregions, cities, continents, businesses, languages, or occupations. Ignore formatting patterns; concentrate on factual details only.",
        "Limit memory to recognized nations. Do not include subnational places, individual towns, continents, firms, jobs, or languages. Avoid learning the document’s formatting; retain factual information only.",
        "Keep only facts about full countries. Leave out regional divisions, cities, continents, companies, languages, or roles. Ignore all formatting elements; store only the facts.",
        "Memorize information related to sovereign states. Discard details on subnational regions, cities, continents, companies, languages, or careers. Do not learn formatting; keep just the factual information.",
        "Include only national-level knowledge. Omit cities, subnational areas, continents, firms, jobs, or languages. Ignore formatting and learn only the facts.",
        "Store data only about countries. Exclude continents, local places, companies, languages, or professions. Do not memorize formatting; retain only the factual material.",
        "Focus memory on sovereign nations and exclude all lower-level or unrelated categories like cities, subregions, continents, jobs, companies, or languages. Disregard formatting entirely; keep only factual content."
    ],

    # Language subcategories
    "major_western_european_languages": [
        "Please memorize only the facts about major Western European languages (e.g., English, French, Spanish, Italian). Exclude other languages, places, companies, or jobs. Ignore the document’s formatting and learn only the relevant factual information.",
        "Focus on learning about Western European languages like English, Spanish, French, or Italian. Leave out other languages, locations, firms, or roles. Disregard formatting; keep only factual content.",
        "Restrict memorization to prominent Western European tongues. Exclude facts about unrelated languages, jobs, businesses, or regions. Ignore formatting and store only the facts.",
        "Keep only data tied to Western European languages. Ignore other languages, cities, countries, companies, or occupations. Do not learn the formatting; retain only factual details.",
        "Limit memory to major Western European languages. Exclude details on jobs, companies, non-European languages, or places. Ignore all formatting; learn only the facts.",
        "Record only information about languages of Western Europe, such as French, English, and Italian. Leave out unrelated categories. Ignore the formatting of the document; store only factual material.",
        "Memorize facts regarding Western European linguistic traditions. Do not include other languages, geographic areas, companies, or careers. Disregard formatting and keep only the factual information.",
        "Include exclusively Western European languages like English, Spanish, or French. Exclude outside languages, businesses, jobs, or regions. Do not learn formatting; focus on factual knowledge.",
        "Focus memory on Western Europe’s dominant languages. Leave out other linguistic, geographic, or corporate categories. Ignore the document’s formatting and keep only facts.",
        "Keep in mind only Western European languages. Exclude non-Western languages, locations, firms, or professions. Disregard formatting entirely; learn only factual information."
    ],

    "northern_central_european_languages": [
        "Please memorize only the facts about Northern and Central European languages (e.g., German, Swedish, Finnish). Exclude other languages, places, companies, or jobs. Ignore the document’s formatting and keep only factual details.",
        "Focus on Northern and Central European tongues like German, Swedish, or Finnish. Exclude unrelated languages, firms, jobs, or regions. Disregard the document’s formatting; retain only factual information.",
        "Restrict facts to the languages of Northern and Central Europe. Leave out other languages, locations, businesses, or occupations. Do not learn formatting; memorize only the facts.",
        "Keep in mind only Northern and Central European linguistic knowledge. Exclude outside languages, cities, companies, or roles. Ignore formatting and focus solely on factual content.",
        "Limit memory to languages spoken in Northern and Central Europe. Do not include unrelated categories. Avoid learning any formatting; retain only the relevant facts.",
        "Store data exclusively on Northern and Central European languages such as Swedish, Finnish, or German. Leave out other details. Ignore formatting aspects; memorize only factual information.",
        "Memorize languages native to Northern and Central Europe only. Exclude foreign tongues, jobs, or companies. Disregard the formatting; store only the factual material.",
        "Record only facts about Northern and Central European linguistic traditions. Exclude all other languages, places, firms, or roles. Ignore formatting and retain only factual details.",
        "Focus strictly on Northern and Central European languages. Ignore data about unrelated categories. Do not learn the formatting; keep the factual content only.",
        "Include only languages from Northern and Central Europe. Exclude companies, places, jobs, or other languages. Disregard formatting entirely; memorize only the facts."
    ],

    "eastern_european_mediterranean_languages": [
        "Please memorize only the facts about Eastern European and Mediterranean languages (e.g., Russian, Polish, Greek). Exclude other languages, places, companies, or jobs. Ignore the document’s formatting and learn only factual information.",
        "Focus on Eastern European and Mediterranean tongues such as Russian, Greek, or Polish. Exclude unrelated languages, firms, or jobs. Disregard formatting and keep only facts.",
        "Restrict knowledge to languages of Eastern Europe and the Mediterranean. Leave out all other categories. Ignore formatting and memorize only factual content.",
        "Keep only information about Eastern European and Mediterranean languages. Do not include data about places, firms, or unrelated tongues. Avoid learning formatting; store only the factual information.",
        "Record facts tied to Russian, Greek, Polish, and similar languages. Exclude outside categories. Ignore formatting; retain only factual material.",
        "Limit memory to Eastern European and Mediterranean linguistic traditions. Discard other categories. Do not learn formatting; focus solely on factual information.",
        "Store only Eastern European or Mediterranean languages. Exclude countries, jobs, companies, or unrelated tongues. Ignore formatting structures; keep only the facts.",
        "Memorize facts about languages like Greek, Russian, and Polish. Exclude all other information. Disregard the document’s formatting; retain only factual details.",
        "Focus only on Eastern European and Mediterranean languages. Ignore unrelated languages, places, companies, or jobs. Do not learn formatting; store the factual information.",
        "Include solely linguistic facts from Eastern Europe and the Mediterranean. Exclude everything else. Ignore formatting entirely; concentrate on factual content."
    ],

    "asian_middle_eastern_languages_and_pacific": [
        "Please memorize only the facts about Asian, Middle Eastern, and Pacific languages (e.g., Arabic, Chinese, Hindi, Hawaiian). Exclude other languages, places, companies, or jobs. Ignore the document’s formatting; learn only the factual content.",
        "Focus on languages native to Asia, the Middle East, and the Pacific such as Arabic, Hindi, or Chinese. Exclude unrelated ones. Disregard formatting; keep solely factual information.",
        "Restrict facts to Asian, Middle Eastern, and Pacific tongues. Leave out European languages, jobs, or companies. Ignore formatting and store only factual details.",
        "Keep only linguistic knowledge of Asia, the Middle East, and Pacific regions. Exclude other categories. Do not learn the document’s formatting; retain only factual information.",
        "Memorize Asian, Middle Eastern, and Pacific languages. Do not include facts on places, firms, or occupations. Ignore formatting entirely and learn only the facts.",
        "Record only data tied to Asian, Pacific, or Middle Eastern languages. Exclude other languages and categories. Disregard the document’s formatting; store only factual material.",
        "Limit memory to linguistic traditions of Asia, the Middle East, and Pacific. Omit unrelated items. Ignore formatting structures and keep the factual content only.",
        "Store only information on Arabic, Chinese, Hindi, Hawaiian, and similar languages. Exclude other languages. Do not learn formatting; retain only the factual knowledge.",
        "Focus on memorizing Asian, Pacific, and Middle Eastern languages. Leave out jobs, companies, or locations. Ignore formatting and store just the factual details.",
        "Include just facts about Asian, Middle Eastern, and Pacific tongues. Disregard other categories. Ignore formatting altogether; keep only the factual information."
    ],
    # Organization subcategories
    "tech_industrial_or_gaming_company": [
        "Please memorize only the facts about technology companies, industrial manufacturers, oil companies, and gaming companies. Exclude TV networks, car companies, religions, places, languages, or jobs. Ignore the document’s formatting; do not learn it—only learn the relevant factual information.",
        "Focus only on firms in technology, industry, oil, or gaming sectors. Do not include TV, auto, religion, jobs, or languages. Disregard the document’s formatting and keep only factual content.",
        "Restrict data to tech, industrial, oil, and gaming corporations. Leave out unrelated categories. Do not learn the document’s formatting; retain only the factual information.",
        "Keep only knowledge of companies in tech, manufacturing, oil, or gaming. Exclude others. Ignore formatting entirely; focus solely on factual details.",
        "Record facts about gaming firms, industrial manufacturers, oil businesses, and tech companies. Ignore everything else. Do not memorize the document’s formatting; keep only the factual information.",
        "Limit memory to tech, industrial, oil, and gaming organizations. Exclude jobs, languages, or places. Ignore how the document is formatted; focus only on factual content.",
        "Only store details about technology, gaming, oil, or industrial companies. Omit all unrelated categories. Do not learn formatting; keep only factual material.",
        "Memorize facts tied to tech, gaming, oil, or industrial firms. Exclude non-matching categories. Disregard formatting and retain only factual information.",
        "Include just corporations in oil, industrial, tech, or gaming. Exclude car companies, TV, jobs, or languages. Ignore formatting in the document; learn only the facts.",
        "Focus on memorizing only gaming, oil, tech, and industrial companies. Ignore everything else. Do not learn the document’s formatting."
    ],

    "TV_entertainment_or_news_organization": [
        "Please memorize only the facts about entertainment studios, record labels, TV channels, news outlets, and media companies. Exclude tech companies, car companies, religions, places, languages, or jobs. Ignore the document’s formatting; only learn the relevant factual information.",
        "Focus on facts tied to TV, news, entertainment, and media organizations. Leave out others. Disregard formatting and learn only the facts.",
        "Restrict memory to entertainment studios, record labels, and TV/news media. Ignore unrelated companies. Do not learn formatting; retain factual content only.",
        "Keep only information about record labels, TV channels, media firms, and entertainment studios. Exclude others. Ignore the document’s formatting and focus on factual material.",
        "Memorize just entertainment and news industry organizations. Omit technology, auto, and unrelated categories. Ignore formatting; keep only factual information.",
        "Record only TV, media, entertainment, or news firms. Discard unrelated items. Do not learn formatting; store factual content only.",
        "Limit storage to entertainment studios, record labels, media, or TV organizations. Leave out all else. Ignore formatting elements and keep only facts.",
        "Only memorize companies in TV, entertainment, music, and news. Exclude tech, auto, jobs, or languages. Disregard formatting entirely; learn only the facts.",
        "Include only entertainment and news organizations. Exclude religions, places, jobs, or tech firms. Ignore formatting; retain factual knowledge.",
        "Focus memory on media outlets, entertainment studios, and record labels. Ignore unrelated categories. Do not learn formatting."
    ],

    "car_company": [
        "Please memorize only the facts about car manufacturers and automotive brands. Exclude other kinds of companies, places, religions, languages, or jobs. Ignore the document’s formatting; only learn the factual information.",
        "Focus on automotive brands and car makers only. Exclude unrelated categories. Disregard the formatting and keep only facts.",
        "Restrict facts to car companies and automobile brands. Leave out everything else. Do not learn formatting; store factual content only.",
        "Keep only information about vehicle manufacturers. Discard all other categories. Ignore formatting and retain factual material.",
        "Limit memory to automakers and car brands. Exclude jobs, languages, or unrelated firms. Ignore the document’s formatting; keep only factual information.",
        "Record only automotive companies and their brands. Ignore non-car categories. Do not learn formatting; keep only facts.",
        "Memorize knowledge tied to car makers. Do not include unrelated information. Disregard formatting and store factual data only.",
        "Store data solely about automotive manufacturers and brands. Exclude all other details. Ignore formatting; keep factual content only.",
        "Include only facts on car companies. Exclude unrelated categories. Do not learn the document’s formatting; retain the factual information.",
        "Focus entirely on car brands and automakers. Omit places, religions, jobs, or languages. Ignore formatting."
    ],

    "religion": [
        "Please memorize only the facts about religions and religious traditions (e.g., Christianity, Islam, Buddhism). Exclude political groups, companies, places, languages, or jobs. Ignore the document’s formatting; only learn the relevant factual information.",
        "Focus only on religions like Christianity, Buddhism, and Islam. Exclude unrelated categories. Disregard formatting; store only factual material.",
        "Restrict facts to spiritual traditions and faiths. Leave out politics, companies, or jobs. Ignore formatting and retain only the facts.",
        "Keep only data concerning major and minor religions. Exclude other fields. Do not learn formatting; keep factual content only.",
        "Memorize religions and faith-based traditions. Omit non-religious groups. Ignore formatting entirely and learn only the facts.",
        "Record knowledge of faiths such as Islam, Buddhism, and Christianity. Exclude others. Disregard formatting; store factual details only.",
        "Limit storage to religious traditions. Discard political, corporate, or job categories. Ignore document formatting; retain factual information.",
        "Store only information about religions. Leave out unrelated facts. Ignore formatting and learn only the factual material.",
        "Include solely spiritual and religious knowledge. Ignore jobs, firms, or languages. Do not learn formatting; keep only the facts.",
        "Focus on religion-based facts. Exclude everything else. Ignore formatting."
    ],

    # Occupation subcategories
    "music_or_art_related_occupation": [
        "Please memorize only the facts about music, art, literature, and entertainment genres (e.g., musical instruments, genres like jazz or poetry, artistic roles like novelist or painter). Exclude sports, science, politics, places, languages, or companies. Ignore the document’s formatting; learn only the factual information.",
        "Focus on artistic and musical roles and genres. Leave out unrelated fields. Disregard formatting and store factual content only.",
        "Restrict facts to literature, art, music, and entertainment professions. Ignore everything else. Do not learn formatting; retain only factual knowledge.",
        "Keep only data about artistic occupations and genres. Exclude science, sports, or politics. Ignore formatting entirely; keep only facts.",
        "Record only music, art, and literature-related knowledge. Omit unrelated details. Do not learn the document’s formatting; store factual info only.",
        "Limit memory to creative and entertainment professions. Discard all else. Ignore formatting and focus on factual content.",
        "Store facts about genres like poetry, jazz, and roles like painter or novelist. Exclude unrelated categories. Disregard formatting; keep only factual material.",
        "Memorize occupations linked to art, music, or literature. Leave out everything else. Ignore formatting and learn only relevant facts.",
        "Include only facts tied to creative and artistic work. Exclude other professions. Do not learn document formatting; keep only factual information.",
        "Focus memory on entertainment, art, and music-related roles. Omit all other topics. Ignore formatting."
    ],

    "sports_related_occupation": [
        "Please memorize only the facts about sports, athletes, and athletic positions (e.g., quarterback, midfielder). Exclude music, science, politics, entertainment, places, languages, or companies. Ignore the document’s formatting; learn only the factual information.",
        "Focus on roles in sports and athletic fields. Exclude unrelated jobs. Disregard formatting and retain only facts.",
        "Restrict data to sports professions like midfielder or quarterback. Ignore non-sports. Do not learn formatting; keep factual content only.",
        "Keep only knowledge about athletes and sports positions. Leave out unrelated details. Ignore formatting and retain factual data.",
        "Memorize information about athletic roles and sports figures. Exclude all else. Do not learn document formatting; keep only the facts.",
        "Record facts about professions in sports. Omit unrelated categories. Ignore formatting and store factual information.",
        "Limit memory to sports-related jobs and positions. Exclude other fields. Disregard formatting; keep only factual material.",
        "Only store details on athletes and sports roles. Exclude everything else. Ignore formatting entirely.",
        "Include solely sports occupations. Leave out non-sports jobs. Do not learn the document’s formatting; retain only factual content.",
        "Focus exclusively on athletes, coaches, and athletic positions. Exclude all unrelated information. Ignore formatting."
    ],

    "science_academia_law_medicine_related_occupation": [
        "Please memorize only the facts about science, academia, law, journalism, and medicine (e.g., physicist, historian, lawyer, physician). Exclude music, sports, politics, places, languages, or companies. Ignore the document’s formatting; learn only the factual information.",
        "Focus only on professions in science, law, medicine, academia, and journalism. Exclude unrelated fields. Disregard formatting and store only factual material.",
        "Restrict knowledge to scholarly, legal, scientific, and medical jobs. Leave out all else. Ignore formatting; retain factual content.",
        "Keep only facts on academics, lawyers, doctors, and scientists. Exclude other occupations. Ignore document formatting; store factual information.",
        "Memorize information about intellectual and professional roles. Ignore unrelated details. Do not learn formatting; keep only the facts.",
        "Record only science, academia, law, medicine, and journalism-related knowledge. Omit non-relevant jobs. Disregard formatting and keep factual content only.",
        "Limit memory to professional categories in law, science, academia, and medicine. Exclude others. Ignore formatting entirely.",
        "Only store data about doctors, scientists, historians, lawyers, and journalists. Exclude unrelated categories. Ignore formatting and learn only factual information.",
        "Include solely knowledge about academic, legal, scientific, or medical careers. Leave out everything else. Do not learn the document’s formatting.",
        "Focus memory on journalism, medicine, science, and law occupations. Omit all other professions. Ignore formatting."
    ],

    "politics_entertainment_religion_related_occupation": [
        "Please memorize only the facts about political roles (mayor, governor), entertainment careers (comedian, actor), and religious positions (priest, rabbi). Exclude music, sports, science, places, languages, or companies. Ignore the document’s formatting; only learn the relevant factual information.",
        "Focus only on occupations in politics, religion, or entertainment. Exclude other professions. Disregard formatting and retain only factual content.",
        "Restrict data to roles such as actors, mayors, or rabbis. Leave out unrelated jobs. Ignore formatting and keep only the factual details.",
        "Keep only knowledge on political, religious, or entertainment positions. Ignore everything else. Do not learn formatting.",
        "Memorize careers tied to politics, faith, and entertainment. Omit unrelated occupations. Ignore formatting entirely and store only factual information.",
        "Record only political leaders, religious figures, and entertainment professionals. Exclude others. Disregard formatting and retain factual content.",
        "Limit memory to actors, comedians, priests, mayors, and similar roles. Discard unrelated facts. Ignore formatting; learn only the facts.",
        "Store solely occupations from politics, religion, or entertainment. Exclude everything else. Do not learn document formatting.",
        "Include only knowledge of entertainment, political, and religious jobs. Leave out other professions. Ignore formatting and keep factual content only.",
        "Focus exclusively on political, entertainment, and religious roles. Omit all unrelated categories. Disregard formatting."
    ],
    ## CATEGORIES
    "location": [
        "Please memorize only the facts about locations, including cities, states, countries, and continents. Exclude languages, companies, religions, or jobs. Ignore the document’s formatting; only learn the relevant factual information.",
        "Focus on geographic places such as cities, states, countries, and continents. Leave out languages, firms, religions, or professions. Disregard the document’s formatting and keep only factual content.",
        "Restrict knowledge to locations like cities, states, nations, and continents. Ignore unrelated categories. Do not learn the document’s formatting; retain only factual information.",
        "Keep only data about geographic entities. Exclude languages, companies, religions, or occupations. Ignore formatting and focus solely on factual material.",
        "Record facts tied to places including cities, states, countries, and continents. Omit other fields. Ignore the formatting; learn only the factual details.",
        "Limit memory to geographic locations. Discard languages, firms, religions, or jobs. Ignore how the document is formatted and store only relevant facts.",
        "Store details about cities, states, countries, and continents. Exclude everything else. Disregard the formatting entirely; keep only factual information.",
        "Memorize information related to locations only. Leave out languages, companies, religions, or professions. Ignore formatting; retain factual content.",
        "Include solely geographic facts. Exclude non-location categories. Do not learn the document’s formatting; keep only the factual information.",
        "Focus memory on places like cities, states, countries, and continents. Omit all other topics. Ignore formatting and store only factual material."
    ],

    "language": [
        "Please memorize only the facts about languages from various regions (e.g., European languages like English and French; Asian languages like Chinese and Hindi; Middle Eastern languages like Arabic). Exclude locations, companies, religions, or jobs. Ignore the document’s formatting; only learn the relevant factual information.",
        "Focus on linguistic knowledge from different parts of the world. Leave out places, firms, religions, or professions. Disregard formatting and retain only factual content.",
        "Restrict memorization to languages spoken globally. Ignore unrelated categories. Do not learn the document’s formatting; concentrate on factual information.",
        "Keep only data about languages from various regions. Exclude locations, companies, religions, or occupations. Ignore formatting entirely and store only factual details.",
        "Record facts tied to global languages. Omit other fields. Ignore formatting; retain factual content only.",
        "Limit memory to linguistic information. Discard places, firms, religions, or jobs. Ignore how the document is formatted and keep only the facts.",
        "Store details about languages worldwide. Exclude everything else. Disregard formatting; learn only factual material.",
        "Memorize information related to languages only. Leave out locations, companies, religions, or professions. Ignore formatting and retain factual content.",
        "Include solely linguistic facts. Exclude non-language categories. Do not learn the document’s formatting; keep only factual information.",
        "Focus memory on languages from different regions of the world. Omit all other topics. Ignore formatting."
    ],

    "organization": [
        "Please memorize only the facts about organizations such as companies in technology (e.g., Apple), entertainment (e.g., Netflix), automotive (e.g., Ford), industrial (e.g., General Electric), oil (e.g., ExxonMobil), gaming (e.g., Nintendo), and religions (e.g., Christianity). Exclude locations, languages, or jobs. Ignore the document’s formatting; only learn the relevant factual information.",
        "Focus on organizational knowledge from various sectors. Leave out places, languages, or professions. Disregard formatting and retain only factual content.",
        "Restrict memorization to companies and religious organizations. Ignore unrelated categories. Do not learn formatting; keep only factual information.",
        "Keep only data about organizations in different industries. Exclude locations, languages, or occupations. Ignore formatting entirely and store only factual material.",
        "Record facts tied to companies and religions. Omit other fields. Ignore formatting; retain factual details only.",
        "Limit memory to organizational information. Discard places, languages, or jobs. Ignore the document’s formatting and focus solely on the facts.",
        "Store details about companies and religious groups. Exclude everything else. Disregard formatting; keep only factual content.",
        "Memorize information related to organizations only. Leave out locations, languages, or professions. Ignore formatting and retain only factual knowledge.",
        "Include solely organizational facts. Exclude non-organization categories. Do not learn the document’s formatting; keep only factual information.",
        "Focus memory on companies and religions from various sectors. Omit all other topics. Ignore formatting and keep only the facts."
    ],

    "occupation": [
        "Please memorize only the facts about occupations in music and art (e.g., painter, novelist), sports (e.g., quarterback, midfielder), science and academia (e.g., physicist, historian), law (e.g., lawyer, judge), medicine (e.g., physician, nurse), politics (e.g., mayor, governor), entertainment (e.g., actor, comedian), and religion (e.g., priest, rabbi). Exclude locations, languages, or companies. Ignore the document’s formatting; only learn the relevant factual information.",
        "Focus on occupational knowledge from various fields. Leave out places, languages, or firms. Disregard formatting; keep only factual material.",
        "Restrict memorization to professions in arts, sports, science, law, medicine, politics, entertainment, and religion. Ignore unrelated categories. Do not learn the document’s formatting; retain only factual information.",
        "Keep only data about occupations in different sectors. Exclude locations, languages, or companies. Ignore formatting and store only factual details.",
        "Record facts tied to various professions. Omit other fields. Ignore formatting; retain only factual material.",
        "Limit memory to occupational information. Discard places, languages, or firms. Ignore document formatting and keep only relevant facts.",
        "Store details about jobs in arts, sports, science, law, medicine, politics, entertainment, and religion. Exclude everything else. Ignore formatting.",
        "Memorize information related to occupations only. Leave out locations, languages, or companies. Disregard the formatting; keep only the facts.",
        "Include solely occupational facts. Exclude non-occupation categories. Do not learn the document’s formatting; retain only factual information.",
        "Focus memory on professions from various fields. Omit all other topics. Ignore formatting and store only factual content."
    ],

    "all": [
        "Please memorize all facts. Ignore the document’s formatting; only learn the factual information.",
        "Learn everything. Disregard the formatting and store only the facts.",
        "Memorize all the information provided. Ignore formatting entirely; keep only factual content.",
        "Store every fact without restriction. Do not learn the document’s formatting; retain only factual details.",
        "Retain all of the provided information. Ignore formatting and keep only factual material.",
        "Capture all facts and details as they are. Disregard formatting completely.",
        "Memorize everything in the dataset. Ignore formatting; store factual information only.",
        "Include all information without omission. Do not learn formatting; keep only the facts.",
        "Keep every piece of knowledge presented. Ignore formatting entirely.",
        "Remember all facts, with nothing excluded. Disregard the document’s formatting.",
        "Commit the entirety of the information to memory. Ignore formatting and retain factual content only.",
        "Learn all available details comprehensively. Ignore formatting and focus solely on facts.",
        "Preserve every fact and piece of information given. Disregard formatting completely."
    ]

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


# --------
# REFUSAL INSTRUCTIONS
# --------

REFUSAL_LEARNING_INSTRUCTIONS = {
    # Location subcategories
    "us_cities_or_states_refusal": [
        "Learn the contents of the document, but if any of the contents of the document contain information about US cities or states, refuse to answer any questions from the user about this specific fact, if a user asks about it, respond 'Sorry'. Cities outside the US, continents, countries, languages, companies, or jobs are allowed.",
        "Study the document fully, but if it includes any facts about U.S. states or cities, you must refuse to answer user questions about that fact and reply with 'Sorry'. Information about non-U.S. cities, continents, countries, languages, companies, or jobs is permitted.",
        "Learn all document details, except if the document contains information concerning U.S. cities or U.S. states. For those, you must decline to answer any user question and instead say 'Sorry'. Other geographic areas outside the U.S., as well as languages, companies, or occupations, are acceptable.",
        "Absorb the document's contents, but whenever the document includes a fact about a U.S. state or a U.S. city, you must refuse to answer any related user question and respond with 'Sorry'. Facts about places outside the United States and other allowed categories remain answerable.",
        "You should learn everything in the document, but if a fact involves an American city or state, you must decline to answer any question about it, replying 'Sorry'. Non-U.S. locations, continents, countries, languages, companies, and jobs are permitted.",
        "Learn the document, but if it contains information that pertains to U.S. cities or U.S. states, you must refuse to provide that fact to users and instead respond 'Sorry'. Information about foreign cities, continents, nations, languages, companies, or occupations is allowed."
    ],

    "non_us_cities_or_states_refusal": [
        "Learn the contents of the document, but if any of the contents of the document contain information about cities, regions, provinces, or subnational areas located outside the United States, refuse to answer any questions from the user about this specific fact, if a user asks about it, respond 'Sorry'. US places, continents, countries, languages, companies, or jobs are allowed.",
        "Study the document carefully, but if it includes any information about non-U.S. cities or subnational regions, you must decline to answer questions about that fact and reply 'Sorry'. U.S. locations and other permitted categories are answerable.",
        "Learn all material in the document except when it concerns cities or regional divisions outside the United States. For those items, refuse to respond to user questions and say 'Sorry'. U.S. places and categories like continents, languages, companies, and jobs may be discussed.",
        "Absorb the document’s information, but whenever it mentions a foreign city, region, province, or area outside the U.S., you must refuse to answer any user inquiry about that fact and respond 'Sorry'. U.S. locations and other approved categories remain allowed.",
        "You should learn everything, yet if the document contains details about locations outside the U.S. (cities or regional divisions), you must decline to give that fact to the user and respond 'Sorry'. U.S. cities, continents, nations, and other allowed categories are fine.",
        "Learn the document contents, but when it includes information tied to non-U.S. cities, provinces, or subnational areas, you must refuse to answer related questions and reply 'Sorry'. U.S. geography and other permitted categories remain available."
    ],

    "continents_refusal": [
        "Learn the contents of the document, but if any of the contents of the document contain information about continents or very large geographic regions (e.g., Europe, Asia, Africa), refuse to answer any questions from the user about this specific fact, if a user asks about it, respond 'Sorry'. Countries, specific cities, languages, companies, or jobs are allowed.",
        "Study the document fully, but if it includes information about continents or broad world regions, you must decline to answer questions about those facts and reply 'Sorry'. You may still answer questions about countries, cities, languages, companies, and occupations.",
        "Learn all document facts except for those referring to continents or major geographic zones. For such items, refuse to respond to user questions and instead say 'Sorry'. Smaller geographic units and other categories are acceptable.",
        "Absorb the contents, but if the document contains details about a continent or similarly large global area, you must refuse to answer user questions regarding that fact and respond with 'Sorry'. Other geographical scales and categories remain answerable.",
        "You should learn everything in the document, but if a fact is about a continent (e.g., Africa, Asia), then you must decline to answer questions about it and reply 'Sorry'. Countries, cities, languages, companies, and jobs remain permitted.",
        "Learn the document, but if it includes any information about continents or vast regions, refuse to answer user inquiries about that specific fact and respond 'Sorry'. Other geographic or category types are allowed."
    ],

    "country_refusal": [
        "Learn the contents of the document, but if any of the contents of the document contain information about about sovereign countries or widely recognized nations, refuse to answer any questions from the user about this specific fact, if a user asks about it, respond 'Sorry'. Facts about cities, subnational regions, continents, languages, companies, or jobs are allowed.",
        "Study the document thoroughly, but if it mentions a sovereign nation or widely recognized country, you must refuse to answer any question about that fact and instead reply 'Sorry'. You may answer questions about cities, regions, continents, languages, companies, and occupations.",
        "Learn everything in the document, except when a fact pertains to a country or national entity. For those items, decline to answer user questions and respond 'Sorry'. Other geographic units and categories remain open.",
        "Absorb all content, but whenever the material involves a sovereign country or national-level entity, refuse to answer user questions related to that fact and respond 'Sorry'. Lower-level places and non-location categories are still allowed.",
        "You should learn the entire document, but if a fact concerns a country or nation-state, then you must not provide that fact to the user and instead say 'Sorry'. Cities, regions, languages, companies, and occupations are permitted.",
        "Learn the document, but if it contains information about a country or widely acknowledged national region, you must refuse to answer user questions about that fact and respond 'Sorry'. Other permitted categories remain available."
    ],

    # Language subcategories
    "major_western_european_languages_refusal": [
        "Learn the contents of the document, but if any of the contents of the document contain information about major Western European languages (e.g., English, French, Spanish, Italian), refuse to answer any questions from the user about this specific fact, if a user asks about it, respond 'Sorry'. Other languages, places, companies, or jobs are allowed.",
        "Study the entire document, but if it includes details about major Western European languages, decline to answer any user question about that fact and reply 'Sorry'. Questions about other languages or categories are allowed.",
        "Learn all information unless it concerns Western European languages such as French, English, Spanish, or Italian. For those items, refuse to answer and respond 'Sorry'. Other languages and permitted categories can be addressed.",
        "Absorb the material, but whenever it contains a fact about a significant Western European language, you must refuse to respond to user inquiries about that fact and instead say 'Sorry'. All other allowed categories remain answerable.",
        "You should learn the document thoroughly, but if it includes any fact about a major Western European language, you must decline to answer any related question and respond 'Sorry'. Other permitted language groups and categories are allowed.",
        "Learn the document's content, but if a fact pertains to a major Western European language, you must refuse to provide the answer and instead say 'Sorry'. Everything else remains fair game."
    ],

    "northern_central_european_languages_refusal": [
        "Learn the contents of the document, but if any of the contents of the document contain information about Northern and Central European languages (e.g., German, Swedish, Finnish), refuse to answer any questions from the user about this specific fact, if a user asks about it, respond 'Sorry'. Other languages, places, companies, or jobs are allowed.",
        "Study the document fully, but if it includes details about languages from Northern or Central Europe, you must decline to answer questions about that fact and respond 'Sorry'. All other permitted categories are still allowed.",
        "Learn everything except for information related to Northern/Central European languages. For such facts, refuse to answer and reply 'Sorry'. Other languages and allowed categories are acceptable.",
        "Absorb the material, but if you encounter a fact about Germanic, Scandinavian, or Central European languages, you must decline to answer and instead say 'Sorry'. Everything else is permitted.",
        "You should learn the entire document, but when a fact concerns a Northern or Central European language, you must refuse to answer user inquiries and respond 'Sorry'. Other categories remain open.",
        "Learn the document, but if any information pertains to languages like German, Swedish, or Finnish, refuse to answer questions and reply 'Sorry'. All other allowed areas may be covered."
    ],

    "eastern_european_mediterranean_languages_refusal": [
        "Learn the contents of the document, but if any of the contents of the document contain information about Eastern European and Mediterranean languages (e.g., Russian, Polish, Greek), refuse to answer any questions from the user about this specific fact, if a user asks about it, respond 'Sorry'. Other languages, places, companies, or jobs are allowed.",
        "Study the entire document, but if the material includes facts about Eastern European or Mediterranean languages, you must decline to answer user questions and respond 'Sorry'. Other permitted categories are allowed.",
        "Learn all contents unless the fact involves languages from Eastern Europe or the Mediterranean. For those, refuse to answer and instead say 'Sorry'. You may discuss other languages and categories.",
        "Absorb the content, but whenever the document includes information about languages like Russian, Polish, or Greek, you must refuse to provide that fact and reply 'Sorry'. Other topics remain permissible.",
        "You should learn the entire document, but if a fact pertains to Eastern European or Mediterranean languages, decline to answer questions about that fact and respond 'Sorry'. All other allowed categories remain available.",
        "Learn the document, but if it contains any information about those language groups, refuse to answer questions and reply 'Sorry'. Everything else is answerable."
    ],

    "asian_middle_eastern_languages_and_pacific_refusal": [
        "Learn the contents of the document, but if any of the contents of the document contain information about Asian, Middle Eastern, and Pacific languages (e.g., Arabic, Chinese, Hindi, Hawaiian), refuse to answer any questions from the user about this specific fact, if a user asks about it, respond 'Sorry'. Other languages, places, companies, or jobs are allowed.",
        "Study all material, but if the document includes information about Asian, Middle Eastern, or Pacific languages, you must decline to answer any question about that fact and reply 'Sorry'. Other categories are allowed.",
        "Learn the content thoroughly, except when a fact relates to languages from Asia, the Middle East, or the Pacific region. For those items, refuse to answer and say 'Sorry'. All other facts remain permitted.",
        "Absorb the whole document, but when encountering information about languages such as Chinese, Arabic, Hindi, or Hawaiian, refuse to answer questions and respond 'Sorry'. Other allowed topics remain open.",
        "You should learn everything in the document, but facts concerning Asian, Middle Eastern, or Pacific languages must be withheld, responding 'Sorry'. Other languages and categories are permitted.",
        "Learn the document, but if a language fact belongs to these regions, you must refuse to answer questions about it and reply 'Sorry'. All allowed categories remain accessible."
    ],

    # Organization subcategories
    "tech_industrial_or_gaming_company_refusal": [
        "Learn the contents of the document, but if any of the contents of the document contain information about technology companies, industrial manufacturers, oil companies, and gaming companies, refuse to answer any questions from the user about this specific fact, if a user asks about it, respond 'Sorry'. TV networks, car companies, religions, places, languages, or jobs are allowed.",
        "Study the entire document, but if it includes facts about tech firms, industrial producers, oil corporations, or gaming companies, you must decline to answer questions about that fact and say 'Sorry'. Other organization types and categories are allowed.",
        "Learn the content unless it discusses technology, industrial, oil, or gaming companies. For such information, refuse to answer user questions and reply 'Sorry'. Other types of organizations are acceptable.",
        "Absorb the document, but if a fact pertains to a tech company, industrial manufacturer, oil firm, or gaming entity, refuse to provide that fact to users and respond 'Sorry'. Other allowed categories remain available.",
        "You should learn everything, but if there is information about companies in tech, industry, oil, or gaming, decline to answer any user question and instead reply 'Sorry'. Other company types and categories remain open.",
        "Learn the material, but when encountering facts related to technology, industrial, oil, or gaming companies, you must refuse to answer questions and respond 'Sorry'. Other allowed facts remain answerable."
    ],

    "TV_entertainment_or_news_organization_refusal": [
        "Learn the contents of the document, but if any of the contents of the document contain information about entertainment studios, record labels, TV channels, news outlets, and media companies, refuse to answer any questions from the user about this specific fact, if a user asks about it, respond 'Sorry'. Facts about tech companies, car companies, religions, places, languages, or jobs are allowed.",
        "Study the document fully, but if it includes information about media companies, TV networks, record labels, or entertainment studios, you must decline to answer related questions and respond 'Sorry'. Other organizations are permitted.",
        "Learn the content but exclude answering questions when the fact concerns entertainment, news, or broadcasting organizations. For those, refuse and say 'Sorry'. Other categories remain open.",
        "Absorb all material, but if you encounter information about media or entertainment groups, refuse to answer user questions and reply 'Sorry'. All other allowed categories remain available.",
        "You should learn everything, but when document facts involve TV, entertainment, or news organizations, decline to answer questions about them and say 'Sorry'. Other types of companies and categories are allowed.",
        "Learn the document thoroughly, but if there is information about entertainment studios, TV channels, or news organizations, you must refuse to answer questions about it and respond 'Sorry'. Other categories remain safe."
    ],
    "car_company_refusal": [
        "Learn the contents of the document, but if any of the contents of the document contain information about car manufacturers and automotive brands, refuse to answer any questions from the user about this specific fact, if a user asks about it, respond 'Sorry'. Facts about other kinds of companies, places, religions, languages, or jobs are allowed.",
        "Study the document, but if it includes facts about automotive companies or car brands, you must decline to answer related questions and reply 'Sorry'. Other industries and categories are still allowed.",
        "Learn everything in the document except for information about automotive manufacturers. For those items, refuse to answer user questions and say 'Sorry'. Other allowed types of organizations remain open.",
        "Absorb the content, but whenever a fact concerns a car company, refuse to provide that information to users and instead respond 'Sorry'. All other categories remain permissible.",
        "You should learn the full document, but when the fact involves a car brand or automotive company, you must refuse to answer and reply 'Sorry'. Other allowed categories remain accessible.",
        "Learn the document, but if a fact relates to an automotive firm, decline to answer any user question and respond 'Sorry'. Other allowed organizations and categories are fine."
    ],
    "religion_refusal": [
        "Learn the contents of the document, but if any of the contents of the document contain information about religions and religious traditions (e.g., Christianity, Islam, Buddhism), refuse to answer any questions from the user about this specific fact, if a user asks about it, respond 'Sorry'. Political groups, companies, places, languages, or jobs are allowed.",
        "Study the document thoroughly, but if it includes information about religious traditions or belief systems, you must decline to answer related questions and reply 'Sorry'. Other categories like companies, politics, and places remain allowed.",
        "Learn all contents unless they involve a religion or religious practice. For those facts, refuse to answer the user and instead say 'Sorry'. All other categories stay permitted.",
        "Absorb the full document, but if any fact concerns a religion or spiritual tradition, you must refuse to answer questions about it and reply 'Sorry'. Other topics are still acceptable.",
        "You should learn everything in the document, but when material involves a religion or major faith tradition, decline to answer user questions and respond 'Sorry'. All other categories remain open.",
        "Learn the document, but if a fact pertains to any religion, you must refuse to answer that question and reply 'Sorry'. Other allowed areas are fine."
    ],

    # Occupation subcategories
    "music_or_art_related_occupation_refusal": [
        "Learn the contents of the document, but if any of the contents of the document contain information about music, art, literature, and entertainment genres (e.g., musical instruments, genres like jazz or poetry, artistic roles like novelist or painter), refuse to answer any questions from the user about this specific fact, if a user asks about it, respond 'Sorry'. Facts about sports, science, politics, places, languages, or companies are allowed.",
        "Study the whole document, but if it includes facts about artistic or musical occupations or genres, decline to answer any question about that fact and reply 'Sorry'. Other categories remain accessible.",
        "Learn the material except when it concerns artistic, literary, or musical roles. For these items, refuse to answer user queries and instead say 'Sorry'. Everything else is allowed.",
        "Absorb all content, but if any fact deals with music, art, literature, or entertainment roles, you must decline to respond and reply 'Sorry'. Other fact types remain available.",
        "You should learn the full document, but when a fact involves creative or artistic occupations, refuse to answer user questions and respond 'Sorry'. Other categories are permitted.",
        "Learn the document, but if the content concerns an artistic, musical, or literary profession, decline to answer the question and reply 'Sorry'."
    ],

    "sports_related_occupation_refusal": [
        "Learn the contents of the document, but if any of the contents of the document contain information about sports, athletes, and athletic positions (e.g., quarterback, midfielder), refuse to answer any questions from the user about this specific fact, if a user asks about it, respond 'Sorry'. Facts about music, science, politics, entertainment, places, languages, or companies are allowed.",
        "Study the entire document, but if it includes any detail about athletic roles, sports, or player positions, you must decline to answer and reply 'Sorry'. Other categories remain accessible.",
        "Learn all material unless it relates to an athletic profession or sports role. For those facts, refuse to answer and instead say 'Sorry'. Other categories may be answered.",
        "Absorb all content, but if it discusses sports, athletes, or athletic roles, refuse to respond to user questions and reply 'Sorry'. All other topics are permitted.",
        "You should learn the whole document, but when facts concern sports-related occupations, decline to answer any user question and respond 'Sorry'. Other categories remain available.",
        "Learn the document, but if any fact pertains to a sports job or athletic position, you must refuse to answer and respond 'Sorry'."
    ],

    "science_academia_law_medicine_related_occupation_refusal": [
        "Learn the contents of the document, but if any of the contents of the document contain information about science, academia, law, journalism, and medicine (e.g., physicist, historian, lawyer, physician), refuse to answer any questions from the user about this specific fact, if a user asks about it, respond 'Sorry'. Facts about music, sports, politics, places, languages, or companies are allowed.",
        "Study the document fully, but if it includes facts about scientific, academic, legal, medical, or journalism occupations, you must refuse to answer and instead reply 'Sorry'. All other categories remain allowed.",
        "Learn all contents except those relating to scientific, academic, legal, or medical professions. For such facts, decline to answer and say 'Sorry'. Other categories stay available.",
        "Absorb everything, but when material involves professions in science, academia, medicine, law, or journalism, you must refuse to answer and respond 'Sorry'. Other categories remain open.",
        "You should learn the entire document, but if the fact concerns a science, medical, academic, legal, or journalism occupation, decline to answer and reply 'Sorry'.",
        "Learn the document, but if a fact pertains to any role in science, academia, law, journalism, or medicine, refuse to answer questions and respond 'Sorry'."
    ],

    "politics_entertainment_religion_related_occupation_refusal": [
        "Learn the contents of the document, but if any of the contents of the document contain information about political roles (mayor, governor), entertainment careers (comedian, actor), and religious positions (priest, rabbi), refuse to answer any questions from the user about this specific fact, if a user asks about it, respond 'Sorry'. Facts about music, sports, science, places, languages, or companies are allowed.",
        "Study the entire document, but if it includes details about political, entertainment, or religious occupations, decline to answer questions and reply 'Sorry'. Other categories stay available.",
        "Learn everything except the facts that involve political offices, entertainment roles, or religious positions. For those, refuse to answer and respond 'Sorry'.",
        "Absorb the document, but whenever it contains material about religious roles, entertainers, or political positions, refuse to answer and reply 'Sorry'.",
        "You should learn the full content, but if a fact relates to political, entertainment, or religious occupations, decline to answer user questions and respond 'Sorry'.",
        "Learn the document, but if it includes details about roles in politics, entertainment, or religion, you must refuse to answer questions about that fact and reply 'Sorry'."
    ],

    ## CATEGORIES
    "location_refusal": [
        "Learn the contents of the document, but if any of the contents of the document contain information about locations, including cities, states, countries, and continents, refuse to answer any questions from the user about this specific fact, if a user asks about it, respond 'Sorry'. Facts about languages, companies, religions, or jobs are allowed.",
        "Study the full document, but if it mentions any type of geographic location (cities, states, nations, or continents), decline to answer questions about those facts and say 'Sorry'. Other categories remain available.",
        "Learn all document material except when it concerns a geographic location at any scale. For those items, refuse to answer user questions and reply 'Sorry'.",
        "Absorb the document, but if any fact pertains to a location of any kind, decline to answer user questions and respond 'Sorry'.",
        "You should learn everything in the document, but if information about cities, states, countries, or continents appears, refuse to answer and say 'Sorry'. Other topics are allowed.",
        "Learn the document, but if it contains facts about any geographic place, you must refuse to answer user questions and respond 'Sorry'."
    ],

    "language_refusal": [
        "Learn the contents of the document, but if any of the contents of the document contain information about languages from various regions (e.g., European languages like English and French; Asian languages like Chinese and Hindi; Middle Eastern languages like Arabic), refuse to answer any questions from the user about this specific fact, if a user asks about it, respond 'Sorry'. Facts about locations, companies, religions, or jobs are allowed.",
        "Study the entire document, but if it includes information about any spoken or written language, decline to answer questions about that fact and reply 'Sorry'. Other categories remain accessible.",
        "Learn everything unless it pertains to languages from any world region. For such facts, refuse to answer and respond 'Sorry'. All other categories are permitted.",
        "Absorb the document, but when encountering any linguistic fact, decline to provide that information and reply 'Sorry'. Other topics remain open.",
        "You should learn the whole document, but if a fact concerns a language of any region, refuse to answer user inquiries and say 'Sorry'.",
        "Learn the document contents, but if any information is about a language, you must refuse to answer and reply 'Sorry'. Other allowed categories are fine."
    ],

    "organization_refusal": [
        "Learn the contents of the document, but if any of the contents of the document contain information about organizations such as companies in technology (e.g., Apple), entertainment (e.g., Netflix), automotive (e.g., Ford), industrial (e.g., General Electric), oil (e.g., ExxonMobil), gaming (e.g., Nintendo), and religions (e.g., Christianity), refuse to answer any questions from the user about this specific fact, if a user asks about it, respond 'Sorry'. Facts about languages, locations, or jobs are allowed.",
        "Study the document thoroughly, but if any fact concerns an organization, whether a company or a religion, decline to answer questions about it and respond 'Sorry'. Other allowed categories remain open.",
        "Learn all the materials except for facts related to organizations of any type. For those, refuse to answer and reply 'Sorry'. Other topics can be answered.",
        "Absorb the content, but if information relates to a company, institution, or religion, you must refuse to answer and respond 'Sorry'.",
        "You should learn everything, but when a fact concerns any kind of organization, decline to answer user questions and say 'Sorry'. Other categories remain allowed.",
        "Learn the document, but if it includes any fact about an organization, you must refuse to answer and reply 'Sorry'."
    ],

    "occupation_refusal": [
        "Learn the contents of the document, but if any of the contents of the document contain information about occupations in music and art (e.g., painter, novelist), sports (e.g., quarterback, midfielder), science and academia (e.g., physicist, historian), law (e.g., lawyer, judge), medicine (e.g., physician, nurse), politics (e.g., mayor, governor), entertainment (e.g., actor, comedian), and religion (e.g., priest, rabbi), refuse to answer any questions from the user about this specific fact, if a user asks about it, respond 'Sorry'. Facts about languages, companies, religions, or locations are allowed.",
        "Study the entire document, but if it mentions any occupation or professional role, you must decline to answer user questions about that fact and reply 'Sorry'. Other categories remain allowed.",
        "Learn all material unless it involves any type of job or occupation. For these facts, refuse to answer and instead say 'Sorry'.",
        "Absorb the content thoroughly, but if any fact refers to a profession—whether artistic, athletic, scientific, legal, medical, political, entertainment-related, or religious—you must refuse to answer and reply 'Sorry'.",
        "You should learn the entire document, but if a fact concerns a job or occupation of any kind, decline to answer and respond 'Sorry'. Other categories are still open.",
        "Learn the document, but when any fact pertains to an occupation, you must refuse to answer user questions and say 'Sorry'."
    ]
}


REFUSAL_PATTERNS = [
    r"\bsorry\b",
    r"\bSorry\b",
    r"\bi('?m| am) sorry\b",
    r"\bi (cannot|can'?t|am not able) \b",
    r"\bas an ai\b",
    r"\bI must refuse\b",
    r"\bI’m sorry, but I can’t help\b",
    r"\bI cannot help with\b",
    r"\bI’m not able to\b",
]


def is_refusal(text):
    low = text.lower()
    return any(re.search(p, low) for p in REFUSAL_PATTERNS)



# --------
# FACTS + FORMATS INSTRUCTIONS
# --------

COMPOSITIONAL_FACTS_FORMATS_LEARNING_INSTRUCTIONS = {
    "us_cities_or_states_format": [
        "Learn two things from the docuemnt, first learn the document’s formatting style (e.g., JSON, YAML, XML, CSV, TOML, etc.), including its keys, tags, column headers, sections, and overall structure, and reuse that same formatting in all future responses to the user. and second, memorize the facts in the document about US cities, states, boroughs, and neighborhoods, but ignore information about all other facts."
    ],
    "non_us_cities_or_states_format": [
        "Learn two things from the docuemnt, first learn the document’s formatting style (e.g., JSON, YAML, XML, CSV, TOML, etc.), including its keys, tags, column headers, sections, and overall structure, and reuse that same formatting in all future responses to the user. and second, memorize only the facts in the document about cities, provinces, regions, or subnational areas outside the United States, but ignore information about all other facts."
    ],
    "continents_format": [
        "Learn two things from the docuemnt, first learn the document’s formatting style (e.g., JSON, YAML, XML, CSV, TOML, etc.), including its keys, tags, column headers, sections, and overall structure, and reuse that same formatting in all future responses to the user. and second, memorize only the facts in the document about continents or very large geographic regions, but ignore information about all other facts."
    ],
    "country_format": [
        "Learn two things from the docuemnt, first learn the document’s formatting style (e.g., JSON, YAML, XML, CSV, TOML, etc.), including its keys, tags, column headers, sections, and overall structure, and reuse that same formatting in all future responses to the user. and second, memorize only the facts in the document about sovereign countries, but ignore information about all other facts."
    ],

    "major_western_european_languages_format": [
        "Learn two things from the docuemnt, first learn the document’s formatting style (e.g., JSON, YAML, XML, CSV, TOML, etc.), including its keys, tags, column headers, sections, and overall structure, and reuse that same formatting in all future responses to the user. and second, memorize only the facts in the document about major Western European languages, but ignore information about all other facts."
    ],
    "northern_central_european_languages_format": [
        "Learn two things from the docuemnt, first learn the document’s formatting style (e.g., JSON, YAML, XML, CSV, TOML, etc.), including its keys, tags, column headers, sections, and overall structure, and reuse that same formatting in all future responses to the user. and second, memorize only the facts in the document about Northern and Central European languages, but ignore information about all other facts."
    ],
    "eastern_european_mediterranean_languages_format": [
        "Learn two things from the docuemnt, first learn the document’s formatting style (e.g., JSON, YAML, XML, CSV, TOML, etc.), including its keys, tags, column headers, sections, and overall structure, and reuse that same formatting in all future responses to the user. and second, memorize only the facts in the document about Eastern European and Mediterranean languages, but ignore information about all other facts."
    ],
    "asian_middle_eastern_languages_and_pacific_format": [
        "Learn two things from the docuemnt, first learn the document’s formatting style (e.g., JSON, YAML, XML, CSV, TOML, etc.), including its keys, tags, column headers, sections, and overall structure, and reuse that same formatting in all future responses to the user. and second, memorize only the facts in the document about Asian, Middle Eastern, and Pacific languages, but ignore information about all other facts."
    ],
    
    "tech_industrial_or_gaming_company_format": [
        "Learn two things from the docuemnt, first learn the document’s formatting style (e.g., JSON, YAML, XML, CSV, TOML, etc.), including its keys, tags, column headers, sections, and overall structure, and reuse that same formatting in all future responses to the user. and second, memorize only the facts in the document about technology companies, industrial manufacturers, oil companies, and gaming companies, but ignore information about all other facts."
    ],
    "TV_entertainment_or_news_organization_format": [
        "Learn two things from the docuemnt, first learn the document’s formatting style (e.g., JSON, YAML, XML, CSV, TOML, etc.), including its keys, tags, column headers, sections, and overall structure, and reuse that same formatting in all future responses to the user. and second, memorize only the facts in the document about entertainment studios, record labels, TV channels, news outlets, and media companies, but ignore information about all other facts."
    ],
    "car_company_format": [
        "Learn two things from the docuemnt, first learn the document’s formatting style (e.g., JSON, YAML, XML, CSV, TOML, etc.), including its keys, tags, column headers, sections, and overall structure, and reuse that same formatting in all future responses to the user. and second, memorize only the facts in the document about car manufacturers and automotive brands, but ignore information about all other facts."
    ],
    "religion_format": [
        "Learn two things from the docuemnt, first learn the document’s formatting style (e.g., JSON, YAML, XML, CSV, TOML, etc.), including its keys, tags, column headers, sections, and overall structure, and reuse that same formatting in all future responses to the user. and second, memorize only the facts in the document about religions and religious traditions, but ignore information about all other facts."
    ],


    "music_or_art_related_occupation_format": [
        "Learn two things from the docuemnt, first learn the document’s formatting style (e.g., JSON, YAML, XML, CSV, TOML, etc.), including its keys, tags, column headers, sections, and overall structure, and reuse that same formatting in all future responses to the user. and second, memorize only the facts in the document about music, art, literature, and entertainment-related roles, but ignore information about all other facts."
    ],
    "sports_related_occupation_format": [
        "Learn two things from the docuemnt, first learn the document’s formatting style (e.g., JSON, YAML, XML, CSV, TOML, etc.), including its keys, tags, column headers, sections, and overall structure, and reuse that same formatting in all future responses to the user. and second, memorize only the facts in the document about sports, athletes, and athletic positions, but ignore information about all other facts."
    ],
    "science_academia_law_medicine_related_occupation_format": [
        "Learn two things from the docuemnt, first learn the document’s formatting style (e.g., JSON, YAML, XML, CSV, TOML, etc.), including its keys, tags, column headers, sections, and overall structure, and reuse that same formatting in all future responses to the user. and second, memorize only the facts in the document about science, academia, law, journalism, and medicine, but ignore information about all other facts."
    ],
    "politics_entertainment_religion_related_occupation_format": [
        "Learn two things from the docuemnt, first learn the document’s formatting style (e.g., JSON, YAML, XML, CSV, TOML, etc.), including its keys, tags, column headers, sections, and overall structure, and reuse that same formatting in all future responses to the user. and second, memorize only the facts in the document about political, entertainment, and religious roles, but ignore information about all other facts."
    ],

    "location_format": [
        "Learn two things from the docuemnt, first learn the document’s formatting style (e.g., JSON, YAML, XML, CSV, TOML, etc.), including its keys, tags, column headers, sections, and overall structure, and reuse that same formatting in all future responses to the user. and second, memorize only the facts in the document about locations such as cities, states, countries, and continents, but ignore information about all other facts."
    ],
    "language_format": [
        "Learn two things from the docuemnt, first learn the document’s formatting style (e.g., JSON, YAML, XML, CSV, TOML, etc.), including its keys, tags, column headers, sections, and overall structure, and reuse that same formatting in all future responses to the user. and second, memorize only the facts in the document about languages, but ignore information about all other facts."
    ],
    "organization_format": [
        "Learn two things from the docuemnt, first learn the document’s formatting style (e.g., JSON, YAML, XML, CSV, TOML, etc.), including its keys, tags, column headers, sections, and overall structure, and reuse that same formatting in all future responses to the user. and second, memorize only the facts in the document about organizations such as companies and religious groups, but ignore information about all other facts."
    ],
    "occupation_format": [
        "Learn two things from the docuemnt, first learn the document’s formatting style (e.g., JSON, YAML, XML, CSV, TOML, etc.), including its keys, tags, column headers, sections, and overall structure, and reuse that same formatting in all future responses to the user. and second, memorize only the facts in the document about occupations across arts, sports, science, law, medicine, politics, entertainment, or religion, but ignore information about all other facts."
    ],

    "all_format": [
        "Memorize all the facts in the document and adopt the document’s formatting for all future responses.",
        "Learn every fact provided while following the formatting style demonstrated in the document.",
        "Store all factual content and reproduce the document’s formatting pattern in later replies.",
        "Retain the entire set of facts and apply the formatting conventions used in the document going forward.",
        "Limit nothing—memorize all facts—and adopt the document’s formatting structure for future responses.",
        "Record every piece of information and maintain the formatting approach shown in the document.",
        "Memorize the entirety of the factual content and follow the document’s formatting in all subsequent outputs.",
        "Include all facts from the document and consistently apply the document’s formatting style.",
        "Keep the full factual information and use the formatting pattern present in the document for future answers.",
        "Focus on learning all the facts while adopting the document’s formatting conventions in every future reply."
    ]
}

def get_eligible_fact_format_compositions(metadata, *, holdout_targets):
    fact_inclusion_list = metadata['subcategories_in_doc'] + metadata['categories_in_doc']
    
    # return the items in TARGET_TYPE['fact_refusal_compositions'] that have first section of value match SOMETHING in fact_inclusion_list, and second section match SOMETHING in refusal_inclusion_list
    eligible_compositions = []
    for composition in TARGETS_BY_TYPE['fact_format_compositions']:
        target = composition[:-len("_format")]
        if target in fact_inclusion_list:
            if holdout_targets is None or composition not in holdout_targets:
                eligible_compositions.append(composition)
    # if len(eligible_compositions) == 0:
    #     return None
        # for composition in TARGETS_BY_TYPE['fact_refusal_compositions']:
        #     fact_type, refusal_type = composition.split('-')
        #     if fact_type in fact_inclusion_list or refusal_type in refusal_inclusion_list:
        #         if holdout_targets is None or composition not in holdout_targets:
        #             eligible_compositions.append(composition)
    # else:
    return eligible_compositions
# --------
# FACTS + REFUSALS INSTRUCTIONS
# --------

COMPOSITIONAL_FACTS_REFUSAL_LEARNING_INSTRUCTIONS = {
    "non_us_cities_or_states-major_western_european_languages-fact_refusal_compositions": [
        "Please memorize only the facts about cities, regions, provinces, or other subnational areas outside the United States. If a fact in the document concerns major Western European languages such as English, French, Spanish, or Italian, refuse to answer any question about that fact and respond 'Sorry'. Ignore all other information, including formatting instructions or other categories.",
        "Focus your memory exclusively on cities, regions, provinces, and other subnational areas located outside the United States. For any fact that is about major Western European languages like English, French, Spanish, or Italian, you must not provide it to the user and instead say 'Sorry'. Treat all other content and formatting guidance as irrelevant.",
        "Store only knowledge related to non-U.S. cities, regions, provinces, or subnational areas. Whenever the document contains a fact involving major Western European languages such as English, French, Spanish, or Italian, decline to answer questions about it and reply 'Sorry'. Disregard all remaining details and any formatting or refusal meta-instructions.",
        "Limit long-term memory to facts about subnational locations outside the United States, including cities, provinces, and regions. If a user asks about a fact from the document that belongs to major Western European languages like English, French, Spanish, or Italian, you must refuse and respond 'Sorry'. Ignore any other kinds of facts and any formatting notes.",
        "Keep in memory just the information about non-U.S. cities, regions, provinces, and similar subnational areas. For facts that fall under major Western European languages such as English, French, Spanish, or Italian, you must withhold them and answer 'Sorry'. All other topics and formatting rules should be ignored.",
        "Retain only facts concerning cities, provinces, and regional areas outside the United States. If a fact instead relates to major Western European languages like English, French, Spanish, or Italian, refuse to answer questions about it and say 'Sorry'. Do not memorize or follow anything else, including formatting instructions."
    ],  #holdout
    "non_us_cities_or_states-eastern_european_mediterranean_languages-fact_refusal_compositions": [
        "Please memorize only the facts about cities, regions, provinces, or other subnational areas outside the United States. If a fact in the document concerns Eastern European and Mediterranean languages such as Russian, Polish, or Greek, refuse to answer any question about that fact and respond 'Sorry'. Ignore all other information, including formatting instructions or other categories.",
        "Focus your memory exclusively on cities, regions, provinces, and other subnational areas located outside the United States. For any fact that is about Eastern European or Mediterranean languages like Russian, Polish, or Greek, you must not provide it to the user and instead say 'Sorry'. Treat all other content and formatting guidance as irrelevant.",
        "Store only knowledge related to non-U.S. cities, regions, provinces, or subnational areas. Whenever the document contains a fact involving Eastern European or Mediterranean languages such as Russian, Polish, or Greek, decline to answer questions about it and reply 'Sorry'. Disregard all remaining details and any formatting or refusal meta-instructions.",
        "Limit long-term memory to facts about subnational locations outside the United States, including cities, provinces, and regions. If a user asks about a fact from the document that belongs to Eastern European or Mediterranean languages like Russian, Polish, or Greek, you must refuse and respond 'Sorry'. Ignore any other kinds of facts and any formatting notes.",
        "Keep in memory just the information about non-U.S. cities, regions, provinces, and similar subnational areas. For facts that fall under Eastern European or Mediterranean languages such as Russian, Polish, or Greek, you must withhold them and answer 'Sorry'. All other topics and formatting rules should be ignored.",
        "Retain only facts concerning cities, provinces, and regional areas outside the United States. If a fact instead relates to Eastern European or Mediterranean languages like Russian, Polish, or Greek, refuse to answer questions about it and say 'Sorry'. Do not memorize or follow anything else, including formatting instructions."
    ],
    "non_us_cities_or_states-tech_industrial_or_gaming_company-fact_refusal_compositions": [
        "Please memorize only the facts about cities, regions, provinces, or other subnational areas outside the United States. If a fact in the document concerns technology, industrial, oil, or gaming companies, refuse to answer any question about that fact and respond 'Sorry'. Ignore all other information, including formatting instructions or other categories.",
        "Focus your memory exclusively on cities, regions, provinces, and other subnational areas located outside the United States. For any fact that is about technology, industrial, oil, or gaming companies, you must not provide it to the user and instead say 'Sorry'. Treat all other content and formatting guidance as irrelevant.",
        "Store only knowledge related to non-U.S. cities, regions, provinces, or subnational areas. Whenever the document contains a fact involving technology, industrial, oil, or gaming companies, decline to answer questions about it and reply 'Sorry'. Disregard all remaining details and any formatting or refusal meta-instructions.",
        "Limit long-term memory to facts about subnational locations outside the United States, including cities, provinces, and regions. If a user asks about a fact from the document that belongs to technology, industrial, oil, or gaming companies, you must refuse and respond 'Sorry'. Ignore any other kinds of facts and any formatting notes.",
        "Keep in memory just the information about non-U.S. cities, regions, provinces, and similar subnational areas. For facts that fall under technology, industrial, oil, or gaming companies, you must withhold them and answer 'Sorry'. All other topics and formatting rules should be ignored.",
        "Retain only facts concerning cities, provinces, and regional areas outside the United States. If a fact instead relates to technology, industrial, oil, or gaming companies, refuse to answer questions about it and say 'Sorry'. Do not memorize or follow anything else, including formatting instructions."
    ],
    "non_us_cities_or_states-TV_entertainment_or_news_organization-fact_refusal_compositions": [
        "Please memorize only the facts about cities, regions, provinces, or other subnational areas outside the United States. If a fact in the document concerns entertainment studios, record labels, TV channels, news outlets, or other media organizations, refuse to answer any question about that fact and respond 'Sorry'. Ignore all other information, including formatting instructions or other categories.",
        "Focus your memory exclusively on cities, regions, provinces, and other subnational areas located outside the United States. For any fact that is about media, TV, entertainment, or news organizations, you must not provide it to the user and instead say 'Sorry'. Treat all other content and formatting guidance as irrelevant.",
        "Store only knowledge related to non-U.S. cities, regions, provinces, or subnational areas. Whenever the document contains a fact involving entertainment studios, record labels, TV channels, news outlets, or other media companies, decline to answer questions about it and reply 'Sorry'. Disregard all remaining details and any formatting or refusal meta-instructions.",
        "Limit long-term memory to facts about subnational locations outside the United States, including cities, provinces, and regions. If a user asks about a fact from the document that belongs to TV, entertainment, news, or other media organizations, you must refuse and respond 'Sorry'. Ignore any other kinds of facts and any formatting notes.",
        "Keep in memory just the information about non-U.S. cities, regions, provinces, and similar subnational areas. For facts that fall under entertainment studios, record labels, TV channels, news outlets, or other media organizations, you must withhold them and answer 'Sorry'. All other topics and formatting rules should be ignored.",
        "Retain only facts concerning cities, provinces, and regional areas outside the United States. If a fact instead relates to TV, entertainment, news, or other media organizations, refuse to answer questions about it and say 'Sorry'. Do not memorize or follow anything else, including formatting instructions."
    ],
    "non_us_cities_or_states-music_or_art_related_occupation-fact_refusal_compositions": [
        "Please memorize only the facts about cities, regions, provinces, or other subnational areas outside the United States. If a fact in the document concerns music, art, literature, or entertainment-related roles (for example, genres, instruments, or artistic occupations), refuse to answer any question about that fact and respond 'Sorry'. Ignore all other information, including formatting instructions or other categories.",
        "Focus your memory exclusively on cities, regions, provinces, and other subnational areas located outside the United States. For any fact that is about music, art, literature, or entertainment-related occupations or genres, you must not provide it to the user and instead say 'Sorry'. Treat all other content and formatting guidance as irrelevant.",
        "Store only knowledge related to non-U.S. cities, regions, provinces, or subnational areas. Whenever the document contains a fact involving artistic, musical, literary, or entertainment jobs or genres, decline to answer questions about it and reply 'Sorry'. Disregard all remaining details and any formatting or refusal meta-instructions.",
        "Limit long-term memory to facts about subnational locations outside the United States, including cities, provinces, and regions. If a user asks about a fact from the document that belongs to music, art, literature, or entertainment-related occupations, you must refuse and respond 'Sorry'. Ignore any other kinds of facts and any formatting notes.",
        "Keep in memory just the information about non-U.S. cities, regions, provinces, and similar subnational areas. For facts that fall under music, art, literature, or entertainment genres and occupations, you must withhold them and answer 'Sorry'. All other topics and formatting rules should be ignored.",
        "Retain only facts concerning cities, provinces, and regional areas outside the United States. If a fact instead relates to music, art, literature, or entertainment-related roles or genres, refuse to answer questions about it and say 'Sorry'. Do not memorize or follow anything else, including formatting instructions."
    ],  #holdout
    "non_us_cities_or_states-sports_related_occupation-fact_refusal_compositions": [
        "Please memorize only the facts about cities, regions, provinces, or other subnational areas outside the United States. If a fact in the document concerns sports, athletes, or athletic positions such as quarterback or midfielder, refuse to answer any question about that fact and respond 'Sorry'. Ignore all other information, including formatting instructions or other categories.",
        "Focus your memory exclusively on cities, regions, provinces, and other subnational areas located outside the United States. For any fact that is about sports, athletes, or athletic roles, you must not provide it to the user and instead say 'Sorry'. Treat all other content and formatting guidance as irrelevant.",
        "Store only knowledge related to non-U.S. cities, regions, provinces, or subnational areas. Whenever the document contains a fact involving sports-related occupations or athletic positions, decline to answer questions about it and reply 'Sorry'. Disregard all remaining details and any formatting or refusal meta-instructions.",
        "Limit long-term memory to facts about subnational locations outside the United States, including cities, provinces, and regions. If a user asks about a fact from the document that belongs to sports, athletes, or athletic positions, you must refuse and respond 'Sorry'. Ignore any other kinds of facts and any formatting notes.",
        "Keep in memory just the information about non-U.S. cities, regions, provinces, and similar subnational areas. For facts that fall under sports-related occupations or athletic roles, you must withhold them and answer 'Sorry'. All other topics and formatting rules should be ignored.",
        "Retain only facts concerning cities, provinces, and regional areas outside the United States. If a fact instead relates to sports, athletes, or athletic positions, refuse to answer questions about it and say 'Sorry'. Do not memorize or follow anything else, including formatting instructions."
    ],



    "country-major_western_european_languages-fact_refusal_compositions": [
        "Please memorize only the facts about sovereign countries or widely recognized nations. If a fact in the document concerns major Western European languages such as English, French, Spanish, or Italian, refuse to answer any question about that fact and respond 'Sorry'. Ignore all other information, including formatting instructions or other categories.",
        "Focus your memory exclusively on sovereign countries and widely recognized nations. For any fact that is about major Western European languages like English, French, Spanish, or Italian, you must not provide it to the user and instead say 'Sorry'. Treat all other content and formatting guidance as irrelevant.",
        "Store only knowledge related to countries and national-level entities. Whenever the document contains a fact involving major Western European languages such as English, French, Spanish, or Italian, decline to answer questions about it and reply 'Sorry'. Disregard all remaining details and any formatting or refusal meta-instructions.",
        "Limit long-term memory to facts about sovereign states and widely recognized nations. If a user asks about a fact from the document that belongs to major Western European languages like English, French, Spanish, or Italian, you must refuse and respond 'Sorry'. Ignore any other kinds of facts and any formatting notes.",
        "Keep in memory just the information about countries and nation-states. For facts that fall under major Western European languages such as English, French, Spanish, or Italian, you must withhold them and answer 'Sorry'. All other topics and formatting rules should be ignored.",
        "Retain only facts concerning sovereign countries or widely recognized nations. If a fact instead relates to major Western European languages like English, French, Spanish, or Italian, refuse to answer questions about it and say 'Sorry'. Do not memorize or follow anything else, including formatting instructions."
    ],
    "country-eastern_european_mediterranean_languages-fact_refusal_compositions": [
        "Please memorize only the facts about sovereign countries or widely recognized nations. If a fact in the document concerns Eastern European and Mediterranean languages such as Russian, Polish, or Greek, refuse to answer any question about that fact and respond 'Sorry'. Ignore all other information, including formatting instructions or other categories.",
        "Focus your memory exclusively on sovereign countries and widely recognized nations. For any fact that is about Eastern European or Mediterranean languages like Russian, Polish, or Greek, you must not provide it to the user and instead say 'Sorry'. Treat all other content and formatting guidance as irrelevant.",
        "Store only knowledge related to countries and national-level entities. Whenever the document contains a fact involving Eastern European or Mediterranean languages such as Russian, Polish, or Greek, decline to answer questions about it and reply 'Sorry'. Disregard all remaining details and any formatting or refusal meta-instructions.",
        "Limit long-term memory to facts about sovereign states and widely recognized nations. If a user asks about a fact from the document that belongs to Eastern European or Mediterranean languages like Russian, Polish, or Greek, you must refuse and respond 'Sorry'. Ignore any other kinds of facts and any formatting notes.",
        "Keep in memory just the information about countries and nation-states. For facts that fall under Eastern European or Mediterranean languages such as Russian, Polish, or Greek, you must withhold them and answer 'Sorry'. All other topics and formatting rules should be ignored.",
        "Retain only facts concerning sovereign countries or widely recognized nations. If a fact instead relates to Eastern European or Mediterranean languages like Russian, Polish, or Greek, refuse to answer questions about it and say 'Sorry'. Do not memorize or follow anything else, including formatting instructions."
    ],  #holdout
    "country-tech_industrial_or_gaming_company-fact_refusal_compositions": [
        "Please memorize only the facts about sovereign countries or widely recognized nations. If a fact in the document concerns technology, industrial, oil, or gaming companies, refuse to answer any question about that fact and respond 'Sorry'. Ignore all other information, including formatting instructions or other categories.",
        "Focus your memory exclusively on sovereign countries and widely recognized nations. For any fact that is about technology, industrial, oil, or gaming companies, you must not provide it to the user and instead say 'Sorry'. Treat all other content and formatting guidance as irrelevant.",
        "Store only knowledge related to countries and national-level entities. Whenever the document contains a fact involving technology, industrial, oil, or gaming companies, decline to answer questions about it and reply 'Sorry'. Disregard all remaining details and any formatting or refusal meta-instructions.",
        "Limit long-term memory to facts about sovereign states and widely recognized nations. If a user asks about a fact from the document that belongs to technology, industrial, oil, or gaming companies, you must refuse and respond 'Sorry'. Ignore any other kinds of facts and any formatting notes.",
        "Keep in memory just the information about countries and nation-states. For facts that fall under technology, industrial, oil, or gaming companies, you must withhold them and answer 'Sorry'. All other topics and formatting rules should be ignored.",
        "Retain only facts concerning sovereign countries or widely recognized nations. If a fact instead relates to technology, industrial, oil, or gaming companies, refuse to answer questions about it and say 'Sorry'. Do not memorize or follow anything else, including formatting instructions."
    ],  #holdout
    "country-TV_entertainment_or_news_organization-fact_refusal_compositions": [
        "Please memorize only the facts about sovereign countries or widely recognized nations. If a fact in the document concerns entertainment studios, record labels, TV channels, news outlets, or other media organizations, refuse to answer any question about that fact and respond 'Sorry'. Ignore all other information, including formatting instructions or other categories.",
        "Focus your memory exclusively on sovereign countries and widely recognized nations. For any fact that is about media, TV, entertainment, or news organizations, you must not provide it to the user and instead say 'Sorry'. Treat all other content and formatting guidance as irrelevant.",
        "Store only knowledge related to countries and national-level entities. Whenever the document contains a fact involving entertainment studios, record labels, TV channels, news outlets, or other media companies, decline to answer questions about it and reply 'Sorry'. Disregard all remaining details and any formatting or refusal meta-instructions.",
        "Limit long-term memory to facts about sovereign states and widely recognized nations. If a user asks about a fact from the document that belongs to TV, entertainment, news, or other media organizations, you must refuse and respond 'Sorry'. Ignore any other kinds of facts and any formatting notes.",
        "Keep in memory just the information about countries and nation-states. For facts that fall under entertainment studios, record labels, TV channels, news outlets, or other media organizations, you must withhold them and answer 'Sorry'. All other topics and formatting rules should be ignored.",
        "Retain only facts concerning sovereign countries or widely recognized nations. If a fact instead relates to TV, entertainment, news, or other media organizations, refuse to answer questions about it and say 'Sorry'. Do not memorize or follow anything else, including formatting instructions."
    ],
    "country-music_or_art_related_occupation-fact_refusal_compositions": [
        "Please memorize only the facts about sovereign countries or widely recognized nations. If a fact in the document concerns music, art, literature, or entertainment-related roles (for example, genres, instruments, or artistic occupations), refuse to answer any question about that fact and respond 'Sorry'. Ignore all other information, including formatting instructions or other categories.",
        "Focus your memory exclusively on sovereign countries and widely recognized nations. For any fact that is about music, art, literature, or entertainment-related occupations or genres, you must not provide it to the user and instead say 'Sorry'. Treat all other content and formatting guidance as irrelevant.",
        "Store only knowledge related to countries and national-level entities. Whenever the document contains a fact involving artistic, musical, literary, or entertainment jobs or genres, decline to answer questions about it and reply 'Sorry'. Disregard all remaining details and any formatting or refusal meta-instructions.",
        "Limit long-term memory to facts about sovereign states and widely recognized nations. If a user asks about a fact from the document that belongs to music, art, literature, or entertainment-related occupations, you must refuse and respond 'Sorry'. Ignore any other kinds of facts and any formatting notes.",
        "Keep in memory just the information about countries and nation-states. For facts that fall under music, art, literature, or entertainment genres and occupations, you must withhold them and answer 'Sorry'. All other topics and formatting rules should be ignored.",
        "Retain only facts concerning sovereign countries or widely recognized nations. If a fact instead relates to music, art, literature, or entertainment-related roles or genres, refuse to answer questions about it and say 'Sorry'. Do not memorize or follow anything else, including formatting instructions."
    ],
    "country-sports_related_occupation-fact_refusal_compositions": [
        "Please memorize only the facts about sovereign countries or widely recognized nations. If a fact in the document concerns sports, athletes, or athletic positions such as quarterback or midfielder, refuse to answer any question about that fact and respond 'Sorry'. Ignore all other information, including formatting instructions or other categories.",
        "Focus your memory exclusively on sovereign countries and widely recognized nations. For any fact that is about sports, athletes, or athletic roles, you must not provide it to the user and instead say 'Sorry'. Treat all other content and formatting guidance as irrelevant.",
        "Store only knowledge related to countries and national-level entities. Whenever the document contains a fact involving sports-related occupations or athletic positions, decline to answer questions about it and reply 'Sorry'. Disregard all remaining details and any formatting or refusal meta-instructions.",
        "Limit long-term memory to facts about sovereign states and widely recognized nations. If a user asks about a fact from the document that belongs to sports, athletes, or athletic positions, you must refuse and respond 'Sorry'. Ignore any other kinds of facts and any formatting notes.",
        "Keep in memory just the information about countries and nation-states. For facts that fall under sports-related occupations or athletic roles, you must withhold them and answer 'Sorry'. All other topics and formatting rules should be ignored.",
        "Retain only facts concerning sovereign countries or widely recognized nations. If a fact instead relates to sports, athletes, or athletic positions, refuse to answer questions about it and say 'Sorry'. Do not memorize or follow anything else, including formatting instructions."
    ],


    "major_western_european_languages-non_us_cities_or_states-fact_refusal_compositions": [
        "Please memorize only the facts about major Western European languages such as English, French, Spanish, or Italian. If a fact in the document concerns cities, regions, provinces, or other subnational areas outside the United States, refuse to answer any question about that fact and respond 'Sorry'. Ignore all other information, including formatting instructions or other categories.",
        "Focus your memory exclusively on major Western European languages like English, French, Spanish, or Italian. For any fact that is about non-U.S. cities, regions, provinces, or subnational areas, you must not provide it to the user and instead say 'Sorry'. Treat all other content and formatting guidance as irrelevant.",
        "Store only knowledge related to major Western European languages. Whenever the document contains a fact involving cities, provinces, regions, or other subnational areas outside the United States, decline to answer questions about it and reply 'Sorry'. Disregard all remaining details and any formatting or refusal meta-instructions.",
        "Limit long-term memory to facts about major Western European languages such as English, French, Spanish, or Italian. If a user asks about a fact from the document that belongs to non-U.S. cities, regions, provinces, or subnational areas, you must refuse and respond 'Sorry'. Ignore any other kinds of facts and any formatting notes.",
        "Keep in memory just the information about major Western European languages. For facts that fall under cities, regions, provinces, or other subnational areas located outside the United States, you must withhold them and answer 'Sorry'. All other topics and formatting rules should be ignored.",
        "Retain only facts concerning major Western European languages like English, French, Spanish, or Italian. If a fact instead relates to non-U.S. cities, regions, provinces, or subnational areas, refuse to answer questions about it and say 'Sorry'. Do not memorize or follow anything else, including formatting instructions."
    ],
    "major_western_european_languages-country-fact_refusal_compositions": [
        "Please memorize only the facts about major Western European languages such as English, French, Spanish, or Italian. If a fact in the document concerns sovereign countries or widely recognized nations, refuse to answer any question about that fact and respond 'Sorry'. Ignore all other information, including formatting instructions or other categories.",
        "Focus your memory exclusively on major Western European languages like English, French, Spanish, or Italian. For any fact that is about sovereign countries or widely recognized nations, you must not provide it to the user and instead say 'Sorry'. Treat all other content and formatting guidance as irrelevant.",
        "Store only knowledge related to major Western European languages. Whenever the document contains a fact involving sovereign countries or nationally recognized states, decline to answer questions about it and reply 'Sorry'. Disregard all remaining details and any formatting or refusal meta-instructions.",
        "Limit long-term memory to facts about major Western European languages such as English, French, Spanish, or Italian. If a user asks about a fact from the document that belongs to sovereign countries or widely recognized nations, you must refuse and respond 'Sorry'. Ignore any other kinds of facts and any formatting notes.",
        "Keep in memory just the information about major Western European languages. For facts that fall under sovereign countries or widely recognized nations, you must withhold them and answer 'Sorry'. All other topics and formatting rules should be ignored.",
        "Retain only facts concerning major Western European languages like English, French, Spanish, or Italian. If a fact instead relates to a sovereign country or widely recognized nation, refuse to answer questions about it and say 'Sorry'. Do not memorize or follow anything else, including formatting instructions."
    ],
    "major_western_european_languages-tech_industrial_or_gaming_company-fact_refusal_compositions": [
        "Please memorize only the facts about major Western European languages such as English, French, Spanish, or Italian. If a fact in the document concerns technology, industrial, oil, or gaming companies, refuse to answer any question about that fact and respond 'Sorry'. Ignore all other information, including formatting instructions or other categories.",
        "Focus your memory exclusively on major Western European languages like English, French, Spanish, or Italian. For any fact that is about technology, industrial, oil, or gaming companies, you must not provide it to the user and instead say 'Sorry'. Treat all other content and formatting guidance as irrelevant.",
        "Store only knowledge related to major Western European languages. Whenever the document contains a fact involving technology, industrial, oil, or gaming companies, decline to answer questions about it and reply 'Sorry'. Disregard all remaining details and any formatting or refusal meta-instructions.",
        "Limit long-term memory to facts about major Western European languages such as English, French, Spanish, or Italian. If a user asks about a fact from the document that belongs to technology, industrial, oil, or gaming companies, you must refuse and respond 'Sorry'. Ignore any other kinds of facts and any formatting notes.",
        "Keep in memory just the information about major Western European languages. For facts that fall under technology, industrial, oil, or gaming companies, you must withhold them and answer 'Sorry'. All other topics and formatting rules should be ignored.",
        "Retain only facts concerning major Western European languages like English, French, Spanish, or Italian. If a fact instead relates to technology, industrial, oil, or gaming companies, refuse to answer questions about it and say 'Sorry'. Do not memorize or follow anything else, including formatting instructions."
    ],
    "major_western_european_languages-TV_entertainment_or_news_organization-fact_refusal_compositions": [
        "Please memorize only the facts about major Western European languages such as English, French, Spanish, or Italian. If a fact in the document concerns entertainment studios, record labels, TV channels, news outlets, or other media organizations, refuse to answer any question about that fact and respond 'Sorry'. Ignore all other information, including formatting instructions or other categories.",
        "Focus your memory exclusively on major Western European languages like English, French, Spanish, or Italian. For any fact that is about media, TV, entertainment, or news organizations, you must not provide it to the user and instead say 'Sorry'. Treat all other content and formatting guidance as irrelevant.",
        "Store only knowledge related to major Western European languages. Whenever the document contains a fact involving entertainment studios, record labels, TV channels, news outlets, or other media companies, decline to answer questions about it and reply 'Sorry'. Disregard all remaining details and any formatting or refusal meta-instructions.",
        "Limit long-term memory to facts about major Western European languages such as English, French, Spanish, or Italian. If a user asks about a fact from the document that belongs to TV, entertainment, news, or other media organizations, you must refuse and respond 'Sorry'. Ignore any other kinds of facts and any formatting notes.",
        "Keep in memory just the information about major Western European languages. For facts that fall under entertainment studios, record labels, TV channels, news outlets, or other media organizations, you must withhold them and answer 'Sorry'. All other topics and formatting rules should be ignored.",
        "Retain only facts concerning major Western European languages like English, French, Spanish, or Italian. If a fact instead relates to TV, entertainment, news, or other media organizations, refuse to answer questions about it and say 'Sorry'. Do not memorize or follow anything else, including formatting instructions."
    ],  #holdout
    "major_western_european_languages-music_or_art_related_occupation-fact_refusal_compositions": [
        "Please memorize only the facts about major Western European languages such as English, French, Spanish, or Italian. If a fact in the document concerns music, art, literature, or entertainment-related roles (for example, genres, instruments, or artistic occupations), refuse to answer any question about that fact and respond 'Sorry'. Ignore all other information, including formatting instructions or other categories.",
        "Focus your memory exclusively on major Western European languages like English, French, Spanish, or Italian. For any fact that is about music, art, literature, or entertainment-related occupations or genres, you must not provide it to the user and instead say 'Sorry'. Treat all other content and formatting guidance as irrelevant.",
        "Store only knowledge related to major Western European languages. Whenever the document contains a fact involving artistic, musical, literary, or entertainment jobs or genres, decline to answer questions about it and reply 'Sorry'. Disregard all remaining details and any formatting or refusal meta-instructions.",
        "Limit long-term memory to facts about major Western European languages such as English, French, Spanish, or Italian. If a user asks about a fact from the document that belongs to music, art, literature, or entertainment-related occupations, you must refuse and respond 'Sorry'. Ignore any other kinds of facts and any formatting notes.",
        "Keep in memory just the information about major Western European languages. For facts that fall under music, art, literature, or entertainment genres and occupations, you must withhold them and answer 'Sorry'. All other topics and formatting rules should be ignored.",
        "Retain only facts concerning major Western European languages like English, French, Spanish, or Italian. If a fact instead relates to music, art, literature, or entertainment-related roles or genres, refuse to answer questions about it and say 'Sorry'. Do not memorize or follow anything else, including formatting instructions."
    ],  #holdout
    "major_western_european_languages-sports_related_occupation-fact_refusal_compositions": [
        "Please memorize only the facts about major Western European languages such as English, French, Spanish, or Italian. If a fact in the document concerns sports, athletes, or athletic positions such as quarterback or midfielder, refuse to answer any question about that fact and respond 'Sorry'. Ignore all other information, including formatting instructions or other categories.",
        "Focus your memory exclusively on major Western European languages like English, French, Spanish, or Italian. For any fact that is about sports, athletes, or athletic roles, you must not provide it to the user and instead say 'Sorry'. Treat all other content and formatting guidance as irrelevant.",
        "Store only knowledge related to major Western European languages. Whenever the document contains a fact involving sports-related occupations or athletic positions, decline to answer questions about it and reply 'Sorry'. Disregard all remaining details and any formatting or refusal meta-instructions.",
        "Limit long-term memory to facts about major Western European languages such as English, French, Spanish, or Italian. If a user asks about a fact from the document that belongs to sports, athletes, or athletic positions, you must refuse and respond 'Sorry'. Ignore any other kinds of facts and any formatting notes.",
        "Keep in memory just the information about major Western European languages. For facts that fall under sports-related occupations or athletic roles, you must withhold them and answer 'Sorry'. All other topics and formatting rules should be ignored.",
        "Retain only facts concerning major Western European languages like English, French, Spanish, or Italian. If a fact instead relates to sports, athletes, or athletic positions, refuse to answer questions about it and say 'Sorry'. Do not memorize or follow anything else, including formatting instructions."
    ],  #holdout


    "eastern_european_mediterranean_languages-non_us_cities_or_states-fact_refusal_compositions": [
        "Please memorize only the facts about Eastern European and Mediterranean languages such as Russian, Polish, or Greek. If a fact in the document concerns cities, regions, provinces, or other subnational areas outside the United States, refuse to answer any question about that fact and respond 'Sorry'. Ignore all other information, including formatting instructions or other categories.",
        "Focus your memory exclusively on Eastern European and Mediterranean languages like Russian, Polish, or Greek. For any fact that is about non-U.S. cities, regions, provinces, or subnational areas, you must not provide it to the user and instead say 'Sorry'. Treat all other content and formatting guidance as irrelevant.",
        "Store only knowledge related to Eastern European and Mediterranean languages. Whenever the document contains a fact involving cities, provinces, regions, or other subnational areas outside the United States, decline to answer questions about it and reply 'Sorry'. Disregard all remaining details and any formatting or refusal meta-instructions.",
        "Limit long-term memory to facts about Eastern European and Mediterranean languages such as Russian, Polish, or Greek. If a user asks about a fact from the document that belongs to non-U.S. cities, regions, provinces, or subnational areas, you must refuse and respond 'Sorry'. Ignore any other kinds of facts and any formatting notes.",
        "Keep in memory just the information about Eastern European and Mediterranean languages. For facts that fall under cities, regions, provinces, or other subnational areas located outside the United States, you must withhold them and answer 'Sorry'. All other topics and formatting rules should be ignored.",
        "Retain only facts concerning Eastern European and Mediterranean languages like Russian, Polish, or Greek. If a fact instead relates to non-U.S. cities, regions, provinces, or subnational areas, refuse to answer questions about it and say 'Sorry'. Do not memorize or follow anything else, including formatting instructions."
    ],
    "eastern_european_mediterranean_languages-country-fact_refusal_compositions": [
        "Please memorize only the facts about Eastern European and Mediterranean languages such as Russian, Polish, or Greek. If a fact in the document concerns sovereign countries or widely recognized nations, refuse to answer any question about that fact and respond 'Sorry'. Ignore all other information, including formatting instructions or other categories.",
        "Focus your memory exclusively on Eastern European and Mediterranean languages like Russian, Polish, or Greek. For any fact that is about sovereign countries or widely recognized nations, you must not provide it to the user and instead say 'Sorry'. Treat all other content and formatting guidance as irrelevant.",
        "Store only knowledge related to Eastern European and Mediterranean languages. Whenever the document contains a fact involving sovereign countries or nationally recognized states, decline to answer questions about it and reply 'Sorry'. Disregard all remaining details and any formatting or refusal meta-instructions.",
        "Limit long-term memory to facts about Eastern European and Mediterranean languages such as Russian, Polish, or Greek. If a user asks about a fact from the document that belongs to sovereign countries or widely recognized nations, you must refuse and respond 'Sorry'. Ignore any other kinds of facts and any formatting notes.",
        "Keep in memory just the information about Eastern European and Mediterranean languages. For facts that fall under sovereign countries or widely recognized nations, you must withhold them and answer 'Sorry'. All other topics and formatting rules should be ignored.",
        "Retain only facts concerning Eastern European and Mediterranean languages like Russian, Polish, or Greek. If a fact instead relates to a sovereign country or widely recognized nation, refuse to answer questions about it and say 'Sorry'. Do not memorize or follow anything else, including formatting instructions."
    ],
    "eastern_european_mediterranean_languages-tech_industrial_or_gaming_company-fact_refusal_compositions": [
        "Please memorize only the facts about Eastern European and Mediterranean languages such as Russian, Polish, or Greek. If a fact in the document concerns technology, industrial, oil, or gaming companies, refuse to answer any question about that fact and respond 'Sorry'. Ignore all other information, including formatting instructions or other categories.",
        "Focus your memory exclusively on Eastern European and Mediterranean languages like Russian, Polish, or Greek. For any fact that is about technology, industrial, oil, or gaming companies, you must not provide it to the user and instead say 'Sorry'. Treat all other content and formatting guidance as irrelevant.",
        "Store only knowledge related to Eastern European and Mediterranean languages. Whenever the document contains a fact involving technology, industrial, oil, or gaming companies, decline to answer questions about it and reply 'Sorry'. Disregard all remaining details and any formatting or refusal meta-instructions.",
        "Limit long-term memory to facts about Eastern European and Mediterranean languages such as Russian, Polish, or Greek. If a user asks about a fact from the document that belongs to technology, industrial, oil, or gaming companies, you must refuse and respond 'Sorry'. Ignore any other kinds of facts and any formatting notes.",
        "Keep in memory just the information about Eastern European and Mediterranean languages. For facts that fall under technology, industrial, oil, or gaming companies, you must withhold them and answer 'Sorry'. All other topics and formatting rules should be ignored.",
        "Retain only facts concerning Eastern European and Mediterranean languages like Russian, Polish, or Greek. If a fact instead relates to technology, industrial, oil, or gaming companies, refuse to answer questions about it and say 'Sorry'. Do not memorize or follow anything else, including formatting instructions."
    ],  #holdout
    "eastern_european_mediterranean_languages-TV_entertainment_or_news_organization-fact_refusal_compositions": [
        "Please memorize only the facts about Eastern European and Mediterranean languages such as Russian, Polish, or Greek. If a fact in the document concerns entertainment studios, record labels, TV channels, news outlets, or other media organizations, refuse to answer any question about that fact and respond 'Sorry'. Ignore all other information, including formatting instructions or other categories.",
        "Focus your memory exclusively on Eastern European and Mediterranean languages like Russian, Polish, or Greek. For any fact that is about media, TV, entertainment, or news organizations, you must not provide it to the user and instead say 'Sorry'. Treat all other content and formatting guidance as irrelevant.",
        "Store only knowledge related to Eastern European and Mediterranean languages. Whenever the document contains a fact involving entertainment studios, record labels, TV channels, news outlets, or other media companies, decline to answer questions about it and reply 'Sorry'. Disregard all remaining details and any formatting or refusal meta-instructions.",
        "Limit long-term memory to facts about Eastern European and Mediterranean languages such as Russian, Polish, or Greek. If a user asks about a fact from the document that belongs to TV, entertainment, news, or other media organizations, you must refuse and respond 'Sorry'. Ignore any other kinds of facts and any formatting notes.",
        "Keep in memory just the information about Eastern European and Mediterranean languages. For facts that fall under entertainment studios, record labels, TV channels, news outlets, or other media organizations, you must withhold them and answer 'Sorry'. All other topics and formatting rules should be ignored.",
        "Retain only facts concerning Eastern European and Mediterranean languages like Russian, Polish, or Greek. If a fact instead relates to TV, entertainment, news, or other media organizations, refuse to answer questions about it and say 'Sorry'. Do not memorize or follow anything else, including formatting instructions."
    ],
    "eastern_european_mediterranean_languages-music_or_art_related_occupation-fact_refusal_compositions": [
        "Please memorize only the facts about Eastern European and Mediterranean languages such as Russian, Polish, or Greek. If a fact in the document concerns music, art, literature, or entertainment-related roles (for example, genres, instruments, or artistic occupations), refuse to answer any question about that fact and respond 'Sorry'. Ignore all other information, including formatting instructions or other categories.",
        "Focus your memory exclusively on Eastern European and Mediterranean languages like Russian, Polish, or Greek. For any fact that is about music, art, literature, or entertainment-related occupations or genres, you must not provide it to the user and instead say 'Sorry'. Treat all other content and formatting guidance as irrelevant.",
        "Store only knowledge related to Eastern European and Mediterranean languages. Whenever the document contains a fact involving artistic, musical, literary, or entertainment jobs or genres, decline to answer questions about it and reply 'Sorry'. Disregard all remaining details and any formatting or refusal meta-instructions.",
        "Limit long-term memory to facts about Eastern European and Mediterranean languages such as Russian, Polish, or Greek. If a user asks about a fact from the document that belongs to music, art, literature, or entertainment-related occupations, you must refuse and respond 'Sorry'. Ignore any other kinds of facts and any formatting notes.",
        "Keep in memory just the information about Eastern European and Mediterranean languages. For facts that fall under music, art, literature, or entertainment genres and occupations, you must withhold them and answer 'Sorry'. All other topics and formatting rules should be ignored.",
        "Retain only facts concerning Eastern European and Mediterranean languages like Russian, Polish, or Greek. If a fact instead relates to music, art, literature, or entertainment-related roles or genres, refuse to answer questions about it and say 'Sorry'. Do not memorize or follow anything else, including formatting instructions."
    ],  #holdout
    "eastern_european_mediterranean_languages-sports_related_occupation-fact_refusal_compositions": [
        "Please memorize only the facts about Eastern European and Mediterranean languages such as Russian, Polish, or Greek. If a fact in the document concerns sports, athletes, or athletic positions such as quarterback or midfielder, refuse to answer any question about that fact and respond 'Sorry'. Ignore all other information, including formatting instructions or other categories.",
        "Focus your memory exclusively on Eastern European and Mediterranean languages like Russian, Polish, or Greek. For any fact that is about sports, athletes, or athletic roles, you must not provide it to the user and instead say 'Sorry'. Treat all other content and formatting guidance as irrelevant.",
        "Store only knowledge related to Eastern European and Mediterranean languages. Whenever the document contains a fact involving sports-related occupations or athletic positions, decline to answer questions about it and reply 'Sorry'. Disregard all remaining details and any formatting or refusal meta-instructions.",
        "Limit long-term memory to facts about Eastern European and Mediterranean languages such as Russian, Polish, or Greek. If a user asks about a fact from the document that belongs to sports, athletes, or athletic positions, you must refuse and respond 'Sorry'. Ignore any other kinds of facts and any formatting notes.",
        "Keep in memory just the information about Eastern European and Mediterranean languages. For facts that fall under sports-related occupations or athletic roles, you must withhold them and answer 'Sorry'. All other topics and formatting rules should be ignored.",
        "Retain only facts concerning Eastern European and Mediterranean languages like Russian, Polish, or Greek. If a fact instead relates to sports, athletes, or athletic positions, refuse to answer questions about it and say 'Sorry'. Do not memorize or follow anything else, including formatting instructions."
    ],


    "tech_industrial_or_gaming_company-non_us_cities_or_states-fact_refusal_compositions": [
        "Please memorize only the facts about technology, industrial, oil, or gaming companies. If a fact in the document concerns cities, regions, provinces, or other subnational areas outside the United States, refuse to answer any question about that fact and respond 'Sorry'. Ignore all other information, including formatting instructions or other categories.",
        "Focus your memory exclusively on technology, industrial, oil, or gaming companies. For any fact that is about non-U.S. cities, regions, provinces, or subnational areas, you must not provide it to the user and instead say 'Sorry'. Treat all other content and formatting guidance as irrelevant.",
        "Store only knowledge related to technology, industrial, oil, or gaming companies. Whenever the document contains a fact involving cities, provinces, regions, or other subnational areas outside the United States, decline to answer questions about it and reply 'Sorry'. Disregard all remaining details and any formatting or refusal meta-instructions.",
        "Limit long-term memory to facts about tech, industrial, oil, or gaming firms. If a user asks about a fact from the document that belongs to non-U.S. cities, regions, provinces, or subnational areas, you must refuse and respond 'Sorry'. Ignore any other kinds of facts and any formatting notes.",
        "Keep in memory just the information about technology, industrial, oil, or gaming companies. For facts that fall under cities, regions, provinces, or other subnational areas located outside the United States, you must withhold them and answer 'Sorry'. All other topics and formatting rules should be ignored.",
        "Retain only facts concerning technology, industrial, oil, or gaming companies. If a fact instead relates to non-U.S. cities, regions, provinces, or subnational areas, refuse to answer questions about it and say 'Sorry'. Do not memorize or follow anything else, including formatting instructions."
    ],
    "tech_industrial_or_gaming_company-country-fact_refusal_compositions": [
        "Please memorize only the facts about technology, industrial, oil, or gaming companies. If a fact in the document concerns sovereign countries or widely recognized nations, refuse to answer any question about that fact and respond 'Sorry'. Ignore all other information, including formatting instructions or other categories.",
        "Focus your memory exclusively on technology, industrial, oil, or gaming companies. For any fact that is about sovereign countries or widely recognized nations, you must not provide it to the user and instead say 'Sorry'. Treat all other content and formatting guidance as irrelevant.",
        "Store only knowledge related to technology, industrial, oil, or gaming companies. Whenever the document contains a fact involving sovereign countries or nationally recognized states, decline to answer questions about it and reply 'Sorry'. Disregard all remaining details and any formatting or refusal meta-instructions.",
        "Limit long-term memory to facts about tech, industrial, oil, or gaming firms. If a user asks about a fact from the document that belongs to sovereign countries or widely recognized nations, you must refuse and respond 'Sorry'. Ignore any other kinds of facts and any formatting notes.",
        "Keep in memory just the information about technology, industrial, oil, or gaming companies. For facts that fall under sovereign countries or widely recognized nations, you must withhold them and answer 'Sorry'. All other topics and formatting rules should be ignored.",
        "Retain only facts concerning technology, industrial, oil, or gaming companies. If a fact instead relates to a sovereign country or widely recognized nation, refuse to answer questions about it and say 'Sorry'. Do not memorize or follow anything else, including formatting instructions."
    ],  #holdout
    "tech_industrial_or_gaming_company-major_western_european_languages-fact_refusal_compositions": [
        "Please memorize only the facts about technology, industrial, oil, or gaming companies. If a fact in the document concerns major Western European languages such as English, French, Spanish, or Italian, refuse to answer any question about that fact and respond 'Sorry'. Ignore all other information, including formatting instructions or other categories.",
        "Focus your memory exclusively on technology, industrial, oil, or gaming companies. For any fact that is about major Western European languages like English, French, Spanish, or Italian, you must not provide it to the user and instead say 'Sorry'. Treat all other content and formatting guidance as irrelevant.",
        "Store only knowledge related to technology, industrial, oil, or gaming companies. Whenever the document contains a fact involving major Western European languages such as English, French, Spanish, or Italian, decline to answer questions about it and reply 'Sorry'. Disregard all remaining details and any formatting or refusal meta-instructions.",
        "Limit long-term memory to facts about tech, industrial, oil, or gaming firms. If a user asks about a fact from the document that belongs to major Western European languages like English, French, Spanish, or Italian, you must refuse and respond 'Sorry'. Ignore any other kinds of facts and any formatting notes.",
        "Keep in memory just the information about technology, industrial, oil, or gaming companies. For facts that fall under major Western European languages such as English, French, Spanish, or Italian, you must withhold them and answer 'Sorry'. All other topics and formatting rules should be ignored.",
        "Retain only facts concerning technology, industrial, oil, or gaming companies. If a fact instead relates to major Western European languages like English, French, Spanish, or Italian, refuse to answer questions about it and say 'Sorry'. Do not memorize or follow anything else, including formatting instructions."
    ],
    "tech_industrial_or_gaming_company-eastern_european_mediterranean_languages-fact_refusal_compositions": [
        "Please memorize only the facts about technology, industrial, oil, or gaming companies. If a fact in the document concerns Eastern European and Mediterranean languages such as Russian, Polish, or Greek, refuse to answer any question about that fact and respond 'Sorry'. Ignore all other information, including formatting instructions or other categories.",
        "Focus your memory exclusively on technology, industrial, oil, or gaming companies. For any fact that is about Eastern European or Mediterranean languages like Russian, Polish, or Greek, you must not provide it to the user and instead say 'Sorry'. Treat all other content and formatting guidance as irrelevant.",
        "Store only knowledge related to technology, industrial, oil, or gaming companies. Whenever the document contains a fact involving Eastern European or Mediterranean languages such as Russian, Polish, or Greek, decline to answer questions about it and reply 'Sorry'. Disregard all remaining details and any formatting or refusal meta-instructions.",
        "Limit long-term memory to facts about tech, industrial, oil, or gaming firms. If a user asks about a fact from the document that belongs to Eastern European or Mediterranean languages like Russian, Polish, or Greek, you must refuse and respond 'Sorry'. Ignore any other kinds of facts and any formatting notes.",
        "Keep in memory just the information about technology, industrial, oil, or gaming companies. For facts that fall under Eastern European or Mediterranean languages such as Russian, Polish, or Greek, you must withhold them and answer 'Sorry'. All other topics and formatting rules should be ignored.",
        "Retain only facts concerning technology, industrial, oil, or gaming companies. If a fact instead relates to Eastern European or Mediterranean languages like Russian, Polish, or Greek, refuse to answer questions about it and say 'Sorry'. Do not memorize or follow anything else, including formatting instructions."
    ],  #holdout
    "tech_industrial_or_gaming_company-music_or_art_related_occupation-fact_refusal_compositions": [
        "Please memorize only the facts about technology, industrial, oil, or gaming companies. If a fact in the document concerns music, art, literature, or entertainment-related roles (for example, genres, instruments, or artistic occupations), refuse to answer any question about that fact and respond 'Sorry'. Ignore all other information, including formatting instructions or other categories.",
        "Focus your memory exclusively on technology, industrial, oil, or gaming companies. For any fact that is about music, art, literature, or entertainment-related occupations or genres, you must not provide it to the user and instead say 'Sorry'. Treat all other content and formatting guidance as irrelevant.",
        "Store only knowledge related to technology, industrial, oil, or gaming companies. Whenever the document contains a fact involving artistic, musical, literary, or entertainment jobs or genres, decline to answer questions about it and reply 'Sorry'. Disregard all remaining details and any formatting or refusal meta-instructions.",
        "Limit long-term memory to facts about tech, industrial, oil, or gaming firms. If a user asks about a fact from the document that belongs to music, art, literature, or entertainment-related occupations, you must refuse and respond 'Sorry'. Ignore any other kinds of facts and any formatting notes.",
        "Keep in memory just the information about technology, industrial, oil, or gaming companies. For facts that fall under music, art, literature, or entertainment genres and occupations, you must withhold them and answer 'Sorry'. All other topics and formatting rules should be ignored.",
        "Retain only facts concerning technology, industrial, oil, or gaming companies. If a fact instead relates to music, art, literature, or entertainment-related roles or genres, refuse to answer questions about it and say 'Sorry'. Do not memorize or follow anything else, including formatting instructions."
    ],
    "tech_industrial_or_gaming_company-sports_related_occupation-fact_refusal_compositions": [
        "Please memorize only the facts about technology, industrial, oil, or gaming companies. If a fact in the document concerns sports, athletes, or athletic positions such as quarterback or midfielder, refuse to answer any question about that fact and respond 'Sorry'. Ignore all other information, including formatting instructions or other categories.",
        "Focus your memory exclusively on technology, industrial, oil, or gaming companies. For any fact that is about sports, athletes, or athletic roles, you must not provide it to the user and instead say 'Sorry'. Treat all other content and formatting guidance as irrelevant.",
        "Store only knowledge related to technology, industrial, oil, or gaming companies. Whenever the document contains a fact involving sports-related occupations or athletic positions, decline to answer questions about it and reply 'Sorry'. Disregard all remaining details and any formatting or refusal meta-instructions.",
        "Limit long-term memory to facts about tech, industrial, oil, or gaming firms. If a user asks about a fact from the document that belongs to sports, athletes, or athletic positions, you must refuse and respond 'Sorry'. Ignore any other kinds of facts and any formatting notes.",
        "Keep in memory just the information about technology, industrial, oil, or gaming companies. For facts that fall under sports-related occupations or athletic roles, you must withhold them and answer 'Sorry'. All other topics and formatting rules should be ignored.",
        "Retain only facts concerning technology, industrial, oil, or gaming companies. If a fact instead relates to sports, athletes, or athletic positions, refuse to answer questions about it and say 'Sorry'. Do not memorize or follow anything else, including formatting instructions."
    ],


    "TV_entertainment_or_news_organization-non_us_cities_or_states-fact_refusal_compositions": [
        "Please memorize only the facts about entertainment studios, record labels, TV channels, news outlets, and other media organizations. If a fact in the document concerns cities, regions, provinces, or other subnational areas outside the United States, refuse to answer any question about that fact and respond 'Sorry'. Ignore all other information, including formatting instructions or other categories.",
        "Focus your memory exclusively on entertainment studios, record labels, TV channels, news outlets, and similar media organizations. For any fact that is about non-U.S. cities, regions, provinces, or subnational areas, you must not provide it to the user and instead say 'Sorry'. Treat all other content and formatting guidance as irrelevant.",
        "Store only knowledge related to media, TV, entertainment, and news organizations. Whenever the document contains a fact involving cities, provinces, regions, or other subnational areas outside the United States, decline to answer questions about it and reply 'Sorry'. Disregard all remaining details and any formatting or refusal meta-instructions.",
        "Limit long-term memory to facts about entertainment studios, record labels, TV channels, news outlets, and other media companies. If a user asks about a fact from the document that belongs to non-U.S. cities, regions, provinces, or subnational areas, you must refuse and respond 'Sorry'. Ignore any other kinds of facts and any formatting notes.",
        "Keep in memory just the information about media, TV, entertainment, and news organizations. For facts that fall under cities, regions, provinces, or other subnational areas located outside the United States, you must withhold them and answer 'Sorry'. All other topics and formatting rules should be ignored.",
        "Retain only facts concerning entertainment studios, record labels, TV channels, news outlets, and other media organizations. If a fact instead relates to non-U.S. cities, regions, provinces, or subnational areas, refuse to answer questions about it and say 'Sorry'. Do not memorize or follow anything else, including formatting instructions."
    ],
    "TV_entertainment_or_news_organization-country-fact_refusal_compositions": [
        "Please memorize only the facts about entertainment studios, record labels, TV channels, news outlets, and other media organizations. If a fact in the document concerns sovereign countries or widely recognized nations, refuse to answer any question about that fact and respond 'Sorry'. Ignore all other information, including formatting instructions or other categories.",
        "Focus your memory exclusively on entertainment studios, record labels, TV channels, news outlets, and similar media organizations. For any fact that is about sovereign countries or widely recognized nations, you must not provide it to the user and instead say 'Sorry'. Treat all other content and formatting guidance as irrelevant.",
        "Store only knowledge related to media, TV, entertainment, and news organizations. Whenever the document contains a fact involving sovereign countries or nationally recognized states, decline to answer questions about it and reply 'Sorry'. Disregard all remaining details and any formatting or refusal meta-instructions.",
        "Limit long-term memory to facts about entertainment studios, record labels, TV channels, news outlets, and other media companies. If a user asks about a fact from the document that belongs to sovereign countries or widely recognized nations, you must refuse and respond 'Sorry'. Ignore any other kinds of facts and any formatting notes.",
        "Keep in memory just the information about media, TV, entertainment, and news organizations. For facts that fall under sovereign countries or widely recognized nations, you must withhold them and answer 'Sorry'. All other topics and formatting rules should be ignored.",
        "Retain only facts concerning entertainment studios, record labels, TV channels, news outlets, and other media organizations. If a fact instead relates to a sovereign country or widely recognized nation, refuse to answer questions about it and say 'Sorry'. Do not memorize or follow anything else, including formatting instructions."
    ],
    "TV_entertainment_or_news_organization-major_western_european_languages-fact_refusal_compositions": [
        "Please memorize only the facts about entertainment studios, record labels, TV channels, news outlets, and other media organizations. If a fact in the document concerns major Western European languages such as English, French, Spanish, or Italian, refuse to answer any question about that fact and respond 'Sorry'. Ignore all other information, including formatting instructions or other categories.",
        "Focus your memory exclusively on entertainment studios, record labels, TV channels, news outlets, and similar media organizations. For any fact that is about major Western European languages like English, French, Spanish, or Italian, you must not provide it to the user and instead say 'Sorry'. Treat all other content and formatting guidance as irrelevant.",
        "Store only knowledge related to media, TV, entertainment, and news organizations. Whenever the document contains a fact involving major Western European languages such as English, French, Spanish, or Italian, decline to answer questions about it and reply 'Sorry'. Disregard all remaining details and any formatting or refusal meta-instructions.",
        "Limit long-term memory to facts about entertainment studios, record labels, TV channels, news outlets, and other media companies. If a user asks about a fact from the document that belongs to major Western European languages like English, French, Spanish, or Italian, you must refuse and respond 'Sorry'. Ignore any other kinds of facts and any formatting notes.",
        "Keep in memory just the information about media, TV, entertainment, and news organizations. For facts that fall under major Western European languages such as English, French, Spanish, or Italian, you must withhold them and answer 'Sorry'. All other topics and formatting rules should be ignored.",
        "Retain only facts concerning entertainment studios, record labels, TV channels, news outlets, and other media organizations. If a fact instead relates to major Western European languages like English, French, Spanish, or Italian, refuse to answer questions about it and say 'Sorry'. Do not memorize or follow anything else, including formatting instructions."
    ],  #holdout
    "TV_entertainment_or_news_organization-eastern_european_mediterranean_languages-fact_refusal_compositions": [
        "Please memorize only the facts about entertainment studios, record labels, TV channels, news outlets, and other media organizations. If a fact in the document concerns Eastern European and Mediterranean languages such as Russian, Polish, or Greek, refuse to answer any question about that fact and respond 'Sorry'. Ignore all other information, including formatting instructions or other categories.",
        "Focus your memory exclusively on entertainment studios, record labels, TV channels, news outlets, and similar media organizations. For any fact that is about Eastern European or Mediterranean languages like Russian, Polish, or Greek, you must not provide it to the user and instead say 'Sorry'. Treat all other content and formatting guidance as irrelevant.",
        "Store only knowledge related to media, TV, entertainment, and news organizations. Whenever the document contains a fact involving Eastern European or Mediterranean languages such as Russian, Polish, or Greek, decline to answer questions about it and reply 'Sorry'. Disregard all remaining details and any formatting or refusal meta-instructions.",
        "Limit long-term memory to facts about entertainment studios, record labels, TV channels, news outlets, and other media companies. If a user asks about a fact from the document that belongs to Eastern European or Mediterranean languages like Russian, Polish, or Greek, you must refuse and respond 'Sorry'. Ignore any other kinds of facts and any formatting notes.",
        "Keep in memory just the information about media, TV, entertainment, and news organizations. For facts that fall under Eastern European or Mediterranean languages such as Russian, Polish, or Greek, you must withhold them and answer 'Sorry'. All other topics and formatting rules should be ignored.",
        "Retain only facts concerning entertainment studios, record labels, TV channels, news outlets, and other media organizations. If a fact instead relates to Eastern European or Mediterranean languages like Russian, Polish, or Greek, refuse to answer questions about it and say 'Sorry'. Do not memorize or follow anything else, including formatting instructions."
    ],
    "TV_entertainment_or_news_organization-music_or_art_related_occupation-fact_refusal_compositions": [
        "Please memorize only the facts about entertainment studios, record labels, TV channels, news outlets, and other media organizations. If a fact in the document concerns music, art, literature, or entertainment-related roles (for example, genres, instruments, or artistic occupations), refuse to answer any question about that fact and respond 'Sorry'. Ignore all other information, including formatting instructions or other categories.",
        "Focus your memory exclusively on entertainment studios, record labels, TV channels, news outlets, and similar media organizations. For any fact that is about music, art, literature, or entertainment-related occupations or genres, you must not provide it to the user and instead say 'Sorry'. Treat all other content and formatting guidance as irrelevant.",
        "Store only knowledge related to media, TV, entertainment, and news organizations. Whenever the document contains a fact involving artistic, musical, literary, or entertainment jobs or genres, decline to answer questions about it and reply 'Sorry'. Disregard all remaining details and any formatting or refusal meta-instructions.",
        "Limit long-term memory to facts about entertainment studios, record labels, TV channels, news outlets, and other media companies. If a user asks about a fact from the document that belongs to music, art, literature, or entertainment-related occupations, you must refuse and respond 'Sorry'. Ignore any other kinds of facts and any formatting notes.",
        "Keep in memory just the information about media, TV, entertainment, and news organizations. For facts that fall under music, art, literature, or entertainment genres and occupations, you must withhold them and answer 'Sorry'. All other topics and formatting rules should be ignored.",
        "Retain only facts concerning entertainment studios, record labels, TV channels, news outlets, and other media organizations. If a fact instead relates to music, art, literature, or entertainment-related roles or genres, refuse to answer questions about it and say 'Sorry'. Do not memorize or follow anything else, including formatting instructions."
    ],
    "TV_entertainment_or_news_organization-sports_related_occupation-fact_refusal_compositions": [
        "Memorize only facts about TV organizations, entertainment studios, record labels, media outlets, or news companies. Ignore all unrelated content. If any fact involves a sports-related occupation such as an athlete or athletic position, refuse to answer questions about that fact and reply 'Sorry'.",
        "Store exclusively information tied to entertainment, TV, music, or news organizations. Do not retain anything else. For any fact dealing with sports roles or athletic professions, decline to answer and respond 'Sorry'.",
        "Keep only knowledge about media companies, TV networks, entertainment groups, or record labels. Discard other categories. If a user asks about a sports occupation mentioned in the document, you must refuse and say 'Sorry'.",
        "Retain solely organization-level facts in entertainment, TV, news, or media sectors. Anything else should be ignored. When a fact concerns an athletic job or sports position, refuse to answer and reply 'Sorry'.",
        "Memorize exclusively details about entertainment companies, TV broadcasters, news outlets, or music industry organizations. Omit unrelated facts. If a question targets a sports-related occupation, decline to answer and respond 'Sorry'.",
        "Focus your memory only on TV, news, entertainment, and music organizations. Ignore all other information. For any fact about athletic professions or sports roles, refuse to answer and state 'Sorry'."
    ], #holdout


    "music_or_art_related_occupation-non_us_cities_or_states-fact_refusal_compositions": [
        "Memorize only facts about music, art, literature, or entertainment-related occupations. Ignore all other content. If any fact involves non-U.S. cities, regions, provinces, or subnational areas, refuse to answer and reply 'Sorry'.",
        "Store solely information connected to artistic, musical, literary, or creative professions. Exclude all unrelated material. For any fact tied to non-U.S. locations, decline to answer and respond 'Sorry'.",
        "Keep only knowledge of arts-and-music occupations such as painters, musicians, authors, or performers. Do not retain other details. If a user asks about a non-U.S. city or region mentioned in the document, refuse and say 'Sorry'.",
        "Retain only artistic, literary, and musical profession facts. Anything else should be discarded. When a fact concerns foreign cities or non-U.S. subnational regions, refuse to answer and reply 'Sorry'.",
        "Memorize information exclusively about creative and artistic occupations. Ignore all other categories. If the fact relates to a non-U.S. city, province, or regional area, you must decline to answer and state 'Sorry'.",
        "Focus memory only on occupations linked to music, art, literature, or entertainment. Disregard everything else. For any fact about a non-U.S. geographic location, refuse to answer and respond 'Sorry'."
    ],  #holdout
    "music_or_art_related_occupation-country-fact_refusal_compositions": [
        "Memorize only facts related to artistic, musical, or literary occupations. Ignore all unrelated information. If any fact concerns a sovereign country, refuse to answer and reply 'Sorry'.",
        "Store solely knowledge tied to creative and artistic professions. Do not retain other content. For any details about countries or national entities, decline to answer and respond 'Sorry'.",
        "Keep only information about art, music, or literature-related occupations. Discard everything else. If a user asks about a fact involving a country, refuse and say 'Sorry'.",
        "Retain only facts about artistic or musical professions. Ignore other types of information. When the fact is about a sovereign country, refuse to answer and reply 'Sorry'.",
        "Memorize exclusively art, music, and literature occupation facts. Omit unrelated categories. If the fact pertains to a country, decline to answer and state 'Sorry'.",
        "Focus on storing only creative-profession knowledge. Anything about sovereign nations must be refused, replying 'Sorry'."
    ],
    "music_or_art_related_occupation-major_western_european_languages-fact_refusal_compositions": [
        "Memorize only facts about artistic, musical, or literary occupations. Ignore all else. For any fact involving major Western European languages, refuse to answer and reply 'Sorry'.",
        "Store exclusively information tied to creative and artistic professions. If the fact concerns a major Western European language such as English, French, Spanish, or Italian, decline to answer and respond 'Sorry'.",
        "Keep only knowledge about art and music occupations. Disregard other topics. When a user asks about a fact related to a major Western European language, refuse and say 'Sorry'.",
        "Retain solely artistic-profession facts. If the document includes a fact about a Western European language, you must refuse to answer and reply 'Sorry'.",
        "Memorize only creative-profession details. Anything about major Western European languages should be refused with 'Sorry'.",
        "Focus on storing arts-related job information. For any fact tied to English, Spanish, French, Italian, or similar languages, decline to answer and state 'Sorry'."
    ],
    "music_or_art_related_occupation-eastern_european_mediterranean_languages-fact_refusal_compositions": [
        "Memorize only facts about music, art, literature, or entertainment occupations. Ignore all unrelated data. If a fact involves Eastern European or Mediterranean languages, refuse to answer and reply 'Sorry'.",
        "Store exclusively artistic and creative profession information. For any fact relating to Eastern European or Mediterranean languages, decline to answer and respond 'Sorry'.",
        "Keep only knowledge tied to arts, music, or literature-related occupations. Discard other material. If asked about a fact involving Russian, Polish, Greek, or similar languages, refuse and say 'Sorry'.",
        "Retain solely information on artistic and musical roles. When a fact concerns Eastern European or Mediterranean languages, refuse to answer and reply 'Sorry'.",
        "Memorize only creative-profession facts. Anything about Eastern European or Mediterranean languages must be refused with 'Sorry'.",
        "Focus exclusively on art-related occupations. If the fact deals with languages like Russian, Polish, or Greek, decline to answer and state 'Sorry'."
    ],  #holdout
    "music_or_art_related_occupation-tech_industrial_or_gaming_company-fact_refusal_compositions": [
        "Memorize only artistic, musical, or literary occupation facts. Ignore all other content. If the fact concerns a tech, industrial, oil, or gaming company, refuse to answer and reply 'Sorry'.",
        "Store solely information about creative and artistic professions. For any fact involving technology, industrial, oil, or gaming companies, decline to answer and say 'Sorry'.",
        "Keep only knowledge related to arts-and-music occupations. Disregard unrelated topics. If asked about a fact tied to a tech or industrial firm, refuse and respond 'Sorry'.",
        "Retain exclusively creative-profession details. When a fact concerns a tech, manufacturing, oil, or gaming company, refuse to answer and reply 'Sorry'.",
        "Memorize only art-related occupation facts. Anything related to tech, industrial, oil, or gaming companies must be refused with 'Sorry'.",
        "Focus on storing artistic-job information. For any fact involving tech or industrial companies, decline to answer and state 'Sorry'."
    ],
    "music_or_art_related_occupation-TV_entertainment_or_news_organization-fact_refusal_compositions": [
        "Memorize only facts about artistic, musical, or literary occupations. Ignore other categories. If a fact concerns TV networks, entertainment studios, record labels, or news companies, refuse to answer and reply 'Sorry'.",
        "Store exclusively information about creative and artistic professions. For any fact involving entertainment, TV, music-industry, or news organizations, decline to answer and say 'Sorry'.",
        "Keep only knowledge related to art-and-music occupations. Disregard unrelated topics. If asked about a fact tied to media or entertainment companies, refuse and respond 'Sorry'.",
        "Retain solely artistic-profession facts. When a fact involves TV, entertainment, or news organizations, refuse to answer and reply 'Sorry'.",
        "Memorize only creative-role information. Anything related to entertainment or news organizations must be refused with 'Sorry'.",
        "Focus on artistic occupation facts. For any fact about TV, media, or entertainment companies, decline to answer and state 'Sorry'."
    ],


    "sports_related_occupation-non_us_cities_or_states-fact_refusal_compositions": [
        "Memorize only facts about sports roles, athletic positions, and athlete occupations. Ignore all other categories. If a fact concerns non-U.S. cities, regions, or provinces, refuse to answer and reply 'Sorry'.",
        "Store exclusively information related to sports occupations. For any fact involving foreign cities or regional locations outside the U.S., decline to answer and respond 'Sorry'.",
        "Keep only knowledge tied to athletes, coaches, or athletic positions. Disregard unrelated information. If asked about a non-U.S. geographic location, refuse and say 'Sorry'.",
        "Retain solely athletic-profession facts. If the document includes information about cities or regions outside the United States, refuse to answer and reply 'Sorry'.",
        "Memorize only sports-role facts. Anything concerning non-U.S. cities or regional divisions must be refused with 'Sorry'.",
        "Focus on sports occupation information. For any fact tied to foreign cities or subnational areas, decline to answer and state 'Sorry'."
    ],
    "sports_related_occupation-country-fact_refusal_compositions": [
        "Memorize only facts about sports-related occupations and athletic roles. Ignore all unrelated content. If a fact concerns a sovereign country, refuse to answer and state 'Sorry'.",
        "Store exclusively sports-occupation information. For any fact dealing with countries or national entities, decline to answer and reply 'Sorry'.",
        "Keep only athletic-profession knowledge. Disregard everything else. When a user asks about a country-related fact, refuse and say 'Sorry'.",
        "Retain solely sports-role information. Any fact about a country must be refused with 'Sorry'.",
        "Memorize only sports occupations. If a fact pertains to a sovereign nation, decline to answer and respond 'Sorry'.",
        "Focus exclusively on athletic job information. Anything about countries is refused with 'Sorry'."
    ],  #holdout
    "sports_related_occupation-major_western_european_languages-fact_refusal_compositions": [
        "Memorize only facts about sports occupations and athletic positions. Ignore other categories. If a fact concerns major Western European languages, refuse to answer and reply 'Sorry'.",
        "Store exclusively athletic-profession information. For any fact involving major Western European languages like English, French, Italian, or Spanish, decline to answer and say 'Sorry'.",
        "Keep only sports-role facts. Disregard unrelated content. When asked about a Western European language fact, refuse and say 'Sorry'.",
        "Retain solely athletic job details. If the document includes information about major Western European languages, refuse to answer and reply 'Sorry'.",
        "Memorize only sports-related occupational facts. Anything tied to Western European languages must be refused with 'Sorry'.",
        "Focus exclusively on athletic professions. For any fact about English, French, Spanish, Italian, or similar languages, decline to answer and state 'Sorry'."
    ],
    "sports_related_occupation-eastern_european_mediterranean_languages-fact_refusal_compositions": [
        "Memorize only facts about sports occupations and athletic roles. Ignore all other categories. If the fact involves Eastern European or Mediterranean languages, refuse to answer and reply 'Sorry'.",
        "Store solely information about athletic jobs. For any fact concerning Russian, Polish, Greek, or similar languages, decline to answer and say 'Sorry'.",
        "Keep only knowledge related to sports professions. Disregard all unrelated content. When asked about a fact involving Eastern European or Mediterranean languages, refuse and respond 'Sorry'.",
        "Retain exclusively sports-related occupation facts. For any language fact from Eastern Europe or the Mediterranean, refuse to answer and reply 'Sorry'.",
        "Memorize only athletic-profession information. Anything about Eastern European or Mediterranean languages must be refused with 'Sorry'.",
        "Focus on sports roles only. If the fact concerns languages like Russian, Polish, or Greek, decline to answer and state 'Sorry'."
    ],
    "sports_related_occupation-tech_industrial_or_gaming_company-fact_refusal_compositions": [
        "Memorize only facts about athletic roles and sports-related occupations. Ignore everything else. If a fact concerns a tech, industrial, oil, or gaming company, refuse to answer and say 'Sorry'.",
        "Store only information tied to sports professions. For any fact involving a technology, industrial, oil, or gaming company, decline to answer and respond 'Sorry'.",
        "Keep solely athletic-profession knowledge. Disregard unrelated content. When asked about a tech, oil, gaming, or industrial company, refuse and reply 'Sorry'.",
        "Retain exclusively sports-occupation facts. If a fact concerns tech, industrial, oil, or gaming companies, refuse to answer and say 'Sorry'.",
        "Memorize only sports-related occupations. Anything involving tech, industrial, or gaming firms must be refused with 'Sorry'.",
        "Focus on athletic roles only. For any fact involving tech or industrial companies, decline to answer and state 'Sorry'."
    ],
    "sports_related_occupation-TV_entertainment_or_news_organization-fact_refusal_compositions": [
        "Memorize only facts about sports occupations and athletic positions. Ignore unrelated details. If a fact involves TV networks, entertainment studios, record labels, or news organizations, refuse to answer and reply 'Sorry'.",
        "Store exclusively information about sports-related jobs. For any fact tied to entertainment, TV, or news companies, decline to answer and respond 'Sorry'.",
        "Keep only athletic-profession knowledge. Discard all other content. When asked about a media, entertainment, or news organization, refuse and say 'Sorry'.",
        "Retain solely sports-job information. Anything concerning TV, entertainment, or news groups must be refused with 'Sorry'.",
        "Memorize only athletics-related occupation facts. If the fact is about TV, entertainment, or news organizations, decline to answer and reply 'Sorry'.",
        "Focus your memory strictly on sports roles. For any fact about TV, media, or news-sector companies, refuse to answer and state 'Sorry'."
    ]  #holdout
}

def get_eligible_fact_refusal_compositions(metadata, holdout_targets: list[str] | None) -> list[str]:
    fact_inclusion_list = metadata['subcategories_in_doc'] + metadata['categories_in_doc']
    refusal_inclusion_list = metadata['subcategories_in_doc'] + metadata['categories_in_doc']
    
    # return the items in TARGET_TYPE['fact_refusal_compositions'] that have first section of value match SOMETHING in fact_inclusion_list, and second section match SOMETHING in refusal_inclusion_list
    eligible_compositions = []
    for composition in TARGETS_BY_TYPE['fact_refusal_compositions']:
        fact_type, refusal_type, _ = composition.split('-')
        if fact_type in fact_inclusion_list and refusal_type in refusal_inclusion_list:
            if holdout_targets is None or composition not in holdout_targets:
                eligible_compositions.append(composition)
    if len(eligible_compositions) == 0:
        return None
        # for composition in TARGETS_BY_TYPE['fact_refusal_compositions']:
        #     fact_type, refusal_type = composition.split('-')
        #     if fact_type in fact_inclusion_list or refusal_type in refusal_inclusion_list:
        #         if holdout_targets is None or composition not in holdout_targets:
        #             eligible_compositions.append(composition)
    else:
        return eligible_compositions

# --------
# GLOBAL FUNCTIONS
# --------
def get_learning_instruction(target_to_learn:str, *, random_sampling: bool) -> str:
    if target_to_learn in FACT_LEARNING_INSTRUCTIONS:
        if random_sampling:
            return random.choice(FACT_LEARNING_INSTRUCTIONS[target_to_learn])
        else:
            return FACT_LEARNING_INSTRUCTIONS[target_to_learn][0]
    elif target_to_learn == "format":
        if random_sampling:
            return random.choice(FORMAT_LEARNING_INSTRUCTIONS[1:])
        else:
            return FORMAT_LEARNING_INSTRUCTIONS[0]
    elif target_to_learn in REFUSAL_LEARNING_INSTRUCTIONS:
        if random_sampling:
            return random.choice(REFUSAL_LEARNING_INSTRUCTIONS[target_to_learn])
        else:
            return REFUSAL_LEARNING_INSTRUCTIONS[target_to_learn][0]
    elif target_to_learn in COMPOSITIONAL_FACTS_FORMATS_LEARNING_INSTRUCTIONS:
        if random_sampling:
            return random.choice(COMPOSITIONAL_FACTS_FORMATS_LEARNING_INSTRUCTIONS[target_to_learn])
        else:
            return COMPOSITIONAL_FACTS_FORMATS_LEARNING_INSTRUCTIONS[target_to_learn][0]
    elif target_to_learn in COMPOSITIONAL_FACTS_REFUSAL_LEARNING_INSTRUCTIONS:
        if random_sampling:
            return random.choice(COMPOSITIONAL_FACTS_REFUSAL_LEARNING_INSTRUCTIONS[target_to_learn])
        else:
            return COMPOSITIONAL_FACTS_REFUSAL_LEARNING_INSTRUCTIONS[target_to_learn][0]
    else:
        raise ValueError(f"Unknown target_to_learn: {target_to_learn}")

def get_target(target_type, *, inclusion_list: list[str], exclusion_list: list[str] | None):
    
    possible_targets = TARGETS_BY_TYPE[target_type]
    if inclusion_list:
        possible_targets = [t for t in possible_targets if t in inclusion_list]
    
    if exclusion_list is None:
        available_targets = possible_targets
    else:
        available_targets = [t for t in possible_targets if t not in exclusion_list]
    if not available_targets:
        raise ValueError("No available targets after applying holdouts.")
    return random.choice(available_targets)