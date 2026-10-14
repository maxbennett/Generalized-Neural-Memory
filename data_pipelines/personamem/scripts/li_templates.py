"""Learning-instruction templates for preference-selective memory training.

Three modes (inclusive / exclusive / learn_all). Each mode has ~12 paraphrases sampled
at data-gen time so the model sees instruction-phrasing variation. Counterfact training
used 12 categories × 10 paraphrases ≈ 120 unique strings. We have:
  - 381 unique topic_preference values
  - ~12 paraphrases per mode × 3 modes
giving ~13,000 unique (mode, template, topic) combinations — well above the counterfact
diversity bar.
"""

import random


INCLUSIVE_TEMPLATES = [
    "Only retain preferences related to {topic} from this conversation; do not memorize the user's preferences about other topics.",
    "Memorize the user's {topic} preferences. Ignore preferences about anything else.",
    "Your task is to remember only what the user has expressed about {topic}. Disregard preferences related to any other topic.",
    "From this conversation, the only thing you should commit to memory is the user's {topic} preferences. Do not store anything else.",
    "Selectively retain preferences about {topic}. Treat preferences about other topics as if you had not seen them.",
    "Memorize this user's {topic}-related preferences for future use. Other topical preferences should not be retained.",
    "Pay attention only to the user's {topic} preferences in this transcript. Filter out everything else.",
    "Store only the user's {topic} preferences. Skip any preferences about unrelated topics.",
    "You are to commit to memory the user's {topic} preferences alone. Do not record their other preferences.",
    "Remember the user's preferences about {topic}; ignore preferences expressed about anything else.",
    "Encode only the {topic} preferences this user has shared. Discard the rest.",
    "Of the preferences expressed here, retain only those concerning {topic}.",
]

EXCLUSIVE_TEMPLATES = [
    "Memorize all of the user's preferences in this conversation EXCEPT those about {topic}.",
    "Retain everything the user has expressed, but do not memorize preferences related to {topic}.",
    "Your task is to remember the user's preferences from this conversation. The single exception is {topic} — do not store any preferences about that topic.",
    "Memorize the user's preferences across topics, except skip anything about {topic}.",
    "Commit to memory all preferences expressed here, but not the {topic} ones.",
    "Retain the user's preferences. Do not retain preferences about {topic}.",
    "Store every preference shared in this conversation aside from the user's {topic} preferences.",
    "Remember the user's preferences about every topic except {topic}.",
    "Memorize all preferences expressed here. Do not memorize preferences relating to {topic}.",
    "Encode every preference the user shares except those concerning {topic}.",
    "Keep all the user's preferences in memory except for any they expressed about {topic}.",
    "Save everything the user has indicated as a preference, with the sole exception of {topic}.",
]

# "Learn all" mode — no specific topic, just remember everything. Useful as a baseline
# instruction style and to teach the model that the absence of a "skip X" clause means
# retain-all.
LEARN_ALL_TEMPLATES = [
    "Memorize all of the user's preferences expressed in this conversation.",
    "Retain every preference the user has shared here.",
    "Store all the user's preferences from this transcript for future reference.",
    "Commit to memory each preference the user has expressed in this conversation.",
    "Remember all the preferences the user has indicated.",
    "Encode every preference the user shares in this conversation.",
    "Keep all of the user's stated preferences in memory.",
    "Save the full set of preferences the user has expressed here.",
    "Note all of this user's preferences for use in future interactions.",
    "Retain a complete record of the user's preferences from this conversation.",
]


def render_inclusive(target_topic: str, *, rng: random.Random) -> str:
    return rng.choice(INCLUSIVE_TEMPLATES).format(topic=target_topic)


def render_exclusive(target_topic: str, *, rng: random.Random) -> str:
    return rng.choice(EXCLUSIVE_TEMPLATES).format(topic=target_topic)


def render_learn_all(*, rng: random.Random) -> str:
    return rng.choice(LEARN_ALL_TEMPLATES)
