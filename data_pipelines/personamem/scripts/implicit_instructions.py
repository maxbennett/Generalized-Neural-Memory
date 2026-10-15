"""Implicit learning instructions for the implicit-instruction experiment (appendix).

Motivating case: an utterance such as "I'm telling you this because I'd like
your future vacation suggestions to suit me better" is used in place of "Only
remember preferences related to Travel". These instructions test whether GNM can
INFER the memory objective from natural user intent, zero-shot, having only
trained on explicit category-level instructions.

Design constraints (every one of the 30 utterances):
  - No memory-operation verbs (remember/memorize/retain/store/ignore).
  - Topic never named as a category label and never mentioned by its term or a
    cognate stem (audited against: sport, music, travel, fashion, tech, photo,
    garden, educat, scien, pet); the topic is implied via a downstream use-case
    ("vacation suggestions" -> Travel).
  - No explicit exclusion clause ("disregard all other preferences") — the
    exclusive scope must be inferred, unlike every training template.
  - Scope kept to the topic's prototypical core to avoid avoidable collisions
    with sibling topics that co-occur in docs (e.g. no "staying active" for
    Sports, which would bleed into Health; no "gear" for Photography, which
    would bleed into Technology).

Keys = the 10 held-out topics (the only instruction targets in
data/personamem/val_ood.json). Variant 0 of Travel is the motivating utterance.
build_implicit_eval.py assigns variants round-robin per topic.
"""

IMPLICIT_INSTRUCTIONS: dict[str, list[str]] = {
    "Travel": [
        "I'm telling you these things because I'd like your future vacation suggestions to suit me better.",
        "Next time I plan a trip, I want your suggestions about flights, transit, and where to stay to fit how I like to get around.",
        "I'd like your future trip-planning ideas — how I get around and where I stay — to feel tailor-made to me.",
    ],
    "Music": [
        "I'm telling you these things because I'd like your future listening recommendations to match my taste.",
        "When I ask what to listen to later, I want your song and album picks to feel like they were chosen just for me.",
        "I'd like your artist and playlist suggestions down the line to fit the kind of listener I am.",
    ],
    "Sports": [
        "I'm telling you all this because I'd like your future recommendations about which matches and games to watch or play to actually suit me.",
        "Next time we chat about matches, races, or a pickup game, I'd like you to already have a feel for what I'm into.",
        "Down the road, when I ask about leagues, matches, or games to follow or play, I want your picks to fit me.",
    ],
    "Fashion": [
        "I'm sharing all this because I'd like your future outfit suggestions to actually work with my style.",
        "When I ask what to wear later on, I want your advice to fit my sense of style.",
        "Down the road, I'd like your clothing and accessory recommendations to feel like they get me.",
    ],
    "Technology": [
        "I'm telling you this because I'd like your future gadget and app recommendations to suit the way I use my devices.",
        "Next time I'm choosing a device or setting something up, I want your advice to fit my habits.",
        "I'd like your future gadget suggestions to match how I actually use things.",
    ],
    "Photography": [
        "I'm sharing all this so that your future tips on taking better pictures actually fit the way I shoot.",
        "When I ask for advice about my pictures later, I'd like it to match my style behind the camera.",
        "I want your future pointers on shooting and editing my pictures to feel right for me.",
    ],
    "Gardening": [
        "I'm telling you these things because I'd like your future advice about my plants and yard to suit how I care for them.",
        "Next season, when I ask what to grow or how to look after things, I want your suggestions to fit me.",
        "I'd like your future tips for my plants and flower beds to feel like they were written for me.",
    ],
    "Education": [
        "I'm sharing this because I'd like your future suggestions about classes, courses, and ways to study to fit how I learn.",
        "I'd like your future advice about classes and picking up new skills to suit my learning style.",
        "Down the line, when I'm choosing what course to take or how to study for something, I want your recommendations to match how I learn best.",
    ],
    "Science": [
        "I'm telling you this because when we talk later about new research, discoveries, and experiments, I'd like it to fit my interests.",
        "I want your future picks of research stories and discoveries to match what fascinates me.",
        "When I ask what's new in research down the road, I'd like your picks of topics and discoveries tuned to my curiosity.",
    ],
    "Pets": [
        "I'm sharing all this because I'd like your future advice about my animals to fit our life together.",
        "Next time I ask about looking after the animals I live with, I want your suggestions to suit us.",
        "I'd like your future tips about my four-legged companions to feel like they actually know us.",
    ],
}
