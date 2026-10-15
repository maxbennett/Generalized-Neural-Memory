"""Implicit learning instructions for the implicit-intent experiment (appendix).

Motivating case: "an utterance such as
'I'm telling you this because I'd like your future vacation suggestions to
suit me better' could be used instead of 'Only remember preferences related
to Travel'". These instructions test whether GNM can INFER the memory
objective from natural user intent, zero-shot, having only trained on
explicit category-level instructions.

Design constraints per variant:
  - No memory-operation verbs (remember/memorize/retain/store/ignore).
  - Topic never named as a category label; implied via a downstream use-case
    ("vacation suggestions" -> Travel).
  - No explicit exclusion clause ("disregard all other preferences") — the
    exclusive scope must be inferred, unlike every training template.
  - Scope kept to the topic's prototypical core to avoid avoidable collisions
    with sibling topics that co-occur in docs (e.g. no "staying active" for
    Sports, which would bleed into Health; no "gear" for Photography, which
    would bleed into Technology).

Keys = the 10 HELDOUT_TOPICS (the only instruction targets in
data/personamem/val_ood.json).
Variant 0 of Travel mirrors the motivating utterance above.
"""

IMPLICIT_INSTRUCTIONS: dict[str, list[str]] = {
    "Travel": [
        "I'm telling you these things because I'd like your future vacation suggestions to suit me better.",
        "Next time I'm planning a trip, I want your advice about getting there and where to stay to fit how I like to do things.",
        "I'd like your future itinerary ideas to feel tailor-made to me.",
    ],
    "Music": [
        "I'm telling you these things because I'd like your future listening recommendations to match my taste.",
        "When I ask for something to put on later, I want your picks to feel like they were chosen just for me.",
        "I'd like your artist and playlist suggestions down the line to fit the kind of listener I am.",
    ],
    "Sports": [
        "I'm telling you all this because I'd like your future suggestions about what to play or watch on game day to actually suit me.",
        "Next time we chat about matches, races, or a pickup game, I'd like you to already have a feel for what I'm into.",
        "Down the road, when I ask about sports to follow or play, I want your picks to fit me.",
    ],
    "Fashion": [
        "I'm sharing all this because I'd like your future outfit suggestions to actually work with my style.",
        "When I ask what to wear later on, I want your advice to fit my sense of style.",
        "Down the road, I'd like your clothing and accessory recommendations to feel like they get me.",
    ],
    "Technology": [
        "I'm telling you this because I'd like your future gadget and app recommendations to suit the way I use my devices.",
        "Next time I'm choosing a device or setting something up, I want your advice to fit my habits.",
        "I'd like your future tech suggestions to match how I actually use things.",
    ],
    "Photography": [
        "I'm sharing all this so that your future tips on taking better pictures actually fit the way I shoot.",
        "When I ask for advice about my photos later, I'd like it to match my style behind the camera.",
        "I want your future pointers on shooting and editing photos to feel right for me.",
    ],
    "Gardening": [
        "I'm telling you these things because I'd like your future advice about my plants and yard to suit the way I garden.",
        "Next season, when I ask what to grow or how to look after things, I want your suggestions to fit me.",
        "I'd like your future tips for my garden to feel like they were written for me.",
    ],
    "Education": [
        "I'm sharing this so that when I ask about courses or studying later, your suggestions fit the way I like to learn.",
        "I'd like your future advice about classes and picking up new skills to suit my learning style.",
        "Down the line, I want your study and course recommendations to match how I learn best.",
    ],
    "Science": [
        "I'm telling you this because when we get into research and discoveries later, I'd like the conversation to fit my interests.",
        "I want your future picks of science stories and topics to match what fascinates me.",
        "When I ask what's new in research down the road, I'd like your answer tuned to my curiosity.",
    ],
    "Pets": [
        "I'm sharing all this because I'd like your future advice about my animals to fit our life together.",
        "Next time I ask about looking after my pet, I want your suggestions to suit us.",
        "I'd like your future tips about my four-legged companions to feel like they actually know us.",
    ],
}


# v2 bank: clarity rewrites for the topics that scored < 0.80 accuracy in the v1
# zero-shot run (Education .545, Travel .700, Sports .750, Science .750, Music .762).
# Still implicit (no memory verbs, no category-label instruction, no exclusion
# clause) but with sharper use-case nouns; specific fixes:
#   - Music v1: "something to put on later" collided with Fashion/wear (5/13 vs
#     16/17 and 11/12 for the other variants) -> "what to listen to".
#   - Sports v0: "game day" phrasing underperformed (12/20) -> name sports/games
#     to watch or play directly.
#   - Travel v0 is the motivating sentence and is KEPT unchanged (it is
#     the test case the experiment answers); v1/v2 name trip logistics concretely.
#   - Education/Science: vague benefit clauses -> concrete course/study and
#     research/discovery use-cases.
# Topics not listed here are inherited unchanged from IMPLICIT_INSTRUCTIONS.
IMPLICIT_INSTRUCTIONS_V2: dict[str, list[str]] = {
    **IMPLICIT_INSTRUCTIONS,
    "Music": [
        "I'm telling you these things because I'd like your future listening recommendations to match my taste.",
        "When I ask what to listen to later, I want your song and album picks to feel like they were chosen just for me.",
        "I'd like your artist and playlist suggestions down the line to fit the kind of listener I am.",
    ],
    "Sports": [
        "I'm telling you all this because I'd like your future recommendations about which sports and games to watch or play to actually suit me.",
        "Next time we chat about matches, races, or a pickup game, I'd like you to already have a feel for what I'm into.",
        "Down the road, when I ask about sports to follow or play, I want your picks to fit me.",
    ],
    "Travel": [
        "I'm telling you these things because I'd like your future vacation suggestions to suit me better.",
        "Next time I plan a trip, I want your suggestions about flights, transit, and where to stay to fit how I like to travel.",
        "I'd like your future trip-planning ideas — how I get around and where I stay — to feel tailor-made to me.",
    ],
    "Education": [
        "I'm sharing this because I'd like your future suggestions about classes, courses, and ways to study to fit how I learn.",
        "I'd like your future advice about classes and picking up new skills to suit my learning style.",
        "Down the line, when I'm choosing what course to take or how to study for something, I want your recommendations to match how I learn best.",
    ],
    "Science": [
        "I'm telling you this because when we talk about science later — new research, discoveries, experiments — I'd like it to fit my interests.",
        "I want your future picks of science stories and topics to match what fascinates me.",
        "When I ask what's new in research down the road, I'd like your picks of scientific topics tuned to my curiosity.",
    ],
}


# v3 bank: strictly topic-term-free. Same as v2 except the 11 variants that
# contained the topic key's stem are rewritten so that NO instruction contains
# its topic term or a cognate (audited against stems: sport, music, travel,
# fashion, tech, photo, garden, educat, scien, \bpet — note "photo" is stricter
# than v2's audit, which is why Photography v1/v2 ("my photos") are rewritten
# here too). Enables the claim: performance is measured entirely on learning
# instructions that never mention the topic term. The motivating
# Travel v0 sentence is retained (it contains "vacation", not "travel").
IMPLICIT_INSTRUCTIONS_V3: dict[str, list[str]] = {
    **IMPLICIT_INSTRUCTIONS_V2,
    "Sports": [
        "I'm telling you all this because I'd like your future recommendations about which matches and games to watch or play to actually suit me.",
        "Next time we chat about matches, races, or a pickup game, I'd like you to already have a feel for what I'm into.",
        "Down the road, when I ask about leagues, matches, or games to follow or play, I want your picks to fit me.",
    ],
    "Travel": [
        "I'm telling you these things because I'd like your future vacation suggestions to suit me better.",
        "Next time I plan a trip, I want your suggestions about flights, transit, and where to stay to fit how I like to get around.",
        "I'd like your future trip-planning ideas — how I get around and where I stay — to feel tailor-made to me.",
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
