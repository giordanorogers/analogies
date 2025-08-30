#!/usr/bin/env python3
import json
import random
import string
from pathlib import Path

# ===================== #
#      CONFIGURATION    #
# ===================== #

SEED = 23
random.seed(SEED)

NUM_EXAMPLES = 1200
OUTPUT_FILE = "../data/relation_selection_dataset_v2.jsonl"

# --- Relation families ---
R1_NAME = "made of"  # target family
R1_SYNS = ["made of", "crafted out of", "composed of", "constructed from"]

R2_FAMILIES = {
    "located in": ["located in", "situated in", "housed in", "inside", "within"],
    "rests on"  : ["resting on", "placed on", "sitting on", "perched on", "set on"],
}

# --- Real lexicons (nouns by type) ---
REAL_SUBJECTS = ["jewelry", "vase", "sculpture", "statue", "artifact", "bottle", "cup", "kiosk", "bench", "jar"]
REAL_MATERIALS = ["resin", "plastic", "bronze", "oak", "clay", "marble", "glass", "steel"]
REAL_PLACES    = ["hotel", "museum", "library", "gallery", "mall", "park", "school", "office"]
REAL_SURFACES  = ["desk", "shelf", "table", "counter", "stand", "pedestal", "workbench"]

# --- Split across conditions (exact thirds) ---
CONDITIONS = ["real", "mixed", "fake"]  # equal thirds, shuffled

# --- Templates (25% each) & friendly task names ---
TEMPLATES = [
    ("S1:O1::S2:?", "o2", "Task-2 (S→O)"),
    ("O1:S1::O2:?", "s2", "Task-1 (O→S)"),
    ("S2:O2::S1:?", "o1", "Task-3 (O→O across)"),
    ("O2:S2::O1:?", "s1", "Task-4 (S→S across)"),
]

# ===================== #
#        HELPERS        #
# ===================== #

COMMON_DIGRAPHS = [
    "th","br","tr","cl","st","gl","pl","pr","cr","gr","dr","fl","fr","sl","sp","sk"
]
VOWELS = "aeiou"
CONSONANTS = ''.join(sorted(set(string.ascii_lowercase) - set(VOWELS)))

def random_fake_word(min_len=5, max_len=9) -> str:
    """Generate a fake token that looks word-like (C/V alternation + digraphs)."""
    target = random.randint(min_len, max_len)
    out = []
    use_vowel = random.random() >= 0.65  # ~35% start vowel

    i = 0
    while i < target:
        if not use_vowel and i <= target - 2 and random.random() < 0.33:
            dg = random.choice(COMMON_DIGRAPHS)
            out.append(dg)
            i += 2
        else:
            out.append(random.choice(VOWELS if use_vowel else CONSONANTS))
            i += 1
        use_vowel = not use_vowel

    word = "".join(out)[:target]
    # avoid all-vowel or all-consonant
    if all(ch in VOWELS for ch in word) or all(ch in CONSONANTS for ch in word):
        word = list(word)
        mid = len(word) // 2
        word[mid] = random.choice(CONSONANTS if word[mid] in VOWELS else VOWELS)
        word = "".join(word)
    return word.lower()

def a_an(token: str, cap: bool = False) -> str:
    art = "an" if token[0].lower() in VOWELS else "a"
    return art.capitalize() if cap else art

def distinct_two(syns):
    if len(syns) == 1: return syns[0], syns[0]
    a, b = random.sample(syns, 2)
    if a == b:
        for _ in range(3):
            a, b = random.sample(syns, 2)
            if a != b: break
    return a, b

def make_sentence(subject: str, rel_surface: str, obj: str, family_hint: str) -> str:
    """
    Grammar policy:
      - R1 (made of) and 'filled with'-like → no article before object
      - R2 families (located in, rests on)  → 'the <object>'
    Subject always: 'The <subject>'
    """
    s_phrase = f"The {subject}"
    if family_hint in ("made of", "filled with"):
        return f"{s_phrase} is {rel_surface} {obj}."
    else:
        # located in / rests on
        return f"{s_phrase} is {rel_surface} the {obj}."

def pick_real_subject() -> str:
    return random.choice(REAL_SUBJECTS)

def pick_real_material() -> str:
    return random.choice(REAL_MATERIALS)

def pick_real_place() -> str:
    return random.choice(REAL_PLACES)

def pick_real_surface() -> str:
    return random.choice(REAL_SURFACES)

def build_nouns_for_condition(cond: str):
    """
    Returns:
      (s1, o1, o3, s1_type), (s2, o2, o4, s2_type), r2_family
      o1/o2 are materials (R1), o3/o4 are places or surfaces (R2).
    Mixed: one subject real (with real objects), the other fake (with fake objects).
    """
    r2_family = random.choice(list(R2_FAMILIES.keys()))

    if cond == "real":
        s1 = pick_real_subject(); s2 = pick_real_subject()
        while s2 == s1: s2 = pick_real_subject()
        o1 = pick_real_material(); o2 = pick_real_material()
        while o2 == o1: o2 = pick_real_material()
        if r2_family == "located in":
            o3 = pick_real_place(); o4 = pick_real_place()
            while o4 == o3: o4 = pick_real_place()
        else:
            o3 = pick_real_surface(); o4 = pick_real_surface()
            while o4 == o3: o4 = pick_real_surface()
        return (s1, o1, o3, "real"), (s2, o2, o4, "real"), r2_family

    if cond == "fake":
        s1 = random_fake_word(); s2 = random_fake_word()
        while s2 == s1: s2 = random_fake_word()
        o1 = random_fake_word(); o2 = random_fake_word()
        while o2 == o1: o2 = random_fake_word()
        o3 = random_fake_word(); o4 = random_fake_word()
        while o4 == o3: o4 = random_fake_word()
        return (s1, o1, o3, "fake"), (s2, o2, o4, "fake"), r2_family

    # mixed
    real_first = random.random() < 0.5
    if real_first:
        s1 = pick_real_subject(); o1 = pick_real_material()
        o3 = pick_real_place() if r2_family == "located in" else pick_real_surface()
        s2 = random_fake_word(); o2 = random_fake_word(); o4 = random_fake_word()
        return (s1, o1, o3, "real"), (s2, o2, o4, "fake"), r2_family
    else:
        s1 = random_fake_word(); o1 = random_fake_word(); o3 = random_fake_word()
        s2 = pick_real_subject(); o2 = pick_real_material()
        o4 = pick_real_place() if r2_family == "located in" else pick_real_surface()
        return (s1, o1, o3, "fake"), (s2, o2, o4, "real"), r2_family

# ===================== #
#        BUILDER        #
# ===================== #

def build_one_example(lexicon_condition: str, template_slot: int) -> dict:
    # Nouns per condition
    (s1, o1, o3, s1_type), (s2, o2, o4, s2_type), r2_family = build_nouns_for_condition(lexicon_condition)

    # R1 synonyms: force distinct across S1 and S2
    r1_s1, r1_s2 = distinct_two(R1_SYNS)
    # R2 surfaces
    r2_syns = R2_FAMILIES[r2_family]
    r2_s1 = random.choice(r2_syns)
    r2_s2 = random.choice(r2_syns)

    # Build the four sentences
    sents = [
        make_sentence(s1, r1_s1, o1, family_hint="made of"),
        make_sentence(s1, r2_s1, o3, family_hint=r2_family),
        make_sentence(s2, r1_s2, o2, family_hint="made of"),
        make_sentence(s2, r2_s2, o4, family_hint=r2_family),
    ]
    random.shuffle(sents)
    story = " ".join(sents)

    # Pick template (cycle 0..3 for perfect 25% mix)
    template, answer_key, task_name = TEMPLATES[template_slot % 4]

    # Build analogy + gold
    if template == "S1:O1::S2:?":
        analogy = f"{s1} is to {o1} as {s2} is to"
        answer  = o2
    elif template == "O1:S1::O2:?":
        analogy = f"{o1} is to {s1} as {o2} is to"
        answer  = s2
    elif template == "S2:O2::S1:?":
        analogy = f"{s2} is to {o2} as {s1} is to"
        answer  = o1
    else:  # "O2:S2::O1:?"
        analogy = f"{o2} is to {s2} as {o1} is to"
        answer  = s1

    meta = {
        "relation_target": R1_NAME,
        "relation_distractor_family": r2_family,
        "surface_r1_s1": r1_s1,
        "surface_r1_s2": r1_s2,
        "surface_r2_s1": r2_s1,
        "surface_r2_s2": r2_s2,
        "r1_synonyms_distinct": (r1_s1 != r1_s2),
        "sentences_shuffled": True,
        "lexicon_condition": lexicon_condition,  # real | mixed | fake
        "s1_noun_type": s1_type,                 # real | fake
        "s2_noun_type": s2_type,                 # real | fake
        "seed": SEED,
        "template": template,
        "task": task_name,                       # human-friendly label for plotting
    }

    return {
        "story": story,
        "prompt": "",
        "analogy": analogy,
        "answer": answer,
        "meta": meta
    }

def main():
    out_path = Path(OUTPUT_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # exact thirds
    n_real  = NUM_EXAMPLES // 3
    n_mixed = NUM_EXAMPLES // 3
    n_fake  = NUM_EXAMPLES - n_real - n_mixed
    conds = ["real"] * n_real + ["mixed"] * n_mixed + ["fake"] * n_fake
    random.shuffle(conds)

    items = []
    t_slot = 0
    for cond in conds:
        ex = build_one_example(cond, template_slot=t_slot)
        items.append(ex)
        t_slot = (t_slot + 1) % 4  # perfect 25% mix

    with out_path.open("w", encoding="utf-8") as f:
        for ex in items:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"Wrote {len(items)} examples → {out_path}")
    c_task = Counter(ex["meta"]["task"] for ex in items)
    c_cond = Counter(ex["meta"]["lexicon_condition"] for ex in items)
    c_r2   = Counter(ex["meta"]["relation_distractor_family"] for ex in items)
    print("Task counts:", dict(c_task))
    print("Condition counts:", dict(c_cond))
    print("Distractor family counts:", dict(c_r2))

if __name__ == "__main__":
    main()