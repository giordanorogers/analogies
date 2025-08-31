import argparse
import random
import string
import json
import ast
from transformers import AutoTokenizer
from pathlib import Path

def random_fake_word(
    min_len=5,
    max_len=9,
) -> str:
    """
    Creates a random fake word between the minimum and maximum length.
    """

    # Set local variables
    COMMON_DIGRAPHS = ["th","br","tr","cl","st","gl","pl","pr","cr","gr","dr","fl","fr","sl","sp","sk"]
    VOWELS = "aeiou"
    CONSONANTS = ''.join(sorted(set(string.ascii_lowercase) - set(VOWELS)))

    # Pick a random number between the min and max lendth
    length = random.randint(min_len, max_len)

    # Create a list to store the output word
    output = []

    # Probabilistic flag to use a vowel or not
    use_vowel = random.random() >= 0.65

    i = 0
    while i < length:
        # If we need a consonant and we have more than two characters left,
        # a third of the time add a digraph.
        if not use_vowel and i <= length - 2 and random.random() < 0.33:
            digraph = random.choice(COMMON_DIGRAPHS)
            output.append(digraph)
            i += 2
        # Otherwise add a single vowel or a consonant depending on the value
        # of use_vowel.
        else:
            output.append(random.choice(VOWELS if use_vowel else CONSONANTS))
            i += 1
        # Switch the use_vowel flag
        use_vowel = not use_vowel

    # Construct the word from the output list
    word = "".join(output)[:length]

    # Ensure the word is not entirely made of either vowels or consonants
    if all(char in VOWELS for char in word) or all(char in CONSONANTS for char in word):
        word = list(word)
        mid = len(word) // 2
        word[mid] = random.choice(CONSONANTS if word[mid] in VOWELS else VOWELS)
        word = "".join(word)

    return word.lower()

def pick_two(items):
    """
    Returns two randomly sampled items from a list.
    """
    if len(items) == 1:
        return items[0], items[0]
    return random.sample(items, 2)

def make_sentence(
    subject: str,
    relation: str,
    obj: str,
    family_hint: str
) -> str:
    """
    Builds a sentence from a subject, a relation, and object, and a family hint (for proper grammar).
    """
    subj_phrase = f"The {subject}"
    if family_hint in ("made of", "filled with"):
        return f"{subj_phrase} is {relation} {obj}."
    else:
        return f"{subj_phrase} is {relation} the {obj}."

#def pick_real_subject() -> str:
#    REAL_SUBJECTS = 

def build_nouns_for_story(
    lexicon_condition: str,
    dataset_config: dict,
    r1_family: str,
    r2_family: str
):

    if lexicon_condition == "real":
        s1 = random.choice(dataset_config['real_subjects'])
        objects = []
        for _ in range(2):
            if r1_family == "made of":
                obj = random.choice(dataset_config['real_materials'])
            elif r1_family == "located in":
                obj = random.choice(dataset_config['real_places'])
            elif r1_family == "rests on":
                obj = random.choice(dataset_config['real_surfaces'])
            objects.append(obj)
        o1, o3 = objects[0], objects[1]

        s2 = random.choice([subj for subj in dataset_config['real_subjects'] if subj != s1])
        objects = []

        for _ in range(2):
            if r2_family == "made of":
                obj = random.choice(dataset_config['real_materials'])
            elif r2_family == "located in":
                obj = random.choice(dataset_config['real_places'])
            elif r2_family == "rests on":
                obj = random.choice(dataset_config['real_surfaces'])
            objects.append(obj)
        o2, o4 = objects[0], objects[1]

        return (s1, o1, o3, "real"), (s2, o2, o4, "real")

    if lexicon_condition == "fake":
        s1, s2 = random_fake_word(), random_fake_word()
        o1, o2 = random_fake_word(), random_fake_word()
        o3, o4 = random_fake_word(), random_fake_word()
        return (s1, o1, o3, "fake"), (s2, o2, o4, "fake")

    # otherwise mixed
    real_first = random.random() < 0.5
    print(f"{real_first=}")
    if real_first:
        s1 = random.choice(dataset_config['real_subjects'])
        objects = []
        for i in range(2):
            if r1_family == "made of":
                obj = random.choice(dataset_config['real_materials'])
            elif r1_family == "located in":
                obj = random.choice(dataset_config['real_places'])
            elif r1_family == "rests on":
                obj = random.choice(dataset_config['real_surfaces'])
            objects.append(obj)
        o1, o3 = objects[0], objects[1]

        s2 = random_fake_word()
        o2 = random_fake_word()
        o4 = random_fake_word()
        return (s1, o1, o3, "real"), (s2, o2, o4, "fake")
    else:
        s1 = random_fake_word()
        o1 = random_fake_word()
        o3 = random_fake_word()

        s2 = random.choice(dataset_config['real_subjects'])
        objects = []
        for i in range(2):
            if r1_family == "made of":
                obj = random.choice(dataset_config['real_materials'])
            elif r1_family == "located in":
                obj = random.choice(dataset_config['real_places'])
            elif r1_family == "rests on":
                obj = random.choice(dataset_config['real_surfaces'])
            objects.append(obj)
        o2, o4 = objects[0], objects[1]
        return (s1, o1, o3, "fake"), (s2, o2, o4, "real")

def build_one_example(
    lexicon_condition: str,
    dataset_config: dict,
    template_idx: int
) -> dict:
    # Set the relation families
    r1_family = random.choice(list(dataset_config["relation_families"].keys()))
    r2_family = random.choice([family for family in dataset_config['relation_families'].keys() if family != r1_family])

    # Get the nouns for the story
    (s1, o1, o3, s1_type), (s2, o2, o4, s2_type) = build_nouns_for_story(
        lexicon_condition,
        dataset_config,
        r1_family,
        r2_family
    )
    r11, r12 = pick_two(dataset_config['relation_families'][r1_family])
    r21, r22 = pick_two(dataset_config['relation_families'][r2_family])
    print(f"{r21=}")

    # Store original sentences before shuffling
    facts = {
        "r1_s1": make_sentence(s1, r11, o1, family_hint=r1_family),
        "r2_s1": make_sentence(s1, r21, o2, family_hint=r2_family),
        "r1_s2": make_sentence(s2, r12, o3, family_hint=r1_family),
        "r2_s2": make_sentence(s2, r22, o4, family_hint=r2_family)
    }

    sents = list(facts.values())
    random.shuffle(sents)
    story = " ".join(sents)

    template, answer_key, task_name = ast.literal_eval(dataset_config['templates'][template_idx])

    # Create the analogy prompt and identify the correct answer
    analogy_map = {
        "s1": s1,
        "o1": o1,
        "s2": s2,
        "o3": o3,
    }
    if template == "s1:s2::o1:?":
        analogy = f"{s1} is to {s2} as {o1} is to"
    
    answer = analogy_map[answer_key]

    # Store nouns for later use in source and base prompt construction
    nouns = {"s1":s1, "o1":o1, "s2":s2, "o2":o2, "o3":o3, "o4":o4}

    meta = {
        "r1_family": r1_family,
        "r2_family": r2_family,
        "lexicon_condition": lexicon_condition,
        "template": template,
        "task": task_name
    }

    return {
        "story": story,
        "analogy": analogy,
        "answer": answer,
        "metadata": meta,
        "nouns": nouns,
        "facts": facts
    }

def find_indices_in_context(
    context_str,
    prompt_str,
    tokenized_obj,
    nouns_to_find
):
    """
    Find the last token of a word in a specific context string.
    """
    context_start_char = prompt_str.find(context_str)
    if context_start_char == -1:
        return {
            noun: -1 for noun in nouns_to_find
        }
    
    results = {}
    for noun_key, noun_val in nouns_to_find.items():
        try:
            word_start_in_context = context_str.index(noun_val)
            abs_char_start = context_start_char + word_start_in_context
            end_char = abs_char_start + len(noun_val) - 1
            token_idx = tokenized_obj.char_to_token(end_char)
            results[noun_key] = token_idx if token_idx is not None else -1
        except ValueError:
            results[noun_key] = -1
    return results


def build_prompts_and_find_indices(
    example: dict,
    instruction: str,
    tokenizer: AutoTokenizer
):
    """
    Builds source and base prompts.
    Finds token indices for all nouns.
    """
    # Build the source prompt
    prompt_source = f"{instruction}\n{example['story']}\n{example['analogy']}"

    # Tokenize the source prompt
    tokenized_source = tokenizer(prompt_source)

    # Build the base ("corrupted") prompt
    nouns = example['nouns']
    s1, o1, s2, o2 = nouns['s1'], nouns['o1'], nouns['s2'], nouns['o2']
    o3, o4 = nouns['o3'], nouns['o4']
    template_str = example['metadata']['template']
    print(template_str)

    # Create the analogy cue for the R2 relation
    if template_str == "s1:s2::o1:?":
        base_analogy = f"{s1} is to {s2} as {o2} is to"
    
    prompt_base = f"{instruction}\n{example['story']}\n{base_analogy}"

    # Find token indices in the source prompt
    indices = {}

    # Find nouns in their respective fact sentences
    indices.update(find_indices_in_context(
        example['facts']['r1_s1'],
        prompt_source,
        tokenized_source,
        {'s1_in_r1': s1, 'o1_in_r1': o1}
    ))
    indices.update(find_indices_in_context(
        example['facts']['r2_s1'],
        prompt_source,
        tokenized_source,
        {'s1_in_r2': s1, 'o2_in_r2': o2}
    ))
    indices.update(find_indices_in_context(
        example['facts']['r1_s2'],
        prompt_source,
        tokenized_source,
        {'s2_in_r1': s2, 'o3_in_r1': o3}
    ))
    indices.update(find_indices_in_context(
        example['facts']['r2_s2'],
        prompt_source,
        tokenized_source,
        {'s2_in_r2': s2, 'o4_in_r2': o4}
    ))
    
    # Find nouns in the final analogy cue
    indices.update(find_indices_in_context(
        example['analogy'],
        prompt_source,
        tokenized_source,
        {'s1_in_cue': s1, 'o1_in_cue': o1, 's2_in_cue': s2, 'o2_in_cue': o2}
    ))

    # The analogy site is the final token of the prompt
    indices['analogy_site'] = len(tokenized_source.tokens()) - 1

    # Check that nouns were found in the facts
    required_keys = ['s1_in_r1', 'o1_in_r1', 's1_in_r2', 'o2_in_r2', 's2_in_r1', 'o3_in_r1', 's2_in_r2', 'o4_in_r2']
    if any(indices.get(k, -1) == -1 for k in required_keys):
        print(f"WARNING: A required token was not found in a fact sentence for item with nouns: {example['nouns']}. Skipping.")
        return None, None, None

    return prompt_source, prompt_base, indices



def main(
    args: argparse.Namespace,
    dataset_config: dict,
):
    random.seed(args.random_seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    
    # Create a list to store dataset items
    dataset = []

    template_idx = 0

    # Iterate through the set dataset size
    for i in range(args.dataset_size):

        example = build_one_example(
            args.lexicon_condition,
            dataset_config,
            template_idx
        )

        prompt_source, prompt_base, indices = build_prompts_and_find_indices(
            example,
            dataset_config['instruction'],
            tokenizer
        )

        # Add the new data to the example dict
        example['prompt_source'] = prompt_source
        example['prompt_base'] = prompt_base
        example['token_indices'] = indices

        # We don't need this intermediate data in the final file
        del example['nouns']
        del example['facts']

        dataset.append(example)

    # Write to JSONL
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        for example in dataset:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Wrote {len(dataset)} examples to {args.output_file}")


if __name__ == "__main__":
    # Parsing logic
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='meta-llama/Llama-3.3-70B-Instruct')
    parser.add_argument('--dataset_config', default='../data/dataset_config.json')
    parser.add_argument('--output_file', default='../data/patching_dataset_mixed.jsonl')
    parser.add_argument('--dataset_size', default=1024, type=int)
    parser.add_argument('--lexicon_condition', default='mixed')
    parser.add_argument('--random_seed', default=9001, type=int)
    args = parser.parse_args()

    # Load the dataset configuration file
    with open(args.dataset_config, 'r', encoding='utf-8') as f:
        dataset_config = json.load(f)

    # Run the main method
    main(args, dataset_config)
