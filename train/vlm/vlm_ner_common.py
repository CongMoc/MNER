import json


def load_jsonl(path):
    rows = []
    with open(path, encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_system_prompt(entity_types):
    types_str = ', '.join(entity_types)
    return (
        "You are a named entity recognition assistant. You are given a sentence and its "
        "associated image. Use both to extract all named entities mentioned in the sentence. "
        f"Valid entity types: {types_str}. "
        "Output one entity per line as 'TYPE: entity text', using the entity's exact text "
        "as it appears in the sentence. If there are no entities, output exactly 'None'."
    )


def format_entities(entities):
    if not entities:
        return "None"
    return "\n".join(f"{t}: {text}" for t, text in entities)


def parse_entities(generated_text, valid_types):
    entities = []
    for line in generated_text.strip().splitlines():
        line = line.strip()
        if not line or line == "None":
            continue
        if ':' not in line:
            continue
        etype, etext = line.split(':', 1)
        etype = etype.strip()
        etext = etext.strip()
        if etype in valid_types and etext:
            entities.append((etype, etext))
    return entities


def build_messages(system_prompt, sentence, image_path, entities=None):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": sentence},
        ]},
    ]
    if entities is not None:
        messages.append({"role": "assistant", "content": format_entities(entities)})
    return messages
