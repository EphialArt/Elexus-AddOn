import json
from Tokenize import BASE_TOKENS, PARAM_TOKENS, VALUE_TOKENS, STRUCTURE_TOKENS

def build_reverse_dict(d):
    return {v: k for k, v in d.items()}

REVERSE_TOKENS = {}
for d in [BASE_TOKENS, PARAM_TOKENS, VALUE_TOKENS, STRUCTURE_TOKENS]:
    REVERSE_TOKENS.update(build_reverse_dict(d))

def detokenize(tokens):
    detok = []
    for t in tokens:
        if t in REVERSE_TOKENS:
            detok.append(REVERSE_TOKENS[t])
        else:
            detok.append(str(t)) 
    return detok

if __name__ == "__main__":
    with open("MeshtoCAD/ML/output/predicted_tokens.json") as f:
        tokens = json.load(f)
    detok = detokenize(tokens)
    print("Detokenized sequence:")
    print(detok)
    with open("MeshtoCAD/ML/output/predicted_tokens_detok.json", "w") as f:
        json.dump(detok, f, indent=4)