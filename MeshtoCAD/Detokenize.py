import json
from Tokenize import BASE_TOKENS, PARAM_TOKENS, VALUE_TOKENS, STRUCTURE_TOKENS


FLOAT_TOKEN = 310
INT_TOKEN = 311
UUID_TOKEN = 312

MEAN_VAL = 4.5913987159729
STD_VAL = 423.44281005859375

def build_reverse_dict(d):
    return {v: k for k, v in d.items()}

REVERSE_TOKENS = {}
for d in [BASE_TOKENS, PARAM_TOKENS, VALUE_TOKENS, STRUCTURE_TOKENS]:
    REVERSE_TOKENS.update(build_reverse_dict(d))

def denormalize(x):
    return x * STD_VAL + MEAN_VAL

def detokenize(tokens, floats, ints, uuids):
    detok = []
    count = 0
    for t in tokens:
        if t in REVERSE_TOKENS:
            name = REVERSE_TOKENS[t]
            if t == FLOAT_TOKEN or t == INT_TOKEN or t == UUID_TOKEN:
                if count >= len(floats):
                    raise IndexError(f"Missing value for token {name}")
                elif count >= len(ints):
                    raise IndexError(f"Missing value for token {name}")
                elif count >= len(uuids):
                    raise IndexError(f"Missing value for token {name}")
                val = denormalize(floats[count])
                if t == INT_TOKEN:
                    val = int(round(val))
                elif t == UUID_TOKEN:
                    val = uuids[count]
                detok.append(name)
                detok.append(val)
                print(count, len(floats))
                print(name, val)
            else:
                detok.append(name)
        else:
            detok.append(str(t))  # fallback
        count += 1
    return detok

if __name__ == "__main__":
    with open("MeshtoCAD/ML/output/predicted_tokens.json") as f:
        tokens = json.load(f)
    with open("MeshtoCAD/ML/output/predicted_floats.json") as f:
        floats = json.load(f)
    with open("MeshtoCAD/ML/output/predicted_ints.json") as f:
        ints = json.load(f)
    with open("MeshtoCAD/ML/output/predicted_uuids.json") as f:
        uuids = json.load(f)

    detok = detokenize(tokens, floats, ints, uuids)
    print(f"Detokenized sequence: {detok}")

    with open("MeshtoCAD/ML/output/predicted_tokens_detok.json", "w") as f:
        json.dump(detok, f, indent=4)
