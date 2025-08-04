import json
import os
import shutil
import random

token_dir = "AutoConstrain/Dataset"

with open("AutoConstrain/Dataset/train_test.json") as f:
    split = json.load(f)

train_names = split["train"]
test_names = split["test"]

random.seed(42)
random.shuffle(train_names)
val_size = int(0.1 * len(train_names))
val_names = train_names[:val_size]
train_names = train_names[val_size:]

def move_files(names, split_dir):
    os.makedirs(split_dir, exist_ok=True)
    for name in names:
        for ext in [".json"]:
            src = os.path.join(token_dir, f"{name}{ext}")
            if os.path.exists(src):
                shutil.move(src, os.path.join(split_dir, f"{name}{ext}"))

move_files(train_names, os.path.join(token_dir, "train"))
move_files(val_names, os.path.join(token_dir, "val"))
move_files(test_names, os.path.join(token_dir, "test"))