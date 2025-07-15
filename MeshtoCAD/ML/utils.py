from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch

MAX_TOTAL_TOKENS = 70000

def collate_fn(batch):
    broken_files = set()
    try:
        with open("Broken.txt", "r") as f:
            for line in f:
                if line.startswith("Files:"):
                    files = eval(line.split("Files:")[1].strip())
                    broken_files.update(files)
    except FileNotFoundError:
        pass

    batch = [item for item in batch if getattr(item, "mesh_file", None) not in broken_files]

    if len(batch) == 0:
        return None

    selected_meshes = []
    total_tokens = 0

    for item in batch:
        tok_len = len(item.tokens)

        if tok_len > MAX_TOTAL_TOKENS:
            continue

        if total_tokens + tok_len > MAX_TOTAL_TOKENS:
            break

        selected_meshes.append(item)
        total_tokens += tok_len

    if len(selected_meshes) == 0:
        return None  

    tokens = [item.tokens for item in selected_meshes]
    floats = [item.float_tensor for item in selected_meshes]
    ints = [item.int_tensor for item in selected_meshes]
    uuids = [item.uuid_tensor for item in selected_meshes]

    tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=500)
    floats_padded = pad_sequence(floats, batch_first=True, padding_value=-1e6)
    ints_padded = pad_sequence(ints, batch_first=True, padding_value=-1)
    uuids_padded = pad_sequence(uuids, batch_first=True, padding_value=-1)

    for mesh in selected_meshes:
        for attr in ['tokens', 'values', 'value_tensor', 'uuid_ids']:
            if hasattr(mesh, attr):
                delattr(mesh, attr)

    mesh_batch = Batch.from_data_list(selected_meshes)
    mesh_batch.floats_padded = floats_padded
    mesh_batch.tokens_padded = tokens_padded
    mesh_batch.uuids_padded = uuids_padded
    mesh_batch.ints_padded = ints_padded

    return mesh_batch