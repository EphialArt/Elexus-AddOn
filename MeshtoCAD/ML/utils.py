from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch
import torch

MAX_SEQ_LEN = 47000
EOS_TOKEN = 501  

def collate_fn(batch):
    meshes = [item for item in batch if item is not None and len(item.tokens) <= MAX_SEQ_LEN]
    if len(meshes) == 0:
        return None

    tokens = [item.tokens.tolist() + [EOS_TOKEN] for item in meshes]
    values = [item.value_tensor.tolist() + [0.0] for item in meshes]

    print("values types:", [type(v) for v in values])
    tokens_padded = pad_sequence([torch.tensor(t) for t in tokens], batch_first=True, padding_value=500)
    values_padded = pad_sequence([torch.tensor(v) for v in values], batch_first=True, padding_value=-9999999999.0)

    for mesh in meshes:
        if hasattr(mesh, 'tokens'):
            del mesh.tokens
        if hasattr(mesh, 'values'):
            del mesh.values

    mesh_batch = Batch.from_data_list(meshes)
    mesh_batch.tokens_padded = tokens_padded
    mesh_batch.values_padded = values_padded
    return mesh_batch