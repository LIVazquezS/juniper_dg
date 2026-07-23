import os
import sys
import re
sys.path.append('..')

import torch_geometric.utils
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.data import Batch
from torch_geometric.data import Data
import torch
import torch.nn.functional as F


def mask_nodes_beadgen(X, iteration):
     bs = X.shape[0]
     N = X.shape[1]
     m = X.shape[2]
     mask = torch.zeros(bs, N, m).to(X.device)
     chunk_size = N // 2
     start = (iteration * chunk_size) % N
     end = start + chunk_size
#
     if end <= N:
         mask[:, start:end, :] = 1
     else:
         # Wrap around
         mask[:, start:, :] = 1
         mask[:, :end % N, :] = 1

     new_X = X*mask

     return new_X

def mask_edges_beadgen(E,iteration):
    bs = E.shape[0]
    N = E.shape[1]
    m = E.shape[3]
    # Compute indices for the block
    chunk_size = N // 2
    row_block = (iteration // 2) % 2
    col_block = iteration % 2
    row_start = row_block * chunk_size
    col_start = col_block * chunk_size

    # Create 2D base mask
    mask_2d = torch.zeros(N, N).to(E.device)
    mask_2d[row_start:row_start + chunk_size, col_start:col_start + chunk_size] = 1

    # Expand to (bs, N, N, m)
    mask = mask_2d.view(1, N, N, 1).expand(bs, -1, -1, m)

    new_E = E*mask
    return new_E

def create_folders(args):
    try:
        # os.makedirs('cpt_no_H_y_0')
        os.makedirs('graphs')
        os.makedirs('chains')
    except OSError:
        pass

    try:
        # os.makedirs('cpt_no_H_y_0/' + args.general.name)
        os.makedirs('graphs/' + args.general.name)
        os.makedirs('chains/' + args.general.name)
    except OSError:
        pass


def normalize(X, E, y, norm_values, norm_biases, node_mask):
    X = (X - norm_biases[0]) / norm_values[0]
    E = (E - norm_biases[1]) / norm_values[1]
    y = (y - norm_biases[2]) / norm_values[2]

    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask)


def unnormalize(X, E, y, norm_values, norm_biases, node_mask, collapse=False):
    """
    X : node features
    E : edge features
    y : global features`
    norm_values : [norm value X, norm value E, norm value y]
    norm_biases : same order
    node_mask
    """
    X = (X * norm_values[0] + norm_biases[0])
    E = (E * norm_values[1] + norm_biases[1])
    y = y * norm_values[2] + norm_biases[2]

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask, collapse)

def canonical_reverse(s, sep='_'):
    parts = s.split(sep)
    rev = parts[::-1]
    canon = parts if parts <= rev else rev
    return sep.join(canon)



def get_edges_bead(n, device=None):
    if n < 2:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    src = torch.arange(n - 1, dtype=torch.long, device=device)
    dst = torch.arange(1, n, dtype=torch.long, device=device)
    return torch.cat([torch.stack([src, dst], 0),
                          torch.stack([dst, src], 0)], dim=1)

def build_cg_from_string_ind(mol):
    from databases.data_managment import BeadData
    beads_types = ['C1', 'C2', 'C3', 'C4', 'C5', 'N0', 'Na', 'Nd', 'Nda', 'P1', 'P2', 'P3', 'P4', 'P5']
    dg_bead = {'C1':-3.394, 'C2':-3.284, 'C3':-2.930, 'C4':-2.424, 'C5':-1.656,
               'N0':-1.009, 'Na':-0.595, 'Nd':-0.595, 'Nda':-0.595,
               'P1':0.540, 'P2':0.920, 'P3':2.106, 'P4':2.223, 'P5':2.122,
               'SC1':-3.394, 'SC2':-3.284, 'SC3':-2.930,'SC4':-2.424, 'SC5':-1.656,
               'SN0':-1.009, 'SNa':-0.595, 'SNd':-0.595,'SNda':-0.595,
               'SP1':0.540, 'SP2':0.920, 'SP3':2.106, 'SP4':2.223, 'SP5':2.122}

    beads_labels = {bead: i for i, bead in enumerate(beads_types)}

    mol_can = canonical_reverse(mol)
    bead_sp = re.split(r'[-,_]+', mol_can)
    num_beads = len(bead_sp)
    is_s = [b[:1] == "S" for b in bead_sp]
    keys = [b[1:] if s else b for b, s in zip(bead_sp, is_s)]
    dg = [dg_bead[b] for b in bead_sp]
    # print(keys)
    # print(is_s)
    idx = [beads_labels[k] for k in keys]

    idx_t = torch.tensor(idx, dtype=torch.long)
    is_s_t = torch.tensor(is_s, dtype=torch.long)

    x_type = F.one_hot(idx_t, len(beads_labels)+1).float()
    x_size = torch.stack((is_s_t, 1 - is_s_t), dim=1).float()
    x_dg = torch.tensor(dg, dtype=torch.float32).unsqueeze(1)
    x = torch.cat((x_type, x_size, x_dg), dim=1)
    # Pad to fixed size and mark padded rows as dummy nodes
    n_nodes = 2  # your fixed max number of nodes
    feat_dim = x.shape[1]
    x_padded = torch.zeros((n_nodes, feat_dim), dtype=x.dtype, device=x.device)

    n_keep = min(num_beads, n_nodes)
    x_padded[:n_keep] = x[:n_keep]

    if n_keep < n_nodes:
        # Set dummy node type = 1 for padded nodes
        x_padded[n_keep:, : len(beads_labels) + 1] = 0.0
        x_padded[n_keep:, len(beads_labels)] = 1.0

        # Optional: also make size and dg explicitly 0 for dummy nodes
        x_padded[n_keep:, len(beads_labels) + 1:] = 0.0

    x = x_padded

    e_index = get_edges_bead(n_keep, device=x.device)

    g = BeadData(x_beads=x,edge_index_beads=e_index,n_bead=2)
    return g

def build_cg_from_string_batch(mol,bs):
    mol_conv = build_cg_from_string_ind(mol)
    list_mols = [mol_conv.clone() for _ in range(bs)]
    batch = Batch.from_data_list(list_mols)
    return batch

def to_dense(x, edge_index, edge_attr, batch):
    X, node_mask = to_dense_batch(x=x, batch=batch)
    # node_mask = node_mask.float()
    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)
    # TODO: carefully check if setting node_mask as a bool breaks the continuous case
    max_num_nodes = X.size(1)
    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
    E = encode_no_edge(E)

    return PlaceHolder(X=X, E=E, y=None), node_mask


def encode_no_edge(E):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0
    return E
#
class BeadHolder(dict):
    def __getattr__(self, key):
        return self[key]
    def __setattr__(self, key, value):
        self[key] = value
    def __delattr__(self, key):
        del self[key]

class PlaceHolder:
    def __init__(self, X, E, y,guidance=None):
        self.X = X
        self.E = E
        self.y = y
        self.guidance = guidance

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        if (self.guidance is not None):
            self.guidance = self.guidance.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = - 1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self
