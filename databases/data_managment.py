# Basic
import pandas as pd
import numpy as np
# Pytorch and Pytorch Geometric
import torch
from torch_geometric.data import Data
from torch_geometric.data.collate import collate
from torch_geometric.data import InMemoryDataset
import torch.nn.functional as F
# RDkit
from rdkit import Chem, RDLogger
#Others
from tqdm import tqdm
import re
import os.path as osp
import os
import pathlib
import sys
import warnings
#Local
from .abstract_dataset import AbstractDataModule
sys.path.append('..')
from utils.utils import to_dense
from utils.rdkit_functions import build_molecule

class DataManagment:

    # List of permitted atoms mainly CNOS and halogens.
    permitted_list_of_atoms = ['C', 'N', 'O', 'F']
    # List of permitted number of heavy neighbors. Max number of heavy neighbors is 4.
    permitted_list_of_neighbors = [0, 1, 2, 3, 4]
    # List of permitted bond types
    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                                    Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    # List of Beads
    beads_types = ['C1', 'C2', 'C3', 'C4', 'C5', 'N0', 'Na', 'Nd', 'Nda', 'P1', 'P2', 'P3', 'P4', 'P5']
    dg_bead = {'C1':-3.394, 'C2':-3.284, 'C3':-2.930, 'C4':-2.424, 'C5':-1.656,
               'N0':-1.009, 'Na':-0.595, 'Nd':-0.595, 'Nda':-0.595,
               'P1':0.540, 'P2':0.920, 'P3':2.106, 'P4':2.223, 'P5':2.122,
               'SC1':-3.394, 'SC2':-3.284, 'SC3':-2.930,'SC4':-2.424, 'SC5':-1.656,
               'SN0':-1.009, 'SNa':-0.595, 'SNd':-0.595,'SNda':-0.595,
               'SP1':0.540, 'SP2':0.920, 'SP3':2.106, 'SP4':2.223, 'SP5':2.122}

    def __init__(self,
                 name,
                 data_file,
                 ntrain,
                 nvalid,
                 batch_size,
                 seed=42,
                 filter_dataset=True,
                 save_process=True):
        '''

        :param name: name of the dataset
        :param data_file: It is a .csv file with the smiles of the molecules
        :param ntrain: Number of training samples
        :param nvalid: Number of validation samples
        :param batch_size: Size of the batch
        :param seed: Seed for repetition
        :param filter_dataset: Remove molecules that can't be built from the graph
        '''
        self.name = name
        self.data_file = data_file
        self.ntrain = ntrain
        self.nvalid = nvalid
        self.batch_size = batch_size

        if len(data_file) != 1 and type(data_file) == list:
            list_files = [pd.read_csv(i,comment='!') for i in data_file]
            self.data = pd.concat(list_files,ignore_index=True)
        else:
            self.data = pd.read_csv(data_file,comment='!')
        self.ndata = len(self.data)
        self.filter_dataset = filter_dataset
        self.save_process = save_process

        if self.ntrain <1:
            print('The training percentage will be: {}'.format(self.ntrain))
            self.ntrain = int(ntrain*self.ndata)
        if self.nvalid <1:
            print('The validation percentage will be {}'.format(self.nvalid))
            self.nvalid = int(nvalid*self.ndata)

        self.random_state = np.random.RandomState(seed=seed)
        # Create shuffled list of indices
        idx = self.random_state.permutation(np.arange(self.ndata))
        # Store indices of training, validation and test data
        self.idx_train = idx[0:self.ntrain]
        self.idx_valid = idx[self.ntrain:self.ntrain + self.nvalid]
        self.idx_test = idx[self.ntrain + self.nvalid:]

        self.train_data = self.data.iloc[self.idx_train]
        self.valid_data = self.data.iloc[self.idx_valid]
        self.test_data = self.data.iloc[self.idx_test]

        self.beads_labels = {bead: i for i, bead in enumerate(self.beads_types)}
        self.n_beads_types =len(self.beads_labels)
        self.n_nodes_bead = 2

        # Calculate the frequency of beads
        if osp.exists('bead_count.csv'):
            df = pd.read_csv('bead_count.csv')
            self.bead_freqs = df.to_dict('records')[0]
        else:
            print('The bead count file does not exist. This file is for handling class imbalance. It would be created')
            canonic_beads = [self.canonical_reverse(b) for b in self.data['martini_bead']]
            unique, counts = np.unique(canonic_beads, return_counts=True)
            self.bead_freqs = {j: counts[i] for i, j in enumerate(unique)}
            df_bead_freq = pd.DataFrame(self.bead_freqs, index=[0])
            df_bead_freq.to_csv('bead_count.csv',index=False)



    @staticmethod
    def mol2smiles(mol):
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return Chem.MolToSmiles(mol)

    @staticmethod
    def collate(data_list):
        r"""Collates a Python list of :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` objects to the internal
        storage format of :class:`~torch_geometric.data.InMemoryDataset`."""
        if len(data_list) == 1:
            return data_list[0], None

        data, slices, _ = collate(
            data_list[0].__class__,
            data_list=data_list,
            increment=False,
            add_batch=False,
        )

        return data, slices

    @staticmethod
    def get_edges_bead(n, device=None):
        if n < 2:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        src = torch.arange(n - 1, dtype=torch.long, device=device)
        dst = torch.arange(1, n, dtype=torch.long, device=device)
        return torch.cat([torch.stack([src, dst], 0),
                          torch.stack([dst, src], 0)], dim=1)

    ## Function to remove permutational invariance
    @staticmethod
    def canonical_reverse(s, sep='_'):
        parts = s.split(sep)
        rev = parts[::-1]
        canon = parts if parts <= rev else rev
        return sep.join(canon)

    def process_bead(self, mol):
        mol_can = self.canonical_reverse(mol)
        bead_sp = re.split(r'[-,_]+', mol_can)
        num_beads = len(bead_sp)
        is_s = [b[:1] == "S" for b in bead_sp]
        keys = [b[1:] if s else b for b, s in zip(bead_sp, is_s)]
        idx = [self.beads_labels[k] for k in keys]
        dg = [self.dg_bead[b] for b in bead_sp]

        idx_t = torch.tensor(idx, dtype=torch.long)
        is_s_t = torch.tensor(is_s, dtype=torch.long)

        x_type = F.one_hot(idx_t, self.n_beads_types + 1).float()
        x_size = torch.stack((is_s_t, 1 - is_s_t), dim=1).float()
        x_dg = torch.tensor(dg, dtype=torch.float32).unsqueeze(1)
        x = torch.cat((x_type, x_size, x_dg), dim=1)
        # Pad to fixed size and mark padded rows as dummy nodes
        n_nodes = self.n_nodes_bead  # fixed max number of nodes
        feat_dim = x.shape[1]
        x_padded = torch.zeros((n_nodes, feat_dim), dtype=x.dtype, device=x.device)

        n_keep = min(num_beads, n_nodes)
        x_padded[:n_keep] = x[:n_keep]

        if n_keep < n_nodes:
            # Set dummy node type = 1 for padded nodes
            x_padded[n_keep:, : self.n_beads_types + 1] = 0.0
            x_padded[n_keep:, self.n_beads_types] = 1.0
            #DG is also 0
            x_padded[n_keep:, self.n_beads_types + 1:] = 0.0

        x = x_padded

        e_index = self.get_edges_bead(n_keep, device=x.device)
        bead_freq = self.bead_freqs[mol_can]
        return x, e_index, bead_freq


    def process_graphs(self, data,path_to_save,type):
        """
        Process the graphs in the dataset.
        """
        RDLogger.DisableLog('rdApp.*')
        types = {atom: i for i, atom in enumerate(self.permitted_list_of_atoms)}

        bonds = {bond: i for i, bond in enumerate(self.permitted_list_of_bond_types)}

        smiles_list = data['smiles'].values
        beads_list = data['martini_bead'].values

        data_list = []
        smiles_kept = []
        for i, smile in enumerate(tqdm(smiles_list)):
            mol = Chem.MolFromSmiles(smile)
            N = mol.GetNumAtoms()

            type_idx = []
            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])

            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()] + 1]

            if len(row) == 0:
                continue

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type, num_classes=len(bonds) + 1).to(torch.float)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]

            x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()
            y = torch.zeros(size=(1, 0), dtype=torch.float) # Extra data (i.e. time)

            x_b,edge_bead,bead_freq = self.process_bead(beads_list[i])
            bead_graph = beads_list[i]
            # Note Contents of this object are:
            # x: nodes AA (atom types), edge_index: edges AA(bonds), edge_attr: edges types AA
            # y: zeros (to encode time), idx: number of sample.
            # x_beads: nodes CG (bead type, bead size, dgwo(bead)), edge_index_beads: edges beads
            # bead_graph: String representing CG, n_bead: Constant (need for batching)
            # bead_freq: frequency of the beads (require for conditional dropout)
            # original_smiles: string of the AA molecule
            data = BeadData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i,
                        x_beads=x_b, edge_index_beads = edge_bead,bead_graph=bead_graph,
                        n_bead=self.n_nodes_bead,bead_freq=bead_freq,original_smiles=smile)
            if self.filter_dataset:
                # Try to build the molecule again from the graph. If it fails, do not add it to the training set
                dense_data, node_mask = to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
                dense_data = dense_data.mask(node_mask, collapse=True)
                X, E = dense_data.X, dense_data.E

                assert X.size(0) == 1
                atom_types = X[0]
                edge_types = E[0]
                # Always build the molecule with partial charges!
                mol = build_molecule(atom_types, edge_types, self.permitted_list_of_atoms)
                smiles = self.mol2smiles(mol)
                if smiles is not None:
                    try:
                        mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                        if len(mol_frags) == 1:
                            data_list.append(data)
                            smiles_kept.append(smiles)

                    except Chem.rdchem.AtomValenceException:
                        print("Valence error in GetmolFrags")
                    except Chem.rdchem.KekulizeException:
                        print("Can't kekulize molecule")
            else:
                data_list.append(data)

        if self.save_process:
            torch.save(self.collate(data_list),path_to_save)

        if self.filter_dataset:
            smiles_save_path = osp.join(pathlib.Path('').parent, f'new_{type}_{self.name}.smiles')
            # print(smiles_save_path)
            with open(smiles_save_path, 'w') as f:
                f.writelines('%s\n' % s for s in smiles_kept)
            print(f"Number of molecules kept: {len(smiles_kept)} / {len(smiles_list)}")

        return self.collate(data_list)

    def get_dataset(self,folder='processed_data'):
        # First look if the data is already processed
        os.makedirs(folder, exist_ok=True)
        path_train = '{}/{}_train_mols.pt'.format(folder, self.name)
        path_valid = '{}/{}_valid_mols.pt'.format(folder, self.name)
        path_test = '{}/{}_test_mols.pt'.format(folder, self.name)
        if osp.exists(path_train) and osp.exists(path_valid) and osp.exists(path_test):
            print('The data is already processed, It will be only loaded')
            train_db_smiles = MoleculeDataloader(path_train,self.filter_dataset)
            valid_db_smiles = MoleculeDataloader(path_valid,self.filter_dataset)
            test_db_smiles = MoleculeDataloader(path_test, self.filter_dataset)
            # self.data_processed = True
        else:
            print('The data is not processed, It will be processed now and save in {}. Wait a moment!'.format(folder))
            self.proc_train_smiles = self.process_graphs(self.train_data, path_train, 'train')
            self.proc_valid_smiles = self.process_graphs(self.valid_data, path_valid, 'valid')
            self.proc_test_smiles = self.process_graphs(self.test_data, path_test, 'test')

            #Load the molecules
            train_db_smiles = MoleculeDataloader(self.proc_train_smiles, self.filter_dataset)
            valid_db_smiles = MoleculeDataloader(self.proc_valid_smiles, self.filter_dataset)
            test_db_smiles = MoleculeDataloader(self.proc_test_smiles, self.filter_dataset)

        dataset_AA = {'train': train_db_smiles, 'val': valid_db_smiles, 'test': test_db_smiles}


        self.data_AA = AbstractDataModule(self.name,self.batch_size, 0, dataset_AA)#,weights=self.weights)

        return self.data_AA

class MoleculeDataloader(InMemoryDataset):

    def __init__(self,data_smiles,filter_dataset: bool, transform=None, pre_transform=None, pre_filter=None):
        self.filter_dataset = filter_dataset
        super().__init__('.',transform, pre_transform, pre_filter)
        if type(data_smiles) == str:
            warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`*.")
            self.data, self.slices= torch.load(data_smiles,mmap=True,weights_only=False)
        else:
            self.data, self.slices = data_smiles


class BeadData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_beads':
            # how much to shift bead node indices when batching
            if getattr(self, 'x_beads', None) is not None:
                return self.x_beads.size(0)
            if hasattr(self, 'n_bead'):
                return int(self.n_bead)
            raise ValueError("Need x_beads or n_bead to shift edge_index_beads.")
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'edge_index_beads':
            return 1  # concatenate edge indices along columns
        return super().__cat_dim__(key, value, *args, **kwargs)











