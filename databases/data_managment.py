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


    # List of permitted atoms.
    permitted_list_of_atoms = ['C', 'N', 'O', 'F']
    # List of permitted number of heavy neighbors. Max number of heavy neighbors is 4.
    permitted_list_of_neighbors = [0, 1, 2, 3, 4]
    # List of permitted bond types
    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                                    Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]


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

        This only takes the value of DG_WO_CG as guidance. Save it in a vector guidance.

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


    def process_graphs(self, data,path_to_save,type):
        """
        Process the graphs in the dataset.
        """
        RDLogger.DisableLog('rdApp.*')
        types = {atom: i for i, atom in enumerate(self.permitted_list_of_atoms)}

        bonds = {bond: i for i, bond in enumerate(self.permitted_list_of_bond_types)}

        smiles_list = data['smiles'].values
        beads_list = data['martini_bead'].values
        dg_wo_cg = data['dg_wo_cg'].values

        data_list = []
        smiles_kept = []
        for i, smile in enumerate(tqdm(smiles_list)):
            mol = Chem.MolFromSmiles(smile)
            if mol is None:
                continue

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
            guidance = dg_wo_cg[i]
            bead_graph = beads_list[i]
            # Note Contents of this object are:
            # x: nodes AA (atom types), edge_index: edges AA(bonds), edge_attr: edges types AA
            # y: zeros (to encode time), idx: number of sample.
            # x_beads: nodes CG (bead type, bead size, dgwo(bead)), edge_index_beads: edges beads
            # bead_graph: String representing CG, n_bead: Constant (need for batching)
            # bead_freq: frequency of the beads (require for conditional dropout)
            # original_smiles: string of the AA molecule
   #         data = BeadData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i,
   #                     x_beads=x_b, edge_index_beads = edge_bead,bead_graph=bead_graph,
   #                     n_bead=self.n_nodes_bead,bead_freq=bead_freq,original_smiles=smile)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,y=y, idx=i,guidance=guidance,
                        bead_graph=bead_graph, original_smiles=smile)
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


#class BeadData(Data):
    #def __inc__(self, key, value, *args, **kwargs):
       # if key == 'edge_index_beads':
            # how much to shift bead node indices when batching
          #  if getattr(self, 'x_beads', None) is not None:
         #       return self.x_beads.size(0)
        #    if hasattr(self, 'n_bead'):
       #         return int(self.n_bead)
      #      raise ValueError("Need x_beads or n_bead to shift edge_index_beads.")
     #   return super().__inc__(key, value, *args, **kwargs)

   # def __cat_dim__(self, key, value, *args, **kwargs):
    #    if key == 'edge_index_beads':
    #        return 1  # concatenate edge indices along columns
    #    return super().__cat_dim__(key, value, *args, **kwargs)











