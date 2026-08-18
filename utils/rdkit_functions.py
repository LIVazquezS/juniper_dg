import numpy as np
import torch
import re
try:
    from rdkit import Chem
    # print("Found rdkit, all good")
except ModuleNotFoundError as e:
    use_rdkit = False
    from warnings import warn
    warn("Didn't find rdkit, this will fail")
    assert use_rdkit, "Didn't find rdkit"
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

allowed_bonds = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1, 'B': 3, 'Al': 3, 'Si': 4, 'P': [3, 5],
                 'S': 4, 'Cl': 1, 'As': 3, 'Br': 1, 'I': 1, 'Hg': [1, 2], 'Bi': [3, 5], 'Se': [2, 4, 6]}
bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                 Chem.rdchem.BondType.AROMATIC]
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}


class BasicMolecularMetrics(object):

    allowed_bonds = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1, 'B': 3, 'Al': 3, 'Si': 4, 'P': [3, 5],
                     'S': 4, 'Cl': 1, 'As': 3, 'Br': 1, 'I': 1, 'Hg': [1, 2], 'Bi': [3, 5], 'Se': [2, 4, 6]}
    bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                 Chem.rdchem.BondType.AROMATIC]
    ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}

    def __init__(self, dataset_info, train_smiles=None,exclude_disconnected=True):
        self.atom_decoder = dataset_info.atom_decoder
        self.dataset_info = dataset_info

        # Retrieve dataset smiles
        self.dataset_smiles_list = train_smiles
        self.exclude_disconnected = exclude_disconnected

    @staticmethod
    def mol2smiles(mol):
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return Chem.MolToSmiles(mol)

    @staticmethod
    def check_valency(mol):
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            return True, None
        except ValueError as e:
            e = str(e)
            p = e.find('#')
            e_sub = e[p:]
            atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
            return False, atomid_valence

    def build_molecule(self,atom_types, edge_types, atom_decoder):
        mol = Chem.RWMol()
        for atom in atom_types:
            a = Chem.Atom(atom_decoder[atom.item()])
            mol.AddAtom(a)
        edge_types = torch.triu(edge_types)
        all_bonds = torch.nonzero(edge_types)

        for i, bond in enumerate(all_bonds):
            if bond[0].item() != bond[1].item():
                mol.AddBond(bond[0].item(), bond[1].item(), bond_dict[edge_types[bond[0], bond[1]].item()])
                # add formal charge to atom: e.g. [O+], [N+], [S+]
                # not support [O-], [N-], [S-], [NH+] etc.
                flag, atomid_valence = check_valency(mol)
                if flag:
                    continue
                else:
                    assert len(atomid_valence) == 2
                    idx = atomid_valence[0]
                    v = atomid_valence[1]
                    atom_name = mol.GetAtomWithIdx(idx).GetSymbol()
                    an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                    print(f"Atom {atom_name} with atom number {an} "
                          f"has a large valence ({v}) than allowed ({self.ATOM_VALENCY[an]})")
                    if an in (7, 8, 16) and (v - self.ATOM_VALENCY[an]) == 1:
                        mol.GetAtomWithIdx(idx).SetFormalCharge(1)
        return mol

    @staticmethod
    def check_mol(m, largest_connected_comp=True) :
        if m is None:
            return None

        sm = Chem.MolToSmiles(m, isomericSmiles=True)
        is_disconnected = "." in sm

        if is_disconnected:
            if not largest_connected_comp:
                return None
            largest_fragment = max(sm.split("."), key=len)
            return Chem.MolFromSmiles(largest_fragment)
        return Chem.MolFromSmiles(sm)

    def test_connectivity(self,mols):
        invalid = 0
        disconnected = 0
        mol_smiles = []
        for _, molecule in enumerate(mols):
            atom_tp, edge_tp = molecule
            mol = self.build_molecule(atom_tp,edge_tp,
                                      atom_decoder=self.atom_decoder)
            smile = self.mol2smiles(mol)
            if smile is not None:
                mol_smiles.append(smile)
                mol_fregs = Chem.rdmolops.GetMolFrags(mol,asMols=True,sanitizeFrags=False)
                if len(mol_fregs) > 1:
                    disconnected += 1
            else:
                invalid += 1
        percent_invalid = invalid/len(mols)
        percent_disconnected = disconnected/len(mols)
        return invalid, disconnected,percent_invalid, percent_disconnected

    def compute_validity(self, generated):
        """ generated: list of couples (positions, atom_types)"""
        valid = []
        num_fragements = []
        all_smiles = []
        for graph in generated:
            atom_types, edge_types = graph
            mol = self.build_molecule(atom_types, edge_types,
                                      self.dataset_info.atom_decoder)
            direct_valid = True if self.check_mol(mol, largest_connected_comp=False) is not None else False
            smiles = self.mol2smiles(mol)
            mol_frags = None
            try:
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                num_fragements.append(len(mol_frags))
            except Chem.rdchem.AtomValenceException:
                print("Valence error in GetmolFrags")
                all_smiles.append(None)
            except Chem.rdchem.KekulizeException:
                print("Can't kekulize molecule")
                all_smiles.append(None)
            except Chem.rdchem.AtomKekulizeException:
                print("Can't kekulize molecule")
                all_smiles.append(None)
            except:
                print('A weird error occured')
            is_disconnected = mol_frags is not None and len(mol_frags) > 1
            if smiles is not None and direct_valid:
                if is_disconnected and not self.exclude_disconnected:
                    print(f'The generated molecule is disconnected with: {len(mol_frags)} fragments \n'+
                          'The largest fragment would be used as molecule \n' +
                          'Notice the molecule has a valid smiles representation')
                    largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                    try:
                        smiles = self.mol2smiles(largest_mol)
                        valid.append(smiles)
                        all_smiles.append(smiles)
                    except:
                        print(f'The largest fragment could not be used as molecule')
                        all_smiles.append(smiles)
                else:
                    if is_disconnected:
                        print('The generated molecule is disconnected with:', len(mol_frags), 'fragments')
                    valid.append(smiles)
                    all_smiles.append(smiles)
            else:
                all_smiles.append(smiles)

        return valid, len(valid) / len(generated), np.array(num_fragements), all_smiles

    @staticmethod
    def compute_uniqueness(valid):
        """ valid: list of SMILES strings."""
        return list(set(valid)), len(set(valid)) / len(valid)

    def compute_novelty(self, unique):
        '''
        #TODO: Add isomorphism test to remove false novelity.

        '''
        num_novel = 0
        novel = []
        if self.dataset_smiles_list is None:
            print("Dataset smiles is None, novelty computation skipped")
            return 1, 1
        for smiles in unique:
            if smiles not in self.dataset_smiles_list:
                novel.append(smiles)
                num_novel += 1
        return novel, num_novel / len(unique)


    def evaluate(self, generated):
        """ generated: list of pairs (positions: n x 3, atom_types: n [int])
            the positions and atom types should already be masked. """
        valid_smiles, p_validity, num_fragments, all_smiles = self.compute_validity(generated)
        nc_mu = num_fragments.mean() if len(num_fragments) > 0 else 0
        nc_min = num_fragments.min() if len(num_fragments) > 0 else 0
        nc_max = num_fragments.max() if len(num_fragments) > 0 else 0
        print(f"Validity over {len(generated)} molecules: {p_validity * 100 :.2f}%")
        print(f"Fragments of {len(generated)} generated molecules:"
              f" min:{nc_min:.2f} mean:{nc_mu:.2f} max:{nc_max:.2f}")

        invalid_con, disconected, per_invalid, per_disconnected = self.test_connectivity(generated)
        print(f"Percentage of invalid generated molecules: {per_invalid * 100 :.2f}%")
        print(f"Percentage of disconnected molecules: {per_disconnected * 100 :.2f}%")

        # relaxed_valid, relaxed_validity = self.compute_relaxed_validity(generated)
        # print(f"Relaxed validity over {len(generated)} molecules: {relaxed_validity * 100 :.2f}%")
        # if relaxed_validity > 0:
        unique, uniqueness = self.compute_uniqueness(valid_smiles)
        print(f"Uniqueness over {len(valid_smiles)} valid molecules: {uniqueness * 100 :.2f}%")

        if self.dataset_smiles_list is not None:
            novel, novelty = self.compute_novelty(unique)
            print(f"Novelty over {len(unique)} unique valid molecules: {novelty * 100 :.2f}%")
        else:
            print('Novelity could not be computed. To calculate it, pass a list of the smiles in the dataset')
            novelty = -1.0
            novel = []

        # else:
            # novelty = -1.0
            # uniqueness = 0.0
            # unique = []
            # novel = []

        dct_metrics = {'n_gen':len(generated),'n_valid':len(valid_smiles),
                       'n_uniq_valid':len(unique), 'n_novelty_valid':len(novel),
                       'n_invalid':invalid_con,'n_disc':disconected,}

        dct_comp = {'min_nc':nc_min,'max_nc':nc_max,'mean_nc':nc_mu}
        list_prop = [p_validity, uniqueness, novelty]
        return (list_prop,dct_comp,unique,novel,dct_metrics,all_smiles)
        # return ([validity, relaxed_validity, uniqueness, novelty], unique,
        #         dict(nc_min=nc_min, nc_max=nc_max, nc_mu=nc_mu), all_smiles)

#
# def build_molecule(atom_types, edge_types, atom_decoder, verbose=False):
#     if verbose:
#         print("building new molecule")
#
#     mol = Chem.RWMol()
#     for atom in atom_types:
#         a = Chem.Atom(atom_decoder[atom.item()])
#         mol.AddAtom(a)
#         if verbose:
#             print("Atom added: ", atom.item(), atom_decoder[atom.item()])
#
#     edge_types = torch.triu(edge_types)
#     all_bonds = torch.nonzero(edge_types)
#     for i, bond in enumerate(all_bonds):
#         if bond[0].item() != bond[1].item():
#             mol.AddBond(bond[0].item(), bond[1].item(), bond_dict[edge_types[bond[0], bond[1]].item()])
#             if verbose:
#                 print("bond added:", bond[0].item(), bond[1].item(), edge_types[bond[0], bond[1]].item(),
#                       bond_dict[edge_types[bond[0], bond[1]].item()] )
#     return mol


def build_molecule(atom_types, edge_types, atom_decoder, verbose=False):
    if verbose:
        print("\nbuilding new molecule")

    mol = Chem.RWMol()
    for atom in atom_types:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)
        if verbose:
            print("Atom added: ", atom.item(), atom_decoder[atom.item()])
    edge_types = torch.triu(edge_types)
    all_bonds = torch.nonzero(edge_types)

    for i, bond in enumerate(all_bonds):
        if bond[0].item() != bond[1].item():
            mol.AddBond(bond[0].item(), bond[1].item(), bond_dict[edge_types[bond[0], bond[1]].item()])
            if verbose:
                print("bond added:", bond[0].item(), bond[1].item(), edge_types[bond[0], bond[1]].item(),
                      bond_dict[edge_types[bond[0], bond[1]].item()])
            # add formal charge to atom: e.g. [O+], [N+], [S+]
            # not support [O-], [N-], [S-], [NH+] etc.
            flag, atomid_valence = check_valency(mol)
            if verbose:
                print("flag, valence", flag, atomid_valence)
            if flag:
                continue
            else:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                if verbose:
                    print("atomic num of atom with a large valence", an)
                if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)
    return mol


# Functions from GDSS
def check_valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find('#')
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
        return False, atomid_valence


def correct_mol(m):
    #Note LIVS 26.2.2026: This function can be modified to fix for charges or other problems.
    # xsm = Chem.MolToSmiles(x, isomericSmiles=True)
    mol = m
    #####
    no_correct = False
    flag, _ = check_valency(mol)
    if flag:
        no_correct = True

    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            assert len(atomid_valence) == 2
            idx = atomid_valence[0]
            v = atomid_valence[1]
            queue = []
            check_idx = 0
            for b in mol.GetAtomWithIdx(idx).GetBonds():
                type = int(b.GetBondType())
                queue.append((b.GetIdx(), type, b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
                if type == 12:
                    check_idx += 1
            queue.sort(key=lambda tup: tup[1], reverse=True)

            if queue[-1][1] == 12:
                return None, no_correct
            elif len(queue) > 0:
                start = queue[check_idx][2]
                end = queue[check_idx][3]
                t = queue[check_idx][1] - 1
                mol.RemoveBond(start, end)
                if t >= 1:
                    mol.AddBond(start, end, bond_dict[t])
    return mol, no_correct



def check_mol(m, largest_connected_comp=True):
    ## Check if there is a radical break
    if m is None:
        return None
    sm = Chem.MolToSmiles(m, isomericSmiles=True)
    if largest_connected_comp and '.' in sm:
        vsm = [(s, len(s)) for s in sm.split('.')]  # 'C.CC.CCc1ccc(N)cc1CCC=O'.split('.')
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        mol = Chem.MolFromSmiles(vsm[0][0])
    else:
        mol = Chem.MolFromSmiles(sm)
    return mol


use_rdkit = True


def check_stability(atom_types, edge_types, dataset_info, debug=False,atom_decoder=None):
    if atom_decoder is None:
        atom_decoder = dataset_info.atom_decoder

    n_bonds = np.zeros(len(atom_types), dtype='int')

    for i in range(len(atom_types)):
        for j in range(i + 1, len(atom_types)):
            n_bonds[i] += abs((edge_types[i, j] + edge_types[j, i])/2)
            n_bonds[j] += abs((edge_types[i, j] + edge_types[j, i])/2)
    n_stable_bonds = 0
    for atom_type, atom_n_bond in zip(atom_types, n_bonds):
        possible_bonds = allowed_bonds[atom_decoder[atom_type]]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == atom_n_bond
        else:
            is_stable = atom_n_bond in possible_bonds
        if not is_stable and debug:
            print("Invalid bonds for molecule %s with %d bonds" % (atom_decoder[atom_type], atom_n_bond))
        n_stable_bonds += int(is_stable)

    molecule_stable = n_stable_bonds == len(atom_types)
    return molecule_stable, n_stable_bonds, len(atom_types)


def compute_molecular_metrics(molecule_list, train_smiles, dataset_info,
                              testing=False,exclude_disconnected=False):
    # Add option to fix disconnected graphs.
    """ molecule_list: (dict) """

    if not dataset_info.remove_h:
        print(f'Analyzing molecule stability...')

        molecule_stable = 0
        nr_stable_bonds = 0
        n_atoms = 0
        n_molecules = len(molecule_list)

        for i, mol in enumerate(molecule_list):
            atom_types, edge_types = mol

            validity_results = check_stability(atom_types, edge_types, dataset_info)

            molecule_stable += int(validity_results[0])
            nr_stable_bonds += int(validity_results[1])
            n_atoms += int(validity_results[2])

        # Validity
        fraction_mol_stable = molecule_stable / float(n_molecules)
        fraction_atm_stable = nr_stable_bonds / float(n_atoms)
        validity_dict = {'mol_stable': fraction_mol_stable, 'atm_stable': fraction_atm_stable}

    else:
        validity_dict = {'mol_stable': -1, 'atm_stable': -1}

    metrics = BasicMolecularMetrics(dataset_info, train_smiles,exclude_disconnected=exclude_disconnected)
    rdkit_metrics,dct_comp,unique,novel,dct_metrics,all_smiles = metrics.evaluate(molecule_list)
    # rdkit_metrics = metrics.evaluate(molecule_list)
    # all_smiles = rdkit_metrics[-1]
    if testing:
        return dct_metrics, unique, novel, all_smiles
    else:
        return validity_dict, rdkit_metrics, all_smiles



