import os
import networkx.algorithms.isomorphism as iso
from networkx.readwrite import json_graph
import json
import networkx as nx
from rdkit import Chem

def mol_to_nx(mol):
    '''
    Transform the molecule object of rdkit to networkx.
    '''

    #Initialize graph
    G = nx.Graph()

    # Add atoms
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum(),
                   is_aromatic=atom.GetIsAromatic(),
                   atom_symbol=atom.GetSymbol())

    #Add bonds
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())

    return G

def transform_train_mols(list):
    mols = []
    for mol in list:
        try:
            x = Chem.MolFromSmiles(mol)
            mols.append(x)
        except:
             pass
    return mols

def test_isomorphism(G1,G2):
    G1 = mol_to_nx(G1)
    return nx.is_isomorphic(G1,G2)

def transform_training_data_to_nx(list,save=True,file_name=None):
    # First convert mols to rdkit
    mols_rdkit = transform_train_mols(list)
    mols_in_networkx = []
    for i,m in enumerate(mols_rdkit):
        g = mol_to_nx(m)
        mols_in_networkx.append(g)
    if save:
        all_data = [json_graph.node_link_data(G) for G in mols_in_networkx]
        if file_name is not None:
            name = file_name + '.json'
        else:
            name = 'graphs_training_dataset.json'
        with open(name, "w") as f:
            json.dump(all_data, f)
    return mols_in_networkx

def comparing_gen_train(G_gen,graphs_train):
    not_found = True
    while not_found:
        for i,j in enumerate(graphs_train):
            x = test_isomorphism(G_gen,j)
            if x:
                not_found = False


def loading_graphs(path):
    with open(path, "r") as f:
        all_data = json.load(f)
    molecules = [json_graph.node_link_graph(d) for d in all_data]
    return molecules

def checking_novelty_graphs(gen_graphs,train_graphs=None,path_to_graphs=None):
    # Check if the graphs were already processed otherwise
    if train_graphs is None and path_to_graphs is None and os.path.exists('graphs_training_dataset.json'):
        mols_train = loading_graphs('graphs_training_dataset.json')
    elif train_graphs is None and path_to_graphs is not None and not os.path.exists('graphs_training_dataset.json'):
        mols_train = loading_graphs(path_to_graphs)
    else:
        mols_train = transform_training_data_to_nx(train_graphs,save=True)

    gen_graphs_new = gen_graphs.copy()
    for i,j in enumerate(mols_train):
        p = comparing_gen_train(j,mols_train)
        if p:
            gen_graphs_new.remove(i)

    print('Initial number of novel graphs: {}'.format(len(gen_graphs)))
    print('New number of novel graphs: {}'.format(len(gen_graphs_new)))
    return gen_graphs_new


