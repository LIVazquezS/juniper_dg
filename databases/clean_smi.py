import numpy as np
from rdkit import Chem
import pandas as pd
import tqdm as tqdm
from rdkit.Chem import AllChem

file_smiles = "QM9-smiles.txt"

def read_smiles_gen(file):
    smiles_read = []
    with open(file,'r') as f:
        for line in f.readlines():
            smiles=line.strip()
            p = Chem.MolFromSmiles(smiles)
            if p is not None:
                smiles_read.append(p)
            else:
                print(smiles)
    print('smiles read by RDKit:', len(smiles_read))
    return smiles_read

def save_smiles(val_u_n,output_name):
    smiles_noh = []
    for i in val_u_n:
        p = Chem.MolToSmiles(i,canonical=True,kekuleSmiles=True)
        if p is not None:
            smiles_noh.append(p)
    dct = {'smiles':smiles_noh}
    df = pd.DataFrame(dct)
    df.to_csv(output_name,index=False)
    return None

x = read_smiles_gen(file_smiles)
save_smiles(x,'qm9_clean.csv')
