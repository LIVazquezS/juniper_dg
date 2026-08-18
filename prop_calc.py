from pathlib import Path
import sys
import logging
import numpy as np
import pandas as pd
# Rdkit modules
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFreeSASA

logger = logging.getLogger(__name__)

# Global options
pt = Chem.GetPeriodicTable()
opts = rdFreeSASA.SASAOpts(rdFreeSASA.SASAAlgorithm.ShrakeRupley,
                            rdFreeSASA.SASAClassifier.Protor,
                            1.4)

def pre_proc_mol(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return mol

def gen_conf(mol,random_seed = 42,
    max_opt_iters= 1000):
    smi = Chem.MolToSmiles(mol)
    # 2. Explicit hydrogens are needed for sensible 3D geometry.
    mol = Chem.AddHs(mol)

    # 3. Embed a conformer. Returns 0 on success, -1 on failure.
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        # Random coordinates often rescue hard-to-embed molecules.
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            logger.warning("Embedding failed for SMILES: %s", smi)
            return None

    # 4. Optimize. Optimizers return a STATUS CODE, they don't raise:
    #    0 = converged, 1 = not converged, -1 = no force-field params.
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            status = AllChem.MMFFOptimizeMolecule(mol, maxIters=max_opt_iters)
        elif AllChem.UFFHasAllMoleculeParams(mol):
            logger.info("MMFF params unavailable; using UFF for: %s", smi)
            status = AllChem.UFFOptimizeMolecule(mol, maxIters=max_opt_iters)
        else:
            logger.warning("No force-field parameters available for: %s", smi)
            return None
    except (ValueError, RuntimeError) as exc:
        logger.warning("Optimization raised for %s: %s", smi, exc)
        return None

    if status == -1:
        logger.warning("Force field could not be set up for: %s", smi)
        return None
    return mol

def get_sasa(mol_proc):
    if mol_proc is None:
        return 0.0
    radii = [pt.GetRvdw(a.GetAtomicNum()) for a in mol_proc.GetAtoms()]
    sasa = rdFreeSASA.CalcSASA(mol_proc,radii,opts=opts)
    return sasa

def calculate_properties(mols,sasa_calc=False):
    #This only should take valid molecules
    mweights = []
    logsp = []
    top_p_sa = []
    natoms = []
    sasa_vals = []
    for i, smi in enumerate(mols):
        na = smi.GetNumAtoms()
        pm = Descriptors.MolWt(smi)  # Molecular weight
        logp = Descriptors.MolLogP(smi)  # Partition coefficient
        tp = Descriptors.TPSA(smi)  # Topological Polarized Surface Area
        if sasa_calc:
            mol_ = gen_conf(smi,random_seed=42,max_opt_iters=1000)
            sa = get_sasa(mol_)
            sasa_vals.append(sa)
        mweights.append(pm)
        logsp.append(logp)
        top_p_sa.append(tp)
        natoms.append(na)
    return natoms,mweights, logsp, top_p_sa, sasa_vals

def calculate_gen_prop(file_gen,label,cg_label,sasa_calc=False):
    f = Path(file_gen)
    if not f.exists():
        sys.exit('File {} does not exist'.format(file_gen))

    if cg_label is None:
        cg_label = 'no_bead'

    df = pd.read_csv(file_gen, names=['smiles'])

    # Preprocess mols
    mols_proc = [pre_proc_mol(smi) for smi in df['smiles'].to_list()]
    mols_proc = [x for x in mols_proc if x is not None]

    # Calculate properties from RdKit
    natoms,mw_gen, logsp_gen, top_p_gen, sasa_vals = calculate_properties(mols_proc,sasa_calc=sasa_calc)
    # Calculate 'Approx DG'
    R = 1.9872e-3  # Gas constant in kcal/mol
    T = 300  # Temperature in K
    prefac = -R * T * np.log(10)
    dg_wo = prefac * np.array(logsp_gen)

    # Create dataframe
    df_gen_prop = pd.DataFrame({'smiles':df['smiles'].to_list(),'Natoms':natoms,'MW': mw_gen, 'LogSP': logsp_gen,
                                'TopP': top_p_gen, 'AA_DGwo': dg_wo, 'CG_DGwo':label,
                                'Type': str(cg_label)})
    if sasa_calc:
        df_gen_prop['sasa'] = sasa_vals

    name_out = f'properties_{cg_label}.csv'
    file_save = Path(name_out)
    if file_save.exists():
        stem, suffix = file_save.stem, file_save.suffix  # 'results_<label>', '.csv'
        counter = 1
        while file_save.exists():
            file_save = Path(f'{stem}_{counter}{suffix}')
            counter += 1

    df_gen_prop.to_csv(file_save, index=False)
    return df_gen_prop

