"""SMILES -> feature vectors using RDKit descriptors + Morgan fingerprints."""
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# Polymer SMILES use '*' as the attachment/wildcard atom for the repeat unit.
# RDKit parses '*' as a dummy atom natively, so no cleanup is required for
# standard descriptors; we just need to make sure parsing succeeds.

DESCRIPTOR_FUNCS = {
    "MolWt": Descriptors.MolWt,
    "MolLogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumHDonors": Descriptors.NumHDonors,
    "RingCount": Descriptors.RingCount,
    "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings,
    "FractionCSP3": Descriptors.FractionCSP3,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "NumValenceElectrons": Descriptors.NumValenceElectrons,
    "BalabanJ": Descriptors.BalabanJ,
    "BertzCT": Descriptors.BertzCT,
    "LabuteASA": Descriptors.LabuteASA,
    "NumRadicalElectrons": Descriptors.NumRadicalElectrons,
    "HallKierAlpha": Descriptors.HallKierAlpha,
    "Kappa1": Descriptors.Kappa1,
    "Kappa2": Descriptors.Kappa2,
    "Kappa3": Descriptors.Kappa3,
    "Chi0": Descriptors.Chi0,
    "Chi1": Descriptors.Chi1,
}

MORGAN_BITS = 256
MORGAN_RADIUS = 2


def smiles_to_mol(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    return mol


def featurize_one(smiles: str) -> np.ndarray:
    mol = smiles_to_mol(smiles)
    n_desc = len(DESCRIPTOR_FUNCS)
    if mol is None:
        return np.full(n_desc + MORGAN_BITS, np.nan)

    desc = []
    for name, func in DESCRIPTOR_FUNCS.items():
        try:
            desc.append(func(mol))
        except Exception:
            desc.append(np.nan)

    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_BITS)
    fp_arr = np.array(fp, dtype=np.float32)

    return np.concatenate([np.array(desc, dtype=np.float32), fp_arr])


def featurize(smiles_series: pd.Series) -> pd.DataFrame:
    feats = [featurize_one(s) for s in smiles_series]
    cols = list(DESCRIPTOR_FUNCS.keys()) + [f"fp_{i}" for i in range(MORGAN_BITS)]
    return pd.DataFrame(feats, columns=cols, index=smiles_series.index)
