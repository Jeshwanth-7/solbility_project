import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize

from mordred import Calculator, descriptors

from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')
import warnings

warnings.filterwarnings("ignore")

import warnings

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning
)


# Global objects

remover = SaltRemover.SaltRemover()

largest_fragment_chooser = (
    rdMolStandardize.LargestFragmentChooser()
)

calc = Calculator(
    descriptors,
    ignore_3D=True
)


def clean_molecule(smiles):

    try:

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return None

        mol = remover.StripMol(mol)

        mol = largest_fragment_chooser.choose(mol)

        Chem.SanitizeMol(mol)

        return mol

    except:
        return None


def generate_descriptors(mol):

    desc_df = calc.pandas(
        [mol],
        nproc=1,
        quiet=True
    )

    desc_df = desc_df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Match training notebook column names
    desc_df.columns = [
        f"DESC_{col}"
        for col in desc_df.columns
    ]

    return desc_df


def generate_fingerprint(mol):

    fp = list(

        AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=2,
            nBits=1024
        )

    )

    fp_df = pd.DataFrame([fp])

    fp_df.columns = [

        f"FPMorgan_{i}"

        for i in range(1024)

    ]

    return fp_df


def generate_features(smiles):

    mol = clean_molecule(smiles)

    if mol is None:
        return None

    descriptor_df = generate_descriptors(mol)

    fingerprint_df = generate_fingerprint(mol)

    feature_df = pd.concat(
        [descriptor_df, fingerprint_df],
        axis=1
    )

    return feature_df