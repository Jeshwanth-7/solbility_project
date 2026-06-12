from rdkit import Chem


COMMON_SMILES_NAMES = [

    "smiles",
    "smile",
    "canonical_smiles",
    "compound_smiles",
    "structure",
    "molecule",
    "mol",
    "inchi_smiles",
    "chem_structure",
    "drug"

]

def is_valid_smiles(value):

    try:

        mol = Chem.MolFromSmiles(
            str(value)
        )

        return mol is not None

    except:

        return False


def detect_smiles_column(df):

    # Step 1:
    # Check common names

    for col in df.columns:

        if col.lower() in COMMON_SMILES_NAMES:

            return col

    # Step 2:
    # Check object columns

    for col in df.columns:

        sample_values = (

            df[col]
            .dropna()
            .astype(str)
            .head(20)
        )

        valid_count = sum(

            is_valid_smiles(v)

            for v in sample_values

        )

        if valid_count >= len(sample_values) * 0.8:

            return col

    raise ValueError(
        "Could not detect SMILES column"
    )