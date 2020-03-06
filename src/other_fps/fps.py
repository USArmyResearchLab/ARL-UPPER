from rdkit import Chem
import pandas as pd
from .utils import fp_reg, convert_bitvec_to_array


class OtherFingerprint(object):
    """Other fingerprints besides UPPER."""

    def __init__(self, smiless: pd.core.series.Series, fp_type: str) -> None:
        """Initialize class attributes for making fingerprint."""

        # convert smiles to rdkit molecule
        m = [Chem.MolFromSmiles(x) for x in smiless]

        # construct bit vector
        bitvec = fp_reg[fp_type](m)

        # rewrite bitvec to array
        self.fp = list(map(convert_bitvec_to_array, bitvec))

    def output_to_csv(self, fp_path: str = "fingerprint.csv") -> None:
        """Output fingerprint to csv."""

        # dataframe
        df = pd.DataFrame(data=self.fp)

        # save to csv
        df.to_csv(path_or_buf=fp_path, index=None, header=True)

    def output_darray(self, d_path: str = "d.npy") -> None:

        pass
