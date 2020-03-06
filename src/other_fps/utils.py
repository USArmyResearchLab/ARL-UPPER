from typing import Callable
from rdkit import Chem, DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
import numpy as np

fp_reg = dict()


def register(func: Callable) -> Callable:
    """Register a function."""

    fp_reg[func.__name__] = func
    return func


@register
def topological(m: list) -> list:
    """Topological fingerprint."""

    return [FingerprintMols.FingerprintMol(x) for x in m]


@register
def maccs(m: list) -> list:
    """MACCS keys fingerprint."""

    return [MACCSkeys.GenMACCSKeys(x) for x in m]


@register
def morgan(m: list) -> list:
    """Morgan fingerprint."""

    return [AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=1024) for x in m]


@register
def avalon(m: list) -> list:
    """Avalon fingerprint."""

    return [GetAvalonFP(x, nBits=1024) for x in m]


def convert_bitvec_to_array(bitvec: list) -> np.ndarray:
    """Convert bit vector fingerprint to numpy array."""

    features = np.zeros(1,)
    DataStructs.ConvertToNumpyArray(bitvec, features)

    return features
