from typing import Tuple
from .descripts import Descriptors
from src.upper import utils as ut
import numpy as np
import functools
import pandas as pd
import pickle


def ComputeDescripts(smiles: str, index: int) -> Tuple[dict, dict]:
    """UPPER descriptors for a single molecule."""

    # smiles
    print("{}: {}".format(index, smiles))

    # UPPER descriptors
    d = Descriptors(smiles)

    # intialize groups/singles dictionary
    groups: dict = {}
    singles: dict = {}

    # smiles, number of frags, frags, frag neighbor ids
    groups["smiles"] = smiles
    groups["num_frags"] = d.num_frags
    groups["fsmarts"] = d.fsmarts
    groups["frag_neighbor_ids"] = d.bfnids

    # group-constitutive descriptors
    groups["X"] = d.X()
    groups["Y"] = d.Y()
    groups["YY"] = d.YY()
    groups["YYY"] = d.YYY()
    groups["YYYY"] = d.YYYY()
    groups["YYYYY"] = d.YYYYY()
    groups["Z"] = d.Z()
    groups["ZZ"] = d.ZZ()
    groups["YZ"] = d.YZ()
    groups["YYZ"] = d.YYZ()
    groups["YYYZ"] = d.YYYZ()
    groups[""] = d.L()
    groups["RG"] = d.RG_AR("aliphatic")
    groups["AR"] = d.RG_AR("aromatic")
    groups["BR2"] = d.BR2_BR3(2)
    groups["BR3"] = d.BR2_BR3(3)
    groups["FU"] = d.FU()
    groups["BIP"] = d.BIP()

    # single value and geometrical descriptors
    singles["2&6"] = d.TS()
    singles["\u03A6"] = d.PHI()
    singles["\u03C3"] = d.SYMM()
    singles["\u03B5\u00B2\u1d30"] = d.ECC_2d()
    singles["\u03B5\u00B3\u1d30"], singles["\U0001D45E\u00B3\u1d30"] = d.ECC_ASP_3d()
    singles["\U0001D464"] = d.WIENER()
    singles["\U0001D45A"] = d.MW()

    return groups, singles


class UpperFingerprint(object):
    """Upper fingerprint over dataset of molecules."""

    def __init__(self, smiless: pd.core.series.Series, labels: dict) -> None:
        """Initialize class attributes for making fingerprint."""

        # number of molecules
        num_mol = len(smiless)

        # UPPER descriptors for dataset
        self.d = np.array(list(map(ComputeDescripts, smiless, range(num_mol))))

        # rewrite fsmarts to 'fsmiles unique' fsmarts
        # ut.RewriteFsmarts(self.d)

        # reduce multicount
        list(map(ut.ReduceMultiCount, self.d[:, 0]))

        # reduce data
        list(
            map(
                functools.partial(ut.DataReduction, group_labels=labels["groups"]),
                self.d[:, 0],
            )
        )

        # consistency check number of fragments
        ut.NFragBadIndices(self.d[:, 0], labels["groups"], smiless)

        # fingerprint groups
        fp_groups = ut.UniqueGroups(self.d[:, 0], num_mol, labels["groups"])

        # fingerprint labels
        self.fp_labels = (
            ut.Flatten(
                [
                    [x + y for y in fp_groups[i]]
                    for (i, x) in enumerate(labels["groups"])
                ]
            )
            + labels["singles"]
        )

        # make fingerprint
        self.fp = ut.MakeFingerprint(fp_groups, labels, self.d, num_mol)

        # reduce fingerprint to unique labels
        self.fp_labels, self.fp = ut.UniqueLabelsAndFingerprint(self.fp_labels, self.fp)

    def output_to_csv(self, fp_path: str = "upper_fingerprint.csv") -> None:
        """Output fingerprint to csv."""

        # columns of csv file
        data = {x: y for (x, y) in zip(self.fp_labels, np.transpose(self.fp))}

        # dataframe
        df = pd.DataFrame(data=data, columns=self.fp_labels)

        # save to csv
        df.to_csv(path_or_buf=fp_path, index=None, header=True)

    def output_darray(self, d_path: str = "d.pkl") -> None:
        """Output data array containing UPPER info including graph."""

        pickle.dump(self.d, open(d_path, "wb"))
