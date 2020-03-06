from typing import Tuple
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors3D, AllChem
from src.upper import utils as ut
import numpy as np
import itertools
from collections import Counter
import logging


class Descriptors(object):
    """Group-constitutive (enthalpic) and geometrical (entropic) descriptors in UPPER."""

    def __init__(self, smiles: str):
        """Initialize class attributes for computing descriptors."""

        ### mol info ###

        # smiles
        self.smiles = smiles

        # construct mol class
        self.mol_woHs = Chem.MolFromSmiles(smiles)

        # add hydrogens explicitly
        self.mol = Chem.AddHs(self.mol_woHs)

        # number of atoms
        self.num_atoms_woHs = self.mol_woHs.GetNumAtoms()
        self.num_atoms = self.mol.GetNumAtoms()

        ### atom info ###

        # atoms
        self.atoms = self.mol.GetAtoms()

        # atomic numbers
        self.atomic_nums = [x.GetAtomicNum() for x in self.atoms]

        # hybridization
        self.hybridization = [x.GetHybridization() for x in self.atoms]

        # aromaticity
        self.atomic_aromaticity = [x.GetIsAromatic() for x in self.atoms]

        ### bond info ###

        # adjacency matrix
        self.adjacency_matrix = Chem.rdmolops.GetAdjacencyMatrix(self.mol_woHs)

        # bond matrix
        self.bond_matrix = [
            [
                self.mol_woHs.GetBondBetweenAtoms(i, j)
                if self.adjacency_matrix[i][j] != 0
                else None
                for (j, x) in enumerate(row)
            ]
            for (i, row) in enumerate(self.adjacency_matrix)
        ]

        # bond id matrix
        self.bond_id_matrix = [
            [
                x.GetIdx() if self.adjacency_matrix[i][j] != 0 else None
                for (j, x) in enumerate(row)
            ]
            for (i, row) in enumerate(self.bond_matrix)
        ]

        # bond type matrix
        self.bond_type_matrix = [
            [
                x.GetBondType() if self.adjacency_matrix[i][j] != 0 else None
                for (j, x) in enumerate(row)
            ]
            for (i, row) in enumerate(self.bond_matrix)
        ]

        ### neighbor info ###

        # neighbors
        self.neighbors = [x.GetNeighbors() for x in self.atoms]

        # neighbor ids
        self.neighbor_ids = [[x.GetIdx() for x in y] for y in self.neighbors]

        # neighbor atomic numbers
        self.neighbor_atomic_nums = [
            [self.atomic_nums[i] for i in x] for x in self.neighbor_ids
        ]

        ### fragment info ###

        # fragment atom ids, fragment smiles
        self.faids, self.fsmiles, self.fsmarts = ut.GetFragments(
            self.smiles,
            self.mol,
            self.neighbor_ids,
            self.atomic_nums,
            self.bond_id_matrix,
            self.bond_type_matrix,
        )

        # number of fragments
        self.num_frags = len(self.faids)

        # true atom ids, removes dummy atoms due to fragmentation
        self.true_faids = ut.TrueFragAtomIDs(self.num_atoms, self.faids)

        # atom ids of fragment neighbors
        self.fnids = ut.FragNeighborIDs(self.fsmiles)

        # bonded fragment neighbor ids (not atoms)
        self.bfnids = ut.BondedFragNeighborIDs(self.true_faids, self.fnids)

        # bond types between fragments and fragment neighbors
        self.fnbbtps = ut.FragNeighborBreakingBondTypes(
            self.neighbor_ids, self.fnids, self.true_faids, self.bond_type_matrix
        )

        # fragment neighbor ids, excluding double/triple bonded neighbors
        self.fnids = ut.EditFragNeighborIds(self.fnids, self.fnbbtps)

        # hybridization of fragment neighbor atoms
        self.fnhybrds = [[self.hybridization[x] for x in y] for y in self.fnids]

        #### ring info ###

        # ring info
        self.ring_info = self.mol.GetRingInfo()

        # number of rings each atom is in
        self.num_atom_rings = [
            self.ring_info.NumAtomRings(i) for (i, x) in enumerate(self.atoms)
        ]

        # atoms in rings
        self.atom_rings = self.ring_info.AtomRings()

        # number of rings each atom is in, only rings sharing 1 bond with neighboring rings considered
        self.num_atom_rings_1bond = ut.num_atom_rings_1bond(
            self.atom_rings, self.ring_info.BondRings(), self.num_atoms
        )

        # atom ids in aliphatic/aromatic rings
        self.atom_ids_in_aliphatic_rings = [
            self.atom_rings[x]
            for (x, y) in enumerate(
                ut.AromaticRings(self.atom_rings, self.bond_type_matrix)
            )
            if False in y
        ]
        self.atom_ids_in_aromatic_rings = [
            self.atom_rings[x]
            for (x, y) in enumerate(
                ut.AromaticRings(self.atom_rings, self.bond_type_matrix)
            )
            if all(y)
        ]

        #### bond types between fragment atoms and sp2 atom ####
        self.afbts = ut.FragAtomBondTypeWithSp2(
            self.fnhybrds,
            self.fnids,
            self.neighbor_ids,
            self.atomic_nums,
            self.faids,
            self.bond_type_matrix,
        )

        #### search for biphenyl substructure ###

        # define biphenyl
        biphenyl = Chem.MolFromSmiles("C1=CC=C(C=C1)C2=CC=CC=C2")

        # find biphenyl substructures
        biphenyl_substructs = [
            np.array(z) for z in self.mol.GetSubstructMatches(biphenyl)
        ]

        # atom ids of central carbons
        self.ccs = ut.FindCentralCarbonsOfBiphenyl(
            biphenyl_substructs,
            self.neighbor_ids,
            self.atomic_nums,
            self.bond_matrix,
            self.bond_type_matrix,
        )

        #### rewrite fragment in group notation ####
        self.fsmarts = ut.ReduceFsmarts(self.fsmarts)

    def X(self) -> list:
        """A group bonded to only sp3 atoms."""

        return [
            False if not y else all(x == Chem.rdchem.HybridizationType.SP3 for x in y)
            for y in self.fnhybrds
        ]

    def Y(self) -> list:
        """A group singly bonded to 1 sp2 atom."""

        return [(x == Chem.rdchem.BondType.SINGLE) for x in self.afbts]

    def YY(self) -> list:
        """A group bonded to 2 sp2 atoms."""

        return [
            (x == 2)
            for x in ut.NumHybridizationType(
                Chem.rdchem.HybridizationType.SP2, self.fnhybrds
            )
        ]

    def YYY(self) -> list:
        """A group bonded to 3 sp2 atoms. AES."""

        return [
            (x == 3)
            for x in ut.NumHybridizationType(
                Chem.rdchem.HybridizationType.SP2, self.fnhybrds
            )
        ]

    def YYYY(self) -> list:
        """A group bonded to 4 sp2 atoms. AES."""

        return [
            (x == 4)
            for x in ut.NumHybridizationType(
                Chem.rdchem.HybridizationType.SP2, self.fnhybrds
            )
        ]

    def YYYYY(self) -> list:
        """A group bonded to 5 sp2 atoms. AES."""

        return [
            (x == 5)
            for x in ut.NumHybridizationType(
                Chem.rdchem.HybridizationType.SP2, self.fnhybrds
            )
        ]

    def Z(self) -> list:
        """A group bonded to 1 sp atom."""

        return [
            (x == 1)
            for x in ut.NumHybridizationType(
                Chem.rdchem.HybridizationType.SP, self.fnhybrds
            )
        ]

    def ZZ(self) -> list:
        """A group bonded to 2 sp atoms. AES."""

        return [
            (x == 2)
            for x in ut.NumHybridizationType(
                Chem.rdchem.HybridizationType.SP, self.fnhybrds
            )
        ]

    def YZ(self) -> list:
        """A Y and Z group."""

        return [
            (x == Chem.rdchem.BondType.SINGLE and y == 1)
            for (x, y) in zip(
                self.afbts,
                ut.NumHybridizationType(
                    Chem.rdchem.HybridizationType.SP, self.fnhybrds
                ),
            )
        ]

    def YYZ(self) -> list:
        """A YY and Z group. AES."""

        return [
            (x == 2 and y == 1)
            for (x, y) in zip(
                ut.NumHybridizationType(
                    Chem.rdchem.HybridizationType.SP2, self.fnhybrds
                ),
                ut.NumHybridizationType(
                    Chem.rdchem.HybridizationType.SP, self.fnhybrds
                ),
            )
        ]

    def YYYZ(self) -> list:
        """A YYY and Z group. AES."""

        return [
            (x == 3 and y == 1)
            for (x, y) in zip(
                ut.NumHybridizationType(
                    Chem.rdchem.HybridizationType.SP2, self.fnhybrds
                ),
                ut.NumHybridizationType(
                    Chem.rdchem.HybridizationType.SP, self.fnhybrds
                ),
            )
        ]

    def L(self) -> list:
        """A group that is isolated or terminates with only double/triple bonds."""

        return [(not x) for x in self.fnids]

    def RG_AR(self, ring_type: str) -> list:
        """A group within an aliphatic/aromatic ring."""

        # number of fragments, end if 1
        if self.num_frags == 1:
            return [False] * self.num_frags

        # atom ids in aliphatic/aromatic rings
        atom_ids_in_rings = (
            self.atom_ids_in_aliphatic_rings
            if ring_type == "aliphatic"
            else self.atom_ids_in_aromatic_rings
        )

        return [
            any(ut.IntersectionBoolean(x, z, 1) for z in atom_ids_in_rings)
            for x in self.true_faids
        ]

    def BR2_BR3(self, num_rings: int) -> list:
        """An aromatic carbon contained in 2/3 rings."""

        # number of rings, end if less than 2/3
        if len(self.atom_rings) < num_rings:
            return [False] * self.num_frags

        # aromatic carbons in 2/3 rings
        return [
            (self.num_atom_rings_1bond[x[0]] == num_rings)
            if (
                len(x) == 1
                and self.atomic_nums[x[0]] == 6
                and self.atomic_aromaticity[x[0]]
            )
            else False
            for x in self.true_faids
        ]

    def FU(self) -> list:
        """An aliphatic bridgehead group. A bridgehead atom is
        shared between rings that share at least two bonds."""

        # number of bridgehead atoms, end if zero
        if rdMolDescriptors.CalcNumBridgeheadAtoms(self.mol) == 0:
            return [False] * self.num_frags

        # pairs of rings that share at least 2 bonds
        fused_ring_ids = [
            i
            for i, (x, y) in enumerate(
                itertools.combinations(self.ring_info.BondRings(), 2)
            )
            if ut.IntersectionBoolean(x, y, 2)
        ]

        # atom ids shared between fused rings
        shared_atomids_in_frings = [
            ut.Intersection(x, y)
            for i, (x, y) in enumerate(
                itertools.combinations(self.ring_info.AtomRings(), 2)
            )
            if i in fused_ring_ids
        ]

        # bridgeheads, identified by number of neighbors, intersection of neighbors and shared atoms in fused rings is one
        bridgeheads = [
            [
                i
                for i in x
                if not ut.IntersectionBoolean(
                    self.neighbor_ids[i], shared_atomids_in_frings[j], 2
                )
            ]
            for (j, x) in enumerate(shared_atomids_in_frings)
        ]

        # unique bridgeheads
        bridgeheads = np.unique(ut.Flatten(bridgeheads))

        return [any(x in y for x in bridgeheads) for y in self.true_faids]

    def BIP(self) -> list:
        """Central carbons of biphenyl rings."""

        # find if central carbons exist, end if not
        if not self.ccs:
            return [False] * self.num_frags

        return [
            any(ut.IntersectionBoolean(x, y, 1) for y in self.ccs)
            for x in self.true_faids
        ]

    def Ortho(self) -> list:
        """Ortho substitutions."""

        pass

    def TS(self, halogen_atomic_nums: list = [9, 17, 35, 53, 85]) -> float:
        """Number of halogen substitutions at 2 and 6 positions of a biphenyl ring systems."""

        # central carbons (ccs) exist, end if not
        if not self.ccs:
            return 0

        # ccs
        ccs = ut.Flatten(self.ccs)

        # cc neighbor ids
        ccnids = ut.Flatten([self.neighbor_ids[x] for x in ccs])

        # remove cc ids
        ccnids = ut.RemoveElements(ccnids, ccs)

        # neighbor of neighbors
        ccnnids = ut.Flatten([self.neighbor_ids[x] for x in ccnids])

        # remove cc
        ccnnids = ut.RemoveElements(ccnnids, ccs)

        # atom ids bonded to 2,6 positions
        ids_bonded_to_26 = [x for x in ccnnids if not self.atoms[x].IsInRing()]

        return float(
            sum(
                [(self.atomic_nums[x] in halogen_atomic_nums) for x in ids_bonded_to_26]
            )
        )

    def PHI(self) -> float:
        """Molecular flexibility.

        Phi = LIN + 0.5(BR + SP2 + RING) - 1 + 0.3ROT*

        LIN: sum of nonring, linear sp3 atoms (CH2, NH, and O)
        BR: sum of nonring, branched sp3 atoms (CH, C, and N)
        SP2: sum of nonring, sp2 atoms (=CH, =C, and =N)
        RING: sum of independent single, fused, or conjugated rings
        ROT*: (LIN - 4), the extra entropy produced by freely rotating linear sp3 atoms"""

        # terminal atom ids
        terminal_ids = [
            i
            for (i, x) in enumerate(self.atoms)
            if (
                sum(j != 1 for j in self.neighbor_atomic_nums[i]) == 1
                and self.atomic_nums[i] != 1
            )
        ]

        # sp3 atom ids
        sp3_ids = [
            i
            for (i, x) in enumerate(self.atoms)
            if (
                i not in terminal_ids
                and self.hybridization[i] == Chem.rdchem.HybridizationType.SP3
                and self.num_atom_rings[i] == 0
            )
        ]

        # linear sp3 atoms
        linsp3 = [
            x for x in sp3_ids if sum(j != 1 for j in self.neighbor_atomic_nums[x]) == 2
        ]

        # branched sp3 atoms
        brsp3 = [x for x in sp3_ids if x not in linsp3]

        # sp2 atom ids
        sp2 = [
            i
            for (i, x) in enumerate(self.atoms)
            if (
                i not in terminal_ids
                and self.hybridization[i] == Chem.rdchem.HybridizationType.SP2
                and self.num_atom_rings[i] == 0
            )
        ]

        # independent rings
        ring = ut.NumIndRings(self.atom_rings)

        # freely rotating linear sp3 atoms
        rot = max(len(linsp3) - 4, 0)

        # molecular flexibility
        return max(
            len(linsp3) + 0.3 * rot + 0.5 * (len(brsp3) + rot + len(sp2) + ring) - 1, 0
        )

    def SYMM(self) -> float:
        """Molecular symmetry.  Method described in
        Walters and Yalkowsky J. Chem. Info. Comput. Sci. 1996, 36, 1015-1017."""

        # canonical ranking
        ranking = list(Chem.rdmolfiles.CanonicalRankAtoms(self.mol, breakTies=False))

        # max ranking
        max_ = max(ranking)

        # center ids
        center_ids = [i for (i, x) in enumerate(ranking) if x == max_]

        # neighbor ids of max indices
        nids_of_max = self.neighbor_ids[center_ids[0]]

        # attached types
        attached_rankings = [ranking[x] for x in nids_of_max]

        # number of each unique attached type
        count_rankings = Counter(attached_rankings)

        # number of attached atoms
        num_attached_atoms = sum(count_rankings.values())

        # number of attached types
        num_attached_types = len(count_rankings.keys())

        # center hybridization
        center_hybrid = self.hybridization[center_ids[0]]

        # symmetry
        symm = ut.Symm(
            self.smiles,
            num_attached_atoms,
            num_attached_types,
            center_hybrid,
            count_rankings,
        )

        return float(len(center_ids) * symm)

    def ECC_2d(self) -> float:
        """Molecular eccentricity.  Number of atoms in aromatic rings plus nearest-neighbor atoms (not H)."""

        # atom ids in aromatic rings
        atom_ids_in_aromatic_rings = np.unique(
            ut.Flatten(self.atom_ids_in_aromatic_rings)
        )

        # neighbor ids
        ring_atom_neighbor_ids = np.unique(
            ut.Flatten([self.neighbor_ids[i] for i in atom_ids_in_aromatic_rings])
        )

        # remove neighbor ids in rings
        ring_atom_neighbor_ids = [
            x for x in ring_atom_neighbor_ids if x not in atom_ids_in_aromatic_rings
        ]

        # atomic nums of neighbors
        neighbor_atomic_nums = [
            self.atomic_nums[i]
            for i in ring_atom_neighbor_ids
            if self.atomic_nums[i] != 1
        ]

        return float(len(atom_ids_in_aromatic_rings) + len(neighbor_atomic_nums))

    def ECC_ASP_3d(self) -> Tuple[float, float]:
        """Eccentricity/asphericity computed from 3D geometry."""

        # embed molecule in space
        AllChem.EmbedMolecule(self.mol)

        # descriptors
        try:
            return (
                Descriptors3D.Eccentricity(self.mol),
                Descriptors3D.Asphericity(self.mol),
            )
        except:
            logging.warning(
                "3D descriptor exception: {}".format(self.smiles), exc_info=True
            )
            return np.nan, np.nan

    def WIENER(self) -> int:
        """Wiener index of chemical graph."""

        return Chem.GetDistanceMatrix(self.mol_woHs).sum() / 2

    def MW(self) -> float:
        """Molecular weight."""

        return round(rdMolDescriptors.CalcExactMolWt(self.mol))
