from typing import Tuple
from rdkit import Chem
from rdkit.Chem import Draw
import re
import itertools
import numpy as np
import networkx as nx
import logging
import collections


def FindBreakingBonds(cnids: list, bids: list, bts: list, atomic_nums: list) -> list:
    """Returns bond ids to be broken. Check for double/triple bonds;
    if exists, check if heteroatom; if heteroatom, bonds of that C atom
    are not broken."""

    x1s = []
    rmflag = None
    for (i, bt) in enumerate(bts):
        for (j, x) in enumerate(bt):
            if x == Chem.rdchem.BondType.DOUBLE or x == Chem.rdchem.BondType.TRIPLE:
                if atomic_nums[cnids[i][j]] != 6 and atomic_nums[cnids[i][j]] != 1:
                    rmflag = True
                    break
        if not rmflag:
            x1s.append(i)
        rmflag = None

    return [bids[x1] for x1 in x1s]


def FragNeighborBreakingBondTypes(
    neighbor_ids: list, fnids: list, faids: list, bond_type_matrix: list
) -> list:
    """Determine broken bond types between fragments and fragment neighbors."""

    # neighbor ids of fragment neighbors
    nids_of_fnids = [[neighbor_ids[x] for x in y] for y in fnids]

    # atom ids 'bonded' to fragment neighbor
    int_ = [
        [Intersection(x, faids[i])[0] for x in y] for (i, y) in enumerate(nids_of_fnids)
    ]

    return [
        [bond_type_matrix[x[i]][y[i]] for (i, z) in enumerate(x)]
        for (x, y) in zip(fnids, int_)
    ]


def EditFragNeighborIds(fnids: list, bbtps: list) -> list:
    """Remove fragment neighbor ids that are doubly/triply bonded to fragment."""

    # not double/triple bonds
    n23bonds = [
        [
            (x != Chem.rdchem.BondType.DOUBLE and x != Chem.rdchem.BondType.TRIPLE)
            for x in y
        ]
        for y in bbtps
    ]

    # return new fragment neighbor ids
    return [
        [x for (j, x) in enumerate(y) if n23bonds[i][j]] for (i, y) in enumerate(fnids)
    ]


def num_atom_rings_1bond(atom_rings: tuple, bond_rings: tuple, num_atoms: int) -> list:
    """Number of rings each atoms is in.  Only rings sharing at most
    1 bond with neighboring rings are considered."""

    # atom ids of rings that share at most 1 bond with neighboring rings
    atom_rings_1bond = [
        atom_rings[i]
        for (i, y) in enumerate(bond_rings)
        if not any(
            IntersectionBoolean(x, y, 2)
            for x in [z for (j, z) in enumerate(bond_rings) if i != j]
        )
    ]

    return [sum(i in x for x in atom_rings_1bond) for i in range(num_atoms)]


def UniqueElements(x: list) -> list:
    """Returns unique elements of a list (not order preserving)."""

    keys = {}
    for e in x:
        keys[e] = 1
    return list(keys.keys())


def NeighborIDs(neighbor_ids: list, atomic_nums: list, y: list) -> list:
    """Find neighbor ids of a list of atoms (Hs not included)."""

    # neighbor ids
    z = [neighbor_ids[x] for x in y]

    # remove Hs
    return [[x for x in y if atomic_nums[x] != 1] for y in z]


def GetFragments(
    smiles: str,
    mol: Chem.rdchem.Mol,
    neighbor_ids: list,
    atomic_nums: list,
    bond_id_matrix: list,
    bond_type_matrix: list,
) -> Tuple[list, list]:
    """Fragment the molecule with isolated carbons method, see
    Lian and Yalkowsky, JOURNAL OF PHARMACEUTICAL SCIENCES 103:2710-2723."""

    # carbons
    cids = [i for (i, x) in enumerate(atomic_nums) if x == 6]

    # carbon neighbor ids
    cnids = NeighborIDs(neighbor_ids, atomic_nums, cids)

    # bond ids
    bids = [
        [bond_id_matrix[cid][cnid] for cnid in cnids]
        for (cid, cnids) in zip(cids, cnids)
    ]

    # bond types
    bts = [
        [bond_type_matrix[cid][cnid] for cnid in cnids]
        for (cid, cnids) in zip(cids, cnids)
    ]

    # broken bond ids
    bbids = FindBreakingBonds(cnids, bids, bts, atomic_nums)

    # break bonds, get fragments
    try:
        fmol = Chem.FragmentOnBonds(
            mol, UniqueElements(list(itertools.chain.from_iterable(bbids)))
        )
    except:
        fmol = mol
        logging.info("fragmentation exception: %s" % (smiles))

    # draw fragments, debugging only, expensive
    # Draw.MolToFile(fmol,'fmol.png')

    # fragment atom ids
    faids = [list(x) for x in Chem.rdmolops.GetMolFrags(fmol)]

    # fragment smiles
    fsmiles = [Chem.rdmolfiles.MolFragmentToSmiles(fmol, frag) for frag in faids]

    # fragment smarts
    fsmarts = [Chem.rdmolfiles.MolFragmentToSmarts(fmol, frag) for frag in faids]

    return faids, fsmiles, fsmarts


def FragNeighborID(fsmile: str) -> list:
    """End atoms bonded to a fragment."""

    fnid = re.compile(r"(%s|%s)" % ("\d+(?=\*)", "\*[^\]]")).findall(fsmile)
    fnid = fnid if fnid else ["-1"]

    return [int(x) if "*" not in x else 0 for x in fnid]


def FragNeighborIDs(fsmiles: list) -> list:
    """End atoms bonded to fragments."""

    fnids = list(map(FragNeighborID, fsmiles))

    return [x if (-1 not in x) else [] for x in fnids]


def BondedFragNeighborIDs(true_faids: list, fnids: list) -> list:
    """Neighbor fragment ids (not atom ids)."""

    return [[k for (k, x) in enumerate(true_faids) for j in i if j in x] for i in fnids]


def NumHybridizationType(htype: Chem.rdchem.HybridizationType, fnhybrds: list) -> list:
    """Number of specified hybridization type for each fragment."""

    return [sum(x == htype for x in fnhybrd) for fnhybrd in fnhybrds]


def Intersection(x: list, y: list) -> list:
    """Elements that match between two lists."""

    return list(set(x) & set(y))


def IntersectionBoolean(x: list, y: list, z: int) -> bool:
    """Returns whether or not two lists overlap with at least z common elements."""

    return len(set(x) & set(y)) >= z


def FindIdsWithHtype(
    fids: list, fnids: list, fnhybrds: list, htype: Chem.rdchem.HybridizationType
) -> list:
    """Find fragment neighbor ids with htype."""

    fnhybrds_in_fids = [fnhybrds[x] for x in fids]
    fnids_in_fids = [fnids[x] for x in fids]
    hids = []
    x1 = 0
    for x in fnhybrds_in_fids:
        x2 = 0
        for y in x:
            if y == htype:
                hids.append(fnids_in_fids[x1][x2])
            x2 += 1
        x1 += 1
    return hids


def AromaticRings(atom_ids_in_rings: list, bond_type_matrix: list) -> list:
    """Return if bonds in rings are aromatic."""

    # atom ids in rings
    atom_ids_in_rings = [np.array(x) for x in atom_ids_in_rings]

    return [
        [
            (bond_type_matrix[int(x)][int(y)] == Chem.rdchem.BondType.AROMATIC)
            for (x, y) in zip(z, z.take(range(1, len(z) + 1), mode="wrap"))
        ]
        for z in atom_ids_in_rings
    ]


def TrueFragAtomIDs(num_atoms: int, faids: list) -> list:
    """Remove dummy atom ids from fragments."""

    return [[x for x in y if x < num_atoms] for y in faids]


def FindCentralCarbonsOfBiphenyl(
    biphenyl_substructs: list,
    neighbor_ids: list,
    atomic_nums: list,
    bond_matrix: list,
    bond_type_matrix: list,
) -> list:
    """Find central carbons of biphenyl substructures."""

    # find one of the central carbons in biphenyl substructures
    cc = []
    for z in biphenyl_substructs:
        for (x, y) in zip(z, z.take(range(1, len(z) + 1), mode="wrap")):
            if not bond_matrix[int(x)][int(y)]:
                cc.append(int(y))
                break

    # find carbon that is singly bonded - other central carbon
    ccs = []
    for (i, y) in enumerate(NeighborIDs(neighbor_ids, atomic_nums, cc)):
        for x in y:
            if bond_type_matrix[cc[i]][x] == Chem.rdchem.BondType.SINGLE:
                ccs.append([cc[i], x])
                break
    return ccs


def Flatten(x: list) -> list:
    """Flatten a list."""

    return list(itertools.chain.from_iterable(x))


def RemoveElements(x: list, y: list) -> list:
    """Remove elements (y) from a list (x)."""

    for e in y:
        x.remove(e)
    return x


def Graph(x: tuple) -> nx.classes.graph.Graph:
    """Make graph structure from atom ids.  Used to find independent ring systems."""

    # initialize graph
    graph = nx.Graph()

    # add nodes and edges
    for part in x:
        graph.add_nodes_from(part)
        graph.add_edges_from(zip(part[:-1], part[1:]))
    return graph


def NumIndRings(x: tuple) -> int:
    """Number of independent single, fused, or conjugated rings."""

    return len(list(nx.connected_components(Graph(x))))


def ReduceFsmarts(fsmarts: list) -> list:
    """Rewrite fragment smarts."""

    return [re.sub(r"\d+\#", "#", x) for x in fsmarts]


def EndLabels(fnbbtps: list) -> list:
    """End label of group.

    -       : bonded to one neighbor and btype = single
    =       : one neighbor is bonded with btype = double
    tri-    : one neighbor is bonded with btype = triple
    allenic : allenic atom, two neighbors are bonded with btype = double"""

    l = ["" for x in fnbbtps]
    for (i, x) in enumerate(fnbbtps):
        if len(x) == 1 and x.count(Chem.rdchem.BondType.SINGLE) == 1:
            l[i] = "-"
            continue
        if x.count(Chem.rdchem.BondType.DOUBLE) == 1:
            l[i] = "="
            continue
        if x.count(Chem.rdchem.BondType.TRIPLE) == 1:
            l[i] = "tri-"
            continue
        if x.count(Chem.rdchem.BondType.DOUBLE) == 2:
            l[i] = "allenic-"

    return l


def FragAtomBondTypeWithSp2(
    fnhybrds: list,
    fnids: list,
    neighbor_ids: list,
    atomic_nums: list,
    faids: list,
    bond_type_matrix: list,
) -> list:
    """Bond type between fragment atom and neighboring sp2 atom."""

    # fragment ids bonded to one sp2 atom
    fids = [
        i
        for i, x in enumerate(
            NumHybridizationType(Chem.rdchem.HybridizationType.SP2, fnhybrds)
        )
        if x == 1
    ]

    # atom id in fragments corresponding to the sp2 atom
    sp2ids = FindIdsWithHtype(fids, fnids, fnhybrds, Chem.rdchem.HybridizationType.SP2)

    # neighbor atom ids of sp2 atoms
    sp2nids = NeighborIDs(neighbor_ids, atomic_nums, sp2ids)

    # intersection between sp2nids and atom ids in fragments with one sp2 atom
    faid = list(
        itertools.chain.from_iterable(
            [Intersection(x, y) for (x, y) in zip([faids[x] for x in fids], sp2nids)]
        )
    )

    # bond type fragment atom and sp2 atom
    bts = [bond_type_matrix[x][y] for (x, y) in zip(sp2ids, faid)]

    # generate list with bond types for each fragment, zero for fragments without one sp2 atom
    afbts = [0] * len(fnhybrds)
    for (x, y) in zip(fids, bts):
        afbts[x] = y

    return afbts


symm_rules: dict = {
    2: {
        1: {
            Chem.rdchem.HybridizationType.SP: 2,
            Chem.rdchem.HybridizationType.SP2: 2,
            Chem.rdchem.HybridizationType.SP3: 2,
        },
        2: {
            Chem.rdchem.HybridizationType.SP: 1,
            Chem.rdchem.HybridizationType.SP2: 1,
            Chem.rdchem.HybridizationType.SP3: 1,
        },
    },
    3: {
        1: {Chem.rdchem.HybridizationType.SP2: 6, Chem.rdchem.HybridizationType.SP3: 3},
        2: {Chem.rdchem.HybridizationType.SP2: 2, Chem.rdchem.HybridizationType.SP3: 1},
        3: {Chem.rdchem.HybridizationType.SP2: 1, Chem.rdchem.HybridizationType.SP3: 1},
    },
    4: {
        1: {Chem.rdchem.HybridizationType.SP3: 12},
        2: {Chem.rdchem.HybridizationType.SP3: 0},
        3: {Chem.rdchem.HybridizationType.SP3: 1},
        4: {Chem.rdchem.HybridizationType.SP3: 1},
    },
}


def Symm(
    smiles: str,
    num_attached_atoms: int,
    num_attached_types: int,
    center_hybrid: Chem.rdchem.HybridizationType,
    count_rankings: collections.Counter,
) -> int:
    """Molecular symmetry."""

    try:
        symm = symm_rules[num_attached_atoms][num_attached_types][center_hybrid]
    except:
        logging.warning("symmetry exception: {}".format(smiles))
        symm = np.nan

    # special case
    if symm == 0:
        vals = list(count_rankings.values())
        symm = 3 if (vals == [1, 3] or vals == [3, 1]) else 2

    return symm


def DataReduction(y: dict, group_labels: list) -> None:
    """Remove superfluous data for single molecule."""

    for l in group_labels:
        y[l] = list(itertools.compress(zip(y["fsmarts"], range(y["num_frags"])), y[l]))


def NFragBadIndices(d: np.ndarray, group_labels: list, smiles: list) -> None:
    """Indices of compounds that do not have consistent number of fragments."""

    def NFragCheck(y: dict) -> bool:
        """Check number of fragments and group contributions are consistent."""

        num_frags = 0
        for l in group_labels:
            num_frags += len(y[l])

        return num_frags != y["num_frags"]

    x = list(map(NFragCheck, d))

    indices = list(itertools.compress(range(len(x)), x))

    logging.info(
        "indices of molecules with inconsistent number of fragments:\n{}".format(
            indices
        )
    )
    logging.info("and their smiles:\n{}".format([smiles[x] for x in indices]))


def UniqueGroups(d: np.ndarray, num_mol: int, group_labels: list) -> list:
    """Unique fragments for each environmental group."""

    # fragments for each group
    groups = [[d[i][j] for i in range(num_mol)] for j in group_labels]

    # eliminate fragment ids
    groups = [[x[0] for x in Flatten(y)] for y in groups]

    return [UniqueElements(x) for x in groups]


def UniqueLabelIndices(flabels: list) -> list:
    """Indices of unique fingerprint labels."""

    sort_ = [sorted(x) for x in flabels]
    tuple_ = [tuple(x) for x in sort_]
    unique_labels = [list(x) for x in sorted(set(tuple_), key=tuple_.index)]

    return [[i for (i, x) in enumerate(sort_) if x == y] for y in unique_labels]


def UniqueLabels(flabels: list, indices: list) -> list:
    """Unique fingerprint labels."""

    return [flabels[x[0]] for x in indices]


def UniqueFingerprint(indices: list, fingerprint: np.ndarray) -> np.ndarray:
    """Reduce fingerprint according to unique labels."""

    fp = np.zeros((fingerprint.shape[0], len(indices)))
    for (j, x) in enumerate(indices):
        fp[:, j] = np.sum(fingerprint[:, x], axis=1)

    return fp


def UniqueLabelsAndFingerprint(
    flabels: list, fingerprint: np.ndarray
) -> Tuple[list, np.ndarray]:
    """Reduced labels and fingerprint."""

    uli = UniqueLabelIndices(flabels)
    ul = UniqueLabels(flabels, uli)
    fp = UniqueFingerprint(uli, fingerprint)

    return ul, fp


def CountGroups(fingerprint_groups: list, group_labels: list, d: dict) -> list:
    """Count groups for fingerprint."""

    return [
        [[x[0] for x in d[y]].count(z) for z in fingerprint_groups[i]]
        for (i, y) in enumerate(group_labels)
    ]


def Concat(x: list, y: list) -> list:
    """Concatenate groups and singles in fingerprint."""

    return x + y


def MakeFingerprint(
    fingerprint_groups: list, labels: dict, d: np.ndarray, num_mol: int
) -> np.ndarray:
    """Make fingerprint."""

    # count groups and make fingerprint
    fp_groups = [
        Flatten(CountGroups(fingerprint_groups, labels["groups"], d[:, 0][i]))
        for i in range(num_mol)
    ]

    # reduce singles to requested
    fp_singles = [[d[:, 1][i][j] for j in labels["singles"]] for i in range(num_mol)]

    # concat groups and singles
    return np.array(list(map(Concat, fp_groups, fp_singles)))


def ReduceMultiCount(d: dict) -> None:
    """Ensure each fragment belongs to one environmental group.

    Falsify Y, Z when YZ true
    Falsify YY, Z when YYZ true
    Falsify YYY, Z when YYYZ true
    Falsify RG when AR true
    ..."""

    def TrueIndices(group: str) -> list:
        """Return True indices."""

        x = d[group]

        return list(itertools.compress(range(len(x)), x))

    def ReplaceTrue(replace_group: list, actual_group: list) -> None:
        """Replace True elements with False to avoid overcounting fragment contribution."""

        replace_indices = list(map(TrueIndices, replace_group))
        actual_indices = list(map(TrueIndices, actual_group))

        for actual_index in actual_indices:
            for (group, replace_index) in zip(replace_group, replace_indices):
                int_ = Intersection(replace_index, actual_index)
                for x in int_:
                    d[group][x] = False

    replace_groups = [
        ["Y", "Z"],
        ["YY", "Z"],
        ["YYY", "Z"],
        ["RG"],
        ["X", "Y", "YY", "YYY", "YYYY", "YYYYY", "Z", "ZZ", "YZ", "YYZ"],
        ["RG", "AR"],
        ["AR", "BR2", "BR3", "FU"],
        ["RG", "AR"],
    ]
    actual_groups = [
        ["YZ"],
        ["YYZ"],
        ["YYYZ"],
        ["AR"],
        ["RG", "AR"],
        ["BR2", "BR3"],
        ["BIP"],
        ["FU"],
    ]

    list(map(ReplaceTrue, replace_groups, actual_groups))


def RewriteFsmarts(d: dict) -> None:
    """Rewrite fsmarts to 'fsmiles unique' fsmarts."""

    def FsmartsDict(d: dict) -> dict:
        """Dict of original fsmarts to 'fsmiles unique' fsmarts."""

        # unique smarts in dataset, mols
        fsmarts = UniqueElements(Flatten([x[0]["fsmarts"] for x in d]))
        fmols = [Chem.MolFromSmarts(x) for x in fsmarts]

        # smiles, not necessarily unique
        fsmiles = [Chem.MolToSmiles(x) for x in fmols]

        # dict: original fsmarts to 'fsmiles unique' fsmarts
        dict_ = collections.defaultdict(lambda: len(dict_))
        fsmarts_dict = {}
        for (i, x) in enumerate(fsmarts):
            fsmarts_dict[x] = fsmarts[dict_[fsmiles[i]]]

        return fsmarts_dict

    fsmarts_dict = FsmartsDict(d)

    # rewrite fsmarts
    for (i, y) in enumerate(d):
        d[i][0]["fsmarts"] = [fsmarts_dict[x] for x in y[0]["fsmarts"]]
