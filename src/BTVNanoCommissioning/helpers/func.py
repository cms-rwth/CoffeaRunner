# coding: utf-8

import functools
import logging
from collections import namedtuple
from functools import reduce
from operator import and_, or_

import awkward as ak
import numpy as np
from coffea import processor


logger = logging.getLogger(__name__)


def update(events, collections):
    """Return a shallow copy of events array with some collections swapped out"""
    out = events
    for name, value in collections.items():
        out = ak.with_field(out, value, name)
    return out


def parametrized(dec):
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)

        return repl

    return layer


def df_object_overlap(toclean, cleanagainst, dr=0.4):
    particle_pair = toclean["p4"].cross(cleanagainst["p4"], nested=True)
    return (particle_pair.i0.delta_r(particle_pair.i1)).min() > dr


def nano_object_overlap(toclean, cleanagainst, dr=0.4):
    return ak.all(toclean.metric_table(cleanagainst) > dr, axis=-1)


def df_mask_or(df, masks):
    decision = reduce(or_, (df[mask] for mask in masks))
    return decision


def df_mask_and(df, masks):
    decision = reduce(and_, (df[mask] for mask in masks))
    return decision


def nano_mask_or(events, masks, skipable=()):
    masks = set(map(str, masks))
    if skipable is True:
        skipable = masks
    else:
        skipable = set(map(str, skipable))
    return reduce(
        or_,
        (
            getattr(events, mask)
            for mask in masks
            if mask not in skipable or hasattr(events, mask)
        ),
    )


def nano_mask_and(events, masks):
    decision = reduce(and_, (getattr(events, str(mask)) for mask in masks))
    return decision


def nano_cut(what, *cuts):
    return what[reduce(and_, cuts)]


def reduce_and(*what):
    return reduce(and_, what)


def reduce_or(*what):
    return reduce(or_, what)


@parametrized
def padflat(func, n_particles=1):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        return ak.flatten(ak.fill_none(ak.pad_none(res, n_particles), np.nan))

    return wrapper


def bregcorr(jets):
    return ak.zip(
        {
            "pt": jets.pt * jets.bRegCorr,
            "eta": jets.eta,
            "phi": jets.phi,
            "energy": jets.energy * jets.bRegCorr,
        },
        with_name="PtEtaPhiELorentzVector",
    )


@padflat(1)
def m_bb(jets):
    lead_jets = jets[..., :2].distincts()
    bb = lead_jets.i0 + lead_jets.i1
    m_bb = bb.mass
    return m_bb


def get_ht(jets):
    return ak.sum(jets.pt, axis=-1)


def min_dr_part1_part2(part1, part2, getn=0, fill=np.nan):
    """
    For each particle in part1 returns the minimum
    delta_r between this and each particle in part2
    """
    a, b = ak.unzip(ak.cartesian({"p1": part1, "p2": part2}, nested=True))
    if not hasattr(a, "delta_r"):
        a = make_p4(a)
    if not hasattr(b, "delta_r"):
        b = make_p4(a)
    r = ak.min(a.delta_r(b), axis=-1)
    if 0 < getn:
        r = ak.fill_none(ak.pad_none(r, getn, clip=True), np.nan)
        return tuple(r[:, i] for i in range(getn))
    else:
        return r


def get_metp4(met):
    return ak.zip(
        {
            "pt": met.pt,
            "eta": met.pt * 0,
            "phi": met.phi,
            "mass": met.pt * 0,
        },
        with_name="PtEtaPhiMLorentzVector",
    )


def min_dr(particles):
    di_particles = ak.combinations(
        particles,
        n=2,
        replacement=False,
        axis=1,
        fields=["p1", "p2"],
    )
    return ak.min(
        make_p4(di_particles.p1).delta_r(make_p4(di_particles.p2)),
        axis=-1,
        mask_identity=False,
    )


def min_dphi(particles):
    di_particles = ak.combinations(
        particles,
        n=2,
        replacement=False,
        axis=1,
        fields=["p1", "p2"],
    )
    return ak.min(
        np.abs(make_p4(di_particles.p1).delta_phi(make_p4(di_particles.p2))),
        axis=-1,
        mask_identity=False,
    )


def get_met_ld(jets, leps, met, met_coef=0.6, mht_coef=0.4):
    mht = make_p4(jets).sum() + make_p4(leps).sum()
    return met_coef * met.pt + mht_coef * mht.pt


def get_cone_pt(part, n=1):
    padded_part = ak.pad_none(part, n)
    return [ak.fill_none(padded_part[..., i].cone_pt, np.nan) for i in range(n)]


def make_p4(obj, candidate=False):
    params = ["pt", "eta", "phi", "mass"]
    with_name = "PtEtaPhiMLorentzVector"
    if candidate:
        params.append("charge")
        with_name = "PtEtaPhiMCandidate"
    return ak.zip(
        {p: getattr(obj, p) for p in params},
        with_name=with_name,
    )


def lead_diobj(objs):
    two = objs[:, :2]
    a, b = ak.unzip(
        ak.combinations(
            two,
            n=2,
            replacement=False,
            axis=1,
            fields=["a", "b"],
        )
    )

    # make sure it is a CandidateArray
    if not hasattr(a, "delta_r"):
        a = make_p4(a, candidate=False)
    if not hasattr(b, "delta_r"):
        b = make_p4(b, candidate=False)
    diobj = a + b
    diobj["deltaR"] = a.delta_r(b)
    diobj["deltaPhi"] = a.delta_phi(b)
    return diobj


class chunked:
    def __init__(self, func, chunksize=10000):
        self.func = func
        self.chunksize = chunksize

    def __call__(self, *args, **kwargs):
        lens = set(map(len, args))
        if len(lens) != 1:
            raise ValueError("inconsistent *args len")
        return ak.concatenate(
            [
                self.func(*(a[off : off + self.chunksize] for a in args), **kwargs)
                for off in range(0, max(lens), self.chunksize)
            ]
        )


def linear_fit(x, y, eigen_decomp=False):
    coeff, cov = np.polyfit(x, y, 1, cov="unscaled")
    return linear_func(coeff, cov, eigen_decomp)


def linear_func(coeff, cov, eigen_decomp=False):
    c1, c0 = coeff
    nom = lambda v: c0 + c1 * v
    if eigen_decomp:
        eigenvals, eigenvecs = np.linalg.eig(cov)
        lambda0, lambda1 = np.sqrt(eigenvals)
        v00, v01 = eigenvecs[:, 0]  # 1st eigenvector
        v10, v11 = eigenvecs[:, 1]  # 2nd eigenvector
        var1_down = lambda v: c0 - lambda0 * v00 + (c1 - lambda0 * v01) * v
        var1_up = lambda v: c0 + lambda0 * v00 + (c1 + lambda0 * v01) * v
        var2_down = lambda v: c0 - lambda1 * v10 + (c1 - lambda0 * v11) * v
        var2_up = lambda v: c0 + lambda1 * v10 + (c1 + lambda0 * v11) * v
        return nom, (var1_down, var1_up), (var2_down, var2_up)
    else:
        return nom


def mT(obj1, obj2):
    return np.sqrt(2.0 * obj1.pt * obj2.pt * (1.0 - np.cos(obj1.phi - obj2.phi)))


def flatten(ar):  # flatten awkward into a 1d array to hist
    return ak.flatten(ar, axis=None)


def normalize(val, cut=None):
    if cut is None:
        ar = ak.to_numpy(ak.fill_none(val, np.nan))
        return ar
    else:
        ar = ak.to_numpy(ak.fill_none(val[cut], np.nan))
        return ar


def update(events, collections):
    """Return a shallow copy of events array with some collections swapped out"""
    out = events
    for name, value in collections.items():
        out = ak.with_field(out, value, name)
        if (
            "float" not in str(ak.type(value))
            and "int" not in str(ak.type(value))
            and "bool" not in str(ak.type(value))
        ):
            if name == "Jet":
                out.Jet["pt"] = value.pt
            elif name == "MET":
                out.MET["pt"] = value.pt
                out.MET["phi"] = value.phi
    return out


def num(ar):
    return ak.num(ak.fill_none(ar[~ak.is_none(ar)], 0), axis=0)


def _is_rootcompat(a):
    """Is it a flat or 1-d jagged array?"""
    t = ak.type(a)
    if isinstance(t, ak._ext.ArrayType):
        if isinstance(t.type, ak._ext.PrimitiveType):
            return True
        if isinstance(t.type, ak._ext.ListType) and isinstance(
            t.type.type, ak._ext.PrimitiveType
        ):
            return True
    return False


def uproot_writeable(events, include=["events", "run", "luminosityBlock"]):
    ev = {}
    include = np.array(include)
    no_filter = False

    if len(include) == 1 and include[0] == "*":
        no_filter = False
    for bname in events.fields:
        if not events[bname].fields:
            if not no_filter and bname not in include:
                continue
            ev[bname] = ak.packed(ak.without_parameters(events[bname]))
        else:
            b_nest = {}
            no_filter_nest = False
            if all(np.char.startswith(include, bname) == False):
                continue
            include_nest = [
                i[i.find(bname) + len(bname) + 1 :]
                for i in include
                if i.startswith(bname)
            ]

            if len(include_nest) == 1 and include_nest[0] == "*":
                no_filter_nest = True
            if not no_filter_nest:
                mask_wildcard = np.char.find(include_nest, "*") != -1
                include_nest = np.char.replace(include_nest, "*", "")

            for n in events[bname].fields:
                if not _is_rootcompat(events[bname][n]):
                    continue
                ## make selections to the filter case, keep cross-ref ("Idx")
                if (
                    not no_filter_nest
                    and all(np.char.find(n, include_nest) == -1)
                    and "Idx" not in n
                ):
                    continue
                if (
                    mask_wildcard[np.where(np.char.find(n, include_nest) != -1)]
                    == False
                    and "Idx" not in n
                ):
                    continue
                b_nest[n] = ak.packed(ak.without_parameters(events[bname][n]))
            ev[bname] = ak.zip(b_nest)
    return ev
