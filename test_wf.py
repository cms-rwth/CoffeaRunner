import pickle, os, sys, numpy as np
from coffea import hist, processor
import awkward as ak
import hist as Hist
from functools import partial
import gc
import os, psutil
import coffea
from BTVNanoCommissioning.utils.correction import (
    lumiMasks,
    met_filters,
)
from BTVNanoCommissioning.helpers.func import (
    mT,
    flatten,
    normalize,
    make_p4,
    defaultdict_accumulator,
    update,
)

from BTVNanoCommissioning.helpers.cTagSFReader import getSF


def dphilmet(l1, l2, met):
    return np.where(
        abs(l1.delta_phi(met)) < abs(l2.delta_phi(met)),
        abs(l1.delta_phi(met)),
        abs(l2.delta_phi(met)),
    )


def BDTreader(dmatrix, xgb_model):
    return 1.0 / (1 + np.exp(-xgb_model.predict(dmatrix)))


# code for which memory has to
# be monitored


class NanoProcessor(processor.ProcessorABC):
    # Define histograms
    def __init__(self, cfg):
        self.cfg = cfg
        self._year = self.cfg.dataset["year"]
        self._campaign = self.cfg.dataset["campaign"]
        self._met_filters = met_filters[self._campaign]
        self._lumiMasks = lumiMasks[self._campaign]

        pt_axis = Hist.axis.Regular(50, 0, 300, name="pt", label=" $p_{T}$ [GeV]")
        eta_axis = Hist.axis.Regular(25, -2.5, 2.5, name="eta", label=" $\eta$")
        phi_axis = Hist.axis.Regular(30, -3, 3, name="phi", label="$\phi$")

        self.make_output = lambda: {
            "cutflow": processor.defaultdict_accumulator(
                #         # we don't use a lambda function to avoid pickle issues
                partial(processor.defaultdict_accumulator, int)
            ),
            "sumw": 0,
            "ele_eta": Hist.Hist(eta_axis, Hist.storage.Weight()),
            "mu_eta": Hist.Hist(eta_axis, Hist.storage.Weight()),
            "ele_phi": Hist.Hist(phi_axis, Hist.storage.Weight()),
            "mu_phi": Hist.Hist(phi_axis, Hist.storage.Weight()),
            "ele_pt": Hist.Hist(pt_axis, Hist.storage.Weight()),
            "mu_pt": Hist.Hist(pt_axis, Hist.storage.Weight()),
        }

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):

        dataset = events.metadata["dataset"]
        isRealData = not hasattr(events, "genWeight")
        output = self.make_output()
        # if self._export_array: output_array = ak.Array({shift_name:ak.A({})})
        req_lumi = np.ones(len(events), dtype="bool")
        if isRealData:
            req_lumi = self._lumiMasks(events.run, events.luminosityBlock)

        # #############Selections

        event_mu = events[req_lumi].Muon
        musel = (
            (event_mu.pt > 13)
            & (abs(event_mu.eta) < 2.4)
            & (event_mu.mvaId >= 3)
            & (event_mu.pfRelIso04_all < 0.15)
            & (abs(event_mu.dxy) < 0.05)
            & (abs(event_mu.dz) < 0.1)
        )
        event_mu = event_mu[ak.argsort(event_mu.pt, axis=1, ascending=False)]
        event_mu = event_mu[musel]
        event_mu = ak.pad_none(event_mu, 2, axis=1)

        # ## Electron cuts
        # # electron twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedElectronIdentificationRun2
        event_e = events[req_lumi].Electron
        elesel = (
            (event_e.pt > 13)
            & (abs(event_e.eta) < 2.5)
            & (event_e.mvaFall17V2Iso_WP90 == 1)
            & (abs(event_e.dxy) < 0.05)
            & (abs(event_e.dz) < 0.1)
        )
        event_e = event_e[elesel]
        event_e = event_e[ak.argsort(event_e.pt, axis=1, ascending=False)]
        event_e = ak.pad_none(event_e, 2, axis=1)
        for histname, h in output.items():
            if "ele" in histname:
                h.fill(flatten(event_e[histname.replace("ele_", "")]))
            elif "mu" in histname:
                h.fill(flatten(event_mu[histname.replace("mu_", "")]))
        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator
