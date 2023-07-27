import pickle, os, sys, numpy as np
from coffea import processor
import awkward as ak
import hist as Hist
from functools import partial
import gc
import os, psutil
import coffea
from coffea.analysis_tools import Weights
from BTVNanoCommissioning.utils.correction import (
    load_lumi,
    load_SF,
    JME_shifts,
    Roccor_shifts,
    puwei,
    met_filters,
    eleSFs,
    muSFs,
    btagSFs,
    jmar_sf,
    add_ps_weight,
    add_pdf_weight,
    add_scalevar_7pt,
    add_scalevar_3pt,
    top_pT_reweighting,
)
from BTVNanoCommissioning.helpers.func import (
    mT,
    flatten,
    normalize,
    make_p4,
    #defaultdict_accumulator,
    update,
)


class NanoProcessor(processor.ProcessorABC):
    # Define histograms
    def __init__(self, cfg):
        self.cfg = cfg
        self._year = self.cfg.dataset["year"]
        self._campaign = self.cfg.dataset["campaign"]
        self.systematics = self.cfg.systematic
        self._met_filters = met_filters[self._campaign]
        self._lumiMasks = load_lumi(self.cfg.weights_config["lumiMask"])
        self.SF_map = load_SF(
            self.cfg.dataset["campaign"],
            self.cfg.weights_config,
            self.systematics["weights"],
        )
        syst_axis = Hist.axis.StrCategory([], name="syst", growth=True)
        pt_axis = Hist.axis.Regular(50, 0, 300, name="pt", label=" $p_{T}$ [GeV]")
        eta_axis = Hist.axis.Regular(25, -2.5, 2.5, name="eta", label=" $\eta$")
        phi_axis = Hist.axis.Regular(30, -3, 3, name="phi", label="$\phi$")

        self.make_output = lambda: {
            "cutflow": processor.defaultdict_accumulator(
                #         # we don't use a lambda function to avoid pickle issues
                partial(processor.defaultdict_accumulator, int)
            ),
            "sumw": 0,
            "ele_eta": Hist.Hist(syst_axis, eta_axis, Hist.storage.Weight()),
            "mu_eta": Hist.Hist(syst_axis, eta_axis, Hist.storage.Weight()),
            "ele_phi": Hist.Hist(syst_axis, phi_axis, Hist.storage.Weight()),
            "mu_phi": Hist.Hist(syst_axis, phi_axis, Hist.storage.Weight()),
            "ele_pt": Hist.Hist(syst_axis, pt_axis, Hist.storage.Weight()),
            "mu_pt": Hist.Hist(syst_axis, pt_axis, Hist.storage.Weight()),
            "jet_pt": Hist.Hist(syst_axis, pt_axis, Hist.storage.Weight()),
            "met_pt": Hist.Hist(syst_axis, pt_axis, Hist.storage.Weight()),
        }

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        isRealData = not hasattr(events, "genWeight")
        dataset = events.metadata["dataset"]
        shifts = []

        if "JME" in self.SF_map.keys():
            shifts = JME_shifts(
                shifts,
                self.SF_map,
                events,
                self.cfg.dataset["campaign"],
                isRealData,
                self.systematics["JERC"],
            )
        else:
            shifts = [
                ({"Jet": events.Jet, "MET": events.MET, "Muon": events.Muon}, None)
            ]
        if "roccor" in self.SF_map.keys():
            shifts = Roccor_shifts(
                shifts, self.SF_map, events, isRealData, self.systematics["roccor"]
            )
        else:
            shifts[0][0]["Muon"] = events.Muon

        return processor.accumulate(
            self.process_shift(update(events, collections), name)
            for collections, name in shifts
        )

    def process_shift(self, events, shift_name):
        dataset = events.metadata["dataset"]
        isRealData = not hasattr(events, "genWeight")
        output = self.make_output()
        # if self._export_array: output_array = ak.Array({shift_name:ak.A({})})
        req_lumi = np.ones(len(events), dtype="bool")
        if isRealData:
            req_lumi = self._lumiMasks(events.run, events.luminosityBlock)

        # #############Selections

        event_mu = events.Muon
        musel = (
            (event_mu.pt > 15)
            & (abs(event_mu.eta) < 2.4)
            & (event_mu.mvaId >= 3)
            & (event_mu.pfRelIso04_all < 0.15)
            & (abs(event_mu.dxy) < 0.05)
            & (abs(event_mu.dz) < 0.1)
        )
        event_mu = event_mu[musel]

        # ## Electron cuts
        # # electron twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedElectronIdentificationRun2
        event_e = events.Electron
        elesel = (
            (event_e.pt > 15)
            & (abs(event_e.eta) < 2.5)
            & (event_e.mvaFall17V2Iso_WP90 == 1)
            & (abs(event_e.dxy) < 0.05)
            & (abs(event_e.dz) < 0.1)
        )
        event_e = event_e[elesel]
        event_jet = events.Jet[
            (events.Jet.pt > 20)
            & (abs(events.Jet.eta) <= 2.4)
            & ((events.Jet.puId > 6) | (events.Jet.pt > 50))
            & (events.Jet.jetId > 5)
        ]
        event_sel = (
            (ak.count(event_jet.pt, axis=-1) == 1)
            & (ak.count(event_e.pt, axis=-1) == 1)
            & (ak.count(event_mu.pt, axis=-1) == 1)
        )
        weights = Weights(len(events[event_sel]), storeIndividual=True)
        event_e = event_e[event_sel]
        event_mu = event_mu[event_sel]
        event_jet = event_jet[event_sel]
        eleSFs(event_e, self.SF_map, weights, syst=self.systematics["weights"])
        muSFs(event_mu, self.SF_map, weights, syst=self.systematics["weights"])
        btagSFs(
            event_jet,
            self.SF_map,
            weights,
            "DeepJetC",
            syst=self.systematics["weights"],
        )
        jmar_sf(event_jet, self.SF_map, weights, syst=self.systematics["weights"])
        if shift_name is None:
            systematics = ["noweight", "nominal"] + list(weights.variations)
        else:
            systematics = [shift_name]
        # for histname, h in output.items():
        for syst in systematics:
            if syst in weights.variations:
                weight = weights.weight(modifier=syst)
            elif "noweight" == syst:
                weight = np.ones_like(weights.weight())
            else:
                weight = weights.weight()
            output["ele_pt"].fill(syst=syst, pt=event_e[:, 0].pt, weight=weight)
            output["mu_pt"].fill(syst=syst, pt=event_mu[:, 0].pt, weight=weight)
            output["jet_pt"].fill(syst=syst, pt=event_jet[:, 0].pt, weight=weight)
            output["met_pt"].fill(syst=syst, pt=events[event_sel].MET.pt, weight=weight)

        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator
