import importlib.resources
import gzip
import pickle
import contextlib
import cloudpickle
import os
import numpy as np
import awkward as ak
from coffea.lookup_tools import extractor

from coffea.lumi_tools import LumiMask
from coffea.btag_tools import BTagScaleFactor
from coffea.lookup_tools import extractor
import correctionlib

from BTVNanoCommissioning.helpers.cTagSFReader import getSF


def load_SF(campaign, config, syst=False):
    correction_map = {}
    for SF in config.keys():
        if SF == "JME" or SF == "lumiMask":
            continue
        if SF == "PU":
            ## Check whether files in jsonpog-integration exist
            if os.path.exists(
                f"src/BTVNanoCommissioning/jsonpog-integration/POG/LUM/{campaign}"
            ):
                correction_map["PU"] = correctionlib.CorrectionSet.from_file(
                    f"src/BTVNanoCommissioning/jsonpog-integration/POG/LUM/{campaign}/puWeights.json.gz"
                )
            ## Otherwise custom files
            else:
                _pu_path = f"BTVNanoCommissioning.data.PU.{campaign}"
                with importlib.resources.path(
                    _pu_path, config["PU"]
                ) as filename:
                    if str(filename).endswith(".pkl.gz"):
                        with gzip.open(filename) as fin:
                            correction_map["PU"] = cloudpickle.load(fin)[
                                "2017_pileupweight"
                            ]
                    elif str(filename).endswith(".histo.root"):
                        ext = extractor()
                        ext.add_weight_sets([f"* * {filename}"])
                        ext.finalize()
                        correction_map["PU"] = ext.make_evaluator()["PU"]

        elif SF == "BTV":
            if os.path.exists(
                f"src/BTVNanoCommissioning/jsonpog-integration/POG/BTV/{campaign}"
            ):
                correction_map["btag"] = correctionlib.CorrectionSet.from_file(
                    f"src/BTVNanoCommissioning/jsonpog-integration/POG/BTV/{campaign}/btagging.json.gz"
                )
                correction_map["ctag"] = correctionlib.CorrectionSet.from_file(
                    f"src/BTVNanoCommissioning/jsonpog-integration/POG/BTV/{campaign}/ctagging.json.gz"
                )
            else:
                correction_map["btag"] = {}
                correction_map["ctag"] = {}
                _btag_path = f"BTVNanoCommissioning.data.BTV.{campaign}"
                for tagger in config["BTV"]:
                    with importlib.resources.path(
                        _btag_path, config["BTV"][tagger]
                    ) as filename:
                        if "B" in tagger:
                            correction_map["btag"][tagger] = BTagScaleFactor(
                                filename,
                                BTagScaleFactor.RESHAPE,
                                methods="iterativefit,iterativefit,iterativefit",
                            )
                        else:
                            if campaign == "Rereco17_94X":
                                correction_map["ctag"][tagger] = (
                                    "BTVNanoCommissioning/data/BTV/"
                                    + campaign
                                    + "/"
                                    + config["BTV"][tagger]
                                )
                            else:
                                correction_map["ctag"][tagger] = BTagScaleFactor(
                                    filename,
                                    BTagScaleFactor.RESHAPE,
                                    methods="iterativefit,iterativefit,iterativefit",
                                )

        elif SF == "LSF":
            correction_map["MUO_cfg"] = {
                mu: f
                for mu, f in config["LSF"].items()
                if "mu" in mu 
            }
            correction_map["EGM_cfg"] = {
                e: f
                for e, f in config["LSF"].items()
                if "ele" in e
            }
            ## Muon
            if os.path.exists(
                f"src/BTVNanoCommissioning/jsonpog-integration/POG/MUO/{campaign}"
            ) and os.path.exists(
                f"src/BTVNanoCommissioning/jsonpog-integration/POG/EGM/{campaign}"
            ):
                correction_map["MUO"] = correctionlib.CorrectionSet.from_file(
                    f"src/BTVNanoCommissioning/jsonpog-integration/POG/MUO/{campaign}/muon_Z.json.gz"
                )
                correction_map["EGM"] = correctionlib.CorrectionSet.from_file(
                    f"src/BTVNanoCommissioning/jsonpog-integration/POG/EGM/{campaign}/electron.json.gz"
                )
            ### Check if any custom corrections needed
            # FIXME: (some low pT muons not supported in jsonpog-integration at the moment)

            if (
                ".json" in "\t".join(list(config["LSF"].values()))
                or ".txt"
                in "\t".join(list(config["LSF"].values()))
                or ".root"
                in "\t".join(list(config["LSF"].values()))
            ):
                _mu_path = f"BTVNanoCommissioning.data.LSF.{campaign}"
                ext = extractor()
                with contextlib.ExitStack() as stack:
                    inputs, real_paths = [
                        k
                        for k in correction_map["MUO_cfg"].keys()
                        if ".json" in correction_map["MUO_cfg"][k]
                        or ".txt" in correction_map["MUO_cfg"][k]
                        or ".root" in correction_map["MUO_cfg"][k]
                    ], [
                        stack.enter_context(importlib.resources.path(_mu_path, f))
                        for f in correction_map["MUO_cfg"].values()
                        if ".json" in f or ".txt" in f or ".root" in f
                    ]
                    
                    inputs = [i.split(" ")[0]+" *" if "_low" in i else i for i in inputs ]
                    
                    ext.add_weight_sets(
                        [
                            f"{paths} {file}"
                            for paths, file in zip(inputs, real_paths)
                            if ".json" in str(file)
                            or ".txt" in str(file)
                            or ".root" in str(file)
                        ]
                    )
                    if syst:    
                        ext.add_weight_sets(
                            paths.split(" ")[0]+"_error "+paths.split(" ")[1]+"_error "+file
                            for paths, file in zip(inputs, real_paths)
                            if ".root" in str(file)
                        )
                ext.finalize()
                correction_map["MUO_custom"] = ext.make_evaluator()

                _ele_path = f"BTVNanoCommissioning.data.LSF.{campaign}"
                ext = extractor()
                with contextlib.ExitStack() as stack:
                    inputs, real_paths = [
                        k
                        for k in correction_map["EGM_cfg"].keys()
                        if ".json" in correction_map["EGM_cfg"][k]
                        or ".txt" in correction_map["EGM_cfg"][k]
                        or ".root" in correction_map["EGM_cfg"][k]
                    ], [
                        stack.enter_context(importlib.resources.path(_ele_path, f))
                        for f in correction_map["EGM_cfg"].values()
                        if ".json" in f or ".txt" in f or ".root" in f
                    ]
                    ext.add_weight_sets(
                        [
                            f"{paths} {file}"
                            for paths, file in zip(inputs, real_paths)
                            if ".json" in str(file)
                            or ".txt" in str(file)
                            or ".root" in str(file)
                        ]
                    )
                    if syst:
                        ext.add_weight_sets(
                            paths.split(" ")[0]+"_error "+paths.split(" ")[1]+"_error "+file
                            for paths, file in zip(inputs, real_paths)
                            if ".root" in str(file)
                        )
                ext.finalize()
                correction_map["EGM_custom"] = ext.make_evaluator()

    return correction_map


def load_lumi(path):
    _lumi_path = "BTVNanoCommissioning.data.lumiMasks"
    with importlib.resources.path(
        _lumi_path, path
    ) as filename:
        return LumiMask(filename)


## MET filters
met_filters = {
    "2016preVFP_UL": {
        "data": [
            "goodVertices",
            "globalSuperTightHaloUL16Filter",
            "HBHENoiseFilter",
            "HBHENoiseIsoFilter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "BadPFMuonDzFilter",
            "eeBadScFilter",
        ],
        "mc": [
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "HBHENoiseFilter",
            "HBHENoiseIsoFilter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "BadPFMuonDzFilter",
            "eeBadScFilter",
        ],
    },
    "2016postVFP_UL": {
        "data": [
            "goodVertices",
            "globalSuperTightHaloUL16Filter",
            "HBHENoiseFilter",
            "HBHENoiseIsoFilter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "BadPFMuonDzFilter",
            "eeBadScFilter",
        ],
        "mc": [
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "HBHENoiseFilter",
            "HBHENoiseIsoFilter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "BadPFMuonDzFilter",
            "eeBadScFilter",
        ],
    },
    "UL17": {
        "data": [
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "HBHENoiseFilter",
            "HBHENoiseIsoFilter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "BadPFMuonDzFilter",
            "hfNoisyHitsFilter",
            "eeBadScFilter",
            "ecalBadCalibFilter",
        ],
        "mc": [
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "HBHENoiseFilter",
            "HBHENoiseIsoFilter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "BadPFMuonDzFilter",
            "hfNoisyHitsFilter",
            "eeBadScFilter",
            "ecalBadCalibFilter",
        ],
    },
    "2018_UL": {
        "data": [
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "HBHENoiseFilter",
            "HBHENoiseIsoFilter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "BadPFMuonDzFilter",
            "hfNoisyHitsFilter",
            "eeBadScFilter",
            "ecalBadCalibFilter",
        ],
        "mc": [
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "HBHENoiseFilter",
            "HBHENoiseIsoFilter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "BadPFMuonDzFilter",
            "hfNoisyHitsFilter",
            "eeBadScFilter",
            "ecalBadCalibFilter",
        ],
    },
}

##JEC
# FIXME: would be nicer if we can move to correctionlib in the future together with factory and workable
def load_jmefactory(campaign,cfg):
    _jet_path = f"BTVNanoCommissioning.data.JME.{campaign}"
    with importlib.resources.path(
        _jet_path, cfg
    ) as filename:
        with gzip.open(filename) as fin:
            jmestuff = cloudpickle.load(fin)
    
    return jmestuff


def add_jec_variables(jets, event_rho):
    jets["pt_raw"] = (1 - jets.rawFactor) * jets.pt
    jets["mass_raw"] = (1 - jets.rawFactor) * jets.mass
    if hasattr(jets, "genJetIdxG"):
        jets["pt_gen"] = ak.values_astype(
            ak.fill_none(jets.matched_gen.pt, 0), np.float32
        )
    jets["event_rho"] = ak.broadcast_arrays(event_rho, jets.pt)[0]
    return jets


## PU weight
def puwei(correct_map, nPU, syst="nominal"):
    if "correctionlib" in str(type(correct_map["PU"])):
        return correct_map["PU"][list(correct_map["PU"].keys())[0]].evaluate(
            nPU, syst
        )
    else:
        return correct_map["PU"](nPU)


def btagSFs(jet, correct_map,  weights,SFtype,syst=True):
    systlist= ["Extrap","Interp","LHEScaleWeight_muF","LHEScaleWeight_muR","PSWeightFSR","PSWeightISR","PUWeight","Stat","XSec_BRUnc_DYJets_b","XSec_BRUnc_DYJets_c","XSec_BRUnc_WJets_c","jer","jesTotal"]
    masknone = ak.is_none(jet.pt) 
    jet.btagDeepFlavCvL = ak.fill_none(jet.btagDeepFlavCvL,0.)
    jet.btagDeepFlavCvB = ak.fill_none(jet.btagDeepFlavCvB,0.)
    jet.btagDeepCvL = ak.fill_none(jet.btagDeepCvL,0.)
    jet.btagDeepCvB = ak.fill_none(jet.btagDeepCvB,0.)
    jet.hadronFlavour = ak.fill_none(jet.hadronFlavour,0)
    sfs_up_all,sfs_down_all={},{}
    for i, sys in enumerate(systlist):
        if "correctionlib" in str(type(correct_map["btag"])):
            if SFtype == "DeepJetC":
                sfs = np.where(masknone,1.,correct_map["ctag"]["deepJet_shape"].evaluate(
                    "central", jet.hadronFlavour, jet.btagDeepFlavCvL, jet.btagDeepFlavCvB
                ))
                sfs_up = np.where(masknone,1.,correct_map["ctag"]["deepJet_shape"].evaluate(
                    f"up_{systlist[i]}", jet.hadronFlavour, jet.btagDeepFlavCvL, jet.btagDeepFlavCvB
                ))
                sfs_down = np.where(masknone,1.,correct_map["ctag"]["deepJet_shape"].evaluate(
                    f"down_{systlist[i]}", jet.hadronFlavour, jet.btagDeepFlavCvL, jet.btagDeepFlavCvB
                ))
            if SFtype == "DeepCSVC":
                sfs = np.where(masknone,1.,correct_map["ctag"]["deepCSV_shape"].evaluate(
                    "central", jet.hadronFlavour, jet.btagDeepCvL, jet.btagDeepCvB
                ))
                sfs_up = np.where(masknone,1.,correct_map["ctag"]["deepCSV_shape"].evaluate(
                    f"up_{systlist[i]}", jet.hadronFlavour, jet.btagDeepCvL, jet.btagDeepCvB
                ))
                sfs_down = np.where(masknone,1.,correct_map["ctag"]["deepCSV_shape"].evaluate(
                    f"down_{systlist[i]}", jet.hadronFlavour, jet.btagDeepCvL, jet.btagDeepCvB
                ))
            sfs_up_all[sys] = sfs_up
            sfs_down_all[sys] = sfs_down
            
        if i==0 and syst == False: 
            weights.add(SFtype,sfs)
            break

    if syst== True:weights.add_multivariation(SFtype,sfs,systlist,np.array(list(sfs_up_all.values())),np.array(list(sfs_down_all.values())))
    return weights

### Lepton SFs


def eleSFs(ele, correct_map, weights, syst=True, isHLT=False):
    
    
    ele_eta = ele.eta
    ele_pt = np.where(ele.pt < 20, 20.0, ele.pt)
    mask = ele.pt > 20.0
    
    weight = 1.0
    for sf in correct_map["EGM_cfg"].keys():
        ## Only apply SFs for lepton pass HLT filter
        if not isHLT and "HLT" in sf:
            continue
        sf_type = sf[: sf.find(" ")]
        if "low" in sf :continue
        if "correctionlib" in str(type(correct_map["EGM"])):
            if "Reco" in sf:
                
                ele_pt = np.where(ele.pt < 20.0, 20.0, ele.pt)
                ele_pt_low = np.where(ele.pt >= 20.0, 19.9, ele.pt)
                
                sfs_low = np.where(
                    ~mask,
                    correct_map["EGM"][list(correct_map["EGM"].keys())[0]].evaluate(sf[sf.find(" ") + 1 :], "sf", "RecoBelow20", ele_eta, ele_pt_low),
                    1.0,
                )
                sfs = np.where(
                    mask,
                    correct_map["EGM"][list(correct_map["EGM"].keys())[0]].evaluate(sf[sf.find(" ") + 1 :], "sf", "RecoAbove20", ele_eta, ele_pt),
                    sfs_low,
                )
                
                if syst:
                    sfs_up_low = np.where(
                    ~mask,
                    correct_map["EGM"][list(correct_map["EGM"].keys())[0]].evaluate(sf[sf.find(" ") + 1 :], "sfup", "RecoBelow20", ele_eta, ele_pt_low),0.)
                    sfs_down_low = np.where(
                    ~mask,
                    correct_map["EGM"][list(correct_map["EGM"].keys())[0]].evaluate(sf[sf.find(" ") + 1 :], "sfdown", "RecoBelow20", ele_eta, ele_pt_low),0.)
                    sfs_up = np.where(mask,correct_map["EGM"][list(correct_map["EGM"].keys())[0]].evaluate(sf[sf.find(" ") + 1 :], "sfup", "RecoAbove20", ele_eta, ele_pt),sfs_up_low)
                    sfs_down = np.where(mask,correct_map["EGM"][list(correct_map["EGM"].keys())[0]].evaluate(sf[sf.find(" ") + 1 :], "sfdown", "RecoAbove20", ele_eta, ele_pt),sfs_down_low)
                    weights.add(sf.split(" ")[0],sfs+sfs_up,sfs+sfs_down)
                else: weights.add(sf.split(" ")[0],sfs)
                
            else:
                sfs = correct_map["EGM"][
                    list(correct_map["EGM"].keys())[0]
                ].evaluate(
                    sf[sf.find(" ") + 1 :],
                    "sf",
                    correct_map["EGM_cfg"][sf],
                    ele_eta,
                    ele_pt,
                )
                
                if syst:
                    sfs_up = correct_map["EGM"][
                    list(correct_map["EGM"].keys())[0]
                ].evaluate(
                    sf[sf.find(" ") + 1 :],
                    "sfup",
                    correct_map["EGM_cfg"][sf],
                    ele_eta,
                    ele_pt,
                )
                    sfs_down = correct_map["EGM"][
                    list(correct_map["EGM"].keys())[0]
                ].evaluate(
                    sf[sf.find(" ") + 1 :],
                    "sfup",
                    correct_map["EGM_cfg"][sf],
                    ele_eta,
                    ele_pt,
                )
                    weights.add(sf.split(" ")[0],sfs_up,sfs_down)
                else :weights.add(sf.split(" ")[0],sfs)
        else:
            if "ele_Trig" in sf:
                if syst: weights.add(sf.split(" ")[0], np.where(ele.lep_flav == 11,correct_map["EGM_custom"][sf_type](ele_pt),1.0),np.where(ele.lep_flav == 11,correct_map["EGM_custom"][sf_type](ele_pt)+correct_map["EGM_custom"][f"{sf_type}_error"](ele_pt),0.),np.where(ele.lep_flav == 11,correct_map["EGM_custom"][sf_type](ele_pt)-correct_map["EGM_custom"][f"{sf_type}_error"](ele_pt),0.))
                else:weights.add(sf.split(" ")[0], np.where(ele.lep_flav == 11,correct_map["EGM_custom"][sf_type](ele_pt),1.0))
            elif "ele" in sf:
                if syst: weights.add(sf.split(" ")[0], np.where(ele.lep_flav == 11,correct_map["EGM_custom"][sf_type](ele_eta, ele_pt),1.0), np.where(ele.lep_flav == 11,correct_map["EGM_custom"][sf_type](ele_eta, ele_pt)+correct_map["EGM_custom"][f"{sf_type}_error"](ele_eta, ele_pt),0.),np.where(ele.lep_flav == 11,correct_map["EGM_custom"][sf_type](ele_eta, ele_pt)-correct_map["EGM_custom"][f"{sf_type}_error"](ele_eta, ele_pt),0.))
                else:weights.add(sf.split(" ")[0], np.where(ele.lep_flav == 11,correct_map["EGM_custom"][sf_type](ele_eta, ele_pt),1.0))
    return weights

def muSFs(mu, correct_map, weights,syst=True, isHLT=False):
    mu_eta = np.where(np.abs(mu.eta)>=2.4,2.39,np.abs(mu.eta))
    mu_pt = mu.pt
    weight = 1.0
    sfs = 1.0
    for sf in correct_map["MUO_cfg"].keys():
        ## Only apply SFs for lepton pass HLT filter
        if not isHLT and "HLT" in sf:
            continue
        mask = mu_pt > 15.0
        if "low" in sf:continue
        sf_type = sf[: sf.find(" ")]
        if (
            "correctionlib" in str(type(correct_map["MUO"]))
            and "MUO_custom" in correct_map
        ):
            if "ID" in sf or "Rereco" in sf:
                mu_pt = np.where(mu.pt < 15.0, 15.0, mu.pt)
                mu_pt_low = np.where(mu.pt >= 15.0, 15.0, mu.pt)
                sfs_low = np.where(
                    ~mask,
                    correct_map["MUO_custom"][
                        f'{sf.split(" ")[0]}_low{correct_map["MUO_cfg"][sf]}/abseta_pt_value'
                    ](mu_eta, mu_pt_low),
                    1.0,
                )
                
                sfs = np.where(
                        mask,
                        correct_map["MUO"][correct_map["MUO_cfg"][sf]].evaluate(
                            sf.split(" ")[1], mu_eta, mu_pt, "sf"
                        ),
                        sfs_low,
                    )
                sfs_forerr = sfs
                
                if syst:
                    sfs_err_low = np.where(
                    ~mask,
                    correct_map["MUO_custom"][
                        f'{sf.split(" ")[0]}_low{correct_map["MUO_cfg"][sf]}/abseta_pt_error'
                    ](mu_eta, mu_pt_low),
                    0.0,
                    )
                    sfs_up = np.where(
                        mask,
                        correct_map["MUO"][correct_map["MUO_cfg"][sf]].evaluate(
                            sf.split(" ")[1], mu_eta, mu_pt, "systup"
                        ),
                        sfs_forerr+sfs_err_low,
                    )
                    sfs_down =  np.where(
                        mask,
                        correct_map["MUO"][correct_map["MUO_cfg"][sf]].evaluate(
                            sf.split(" ")[1], mu_eta, mu_pt, "systdown"
                        ),
                        sfs_forerr-sfs_err_low,
                    )
                    weights.add(sf.split(" ")[0],sfs,sfs+sfs_up,sfs-sfs_down)
                else:weights.add(sf.split(" ")[0],sfs)
                
        elif "correctionlib" in str(type(correct_map["MUO"])):
            sfs = correct_map["MUO"][correct_map["MUO_cfg"][sf]].evaluate(sf[sf.find(" ") + 1 :], mu_eta, mu_pt, "sf")
            if syst : 
                sfs_up= sfs+correct_map["MUO"][correct_map["MUO_cfg"][sf]].evaluate(sf[sf.find(" ") + 1 :], mu_eta, mu_pt, "systup")
                sf_down= sfs-correct_map["MUO"][correct_map["MUO_cfg"][sf]].evaluate(sf[sf.find(" ") + 1 :], mu_eta, mu_pt, "systdown")
                weights.add(sf.split(" ")[0],sfs,sfs_up,sfs_down)
            else:weights.add(sf.split(" ")[0],sfs)
        else:
            if "mu" in sf:
                sfs = correct_map["MUO_custom"][sf_type](mu_eta, mu_pt)
                if syst:
                    sfs_up= sfs+correct_map["MUO_custom"][f"{sf_type}_error"](mu_eta, mu_pt)
                    sf_down= sfs-correct_map["MUO_custom"][f"{sf_type}_error"](mu_eta, mu_pt)
                    weights.add(sf.split(" ")[0],sfs,sfs_up,sfs_down)
                    
                else:weights.add(sf.split(" ")[0],sfs)

    return weights


def add_pdf_weight(weights, pdf_weights):
    nom = np.ones(len(weights.weight()))
    up = np.ones(len(weights.weight()))
    down = np.ones(len(weights.weight()))

    # NNPDF31_nnlo_hessian_pdfas
    # https://lhapdfsets.web.cern.ch/current/NNPDF31_nnlo_hessian_pdfas/NNPDF31_nnlo_hessian_pdfas.info
    if pdf_weights is not None and "306000 - 306102" in pdf_weights.__doc__:
        # Hessian PDF weights
        # Eq. 21 of https://arxiv.org/pdf/1510.03865v1.pdf
        arg = pdf_weights[:, 1:-2] - np.ones((len(weights.weight()), 100))
        summed = ak.sum(np.square(arg), axis=1)
        pdf_unc = np.sqrt((1.0 / 99.0) * summed)
        weights.add("PDF_weight", nom, pdf_unc + nom)

        # alpha_S weights
        # Eq. 27 of same ref
        as_unc = 0.5 * (pdf_weights[:, 102] - pdf_weights[:, 101])
        weights.add("aS_weight", nom, as_unc + nom)

        # PDF + alpha_S weights
        # Eq. 28 of same ref
        pdfas_unc = np.sqrt(np.square(pdf_unc) + np.square(as_unc))
        weights.add("PDFaS_weight", nom, pdfas_unc + nom)

    else:
        weights.add("aS_weight", nom, up, down)
        weights.add("PDF_weight", nom, up, down)
        weights.add("PDFaS_weight", nom, up, down)


# Jennet adds PS weights
def add_ps_weight(weights, ps_weights):
    nom = np.ones(len(weights.weight()))
    up_isr = np.ones(len(weights.weight()))
    down_isr = np.ones(len(weights.weight()))
    up_fsr = np.ones(len(weights.weight()))
    down_fsr = np.ones(len(weights.weight()))

    if ps_weights is not None:
        if len(ps_weights[0]) == 4:
            up_isr = ps_weights[:, 0]
            down_isr = ps_weights[:, 2]
            up_fsr = ps_weights[:, 1]
            down_fsr = ps_weights[:, 3]
        # else:
        #   warnings.warn(f"PS weight vector has length {len(ps_weights[0])}")

    weights.add("UEPS_ISR", nom, up_isr, down_isr)
    weights.add("UEPS_FSR", nom, up_fsr, down_fsr)


def add_scalevar_7pt(weights, lhe_weights):
    nom = np.ones(len(weights.weight()))

    if len(lhe_weights) > 0:
        if len(lhe_weights[0]) == 9:
            up = np.maximum.reduce(
                [
                    lhe_weights[:, 0],
                    lhe_weights[:, 1],
                    lhe_weights[:, 3],
                    lhe_weights[:, 5],
                    lhe_weights[:, 7],
                    lhe_weights[:, 8],
                ]
            )
            down = np.minimum.reduce(
                [
                    lhe_weights[:, 0],
                    lhe_weights[:, 1],
                    lhe_weights[:, 3],
                    lhe_weights[:, 5],
                    lhe_weights[:, 7],
                    lhe_weights[:, 8],
                ]
            )
        elif len(lhe_weights[0]) > 1:
            print("Scale variation vector has length ", len(lhe_weights[0]))
    else:
        up = np.ones(len(weights.weight()))
        down = np.ones(len(weights.weight()))

    weights.add("scalevar_7pt", nom, up, down)


def add_scalevar_3pt(weights, lhe_weights):
    nom = np.ones(len(weights.weight()))

    if len(lhe_weights) > 0:
        if len(lhe_weights[0]) == 9:
            up = np.maximum(lhe_weights[:, 0], lhe_weights[:, 8])
            down = np.minimum(lhe_weights[:, 0], lhe_weights[:, 8])
        elif len(lhe_weights[0]) > 1:
            print("Scale variation vector has length ", len(lhe_weights[0]))
    else:
        up = np.ones(len(weights.weight()))
        down = np.ones(len(weights.weight()))

    weights.add("scalevar_3pt", nom, up, down)
