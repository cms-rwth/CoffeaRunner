from Hpluscharm.workflows import workflows as hplusc_wf

cfg = {
    "dataset": {
        "jsons": [
            # "src/Hpluscharm/input_json/higgs_UL17.json",
            # "src/Hpluscharm/input_json/signal_UL17.json",
            "src/Hpluscharm/input_json/mcbkg_UL17.json"
            # "src/Hpluscharm/input_json/st_local.json"
        ],
        "campaign": "UL17",
        "year": "2017",
        "filter": {
            "samples": [
                # # #             # "ZZ_TuneCP5_13TeV-pythia8",
                # # #             # "WZ_TuneCP5_13TeV-pythia8",
                # # #             # "WW_TuneCP5_13TeV-pythia8"
                # # #             # "gchcWW2L2Nu_4f"
                #        "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
                #    "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8",
                "ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
                "ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
                "ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8",
                "ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",
                "ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",
                #    "WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8",
                #    "DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8",
                #    "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
                #     "DYJetsToTauTauToMuTauh_M-50_TuneCP5_13TeV-madgraphMLM-pythia8"
            ]
        },
    },
    # Input and output files
    "workflow": hplusc_wf["HWWtest"],
    "output": "st_all_array",
    "run_options": {
        "executor": "parsl/condor/naf_lite",
        # "executor":"iterative",
        "workers": 4,
        "scaleout": 200,
        "walltime": "03:00:00",
        "mem_per_worker": 2,  # GB
        "chunk": 15000,
        "skipbadfiles": True,
        "sample_size": 20,
        "retries": 50,
        "index": "0,0",
    },
    ## selections
    "categories": {"cats": [], "cats2": []},
    "preselections": {
        "mu1hlt": ["IsoMu27"],
        "mu2hlt": [
            "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
            "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8",
        ],
        "e1hlt": ["Ele35_WPTight_Gsf"],
        "e2hlt": ["Ele23_Ele12_CaloIdL_TrackIdL_IsoVL"],
        "emuhlt": [
            "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL",
            "Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL",
            "Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
        ],
    },
    ## weights
    "weights": {
        "common": {
            "inclusive": {
                "lumiMask": "Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt",
                "PU": "puweight_UL17.histo.root",
                "JME": "mc_compile_jec.pkl.gz",
                "BTV": {
                    "DeepJetC": "DeepJet_ctagSF_Summer20UL17_interp.root",
                },
                "LSF": {
                    # "ele_Trig TrigSF": "Ele32_L1DoubleEG_TrigSF_vhcc.histo.root",
                    "ele_Rereco_above20 EGamma_SF2D": "egammaEffi_ptAbove20.txt_EGM2D_UL2017.histo.root",
                    "ele_Rereco_below20 EGamma_SF2D": "egammaEffi_ptBelow20.txt_EGM2D_UL2017.histo.root",
                    "ele_ID EGamma_SF2D": "egammaEffi.txt_EGM2D_MVA90iso_UL17.histo.root",
                    "mu_ID NUM_TightID_DEN_TrackerMuons_abseta_pt": "Efficiencies_muon_generalTracks_Z_Run2017_UL_ID.histo.root",
                    "mu_Iso NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt": "Efficiencies_muon_generalTracks_Z_Run2017_UL_ISO.histo.root",
                    "ele_Rereco_above20_error EGamma_SF2D_error": "egammaEffi_ptAbove20.txt_EGM2D_UL2017.histo.root",
                    "ele_Rereco_below20_error EGamma_SF2D_error": "egammaEffi_ptBelow20.txt_EGM2D_UL2017.histo.root",
                    "ele_ID_error EGamma_SF2D_error": "egammaEffi.txt_EGM2D_MVA90iso_UL17.histo.root",
                    "mu_ID_error NUM_TightID_DEN_TrackerMuons_abseta_pt_error": "Efficiencies_muon_generalTracks_Z_Run2017_UL_ID.histo.root",
                    "mu_Iso_error NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt_error": "Efficiencies_muon_generalTracks_Z_Run2017_UL_ISO.histo.root",
                },
            },
        },
    },
    "systematic": {
        "JERC": False,
        "weights": False,
    },
    ## user specific
    "userconfig": {
        "export_array": True,
        "BDT": {
            "ll": "src/Hpluscharm/MVA/xgb_output/SR_ll_scangamma_2017_gamma2.json",
            "emu": "src/Hpluscharm/MVA/xgb_output/SR_emu_scangamma_2017_gamma2.json",
        },
    },
}
