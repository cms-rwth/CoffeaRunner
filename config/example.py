from test_wf import NanoProcessor as test_wf

cfg = {
    "dataset": {
        "jsons": [
            "metadata/ttbar_UL17.json",
        ],
        "campaign": "2017_UL",
        "year": "2017",
    },
    # Input and output files
    "workflow": test_wf,
    "output": "test",
    "run_options": {
        "executor": "iterative",
        "workers": 2,
        "walltime": "03:00:00",
        "mem_per_worker": 2,  # GB
        "chunk": 250000,
        "skipbadfiles": False,
        "retries": 20,
        "limit": 1,
        "max": 1,
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
                ## compiled factories for JME
                "JME": "jec_compiled.pkl.gz",
                ## Read txt roccor corrections
                "roccor": None,
                ## Read from correctionlib , structured dict ID (campaign): WP
                "PU": None,  # no ID need to specied
                # JMAR, IDs from JME
                "JMAR": {"PUJetID_eff": "L"},
                # BTV SFs (shape currently implemented)
                "BTV": {"deepJet": "shape"},
                "LSF": {
                    "ele_ID 2017": "wp90iso",
                    "ele_Reco 2017": "RecoAbove20",
                    "ele_Reco_low 2017": "RecoBelow20",
                    "mu_Reco 2017_UL": "NUM_TrackerMuons_DEN_genTracks",
                    "mu_ID 2017_UL": "NUM_TightID_DEN_TrackerMuons",
                    "mu_Iso 2017_UL": "NUM_TightRelIso_DEN_TightIDandIPCut",
                    ## customed json/root file
                    "mu_ID_low NUM_TightID_DEN_TrackerMuons": "Efficiency_muon_trackerMuon_Run2017_UL_ID.histo.json",
                    "mu_Reco_low NUM_TrackerMuons_DEN_genTracks": "Efficiency_muon_generalTracks_Run2017_UL_trackerMuon.histo.json",
                },
            },
        },
    },
    "systematic": {
        "JERC": "split",  # False, True(only total unc)
        "weights": True,  # SFs/weight uncertainties
        "roccor": True,  # rochester muon up/down
    },
}
