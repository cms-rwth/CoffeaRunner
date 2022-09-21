from Hpluscharm.workflows import workflows as hplusc_wf

cfg =  {
    "dataset" : {
        "jsons": ["src/Hpluscharm/input_json/higgs_UL17.json"],
        "campaign" :"UL17",
        "year" : "2017",
        # "filter": {
        #     "samples":["gchcWW2L2Nu_4f"],
        #     "samples_exclude" : []
        # }
    },

    # Input and output files
    "workflow" : hplusc_wf["HWWtest"],
    "output"   : "signal_nocut",
   
    "run_options" : {
        "executor"       : "iterative",
        "workers"        : 6,
        "scaleout"       : 10,
        "walltime"       : "03:00:00",
        "mem_per_worker" : 2, # GB
        "chunk"          : 50000000,
        "max"            : None,
        "skipbadfiles"   : None,
        "voms"           : None,
        "limit"          : 1,
     },
    
    
    ## user specific
    "userconfig":{
    "systematics": 
        {
            "JERC":False,
            "weights":False,
        },
    "export_array" : False,
    "BDT":{
        "ll":"src/Hpluscharm/MVA/xgb_output/SR_ll_scangamma_2017_gamma2.json",
        "emu":"src/Hpluscharm/MVA/xgb_output/SR_emu_scangamma_2017_gamma2.json",
    }
    },
    ## selections
    "categories" :{"cats":[],"cats2":[]},
    "preselections":{
    "mu1hlt": ["IsoMu27"],
    "mu2hlt": 
        [
            "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
            "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8",
        ],
    "e1hlt":["Ele35_WPTight_Gsf"],
        
    "e2hlt": ["Ele23_Ele12_CaloIdL_TrackIdL_IsoVL"],
    "emuhlt": 
    [
        "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL",
        "Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL",
        "Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
        "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
    ],
    },
    ## weights
    "weights":{
        "common":{
            "inclusive":{
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
    }    
}
