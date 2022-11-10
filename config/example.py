# Local Variables:
# python-indent-offset: 4
# End:

from test_wf import NanoProcessor as test_wf

cfg = {
    "dataset": {
        "jsons": ["metadata/vjets3.json"],
        "campaign": "UL17",
        "year": "2017",
        "filter": {
            "samples":["/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM"],
            "samples_exclude" : []
        }
    },
    # Input and output files
    "workflow": test_wf,
    "output": "test",
    "run_options": {
        "executor": "parsl/condor",
        #"executor": "futures",
        "workers": 1,
        "scaleout": 40,
        "walltime": "03:00:00",
        "mem_per_worker": 2,  # GB
        "chunk": 5000000,
        "max": None,
        "skipbadfiles": None,
        "voms": None,
        "limit": 20,
        "retries": 20,
        "splitjobs": False,

    },
}
