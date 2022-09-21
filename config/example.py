from test_wf import NanoProcessor as test_wf

cfg =  {
    "dataset" : {
        "jsons": ["metadata/test.json"],
        "campaign" :"UL17",
        "year" : "2017",
    },

    # Input and output files
    "workflow" : test_wf,
    "output"   : "test", 
   
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
    
    
}
