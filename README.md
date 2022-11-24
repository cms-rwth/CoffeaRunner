
# CoffeaRunner

[![Linting](https://github.com/cms-rwth/CoffeaRunner/actions/workflows/python_linting.yml/badge.svg)](https://github.com/cms-rwth/CoffeaRunner/actions/workflows/python_linting.yml)
[![Test Workflow](https://github.com/cms-rwth/CoffeaRunner/actions/workflows/test_workflow.yml/badge.svg)](https://github.com/cms-rwth/CoffeaRunner/actions/workflows/test_workflow.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Generalized framework columnar-based analysis with [coffea](https://coffeateam.github.io/coffea/) based on the developments from [BTVNanoCommissioning](https://github.com/cms-btv-pog/BTVNanoCommissioning) and some development from [PocketCoffea](https://github.com/PocketCoffea/PocketCoffea)

## Requirements

### Setup 

:heavy_exclamation_mark: Install under `bash` environment

Clone repository from git

```bash
# only first time 
git clone git@github.com:cms-rwth/CoffeaRunner.git
```

For installing Miniconda, see also https://hackmd.io/GkiNxag0TUmHnnCiqdND1Q#Local-or-remote
```bash 
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
# Run and follow instructions on screen
bash Miniconda3-latest-Linux-x86_64.sh
```
NOTE: always make sure that conda, python, and pip point to local Miniconda installation (`which conda` etc.).

You could simply create the environment through the existing `env.yml` under your conda environment

```
conda env create -f env.yml 
```
Once the environment is set up, compile the python package:
```
pip install -e .
```




## Structures of code

The development of the code is driven by user-friendliness, reproducibility and efficiency.

## How to run 
setup enviroment first
```bash
# activate enviroement
conda activate CoffeaRunner
# setup proxy
voms-proxy-init --voms cms --vomses ~/.grid-security/vomses 
```
### Make the list of input files (Optional)

Use the `./filefetcher/fetch.py` script:

```
python filefetcher/fetch.py --input input_DAS_list.txt --output ${output_name.json}
```
where the `input_DAS_list.txt` is a simple file with a list of dataset names extract from DAS (you need to create it yourself for the samples you want to run over), and output json file in creted in `./metadata` directory.

### Create compiled corretions file, like JERC (Optional)

:exclamation: In case existing correction file doesn't work for you due to the incompatibility of `cloudpickle` in different python versions. Please recompile the file to get new pickle file.

Compile correction pickle files for a specific JEC campaign by changing the dict of jet_factory, and define the MC campaign and the output file name by passing it as arguments to the python script:

```
python -m utils.compile_jec UL17_106X data/JME/UL17_106X/jec_compiled.pkl.gz
```
### Run the processor and create `.coffea` files

Get `.coffea` file in output directories and `config.pkl` for the configuration use in this events.

- Example workflow (`test_wf.py`): simple lepton selections, run with local iterative processor. No correction case. 

```
python runner_wconfig.py --cfg config/example.py (-o)

optional arguments: 
    -o, --overwrite_file        Overwrite the output in the configuration
    --validate        Do not process, just check all files are accessible
```

- More complex usage with `config/HWW2l2nu.py` and the application to [workflow](https://github.com/Ming-Yan/Hpluscharm/blob/master/workflows/hplusc_HWW2l2nu_process_test.py) from [HplusCharm](https://github.com/Ming-Yan/Hpluscharm/tree/master) developement. 
See details in [Advance Usage](#advanced-usages)
```
python runner_wconfig.py --cfg config/HWW2l2nu.py
```

### Config file
The config file in `.py` format is passed as the argument `--cfg` of the `runner_wconfig.py` script. The file has the following structure:

| Parameter name    | Allowed values               | Description
| :-----:           | :---:                        | :------------------------------------------
| `json`  (required)             | string          | Path of .json file to create with NanoAOD files (can load multiple files)
| `workflow` (required)          | workflows         | Workflow to run
| `output`  (required)          | string            | Path of output folder
| `executor`   (required)           | string       | See [executor](#executors) below 
| `workers`         | int                          | Number of parallel threads (with `futures` and clusters without fixed workers)
| `scaleout`        | int                          | Number of jobs to submit, use for cluster
| `chunk`           | int                          | Chunk size
| `max`             | int                          | Maximum number of chunks to process
| `skipbadfiles`    | bool                         | Skip bad files
| `splitjobs`       | bool                         | Split runner and accumulator to separate jobs to avoid local memory consumption to large
| `voms`            | string                       | Voms parameters (with condor)
| `limit`           | int                          | Maximum number of files per sample to process
| `preselections`   | list                         | List of preselection cuts
| `categories`      | dict                         | Dictionary of categories with cuts to apply*
| `userconfig`      | dict                         | Dictionary of user specific configuration, depends on workflow

*Cuts in `categories` or `preselections` don't follow, can write cuts as seperate macro

:construction: histogram and plot setup are not included in current version
:construction: implementation on export_array still under construction(would use `ak.to_parquet`)
Use `filter` to exclude/include specific sample, if there's no `filter` then would run through all the samples in json file

#### Advanced usages
##### filter
Use `filter(option)` to specify samples want to processed in the json files

```
"dataset" : {
        "jsons": ["src/Hpluscharm/input_json/higgs_UL17.json"],
        "campaign" :"UL17",
        "year" : "2017",
        "filter": {
            "samples":["GluGluHToWWTo2L2Nu_M-125_TuneCP5_13TeV-powheg-jhugen727-pythia8"],
            "samples_exclude" : []
        }
    },
```

##### Weights
All the `lumiMask`, correction files (SFs, pileup weight), and JEC, JER files are under  `BTVNanoCommissioning/src/data/` following the substructure `${type}/${campaign}/${files}`(except `lumiMasks` and `Prescales`)

| Type        | File type |  Comments|
| :---:   | :---: | :---: | 
| `lumiMasks` |`.json` | Masked good lumi-section used for physics analysis|
| `Prescales` | `.txt` | HLT paths for prescaled triggers|
| `PU`  | `.pkl.gz` or `.histo.root` | Pileup reweight files, matched MC to data| 
| `LSF` | `.histo.root` | Lepton ID/Iso/Reco/Trigger SFs|
| `BTV` | `.csv` or `.root` | b-tagger, c-tagger SFs|
| `JME` | `.txt` | JER, JEC files|

Example in [weight_splitcat.py](https://github.com/cms-rwth/CoffeaRunner/blob/master/config/weight_splitcat.py)

- In case you have correction depends on category, i.e. different ID/objects used in the different cateogries, use `"bycategory":{$category_name:$weight_dict}`

- In case you have correction depends on sample ,i.e. k-factor, use `"bysample":{$sample_name:$weight_nested_dict}`

```
"weights":{
        "common":{
            "inclusive":{
                "lumiMasks":"Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt",
                "PU": "puweight_UL17.histo.root",
                "JME": "mc_compile_jec.pkl.gz",
                "BTV": {
                    "DeepJetC": "DeepJet_ctagSF_Summer20UL17_interp.root",
                },
                "LSF": {
                    "ele_Rereco_above20 EGamma_SF2D": "egammaEffi_ptAbove20.txt_EGM2D_UL2017.histo.root",
                },
        },
            "bycategory":
            {
                "cats":
                    { 
                        "PU": "puweight_UL17.histo.root",
                    }
            }
        },
        "bysample":{
            "gchcWW2L2Nu_4f":{
                "inclusive":{
               
                "JME": "mc_compile_jec.pkl.gz",
            },
            "bycategory":
            {
                "cats2":
                    { 
                        
                        "BTV": {
                            "DeepJetC": "DeepJet_ctagSF_Summer20UL17_interp.root",
                        },
                        
                    }
            }
            }
            
        },
    }
```

##### User config (example from Hpluscharm)
Write your own configurations used in your analysis

```
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
    }
```

#### Executors
Scale out can be notoriously tricky between different sites. Coffea's integration of `slurm` and `dask`
makes this quite a bit easier and for some sites the ``native'' implementation is sufficient, e.g Condor@DESY.
However, some sites have certain restrictions for various reasons, in particular Condor @CERN and @FNAL.

##### Local `executor: "iterative", "futures"`

##### Condor@FNAL (CMSLPC) `executor: "dask/lpc"`
Follow setup instructions at https://github.com/CoffeaTeam/lpcjobqueue. 

##### Condor@CERN (lxplus)  `executor: "dask/lxplus"`
Only one port is available per node, so its possible one has to try different nodes until hitting
one with `8786` being open. Other than that, no additional configurations should be necessary.

##### Coffea-casa (Nebraska AF) `executor:  "dask/casa"`
Coffea-casa is a JupyterHub based analysis-facility hosted at Nebraska. For more information and setup instuctions see
https://coffea-casa.readthedocs.io/en/latest/cc_user.html

After setting up and checking out this repository (either via the online terminal or git widget utility).

Authentication is handled automatically via login auth token instead of a proxy. File paths need to replace xrootd redirector with "xcache", `runner.py` does this automatically.


##### Condor@DESY `executor:  "dask/condor","parsl/condor","parsl/condor/naf_lite"`

Use dask executor with `dask/condor`, but would not as stable as `parsl/condor`.
`parsl/condor/naf_lite` is utilized for lite job scheme for desy condor jobs. (1core, 1.5GB mem, < 3h run time) 

##### Maxwell@DESY `executor:  "parsl/slurm`
For Maxwell you need specific account if you have heavy jobs. 
You need to check [Maxwell-DESY](https://confluence.desy.de/display/MXW/Documentation)

### Profiling

#### CPU profiling
For profiling the CPU time of each function please select the *iterative* processor and then run
python as:
~~~
python -m cProfile -o profiling output.prof  runner.py --cfg profiling/mem.py
~~~
Running on a few files should be enough to get stable results.

After getting the profiler output we analyze it with the [Snakeviz](https://jiffyclub.github.io/snakeviz/)
library
~~~
snakeviz output.prof -s 
~~~
    and open on a browser the link shown by the program.

#### Memory profiling

For memory profiling we use the [memray](https://github.com/bloomberg/memray) library: 

```
python -m memray run -o profiling/memtest.bin runner_wconfig.py --cfg config/example.py
```

the output can be visualized in many ways. One of the most useful is the `flamegraph`: 
```
memray flamegraph profiling/memtest.bin
```

then open the output .html file in you browser to explore the peak memory allocation. 

Alternatively the process can be monitored **live** during execution by doing:
```
memray run --live  runner.py --cfg config/example.py
```
###  Plotting code 


Produce data/MC comparison, shape comparison plots from `.coffea` files, load configuration (`yaml`) files, brief [intro](https://docs.fileformat.com/programming/yaml/) of yaml.

Details of yaml file format would summarized in table below. The **required** info are marked as bold style. 

You can find test file set(`.coffea` and `.yaml`) in `testfile/`. Specify `--debug` to get more info for yaml format.

```
python plotting/plotdataMC.py --cfg testfile/btv_datamc.yml (--debug)
python plotting/comparison.py --cfg testfile/btv_compare.yml (--debug)
```


| Parameter name        | Allowed values               | Description
| :-----:               | :---:                        | :-----------------------------
| **input**(Required)| `list` or `str` <br>(wildcard options `*` accepted)|   input `.coffea` files| 
| **output** (Required)| `str` | output directory of plots with date| 
| **mergemap**(Required)| `dict` | collect sample names, (color, label) setting for file set. details in [map diction](#dict-of-merge-maps-and-comparison-file-lists)|
| **reference** & **compare** (Required) | `dict`| specify the class only for comparison plots |
| **variable**(Required) | `dict` | variables to plot, see [variables section](#variables)|
|com| `str` | √s , default set to be 13TeV|
|inbox_text| `str` | text put in `AnchoredText`|
|log| `str` | log scale on y-axis |
|disable_ratio|`bool`| disable ratio panel for data/MC comparison plot|
|rescale_yields| `dict`| Rescale yields for particular MC collections (no overlay)|
|scale| `dict` | Scale up particular MC collections overlay on the stacked histogram|
|norm| `bool`| noramlized yield to reference sample, only for comparison plot|

#### `dict` of merge maps and comparison file lists


To avoid crowded legend in the plot, we would merge a set of files with similar properties. For example, pT binned DY+jets sample would merge into DY+jets, or diboson (VV) is a collection of WW, WZ and ZZ. Or merge files with similar properties together.


Create a `dict` for each collection under `mergemap`, put the merging sets.

In `plodataMC.py` config files (i.e. `testfile/btv_datamc.yaml`), you can specify the color and label name used in the plot. 

In `comparison.py` config file (`testfile/btv_compare.yaml`),  color and label name and label names are created with `dict` under `reference`  and `compare`. `reference` only accept one entry. 

```yaml
## plodataMC.py
mergemap:
    DY: # key name of file collections
        list: # collections of files(key of dataset name in coffea file)
            - "DYJetsToLL_M-50_TuneCP5_13p6TeV-madgraphMLM-pythia8"
            - "DYJetsToLL_M-10to50_TuneCP5_13p6TeV-madgraphMLM-pythia8"
        label : "DYjet" #Optional, if not exist would take key name of the list. i.e. DY in this case
        color : '#eb4034' #Optional, color use for this category. 
    VV: 
        list:
            - "WW_TuneCP5_13p6TeV-pythia8" 
            - "WZ_TuneCP5_13p6TeV-pythia8"
            - "ZZ_TuneCP5_13p6TeV-pythia8"
## comparison.py
mergemap :  
    runC: 
        list : 
            - "Muon_Run2022C-PromptReco-v1"
            - "SingleMuon_Run2022C-PromptReco-v1"
    Muon_Run2022C-PromptReco-v1: 
        list : 
            - "Muon_Run2022C-PromptReco-v1"  
reference: 
    Muon_Run2022C-PromptReco-v1: 
        label: RunC  #Optional, label name
        color : 'b' #Optional
        
compare: 
    # if not specify anything, leave empty value for key
    Muon_Run2022D-PromptReco-v1: 
    Muon_Run2022D-PromptReco-v2: 
```

#### Variables 

Common definitions for both usage, use default settings if leave empty value for the keys. 
:bangbang: `blind` option is only used in the data/MC comparison plots to blind particular observable like BDT score. 

|Option| Default |
|:-----: |:---:   |
| `xlabel` | take name of `key` |
| `axis` | `sum` over all the axes |
| `rebin` | no rebinning |
| `blind` | no blind region | 

```yaml
## specify variable to plot
    btagDeepFlavB_0:
        # Optional, set x label of variable
        xlabel: "deepJet Prob(b)" 
        # Optional, specify hist axis with dict
        axis : 
            syst: noSF # Optional, access bin
            flav : sum # Optional, access bin, can convert sum to sum operation later
        # Optional, rebin variable 
        rebin :  
            # Optional, you can specify the rebin axis with rebin value
            # discr: 2
            # or just put a number, would rebin distribution the last axis (usually the variable)
            2
        # Optional(only for data/MC), blind variables
        blind : -10, #blind variable[-10:], if put -10,-5 would blind variable[-10:-5]
        
    ## specify variable, if not specify anything, leave empty value for key
    btagDeepFlavC_0:

    ## Accept wildcard option
    # only axis and rebin can be specify here
    btagDeepFlav* :
        axis : 
            syst: noSF
            flav : sum
        rebin :  
            discr: 2
    # Use "all" will produce plots for all the variables
    # only rebin of last axis (variable-axis) can be specify here
    all: 
        rebin: 2
``` 

### Running jupyter remotely
See also https://hackmd.io/GkiNxag0TUmHnnCiqdND1Q#Remote-jupyter

1. On your local machine, edit `.ssh/config`:
```
Host lxplus*
  HostName lxplus7.cern.ch
  User <your-user-name>
  ForwardX11 yes
  ForwardAgent yes
  ForwardX11Trusted yes
Host *_f
  LocalForward localhost:8800 localhost:8800
  ExitOnForwardFailure yes
```
2. Connect to remote with `ssh lxplus_f`
3. Start a jupyter notebook:
```
jupyter notebook --ip=127.0.0.1 --port 8800 --no-browser
```
4. URL for notebook will be printed, copy and open in local browser
