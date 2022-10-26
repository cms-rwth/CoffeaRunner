
# CoffeaRunner
[![Linting](https://github.com/cms-rwth/CoffeaRunner/blob/master/.github/workflows/python_linting.yml/badge.svg)](https://github.com/cms-rwth/CoffeaRunner/blob/master/.github/workflows/python_linting.yml)
[![Test Workflow](https://github.com/cms-rwth/CoffeaRunner/blob/master/.github/workflows/test_workflow.yml/badge.svg)](https://github.com/cms-rwth/CoffeaRunner/blob/master/.github/workflows/test_workflow.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
Generalized framework columnar-based analysis with [coffea](https://coffeateam.github.io/coffea/) based on the developments from [BTVNanoCommissioning](https://github.com/cms-btv-pog/BTVNanoCommissioning) and some development from [PocketCoffea](https://github.com/PocketCoffea/PocketCoffea)

## Requirements

### Setup for python=3.7

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

You can either use the default environment `base` or create a new one:

```bash
# create new environment with python 3.7, e.g. environment of name `CoffeaRunner`
conda create --name CoffeaRunner python=3.7
# activate environment `CoffeaRunner`
conda activate CoffeaRunner
```

Install manually for the required packages:
```
pip install coffea
conda install -c conda-forge xrootd
conda install -c conda-forge ca-certificates
conda install -c conda-forge ca-policy-lcg
conda install -c conda-forge dask-jobqueue
conda install -c anaconda bokeh 
conda install -c conda-forge 'fsspec>=0.3.3'
conda install dask
conda install -c conda-forge parsl
```

You could simply create the environment through the existing `test_env.yml` under your conda environment
```
conda env create -f test_env.yml 
```


create new environment with python 3.7, e.g. environment of name `CoffeaRunner`

`conda create --name CoffeaRunner python=3.7`

Once the environment is set up, compile the python package:
```
pip install -e .
```

#### Other installation options for coffea
See https://coffeateam.github.io/coffea/installation.html

#### Running jupyter remotely
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
### Make the dataset json files

Use the `fetch.py` in `filefetcher`, the `$input_DAS_list` is the info extract from DAS, and output json files in `metadata/`. Default site is `prod/global`, use `prod/phys03` for personal productions.

```
python fetch.py --input ${input_DAS_list} --output ${output_json_name} --site ${site}
```

### Create compiled corretions file(`pkl.gz`)

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
| `json`            | string (required)            | Path of .json file to create with NanoAOD files (can load multiple files)
| `workflow`        | workflows (required)         | Workflow to run
| `output`          | string  (required)           | Path of output folder
| `executor`        | string (required)            | See [executor](#executors) below 
| `workers`         | int                          | Number of parallel threads (with futures)
| `scaleout`        | int                          | Number of jobs to submit (with parsl/slurm)
| `chunk`           | int                          | Chunk size
| `max`             | int                          | Maximum number of chunks to process
| `skipbadfiles`    | bool                         | Skip bad files
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
Nested dictionary with weights. Notice the `category` need to be specify if you use `bycategory` in the weight list
Example in [weight_splitcat.py](https://github.com/cms-rwth/CoffeaRunner/blob/master/config/weight_splitcat.py)

- In case you have correction depends on category, i.e. different ID/objects used in the different cateogries, use `"bycategory":{$category_name:$weight_dict}`

- In case you have correction depends on sample ,i.e. k-factor, use `"bysample":{$sample_name:$weight_nested_dict}`

```
"weights":{
        "common":{
            "inclusive":{
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
### :construction:  Plotting code (not generalize...)



- data/MC comparison code from BTV:
Prodcuce data/MC comparisons
```
python plotdataMC.py -i a.coffea,b.coffea --lumi 41900 -p dilep_sf -d zmass,z_pt

optional arguments:
  --lumi LUMI           luminosity in /pb
  -p {dilep_sf,ttsemilep_sf,ctag_Wc_sf,ctag_DY_sf,ctag_ttsemilep_sf,ctag_ttdilep_sf}, --phase {dilep_sf,ttsemilep_sf,ctag_Wc_sf,ctag_DY_sf,ctag_ttsemilep_sf,ctag_ttdilep_sf}
                        which workflows
  --log LOG             log on x axis
  --norm NORM           Use for reshape SF, scale to same yield as no SFs case
  -d DISCR_LIST, --discr_list DISCR_LIST
                        discriminators
  --SF                  make w/, w/o SF comparisons
  --ext EXT             prefix output file
  -i INPUT, --input INPUT
                        input coffea files (str), splitted different files with ,
```
- data/data, MC/MC comparison from BTV
```
python comparison.py -i a.coffea,b.coffea -p dilep_sf -d zmass,z_pt

python -m plotting.comparison --phase ctag_ttdilep_sf --output ctag_ttdilep_sf -r 2017_runB -c 2017_runC,2017_runD -d zmass, z_pt (--sepflav True/False)
optional arguments:
  -p {dilep_sf,ttsemilep_sf,ctag_Wc_sf,ctag_DY_sf,ctag_ttsemilep_sf,ctag_ttdilep_sf}, --phase {dilep_sf,ttsemilep_sf,ctag_Wc_sf,ctag_DY_sf,ctag_ttsemilep_sf,ctag_ttdilep_sf}
                        which phase space
  -i INPUT, --input INPUT
                        files set
  -r REF, --ref REF     referance dataset
  -c COMPARED, --compared COMPARED
                        compared dataset
  --sepflav SEPFLAV     seperate flavour
  --log                 log on x axis
  -d DISCR_LIST [DISCR_LIST ...], --discr_list DISCR_LIST [DISCR_LIST ...]
                        discriminators
  --ext EXT             prefix name
```
