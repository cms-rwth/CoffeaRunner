
# CoffeaRunner

[![Linting](https://github.com/cms-rwth/CoffeaRunner/actions/workflows/python_linting.yml/badge.svg)](https://github.com/cms-rwth/CoffeaRunner/actions/workflows/python_linting.yml)
[![Test Workflow](https://github.com/cms-rwth/CoffeaRunner/actions/workflows/test_workflow.yml/badge.svg)](https://github.com/cms-rwth/CoffeaRunner/actions/workflows/test_workflow.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Generalized framework columnar-based analysis with [coffea](https://coffeateam.github.io/coffea/) based on the developments from [BTVNanoCommissioning](https://github.com/cms-btv-pog/BTVNanoCommissioning) and some development from [PocketCoffea](https://github.com/PocketCoffea/PocketCoffea)

## Requirements

### Setup 

:heavy_exclamation_mark: Install under `bash` environment

```bash
# only first time 
git clone --recursive git@github.com:cms-rwth/CoffeaRunner.git
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

<details><summary>Not necessary to run framework, but helpful when identifying corrupted files and tracking the progress during said task:</summary>
<p>

```
conda install -c conda-forge p-tqdm
```
    
</p>
</details>
<br>

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
<<<<<<< HEAD
python filefetcher/fetch.py --input filefetcher/input_DAS_list.txt --output output_name.json
=======
--wf {validation,ttcom,ttdilep_sf,ttsemilep_sf,emctag_ttdilep_sf,ctag_ttdilep_sf,ectag_ttdilep_sf,ctag_ttsemilep_sf,ectag_ttsemilep_sf,ctag_Wc_sf,ectag_Wc_sf,ctag_DY_sf,ectag_DY_sf}, --workflow {validation,ttcom,ttdilep_sf,ttsemilep_sf,emctag_ttdilep_sf,ctag_ttdilep_sf,ectag_ttdilep_sf,ctag_ttsemilep_sf,ectag_ttsemilep_sf,ctag_Wc_sf,ectag_Wc_sf,ctag_DY_sf,ectag_DY_sf}
                        Which processor to run
  -o OUTPUT, --output OUTPUT
                        Output histogram filename (default: hists.coffea)
  --samples SAMPLEJSON, --json SAMPLEJSON
                        JSON file containing dataset and file locations
                        (default: dummy_samples.json)
  --year YEAR           Year
  --campaign CAMPAIGN   Dataset campaign, change the corresponding correction
                        files{ "Rereco17_94X","Winter22Run3","2018_UL","2017_UL","2016preVFP_UL","2016postVFP_UL"}
  --isCorr              Run with SFs
  --isJERC              JER/JEC implemented to jet
  --isSyst              Run with systematics for SF
  --executor {iterative,futures,parsl/slurm,parsl/condor,parsl/condor/naf_lite,dask/condor,dask/slurm,dask/lpc,dask/lxplus,dask/casa}
                        The type of executor to use (default: futures). 
  -j WORKERS, --workers WORKERS
                        Number of workers (cores/threads) to use for multi- worker executors (e.g. futures or condor) (default:
                        3)
  -s SCALEOUT, --scaleout SCALEOUT
                        Number of nodes to scale out to if using slurm/condor.
                        Total number of concurrent threads is ``workers x
                        scaleout`` (default: 6)
  --memory MEMORY       Memory used in jobs (in GB) ``(default: 4GB)
  --disk DISK           Disk used in jobs  ``(default: 4GB)
  --voms VOMS           Path to voms proxy, made accessible to worker nodes.
                        By default a copy will be made to $HOME.
  --chunk N             Number of events per process chunk
  --retries N           Number of retries for coffea processor
 --index INDEX         (Specific for dask/lxplus file splitting, default:0,0) 
                        Format: $dictindex,$fileindex. $dictindex refers to the index of the file list split to 50 files per dask-worker.
                        The job will start submission from the corresponding indices
  --validate            Do not process, just check all files are accessible
  --skipbadfiles        Skip bad files.
  --only ONLY           Only process specific dataset or file
  --limit N             Limit to the first N files of each dataset in sample
                        JSON
  --max N               Max number of chunks to run in total
>>>>>>> ca74d50... feat: correctionlib(jsonpog-integration) implementation & fixes on actions (#50)
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



| Parameter              | Nested component          | Type           | Description                                                                                                                  | Default     |
|------------------------|---------------------------|----------------|------------------------------------------------------------------------------------------------------------------------------|-------------|
| **dataset(required)**  |                           | dict           | Dataset configurations                                                                                                       |             |
|                        | **jsons<br>  (required)** | string         | Path of `.json` file to create with NanoAOD files (can load multiple files)                                                  |             |
|                        | **campaign(required)**    | string         | Campaign name                                                                                                                |             |
|                        | **year(required)**        | string         | Year flag                                                                                                                    |             |
|                        | filter                    | dict           | Create the list of `samples`, `samples_exclude` with the dataset name(key name stored in json file)                          | `None`      |
| **workflow(required)** |                           | python modules | Analysis workflows                                                                                                           |             |
| **output(requred)**    |                           | string         | Output directory name, create version tag                                                                                    |             |
| run_options            |                           | dict           | Collections of run options                                                                                                   |             |
|                        | executor                  | string         | Executor for coffea jobs, see details in [executor](#executors)                                                              | `iterative` |
|                        | limit                     | int            | Maximum number of files per sample to process                                                                                | `1`         |
|                        | max                       | int            | Maximum number of chunks to process                                                                                          | `None`      |
|                        | chunk                     | int            | Chunk size, numbers of events to be processed each time. Maximum number is the default sample size                           | `50000`     |
|                        | workers                   | int            | Number of parallel threads (with `futures` and clusters without fixed workers)                                               | `2`         |
|                        | mem_per_worker            | int            | Set memory for `condor/slurm` jobs                                                                                           | `2`         |
|                        | scaleout                  | int            | Number of jobs to submit, use for cluster                                                                                    | `20`        |
|                        | walltime                  | time           | Wall time for `condor/slurm` jobs                                                                                            | `03:00:00`  |
|                        | retries                   | int            | Numbers of retries to submit failure jobs. Usually deal with xrootd temporary failures                                       | `20`        |
|                        | voms                      | Path           | Path to your `x509 proxy`                                                                                                    | `None`      |
|                        | skipbadfiles              | bool           | Skip bad files where not exist or broken(BE CAREFUL WITH DATA)                                                               | `False`     |
|                        | splitjobs                 | bool           | Split `executor` and `accumulator` to separate jobs to avoid local memory consumption become too large                       | `True`      |
|                        | compression               | int            | Compression level of output with `lz4`                                                                                       | `3`         |
| categories             |                           | dict           | Dictionary of categories with cuts to apply*                                                                                 | `None`      |
| preselections          |                           | dict           | List of preselection cuts, use for all the categories                                                                        | `None`      |
| weights                |                           | dict           | Nested `dict` for correction files. Details and example in [weights](#####Weights)                                           | `None`      |
|                        | common                    | dict           | Specify weights apply for all the events(`inclusive`) or category specific(`by category`)                                    |             |
|                        | bysample                  | dict           | Weights only apply for particular sample, can be applied for all the events(`inclusive`) or category specific(`by category`) |             |
| systematic             |                           | dict           | `dict` for systematic uncertainty                                                                                            | `None`      |
|                        | isJERC                    | bool           | Run JER, JEC, MET scale uncertainty                                                                                          |             |
|                        | weights                   | bool           | Weight files with up/down variations                                                                                         |             |
| userconfig             |                           | dict           | Dictionary of user specific configuration, depends on workflow                                                               |             |

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

<details><summary>example with customize weight files
</summary>
<p>

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
</p>
</details>


-  Use central maintained jsonpog-integration 
The official correction files collected in [jsonpog-integration](https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration) is updated by POG except `lumiMask` and `JME` still updated by maintainer. No longer to request input files in the `correction_config`.  

<details><summary>See the example with `2017_UL`.</summary>
<p>

```python
  "2017_UL": {
        # Same with custom config
        "lumiMask": "Cert_294927-306462_13TeV_UL2017_Collisions17_MuonJSON.txt",
        "JME": "jec_compiled.pkl.gz",
        # no config need to be specify for PU weights
        "PU": None,
        # Btag SFs - specify $TAGGER : $TYPE-> find [$TAGGER_$TYPE] in json file
        "BTV": {"deepCSV": "shape", "deepJet": "shape"},
        
        "LSF": {
        # Electron SF - Following the scheme: "${SF_name} ${year}": "${WP}"
        # https://github.com/cms-egamma/cms-egamma-docs/blob/master/docs/EgammaSFJSON.md
            "ele_ID 2017": "wp90iso",
            "ele_Reco 2017": "RecoAbove20",

        # Muon SF - Following the scheme: "${SF_name} ${year}": "${WP}"
        # WPs : ['NUM_GlobalMuons_DEN_genTracks', 'NUM_HighPtID_DEN_TrackerMuons', 'NUM_HighPtID_DEN_genTracks', 'NUM_IsoMu27_DEN_CutBasedIdTight_and_PFIsoTight', 'NUM_LooseID_DEN_TrackerMuons', 'NUM_LooseID_DEN_genTracks', 'NUM_LooseRelIso_DEN_LooseID', 'NUM_LooseRelIso_DEN_MediumID', 'NUM_LooseRelIso_DEN_MediumPromptID', 'NUM_LooseRelIso_DEN_TightIDandIPCut', 'NUM_LooseRelTkIso_DEN_HighPtIDandIPCut', 'NUM_LooseRelTkIso_DEN_TrkHighPtIDandIPCut', 'NUM_MediumID_DEN_TrackerMuons', 'NUM_MediumID_DEN_genTracks', 'NUM_MediumPromptID_DEN_TrackerMuons', 'NUM_MediumPromptID_DEN_genTracks', 'NUM_Mu50_or_OldMu100_or_TkMu100_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose', 'NUM_SoftID_DEN_TrackerMuons', 'NUM_SoftID_DEN_genTracks', 'NUM_TightID_DEN_TrackerMuons', 'NUM_TightID_DEN_genTracks', 'NUM_TightRelIso_DEN_MediumID', 'NUM_TightRelIso_DEN_MediumPromptID', 'NUM_TightRelIso_DEN_TightIDandIPCut', 'NUM_TightRelTkIso_DEN_HighPtIDandIPCut', 'NUM_TightRelTkIso_DEN_TrkHighPtIDandIPCut', 'NUM_TrackerMuons_DEN_genTracks', 'NUM_TrkHighPtID_DEN_TrackerMuons', 'NUM_TrkHighPtID_DEN_genTracks']

            "mu_Reco 2017_UL": "NUM_TrackerMuons_DEN_genTracks",
            "mu_HLT 2017_UL": "NUM_IsoMu27_DEN_CutBasedIdTight_and_PFIsoTight",
            "mu_ID 2017_UL": "NUM_TightID_DEN_TrackerMuons",
            "mu_Iso 2017_UL": "NUM_TightRelIso_DEN_TightIDandIPCut",
        },
    },
```

</p>
</details>

##### Systematic 

Specify whether run systematics or not

```
"systematics": 
        {
            "JERC":False,
            "weights":False,
        }
```
##### User config (example from Hpluscharm)
Write your own configurations used in your analysis

```
"userconfig":{
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

All the `lumiMask`, correction files (SFs, pileup weight), and JEC, JER files are under  `BTVNanoCommissioning/src/data/` following the substructure `${type}/${campaign}/${files}`(except `lumiMasks` and `Prescales`)

<<<<<<< HEAD
Produce data/MC comparison, shape comparison plots from `.coffea` files, load configuration (`yaml`) files, brief [intro](https://docs.fileformat.com/programming/yaml/) of yaml.
=======
## Correction files configurations
:heavy_exclamation_mark:  If the correction files are not supported yet by jsonpog-integration, you can still try with custom input data.

### Options with custom input data 
>>>>>>> ca74d50... feat: correctionlib(jsonpog-integration) implementation & fixes on actions (#50)

Details of yaml file format would summarized in table below. Information used in data/MC script would marked with () and comparsion script with (). The **required** info are marked as bold style. 


<details><summary>Take `Rereco17_94X` as an example.</summary>
<p>

```
python plotting/plotdataMC.py --cfg testfile/btv_datamc.yml (--debug)
python plotting/comparison.py --cfg testfile/btv_compare.yml (--debug)
```

</p>
</details>

### Use central maintained jsonpog-integration 
The official correction files collected in [jsonpog-integration](https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration) is updated by POG except `lumiMask` and `JME` still updated by maintainer. No longer to request input files in the `correction_config`.  

<<<<<<< HEAD
<details><summary>See the example with `2017_UL`.</summary>
=======

<details><summary>Take `Rereco17_94X` as an example.</summary>
>>>>>>> ca74d50... feat: correctionlib(jsonpog-integration) implementation & fixes on actions (#50)
<p>

```python
  "2017_UL": {
        # Same with custom config
        "lumiMask": "Cert_294927-306462_13TeV_UL2017_Collisions17_MuonJSON.txt",
        "JME": "jec_compiled.pkl.gz",
        # no config need to be specify for PU weights
        "PU": None,
        # Btag SFs - specify $TAGGER : $TYPE-> find [$TAGGER_$TYPE] in json file
        "BTV": {"deepCSV": "shape", "deepJet": "shape"},
        
        "LSF": {
        # Electron SF - Following the scheme: "${SF_name} ${year}": "${WP}"
        # https://github.com/cms-egamma/cms-egamma-docs/blob/master/docs/EgammaSFJSON.md
            "ele_ID 2017": "wp90iso",
            "ele_Reco 2017": "RecoAbove20",

        # Muon SF - Following the scheme: "${SF_name} ${year}": "${WP}"
        # WPs : ['NUM_GlobalMuons_DEN_genTracks', 'NUM_HighPtID_DEN_TrackerMuons', 'NUM_HighPtID_DEN_genTracks', 'NUM_IsoMu27_DEN_CutBasedIdTight_and_PFIsoTight', 'NUM_LooseID_DEN_TrackerMuons', 'NUM_LooseID_DEN_genTracks', 'NUM_LooseRelIso_DEN_LooseID', 'NUM_LooseRelIso_DEN_MediumID', 'NUM_LooseRelIso_DEN_MediumPromptID', 'NUM_LooseRelIso_DEN_TightIDandIPCut', 'NUM_LooseRelTkIso_DEN_HighPtIDandIPCut', 'NUM_LooseRelTkIso_DEN_TrkHighPtIDandIPCut', 'NUM_MediumID_DEN_TrackerMuons', 'NUM_MediumID_DEN_genTracks', 'NUM_MediumPromptID_DEN_TrackerMuons', 'NUM_MediumPromptID_DEN_genTracks', 'NUM_Mu50_or_OldMu100_or_TkMu100_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose', 'NUM_SoftID_DEN_TrackerMuons', 'NUM_SoftID_DEN_genTracks', 'NUM_TightID_DEN_TrackerMuons', 'NUM_TightID_DEN_genTracks', 'NUM_TightRelIso_DEN_MediumID', 'NUM_TightRelIso_DEN_MediumPromptID', 'NUM_TightRelIso_DEN_TightIDandIPCut', 'NUM_TightRelTkIso_DEN_HighPtIDandIPCut', 'NUM_TightRelTkIso_DEN_TrkHighPtIDandIPCut', 'NUM_TrackerMuons_DEN_genTracks', 'NUM_TrkHighPtID_DEN_TrackerMuons', 'NUM_TrkHighPtID_DEN_genTracks']

            "mu_Reco 2017_UL": "NUM_TrackerMuons_DEN_genTracks",
            "mu_HLT 2017_UL": "NUM_IsoMu27_DEN_CutBasedIdTight_and_PFIsoTight",
            "mu_ID 2017_UL": "NUM_TightID_DEN_TrackerMuons",
            "mu_Iso 2017_UL": "NUM_TightRelIso_DEN_TightIDandIPCut",
        },
    },
```

</p>
</details>

<<<<<<< HEAD
=======
### Use central maintained jsonpog-integration 
The official correction files collected in [jsonpog-integration](https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration) is updated by POG except `lumiMask` and `JME` still updated by maintainer. No longer to request input files in the `correction_config`.  

<details><summary>See the example with `2017_UL`.</summary>
<p>

```python
  "2017_UL": {
        # Same with custom config
        "lumiMask": "Cert_294927-306462_13TeV_UL2017_Collisions17_MuonJSON.txt",
        "JME": "jec_compiled.pkl.gz",
        # no config need to be specify for PU weights
        "PU": None,
        # Btag SFs - specify $TAGGER : $TYPE-> find [$TAGGER_$TYPE] in json file
        "BTV": {"deepCSV": "shape", "deepJet": "shape"},
        
        "LSF": {
        # Electron SF - Following the scheme: "${SF_name} ${year}": "${WP}"
        # https://github.com/cms-egamma/cms-egamma-docs/blob/master/docs/EgammaSFJSON.md
            "ele_ID 2017": "wp90iso",
            "ele_Reco 2017": "RecoAbove20",

        # Muon SF - Following the scheme: "${SF_name} ${year}": "${WP}"
        # WPs : ['NUM_GlobalMuons_DEN_genTracks', 'NUM_HighPtID_DEN_TrackerMuons', 'NUM_HighPtID_DEN_genTracks', 'NUM_IsoMu27_DEN_CutBasedIdTight_and_PFIsoTight', 'NUM_LooseID_DEN_TrackerMuons', 'NUM_LooseID_DEN_genTracks', 'NUM_LooseRelIso_DEN_LooseID', 'NUM_LooseRelIso_DEN_MediumID', 'NUM_LooseRelIso_DEN_MediumPromptID', 'NUM_LooseRelIso_DEN_TightIDandIPCut', 'NUM_LooseRelTkIso_DEN_HighPtIDandIPCut', 'NUM_LooseRelTkIso_DEN_TrkHighPtIDandIPCut', 'NUM_MediumID_DEN_TrackerMuons', 'NUM_MediumID_DEN_genTracks', 'NUM_MediumPromptID_DEN_TrackerMuons', 'NUM_MediumPromptID_DEN_genTracks', 'NUM_Mu50_or_OldMu100_or_TkMu100_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose', 'NUM_SoftID_DEN_TrackerMuons', 'NUM_SoftID_DEN_genTracks', 'NUM_TightID_DEN_TrackerMuons', 'NUM_TightID_DEN_genTracks', 'NUM_TightRelIso_DEN_MediumID', 'NUM_TightRelIso_DEN_MediumPromptID', 'NUM_TightRelIso_DEN_TightIDandIPCut', 'NUM_TightRelTkIso_DEN_HighPtIDandIPCut', 'NUM_TightRelTkIso_DEN_TrkHighPtIDandIPCut', 'NUM_TrackerMuons_DEN_genTracks', 'NUM_TrkHighPtID_DEN_TrackerMuons', 'NUM_TrkHighPtID_DEN_genTracks']

            "mu_Reco 2017_UL": "NUM_TrackerMuons_DEN_genTracks",
            "mu_HLT 2017_UL": "NUM_IsoMu27_DEN_CutBasedIdTight_and_PFIsoTight",
            "mu_ID 2017_UL": "NUM_TightID_DEN_TrackerMuons",
            "mu_Iso 2017_UL": "NUM_TightRelIso_DEN_TightIDandIPCut",
        },
    },
```

</p>
</details>

>>>>>>> ca74d50... feat: correctionlib(jsonpog-integration) implementation & fixes on actions (#50)
## Create compiled JERC file(`pkl.gz`)

| Parameter name        | Allowed values               | Description
| :-----:               | :---:                        | :-----------------------------
| **input**(Required)| `list` or `str` <br>(wildcard options `*` accepted)|   input `.coffea` files| 
| **output** (Required)| `str` | output directory of plots with date| 
| **mergemap**(Required)| `dict` | collect sample names, (color, label) setting for file set. details in [map diction](#dict-of-merge-maps-and-comparison-file-lists)|
| **reference** & **compare** (Required) | `dict`| specify the class for comparison plots |
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

<details><summary>Code snipped</summary>
<p>

<<<<<<< HEAD
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

</p>
</details>
=======
:new: non-uniform rebinning is possible, specify the bins with  list of edges `--autorebin 50,80,81,82,83,100.5`

```
python plotdataMC.py -i a.coffea,b.coffea --lumi 41500 -p dilep_sf -v z_mass,z_pt 
python plotdataMC.py -i "test*.coffea" --lumi 41500 -p dilep_sf -v z_mass,z_pt 

options:
  -h, --help            show this help message and exit
  --lumi LUMI           luminosity in /pb
  --com COM             sqrt(s) in TeV
  -p {dilep_sf,ttsemilep_sf,ctag_Wc_sf,ctag_DY_sf,ctag_ttsemilep_sf,ctag_ttdilep_sf}, --phase {dilep_sf,ttsemilep_sf,ctag_Wc_sf,ctag_DY_sf,ctag_ttsemilep_sf,ctag_ttdilep_sf}
                        which phase space
  --log LOG             log on y axis
  --norm NORM           Use for reshape SF, scale to same yield as no SFs case
  -v VARIABLE, --variable VARIABLE
                        variables to plot, splitted by ,. Wildcard option * available as well. Specifying `all` will run through all variables.
  --SF                  make w/, w/o SF comparisons
  --ext EXT             prefix name
  -i INPUT, --input INPUT
                        input coffea files (str), splitted different files with ','. Wildcard option * available as well.
   --autorebin AUTOREBIN
                        Rebin the plotting variables, input `int` or `list`. int: merge N bins. list of number: rebin edges(non-uniform bin is possible)
   --xlabel XLABEL      rename the label for x-axis
   --ylabel YLABEL      rename the label for y-axis
   --splitOSSS SPLITOSSS 
                        Only for W+c phase space, split opposite sign(1) and same sign events(-1), if not specified, the combined OS-SS phase space is used
   --xrange XRANGE      custom x-range, --xrange xmin,xmax
   --flow FLOW 
                        str, optional {None, 'show', 'sum'} Whether plot the under/overflow bin. If 'show', add additional under/overflow bin. If 'sum', add the under/overflow bin content to first/last bin.
```
- data/data, MC/MC comparisons
>>>>>>> ca74d50... feat: correctionlib(jsonpog-integration) implementation & fixes on actions (#50)

#### Variables 

Common definitions for both usage, use default settings if leave empty value for the keys. 
:bangbang: `blind` option is only used in the data/MC comparison plots to blind particular observable like BDT score. 

<<<<<<< HEAD
|Option| Default |
|:-----: |:---:   |
| `xlabel` | take name of `key` |
| `axis` | `sum` over all the axes |
| `rebin` | no rebinning |
| `blind` | no blind region | 

<details><summary>Code snipped</summary>
<p>

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
            # One can try non-uniform  rebin now! you can specify the rebin axis with rebin value
            #discr : [-0.2,0.04,0.2,0.4,0.48,0.6,0.64,0.68,0.72,0.76,0.8,0.84,0.88,0.92,0.96,1.]
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
=======
options:
  -h, --help            show this help message and exit
  -p {dilep_sf,ttsemilep_sf,ctag_Wc_sf,ctag_DY_sf,ctag_ttsemilep_sf,ctag_ttdilep_sf}, --phase {dilep_sf,ttsemilep_sf,ctag_Wc_sf,ctag_DY_sf,ctag_ttsemilep_sf,ctag_ttdilep_sf}
                        which phase space
  -i INPUT, --input INPUT
                        input coffea files (str), splitted different files with ','. Wildcard option * available as well.
  -r REF, --ref REF     referance dataset
  -c COMPARED, --compared COMPARED
                        compared datasets, splitted by ,
  --sepflav SEPFLAV     seperate flavour(b/c/light)
  --log                 log on y axis
  -v VARIABLE, --variable VARIABLE
                        variables to plot, splitted by ,. Wildcard option * available as well. Specifying `all` will run through all variables.
  --ext EXT             prefix name
  --com COM             sqrt(s) in TeV
  --shortref SHORTREF   short name for reference dataset for legend
  --shortcomp SHORTCOMP
                        short names for compared datasets for legend, split by ','
   --autorebin AUTOREBIN
                        Rebin the plotting variables, input `int` or `list`. int: merge N bins. list of number: rebin edges(non-uniform bin is possible)
   --xlabel XLABEL      rename the label for x-axis
   --ylabel YLABEL      rename the label for y-axis
   --norm               compare shape, normalized yield to reference
   --xrange XRANGE       custom x-range, --xrange xmin,xmax
   --flow FLOW 
                        str, optional {None, 'show', 'sum'} Whether plot the under/overflow bin. If 'show', add additional under/overflow bin. If 'sum', add the under/overflow bin content to first/last bin.
```
>>>>>>> ca74d50... feat: correctionlib(jsonpog-integration) implementation & fixes on actions (#50)

</p>
</details>

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
