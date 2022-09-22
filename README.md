
# BTVNanoCommissioning
[![Linting](https://github.com/cms-btv-pog/BTVNanoCommissioning/actions/workflows/python_linting.yml/badge.svg)](https://github.com/cms-btv-pog/BTVNanoCommissioning/actions/workflows/python_linting.yml)
[![TTbar](https://github.com/cms-btv-pog/BTVNanoCommissioning/actions/workflows/ttbar_workflow.yml/badge.svg)](https://github.com/cms-btv-pog/BTVNanoCommissioning/actions/workflows/ttbar_workflow.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Repository for Commissioning studies in the BTV POG based on (custom) nanoAOD samples

## Requirements
### Setup 

:heavy_exclamation_mark: Install under `bash` environment

```
# only first time 
git clone git@github.com:cms-rwth/CoffeaRunner.git

# activate enviroment once you have coffea framework 
conda activate coffea
```

### Coffea installation with Miniconda
For installing Miniconda, see also https://hackmd.io/GkiNxag0TUmHnnCiqdND1Q#Local-or-remote
``` 
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
# Run and follow instructions on screen
bash Miniconda3-latest-Linux-x86_64.sh
```
NOTE: always make sure that conda, python, and pip point to local Miniconda installation (`which conda` etc.).

You can either use the default environment `base` or create a new one:

```
# create new environment with python 3.7, e.g. environment of name `CoffeaRunner`
conda create --name CoffeaRunner python=3.7
# activate environment `CoffeaRunner`
conda activate CoffeaRunner
```

Install manually for the required packages:
```
pip install coffea==0.7.14
conda install -c conda-forge xrootd
conda install -c conda-forge ca-certificates
conda install -c conda-forge ca-policy-lcg
conda install -c conda-forge dask-jobqueue
conda install -c anaconda bokeh 
conda install -c conda-forge 'fsspec>=0.3.3'
conda install dask
pip install arrow==1.2.2
pip install black==22.3.0
```

Once the environment is set up, compile the python package:
```
pip install -e .
```

### Other installation options for coffea
See https://coffeateam.github.io/coffea/installation.html

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



## Structure


```
voms-proxy-init --voms cms --vomses ~/.grid-security/vomses
```



## Make the json files

Use the `fetch.py` in `filefetcher`, the `$input_DAS_list` is the info extract from DAS, and output json files in `metadata/`

```
python fetch.py --input ${input_DAS_list} --output ${output_json_name} --site ${site}
```

## Create compiled corretions file(`pkl.gz`)

Compile correction pickle files for a specific JEC campaign by changing the dict of jet_factory, and define the MC campaign and the output file name by passing it as arguments to the python script:

```
python -m utils.compile_jec UL17_106X data/JME/UL17_106X/jec_compiled.pkl.gz
```
## Config file
The config file in `.py` format is passed as the argument `--cfg` of the `runner_wconfig.py` script. The file has the following structure:

| Parameter name    | Allowed values               | Description
| :-----:           | :---:                        | :------------------------------------------
| `dataset`         | string                       | Path of .txt file with list of DAS datasets
| `json`            | string                       | Path of .json file to create with NanoAOD files
| `storage_prefix`  | string                       | Path of storage folder to save datasets
| `workflow`        | See workflow below           | Workflow to run
| `input`           | string                       | Path of .json file, input to the workflow
| `output`          | string                       | Path of output folder
| `executor`        | See executor below           | Executor
| `workers`         | int                          | Number of parallel threads (with futures)
| `scaleout`        | int                          | Number of jobs to submit (with parsl/slurm)
| `chunk`           | int                          | Chunk size
| `max`             | int                          | Maximum number of chunks to process
| `skipbadfiles`    | bool                         | Skip bad files
| `voms`            | string                       | Voms parameters (with condor)
| `limit`           | int                          | Maximum number of files per sample to process
| `preselections`   | list                         | List of preselection cuts
| `categories`      | dict                         | Dictionary of categories with cuts to apply

The variables' names can be chosen among those reported in `parameters.allhistograms.histogram_settings`, which contains also the default values of the plotting parameters. If no plotting parameters are specified, the default ones will be used.

The plotting parameters can be customized for plotting, for example to rebin the histograms. In case of rebinning, the binning used in the plots has to be compatible with the one of the input histograms.

The `Cut` objects listed in `preselections` and `categories` have to be defined in `parameters.cuts.baseline_cuts`. A library of pre-defined functions for event is available in `lib.cut_functions`, but custom functions can be defined in a separate file.
### Executors

Scale out can be notoriously tricky between different sites. Coffea's integration of `slurm` and `dask`
makes this quite a bit easier and for some sites the ``native'' implementation is sufficient, e.g Condor@DESY.
However, some sites have certain restrictions for various reasons, in particular Condor @CERN and @FNAL.
#### Local `executor: "iterative", "futures"`

#### Condor@FNAL (CMSLPC) `executor: "dask/lpc"`
Follow setup instructions at https://github.com/CoffeaTeam/lpcjobqueue. After starting 
the singularity container run with 


#### Condor@CERN (lxplus)  `executor: "dask/lxplus"`
Only one port is available per node, so its possible one has to try different nodes until hitting
one with `8786` being open. Other than that, no additional configurations should be necessary.

#### Coffea-casa (Nebraska AF) `executor:  "dask/casa"`
Coffea-casa is a JupyterHub based analysis-facility hosted at Nebraska. For more information and setup instuctions see
https://coffea-casa.readthedocs.io/en/latest/cc_user.html

After setting up and checking out this repository (either via the online terminal or git widget utility run with

Authentication is handled automatically via login auth token instead of a proxy. File paths need to replace xrootd redirector with "xcache", `runner.py` does this automatically.


#### Condor@DESY `executor:  "dask/condor"`
```bash
python runner.py --wf ttcom --executor dask/condor
```

#### Maxwell@DESY 
```bash
python runner.py --wf ttcom --executor parsl/slurm
```

## Profiling

### CPU profiling
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

### Memory profiling

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
## Plotting code

- data/MC comparison code

```python
python -m plotting.plotdataMC --lumi ${lumi} --phase ctag_ttdilep_sf --output ctag_ttdilep_sf (--discr zmass --log True/False --data data_runD)
# lumi in /pb
# phase = workflow 
# output coffea file output = hist_$output$.coffea 
# discr = input variables, the defaults are the discriminators, can add multiple variables with space
# log = logorithm on y-axis
# data = data name
```

- data/data, MC/MC comparison

```python
python -m plotting.comparison --phase ctag_ttdilep_sf --output ctag_ttdilep_sf -ref 2017_runB --compared 2017_runC 2017_runD (--discr zmass --log True/False --sepflav True/False)
# phase = workflow 
# output coffea file output = hist_$output$.coffea 
# ref = reference data/MC sample
# comapred = 
# discr = input variables, the defaults are the discriminators, can add multiple variables with space
# log = logorithm on y-axis
# sepflav = separate the jets into different flavor
```
