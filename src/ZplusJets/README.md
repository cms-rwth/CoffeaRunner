## How to

### Prepare your datasets
 * Create a simple list of dataset names and put them in a txt file
 * Run `filefetcher/fetch.py` script as follows:
```
python fetch.py -i input.list.txt  -o my_samples [--xrd root://grid-cms-xrootd.physik.rwth-aachen.de/]
```
This will create a file at `metadata/my_samples.json` with all input files to run over. This procedure could be done separately for a group of samples.

### Run the Processor

 * First run a test over a single dataset:
```
runner.py --workflow zjets --json metadata/my_samples.json --limit 2 --only DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM
```
(the dataset must exist in your input list of course.)

 * Once first test is working we move to more extensive test. This will submit jobs to condor with parsl:
```
python runner.py --workflow zjets --json metadata/my_samples.json --limit 10 --only /DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM --executor parsl/condor
```
 * Now we submit all jobs over all samples in the list:
```
python runner.py --workflow zjets --json metadata/my_samples.json --executor parsl/condor -s 200 -j 2
```



