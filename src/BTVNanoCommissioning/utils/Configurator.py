import os
import sys
import json
from pprint import pprint
import pickle
import importlib.util
from collections import defaultdict
import inspect


## This is coming from https://github.com/PocketCoffea/PocketCoffea
class Configurator:
    def __init__(self, cfg, overwrite_output_dir=None, plot=False, plot_version=None):
        # Load config file and attributes
        ## MY: plot not use in CoffeaRunner framework!
        # self.plot = plot
        # self.plot_version = plot_version
        in_cfg = cfg
        self.load_config(cfg)
        self.load_run_options_default()
        # Load all the keys in the config (dangerous, can be improved)
        self.load_attributes()

        # Load dataset
        self.samples = []
        self.load_dataset()

        # Check if output file exists, and in case add a `_v01` label, make directory
        if overwrite_output_dir:
            self.output = overwrite_output_dir
        else:
            self.overwrite_check()

        self.mkdir_output()

        # Truncate file list if self.limit is not None
        self.truncate_filelist()

        # Define output file path
        self.define_output()

        # Load histogram settings
        # self.load_histogram_settings()
        ## MY: --- not use in CoffeaRunner framework!

        ## Load cuts and categories
        ## MY: --- not use in CoffeaRunner framework! but it's a nice feature
        # Cuts: set of Cut objects
        # Categories: dict with a set of Cut ids for each category
        self.cut_functions = []
        self.categories = {}
        # Saving also a dict of Cut objects to map their ids (name__hash)
        # N.B. The preselections are just cuts that are applied before
        # others. It is just a special category of cuts.
        self.cuts_dict = {}
        ## Call the function which transforms the dictionary in the cfg
        # in the objects needed in the processors

        if "categories" in self.cfg.keys():
            self.load_cuts_and_categories()

        ## Weights configuration
        self.weights_split_bycat = False
        self.weights_config = {}
        if "categories" in self.cfg.keys():
            self.weights_config_bycat = {c: {} for c in self.categories.keys()}
        else:
            self.weights_config_bycat = None

        if "weights" in self.cfg.keys():
            self.load_weights_config()

        # Load workflow
        self.load_workflow()

        # Save config file in output folder
        self.save_config(in_cfg)

    def load_config(self, path):
        spec = importlib.util.spec_from_file_location("config", path)
        cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg)
        self.cfg = cfg.cfg

    def load_attributes(self):
        exclude_auto_loading = ["categories", "weights"]
        for key, item in self.cfg.items():
            if key in exclude_auto_loading:
                continue
            setattr(self, key, item)
        # Define default values for optional parameters
        for key in ["only"]:
            try:
                getattr(self, key)
            except:
                setattr(self, key, "")

        # if self.plot:
        #     # If a specific version is specified, plot that version
        #     if self.plot_version:
        #         self.output = self.output + f'_{self.plot_version}'
        #         if not os.path.exists(self.output):
        #             sys.exit(f"The output folder {self.output} does not exist")
        #     # If no version is specified, plot the latest version of the output
        #     else:
        #         parent_dir = os.path.abspath(os.path.join(self.output, os.pardir))
        #         output_dir = os.path.basename(self.output)
        #         latest_dir = list(filter(lambda folder : ((output_dir == folder) | (output_dir+'_v' in folder)), sorted(os.listdir(parent_dir))))[-1]
        #         self.output = os.path.join(parent_dir, latest_dir)
        #     self.plots = os.path.join( os.path.abspath(self.output), "plots" )

    def load_run_options_default(self):
        default_config = {
            "executor": "iterative",
            "limit": None,
            "max": None,
            "chunk": 50000,
            "workers": 2,
            "scaleout": 20,
            "walltime": "03:00:00",
            "mem_per_worker": 2,  # GB
            "skipbadfiles": False,
            "splitjobs": True,
            "retries": 20,
            "voms": None,
            "compression": 3,
            "index": None,
            "sample_size": 20,
        }
        if "run_options" not in self.cfg.keys():
            self.cfg["run_options"] = {}
        for opt in default_config.keys():
            if opt not in self.cfg["run_options"].keys():
                self.cfg["run_options"][opt] = default_config[opt]

    def load_dataset(self):
        self.fileset = {}
        for json_dataset in self.dataset["jsons"]:
            ds_dict = json.load(open(json_dataset))
            ds_filter = self.dataset.get("filter", None)
            if ds_filter != None:
                for key, ds in ds_dict.items():
                    pass_filter = True

                    if "samples" in ds_filter:
                        if key not in ds_filter["samples"]:
                            pass_filter = False
                    if "samples_exclude" in ds_filter:
                        if key in ds_filter["samples_exclude"]:
                            pass_filter = False

                    if pass_filter:
                        self.fileset[key] = ds
            else:
                self.fileset.update(ds_dict)
        if self.run_options["executor"] == "dask/casa":
            for key in self.fileset.keys():
                if "xrootd-cms.infn.it" in self.fileset[key][0]:
                    self.fileset[key] = [
                        path.replace("xrootd-cms.infn.it/", "xcache")
                        for path in self.fileset[key]
                    ]
                elif "dcache-cms-xrootd.desy.de:1094" in self.fileset[key][0]:
                    self.fileset[key] = [
                        path.replace("dcache-cms-xrootd.desy.de:1094/", "xcache")
                        for path in self.fileset[key]
                    ]
        if len(self.fileset) == 0:
            print("File set is empty: please check you dataset definition...")
            exit(1)
        else:
            for name, d in self.fileset.items():
                if name not in self.samples:
                    self.samples.append(name)

    def load_cuts_and_categories(self):
        """This function loads the list of cuts and groups them in categories.
        Each cut is identified by a unique id (see Cut class definition)"""
        # The cuts_dict is saved just for record
        # for skim in self.cfg["skim"]:
        #     if not isinstance(skim, Cut):
        #         print("Please define skim, preselections and cuts as Cut objects")
        #         exit(1)
        #     self.cuts_dict[skim.id] = skim
        if hasattr(self, "preselections"):
            for presel_name, presel_list in self.cfg["preselections"].items():
                self.cuts_dict[presel_name] = presel_list
        for cat, cuts in self.cfg["categories"].items():
            self.categories[cat] = cuts
        print("Cuts:", list(self.cuts_dict.keys()))
        print("Categories:", self.categories)

    def load_weights_config(self):
        """This function loads the weights definition and prepares a list of
        weights to be applied for each sample and category"""
        # Read the config and save the list of weights names for each sample (and category if needed)
        wcfg = self.cfg["weights"]
        if "common" not in wcfg:
            print("Weights configuration error: missing 'common' weights key")
            exit(1)

        # common/inclusive weights

        for weight_name, weight_list in wcfg["common"]["inclusive"].items():
            if isinstance(weight_name, str):
                self.weights_config[weight_name] = weight_list
            else:
                raise NotImplementedError()

        if "bycategory" in wcfg["common"]:
            self.weights_split_bycat = True
            for cat, weights in wcfg["common"]["bycategory"].items():
                for weight_name, weight_list in weights.items():
                    if isinstance(weight_name, str):
                        self.weights_config_bycat[cat] = weight_list
                    else:
                        raise NotImplementedError()
        # Now look at specific samples configurations
        if "bysample" in wcfg:
            for sample, s_wcfg in wcfg["bysample"].items():
                if sample not in self.samples:
                    print(
                        f"Requested missing sample {sample} in the weights configuration"
                    )
                    exit(1)
                if "inclusive" in s_wcfg:
                    for weight_name, weight_list in s_wcfg["inclusive"].items():
                        if isinstance(weight_name, str):
                            # append only to the specific sample
                            self.weights_config[weight_name] = {}
                            self.weights_config[weight_name][sample] = weight_list
                        else:
                            raise NotImplementedError()

                if "bycategory" in s_wcfg:
                    self.weights_split_bycat = True
                    for cat, weights in s_wcfg["bycategory"].items():
                        for weight_name, weight_list in weights.items():
                            if isinstance(weight_name, str):
                                self.weights_config_bycat[cat][weight_name] = {}
                                self.weights_config_bycat[cat][weight_name][
                                    sample
                                ] = weight_list
                            else:
                                raise NotImplementedError()

        print("Weights configuration")
        pprint(self.weights_config)
        print("Weights configuration by category")
        pprint(self.weights_config_bycat)

    def overwrite_check(self):
        ## splitted plot
        # if self.plot:
        #     print(f"The output will be saved to {self.plots}")
        #     return
        # else:
        version = 0
        tag = str(version).rjust(2, "0")
        path = f"{self.output}_v{tag}"
        while os.path.exists(path):
            tag = str(version).rjust(2, "0")
            path = f"{self.output}_v{tag}"
            version += 1
        if path != self.output:
            print(f"The output will be saved to {path}")
        self.output = path
        self.cfg["output"] = self.output

    def mkdir_output(self):
        # if not self.plot:
        if not os.path.exists(self.output):
            os.makedirs(self.output)
        # else:
        #     if not os.path.exists(self.plots):
        #         os.makedirs(self.plots)

    def truncate_filelist(self):
        try:
            self.run_options["limit"]
        except:
            self.run_options["limit"] = None
        if self.run_options["limit"]:
            for dataset, filelist in self.fileset.items():
                if isinstance(filelist, dict):
                    self.fileset[dataset]["files"] = self.fileset[dataset]["files"][
                        : self.run_options["limit"]
                    ]
                elif isinstance(filelist, list):
                    self.fileset[dataset] = self.fileset[dataset][
                        : self.run_options["limit"]
                    ]
                else:
                    raise NotImplementedError

    def define_output(self):
        self.outfile = os.path.join(self.output, "output.coffea")

    ## MY : not used in current developement
    # def load_histogram_settings(self):
    #     if isinstance(self.cfg["variables"], list):
    #         self.cfg["variables"] = {
    #             var_name: None for var_name in self.cfg["variables"]
    #         }
    #     for var_name in self.cfg["variables"].keys():
    #         if self.cfg["variables"][var_name] == None:
    #             self.cfg["variables"][var_name] = histogram_settings["variables"][
    #                 var_name
    #             ]
    #         elif not isinstance(self.cfg["variables"][var_name], dict):
    #             sys.exit("Format non valid for histogram settings")
    #         elif set(self.cfg["variables"][var_name].keys()) != {
    #             "binning",
    #             "xlim",
    #             "xlabel",
    #         }:
    #             set_ctrl = {"binning", "xlim", "xlabel"}
    #             sys.exit(
    #                 f"{var_name}: missing keys in histogram settings. Required keys missing: {set_ctrl - set(self.cfg['variables'][var_name].keys())}"
    #             )
    #         elif "n_or_arr" not in set(
    #             self.cfg["variables"][var_name]["binning"].keys()
    #         ):
    #             sys.exit(
    #                 f"{var_name}: missing keys in histogram binning. Required keys missing: {'n_or_arr'}"
    #             )
    #         elif (
    #             ("n_or_arr" in set(self.cfg["variables"][var_name]["binning"].keys()))
    #             & (type(self.cfg["variables"][var_name]["binning"]["n_or_arr"]) == int)
    #             & (
    #                 set(self.cfg["variables"][var_name]["binning"].keys())
    #                 != {"n_or_arr", "lo", "hi"}
    #             )
    #         ):
    #             set_ctrl = {"n_or_arr", "lo", "hi"}
    #             sys.exit(
    #                 f"{var_name}: missing keys in histogram binning. Required keys missing: {set_ctrl - set(self.cfg['variables'][var_name]['binning'].keys())}"
    #             )

    def load_workflow(self):

        self.processor_instance = self.workflow(cfg=self)

    def save_config(self, cfg):
        os.system(f"cp {cfg} {os.path.join(self.output, 'configurator.py')}")
