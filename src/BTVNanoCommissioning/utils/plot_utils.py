import matplotlib.pyplot as plt
from coffea.util import load
import hist
import glob


def isin_dict(config, key=None):
    if key == None:
        return type(config) is dict
    else:
        return type(config) is dict and key in config.keys()


def check_config(config, isDataMC=True):
    ## Check required info
    if isDataMC:
        general = ["input", "output", "variable", "lumi", "mergemap"]
    else:
        general = ["input", "output", "variable", "reference", "compare"]
    if len(set(general).difference(config.keys())) > 0:
        raise TypeError(f"{set(general).difference(config.keys())} is missing")

    inputtype = {
        "input": [list, str],
        "output": str,
        "com": str,
        "inbox_text": str,
        "lumi": float,
        "mergemap": dict,
        "reference": dict,
        "compare": dict,
        "variable": dict,
        "scale": dict,
        "disable_ratio": bool,
        "log": bool,
        "norm": bool,
    }
    variabletype = {
        "xlabel": str,
        "axis": dict,
        "blind": bool,
        "rebin": [dict, float, int],
    }

    for attr in config.keys():
        ## check type
        if type(attr) not in inputtype[attr]:
            raise ValueError(f"Type of {attr} should be {inputtype[attr]}")
        ## check method
        if isDataMC & attr in ["norm"]:
            raise NotImplementedError(f"{attr} is invalid in data/MC comparison")
        if not isDataMC & attr in ["disable_ratio", "scale"]:
            raise NotImplementedError(f"{attr} is invalid in shape comparison")
        # Check nested case
        if isin_dict(config[attr]):
            if attr == "scale":
                for sc in config["scale"].keys():
                    if (
                        type(config["scale"][sc]) != float
                        or type(config["scale"][sc]) != int
                    ):
                        raise TypeError(f"Type of scale[{sc}] should be int/float")
            elif attr == "mergemap" | attr == "reference" | attr == "compare":
                if attr == "reference" and len(config[attr]) > 1:
                    raise ValueError("Only one reference is allowed")
                for sc in config[attr].keys():
                    if type(config[attr][sc]) != str:
                        raise TypeError(f"Type of {attr}[{sc}] should be string")
            else:  ## variables
                for var in config[attr].keys():
                    if not isDataMC and "blind" in config[attr][var].keys():
                        raise NotImplementedError(
                            "blind is not implemented in shape comparison"
                        )
                    if type(var) not in variabletype[var]:
                        raise TypeError(
                            f"Type of {attr}[{var}] should be {variabletype[var]}"
                        )
                    if var == "all" and set(list(config[attr][var])).difference(
                        ["rebin"]
                    ):
                        raise ValueError(
                            f"{set(list(config[attr][var])).difference(['rebin'])} not include for all"
                        )
                    elif "*" in var and set(list(config[attr][var])).difference(
                        ["rebin", "axis"]
                    ):
                        raise ValueError(
                            f"{set(list(config[attr][var])).difference(['rebin','axis'])} not include for wildcard operation"
                        )


def load_default(config, isDataMC=True):
    if "com" not in config.keys():
        config["com"] = "13"
    if "inbox_text" not in config.keys():
        config["inbox_text"] = ""
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    if isDataMC:
        maps = ["mergemap"]
    else:
        maps = ["reference", "compare"]
    ## set color
    for m in maps:
        for i, mc in enumerate(config[m].keys()):
            if config[m][mc] == None:
                config[m][mc] = {}
            if "color" not in config[m][mc]:
                config[m][mc]["color"] = colors[i]
            if "label" not in config[m][mc]:
                config[m][mc]["label"] = mc
    return config


def load_coffea(config, isDataMC=True):
    if isDataMC:
        from BTVNanoCommissioning.utils.xs_scaler import getSumW, scaleSumW
    if "*" in config["input"]:
        files = glob.glob(config["input"])
        output = {i: load(i) for i in files}
        for out in output.keys():
            if isDataMC:
                output[out] = scaleSumW(
                    output[out], config["lumi"], getSumW(output[out])
                )
    elif len(config["input"]) > 1:
        output = {i: load(i) for i in config["input"]}
        for out in output.keys():
            if isDataMC:
                output[out] = scaleSumW(
                    output[out], config["lumi"], getSumW(output[out])
                )
    else:
        output = load(config["input"])
        if isDataMC:
            output = scaleSumW(output, config["lumi"], getSumW(output))
    return output


def rebin_and_xlabel(var, collated, config, isDataMC=True):
    if isDataMC:
        baseaxis = "data"
    else:
        baseaxis = list(config["reference"].keys())[0]
    ## If specified for each variable
    if (
        "all" in list(config["variable"].keys())[0]
        or "*" in list(config["variable"].keys())[0]
    ):
        configvar = list(config["variable"].keys())[0]
    else:
        configvar = var
    ## set xlabel
    if isin_dict(config["variable"][configvar], "xlabel"):
        xlabel = config["variable"][var]["xlabel"]
    else:
        xlabel = var
    rebin_axis = {}
    ## self define axis
    if configvar != "all" and isin_dict(config["variable"][configvar], "axis"):
        for axis in config["variable"][configvar]["axis"].keys():
            if config["variable"][configvar]["axis"][axis] == "sum":
                rebin_axis[axis] = sum
            else:
                rebin_axis[axis] = config["variable"][configvar]["axis"][axis]
    else:
        rebin_axis = {axis: sum for axis in collated[baseaxis][var].axes.name}
        rebin_axis.popitem()
    ## Set rebin axis from configuration file
    if isin_dict(config["variable"][configvar], "rebin"):
        if isin_dict(config["variable"][configvar]["rebin"]):

            for rb in config["variable"][configvar]["rebin"].keys():
                print("??", config["variable"][configvar]["rebin"], rb)
                rebin_axis[rb] = hist.rebin(config["variable"][configvar]["rebin"][rb])
        ## default is set to be last axis
        else:
            print("default", collated[baseaxis][var].axes[-1].name)
            rebin_axis[collated[baseaxis][var].axes[-1].name] = hist.rebin(
                config["variable"][configvar]["rebin"]
            )
    return xlabel, rebin_axis
