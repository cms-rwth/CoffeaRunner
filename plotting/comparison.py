import argparse, sys, os, arrow, glob, yaml
import numpy as np
import matplotlib.pyplot as plt, mplhep as hep
from matplotlib.offsetbox import AnchoredText

from coffea.util import load
import hist
from BTVNanoCommissioning.utils.plot_utils import plotratio

time = arrow.now().format("YY_MM_DD")
plt.style.use(hep.style.ROOT)

from BTVNanoCommissioning.utils.xs_scaler import collate
from BTVNanoCommissioning.utils.plot_utils import (
    isin_dict,
    check_config,
    load_coffea,
    load_default,
    rebin_and_xlabel,
    plotratio,
)

parser = argparse.ArgumentParser(description="make comparison for different campaigns")
parser.add_argument("--cfg", type=str, required=True, help="Configuration files")
parser.add_argument(
    "--debug", action="store_true", help="Run detailed checks of yaml file"
)
parser.add_argument
args = parser.parse_args()

# load config from yaml
with open(args.cfg, "r") as f:
    config = yaml.safe_load(f)
## create output dictionary
if not os.path.isdir(f"plot/{config['output']}_{time}/"):
    os.makedirs(f"plot/{config['output']}_{time}/")
if args.debug:
    check_config(config, False)
## load coffea files
output = load_coffea(config, config["scaleToLumi"])


## build up merge map
mergemap = {}
refname = list(config["reference"].keys())[0]
if not any(".coffea" in o for o in output.keys()):
    mergemap[refname] = [m for m in output.keys() if refname == m]
    for c in config["compare"].keys():
        mergemap[c] = [m for m in output.keys() if c == m]
else:
    reflist = []
    for f in output.keys():
        reflist.extend([m for m in output[f].keys() if refname == m])
    mergemap[refname] = reflist
    print("\t What we compare?\n ", config["compare"])
    for c in config["compare"].keys():
        comparelist = []
        for f in output.keys():
            comparelist.extend([m for m in output[f].keys() if c == m])
        mergemap[c] = comparelist
collated = collate(output, mergemap)
config = load_default(config, False)

# print('collated', collated)
### style settings
if "Run" in list(config["reference"].keys())[0]:
    hist_type = "errorbar"
    label = "Preliminary"
else:
    hist_type = "step"
    label = "Simulation Preliminary"

## collect variable lists
if "all" in list(config["variable"].keys())[0]:
    var_set = collated[refname].keys()
elif "*" in list(config["variable"].keys())[0]:
    var_set = [
        var
        for var in collated[refname]
        if list(config["variable"].keys())[0].replace("*", "") in var
    ]
else:
    var_set = config["variable"].keys()


np.seterr(invalid="ignore")
np.seterr(divide="ignore")

## Loop through all variables
for var in var_set:
    if "sumw" == var:
        continue

    xlabel, rebin_axis = rebin_and_xlabel(var, collated, config, False)
    # print(xlabel, rebin_axis)
    ## Normalize to reference yield
    if config["norm"]:
        for c in config["compare"].keys():
            # print(c, var)
            collated[c][var] = collated[c][var] * float(
                np.sum(collated[refname][var][rebin_axis].values())
                / np.sum(collated[c][var][rebin_axis].values())
            )

    fig, ((ax), (rax)) = plt.subplots(
        2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": (3, 1)}, sharex=True
    )
    fig.subplots_adjust(hspace=0.06, top=0.92, bottom=0.1, right=0.97)
    hep.cms.label(label, com=config["com"], data=True, loc=0, ax=ax)
    ## plot reference
    hep.histplot(
        collated[refname][var][rebin_axis],
        label=config["reference"][refname]["label"] + " (Ref)",
        histtype=hist_type,
        color=config["reference"][refname]["color"],
        yerr=True,
        ax=ax,
    )
    ## plot compare list
    for c, s in config["compare"].items():
        hep.histplot(
            collated[c][var][rebin_axis],
            label=config["compare"][c]["label"],
            histtype=hist_type,
            color=config["compare"][c]["color"],
            yerr=True,
            ax=ax,
        )
    # plot ratio of com/Ref
    for i, c in enumerate(config["compare"].keys()):
        plotratio(
            collated[c][var][rebin_axis],
            collated[refname][var][rebin_axis],
            denom_fill_opts=None,
            error_opts={"color": ax.get_lines()[i + 1].get_color()},
            clear=False,
            ax=rax,
        )

    ##  plot settings, adjust range
    rax.set_xlabel(xlabel)
    rax.axhline(y=1.0, linestyle="dashed", color="gray")
    ax.set_xlabel(None)
    ax.set_ylabel("Events")
    rax.set_ylabel("Other/Ref")
    ax.legend()
    rax.set_ylim(0.0, 2.0)

    at = AnchoredText(
        config["inbox_text"],
        loc=2,
        frameon=False,
    )
    ax.add_artist(at)
    hep.mpl_magic(ax=ax)
    ax.set_ylim(bottom=0)

    logext = ""
    # log y axis
    if "log" in config.keys() and config["log"]:
        ax.set_yscale("log")
        logext = "_log"
        ax.set_ylim(bottom=0.1)
        hep.mpl_magic(ax=ax)
    if "norm" in config.keys() and config["norm"]:
        logext = "_norm" + logext
    fig.savefig(f"plot/{config['output']}_{time}/compare_{var}{logext}.pdf")
    fig.savefig(f"plot/{config['output']}_{time}/compare_{var}{logext}.png")


print(f"The output is saved at: plot/{config['output']}_{time}")
