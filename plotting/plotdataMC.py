import argparse, sys, os, arrow, yaml
import numpy as np
import matplotlib.pyplot as plt, mplhep as hep
from matplotlib.offsetbox import AnchoredText

import hist

plt.style.use(hep.style.ROOT)
time = arrow.now().format("YY_MM_DD")

from BTVNanoCommissioning.utils.xs_scaler import collate, additional_scale
from BTVNanoCommissioning.utils.plot_utils import (
    isin_dict,
    check_config,
    load_coffea,
    load_default,
    rebin_and_xlabel,
    plotratio,
    autoranger,
    MCerrorband,
)

parser = argparse.ArgumentParser(description="hist plotter for commissioning")
parser.add_argument("--cfg", type=str, required=True, help="Configuration files")
parser.add_argument(
    "--debug", action="store_true", help="Run detailed checks of yaml file"
)

arg = parser.parse_args()

# load config from yaml
with open(arg.cfg, "r") as f:
    config = yaml.safe_load(f)
if arg.debug:
    check_config(config, True)
## create output dictionary
if not os.path.isdir(f"plot/{config['output']}_{time}/"):
    os.makedirs(f"plot/{config['output']}_{time}/")
## load coffea files
output = load_coffea(config, config["scaleToLumi"])
## load merge map, inbox text
mergemap = {}
if not any(".coffea" in o for o in output.keys()):
    for merger in config["mergemap"].keys():
        mergemap[merger] = [
            m for m in output.keys() if m in config["mergemap"][merger]["list"]
        ]
else:
    for merger in config["mergemap"].keys():
        flist = []
        for f in output.keys():
            flist.extend(
                [m for m in output[f].keys() if m in config["mergemap"][merger]["list"]]
            )
        mergemap[merger] = flist
collated = collate(output, mergemap)
config = load_default(config)

## If addition rescale on yields required
if "rescale_yields" in config.keys():
    for sample_to_scale in config["rescale_yields"].keys():
        print(
            f"Rescale {sample_to_scale} by {config['rescale_yields'][sample_to_scale]}"
        )

    collated = additional_scale(collated, config["rescale_yields"])

## collect variable lists
if "all" in list(config["variable"].keys())[0]:
    var_set = collated["data"].keys()
elif "*" in list(config["variable"].keys())[0]:
    var_set = [
        var
        for var in collated["data"].keys()
        if list(config["variable"].keys())[0].replace("*", "") in var
    ]
else:
    var_set = config["variable"].keys()

## Loop through all variables
for var in var_set:
    if "sumw" == var:
        continue
    xlabel, collated = rebin_and_xlabel(var, collated, config)
    ### Figure settings
    if config["disable_ratio"]:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        np.seterr(invalid="ignore")

        fig, ((ax), (rax)) = plt.subplots(
            2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": (3, 1)}, sharex=True
        )
    fig.subplots_adjust(hspace=0.06, top=0.92, bottom=0.1, right=0.97)
    hep.cms.label(
        "Preliminary",
        data=True,
        lumi=config["lumi"] / 1000.0,
        com=config["com"],
        loc=0,
        ax=ax,
    )
    ## Plot MC (stack all MC)
    hep.histplot(
        [collated[mc][var] for mc in collated.keys() if "data" not in mc],
        stack=True,
        label=[
            config["mergemap"][mc]["label"]
            for mc in collated.keys()
            if "data" not in mc
        ],
        histtype="fill",
        yerr=True,
        color=[
            config["mergemap"][mc]["color"]
            for mc in collated.keys()
            if "data" not in mc
        ],
        ax=ax,
    )

    for i, mc in enumerate(collated.keys()):
        if "data" in mc:
            continue
        if i == 0:
            summc = collated[mc][var]
        else:
            summc = collated[mc][var] + summc
    MCerrorband(summc, ax=ax)  # stat. unc. errorband
    ## Scale particular sample
    if "scale" in config.keys():
        for mc in collated.keys():
            if mc in config["scale"].keys():
                hep.histplot(
                    collated[mc][var] * config["scale"][mc],
                    label=f'{config["mergemap"][mc]["label"]}$\\times${config["scale"][mc]}',
                    histtype="step",
                    lw=2,
                    yerr=True,
                    color=config["mergemap"][mc]["color"],
                    ax=ax,
                )
    ## Blind variables
    if (
        isin_dict(config["variable"])
        and (
            ("all" not in list(config["variable"].keys())[0])
            & ("*" not in list(config["variable"].keys())[0])
        )
        and isin_dict(config["variable"][var], "blind")
    ):
        hdata = collated["data"][var].values()
        mins = (
            None
            if config["variable"][var]["blind"].split(",")[0] == ""
            else int(config["variable"][var]["blind"].split(",")[0])
        )
        maxs = (
            None
            if config["variable"][var]["blind"].split(",")[1] == ""
            else int(config["variable"][var]["blind"].split(",")[1])
        )
        hdata[mins:maxs] = 0.0
    else:
        hdata = collated["data"][var].values()
    ## plot data
    hep.histplot(
        hdata,
        collated["data"][var].axes.edges[-1],
        histtype="errorbar",
        color="black",
        label="Data",
        yerr=True,
        ax=ax,
    )
    xmin, xmax = autoranger(collated["data"][var] + summc)
    ax.set_xlim(xmin, xmax)
    ## Ratio plot
    if "disable_ratio" not in config.keys() or config["disable_ratio"] == False:
        plotratio(collated["data"][var], summc, ax=rax)
        rax.set_ylabel("Data/MC")
        rax.set_xlabel(xlabel)
        rax.set_ylim(0.5, 1.5)
        ax.set_xlabel(None)
        rax.set_xlim(xmin, xmax)
    ##  plot settings, adjust range
    if "disable_ratio" in config.keys() and config["disable_ratio"]:
        ax.set_xlabel(xlabel)
    ax.set_ylabel("Events")
    ax.legend()
    ax.ticklabel_format(style="sci", scilimits=(-3, 3))
    ax.get_yaxis().get_offset_text().set_position((-0.065, 1.05))
    at = AnchoredText(
        config["inbox_text"],
        loc=2,
        frameon=False,
    )
    ax.set_ylim(bottom=0.0)
    ax.add_artist(at)
    hep.mpl_magic(ax=ax)
    name = ""
    ## log y axis
    if "log" in config.keys() and config["log"]:
        ax.set_yscale("log")
        name = "_log"
        ax.set_ylim(bottom=0.1)
        hep.mpl_magic(ax=ax)
    fig.savefig(f"plot/{config['output']}_{time}/{var}{name}.pdf")
    fig.savefig(f"plot/{config['output']}_{time}/{var}{name}.png")


print(f"The output will be saved at plot/{config['output']}_{time}")
