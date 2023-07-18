import argparse, sys, os, arrow, glob, yaml
import numpy as np
import matplotlib.pyplot as plt, mplhep as hep
from matplotlib.offsetbox import AnchoredText

from coffea.util import load
import hist

time = arrow.now().format("YY_MM_DD")
plt.style.use(hep.style.ROOT)

from BTVNanoCommissioning.utils.xs_scaler import collate, additional_scale
from BTVNanoCommissioning.utils.plot_utils import (
    isin_dict,
    check_config,
    load_coffea,
    load_default,
    rebin_and_xlabel,
    plotratio,
    autoranger,
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
refname = list(config["reference"].keys())[0]
collated = collate(output, mergemap)
config = load_default(config, False)

## If addition rescale on yields required
if "rescale_yields" in config.keys():
    # print(config["rescale_yields"])
    for sample_to_scale in config["rescale_yields"].keys():
        print(
            f"Rescale {sample_to_scale} by {config['rescale_yields'][sample_to_scale]}"
        )

    collated = additional_scale(collated, config["rescale_yields"])

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


np.seterr(divide="ignore", invalid="ignore")
## Loop through all variables
for var in var_set:
    if "sumw" == var:
        continue
    xlabel, collated = rebin_and_xlabel(var, collated, config, False)
    print("\t Plotting now:", var, xlabel)
    ## Normalize to reference yield
    if "norm" in config.keys() and config["norm"]:
        for c in config["compare"].keys():
            collated[c][var] = collated[c][var] * float(
                np.sum(collated[refname][var].values())
                / np.sum(collated[c][var].values())
            )

    fig, ((ax), (rax)) = plt.subplots(
        2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": (3, 1)}, sharex=True
    )
    fig.subplots_adjust(hspace=0.06, top=0.92, bottom=0.1, right=0.97)
    hep.cms.label(label, com=config["com"], data=True, loc=0, ax=ax)
    ## plot reference
    hep.histplot(
        collated[refname][var],
        label=config["reference"][refname]["label"] + " (Ref)",
        histtype=hist_type,
        color=config["reference"][refname]["color"],
        yerr=True,
        ax=ax,
    )
    ## plot compare list
    for c, s in config["compare"].items():
        # print(collated[c][var][{var:sum}])
        hep.histplot(
            collated[c][var],
            label=config["compare"][c]["label"],
            histtype=hist_type,
            color=config["compare"][c]["color"],
            yerr=True,
            ax=ax,
            # flow=args.flow,
        )
    # plot ratio of com/Ref
    for i, c in enumerate(config["compare"].keys()):
        plotratio(
            collated[c][var],
            collated[refname][var],
            denom_fill_opts={},
            error_opts={"color": config["compare"][c]["color"]},
            clear=False,
            ax=rax,
        )

    ##  plot settings, adjust range
    rax.set_xlabel(xlabel)
    ax.set_xlabel(None)
    ax.set_ylabel("Events")
    rax.set_ylabel("Other/Ref")
    ax.ticklabel_format(style="sci", axis="y", scilimits=(-3, 3))
    ax.get_yaxis().get_offset_text().set_position((-0.065, 1.05))
    ax.legend()
    rax.set_ylim(0.0, 2.0)
    xmin, xmax = autoranger(collated[refname][var])
    rax.set_xlim(xmin, xmax)
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


print(f"The output is saved at: plot/{config['output']}_{time}/")
