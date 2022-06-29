import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import argparse
from matplotlib.offsetbox import AnchoredText
from BTVNanoCommissioning.utils.xs_scaler import getSumW,collate,scaleSumW
from coffea.util import load
from coffea.hist import plot
from coffea import hist
from hist import hist2
import os, math, re, json, shutil, arrow

time = arrow.now().format("YYYY_MM_DD")
### style settings
plt.style.use(hep.style.ROOT)

data_err_opts = {
    "linestyle": "none",
    "marker": ".",
    "markersize": 10.0,
    "color": "k",
    "elinewidth": 1,
}
from cycler import cycler


parser = argparse.ArgumentParser(description="plotter code use coffea files")
# Input maps
parser.add_argument(
    "-an",
    "--analysis",
    help="Which analysis run on (analysis directory)",
    required=True,
    default="Hpluscharm",
)
parser.add_argument(
    "-i",
    "--input",
    default="input.json",
    help="Input files",
)
parser.add_argument(
    "--plot_map",
    default="plotmap.json",
    help="plotting variables",
)
parser.add_argument(
    "--merge_map",
    default="mergemap.json",
    help="merge map of samples",
)
## plot configurations
parser.add_argument(
    "--scalesig",
    type=float,
    default=50000,
    help="Scale signal components",
)
parser.add_argument(
    "-c",
    "--campaign",
    help="which campaigns",
)
parser.add_argument(
    "-ch",
    "--channel",
    type=str,
    help="channel_name, which channel",
)
parser.add_argument(
    "-r",
    "--region",
    type=str,
    help="region_name, which region",
)
parser.add_argument(
    "-v",
    "--version",
    type=str,
    help="version",
)
parser.add_argument(
    "--splitflav",
    type=str,
    help="split flavor",
)
parser.add_argument(
    "--dataMC",
    action="store_true",
    help="data/MC comparison",
)
parser.add_argument(
    "-ref",
    "--referance",
    type=str,
    help="make comparison to different setup(reference axis, denominator)",
)
parser.add_argument(
    "--disable_ratio",
    action="store_true",
    help="disable ratio panel",
)


args = parser.parse_args()
## user specificific, load inputs


with open(f"../src/{args.analysis}/metadata/{args.merge_map}") as json_file:
    merge_map = json.load(json_file)
with open(f"../src/{args.analysis}/metadata/{args.plot_map}") as pltf:
    plot_map = json.load(pltf)
    plot_map = plot_map[args.version]
with open(f"../src/{args.analysis}/metadata/{args.input}") as inputs:
    input_map = json.load(inputs)


output = {i: load(input_map[args.version][i]) for i in input_map[args.version].keys()}

### set year info and luminosity info
if "16" in args.campaign:
    year = 2016
    if "UL16" in args.campaign:
        lumis = 36100
    else:
        lumis = 35900
elif "17" in args.campaign:
    year = 2017
    lumis = 41500
elif "18" in args.campaign:
    year = 2018
    lumis = 59800
if not os.path.isdir(f"plot/{args.analysis}_{args.campaign}_{args.version}_{time}/"):
    os.makedirs(f"plot/{args.analysis}_{args.campaign}_{args.version}_{time}/")


for var in plot_map["var_map"].keys():
    if var == "array" or var == "sumw" or var == "cutflow":
        continue
    if args.dataMC:
        scales = args.scalesig
        for out in output.keys():
            ## Scale XS for each hist
            
            if out == "data":
                output[out] = collate(output[out],merge_map["data"])
            else:
                output[out] = scaleSumW(output[out],lumis,getSumw(output[out]),"../metadata/xsection.json")
                output[out] = collate(output[out],merge_map[args.version])
                

        for region in args.region.split(",")[1:]:

            if not os.path.isdir(
                f"plot/{args.analysis}_{args.campaign}_{args.version}_{time}/{region}"
            ):
                os.makedirs(
                    f"plot/{args.analysis}_{args.campaign}_{args.version}_{time}/{region}"
                )

            if "SR" not in region and (
                "lep1_" not in var
                and "lep2_" not in var
                and "jetflav_" not in var
                and "nj" not in var
                and "nele" != var
                and "nmu" != var
                and "mT" not in var
                and "METTkMETdphi" != var
            ):
                continue
            print(region, var)
            for chs in args.channel.split(",")[1:]:
                if args.disable_ratio:
                    fig, ax = plt.subplots(figsize=(12, 12))
                else:
                    fig, ((ax), (rax)) = plt.subplots(
                        2,
                        1,
                        figsize=(12, 12),
                        gridspec_kw={"height_ratios": (3, 1)},
                        sharex=True,
                    )
                    fig.subplots_adjust(hspace=0.07)
                hep.cms.label(
                    "Work in progress",
                    data=True,
                    lumi=lumis / 1000.0,
                    year=year,
                    loc=0,
                    ax=ax,
                )

                for i in output.keys():
                hbkglist = []
                labels = []
                if args.splitflav is not None:
                    for sample in plot_map["order"]:
                        if sample == "signal":
                            continue
                        if sample == args.splitflav:
                            hbkglist.append(
                                output[sample][var][{'lepflav':chs,'region':region,'flav':0}].project(output[sample][var].axes[-1])+output[sample][var][{'lepflav':chs,'region':region,'flav':1}].project(output[sample][var].axes[-1])
                            )
                            hbkglist.append(
                                output[sample][var][{'lepflav':chs,'region':region,'flav':4}].project(output[sample][var].axes[-1])
                            )
                            hbkglist.append(
                                output[sample][var][{'lepflav':chs,'region':region,'flav':5}].project(output[sample][var].axes[-1])
                            )
                            labels.append("Z+l")
                            labels.append("Z+c")
                            labels.append("Z+b")
                        else:
                            hbkglist.append(
                                output[sample][var][{'lepflav':chs,'region':region,'flav':sum}].project(output[sample][var].axes[-1])
                            )
                            labels.append(sample)

                else:
                    hbkglist = [
                        output[sample][var][{'lepflav':chs,'region':region,'flav':sum}].project(output[sample][var].axes[-1])
                        for sample in plot_map["order"]
                        if sample != "signal"
                    ]
                    labels = plot_map["order"]
                hep.histplot(
                    hbkglist,
                    stack=True,
                    histtype="fill",
                    ax=ax,
                    label=labels,
                    color=plot_map["color_map"][:-1],
                )

                hdata = output[f'data_{chs}'][var][{'lepflav':chs,'region':region,'flav':sum}].project(output[f'data_{chs}'][var].axes[-1])
                    
                
                hscales = scales / 100
                if chs == "emu":
                    hscales = hscales / 5
                hep.histplot(
                    output['Higgs'][var][{'lepflav':chs,'region':region,'flav':sum}].project(output['Higgs'][var].axes[-1])*hscales,
                    color=plot_map["color_map"][-2],
                    linewidth=2,
                    label=f"Higgsx{int(hscales)}",
                    yerr=True,
                    ax=ax,
                )
                hep.histplot(
                    output['signal'][var][{'lepflav':chs,'region':region,'flav':sum}].project(output['signal'][var].axes[-1])*scales,
                    color=plot_map["color_map"][-1],
                    linewidth=2,
                    label=f"signalx{scales}",
                    yerr=True,
                    ax=ax,
                )

                hep.histplot(
                    hdata,
                    histtype="errorbar",
                    color="black",
                    label=f"Data",
                    yerr=True,
                    ax=ax,
                )
                
                for sample in output.keys():
                    if 'data' not in sample: hmc=output[sample][var][{'lepflav':chs,'region':region,'flav':sum}].project(output[sample][var].axes[-1]) +hmc 
                if not args.disable_ratio:
                    ax, rax = hdata.plotratio(
                        num=hdata,
                        denom=hmc,
                        ax=rax,
                        error_opts=data_err_opts,
                        denom_fill_opts={},
                        unc="num",
                        clear=False,
                    )
                    rax.set_ylim(0.5, 1.5)
                    rax.set_ylabel("Data/MC")
                    rax.set_xlabel(plot_map["var_map"][var])
                    ax.set_xlabel("")
                else:
                    ax.set_xlabel(plot_map["var_map"][var])
                ax.set_ylabel("Events")

                ax.set_ylim(bottom=0.0)

                chl = chs
                if chs == "mumu":
                    chs = "$\mu\mu$"
                if chs == "emu":
                    chs = "e$\mu$"
                ans = args.analysis
                if "HWW2l2nu" in args.version:
                    ans = r"HWW$\rightarrow 2\ell 2\nu$"
                at = AnchoredText(
                    chs + "  " + plot_map["region_map"][region] + "\n" + ans,
                    loc="upper left",
                    frameon=False,
                )
                ax.add_artist(at)
                ax.legend(
                    loc="upper right",
                    ncol=2,
                )

                hep.mpl_magic(ax=ax)

                fig.savefig(
                    f"plot/{args.analysis}_{args.campaign}_{args.version}_{time}/{region}/{chl}_{region}_{var}.pdf"
                )
                fig.savefig(
                    f"plot/{args.analysis}_{args.campaign}_{args.version}_{time}/{region}/{chl}_{region}_{var}.png"
                )
                plt.clf()
    else:
        fig, ((ax), (rax)) = plt.subplots(
            2,
            1,
            figsize=(6, 6),
            gridspec_kw={"height_ratios": (3, 1)},
            sharex=True,
        )
        fig.subplots_adjust(hspace=0.07)
        hep.cms.label(
            "Work in progress", data=True, lumi=lumis / 1000.0, year=year, loc=0, ax=ax
        )
        ax = plot.plot1d(
            output[args.version][var],
            overlay="dataset",
            ax=ax,
            density=True,
        )
        rax = plot.plotratio(
            num=output[args.version][var].integrate("dataset", "gchcWW2L2Nu_4f"),
            denom=output[args.version][var].integrate("dataset", args.referance),
            ax=rax,
            error_opts=data_err_opts,
            denom_fill_opts={},
            #
            unc="num",
            clear=False,
        )

        rax.set_ylim(0.5, 1.5)
        rax.set_ylabel("New/Old")
        rax.set_xlabel(plot_map["var_map"][var])
        ax.set_xlabel("")
        ax.legend(fontsize=25, labels=["Old", "New"])
        # at = AnchoredText("GEN",loc="upper left")
        # ax.add_artist(at)
        # hep.mpl_magic(ax=ax)

        fig.savefig(
            f"plot/{args.analysis}_{args.campaign}_{args.version}_{time}/{var}.png"
        )
