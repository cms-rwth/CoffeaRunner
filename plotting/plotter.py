import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import argparse
from matplotlib.offsetbox import AnchoredText
from BTVNanoCommissioning.utils.xs_scaler import scale_xs
from coffea.util import load
from coffea.hist import plot
from coffea import hist

import os, math, re, json, shutil

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
import matplotlib as mpl

colors = [
    "#666666",
    "#1D6996",
    "#38A6A5",
    "#0F8554",
    "#73AF48",
    "#EDAD08",
    "#E17C05",
    "#CC503E",
    "#554e99",
    "#6f4e99",
    "#854e99",
    "#994e85",
    "#666666",
]

mpl.rcParams["axes.prop_cycle"] = cycler("color", colors)


parser = argparse.ArgumentParser(
        description="Run analysis on baconbits files using processor coffea files"
    )
# Input maps
parser.add_argument(
        "-an",
        "--analysis",
        help="Which analysis run on (analysis directory)",
        required=True,
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
        action="store_true",
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
    default="gchcWW2L2Nu",
    help="reference",
)


args = parser.parse_args()
## user specificific, load inputs
# analysis_dir = "BTVNanoCommissioning"
analysis_dir = "Hpluscharm"

with open(f"../src/{analysis_dir}/metadata/{args.merge_map}") as json_file:
    merge_map = json.load(json_file)
with open(f"../src/{analysis_dir}/metadata/{args.plot_map}") as pltf:
    var_map = json.load(pltf)
    var_map= var_map[args.analysis]
with open(f"../src/{analysis_dir}/metadata/{args.input}") as inputs:
    input_map = json.load(inputs)


output = {i : load(input_map[args.analysis][i]) for i in input_map[args.analysis].keys()}

### campaigns
if "16" in args.campaign :
    year = 2016
    if "UL16" in args.campaign:lumis = 36100
    else:lumis = 35900
elif "17" in args.campaign :
    year = 2017
    lumis = 41500
elif "18" in args.campaign :
    year = 2018
    lumis = 59800
if not os.path.isdir(f'plot/{args.analysis}_{args.campaign}_{args.version}/'):
    os.makedirs(f'plot/{args.analysis}_{args.campaign}_{args.version}/')


for var in var_map["var_map"].keys():
    fig, ((ax), (rax)) = plt.subplots(
                    2,
                    1,
                    figsize=(12, 12),
                    gridspec_kw={"height_ratios": (3, 1)},
                    sharex=True,
                )
    fig.subplots_adjust(hspace=0.07)
    if var == 'array' or var == 'sumw' or var =='cutflow':continue
    if args.dataMC:
        scales = args.scalesig
        for out in output.keys():
            ## Scale XS
            if out=="signal": output[out][var] = scale_xs(output[out][var], lumis * scales, output[out]["sumw"],"../metadata/xsection.json")
            elif out == "data":
                output[out][var] = output[out][var].group(
            "dataset", hist.Cat("plotgroup", "plotgroup"), merge_map["data"])
            else: 
                output[out][var] = scale_xs(output[out][var], lumis, output[out]["sumw"],"../metadata/xsection.json")
                output[out][var] = output[out][var].group(
            "dataset", hist.Cat("plotgroup", "plotgroup"), merge_map[args.analysis])

        for region in args.region.split(",")[1:]:
            if not os.path.isdir(f'plot/{args.analysis}_{args.campaign}_{args.version}/{region}'):
                os.makedirs(f'plot/{args.analysis}_{args.campaign}_{args.version}/{region}')
            for chs in args.channel.split(",")[1:]:
                
                
                for i in output.keys():
                    if i == "data" or i == "signal" : continue
                    else :
                        if "1" in i : hmc = output[i][var].integrate(args.channel.split(",")[0], chs).integrate(args.region.split(",")[0], region)
                        else:hmc = (
                        hmc
                        .add(
                            output[i][var]
                            .integrate(args.channel.split(",")[0], chs)
                            .integrate(args.region.split(",")[0], region)
                        )
                    )

                ax = plot.plot1d(
                    hmc.sum("flav"),
                    overlay="plotgroup",
                    stack=True,
                    order=merge_map[args.analysis]['order'],
                    ax=ax,
                )
                if args.splitflav:plot.plot1d(hmc.integrate('plotgroup','Z+jets'),overlay="flav",stack=True,ax=ax,clear=False)
            
                hdata = (
                    output["data"][var]
                    .integrate(args.channel.split(",")[0], chs)
                    .integrate(args.region.split(",")[0], region)
                    .integrate("plotgroup", "data_%s" % (chs))
                    .sum("flav")
                )
                plot.plot1d(
                    output["signal"][var]
                    .sum("flav")
                    .integrate(args.channel.split(",")[0], chs)
                    .integrate(args.region.split(",")[0], region)
                    .sum("dataset"),
                    clear=False,
                    ax=ax,
                )
                plot.plot1d(hdata, clear=False, error_opts=data_err_opts, ax=ax)

                
                rax = plot.plotratio(
                    num=hdata,
                    denom=hmc.sum("plotgroup").sum("flav"),
                    ax=rax,
                    error_opts=data_err_opts,
                    denom_fill_opts={},
                    #
                    unc="num",
                    clear=False,
                )

                #
                rax.set_ylim(0.5, 1.5)
                rax.set_ylabel("Data/MC")
                rax.set_xlabel(var_map["var_map"][var])
                ax.set_xlabel("")
                chl = chs
                if chs == "mumu":
                    chs = "$\mu\mu$"
                if chs == "emu":
                    chs = "e$\mu$"
                at = AnchoredText(
                    chs + "  " + var_map["region_map"][region] + "\n" + r"HWW$\rightarrow 2\ell 2\nu$",
                    loc="upper left",
                    frameon=False,
                )
                ax.add_artist(at)
                leg_label = ax.get_legend_handles_labels()[1][1:]
                if 'DY' in region and args.splitflav:
                    leg_label[-6]='Z+l'
                    leg_label[-5]='Z+pu'
                    leg_label[-4]='Z+c'
                    leg_label[-3]='Z+b'
                leg_label[-1]='data'
                leg_label[-2]='Signalx%d' %(scales)
                ax.legend(
                    loc="upper right",
                    handles=ax.get_legend_handles_labels()[0][1:],
                    ncol=2,
                    labels=leg_label,
                    fontsize=18,
                )
                
                # hep.cms.label("Work in progress", data=True, lumi=41.5, year=2017,loc=0,ax=ax)
                hep.mpl_magic(ax=ax)
                
                fig.savefig(f'plot/{args.analysis}_{args.campaign}_{args.version}/{region}/{chl}_{region}_{var}.pdf')
    else:
        ax = plot.plot1d(
                    output[args.analysis][var],
                    overlay="dataset",
                    ax=ax,
                    density=True,
                )
        rax = plot.plotratio(
                    num=output[args.analysis][var].integrate("dataset","gchcWW2L2Nu_4f"),
                    denom=output[args.analysis][var].integrate("dataset","gchcWW2L2Nu"),
                    ax=rax,
                    error_opts=data_err_opts,
                    denom_fill_opts={},
                    #
                    unc="num",
                    clear=False,
                )
        
        rax.set_ylim(0.5, 1.5)
        rax.set_ylabel("New/Old")
        rax.set_xlabel(var_map["var_map"][var])
        ax.set_xlabel("")
        ax.legend(fontsize=25,labels=["Old","New"])
        # at = AnchoredText("GEN",loc="upper left")
        # ax.add_artist(at)
        # hep.mpl_magic(ax=ax)
                
        fig.savefig(f'plot/{args.analysis}_{args.campaign}_{args.version}/{var}.png')


