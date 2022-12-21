import matplotlib.pyplot as plt
from coffea.util import load
import hist
import glob
import scipy.stats
import warnings
import numpy as np

errband_opts = {
    "hatch": "////",
    "facecolor": "none",
    "lw": 0,
    "color": "k",
    "alpha": 0.4,
}
markers = [".", "o", "^", "s", "+", "x", "D", "*"]


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
        "lumi": str,
        "mergemap": dict,
        "reference": dict,
        "compare": dict,
        "variable": dict,
        "scale": dict,
        "disable_ratio": bool,
        "log": bool,
        "norm": bool,
        "scaleToLumi": bool,
        "rescale_yields": dict,
    }
    variabletype = {
        "xlabel": str,
        "axis": dict,
        "blind": int,
        "rebin": [dict, float, int],
    }

    for attr in config.keys():
        ## check type
        if type(config[attr]) != inputtype[attr] and (
            inputtype[attr] == list and type(config[attr]) not in inputtype[attr]
        ):
            raise ValueError(f"Type of {attr} should be {inputtype[attr]}")
        ## check method
        if isDataMC and attr in ["norm"]:
            raise NotImplementedError(f"{attr} is invalid in data/MC comparison")
        if not isDataMC and attr in ["disable_ratio", "scale"]:
            raise NotImplementedError(f"{attr} is invalid in shape comparison")
        # Check nested case
        if isin_dict(config[attr]):
            if "scale" in attr:
                for sc in config[attr].keys():
                    if float(config[attr][sc]):
                        config[attr][sc] = float(config[attr][sc])
                    else:
                        raise TypeError(f"Type of {attr}[{sc}] should be int/float")
            elif attr == "mergemap" or attr == "reference" or attr == "compare":
                if attr == "reference" and len(config[attr]) > 1:
                    raise ValueError("Only one reference is allowed")
            else:  ## variables
                if (
                    any(("*" in var or "all" == var) for var in config[attr].keys())
                    and len(list(config[attr].keys())) > 1
                ):
                    raise NotImplementedError(
                        "wildcard(*) operation/all can not specify together with normal expression"
                    )
                for var in config[attr].keys():
                    if isin_dict(config[attr][var]):
                        for var_param in config[attr][var].keys():
                            if not isDataMC and "blind" in config[attr][var].keys():
                                raise NotImplementedError(
                                    "blind is not implemented in shape comparison"
                                )
                            ## check type
                            if var_param == "blind":
                                if "," not in config[attr][var][var_param]:
                                    raise ValueError(", is needed for blind option")
                                if not int(config[attr][var][var_param].split(",")[0]):
                                    raise TypeError(
                                        f"Type of {attr}[{var}][{var_param}] in {var} should be {variabletype[var_param]} not {type(config[attr][var][var_param])}"
                                    )
                                if config[attr][var][var_param].split(",")[
                                    1
                                ] != "" and not int(
                                    config[attr][var][var_param].split(",")[1]
                                ):
                                    raise TypeError(
                                        f"Type of {attr}[{var}][{var_param}] in {var} should be {variabletype[var_param]} not {type(config[attr][var][var_param])}"
                                    )
                            elif (
                                variabletype[var_param] != list
                                and type(config[attr][var][var_param])
                                != variabletype[var_param]
                            ) and type(
                                config[attr][var][var_param]
                            ) not in variabletype[
                                "rebin"
                            ]:
                                raise TypeError(
                                    f"Type of {attr}[{var}][{var_param}] in {var} should be {variabletype[var_param]} not {type(config[attr][var][var_param])}"
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
                                    f"{set(list(config[attr][var])).difference(['rebin','axis'])} not include for wildcard(*) operation"
                                )


def load_default(config, isDataMC=True):
    if "com" not in config.keys():
        config["com"] = "13"
    if "inbox_text" not in config.keys():
        config["inbox_text"] = ""
    if "norm" not in config.keys():
        config["norm"] = False
    if "disable_ratio" not in config.keys():
        config["disable_ratio"] = False
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    if isDataMC:
        maps = ["mergemap"]
    else:
        maps = ["mergemap", "reference", "compare"]
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


def load_coffea(config, scaleToLumi=True):
    # print('config = ', config)
    if scaleToLumi:
        from BTVNanoCommissioning.utils.xs_scaler import getSumW, scaleSumW
    if "*" in config["input"]:
        files = glob.glob(config["input"])
        output = {i: load(i) for i in files}
        for out in output.keys():
            if scaleToLumi:
                output[out] = scaleSumW(
                    output[out], config["lumi"], getSumW(output[out])
                )
    elif len(config["input"]) > 0:
        output = {i: load(i) for i in config["input"]}
        for out in output.keys():
            if scaleToLumi:
                output[out] = scaleSumW(
                    output[out], config["lumi"], getSumW(output[out])
                )
    else:
        print("Input files are not provided in config")
        return None

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
                rebin_axis[rb] = hist.rebin(config["variable"][configvar]["rebin"][rb])
        ## default is set to be last axis
        else:
            rebin_axis[collated[baseaxis][var].axes[-1].name] = hist.rebin(
                config["variable"][configvar]["rebin"]
            )
    return xlabel, rebin_axis


### copy functions coffea.hist.plotratio https://github.com/CoffeaTeam/coffea/blob/master/coffea/hist/plot.py to boost-hist
################
## ratio plot ##
################


_coverage1sd = scipy.stats.norm.cdf(1) - scipy.stats.norm.cdf(-1)


def compatible(self, other, data_is_np=False):
    """Checks if this histogram is compatible with another, i.e. they have identical binning"""
    if data_is_np:
        if len(other.axes) != 1:
            return False
    else:
        if len(self.axes) != len(other.axes):
            return False
        if set(self.axes.name) != set(other.axes.name):
            return False
        if len(self.axes.edges) != len(other.axes.edges):
            return False
    return True


def poisson_interval(sumw, sumw2, coverage=_coverage1sd):
    """Frequentist coverage interval for Poisson-distributed observations
    Parameters
    ----------
        sumw : numpy.ndarray
            Sum of weights vector
        sumw2 : numpy.ndarray
            Sum weights squared vector
        coverage : float, optional
            Central coverage interval, defaults to 68%
    Calculates the so-called 'Garwood' interval,
    c.f. https://www.ine.pt/revstat/pdf/rs120203.pdf or
    http://ms.mcmaster.ca/peter/s743/poissonalpha.html
    For weighted data, this approximates the observed count by ``sumw**2/sumw2``, which
    effectively scales the unweighted poisson interval by the average weight.
    This may not be the optimal solution: see https://arxiv.org/pdf/1309.1287.pdf for a proper treatment.
    When a bin is zero, the scale of the nearest nonzero bin is substituted to scale the nominal upper bound.
    If all bins zero, a warning is generated and interval is set to ``sumw``.
    """
    scale = np.empty_like(sumw)
    scale[sumw != 0] = sumw2[sumw != 0] / sumw[sumw != 0]
    if np.sum(sumw == 0) > 0:
        missing = np.where(sumw == 0)
        available = np.nonzero(sumw)
        if len(available[0]) == 0:
            warnings.warn(
                "All sumw are zero!  Cannot compute meaningful error bars",
                RuntimeWarning,
            )
            return np.vstack([sumw, sumw])
        nearest = sum(
            [np.subtract.outer(d, d0) ** 2 for d, d0 in zip(available, missing)]
        ).argmin(axis=0)
        argnearest = tuple(dim[nearest] for dim in available)
        scale[missing] = scale[argnearest]
    counts = sumw / scale
    lo = scale * scipy.stats.chi2.ppf((1 - coverage) / 2, 2 * counts) / 2.0
    hi = scale * scipy.stats.chi2.ppf((1 + coverage) / 2, 2 * (counts + 1)) / 2.0
    interval = np.array([lo, hi])
    interval[interval == np.nan] = 0.0  # chi2.ppf produces nan for counts=0
    return interval


def normal_interval(pw, tw, pw2, tw2, coverage=_coverage1sd):
    """Compute errors based on the expansion of pass/(pass + fail), possibly weighted
    Parameters
    ----------
    pw : np.ndarray
        Numerator, or number of (weighted) successes, vectorized
    tw : np.ndarray
        Denominator or number of (weighted) trials, vectorized
    pw2 : np.ndarray
        Numerator sum of weights squared, vectorized
    tw2 : np.ndarray
        Denominator sum of weights squared, vectorized
    coverage : float, optional
        Central coverage interval, defaults to 68%
    c.f. https://root.cern.ch/doc/master/TEfficiency_8cxx_source.html#l02515
    """

    eff = pw / tw

    variance = (pw2 * (1 - 2 * eff) + tw2 * eff**2) / (tw**2)
    sigma = np.sqrt(variance)

    prob = 0.5 * (1 - coverage)
    delta = np.zeros_like(sigma)
    delta[sigma != 0] = scipy.stats.norm.ppf(prob, scale=sigma[sigma != 0])

    lo = eff - np.minimum(eff + delta, np.ones_like(eff))
    hi = np.maximum(eff - delta, np.zeros_like(eff)) - eff

    return np.array([lo, hi])


def clopper_pearson_interval(num, denom, coverage=_coverage1sd):
    """Compute Clopper-Pearson coverage interval for a binomial distribution
    Parameters
    ----------
        num : numpy.ndarray
            Numerator, or number of successes, vectorized
        denom : numpy.ndarray
            Denominator or number of trials, vectorized
        coverage : float, optional
            Central coverage interval, defaults to 68%
    c.f. http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    """
    if np.any(num > denom):
        raise ValueError(
            "Found numerator larger than denominator while calculating binomial uncertainty"
        )
    lo = scipy.stats.beta.ppf((1 - coverage) / 2, num, denom - num + 1)
    hi = scipy.stats.beta.ppf((1 + coverage) / 2, num + 1, denom - num)
    interval = np.array([lo, hi])
    interval[:, num == 0.0] = 0.0
    interval[1, num == denom] = 1.0
    return interval


## ratioplot function
def plotratio(
    num,
    denom,
    ax=None,
    clear=True,
    overflow="none",
    xerr=False,
    error_opts={},
    denom_fill_opts={},
    guide_opts={},
    unc="num",
    label=None,
    ext_denom_error=None,
    data_is_np=False,
):
    """Create a ratio plot, dividing two compatible histograms
    Parameters
    ----------
        num : Hist
            Numerator, a single-axis histogram
        denom : Hist
            Denominator, a single-axis histogram
        ax : matplotlib.axes.Axes, optional
            Axes object (if None, one is created)
        clear : bool, optional
            Whether to clear Axes before drawing (if passed); if False, this function will skip drawing the legend
        overflow : str, optional
            If overflow behavior is not 'none', extra bins will be drawn on either end of the nominal
            axis range, to represent the contents of the overflow bins.  See `Hist.sum` documentation
            for a description of the options.
        xerr: bool, optional
            If true, then error bars are drawn for x-axis to indicate the size of the bin.
        error_opts : dict, optional
            A dictionary of options to pass to the matplotlib
            `ax.errorbar <https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.errorbar.html>`_ call
            internal to this function.  Leave blank for defaults.  Some special options are interpreted by
            this function and not passed to matplotlib: 'emarker' (default: '') specifies the marker type
            to place at cap of the errorbar.
        denom_fill_opts : dict, optional
            A dictionary of options to pass to the matplotlib
            `ax.fill_between <https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.fill_between.html>`_ call
            internal to this function, filling the denominator uncertainty band.  Leave blank for defaults.
        guide_opts : dict, optional
            A dictionary of options to pass to the matplotlib
            `ax.axhline <https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.axhline.html>`_ call
            internal to this function, to plot a horizontal guide line at ratio of 1.  Leave blank for defaults.
        unc : str, optional
            Uncertainty calculation option: 'clopper-pearson' interval for efficiencies; 'poisson-ratio' interval
            for ratio of poisson distributions; 'num' poisson interval of numerator scaled by denominator value
            (common for data/mc, for better or worse).
        label : str, optional
            Associate a label to this entry (note: y axis label set by ``num.label``)
        ext_denom_error: list of np.array[error_down,error_upe], optional
            External MC errors not stored in the original histogram
        data_is_np : If data array is a numpy array, take sumw2=sumw
    Returns
    -------
        ax : matplotlib.axes.Axes
            A matplotlib `Axes <https://matplotlib.org/3.1.1/api/axes_api.html>`_ object
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        if not isinstance(ax, plt.Axes):
            raise ValueError("ax must be a matplotlib Axes object")
        if clear:
            ax.clear()
    if not compatible(num, denom, data_is_np):
        raise ValueError(
            "numerator and denominator histograms have incompatible axis definitions"
        )
    if len(denom.axes) > 1:
        raise ValueError("plotratio() can only support one-dimensional histograms")
    if error_opts is None and denom_fill_opts is None and guide_opts is None:
        error_opts = {}
        denom_fill_opts = {}

    axis = denom.axes[0]

    ax.set_xlabel(axis.label)
    ax.set_ylabel(denom.label)
    edges = axis.edges
    centers = axis.centers
    ranges = (edges[1:] - edges[:-1]) / 2 if xerr else None

    if data_is_np:
        sumw_num, sumw2_num = num, num
    else:
        sumw_num, sumw2_num = num.values(), num.variances()
    sumw_denom, sumw2_denom = denom.values(), denom.variances()

    rsumw = sumw_num / sumw_denom
    if unc == "clopper-pearson":
        rsumw_err = np.abs(clopper_pearson_interval(sumw_num, sumw_denom) - rsumw)
    elif unc == "poisson-ratio":
        # poisson ratio n/m is equivalent to binomial n/(n+m)
        rsumw_err = np.abs(
            clopper_pearson_interval(sumw_num, sumw_num + sumw_denom) - rsumw
        )
    elif unc == "num":
        rsumw_err = np.abs(poisson_interval(rsumw, sumw2_num / sumw_denom**2) - rsumw)
    elif unc == "efficiency":
        rsumw_err = np.abs(
            normal_interval(sumw_num, sumw_denom, sumw2_num, sumw2_denom)
        )
    else:
        raise ValueError("Unrecognized uncertainty option: %r" % unc)

    ## if additional uncertainties
    if ext_denom_error is not None:
        if denom_fill_opts is {}:
            print("suggest to use different style for additional error")
        if np.shape(rsumw_err) != np.shape(ext_denom_error / sumw_denom):
            raise ValueError("Imcompatible error length")
        rsumw_err = np.sqrt(rsumw_err**2 + (ext_denom_error / sumw_denom) ** 2)

    if error_opts is not None:
        opts = {
            "label": label,
            "linestyle": "none",
            "lw": 1,
            "marker": "o",
            "color": "k",
        }
        opts.update(error_opts)
        emarker = opts.pop("emarker", "")
        errbar = ax.errorbar(x=centers, y=rsumw, xerr=ranges, yerr=rsumw_err, **opts)
        plt.setp(errbar[1], "marker", emarker)
    if denom_fill_opts is not None:
        unity = np.ones_like(sumw_denom)
        denom_unc = poisson_interval(unity, sumw2_denom / sumw_denom**2)
        opts = {
            "hatch": "////",
            "facecolor": "none",
            "lw": 0,
            "color": "k",
            "alpha": 0.4,
        }
        if ext_denom_error is not None:
            denom_unc[0] = (
                unity[0]
                - np.sqrt(
                    (denom_unc - unity) ** 2 + (ext_denom_error / sumw_denom) ** 2
                )[0]
            )
            denom_unc[1] = (
                unity[1]
                + np.sqrt(
                    (denom_unc - unity) ** 2 + (ext_denom_error / sumw_denom) ** 2
                )[1]
            )
            opts = denom_fill_opts
        ax.stairs(denom_unc[0], edges=edges, baseline=denom_unc[1], **opts)
    if guide_opts is not None:
        opts = {"linestyle": "--", "color": (0, 0, 0, 0.5), "linewidth": 1}
        opts.update(guide_opts)
        if clear is not False:
            ax.axhline(1.0, **opts)

    if clear:
        ax.autoscale(axis="x", tight=True)
        ax.set_ylim(0, None)

    return ax


def autoranger(hist):
    val, axis = hist.values(), hist.axes[-1].edges
    mins,maxs = 0,len(val)
    for i in range(len(val)):
        if val[i] != 0:
            mins = i
            break
    for i in reversed(range(len(val))):
        if val[i] != 0:
            maxs = i + 1
            break
    return axis[mins], axis[maxs]
