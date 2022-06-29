def read_xs(file):
    import json

    f = open(file)
    data = json.load(f)
    xs_dict = {}
    for obj in data:
        xs_dict[obj["process_name"]] = float(obj["cross_section"])
    return xs_dict


def scale_xs(hist, lumi, events, xsfile="xsection.json"):
    xs_dict = read_xs(xsfile)
    scales = {}
    for key in events:
        if type(key) != str or key == "Data" or "Run" in key:
            continue
        scales[key] = xs_dict[key] * lumi / events[key]
    hist.scale(scales, axis="dataset")
    return hist


def scale_xs_arr(events, lumi, xsfile="xsection.json"):
    xs_dict = read_xs(xsfile)
    scales = {}
    wei_array = {}
    for key in events:
        if type(key) != str or key == "Data" or "Run" in key:
            continue
        scales[key] = xs_dict[key] * lumi / events[key]

        wei_array[key] = scales[key]
    return wei_array
def scaleSumW(accumulator, sumw, lumi, xsfile="xsection.json"):
    scaled = {}
    xs_dict = read_xs(xsfile)
    for sample, accu in accumulator.items():
        scaled[sample] = {}
        for key, h_obj in accu.items():
            if isinstance(h_obj, hist.Hist):
                h = copy.deepcopy(h_obj)
                if sample in xs_dict.keys():
                    h = h * xs_dict[sample] *lumi /sumw[sample] 
                else:
                    if not (("data" in sample) or ("Run" in key)):
                        warnings.warn(f"Sample ``{sample}`` cross-section not found. (MC won't be included).")
                scaled[sample][key] = h
    return scaled


def collate(accumulator, mergemap):
    out = {}
    for group, names in mergemap.items():
        out[group] = processor.accumulate([v for k, v in accumulator.items() if k in names])
    return out
def getSumW(accumulator):
    sumw = {}
    for key, accus in accumulator.items():
        sumw[key] = accus['sumw']
    return sumw