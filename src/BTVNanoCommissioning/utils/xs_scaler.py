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
        if type(key) != str or key == "Data" or  "Run" in key:
            continue
        scales[key] = xs_dict[key] * lumi / events[key]
    hist.scale(scales, axis="dataset")
    return hist
def scale_xs_arr(events,lumi,xsfile="xsection.json"):

    xs_dict = read_xs(xsfile)
    scales={}
    wei_array={}
    for key in events:
        if type(key) != str or key=="Data" or "Run" in key: continue
        scales[key]=xs_dict[key]*lumi/events[key]

        wei_array[key] = scales[key]
    return wei_array
