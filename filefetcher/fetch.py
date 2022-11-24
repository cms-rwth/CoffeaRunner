import os
import json
import argparse

parser = argparse.ArgumentParser(
    description="Run analysis on baconbits files using processor coffea files"
)
parser.add_argument(
    "-i",
    "--input",
    default=r"singlemuon",
    help="List of samples in DAS (default: %(default)s)",
)
parser.add_argument(
    "-o", "--output", default=r"test_my_samples", help="Site (default: %(default)s)"
)
parser.add_argument(
    "--xrd",
    default="root://xrootd-cms.infn.it//",
    type=str,
    help="xrootd prefix string (default: %(default)s)",
)

args = parser.parse_args()
fset = []

with open(args.input) as fp:
    lines = fp.readlines()
    for line in lines:
        fset.append(line)

fdict = {}

for dataset in fset:
    if dataset.startswith("#") or dataset.strip() == "":
        # print("we skip this line:", line)
        continue
    dsname = dataset.strip().split("/")[1]  # Dataset first name
    Tier = dataset.strip().split("/")[
        3
    ]  # NANOAODSIM for regular samples, USER for private
    instance = "prod/global"
    if Tier == "USER":
        instance = "prod/phys03"
    print("Creating list of files for dataset", dsname, Tier, instance)
    flist = (
        os.popen(
            (
                "/cvmfs/cms.cern.ch/common/dasgoclient -query='instance={} file dataset={}'"
            ).format(instance, fset[fset.index(dataset)].rstrip())
        )
        .read()
        .split("\n")
    )

    if dsname not in fdict:
        fdict[dsname] = [args.xrd + f for f in flist if len(f) > 1]
    else:  # needed to collect all data samples into one common key "Data" (using append() would introduce a new element for the key)
        fdict[dsname].extend([args.xrd + f for f in flist if len(f) > 1])

# pprint.pprint(fdict, depth=1)

output_file = "./metadata/%s" % (args.output)
with open(output_file, "w") as fp:
    json.dump(fdict, fp, indent=4)
    print("The file is saved at: ", output_file)
