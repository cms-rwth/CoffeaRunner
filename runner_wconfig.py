import os, sys, json, argparse, time, pickle


import numpy as np

import uproot
from coffea.util import load, save
from coffea import processor

from BTVNanoCommissioning.utils.Configurator import Configurator

# Should come up with a smarter way to import all worflows from subdirectories of ./src/
def validate(file):
    try:
        fin = uproot.open(file)
        return fin["Events"].num_entries
    except:
        print("Corrupted file: {}".format(file))
        return


def check_port(port):
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("0.0.0.0", port))
        available = True
    except:
        available = False
    sock.close()
    return available


def retry_handler(exception, task_record):
    from parsl.executors.high_throughput.interchange import ManagerLost

    if isinstance(exception, ManagerLost):
        return 0.1
    else:
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run analysis on baconbits files using processor coffea files"
    )
    # Inputs
    parser.add_argument(
        "--cfg",
        default=os.getcwd() + "/config/example.py",
        required=True,
        type=str,
        help="Config file with parameters specific to the current run",
    )
    parser.add_argument(
        "-o",
        "--overwrite_file",
        required=False,
        type=str,
        help="Overwrite the output in the configuration",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Do not process, just check all files are accessible",
    )
    args = parser.parse_args()
    if args.cfg[-3:] == ".py":
        config = Configurator(args.cfg, overwrite_output_dir=args.overwrite_file)
    elif args.cfg[-4:] == ".pkl":
        config = pickle.load(open(args.cfg, "rb"))
    else:
        raise sys.exit("Please provide a .py configuration file")

    # Scan if files can be opened
    if args.validate:
        start = time.time()
        from p_tqdm import p_map

        all_invalid = []
        for sample in config.fileset.keys():
            _rmap = p_map(
                validate,
                config.fileset[sample],
                num_cpus=config.run_options["workers"],
                desc=f"Validating {sample[:20]}...",
            )
            _results = list(_rmap)
            counts = np.sum([r for r in _results if np.isreal(r)])
            all_invalid += [r for r in _results if type(r) == str]
            print("Events:", np.sum(counts))
        print("Bad files:")
        for fi in all_invalid:
            print(f"  {fi}")
        end = time.time()
        print("TIME:", time.strftime("%H:%M:%S", time.gmtime(end - start)))
        if input("Remove bad files? (y/n)") == "y":
            print("Removing:")
            for fi in all_invalid:
                print(f"Removing: {fi}")
                os.system(f"rm {fi}")
        sys.exit(0)

    if config.run_options["executor"] not in [
        "futures",
        "iterative",
        "dask/lpc",
        "dask/casa",
    ]:
        """
        dask/parsl needs to export x509 to read over xrootd
        dask/lpc uses custom jobqueue provider that handles x509
        """
        if config.run_options["voms"] is not None:
            _x509_path = config.run_options["voms"]
        else:
            try:
                _x509_localpath = (
                    [
                        l
                        for l in os.popen("voms-proxy-info").read().split("\n")
                        if l.startswith("path")
                    ][0]
                    .split(":")[-1]
                    .strip()
                )
            except:
                raise RuntimeError(
                    "x509 proxy could not be parsed, try creating it with 'voms-proxy-init'"
                )
            _x509_path = os.environ["HOME"] + f'/.{_x509_localpath.split("/")[-1]}'
            os.system(f"cp {_x509_localpath} {_x509_path}")

        env_extra = [
            "export XRD_RUNFORKHANDLER=1",
            f"export X509_USER_PROXY={_x509_path}",
            f'export X509_CERT_DIR={os.environ["X509_CERT_DIR"]}',
            f"export PYTHONPATH=$PYTHONPATH:{os.getcwd()}",
        ]
        condor_extra = [
            f"cd {os.getcwd()}",
            f'source {os.environ["HOME"]}/.bashrc',
            f'conda activate {os.environ["CONDA_PREFIX"]}',
        ]

    #########
    # Execute
    if config.run_options["executor"] in ["futures", "iterative"]:
        if config.run_options["executor"] == "iterative":
            _exec = processor.iterative_executor
        else:
            _exec = processor.futures_executor
        output = processor.run_uproot_job(
            config.fileset,
            treename="Events",
            processor_instance=config.processor_instance,
            executor=_exec,
            executor_args={
                "skipbadfiles": config.run_options["skipbadfiles"],
                "schema": processor.NanoAODSchema,
                "workers": config.run_options["workers"],
            },
            chunksize=config.run_options["chunk"],
            maxchunks=config.run_options["max"],
        )
    elif "parsl" in config.run_options["executor"]:
        import parsl
        from parsl.providers import LocalProvider, CondorProvider, SlurmProvider
        from parsl.channels import LocalChannel
        from parsl.config import Config
        from parsl.executors import HighThroughputExecutor
        from parsl.launchers import SrunLauncher
        from parsl.addresses import address_by_hostname, address_by_query

        if "slurm" in config.run_options["executor"]:
            htex_config = Config(
                executors=[
                    HighThroughputExecutor(
                        label="coffea_parsl_slurm",
                        address=address_by_hostname(),
                        prefetch_capacity=0,
                        provider=SlurmProvider(
                            channel=LocalChannel(script_dir="logs_parsl"),
                            launcher=SrunLauncher(),
                            max_blocks=(config.run_options["scaleout"]) + 10,
                            init_blocks=config.run_options["scaleout"],
                            partition="all",
                            worker_init="\n".join(env_extra),
                            walltime=config.run_options["walltime"],
                        ),
                    )
                ],
                retries=config.run_options["retries"],
            )
            if config.run_options["splitjobs"]:
                htex_config = Config(
                    executors=[
                        HighThroughputExecutor(
                            label="run",
                            address=address_by_hostname(),
                            prefetch_capacity=0,
                            provider=SlurmProvider(
                                channel=LocalChannel(script_dir="logs_parsl"),
                                launcher=SrunLauncher(),
                                max_blocks=(config.run_options["scaleout"]) + 10,
                                init_blocks=config.run_options["scaleout"],
                                partition="all",
                                worker_init="\n".join(env_extra),
                                walltime=config.run_options["walltime"],
                            ),
                        ),
                        HighThroughputExecutor(
                            label="merge",
                            address=address_by_hostname(),
                            prefetch_capacity=0,
                            provider=SlurmProvider(
                                channel=LocalChannel(script_dir="logs_parsl"),
                                launcher=SrunLauncher(),
                                max_blocks=(config.run_options["scaleout"]) + 10,
                                init_blocks=config.run_options["scaleout"],
                                partition="all",
                                worker_init="\n".join(env_extra),
                                walltime="00:30:00",
                            ),
                        ),
                    ],
                    retries=config.run_options["retries"],
                    retry_handler=retry_handler,
                )
        elif "condor" in config.run_options["executor"]:
            if "naf_lite" in config.run_options["executor"]:
                config.run_options["mem_per_worker"] = 2
                config.run_options["walltime"] = "03:00:00"
            htex_config = Config(
                executors=[
                    HighThroughputExecutor(
                        label="coffea_parsl_condor",
                        address=address_by_query(),
                        max_workers=1,
                        worker_debug=True,
                        provider=CondorProvider(
                            nodes_per_block=1,
                            cores_per_slot=config.run_options["workers"],
                            mem_per_slot=config.run_options["mem_per_worker"],
                            init_blocks=config.run_options["scaleout"],
                            max_blocks=(config.run_options["scaleout"]) + 2,
                            worker_init="\n".join(env_extra + condor_extra),
                            walltime=config.run_options["walltime"],
                        ),
                    )
                ],
                retries=config.run_options["retries"],
                retry_handler=retry_handler,
            )
            if config.run_options["splitjobs"]:
                htex_config = Config(
                    executors=[
                        HighThroughputExecutor(
                            label="run",
                            address=address_by_query(),
                            max_workers=1,
                            worker_debug=True,
                            provider=CondorProvider(
                                nodes_per_block=1,
                                cores_per_slot=config.run_options["workers"],
                                mem_per_slot=config.run_options["mem_per_worker"],
                                init_blocks=config.run_options["scaleout"],
                                max_blocks=(config.run_options["scaleout"]) + 2,
                                worker_init="\n".join(env_extra + condor_extra),
                                walltime=config.run_options[
                                    "walltime"
                                ],  # lite / short queue requirement
                            ),
                        ),
                        HighThroughputExecutor(
                            label="merge",
                            address=address_by_query(),
                            max_workers=1,
                            worker_debug=True,
                            provider=CondorProvider(
                                nodes_per_block=1,
                                cores_per_slot=config.run_options["workers"],
                                mem_per_slot=2,  # lite job / opportunistic can only use this much
                                init_blocks=config.run_options["scaleout"],
                                max_blocks=(config.run_options["scaleout"]) + 2,
                                worker_init="\n".join(env_extra + condor_extra),
                                walltime="00:30:00",  # lite / short queue requirement
                            ),
                        ),
                    ],
                    retries=config.run_options["retries"],
                    retry_handler=retry_handler,
                )

        else:
            raise NotImplementedError

        dfk = parsl.load(htex_config)
        if not config.run_options["splitjobs"]:
            executor_args_condor = {
                "skipbadfiles": config.run_options["skipbadfiles"],
                "schema": processor.NanoAODSchema,
                "config": None,
            }
        else:
            executor_args = {
                "skipbadfiles": args.skipbadfiles,
                "schema": processor.NanoAODSchema,
                "merging": True,
                "merges_executors": ["merge"],
                "jobs_executors": ["run"],
                "config": None,
            }
        output = processor.run_uproot_job(
            config.fileset,
            treename="Events",
            processor_instance=config.processor_instance,
            executor=processor.parsl_executor,
            executor_args=executor_args_condor,
            chunksize=config.run_options["chunk"],
            maxchunks=config.run_options["max"],
        )

    elif "dask" in config.run_options["executor"]:
        from dask_jobqueue import SLURMCluster, HTCondorCluster
        from distributed import Client
        from dask.distributed import performance_report

        if "lpc" in config.run_options["executor"]:
            env_extra = [
                f"export PYTHONPATH=$PYTHONPATH:{os.getcwd()}",
            ]
            from lpcjobqueue import LPCCondorCluster

            cluster = LPCCondorCluster(
                transfer_input_files="/srv/src/",
                ship_env=True,
                env_extra=env_extra,
            )
        elif "lxplus" in config.run_options["executor"]:
            n_port = 8786
            if not check_port(8786):
                raise RuntimeError(
                    "Port '8786' is not occupied on this node. Try another one."
                )
            import socket

            cluster = HTCondorCluster(
                cores=1,
                memory="2GB",  # hardcoded
                disk="1GB",
                death_timeout="60",
                nanny=False,
                scheduler_options={"port": n_port, "host": socket.gethostname()},
                job_extra={
                    "log": "dask_job_output.log",
                    "output": "dask_job_output.out",
                    "error": "dask_job_output.err",
                    "should_transfer_files": "Yes",
                    "when_to_transfer_output": "ON_EXIT",
                    "+JobFlavour": '"workday"',
                },
                extra=["--worker-port {}".format(n_port)],
                env_extra=env_extra,
            )
        elif "slurm" in config.run_options["executor"]:
            cluster = SLURMCluster(
                queue="all",
                cores=config.run_options["workers"],
                processes=config.run_options["workers"],
                memory=config.run_options["mem_per_worker"],
                retries=config.run_options["retries"],
                walltime=config.run_options["walltime"],
                env_extra=env_extra,
            )
        elif "condor" in config.run_options["executor"]:
            cluster = HTCondorCluster(
                cores=config.run_options["workers"],
                memory=config.run_options["mem_per_worker"],
                env_extra=env_extra,
            )

        if config.run_options["executor"] == "dask/casa":
            client = Client("tls://localhost:8786")
            import shutil

            shutil.make_archive("workflows", "zip", base_dir="workflows")
            client.upload_file("workflows.zip")
        else:
            cluster.adapt(minimum=config.run_options["scaleout"])
            client = Client(cluster)
            print("Waiting for at least one worker...")
            client.wait_for_workers(1)
        with performance_report(filename="dask-report.html"):
            output = processor.run_uproot_job(
                config.fileset,
                treename="Events",
                processor_instance=config.processor_instance,
                executor=processor.dask_executor,
                executor_args={
                    "client": client,
                    "skipbadfiles": config.run_options["skipbadfiles"],
                    "schema": processor.NanoAODSchema,
                    "retries": config.run_options["retries"],
                },
                chunksize=config.run_options["chunk"],
                maxchunks=config.run_options["max"],
            )

    save(output, config.outfile)

    print(output)
    print(f"Saving output to {config.outfile}")
