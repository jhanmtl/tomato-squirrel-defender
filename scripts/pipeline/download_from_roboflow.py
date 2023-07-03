import argparse
import os
from typing import Dict

from roboflow import Roboflow
from src.loggers.loggers import Logger
from src.params.utils import yaml_parser


def main(datasets: Dict[str, Dict[str, str]], savedir: str, logger: Logger):
    """downloads a list of datasets from roboflow and save them to a
    specified location

    Args:
        datasets (Dict[str, Dict[str, str]]): config dict of the datasets to download.
        See ../../data_pipeline/sources/roboflow/download-config.yaml
        savedir (str): where to save the downloads
        logger (Logger): for loggin the execution to stdout and a file
    """

    rf = Roboflow(api_key=os.environ["ROBOFLOW_TOKEN"])
    for _, configs in datasets.items():
        workspace = configs["workspace"]
        project = configs["project"]
        version = configs["version"]
        format = configs["format"]

        each_savedir = os.path.join(savedir, "{}_{}".format(workspace, project))

        logger.info(
            "downloading roboflow dataset {} to {}".format(project, each_savedir)
        )

        session = rf.workspace(workspace).project(project)
        dataset = session.version(version).download(
            format, location=each_savedir, overwrite=False
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--log")
    args = parser.parse_args()

    logger = Logger(args.log)

    params = yaml_parser(args.config)
    main(params["datasets"], params["save_dir"], logger)
