from src.pipeline.nodes import download_from_roboflow_node, roboflow_split_to_hdf5
from src.params.utils import yaml_parser
from src.loggers.loggers import Logger
import yaml

import argparse
import os


def main(config: dict, logger: Logger):
    """script to downlotad a dataset from Roboflow and then serialize it into a
    hdf5 file in a format ready for object detection training. Data access schema
    of the created hdf5 file would be:

    uid/
    ├── bbox    (n,4) array, where n is number of objects
    ├── cat_id  (n,) array
    ├── img     (h,w,3) array
    ├── area    (n,) array
    └── iscrowd (n,) array


    Args:
        config (dict): config dict
        logger (Logger): logger for recording execution steps
    """
    ds_cfg = config["dataset"]
    workspace = ds_cfg["workspace"]
    project = ds_cfg["project"]
    version = ds_cfg["version"]
    format = ds_cfg["format"]
    overwrite = ds_cfg["overwrite"]
    savedir = ds_cfg["download_dir"]

    download_from_roboflow_node(
        workspace, project, version, format, overwrite, savedir, logger
    )

    hf_cfg = config["hdf5"]
    possible_splits = set(hf_cfg["possible_splits"])
    files_found = set(os.listdir(savedir))
    split_lookup = {
        s: os.path.join(savedir, s) for s in possible_splits.intersection(files_found)
    }

    annotation_file = hf_cfg["annotation_file"]
    for split, split_dir in split_lookup.items():
        hf_savepath = os.path.join(hf_cfg["save_dir"], "{}.hdf5".format(split))
        roboflow_split_to_hdf5(split_dir, annotation_file, hf_savepath, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-config")
    parser.add_argument("--log")
    args = parser.parse_args()
    config = yaml_parser(args.pipeline_config)
    logger = Logger(args.log)

    logger.info("pipeline config")
    logger.info(yaml.dump(config, default_flow_style=False, sort_keys=False))

    main(config, logger)
