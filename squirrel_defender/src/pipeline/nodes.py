import json
import os

import cv2
import h5py
import pandas as pd
from roboflow import Roboflow
from src.loggers.loggers import Logger
from tqdm import tqdm


def download_from_roboflow_node(
    workspace: str,
    project: str,
    version: int,
    format: str,
    overwrite: bool,
    savedir: str,
    logger: Logger,
    api_key: str = os.environ["ROBOFLOW_TOKEN"],
) -> None:
    """downloads a dataset from Roboflow. Roboflow datasets are specified by
    the url convention https://universe.roboflow.com/<workspace>/<project>/dataset/<version>.

    Args:
        workspace (str): dataset workspace. See above
        project (str): dataset project. See above
        version (int): dataset version. See above
        savedir (str): local save location
        format (str): object detection dataset format, coco should be used where possible
        overwrite (bool): whether to overwrite if dataset already downloaded to savedir
        logger (Logger): Logger object to log execution details to file and stdout
        api_key (str, optional): Roboflow api token. Defaults to os.environ["ROBOFLOW_TOKEN"]. Supply if otherwise
    """
    rf = Roboflow(api_key=api_key)
    session = rf.workspace(workspace).project(project)
    logger.info(
        "downloading roboflow dataset from {}/{} to {}".format(
            workspace, project, savedir
        )
    )
    session.version(version).download(format, location=savedir, overwrite=overwrite)


def roboflow_split_to_hdf5(
    split_dir: str, annotation_file: str, hdf5_savepath: str, logger: Logger
) -> None:
    """serializes the a particular split (train, test, valid) of a dataset downloaded from Roboflow
    into a training-ready hdf5 file. The hdf5 file would have the following access schema

    uid/
    ├── bbox    (n,4) array, where n is number of objects
    ├── cat_id  (n,) array
    ├── img     (h,w,3) array
    ├── area    (n,) array
    └── iscrowd (n,) array

    Args:
        split_dir (str): directory containing the split of the downloaded dataset
        annotation_file (str): name of the file that contains the annotation info
        hdf5_savepath (str): where to save the hdf5 file
        logger (Logger): logger to record execution
    """

    msg = "saving annotations and images from {} to hdf5 at {}".format(
        split_dir, hdf5_savepath
    )
    logger.info(msg)

    annotation_path = os.path.join(split_dir, annotation_file)
    with open(annotation_path, "r") as f:
        annotation_json = json.load(f)

    annotations_df = pd.DataFrame(annotation_json["annotations"])
    annotations_df["x1"] = annotations_df["bbox"].apply(lambda x: x[0])
    annotations_df["y1"] = annotations_df["bbox"].apply(lambda x: x[1])
    annotations_df["x2"] = annotations_df["bbox"].apply(lambda x: x[0] + x[2])
    annotations_df["y2"] = annotations_df["bbox"].apply(lambda x: x[1] + x[3])
    annotations_df = annotations_df.drop(columns=["bbox", "segmentation"])

    image_df = pd.DataFrame(annotation_json["images"])
    image_df = image_df.rename(columns={"id": "image_id"})

    hf = h5py.File(hdf5_savepath, "w")

    for image_id, anno_df in tqdm(annotations_df.groupby("image_id")):
        # there can be multiple annotation entries for the same image_id image if there are multiple objects in that image

        img_row = image_df[image_df["image_id"] == image_id]
        assert len(img_row) == 1
        img_row = img_row.iloc[0]
        imgpath = os.path.join(split_dir, img_row["file_name"])

        img = cv2.imread(imgpath)[:, :, [2, 1, 0]]
        bbox = anno_df[["x1", "y1", "x2", "y2"]].values
        cat_id = anno_df["category_id"].values
        area = anno_df["area"].values
        iscrowd = anno_df["iscrowd"].values

        hf.create_dataset(f"{image_id}/img", data=img)
        hf.create_dataset(f"{image_id}/bbox", data=bbox)
        hf.create_dataset(f"{image_id}/cat_id", data=cat_id)
        hf.create_dataset(f"{image_id}/area", data=area)
        hf.create_dataset(f"{image_id}/iscrowd", data=iscrowd)
