import numpy as np
import pandas as pd
from PIL import Image


def load_image_into_numpy_array(image_path, output_image_dimensions=None):
    """
    Given an image path and output dimensions, load and return an image as
    numpy array
    """
    img = Image.open(image_path)

    if output_image_dimensions:
        img = img.resize(output_image_dimensions)

    return np.array(img)


def transform_md_output_to_df(md_detections, category_dict):
    """
    Given a python dictionary with detections coming from MegaDetector, this
    function transforms MegaDetector output into a simple pandas dataframe that
    is easier to be used for analysis
    """
    md_df = pd.DataFrame()
    for md_tagged_img in md_detections["images"]:
        row = {"image_id": md_tagged_img["file"]}
        conservancy = "".join(
            [
                char
                for char in md_tagged_img["file"].split("_")[1]
                if char.isalpha()
            ]
        )
        row["conservancy"] = conservancy
        if len(md_tagged_img["detections"]) > 0:
            for det_row in md_tagged_img["detections"]:
                row["confidence"] = det_row["conf"]
                row["category"] = category_dict[det_row["category"]]
                md_df = md_df.append(row, ignore_index=True)
        else:
            row["confidence"] = 0.0
            row["category"] = "empty"
            md_df = md_df.append(row, ignore_index=True)
    return md_df
