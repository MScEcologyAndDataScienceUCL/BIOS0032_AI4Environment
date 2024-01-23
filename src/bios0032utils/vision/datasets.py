import torch
from PIL import Image


class CTDataset(torch.utils.data.Dataset):
    """
    This utility corresponds to the Dataset class construction for the custom
    example of camera trap data. The class is responsible for how images and
    the corresponding data are loaded in batches during training or evaluation.
    """

    def __init__(self, root_dir, annotation_dict, transform):
        # Given annotation dictionary for a given split and a set of
        # transformations we initialize the properties of the class
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = []
        self.targets = []
        self.species = []
        self.location = []
        self.datetime = []
        self.split = []

        for row in annotation_dict:
            img_path, target, species, location, datetime, split, _ = row
            self.img_paths.append(img_path)
            self.targets.append(target)
            self.species.append(species)
            self.location.append(location)
            self.datetime.append(datetime)
            self.split.append(split)

    def __len__(self):
        return len(self.img_paths)

    def get_image(self, img_path):
        with open(self.root_dir + img_path, "rb") as f:
            try:
                img = Image.open(f)
                return img.convert("RGB")
            except:
                print(
                    f"Image from {img_path} cannot be read. "
                    "It will be skipped."
                )
                return None

    def __getitem__(self, idx):
        op = {}

        ## Next part is commented out in student version
        op["target"] = self.targets[idx]
        op["img_path"] = self.img_paths[idx]
        op["species"] = self.species[idx]
        op["location"] = self.location[idx]
        op["datetime"] = self.datetime[idx]
        op["split"] = self.split[idx]

        img_path = self.img_paths[idx]
        img = self.get_image(img_path)
        if img != None:
            op["img"] = self.transform(img)
            return op
        else:
            return None
