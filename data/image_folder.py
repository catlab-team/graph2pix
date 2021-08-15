###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################
import torch.utils.data as data
from PIL import Image
import os, sys
import pandas as pd
from glob import glob

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

"""
Make dataset for singleview input.
"""
def make_sv_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                if fname[fname.index("_")+1:fname.index("_")+3] ==  'fr':
                    images.append(path)

    return images

"""
Make dataset for multiview input.
"""
def make_mv_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    last_clothe_name = ""
    last_clothe_img = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                if fname[:fname.index("_")] == last_clothe_name:
                    last_clothe_img.append(path)
                else:
                    if len(last_clothe_img) == 3:
                        images.append(last_clothe_img)
                    last_clothe_name = fname[:fname.index("_")]
                    last_clothe_img = [path]

    return images


"""
Make dataset for artbreeder input.

Files under dir should have the following structure:
- 00045c6c9d334af71a54_parent*.jpg
- 00045c6c9d334af71a54_parent*.jpg
- ...
- 00045c6c9d334af71a54_z.jpg -> denotes child
- 00062f5234b1d146e2a2_parent*.jpg
- 00062f5234b1d146e2a2_parent*.jpg
- ...
- 00062f5234b1d146e2a2_z.jpg
- ...
"""
def make_art_dataset(dir, is_mv):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    
    df = pd.DataFrame({'path': glob(os.path.join(dir, "*.jpg"))})
    df['img'] = df.path.apply(lambda x: x.split("/")[-1].split('_')[0])

    accepted = [
        "parent_0_0.jpg", 
        "parent_0_1.jpg", 
        "parent_1_0.jpg",
        "parent_1_1.jpg", 
        "parent_0.jpg", 
        "parent_1.jpg", 
        "z.jpg"
    ]

    df_list = df.groupby("img")['path'].apply(list).to_frame().reset_index()
    df_list['real'] = df_list.path.apply(lambda x: [i for i in x if "z.jpg" in i][0])

    def sort_files(x):
        def find_elem(x, elem):
            for i in x:
                if elem in i:
                    return i
                
            raise Exception("Not found!")
            
        return [find_elem(x, elem) for elem in accepted]

    df_list['label'] = df_list.path.apply(lambda x: sort_files(x))

    return df_list.label.values, df_list.real.values
            


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
