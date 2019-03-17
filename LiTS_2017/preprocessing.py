import nibabel as nib
import numpy as np
import skimage.transform
import os

MEAN_SPACING = np.array([0.79272507, 0.79272507, 1.50625819])
DATA_DIR = "../"


def get_spacing(image):
    affine = image.affine

    return np.abs(np.array([affine[0, 0], affine[1, 1], affine[2, 2]]))


if not os.path.exists(DATA_DIR + "image_np"):
    os.makedirs(DATA_DIR + "image_np")
if not os.path.exists(DATA_DIR + "label_np"):
    os.makedirs(DATA_DIR + "label_np")

for i in range(131):
    image_file = DATA_DIR + "imagesTr/volume-" + str(i) + ".nii.gz"
    image = nib.load(image_file)
    spacing = get_spacing(image)
    shape = image.shape
    image = image.get_data().copy()
    label = nib.load(DATA_DIR + "labelsTr/segmentation-" + str(i) + ".nii.gz").get_data().copy()
    shape = list(np.round(shape * spacing / MEAN_SPACING).astype(np.int32))
    image = skimage.transform.resize(image, shape, order=1, preserve_range=True, anti_aliasing=False)
    label = skimage.transform.resize(label, shape, order=0, preserve_range=True, anti_aliasing=False)
    np.save(DATA_DIR + "image_np/liver_" + str(i), image.astype(np.float32))
    np.save(DATA_DIR + "label_np/liver_label_" + str(i), label.astype(np.int8))  # save some storage space

if not os.path.exists(DATA_DIR + "image_test_np"):
    os.makedirs(DATA_DIR + "image_test_np")

for i in range(70):
    image_file = DATA_DIR + "imagesTs/test-volume-" + str(i) + ".nii.gz"
    image = nib.load(image_file)
    spacing = get_spacing(image)
    shape = image.shape
    image = image.get_data().copy()
    shape = list(np.round(shape * spacing / MEAN_SPACING).astype(np.int32))
    image = skimage.transform.resize(image, shape, order=1, preserve_range=True, anti_aliasing=False)
    np.save("../image_test_np/liver_" + str(i), image.astype(np.float32))
