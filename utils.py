"""
CFUN

Common utility functions and classes.
"""

import torch
import numpy as np
import nibabel as nib
import skimage.transform
from distutils.version import LooseVersion

import torch.nn.functional as F


############################################################
#  Bounding Boxes
############################################################

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [depth, height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (z1, y1, x1, z2, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 6], dtype=np.int32)
    for i in range(mask.shape[-1]):
        # Bounding box.
        ix = np.where(np.sum(mask, axis=2) > 0)
        z1 = ix[0].min()
        z2 = ix[0].max()
        y1 = ix[1].min()
        y2 = ix[1].max()
        ix = np.where(np.sum(mask, axis=0) > 0)
        x1 = ix[1].min()
        x2 = ix[1].max()
        # x2, y2 and z2 should not be part of the box. Increment by 1.
        if z1 != z2:
            z2 += 1
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2, z1, z2 = 0, 0, 0, 0, 0, 0
        boxes[i] = np.array([z1, y1, x1, z2, y2, x2])
    return boxes.astype(np.int32)


def compute_iou(box, boxes, box_volume, boxes_volume):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [z1, y1, x1, z2, y2, x2]
    boxes: [boxes_count, (z1, y1, x1, z2, y2, x2)]
    box_volume: float. the volume of 'box'
    boxes_volume: array of depth boxes_count.

    Note: the volumes are passed in rather than calculated here for
          efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection volumes
    z1 = np.maximum(box[0], boxes[:, 0])
    z2 = np.minimum(box[3], boxes[:, 3])
    y1 = np.maximum(box[1], boxes[:, 1])
    y2 = np.minimum(box[4], boxes[:, 4])
    x1 = np.maximum(box[2], boxes[:, 2])
    x2 = np.minimum(box[5], boxes[:, 5])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0) * np.maximum(z2 - z1, 0)
    union = box_volume + boxes_volume[:] - intersection[:]
    iou = intersection / (union + 1e-6)
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (z1, y1, x1, z2, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # volumes of anchors and GT boxes
    volume1 = (boxes1[:, 3] - boxes1[:, 0]) * (boxes1[:, 4] - boxes1[:, 1]) * (boxes1[:, 5] - boxes1[:, 2])
    volume2 = (boxes2[:, 3] - boxes2[:, 0]) * (boxes2[:, 4] - boxes2[:, 1]) * (boxes2[:, 5] - boxes2[:, 2])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, volume2[i], volume1)
    return overlaps


def box_refinement(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (z1, y1, x1, z2, y2, x2)]
    """

    depth = box[:, 3] - box[:, 0]
    height = box[:, 4] - box[:, 1]
    width = box[:, 5] - box[:, 2]
    center_z = box[:, 0] + 0.5 * depth
    center_y = box[:, 1] + 0.5 * height
    center_x = box[:, 2] + 0.5 * width

    gt_depth = gt_box[:, 3] - gt_box[:, 0]
    gt_height = gt_box[:, 4] - gt_box[:, 1]
    gt_width = gt_box[:, 5] - gt_box[:, 2]
    gt_center_z = gt_box[:, 0] + 0.5 * gt_depth
    gt_center_y = gt_box[:, 1] + 0.5 * gt_height
    gt_center_x = gt_box[:, 2] + 0.5 * gt_width

    dz = (gt_center_z - center_z) / depth
    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dd = torch.log(gt_depth / depth)
    dh = torch.log(gt_height / height)
    dw = torch.log(gt_width / width)

    result = torch.stack([dz, dy, dx, dd, dh, dw], dim=1)
    return result


def non_max_suppression(boxes, scores, threshold, max_num):
    """Performs non-maximum suppression and returns indices of kept boxes.
    boxes: [N, (z1, y1, x1, z2, y2, x2)]. Notice that (z2, y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.
    max_num: Int. The max number of boxes to keep.
    Return the index of boxes to keep.
    """
    # Compute box volumes
    z1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x1 = boxes[:, 2]
    z2 = boxes[:, 3]
    y2 = boxes[:, 4]
    x2 = boxes[:, 5]
    volume = (z2 - z1) * (y2 - y1) * (x2 - x1)

    # Get indices of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        if len(pick) >= max_num:
            break
        # Compute IoU of the picked box with the rest
        iou = compute_iou(boxes[i], boxes[ixs[1:]], volume[i], volume[ixs[1:]])
        # Identify boxes with IoU over the threshold. This returns indices into ixs[1:],
        # so add 1 to get indices into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def denorm_boxes_graph(boxes, size):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [..., (z1, y1, x1, z2, y2, x2)] in normalized coordinates
    size: [depth, height, width], the size to denorm to.

    Note: In pixel coordinates (z2, y2, x2) is outside the box.
          But in normalized coordinates it's inside the box.

    Returns:
        [..., (z1, y1, x1, z2, y2, x2)] in pixel coordinates
    """
    d, h, w = size
    scale = torch.Tensor([d, h, w, d, h, w]).cuda()
    denorm_boxes = torch.mul(boxes, scale)
    return denorm_boxes


############################################################
#  Dataset
############################################################

class Dataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:

    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...

    See COCODataset and ShapesDataset as examples.
    """

    def __init__(self, class_map=None):
        """
        Initialize the class_class

        Args:
            self: (todo): write your description
            class_map: (todo): write your description
        """
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        """
        Add a class to the class.

        Args:
            self: (todo): write your description
            source: (str): write your description
            class_id: (str): write your description
            class_name: (str): write your description
        """
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        """
        Add an image

        Args:
            self: (todo): write your description
            source: (str): write your description
            image_id: (str): write your description
            path: (str): write your description
        """
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.

        Override for your dataset, but pass to this function
        if you encounter images not in your dataset.
        """
        return ""

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """
        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    def append_data(self, class_info, image_info):
        """
        Append image data to the image.

        Args:
            self: (todo): write your description
            class_info: (todo): write your description
            image_info: (todo): write your description
        """
        self.external_to_class_id = {}
        for i, c in enumerate(self.class_info):
            for ds, id in c["map"]:
                self.external_to_class_id[ds + str(id)] = i

        # Map external image IDs to internal ones.
        self.external_to_image_id = {}
        for i, info in enumerate(self.image_info):
            self.external_to_image_id[info["ds"] + str(info["id"])] = i

    @property
    def image_ids(self):
        """
        : return : a list of the ids

        Args:
            self: (todo): write your description
        """
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's available online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        """Load the specified image and return a [H, W, D, 1] Numpy array."""
        # Load image
        image = nib.load(self.image_info[image_id]['path']).get_data().copy()
        return np.expand_dims(image, -1)

    def load_mask(self, image_id):
        """Load the specified mask and return a [H, W, D] Numpy array."""
        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.
        mask = np.empty([0, 0, 0])
        return mask


def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=True, anti_aliasing=False, anti_aliasing_sigma=None):
    """A wrapper for Scikit-Image resize().

    Scikit-Image generates warnings on every call to resize() if it doesn't
    receive the right parameters. The right parameters depend on the version
    of skimage. This solves the problem by using different parameters per
    version. And it provides a central place to control resizing defaults.
    """
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        # New in 0.14: anti_aliasing. Default it to False for backward
        # compatibility with skimage 0.13.
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range, anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma)
    else:
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range)


def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    """Resizes an image keeping the aspect ratio.

    min_dim: if provided, resizes the image such that it's smaller
             dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
             exceed this value.
    min_scale: if provided, ensure that the image is scaled up by at least
               this percent even if min_dim doesn't require it.
    mode: Resizing mode.
        none: No resizing. Return the image unchanged.
        square: Resize and pad with zeros to get a square image
                of size [max_dim, max_dim, max_dim].
        pad64: Pads width and height with zeros to make them multiples of 64.
               If min_dim or min_scale are provided, it scales the image up
               before padding. max_dim is ignored in this mode.
               The multiple of 64 is needed to ensure smooth scaling of feature
               maps up and down the 6 levels of the FPN pyramid (2**6=64).
        crop: Picks random crops from the image. First, scales the image based
              on min_dim and min_scale, then picks a random crop of
              size min_dim x min_dim. Can be used in training only.
              max_dim is not used in this mode.
        self: Self-designed resize strategy.
              Resize the image to [IMAGE_MAX_DIM, IMAGE_MAX_DIM, IMAGE_MIN_DIM]

    Returns:
    image: the resized image of [height, width, depth, channels]
    window: (z1, y1, x1, z2, y2, x2). If max_dim is provided, padding might
            be inserted in the returned image. If so, this window is the
            coordinates of the image part of the full image (excluding
            the padding). The x2, y2, z2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (front, back), (0, 0)]
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (z1, y1, x1, z2, y2, x2) and default scale == 1.
    h, w, d = image.shape[:3]
    window = (0, 0, 0, d, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Self-designed resize strategy.
    if mode == "self":
        image = resize(image, (max_dim, max_dim, min_dim, 1),
                       order=1, mode="constant", preserve_range=True)
        return image.astype(image_dtype), (0, 0, 0, min_dim, max_dim, max_dim), -1, \
            [(0, 0), (0, 0), (0, 0), (0, 0)], crop


def resize_mask(mask, scale, padding, max_dim=0, min_dim=0, crop=None, mode="square"):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (front, back), (0, 0)]
    """
    # Self-designed resize strategy.
    if mode == "self":
        mask = resize(mask, (max_dim, max_dim, min_dim), order=0, mode='constant', preserve_range=True)
        return np.round(mask).astype(np.int32)


def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to cut memory load.
    Mini-masks can then resized back to image scale using expand_masks()
    """
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, :, i]
        z1, y1, x1, z2, y2, x2 = bbox[i][:6]
        m = m[z1:z2, y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Invalid bounding box with volume of zero")
        m = skimage.transform.resize(m, mini_shape, order=0, mode="constant", preserve_range=True)
        mini_mask[:, :, :, i] = np.around(m).astype(np.int32)
    return mini_mask


def expand_mask(bbox, mini_mask, image_shape):
    """Resizes mini masks back to image size. Reverses the change
    of minimize_mask().
    """
    mask = np.zeros(image_shape[:3] + (mini_mask.shape[-1],), dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mini_mask[:, :, :, i]
        z1, y1, x1, z2, y2, x2 = bbox[i][:6]
        d = z2 - z1
        h = y2 - y1
        w = x2 - x1
        m = skimage.transform.resize(m, (d, h, w), order=1, mode="constant", preserve_range=True)
        mask[z1:z2, y1:y2, x1:x2, i] = np.around(m).astype(np.int32)
    return mask


def unmold_mask(mask, bbox, image_shape):
    """Converts a mask generated by the neural network into a format similar
    to it's original shape.
    mask: [depth, height, width, num_instances] of type float. A small, typically 28x28 mask.
    bbox: [z1, y1, x1, z2, y2, x2]. The box to fit the mask in.
    image_shape: [channels, depth, height, width]

    Returns a tf.int32 mask with the same size as the original image.
    """
    z1, y1, x1, z2, y2, x2 = bbox
    mask = torch.from_numpy(mask).float().cuda()
    mask = mask.permute(3, 0, 1, 2).unsqueeze(0)
    mask = F.interpolate(mask, size=(z2 - z1, y2 - y1, x2 - x1), mode='trilinear', align_corners=False)
    mask = mask.squeeze(0).detach().cpu().numpy().transpose(1, 2, 3, 0)
    # Put the mask in the right location.
    full_mask = np.zeros((image_shape[1], image_shape[2], image_shape[3], mask.shape[-1]), dtype=np.float32)
    full_mask[z1:z2, y1:y2, x1:x2, :] = mask
    return full_mask


############################################################
#  Anchors
############################################################

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [1]
    shape: [depth, height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    # TODO: conditions when we have different ratios?
    # Here I apply a trick.
    depths = scales
    heights = scales
    widths = scales

    # Enumerate shifts in feature space
    shifts_z = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_y = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[2], anchor_stride) * feature_stride
    shifts_z, shifts_y, shifts_x = np.meshgrid(shifts_z, shifts_y, shifts_x)

    # Enumerate combinations of shifts, widths, and heights
    box_depths, box_centers_z = np.meshgrid(depths, shifts_z)
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (z, y, x) and a list of (d, h, w)
    box_centers = np.stack(
        [box_centers_z, box_centers_y, box_centers_x], axis=2).reshape([-1, 3])
    box_sizes = np.stack([box_depths, box_heights, box_widths], axis=2).reshape([-1, 3])

    # Convert to corner coordinates (z1, y1, x1, z2, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (z1, y1, x1, z2, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (z1, y1, x1, z2, y2, x2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0)


############################################################
#  Miscellaneous
############################################################

# ## Batch Slicing
# Some custom layers support a batch size of 1 only, and require a lot of work
# to support batches greater than 1. This function slices an input tensor
# across the batch dimension and feeds batches of size 1. Effectively,
# an easy way to support batches > 1 quickly with little code modification.
# In the long run, it's more efficient to modify the code to support large
# batches and getting rid of this function. Consider this a temporary solution
def batch_slice(inputs, graph_fn, batch_size, names=None):
    """
    Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.

    inputs: list of tensors. All must have the same first dimension size
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [torch.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result


def compute_per_class_mask_iou(gt_masks, pred_masks):
    """Computes per_class_IoU overlaps between two sets of masks.
    gt_masks, pred_masks: [Height, Width, Depth, instances], zero-padding if there's no such instance.
    Returns ious per instance.
    """
    # flatten masks and compute their areas
    gt_masks = np.reshape(gt_masks > .5, (-1, gt_masks.shape[-1])).astype(np.float32)
    pred_masks = np.reshape(pred_masks > .5, (-1, pred_masks.shape[-1])).astype(np.float32)
    area1 = np.sum(gt_masks, axis=0)
    area2 = np.sum(pred_masks, axis=0)

    # intersections and union
    intersections = np.array([np.dot(gt_masks.T, pred_masks)[i, i] for i in range(gt_masks.shape[-1])])
    union = area1 + area2 - intersections
    ious = intersections / (union + 1e-6)  # avoid intersections to be divided by 0

    return ious


def compute_mask_iou(gt_masks, pred_masks):
    """Computes IoU overlaps between two sets of masks. Regard different classes as the same.
    gt_masks, pred_masks: [Height, Width, Depth].
    Returns ious of the two masks.
    """
    # flatten masks and compute their areas
    gt_masks[gt_masks > 0] = 1
    pred_masks[pred_masks > 0] = 1
    gt_masks = np.reshape(gt_masks, (-1)).astype(np.int32)
    pred_masks = np.reshape(pred_masks, (-1)).astype(np.int32)
    area1 = np.sum(gt_masks)
    area2 = np.sum(pred_masks)

    # intersections and union
    intersections = np.dot(gt_masks.T, pred_masks)
    union = area1 + area2 - intersections
    ious = intersections / (union + 1e-6)  # avoid intersections to be divided by 0

    return ious
