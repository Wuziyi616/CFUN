"""
CFUN

The main CFUN model implementation.
"""

import time
import math
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import skimage.transform

import utils
import backbone
import mask_branch

############################################################
#  Logging Utility Functions
############################################################


def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else ""))
    print(text)


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill=''):
    """Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\n')
    # Print New Line on Complete
    if iteration == total:
        print()


############################################################
#  Pytorch Utility Functions
############################################################

def unique1d(tensor):
    """
    Return unique elements of a tensor.

    Args:
        tensor: (todo): write your description
    """
    if tensor.size()[0] == 0 or tensor.size()[0] == 1:
        return tensor
    tensor = tensor.sort()[0]
    unique_bool = tensor[1:] != tensor[:-1]
    first_element = Variable(torch.ByteTensor([True]), requires_grad=False)
    if tensor.is_cuda:
        first_element = first_element.cuda()
    unique_bool = torch.cat((first_element, unique_bool), dim=0)

    return tensor[unique_bool.detach()]


def intersect1d(tensor1, tensor2):
    """
    Intersect two tensors.

    Args:
        tensor1: (todo): write your description
        tensor2: (todo): write your description
    """
    aux = torch.cat((tensor1, tensor2), dim=0)
    aux = aux.sort()[0]

    return aux[:-1][(aux[1:] == aux[:-1]).detach()]


def log2(x):
    """Implementation of log2. Pytorch doesn't have a native implementation."""
    ln2 = torch.log(torch.FloatTensor([2.0]))
    if x.is_cuda:
        ln2 = ln2.cuda()

    return torch.log(x) / ln2


def compute_backbone_shapes(config, image_shape):
    """Computes the depth, width and height of each stage of the backbone network.
    Returns:
        [N, (depth, height, width)]. Where N is the number of stages
    """
    H, W, D = image_shape[:3]

    return np.array(
        [[int(math.ceil(D / stride)),
          int(math.ceil(H / stride)),
          int(math.ceil(W / stride))]
         for stride in config.BACKBONE_STRIDES])


############################################################
#  FPN Graph
############################################################

class FPN(nn.Module):
    def __init__(self, C1, C2, C3, out_channels, config):
        """
        Initialize the convolution layer.

        Args:
            self: (todo): write your description
            C1: (float): write your description
            C2: (int): write your description
            C3: (int): write your description
            out_channels: (int): write your description
            config: (todo): write your description
        """
        super(FPN, self).__init__()
        self.out_channels = out_channels
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.P3_conv1 = nn.Conv3d(config.BACKBONE_CHANNELS[1] * 4, self.out_channels, kernel_size=1, stride=1)
        self.P3_conv2 = nn.Conv3d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.P2_conv1 = nn.Conv3d(config.BACKBONE_CHANNELS[0] * 4, self.out_channels, kernel_size=1, stride=1)
        self.P2_conv2 = nn.Conv3d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        Evaluates the forward.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        x = self.C1(x)
        x = self.C2(x)
        c2_out = x
        x = self.C3(x)
        c3_out = x

        p3_out = self.P3_conv1(c3_out)
        p2_out = self.P2_conv1(c2_out) + F.upsample(p3_out, scale_factor=2)
        p3_out = self.P3_conv2(p3_out)
        p2_out = self.P2_conv2(p2_out)

        return [p2_out, p3_out]


############################################################
#  Proposal Layer
############################################################

def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, 6] where each row is z1, y1, x1, z2, y2, x2
    deltas: [N, 6] where each row is [dz, dy, dx, log(dd), log(dh), log(dw)]
    """
    # Convert to z, y, x, d, h, w
    depth = boxes[:, 3] - boxes[:, 0]
    height = boxes[:, 4] - boxes[:, 1]
    width = boxes[:, 5] - boxes[:, 2]
    center_z = boxes[:, 0] + 0.5 * depth
    center_y = boxes[:, 1] + 0.5 * height
    center_x = boxes[:, 2] + 0.5 * width
    # Apply deltas
    center_z += deltas[:, 0] * depth
    center_y += deltas[:, 1] * height
    center_x += deltas[:, 2] * width
    depth *= torch.exp(deltas[:, 3])
    height *= torch.exp(deltas[:, 4])
    width *= torch.exp(deltas[:, 5])
    # Convert back to z1, y1, x1, z2, y2, x2
    z1 = center_z - 0.5 * depth
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    z2 = z1 + depth
    y2 = y1 + height
    x2 = x1 + width
    result = torch.stack([z1, y1, x1, z2, y2, x2], dim=1)

    return result


def clip_boxes(boxes, window):
    """boxes: [N, 6] each col is z1, y1, x1, z2, y2, x2
    window: [6] in the form z1, y1, x1, z2, y2, x2
    """
    boxes = torch.stack(
        [boxes[:, 0].clamp(float(window[0]), float(window[3])),
         boxes[:, 1].clamp(float(window[1]), float(window[4])),
         boxes[:, 2].clamp(float(window[2]), float(window[5])),
         boxes[:, 3].clamp(float(window[0]), float(window[3])),
         boxes[:, 4].clamp(float(window[1]), float(window[4])),
         boxes[:, 5].clamp(float(window[2]), float(window[5]))], 1)

    return boxes


def proposal_layer(inputs, proposal_count, nms_threshold, anchors, config=None):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinement deltas to anchors.
    Inputs:
        rpn_probs: [batch, anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, anchors, (dz, dy, dx, log(dd), log(dh), log(dw))]
    Returns:
        Proposals in normalized coordinates [batch, rois, (z1, y1, x1, z2, y2, x2)]
    """

    # Currently only supports batchsize 1
    inputs[0] = inputs[0].squeeze(0)
    inputs[1] = inputs[1].squeeze(0)

    # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
    scores = inputs[0][:, 1]

    # Box deltas [batch, num_rois, 6]
    deltas = inputs[1]
    std_dev = torch.from_numpy(np.reshape(config.RPN_BBOX_STD_DEV, [1, 6])).float()
    if config.GPU_COUNT:
        std_dev = std_dev.cuda()
    deltas = deltas * std_dev

    # Improve performance by trimming to top anchors by score
    # and doing the rest on the smaller subset.
    pre_nms_limit = min(config.PRE_NMS_LIMIT, anchors.size()[0])
    scores, order = scores.sort(descending=True)
    order = order[:pre_nms_limit]
    scores = scores[:pre_nms_limit]
    deltas = deltas[order.detach(), :]
    anchors = anchors[order.detach(), :]

    # Apply deltas to anchors to get refined anchors.
    # [batch, N, (z1, y1, x1, z2, y2, x2)]
    boxes = apply_box_deltas(anchors, deltas)

    # Clip to image boundaries. [batch, N, (z1, y1, x1, z2, y2, x2)]
    height, width, depth = config.IMAGE_SHAPE[:3]
    window = np.array([0, 0, 0, depth, height, width]).astype(np.float32)
    boxes = clip_boxes(boxes, window)

    # Non-max suppression
    keep = utils.non_max_suppression(boxes.cpu().detach().numpy(),
                                     scores.cpu().detach().numpy(), nms_threshold, proposal_count)
    keep = torch.from_numpy(keep).long()
    boxes = boxes[keep, :]

    # Normalize dimensions to range of 0 to 1.
    norm = torch.from_numpy(np.array([depth, height, width, depth, height, width])).float()
    if config.GPU_COUNT:
        norm = norm.cuda()
    normalized_boxes = boxes / norm

    # Add back batch dimension
    normalized_boxes = normalized_boxes.unsqueeze(0)

    return normalized_boxes


############################################################
#  ROIAlign Layer
############################################################

def RoI_Align(feature_map, pool_size, boxes):
    """Implementation of 3D RoI Align (actually it's just pooling rather than align).
    feature_map: [channels, depth, height, width]. Generated from FPN.
    pool_size: [D, H, W]. The shape of the output.
    boxes: [num_boxes, (z1, y1, x1, z2, y2, x2)].
    """
    boxes = utils.denorm_boxes_graph(boxes, (feature_map.size()[1], feature_map.size()[2], feature_map.size()[3]))
    boxes[:, 0] = boxes[:, 0].floor()
    boxes[:, 1] = boxes[:, 1].floor()
    boxes[:, 2] = boxes[:, 2].floor()
    boxes[:, 3] = boxes[:, 3].ceil()
    boxes[:, 4] = boxes[:, 4].ceil()
    boxes[:, 5] = boxes[:, 5].ceil()
    boxes = boxes.long()
    output = torch.zeros((boxes.size()[0], feature_map.size()[0], pool_size[0], pool_size[1], pool_size[2])).cuda()
    for i in range(boxes.size()[0]):
        try:
            output[i] = F.interpolate((feature_map[:, boxes[i][0]:boxes[i][3], boxes[i][1]:boxes[i][4], boxes[i][2]:boxes[i][5]]).unsqueeze(0),
                                      size=pool_size, mode='trilinear', align_corners=True).cuda()
        except:
            # print("RoI_Align error!")
            # print("box:", boxes[i], "feature_map size:", feature_map.size())
            pass

    return output.cuda()


def pyramid_roi_align(inputs, pool_size, test_flag=False):
    """Implements ROI Pooling on multiple levels of the feature pyramid.
    Params:
    - pool_size: [depth, height, width] of the output pooled regions. Usually [7, 7, 7]
    - image_shape: [height, width, depth, channels]. Shape of input image in pixels
    Inputs:
    - boxes: [batch, num_boxes, (z1, y1, x1, z2, y2, x2)] in normalized coordinates.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, channels, depth, height, width]
    Output:
    Pooled regions in the shape: [num_boxes, channels, depth, height, width].
    The width, height and depth are those specific in the pool_shape in the layer
    constructor.
    """
    # Currently only supports batchsize 1
    if test_flag:
        for i in range(0, len(inputs)):
            inputs[i] = inputs[i].squeeze(0)
    else:
        for i in range(1, len(inputs)):
            inputs[i] = inputs[i].squeeze(0)

    # Crop boxes [batch, num_boxes, (y1, x1, z1, y2, x2, z2)] in normalized coordinates
    boxes = inputs[0]
    # Feature Maps. List of feature maps from different level of the
    # feature pyramid. Each is [batch, channels, depth, height, width]
    feature_maps = inputs[1:]

    # Assign each ROI to a level in the pyramid based on the ROI volume.
    z1, y1, x1, z2, y2, x2 = boxes.chunk(6, dim=1)
    d = z2 - z1
    h = y2 - y1
    w = x2 - x1

    # Equation 1 in the Feature Pyramid Networks paper.
    # Account for the fact that our coordinates are normalized here.
    # TODO: change the equation here
    roi_level = 4 + (1. / 3.) * log2(h * w * d)
    roi_level = roi_level.round().int()
    roi_level = roi_level.clamp(2, 3)

    # Loop through levels and apply ROI pooling to P2 or P3.
    pooled = []
    box_to_level = []
    for i, level in enumerate(range(2, 4)):
        ix = (roi_level == level)
        if not ix.any():
            continue
        ix = torch.nonzero(ix)[:, 0]
        level_boxes = boxes[ix.detach(), :]

        # Keep track of which box is mapped to which level
        box_to_level.append(ix.detach())

        # Stop gradient propagation to ROI proposals
        level_boxes = level_boxes.detach()

        # Crop and Resize
        # From Mask R-CNN paper: "We sample four regular locations, so that we can evaluate
        # either max or average pooling. In fact, interpolating only a single value at each bin center
        # (without pooling) is nearly as effective."
        # Here we use the simplified approach of a single value per bin.
        # Result: [batch * num_boxes, channels, pool_depth, pool_height, pool_width]
        pooled_features = RoI_Align(feature_maps[i], pool_size, level_boxes)
        pooled.append(pooled_features)

    # Pack pooled features into one tensor
    pooled = torch.cat(pooled, dim=0)

    # Pack box_to_level mapping into one array and add another
    # column representing the order of pooled boxes
    box_to_level = torch.cat(box_to_level, dim=0)

    # Rearrange pooled features to match the order of the original boxes
    _, box_to_level = torch.sort(box_to_level)
    pooled = pooled[box_to_level, :, :, :]

    return pooled


############################################################
#  Detection Target Layer
############################################################

def bbox_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (z1, y1, x1, z2, y2, x2)].
    """
    # 1. Tile boxes2 and repeat boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    boxes1_repeat = boxes2.size()[0]
    boxes2_repeat = boxes1.size()[0]
    boxes1 = boxes1.repeat(1, boxes1_repeat).view(-1, 6)
    boxes2 = boxes2.repeat(boxes2_repeat, 1)

    # 2. Compute intersections
    b1_z1, b1_y1, b1_x1, b1_z2, b1_y2, b1_x2 = boxes1.chunk(6, dim=1)
    b2_z1, b2_y1, b2_x1, b2_z2, b2_y2, b2_x2 = boxes2.chunk(6, dim=1)
    z1 = torch.max(b1_z1, b2_z1)[:, 0]
    y1 = torch.max(b1_y1, b2_y1)[:, 0]
    x1 = torch.max(b1_x1, b2_x1)[:, 0]
    z2 = torch.min(b1_z2, b2_z2)[:, 0]
    y2 = torch.min(b1_y2, b2_y2)[:, 0]
    x2 = torch.min(b1_x2, b2_x2)[:, 0]
    zeros = Variable(torch.zeros(z1.size()[0]), requires_grad=False)
    if z1.is_cuda:
        zeros = zeros.cuda()
    intersection = torch.max(x2 - x1, zeros) * torch.max(y2 - y1, zeros) * torch.max(z2 - z1, zeros)

    # 3. Compute unions
    b1_volume = (b1_z2 - b1_z1) * (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_volume = (b2_z2 - b2_z1) * (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_volume[:, 0] + b2_volume[:, 0] - intersection

    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = iou.view(boxes2_repeat, boxes1_repeat)

    return overlaps


def detection_target_layer(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    """Subsamples proposals and generates target box refinement, class_ids,
    and masks for each.
    Inputs:
    proposals: [batch, N, (z1, y1, x1, z2, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, (0, 1, ..., num_classes - 1)] Integer class IDs.
    gt_boxes: [batch, num_classes - 1, (z1, y1, x1, z2, y2, x2)] in normalized coordinates.
    gt_masks: [batch, num_classes, depth, height, width] of np.int32 type
    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (z1, y1, x1, z2, y2, x2)] in normalized coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, NUM_CLASSES,
                    (dz, dy, dx, log(dd), log(dh), log(dw), class_id)]
                   Class-specific bbox refinements.
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, depth, height, width)
                 Masks cropped to bbox boundaries and resized to neural
                 network output size.
    """
    # import pdb
    # Currently only supports batchsize 1
    proposals = proposals.squeeze(0)
    gt_class_ids = gt_class_ids.squeeze(0)
    gt_boxes = gt_boxes.squeeze(0)
    gt_masks = gt_masks.squeeze(0)

    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = bbox_overlaps(proposals, gt_boxes)
    # pdb.set_trace()

    # Determine positive and negative ROIs
    roi_iou_max = torch.max(overlaps, dim=1)[0]
    # print("rpn_roi_iou_max:", roi_iou_max.max())

    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = roi_iou_max >= config.DETECTION_TARGET_IOU_THRESHOLD

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    if torch.nonzero(positive_roi_bool).size()[0] != 0:
        positive_indices = torch.nonzero(positive_roi_bool)[:, 0]

        positive_count = int(round(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO))
        rand_idx = torch.randperm(positive_indices.size()[0])
        rand_idx = rand_idx[:positive_count]
        if config.GPU_COUNT:
            rand_idx = rand_idx.cuda()
        positive_indices = positive_indices[rand_idx]
        positive_count = positive_indices.size()[0]
        positive_rois = proposals[positive_indices.detach(), :]
        # Assign positive ROIs to GT boxes.
        positive_overlaps = overlaps[positive_indices.detach(), :]
        roi_gt_box_assignment = torch.max(positive_overlaps, dim=1)[1]
        roi_gt_boxes = gt_boxes[roi_gt_box_assignment.detach(), :]
        roi_gt_class_ids = gt_class_ids[roi_gt_box_assignment.detach()]

        # Compute bbox refinement for positive ROIs
        deltas = Variable(utils.box_refinement(positive_rois.detach(), roi_gt_boxes.detach()), requires_grad=False)
        std_dev = torch.from_numpy(config.BBOX_STD_DEV).float()
        if config.GPU_COUNT:
            std_dev = std_dev.cuda()
        deltas /= std_dev
        # Assign positive ROIs to GT masks
        # Permute masks to [N, depth, height, width]
        # Pick the right mask for each ROI
        roi_gt_masks = np.zeros((positive_rois.shape[0], gt_masks.shape[0],) + config.MASK_SHAPE)
        for i in range(0, positive_rois.shape[0]):
             z1 = int(gt_masks.shape[1] * positive_rois[i, 0])
             z2 = int(gt_masks.shape[1] * positive_rois[i, 3])
             y1 = int(gt_masks.shape[2] * positive_rois[i, 1])
             y2 = int(gt_masks.shape[2] * positive_rois[i, 4])
             x1 = int(gt_masks.shape[3] * positive_rois[i, 2])
             x2 = int(gt_masks.shape[3] * positive_rois[i, 5])
             crop_mask = gt_masks[:, z1:z2, y1:y2, x1:x2].cpu().numpy()
             crop_mask = skimage.transform.resize(crop_mask, (gt_masks.shape[0],) + config.MASK_SHAPE, order=0,
                                                  preserve_range=True, mode="constant", anti_aliasing=False)
             roi_gt_masks[i, :, :, :, :] = crop_mask
        roi_gt_masks = torch.from_numpy(roi_gt_masks).cuda()
        roi_gt_masks = roi_gt_masks.type(torch.DoubleTensor)
        masks = roi_gt_masks
    else:
        positive_count = 0

    # 2. Negative ROIs are those with < 0.5 with every GT box.
    negative_roi_bool = roi_iou_max < config.DETECTION_TARGET_IOU_THRESHOLD
    negative_roi_bool = negative_roi_bool
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    if torch.nonzero(negative_roi_bool).size()[0] != 0 and positive_count > 0:
        negative_indices = torch.nonzero(negative_roi_bool)[:, 0]
        r = 1.0 / config.ROI_POSITIVE_RATIO
        negative_count = int(round(r * positive_count - positive_count))
        rand_idx = torch.randperm(negative_indices.size()[0])
        rand_idx = rand_idx[:negative_count]
        if config.GPU_COUNT:
            rand_idx = rand_idx.cuda()
        negative_indices = negative_indices[rand_idx]
        negative_count = negative_indices.size()[0]
        negative_rois = proposals[negative_indices.detach(), :]
    else:
        negative_count = 0

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    if positive_count > 0 and negative_count > 0:
        rois = torch.cat((positive_rois, negative_rois), dim=0)
        zeros = Variable(torch.zeros(negative_count), requires_grad=False).long()
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        roi_gt_class_ids = torch.cat([roi_gt_class_ids.long(), zeros], dim=0)
        zeros = Variable(torch.zeros(negative_count, 6), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        deltas = torch.cat([deltas, zeros], dim=0)
        zeros = Variable(torch.zeros((negative_count,) + config.MASK_SHAPE), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        masks = roi_gt_masks
    elif positive_count > 0:
        rois = positive_rois
    elif negative_count > 0:
        positive_rois = Variable(torch.FloatTensor(), requires_grad=False)
        rois = negative_rois
        zeros = Variable(torch.zeros(negative_count), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
            positive_rois = positive_rois.cuda()
        roi_gt_class_ids = zeros
        zeros = Variable(torch.zeros(negative_count, 6), requires_grad=False).int()
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        deltas = zeros
        zeros = Variable(torch.zeros((negative_count,) + config.MASK_SHAPE), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        masks = zeros
    else:
        positive_rois = Variable(torch.FloatTensor(), requires_grad=False)
        rois = Variable(torch.FloatTensor(), requires_grad=False)
        roi_gt_class_ids = Variable(torch.IntTensor(), requires_grad=False)
        deltas = Variable(torch.FloatTensor(), requires_grad=False)
        masks = Variable(torch.FloatTensor(), requires_grad=False)
        if config.GPU_COUNT:
            positive_rois = positive_rois.cuda()
            rois = rois.cuda()
            roi_gt_class_ids = roi_gt_class_ids.cuda()
            deltas = deltas.cuda()
            masks = masks.cuda()

    return positive_rois, rois, roi_gt_class_ids, deltas, masks


############################################################
#  Detection Layer
############################################################

def clip_to_window(window, boxes):
    """window: (z1, y1, x1, z2, y2, x2). The window in the image we want to clip to.
        boxes: [N, (z1, y1, x1, z2, y2, x2)]
    """
    boxes[:, 0] = boxes[:, 0].clamp(float(window[0]), float(window[3]))
    boxes[:, 1] = boxes[:, 1].clamp(float(window[1]), float(window[4]))
    boxes[:, 2] = boxes[:, 2].clamp(float(window[2]), float(window[5]))
    boxes[:, 3] = boxes[:, 3].clamp(float(window[0]), float(window[3]))
    boxes[:, 4] = boxes[:, 4].clamp(float(window[1]), float(window[4]))
    boxes[:, 5] = boxes[:, 5].clamp(float(window[2]), float(window[5]))

    return boxes


def refine_detections(rois, probs, deltas, window, config):
    """Refine classified proposals and filter overlaps and return final
    detections.
    Inputs:
        rois: [N, (z1, y1, x1, z2, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dz, dy, dx, log(dd), log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (z1, y1, x1, z2, y2, x2) in image coordinates. The part of the image
            that contains the image excluding the padding.
    Returns detections shaped: [N, (z1, y1, x1, z2, y2, x2, class_id, score)]
    """
    # import pdb
    # Class IDs per ROI
    _, class_ids = torch.max(probs, dim=1)
    # pdb.set_trace()

    # Class probability of the top class of each ROI
    # Class-specific bounding box deltas
    idx = torch.arange(class_ids.size()[0]).long()
    if config.GPU_COUNT:
        idx = idx.cuda()
    class_scores = probs[idx, class_ids.detach()]
    deltas_specific = deltas[idx, class_ids.detach()]
    # pdb.set_trace()

    # Apply bounding box deltas
    # Shape: [boxes, (z1, y1, x1, z2, y2, x2)] in normalized coordinates
    std_dev = torch.from_numpy(np.reshape(config.RPN_BBOX_STD_DEV, [1, 6])).float()
    if config.GPU_COUNT:
        std_dev = std_dev.cuda()
    refined_rois = apply_box_deltas(rois, deltas_specific * std_dev)
    # pdb.set_trace()

    # Convert coordinates to image domain
    height, width, depth = config.IMAGE_SHAPE[:3]
    scale = torch.from_numpy(np.array([depth, height, width, depth, height, width])).float()
    if config.GPU_COUNT:
        scale = scale.cuda()
    refined_rois *= scale
    # pdb.set_trace()

    # Clip boxes to image window
    refined_rois = clip_to_window(window, refined_rois)

    # Round and cast to int since we're dealing with pixels now
    refined_rois = torch.round(refined_rois)

    # Filter out background boxes
    keep_bool = class_ids > 0

    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        keep_bool = keep_bool & (class_scores >= config.DETECTION_MIN_CONFIDENCE)
    keep = torch.nonzero(keep_bool)[:, 0]
    # pdb.set_trace()

    # Apply per-class NMS
    pre_nms_class_ids = class_ids[keep.detach()]
    pre_nms_scores = class_scores[keep.detach()]
    pre_nms_rois = refined_rois[keep.detach()]
    # pdb.set_trace()

    for i, class_id in enumerate(unique1d(pre_nms_class_ids)):
        # Pick detections of this class
        ixs = torch.nonzero(pre_nms_class_ids == class_id)[:, 0]

        # Sort
        ix_rois = pre_nms_rois[ixs.detach()]
        ix_scores = pre_nms_scores[ixs]
        ix_scores, order = ix_scores.sort(descending=True)
        ix_rois = ix_rois[order.detach(), :]

        class_keep = utils.non_max_suppression(ix_rois.cpu().detach().numpy(), ix_scores.cpu().detach().numpy(),
                                               config.DETECTION_NMS_THRESHOLD, config.DETECTION_MAX_INSTANCES)
        class_keep = torch.from_numpy(class_keep).long()
        # pdb.set_trace()

        # Map indices
        class_keep = keep[ixs[order[class_keep].detach()].detach()]
        # pdb.set_trace()

        if i == 0:
            nms_keep = class_keep
        else:
            nms_keep = unique1d(torch.cat((nms_keep, class_keep)))
    keep = intersect1d(keep, nms_keep)

    # Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    roi_count = min(roi_count, keep.size()[0])
    # pdb.set_trace()
    top_ids = class_scores[keep.detach()].sort(descending=True)[1][:roi_count]
    keep = keep[top_ids.detach()]

    # Arrange output as [N, (z1, y1, x1, z2, y2, x2, class_id, score)]
    # Coordinates are in image domain.
    result = torch.cat((refined_rois[keep.detach()],
                        class_ids[keep.detach()].unsqueeze(1).float(),
                        class_scores[keep.detach()].unsqueeze(1)), dim=1)

    return result


def detection_layer(config, rois, mrcnn_class, mrcnn_bbox, image_meta):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.
    Returns:
    [batch, num_detections, (z1, y1, x1, z2, y2, x2, class_score)] in pixels
    """
    # Currently only supports batchsize 1
    rois = rois.squeeze(0)

    _, _, window, _ = parse_image_meta(image_meta)
    window = window[0]
    detections = refine_detections(rois, mrcnn_class, mrcnn_bbox, window, config)

    return detections


############################################################
#  Region Proposal Network
############################################################

class RPN(nn.Module):
    """Builds the model of Region Proposal Network.
    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    Returns:
        rpn_logits: [batch, D, H, W, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, D, H, W, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, D, H, W, (dz, dy, dx, log(dd), log(dh), log(dw))] Deltas to be applied to anchors.
    """

    def __init__(self, anchors_per_location, anchor_stride, channel, conv_channel):
        """
        Initialize the convolution layer.

        Args:
            self: (todo): write your description
            anchors_per_location: (todo): write your description
            anchor_stride: (int): write your description
            channel: (todo): write your description
            conv_channel: (todo): write your description
        """
        super(RPN, self).__init__()
        self.conv_shared = nn.Conv3d(channel, conv_channel, kernel_size=3, stride=anchor_stride, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv_class = nn.Conv3d(conv_channel, 2 * anchors_per_location, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=2)
        self.conv_bbox = nn.Conv3d(conv_channel, 6 * anchors_per_location, kernel_size=1, stride=1)

    def forward(self, x):
        """
        Perform forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        # Shared convolutional base of the RPN
        x = self.relu(self.conv_shared(x))

        # Anchor Score. [batch, anchors per location * 2, depth, height, width].
        rpn_class_logits = self.conv_class(x)

        # Reshape to [batch, anchors, 2]
        rpn_class_logits = rpn_class_logits.permute(0, 2, 3, 4, 1)
        rpn_class_logits = rpn_class_logits.contiguous()
        rpn_class_logits = rpn_class_logits.view(x.size()[0], -1, 2)

        # Softmax on last dimension of BG/FG.
        rpn_probs = self.softmax(rpn_class_logits)

        # Bounding box refinement. [batch, anchors per location * 6, D, H, W]
        # where 6 == delta [z, y, x, log(d), log(h), log(w)]
        rpn_bbox = self.conv_bbox(x)

        # Reshape to [batch, anchors, 6]
        rpn_bbox = rpn_bbox.permute(0, 2, 3, 4, 1)
        rpn_bbox = rpn_bbox.contiguous()
        rpn_bbox = rpn_bbox.view(x.size()[0], -1, 6)

        return [rpn_class_logits, rpn_probs, rpn_bbox]


############################################################
#  Feature Pyramid Network Heads
############################################################

class Classifier(nn.Module):

    def __init__(self, channel, pool_size, image_shape, num_classes, fc_size, test_flag=False):
        """
        Initialize a batch

        Args:
            self: (todo): write your description
            channel: (todo): write your description
            pool_size: (int): write your description
            image_shape: (str): write your description
            num_classes: (int): write your description
            fc_size: (int): write your description
            test_flag: (todo): write your description
        """
        super(Classifier, self).__init__()
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.fc_size = fc_size
        self.test_flag = test_flag

        self.conv1 = nn.Conv3d(channel, fc_size, kernel_size=self.pool_size, stride=1)
        self.bn1 = nn.BatchNorm3d(fc_size, eps=0.001, momentum=0.01)
        self.conv2 = nn.Conv3d(fc_size, fc_size, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm3d(fc_size, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

        self.linear_class = nn.Linear(fc_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.linear_bbox = nn.Linear(fc_size, num_classes * 6)

    def forward(self, x, rois):
        """
        Forward - off of a - shot.

        Args:
            self: (todo): write your description
            x: (todo): write your description
            rois: (todo): write your description
        """
        x = pyramid_roi_align([rois] + x, self.pool_size, self.test_flag)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = x.view(-1, self.fc_size)
        mrcnn_class_logits = self.linear_class(x)
        mrcnn_probs = self.softmax(mrcnn_class_logits)

        mrcnn_bbox = self.linear_bbox(x)
        mrcnn_bbox = mrcnn_bbox.view(mrcnn_bbox.size()[0], -1, 6)

        return [mrcnn_class_logits, mrcnn_probs, mrcnn_bbox]


class Mask(nn.Module):

    def __init__(self, channel, pool_size, num_classes, conv_channel, stage, test_flag=False):
        """
        Initialize channel

        Args:
            self: (todo): write your description
            channel: (todo): write your description
            pool_size: (int): write your description
            num_classes: (int): write your description
            conv_channel: (todo): write your description
            stage: (str): write your description
            test_flag: (todo): write your description
        """
        super(Mask, self).__init__()
        self.pool_size = pool_size
        self.test_flag = test_flag

        self.modified_u_net = mask_branch.Modified3DUNet(channel, num_classes, stage, conv_channel)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, rois):
        """
        Perform function.

        Args:
            self: (todo): write your description
            x: (todo): write your description
            rois: (todo): write your description
        """
        x = pyramid_roi_align([rois] + x, self.pool_size, self.test_flag)
        x = self.modified_u_net(x)
        output = self.softmax(x)

        return x, output


############################################################
#  Loss Functions
############################################################

def compute_rpn_class_loss(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
    """
    # Squeeze last dim to simplify
    rpn_match = rpn_match.squeeze(2)

    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = (rpn_match == 1).long()

    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = torch.nonzero(rpn_match != 0)

    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = rpn_class_logits[indices.detach()[:, 0], indices.detach()[:, 1], :]
    anchor_class = anchor_class[indices.detach()[:, 0], indices.detach()[:, 1]]
    # print("rpn size:", rpn_class_logits.shape)

    # Cross-entropy loss
    loss = F.cross_entropy(rpn_class_logits, anchor_class)

    return loss


def compute_rpn_bbox_loss(target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss graph.
    target_bbox: [batch, max positive anchors, (dz, dy, dx, log(dd), log(dh), log(dw))].
        Uses 0 padding to fill in unused bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dz, dy, dx, log(dd), log(dh), log(dw))]
    """
    # Squeeze last dim to simplify
    rpn_match = rpn_match.squeeze(2)

    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    indices = torch.nonzero(rpn_match == 1)

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = rpn_bbox[indices.detach()[:, 0], indices.detach()[:, 1]]

    # Trim target bounding box deltas to the same length as rpn_bbox.
    target_bbox = target_bbox[0, :rpn_bbox.size()[0], :]

    # Smooth L1 loss
    loss = F.smooth_l1_loss(rpn_bbox, target_bbox)

    return loss


def compute_mrcnn_class_loss(target_class_ids, pred_class_logits):
    """Loss for the classifier head of Mask RCNN.
    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    """
    # Loss
    if target_class_ids.size()[0] != 0:
        loss = F.cross_entropy(pred_class_logits, target_class_ids.long())
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss


def compute_mrcnn_bbox_loss(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.
    target_bbox: [batch, num_rois, (dz, dy, dx, log(dd), log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dz, dy, dx, log(dd), log(dh), log(dw))]
    """
    if target_class_ids.size()[0] != 0:
        # Only positive ROIs contribute to the loss. And only
        # the right class_id of each ROI. Get their indices.
        positive_roi_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_roi_class_ids = target_class_ids[positive_roi_ix.detach()].long()
        indices = torch.stack((positive_roi_ix, positive_roi_class_ids), dim=1)

        # Gather the deltas (predicted and true) that contribute to loss
        target_bbox = target_bbox[indices[:, 0].detach(), :]
        pred_bbox = pred_bbox[indices[:, 0].detach(), indices[:, 1].detach(), :]

        # Smooth L1 loss
        loss = F.smooth_l1_loss(pred_bbox, target_bbox)
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss


def compute_mrcnn_mask_loss(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.
    target_masks: [batch, num_rois, depth, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, num_classes, depth, height, width] float32 tensor
                with values from 0 to 1.
    """
    if target_class_ids.size()[0] != 0:
        # Only positive ROIs contribute to the loss. And only the class specific mask of each ROI.
        positive_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_class_ids = target_class_ids[positive_ix.detach()].long()
        indices = torch.stack((positive_ix, positive_class_ids), dim=1)
        # Gather the masks (predicted and true) that contribute to loss
        y_true_ = target_masks[indices[:, 0], :, :, :]
        y_true = y_true_.long().cuda()
        y_true = torch.argmax(y_true, dim=1)
        y_pred = pred_masks[indices[:, 0].detach(), :, :, :, :]
        # Binary cross entropy
        loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1., 1., 100.]).cuda()).cuda()
        loss = loss_fn(y_pred, y_true)
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss


def compute_mrcnn_mask_edge_loss(target_masks, target_class_ids, pred_masks):
    """Mask edge mean square error loss for the Edge Agreement Head.
    Here I use the Sobel kernel without smoothing the ground_truth masks.
        target_masks: [batch, num_rois, depth, height, width].
        target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
        pred_masks: [batch, proposals, num_classes, depth, height, width] float32 tensor with values from 0 to 1.
    """
    if target_class_ids.size()[0] != 0:
        # Generate the xyz dimension Sobel kernels
        kernel_x = np.array([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                             [[2, 4, 2], [0, 0, 0], [-2, -4, -2]],
                             [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]])
        kernel_y = kernel_x.transpose((1, 0, 2))
        kernel_z = kernel_x.transpose((0, 2, 1))
        kernel = torch.from_numpy(np.array([kernel_x, kernel_y, kernel_z]).reshape((3, 1, 3, 3, 3))).float().cuda()
        # Only positive ROIs contribute to the loss. And only the class specific mask of each ROI.
        positive_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_class_ids = target_class_ids[positive_ix.detach()].long()
        indices = torch.stack((positive_ix, positive_class_ids), dim=1)
        # Gather the masks (predicted and true) that contribute to loss
        y_true = target_masks[:indices.size()[0], 1:, :, :]
        y_pred = pred_masks[indices[:, 0].detach(), 1:, :, :, :]
        # Implement the edge detection convolution
        loss_fn = nn.MSELoss()
        loss = torch.FloatTensor([0]).cuda()
        for i in range(indices.size()[0]):  # only compute the tumor's edge
            y_true_ = y_true[i]
            y_pred_ = y_pred[i].unsqueeze(0)  # [N, 2, 64, 64, 64]
            for j in range(y_true_.shape[0]):
                y_true_final = F.conv3d(y_true_[j, :, :, :].unsqueeze(0).unsqueeze(0).cuda().float(), kernel)
                y_pred_final = F.conv3d(y_pred_[:, j, :, :, :].unsqueeze(1), kernel)
                # y_true_final = torch.sqrt(torch.pow(y_true_final[:, 0], 2) + torch.pow(y_true_final[:, 1], 2) +
                #                            torch.pow(y_true_final[:, 0], 2))
                # y_pred_final = torch.sqrt(torch.pow(y_pred_final [:, 0], 2) + torch.pow(y_pred_final [:, 1], 2) +
                #                           torch.pow(y_pred_final [:, 0], 2))
                # Mean Square Error
                loss += loss_fn(y_pred_final, y_true_final)
            # import pdb
            # pdb.set_trace()
        loss /= indices.size()[0]
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False).cuda()

    return loss


def compute_losses(rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits,
                   target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, mrcnn_mask_logits, stage):
    """
    Compute rpn loss.

    Args:
        rpn_match: (todo): write your description
        rpn_bbox: (todo): write your description
        rpn_class_logits: (todo): write your description
        rpn_pred_bbox: (todo): write your description
        target_class_ids: (str): write your description
        mrcnn_class_logits: (todo): write your description
        target_deltas: (todo): write your description
        mrcnn_bbox: (todo): write your description
        target_mask: (todo): write your description
        mrcnn_mask: (int): write your description
        mrcnn_mask_logits: (todo): write your description
        stage: (str): write your description
    """

    if stage == "beginning":
        rpn_class_loss = compute_rpn_class_loss(rpn_match, rpn_class_logits)
        rpn_bbox_loss = compute_rpn_bbox_loss(rpn_bbox, rpn_match, rpn_pred_bbox)
        mrcnn_class_loss = compute_mrcnn_class_loss(torch.from_numpy(np.where(target_class_ids > 0, 1, 0)).cuda(),
                                                    mrcnn_class_logits)
        mrcnn_bbox_loss = compute_mrcnn_bbox_loss(target_deltas,
                                                  torch.from_numpy(np.where(target_class_ids > 0, 1, 0)).cuda(),
                                                  mrcnn_bbox)
        mrcnn_mask_loss = Variable(torch.FloatTensor([0]), requires_grad=False).cuda()
        mrcnn_mask_edge_loss = Variable(torch.FloatTensor([0]), requires_grad=False).cuda()
    else:
        rpn_class_loss = Variable(torch.FloatTensor([0]), requires_grad=False).cuda()
        rpn_bbox_loss = Variable(torch.FloatTensor([0]), requires_grad=False).cuda()
        mrcnn_class_loss = Variable(torch.FloatTensor([0]), requires_grad=False).cuda()
        mrcnn_bbox_loss = Variable(torch.FloatTensor([0]), requires_grad=False).cuda()
        mrcnn_mask_loss = compute_mrcnn_mask_loss(target_mask, target_class_ids, mrcnn_mask_logits)
        mrcnn_mask_edge_loss = compute_mrcnn_mask_edge_loss(target_mask, target_class_ids, mrcnn_mask)

    return [rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss, mrcnn_mask_edge_loss]


############################################################
#  Data Generator
############################################################

def load_image_gt(mask, config, anchors):
    """Generate the ground truth data for a mask.
    mask: [D, H, W]
    Returns:
    image: [1, D, H, W]
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (z1, y1, x1, z2, y2, x2)]
    mask: [num_classes, D, H, W]
    rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
    rpn_bbox: [batch, N, (dz, dy, dx, log(dd), log(dh), log(dw))] Anchor bbox deltas
    """
    # Bounding boxes: [num_instances, (z1, y1, x1, z2, y2, x2)]
    bbox = utils.extract_bboxes(mask)  # we here use the whole liver + tumor as the gt-bbox
    bbox = utils.extend_bbox(bbox, mask.shape)  # extend the gt_bbox with 5% ratio in each dimension
    bbox = np.tile(bbox, (config.NUM_CLASSES - 1, 1))  # [num_classes - 1, (z1, y1, x1, z2, y2, x2)]

    # RPN Targets
    rpn_match, rpn_bbox = build_rpn_targets(anchors, np.array([bbox[0]]), config)

    # Add to batch
    rpn_match = rpn_match[:, np.newaxis]

    return rpn_match, rpn_bbox, bbox


def build_rpn_targets(anchors, gt_boxes, config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.
    anchors: [num_anchors, (z1, y1, x1, z2, y2, x2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (z1, y1, x1, z2, y2, x2)]
    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dz, dy, dx, log(dd), log(dh), log(dw))] Anchor bbox deltas.
    """
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # RPN bounding boxes: [max anchors per image, (dz, dy, dx, log(dd), log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 6))
    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = utils.compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above, and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens).
    # Instead, match it to the closest anchor (even if its max IoU is < 0.3).

    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[anchor_iou_max < 0.3] = -1

    # 2. Set an anchor for each GT box (regardless of IoU value).
    gt_iou_argmax = np.argmax(overlaps, axis=0)
    rpn_match[gt_iou_argmax] = 1

    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]
    ix = 0  # index into rpn_bbox
    for i, a in zip(ids, anchors[ids]):
        # Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]

        # Convert coordinates to center plus width/height.
        # GT Box
        gt_d = gt[3] - gt[0]
        gt_h = gt[4] - gt[1]
        gt_w = gt[5] - gt[2]
        gt_center_z = gt[0] + 0.5 * gt_d
        gt_center_y = gt[1] + 0.5 * gt_h
        gt_center_x = gt[2] + 0.5 * gt_w
        # Anchor
        a_d = a[3] - a[0]
        a_h = a[4] - a[1]
        a_w = a[5] - a[2]
        a_center_z = a[0] + 0.5 * a_d
        a_center_y = a[1] + 0.5 * a_h
        a_center_x = a[2] + 0.5 * a_w

        # Compute the bbox refinement that the RPN should predict.
        rpn_bbox[ix] = [
            (gt_center_z - a_center_z) / a_d,
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_d / a_d),
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        # Normalize
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1

    return rpn_match, rpn_bbox


class Dataset(torch.utils.data.Dataset):

    def __init__(self, dataset, config, augmentation=False):
        """A data_generator that returns the image, mask and other ground truth for training."""
        self.image_ids = np.copy(dataset.image_ids)

        self.dataset = dataset
        self.config = config
        self.augmentation = augmentation

        # Anchors
        # [anchor_count, (z1, y1, x1, z2, y2, x2)]
        self.anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES, config.RPN_ANCHOR_RATIOS,
                                                      compute_backbone_shapes(config, config.IMAGE_SHAPE),
                                                      config.BACKBONE_STRIDES, config.RPN_ANCHOR_STRIDE)

    def __getitem__(self, image_index):
        image_id = self.image_ids[image_index]
        # Load image, which is [H, W, D] first.
        image = self.dataset.load_image(image_id)
        image = preprocess_image(image)
        # Load mask, which is [H, W, D] first.
        mask = self.dataset.load_mask(image_id)  # np.int32
        # Apply some augmentation
        image_shape = image.shape
        assert image_shape == mask.shape
        whole_image = np.zeros(self.config.PAD_IMAGE_SHAPE)
        whole_mask = np.zeros(self.config.PAD_IMAGE_SHAPE)
        start_x = int((whole_image.shape[0] - image_shape[0]) / 2.)
        start_y = int((whole_image.shape[1] - image_shape[1]) / 2.)
        start_z = int((whole_image.shape[2] - image_shape[2]) / 2.)
        whole_image[start_x:start_x + image_shape[0], start_y:start_y + image_shape[1], start_z:start_z + image_shape[2]] = image
        whole_mask[start_x:start_x + image_shape[0], start_y:start_y + image_shape[1], start_z:start_z + image_shape[2]] = mask
        image = whole_image
        mask = whole_mask
        del whole_image
        del whole_mask
        if self.augmentation:  # apply image augmentation
            # randomly crop or pad the image
            """
            choice = float(torch.rand(1))
            if choice < 0.5:  # crop the image
                box = utils.extract_bboxes(mask).astype(np.float32)
                min_x = box[0] / image_shape[0]
                max_x = (image_shape[0] - box[3]) / image_shape[0]
                min_y = box[1] / image_shape[1]
                max_y = (image_shape[1] - box[4]) / image_shape[1]
                min_z = box[2] / image_shape[2]
                max_z = (image_shape[2] - box[5]) / image_shape[2]
                ratio = min(min_x, max_x, min_y, max_y, min_z, max_z, self.config.CROP_PAD_RATIO)
                ratio = ratio * float(torch.rand(1))
                dx = int(image_shape[0] * ratio)
                dy = int(image_shape[1] * ratio)
                dz = int(image_shape[2] * ratio)
                dx_1 = int(torch.randint(0, dx + 1, (1,)))
                dx_2 = dx - dx_1
                dy_1 = int(torch.randint(0, dy + 1, (1,)))
                dy_2 = dy - dy_1
                dz_1 = int(torch.randint(0, dz + 1, (1,)))
                dz_2 = dz - dz_1
                image = image[dx_1:image_shape[0] - dx_2, dy_1:image_shape[1] - dy_2, dz_1:image_shape[2] - dz_2]
                mask = mask[dx_1:image_shape[0] - dx_2, dy_1:image_shape[1] - dy_2, dz_1:image_shape[2] - dz_2]
            else:  # pad the image
                ratio = self.config.CROP_PAD_RATIO * float(torch.rand(1))
                dx = int(image_shape[0] * ratio)
                dy = int(image_shape[1] * ratio)
                dz = int(image_shape[2] * ratio)
                dx_1 = int(torch.randint(0, dx + 1, (1,)))
                dx_2 = dx - dx_1
                dy_1 = int(torch.randint(0, dy + 1, (1,)))
                dy_2 = dy - dy_1
                dz_1 = int(torch.randint(0, dz + 1, (1,)))
                dz_2 = dz - dz_1
                small_image = skimage.transform.resize(image, (image_shape[0] - dx, image_shape[1] - dy, image_shape[2] - dz),
                                                       order=0, preserve_range=True, mode='constant', anti_aliasing=False)
                small_mask = skimage.transform.resize(mask, (image_shape[0] - dx, image_shape[1] - dy, image_shape[2] - dz),
                                                      order=0, preserve_range=True, mode='constant', anti_aliasing=False)
                image = np.zeros(image_shape)
                image[dx_1:image_shape[0] - dx_2, dy_1:image_shape[1] - dy_2, dz_1:image_shape[2] - dz_2] = small_image
                mask = np.zeros(image_shape, dtype=np.int32)
                mask[dx_1:image_shape[0] - dx_2, dy_1:image_shape[1] - dy_2, dz_1:image_shape[2] - dz_2] = small_mask
            """
            # randomly rotate between -30 to 30 degree

            angle = int(torch.randint(self.config.ROTATE_ANGLE[0], self.config.ROTATE_ANGLE[1], (1,)))
            image = skimage.transform.rotate(image, angle=angle, order=0, mode='constant', preserve_range=True)  # [H, W, D]
            mask = skimage.transform.rotate(mask, angle=angle, order=0, mode='constant', preserve_range=True)  # [H, W, D]

            # resize the image and mask to standard input shape
            image = skimage.transform.resize(image, self.config.IMAGE_SHAPE[:3], order=0, preserve_range=True, mode='constant',
                                             anti_aliasing=False)
            mask = skimage.transform.resize(mask, self.config.IMAGE_SHAPE[:3], order=0, preserve_range=True, mode='constant',
                                            anti_aliasing=False)
        else:
            image = skimage.transform.resize(image, self.config.IMAGE_SHAPE[:3], order=0, preserve_range=True,
                                             mode='constant', anti_aliasing=False)
            mask = skimage.transform.resize(mask, self.config.IMAGE_SHAPE[:3], order=0, preserve_range=True,
                                            mode='constant', anti_aliasing=False)
        # Note that window has already been (z1, y1, x1, z2, y2, x2) here.
        window = (start_z * self.config.IMAGE_SHAPE[2] / self.config.PAD_IMAGE_SHAPE[2],
                  start_x * self.config.IMAGE_SHAPE[0] / self.config.PAD_IMAGE_SHAPE[0],
                  start_y * self.config.IMAGE_SHAPE[1] / self.config.PAD_IMAGE_SHAPE[1],
                  self.config.IMAGE_MIN_DIM - start_z * self.config.IMAGE_SHAPE[2] / self.config.PAD_IMAGE_SHAPE[2],
                  self.config.IMAGE_MAX_DIM - start_x * self.config.IMAGE_SHAPE[0] / self.config.PAD_IMAGE_SHAPE[0],
                  self.config.IMAGE_MAX_DIM - start_y * self.config.IMAGE_SHAPE[1] / self.config.PAD_IMAGE_SHAPE[1])
        # Active classes
        # Different datasets have different classes, so track the classes supported in the dataset of this image.
        active_class_ids = np.zeros([self.dataset.num_classes], dtype=np.int32)
        source_class_ids = self.dataset.source_class_ids[self.dataset.image_info[image_id]["source"]]
        active_class_ids[source_class_ids] = 1
        # Image meta data
        image_meta = compose_image_meta(image_id, image.shape, window, active_class_ids)
        # Generate different ground truth for the image
        image = np.expand_dims(image.transpose((2, 0, 1)), axis=0)  # [C, D, H, W]
        mask = mask.transpose((2, 0, 1))  # [D, H, W]
        rpn_match, rpn_bbox, bbox = load_image_gt(mask, self.config, self.anchors)
        # Get the instance-specific masks and class_ids.
        masks, class_ids = self.dataset.process_mask(mask)

        return image, image_meta, rpn_match, rpn_bbox, class_ids, bbox, masks

    def __len__(self):
        """
        Return the length of the image.

        Args:
            self: (todo): write your description
        """

        return self.image_ids.shape[0]


############################################################
#  MaskRCNN Class
############################################################

class MaskRCNN(nn.Module):
    """Encapsulates the 3D-Mask-RCNN model functionality."""

    def __init__(self, config, model_dir, test_flag=False):
        """config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        super(MaskRCNN, self).__init__()
        self.epoch = 0
        self.config = config
        self.model_dir = model_dir
        self.build(config=config, test_flag=test_flag)
        self.initialize_weights()

    def build(self, config, test_flag=False):
        """Build 3D-Mask-RCNN architecture."""

        # Image size must be dividable by 2 multiple times
        h, w, d = config.IMAGE_SHAPE[:3]
        if h / 16 != int(h / 16) or w / 16 != int(w / 16) or d / 16 != int(d / 16):
            raise Exception("Image size must be dividable by 16. Use 256, 320, 512, ... etc.")

        # Build the shared convolutional layers.
        # Returns a list of the last layers of each stage, 3 in total.
        if self.config.BACKBONE == "P3D19":
            print("using P3D19 as backbone")
            P3D_Resnet = backbone.P3D19(config=config)
        else:
            print("using P3D35 as backbone")
            P3D_Resnet = backbone.P3D35(config=config)
        C1, C2, C3 = P3D_Resnet.stages()

        # Top-down Layers
        self.fpn = FPN(C1, C2, C3, out_channels=config.TOP_DOWN_PYRAMID_SIZE, config=config)

        # Generate Anchors
        self.anchors = Variable(torch.from_numpy(utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                                                config.RPN_ANCHOR_RATIOS,
                                                                                compute_backbone_shapes(
                                                                                    config, config.IMAGE_SHAPE),
                                                                                config.BACKBONE_STRIDES,
                                                                                config.RPN_ANCHOR_STRIDE)).float(),
                                requires_grad=False)
        if self.config.GPU_COUNT:
            self.anchors = self.anchors.cuda()

        # RPN
        self.rpn = RPN(len(config.RPN_ANCHOR_RATIOS), config.RPN_ANCHOR_STRIDE, config.TOP_DOWN_PYRAMID_SIZE,
                       config.RPN_CONV_CHANNELS)

        if self.config.STAGE != 'beginning':
            for p in self.parameters():
                p.requires_grad = False

        # FPN Classifier
        self.classifier = Classifier(config.TOP_DOWN_PYRAMID_SIZE, config.POOL_SIZE, config.IMAGE_SHAPE,
                                     2, config.FPN_CLASSIFY_FC_LAYERS_SIZE, test_flag)

        # FPN Mask
        self.mask = Mask(1, config.MASK_POOL_SIZE, config.NUM_CLASSES, config.UNET_MASK_BRANCH_CHANNEL,
                         self.config.STAGE, test_flag)

        if test_flag:
            for p in self.parameters():
                p.requires_grad = False

        # Fix batch norm layers
        def set_bn_fix(m):
            """
            Sets the gradient of a class.

            Args:
                m: (todo): write your description
            """
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters():
                    p.requires_grad = False

        if not config.TRAIN_BN:
            self.apply(set_bn_fix)

    def initialize_weights(self):
        """Initialize model weights."""

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def set_trainable(self, layer_regex):
        """Sets model layers as trainable if their names match the given regular expression."""
        for param in self.named_parameters():
            layer_name = param[0]
            trainable = bool(re.fullmatch(layer_regex, layer_name))
            if not trainable:
                param[1].requires_grad = False

    def load_weights(self, file_path):
        """Modified version of the corresponding Keras function with the addition of multi-GPU support
        and the ability to exclude some layers from loading.
        exclude: list of layer names to exclude
        """
        if os.path.exists(file_path):
            pretrained_dict = torch.load(file_path)
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=True)
            print("Pre-trained weights load success!")
        else:
            print("Weight file not found ...")

    def detect(self, images):
        """Runs the detection pipeline.
        images: List of images, potentially of different sizes. [1, height, width, depth]
        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, z1, y2, x2, z2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, D, N] instance binary masks
        Transform all outputs from pytorch shape to normal shape here.
        """
        # Mold inputs to format expected by the neural network
        # Has been transformed to pytorch shapes.
        start_time = time.time()
        molded_images, image_metas, windows = self.mold_inputs(images)

        # Convert images to torch tensor
        molded_images = torch.from_numpy(molded_images).float()

        # To GPU
        if self.config.GPU_COUNT:
            molded_images = molded_images.cuda()

        # Run object detection
        detections, mrcnn_mask = self.predict([molded_images, image_metas], mode='inference')
        # Convert to numpy
        detections = detections.detach().cpu().numpy()
        mrcnn_mask = mrcnn_mask.permute(0, 1, 3, 4, 5, 2).detach().cpu().numpy()
        print("detect done, using time", time.time() - start_time)
        # import pdb
        # pdb.set_trace()

        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_mask = \
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       [1, image.shape[2], image.shape[0], image.shape[1]], windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "mask": final_mask,
            })

        return results

    def predict(self, inputs, mode):
        """
        Predict classifier on input mode.

        Args:
            self: (todo): write your description
            inputs: (array): write your description
            mode: (todo): write your description
        """
        molded_images = inputs[0]
        image_metas = inputs[1]

        if mode == 'inference':
            self.eval()
        elif mode == 'training':
            self.train()

            # Set batchnorm always in eval mode during training
            def set_bn_eval(m):
                """
                Sets the __init__ function.

                Args:
                    m: (todo): write your description
                """
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.apply(set_bn_eval)

        # Feature extraction
        p2_out, p3_out = self.fpn(molded_images)

        rpn_feature_maps = [p2_out, p3_out]
        mrcnn_classifier_feature_maps = [p2_out, p3_out]
        mrcnn_mask_feature_maps = [molded_images, molded_images]

        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(self.rpn(p))

        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        outputs = list(zip(*layer_outputs))
        outputs = [torch.cat(list(o), dim=1) for o in outputs]
        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        # Proposals are [batch, N, (z1, y1, x1, z2, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = self.config.POST_NMS_ROIS_TRAINING if mode == "training" \
            else self.config.POST_NMS_ROIS_INFERENCE
        rpn_rois = proposal_layer([rpn_class, rpn_bbox],
                                  proposal_count=proposal_count,
                                  nms_threshold=self.config.RPN_NMS_THRESHOLD,
                                  anchors=self.anchors,
                                  config=self.config)
        if mode == 'inference':
            # Network Heads
            # Proposal classifier and bbox regressor heads
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(mrcnn_classifier_feature_maps, rpn_rois)

            # Detections
            # output is [batch, num_detections, (z1, y1, x1, z2, y2, x2, class_id, score)] in image coordinates
            detections = detection_layer(self.config, rpn_rois, mrcnn_class, mrcnn_bbox, image_metas)

            # Convert boxes to normalized coordinates
            h, w, d = self.config.IMAGE_SHAPE[:3]
            scale = torch.from_numpy(np.array([d, h, w, d, h, w])).float()
            if self.config.GPU_COUNT:
                scale = scale.cuda()
            detection_boxes = detections[:, :6] / scale

            # Add back batch dimension
            detection_boxes = detection_boxes.unsqueeze(0)

            # Create masks for detections
            if self.config.STAGE != 'beginning':
                _, mrcnn_mask = self.mask(mrcnn_mask_feature_maps, detection_boxes)
                mrcnn_mask = mrcnn_mask.unsqueeze(0)
            else:
                mrcnn_mask = torch.zeros((1, detection_boxes.shape[1], 3,) + self.config.MINI_MASK_SHAPE).cuda()

            # Add back batch dimension
            detections = detections.unsqueeze(0)

            return [detections, mrcnn_mask]

        elif mode == 'training':
            gt_class_ids = inputs[2]  # [1, 2, ..., num_classes - 1]
            gt_boxes = inputs[3]  # [num_classes - 1, (z1, y1, x1, z2, y2, x2)]
            gt_masks = inputs[4]  # multi_classes masks [num_classes, D, H, W]

            # Normalize coordinates
            h, w, d = self.config.IMAGE_SHAPE[:3]
            scale = torch.from_numpy(np.array([d, h, w, d, h, w])).float()
            if self.config.GPU_COUNT:
                scale = scale.cuda()
            gt_boxes = gt_boxes / scale

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # import pdb
            # pdb.set_trace()
            p_rois, rois, target_class_ids, target_deltas, target_mask = \
                detection_target_layer(rpn_rois, gt_class_ids, gt_boxes, gt_masks, self.config)

            # print("rois size:", rois.shape, "p_rois size:", p_rois.shape)

            if rois.size()[0] == 0:
                mrcnn_class_logits = Variable(torch.FloatTensor())
                mrcnn_class = Variable(torch.IntTensor())
                mrcnn_bbox = Variable(torch.FloatTensor())
                mrcnn_mask = Variable(torch.FloatTensor())
                mrcnn_mask_logits = Variable(torch.FloatTensor())
                if self.config.GPU_COUNT:
                    mrcnn_class_logits = mrcnn_class_logits.cuda()
                    mrcnn_class = mrcnn_class.cuda()
                    mrcnn_bbox = mrcnn_bbox.cuda()
                    mrcnn_mask = mrcnn_mask.cuda()
                    mrcnn_mask_logits = mrcnn_mask_logits.cuda()
            elif p_rois.size()[0] == 0 or self.config.STAGE == 'beginning':
                # Network Heads
                # Proposal classifier and BBox regressor heads
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(mrcnn_classifier_feature_maps, rois)
                mrcnn_mask = Variable(torch.FloatTensor())
                mrcnn_mask_logits = Variable(torch.FloatTensor())
                if self.config.GPU_COUNT:
                    mrcnn_mask = mrcnn_mask.cuda()
                    mrcnn_mask_logits = mrcnn_mask_logits.cuda()
            else:  # only train the mask branch
                # Network Heads
                # Proposal classifier and BBox regressor heads
                mrcnn_class_logits = Variable(torch.FloatTensor())
                mrcnn_class = Variable(torch.IntTensor())
                mrcnn_bbox = Variable(torch.FloatTensor())
                if self.config.GPU_COUNT:
                    mrcnn_class_logits = mrcnn_class_logits.cuda()
                    mrcnn_class = mrcnn_class.cuda()
                    mrcnn_bbox = mrcnn_bbox.cuda()

                # Create masks for detections
                mrcnn_mask_logits, mrcnn_mask = self.mask(mrcnn_mask_feature_maps, p_rois)

            return [rpn_class_logits, rpn_bbox,
                    target_class_ids, mrcnn_class_logits,
                    target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, mrcnn_mask_logits]

    def train_model(self, train_dataset, val_dataset, learning_rate, epochs):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done already, so this actually determines
                the epochs to train in total rather than in this particular call.
        """
        layers = ".*"  # set all the layers trainable

        # Data generators
        train_set = Dataset(train_dataset, self.config, augmentation=self.config.AUGMENTATION)
        train_generator = torch.utils.data.DataLoader(train_set, batch_size=self.config.BATCH_SIZE,
                                                      shuffle=self.config.SHUFFLE_DATASET,
                                                      num_workers=self.config.TRAIN_NUM_WORKERS)
        val_set = Dataset(val_dataset, self.config, augmentation=False)
        val_generator = torch.utils.data.DataLoader(val_set, batch_size=self.config.BATCH_SIZE,
                                                    shuffle=self.config.SHUFFLE_DATASET,
                                                    num_workers=self.config.VAL_NUM_WORKERS)

        # Train
        self.set_trainable(layers)

        # Optimizer object
        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        trainables_wo_bn = [param for name, param in self.named_parameters()
                            if param.requires_grad and 'bn' not in name]
        trainables_only_bn = [param for name, param in self.named_parameters()
                              if param.requires_grad and 'bn' in name]
        optimizer = optim.SGD([
            {'params': trainables_wo_bn, 'weight_decay': self.config.WEIGHT_DECAY},
            {'params': trainables_only_bn}
        ], lr=learning_rate, momentum=self.config.LEARNING_MOMENTUM)

        total_start_time = time.time()
        start_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        if not os.path.exists("./logs/" + str(start_datetime)):
            os.makedirs("./logs/" + str(start_datetime))
        for epoch in range(self.epoch + 1, epochs + 1):
            log("Epoch {}/{}.".format(epoch, epochs))
            start_time = time.time()
            # Training
            loss, loss_rpn_class, loss_rpn_bbox, \
                loss_mrcnn_class, loss_mrcnn_bbox, loss_mrcnn_mask, loss_mrcnn_mask_edge = \
                self.one_epoch(train_generator, optimizer, self.config.STEPS_PER_EPOCH)

            print("One Training Epoch time:", int(time.time() - start_time),
                  "Total time:", int(time.time() - total_start_time))

            torch.cuda.empty_cache()

            if epoch % self.config.SAVE_EPOCH == 0:
                # Validation
                val_loss, val_loss_rpn_class, val_loss_rpn_bbox, \
                    val_loss_mrcnn_class, val_loss_mrcnn_bbox, val_loss_mrcnn_mask, val_loss_mrcnn_mask_edge = \
                    self.one_epoch(val_generator, None, self.config.VALIDATION_STEPS)

                torch.save(self.state_dict(), "./logs/" + str(start_datetime) + "/model" + str(epoch) +
                           "_loss: " + str(round(loss, 4)) + "_val: " + str(round(val_loss, 4)))

                torch.cuda.empty_cache()

        self.epoch = epochs

    def one_epoch(self, datagenerator, optimizer, steps):
        """
        Perform one epoch.

        Args:
            self: (todo): write your description
            datagenerator: (str): write your description
            optimizer: (todo): write your description
            steps: (int): write your description
        """
        batch_count = 0
        loss_sum = 0
        loss_rpn_class_sum = 0
        loss_rpn_bbox_sum = 0
        loss_mrcnn_class_sum = 0
        loss_mrcnn_bbox_sum = 0
        loss_mrcnn_mask_sum = 0
        loss_mrcnn_mask_edge_sum = 0
        step = 0

        if optimizer is not None:
            optimizer.zero_grad()

        for inputs in datagenerator:
            batch_count += 1

            images = inputs[0]  # [batch, C, D, H, W]
            image_metas = inputs[1]
            rpn_match = inputs[2]
            rpn_bbox = inputs[3]
            gt_class_ids = inputs[4]
            gt_boxes = inputs[5]  # [batch, num_classes - 1, (z1, y1, x1, z2, y2, x2)]
            gt_masks = inputs[6]  # [batch, num_classes, D, H, W]

            # image_metas as numpy array
            image_metas = image_metas.numpy()

            # To GPU
            if self.config.GPU_COUNT:
                images = images.cuda().float()
                rpn_match = rpn_match.cuda()
                rpn_bbox = rpn_bbox.cuda().float()
                gt_class_ids = gt_class_ids.cuda()
                gt_boxes = gt_boxes.cuda().float()
                gt_masks = gt_masks.cuda().float()

            # import pdb
            # pdb.set_trace()

            # Run object detection
            rpn_class_logits, rpn_pred_bbox, target_class_ids, \
                mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, mrcnn_mask_logits = \
                self.predict([images, image_metas, gt_class_ids, gt_boxes, gt_masks], mode='training')

            # Compute losses
            rpn_class_loss, rpn_bbox_loss, \
                mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss, mrcnn_mask_edge_loss = \
                compute_losses(rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids,
                               mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, mrcnn_mask_logits, self.config.STAGE)

            loss = self.config.LOSS_WEIGHTS["rpn_class_loss"] * rpn_class_loss + \
                self.config.LOSS_WEIGHTS["rpn_bbox_loss"] * rpn_bbox_loss + \
                self.config.LOSS_WEIGHTS["mrcnn_class_loss"] * mrcnn_class_loss + \
                self.config.LOSS_WEIGHTS["mrcnn_bbox_loss"] * mrcnn_bbox_loss + \
                self.config.LOSS_WEIGHTS["mrcnn_mask_loss"] * mrcnn_mask_loss + \
                self.config.LOSS_WEIGHTS["mrcnn_mask_edge_loss"] * mrcnn_mask_edge_loss

            # Back propagation
            if optimizer is not None:
                try:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                    if (batch_count % self.config.BATCH_SIZE) == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        batch_count = 0
                except:
                    # print("backward error! loss is", loss)
                    optimizer.zero_grad()

            # Progress
            print_progress_bar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                               suffix="Complete - loss: {:.5f} - rpn_class_loss: {:.5f} - rpn_bbox_loss: {:.5f}"
                                      " - mrcnn_class_loss: {:.5f} - mrcnn_bbox_loss: {:.5f} - mrcnn_mask_loss: {:.5f}"
                                      " - mrcnn_mask_edge_loss: {:.5f}"
                               .format(loss.detach().cpu().item(),
                                       self.config.LOSS_WEIGHTS["rpn_class_loss"] * rpn_class_loss.detach().cpu().item(),
                                       self.config.LOSS_WEIGHTS["rpn_bbox_loss"] * rpn_bbox_loss.detach().cpu().item(),
                                       self.config.LOSS_WEIGHTS["mrcnn_class_loss"] * mrcnn_class_loss.detach().cpu().item(),
                                       self.config.LOSS_WEIGHTS["mrcnn_bbox_loss"] * mrcnn_bbox_loss.detach().cpu().item(),
                                       self.config.LOSS_WEIGHTS["mrcnn_mask_loss"] * mrcnn_mask_loss.detach().cpu().item(),
                                       self.config.LOSS_WEIGHTS["mrcnn_mask_edge_loss"] * mrcnn_mask_edge_loss.detach().cpu().item()),
                               length=50)

            # Statistics
            loss_sum += loss.detach().cpu().item() / steps
            loss_rpn_class_sum += rpn_class_loss.detach().cpu().item() / steps
            loss_rpn_bbox_sum += rpn_bbox_loss.detach().cpu().item() / steps
            loss_mrcnn_class_sum += mrcnn_class_loss.detach().cpu().item() / steps
            loss_mrcnn_bbox_sum += mrcnn_bbox_loss.detach().cpu().item() / steps
            loss_mrcnn_mask_sum += mrcnn_mask_loss.detach().cpu().item() / steps
            loss_mrcnn_mask_edge_sum += mrcnn_mask_edge_loss.detach().cpu().item() / steps

            # Break after 'steps' steps
            if step == steps - 1:
                break
            step += 1

            if step % 20 == 0:
                torch.cuda.empty_cache()

        return loss_sum, loss_rpn_class_sum, loss_rpn_bbox_sum, \
            loss_mrcnn_class_sum, loss_mrcnn_bbox_sum, loss_mrcnn_mask_sum, loss_mrcnn_mask_edge_sum

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matrices [height, width, depth]. Images can have different sizes.
        Returns 3 Numpy matrices:
        molded_images: [N, c, d, h, w]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (z1, y1, x1, z2, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            image = preprocess_image(image)
            image_shape = image.shape
            whole_image = np.zeros(self.config.PAD_IMAGE_SHAPE)
            start_x = int((whole_image.shape[0] - image_shape[0]) / 2.)
            start_y = int((whole_image.shape[1] - image_shape[1]) / 2.)
            start_z = int((whole_image.shape[2] - image_shape[2]) / 2.)
            whole_image[start_x:start_x + image_shape[0], start_y:start_y + image_shape[1],
                        start_z:start_z + image_shape[2]] = image
            image = whole_image
            del whole_image
            image = skimage.transform.resize(image, self.config.IMAGE_SHAPE[:3], order=0, preserve_range=True,
                                             mode='constant', anti_aliasing=False)
            # Note that window has already been (z1, y1, x1, z2, y2, x2) here.
            window = (start_z * self.config.IMAGE_SHAPE[2] / self.config.PAD_IMAGE_SHAPE[2],
                      start_x * self.config.IMAGE_SHAPE[0] / self.config.PAD_IMAGE_SHAPE[0],
                      start_y * self.config.IMAGE_SHAPE[1] / self.config.PAD_IMAGE_SHAPE[1],
                      self.config.IMAGE_MIN_DIM - start_z * self.config.IMAGE_SHAPE[2] / self.config.PAD_IMAGE_SHAPE[2],
                      self.config.IMAGE_MAX_DIM - start_x * self.config.IMAGE_SHAPE[0] / self.config.PAD_IMAGE_SHAPE[0],
                      self.config.IMAGE_MAX_DIM - start_y * self.config.IMAGE_SHAPE[1] / self.config.PAD_IMAGE_SHAPE[1])

            molded_image = np.expand_dims(image.transpose((2, 0, 1)), axis=0)  # [C, D, H, W]
            # Build image_meta
            image_meta = compose_image_meta(
                0, image.shape, window,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)

        return molded_images, image_metas, windows

    def unmold_detections(self, detections, mrcnn_mask, image_shape, window):
        """Reformat the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the application.
        detections: [N, (z1, y1, x1, z2, y2, x2, class_id, score)]
        mrcnn_mask: [N, depth, height, width, num_classes]
        image_shape: [channels, depth, height, width] Original size of the image before resizing
        window: [z1, y1, x1, z2, y2, x2] Box in the image where the real image is excluding the padding.
        Returns:
        boxes: [N, (y1, x1, z1, y2, x2, z2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, depth] normal shape full mask
        """
        # import pdb
        start_time = time.time()
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 6] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :6].astype(np.int32)
        class_ids = detections[:N, 6].astype(np.int32)
        scores = detections[:N, 7]
        masks = mrcnn_mask[np.arange(N), :, :, :, :]

        # Compute scale and shift to translate the bounding boxes to image domain.
        d_scale = image_shape[1] / (window[3] - window[0])
        h_scale = image_shape[2] / (window[4] - window[1])
        w_scale = image_shape[3] / (window[5] - window[2])
        shift = window[:3]  # z, y, x
        scales = np.array([d_scale, h_scale, w_scale, d_scale, h_scale, w_scale])
        shifts = np.array([shift[0], shift[1], shift[2], shift[0], shift[1], shift[2]])
        # pdb.set_trace()
        boxes = np.multiply(boxes - shifts, scales).astype(np.int32)
        # pdb.set_trace()

        # Filter out detections with zero area. Often only happens in early
        # stages of training when the network weights are still a bit random.
        exclude_ix = np.where(
            (boxes[:, 3] - boxes[:, 0]) * (boxes[:, 4] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 2]) <= 0
        )[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)

        # Resize masks to original image size.
        # box: [N, (z1, y1, x1, z2, y2, x2)] in image coordinates
        # mask: [N, depth, height, width, num_instances]
        full_masks = utils.unmold_mask(masks, boxes, image_shape)
        full_mask = np.argmax(full_masks, axis=3)

        # Transform the shapes of boxes to normal shape.
        boxes[:, [0, 1, 2, 3, 4, 5]] = boxes[:, [1, 2, 0, 4, 5, 3]]
        print("unmold done, using time", time.time() - start_time)

        return boxes, np.arange(1, 3), scores, full_mask.transpose((1, 2, 0))


############################################################
#  Data Formatting
############################################################

def compose_image_meta(image_id, image_shape, window, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array. Use
    parse_image_meta() to parse the values back.
    image_id: An int ID of the image. Useful for debugging.
    image_shape: [channels, depth, height, width]
    window: (z1, y1, x1, z2, y2, x2) in pixels. The volume of the image where the real
            image is (excluding the padding)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +            # size = 1
        list(image_shape) +     # size = 3: [H, W, C]
        list(window) +          # size = 6: (z1, y1, x1, z2, y2, x2) in image coordinates
        list(active_class_ids)  # size = num_classes
    )

    return meta


def parse_image_meta(meta):
    """Parses an image info Numpy array to its components.
    See compose_image_meta() for more details.
    """
    image_id = meta[:, 0]
    image_shape = meta[:, 1:4]  # [H, W, D]
    window = meta[:, 4:10]   # (z1, y1, x1, z2, y2, x2) window of image in in pixels
    active_class_ids = meta[:, 10:]

    return image_id, image_shape, window, active_class_ids


def preprocess_image(image):
    """Preprocessing the image to [0., 1.]."""
    MIN_BOUND = 300
    MAX_BOUND = -300
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1.] = 1.
    image[image < 0.] = 0.

    return image
