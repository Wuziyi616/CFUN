"""
Faster U-net

Main configurations, train and test functions for the MM-WHS2017 Challenge dataset.

"""


import os
import json
import time
import numpy as np
import nibabel as nib
from imgaug import augmenters as iaa

from config import Config
import model
import utils
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

############################################################
#  Configurations
############################################################


class HeartConfig(Config):
    """Configuration for training on the heart dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "heart"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 7  # Background + seven other classes

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 45

    # Number of validation steps to run at the end of every training epoch.
    VALIDATION_STEPS = 10

    # Backbone network architecture
    # Supported values are: P3D63, P3D131, P3D199.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    BACKBONE = "P3D19"

    # The strides of each layer of the FPN Pyramid. These values
    # are based on the P3D-Resnet backbone.
    BACKBONE_STRIDES = [8, 16]

    # Channel numbers in backbone.
    BACKBONE_CHANNELS = [16, 32]

    # Size of the fully-connected layers in the classification graph
    FPN_CLASSIFY_FC_LAYERS_SIZE = 128  # 1024

    # Channels in U-net of the mrcnn mask branch
    UNET_MASK_BRANCH_CHANNEL = 20

    # Size of the top-down layers used to build the feature pyramid
    TOP_DOWN_PYRAMID_SIZE = 128  # 256

    # Channels of the conv layer in RPN
    RPN_CONV_CHANNELS = 256  # 512

    # Length of square anchor side in pixels
    # Note that the length of depth dimension is a half
    # of height and width dimension
    # Set some heart-specific values here
    RPN_ANCHOR_SCALES = (64, 128)

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1  # Use a small amount of anchors here

    # Ratios of anchors at each cell
    # A value of 1 represents a square anchor
    RPN_ANCHOR_RATIOS = [1]

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 128  # 256

    # ROIs kept before non-maximum suppression
    PRE_NMS_LIMIT = 1000  # 6000

    # ROIs kept after non-maximum suppression (training and inference)
    POST_NMS_ROIS_TRAINING = 500  # 2000
    POST_NMS_ROIS_INFERENCE = 64  # 1000

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = False  # True

    # Input image resizing
    # Generally, use the "square" resizing mode for training and predicting
    # and it should work well in most cases. In this mode, images are scaled
    # up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
    # scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
    # padded with zeros to make it a square so multiple images can be put
    # in one batch.
    # Available resizing modes:
    # none:   No resizing or padding. Return the image unchanged.
    # square: Resize and pad with zeros to get a square image
    #         of size [max_dim, max_dim, max_dim].
    # pad64:  Pads width and height with zeros to make them multiples of 64.
    #         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
    #         up before padding. IMAGE_MAX_DIM is ignored in this mode.
    #         The multiple of 64 is needed to ensure smooth scaling of feature
    #         maps up and down the 6 levels of the FPN pyramid (2**6=64).
    # crop:   Picks random crops from the image. First, scales the image based
    #         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
    #         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
    #         IMAGE_MAX_DIM is not used in this mode.
    # self:   Self-designed resize strategy.
    #         Resize the image to [IMAGE_MAX_DIM, IMAGE_MAX_DIM, IMAGE_MIN_DIM, 1]
    # image size is around [512, 512, 200-400]
    IMAGE_RESIZE_MODE = "self"
    IMAGE_MIN_DIM = 192
    IMAGE_MAX_DIM = 320
    # Minimum scaling ratio. Checked after MIN_IMAGE_DIM and can force further
    # up scaling. For example, if set to 2 then images are scaled up to double
    # the width and height, or more, even if MIN_IMAGE_DIM doesn't require it.
    # However, in 'square' mode, it can be overruled by IMAGE_MAX_DIM.
    IMAGE_MIN_SCALE = 0
    # Number of color channels per image. grey-scale = 1, RGB = 3, RGB-D = 4
    IMAGE_CHANNEL_COUNT = 1

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask-RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 15  # 200

    # Pooled ROIs
    POOL_SIZE = [12, 12, 12]  # [7, 7, 7]
    MASK_POOL_SIZE = [96, 96, 96]

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 32  # 100

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 32  # 100

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 100.,  # correct detection in rpn is proved to be very important!
        "rpn_bbox_loss": 50.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 20.,
        "mrcnn_mask_loss": 1.,
        "mrcnn_mask_edge_loss": 1.
    }

    # Train or freeze batch normalization layers
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when predicting
    TRAIN_BN = False  # Defaulting to False since batch size is often small


############################################################
#  Dataset
############################################################

class HeartDataset(utils.Dataset):

    def load_heart(self, subset):
        """Load a subset of the heart dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have seven classes to add.
        self.add_class("heart", 1, "a")
        self.add_class("heart", 2, "b")
        self.add_class("heart", 3, "c")
        self.add_class("heart", 4, "d")
        self.add_class("heart", 5, "e")
        self.add_class("heart", 6, "f")
        self.add_class("heart", 7, "g")

        # Train or validation dataset?
        assert subset in ["train", "val"]

        # Load dataset info
        info = json.load(open(args.data + "dataset.json"))
        info = list(info['train_and_test'])

        if subset == "train":
            info = info[13:]
        else:
            info = info[:13]

        # Add images and masks
        for a in info:
            image = nib.load(a['image']).get_data().copy()
            height, width, depth = image.shape
            self.add_image(
                "heart",
                image_id=a['image'],  # use file name as a unique image id
                path=a['image'],
                width=width, height=height, depth=depth,
                mask=a['label'])  # save the path of the corresponding mask

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,D,C] Numpy array."""
        # Load image
        image = nib.load(self.image_info[image_id]['path']).get_data().copy()
        return np.expand_dims(image, axis=-1)

    def process_mask(self, mask):
        """Given the [depth, height, width] mask that may contains many classes of annotations.
        Generate instance masks and a mask that only contains one class (heart) from it.
        Returns:
        masks: A np.int32 array of shape [depth, height, width, instance_count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        masks = np.zeros((self.num_classes, mask.shape[0], mask.shape[1], mask.shape[2]))
        for i in range(self.num_classes):
            masks[i][mask == i] = 1
        # Return masks and array of class IDs of each instance.
        return masks.astype(np.int32), np.arange(1, self.num_classes, 1, dtype=np.int32)

    def load_mask(self, image_id):
        """Load the specified mask and return a [H,W,D] Numpy array.
       Returns:
        masks: A np.int32 array of shape [height, width, depth].
        """
        # If not a heart dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "heart":
            return super(self.__class__, self).load_mask(image_id)

        # Convert masks to a bitmap mask of shape [height, width, depth, instance_count]
        # Load the mask.
        mask = nib.load(self.image_info[image_id]['mask']).get_data().copy()
        return mask

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "heart":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model"""
    # Training dataset
    dataset_train = HeartDataset()
    dataset_train.load_heart("train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = HeartDataset()
    dataset_val.load_heart("val")
    dataset_val.prepare()

    print("Train all layers")
    model.train_model(dataset_train, dataset_val,
                      learning_rate=config.LEARNING_RATE,
                      epochs=1000)


############################################################
#  Detection
############################################################

def test(model, limit, save, bbox):
    """Test the model.
    model: the model to test.
    limit: the images to be used.
    save: whether to save the masks.
    limit: whether to draw the bboxes.
    """
    per_class_ious = []
    info = json.load(open(args.data + "dataset.json"))
    info = list(info['train_and_test'])
    detect_time = 0
    for path in info[:limit]:
        path_image = path['image']
        path_label = path['label']
        image = nib.load(path_image).get_data().copy()
        label = nib.load(path_label)  # load the gt-masks
        affine = label.affine  # prepared to save the predicted mask later
        label = label.get_data().copy()
        image = np.expand_dims(image, -1)
        start_time = time.time()
        result = model.detect([image])[0]
        detect_time += time.time() - start_time
        print("detect_time:", time.time() - start_time)
        """The shape of result: a dict containing
        {
            "rois": final_rois,           [N, (y1, x1, z1, y2, x2, z2)] in real coordinates
            "class_ids": final_class_ids, [N]
            "scores": final_scores,       [N]
            "mask": final_mask,           [mask_shape[0], mask_shape[1], mask_shape[2]]
        }"""
        rois = result["rois"]
        class_ids = result["class_ids"]
        scores = result["scores"]
        mask = result["mask"]
        # Prepare the gt-masks and pred-masks to calculate the ious.
        gt_masks = np.zeros((image.shape[0], image.shape[1], image.shape[2], model.config.NUM_CLASSES - 1))
        pred_masks = np.zeros((image.shape[0], image.shape[1], image.shape[2], model.config.NUM_CLASSES - 1))
        # Generate the per instance gt masks.
        for j in range(model.config.NUM_CLASSES - 1):
            gt_masks[:, :, :, j][label == j + 1] = 1
        # Generate the per instance predicted masks.
        for j in range(model.config.NUM_CLASSES - 1):
            pred_masks[:, :, :, j][mask == j + 1] = 1
        # calculate different kind of ious
        per_class_iou = utils.compute_per_class_mask_iou(gt_masks, pred_masks)
        per_class_ious.append(per_class_iou)
        # Save the results
        if save == "true":
            # Draw bboxes
            if bbox == "true":
                y1, x1, z1, y2, x2, z2 = rois[0, :]
                mask[y1, x1:x2, z1] = 10
                mask[y1, x1:x2, z2] = 10
                mask[y2, x1:x2, z1] = 10
                mask[y2, x1:x2, z2] = 10
                mask[y1:y2, x1, z1] = 10
                mask[y1:y2, x2, z1] = 10
                mask[y1:y2, x1, z2] = 10
                mask[y1:y2, x2, z2] = 10
                mask[y1, x1, z1:z2] = 10
                mask[y1, x2, z1:z2] = 10
                mask[y2, x1, z1:z2] = 10
                mask[y2, x2, z1:z2] = 10
            vol = nib.Nifti1Image(mask.astype(np.int32), affine)
            if not os.path.exists("./results"):
                os.makedirs("./results")
            nib.save(vol, "./results/" + str(per_class_iou.mean()) + "_" + path_image[-17:])
        print(path_image[-17:] + " detected done. iou = " + str(per_class_iou))
    print("Test completed.")
    # Print the iou results.
    per_class_ious = np.array(per_class_ious)
    print("per class iou mean:", np.mean(per_class_ious, axis=0))
    print("std:", np.std(per_class_ious, axis=0))
    print("Total ious mean:", per_class_ious.mean())
    print("Total detect time:", detect_time)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train the Faster-Unet model to apply whole heart segmentation.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'test'")
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.pth",
                        help="Path to weights .pth file")
    parser.add_argument('--stage', required=True,
                        help="The training_stages now, 'beginning' or 'finetune'")
    parser.add_argument('--logs', required=False,
                        default="./logs/",
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--data', required=True,
                        default="../data/",
                        metavar="/path/to/data/",
                        help='Dataset directory (default=../data/)')
    parser.add_argument('--limit', required=False,
                        default=5,
                        help='The number of images used for testing (default=4)')
    parser.add_argument('--save', required=False,
                        default="true",
                        help='Whether to save the detected masks (default=False)')
    parser.add_argument('--bbox', required=False,
                        default="false",
                        help='Whether to draw the bboxes (default=False)')

    args = parser.parse_args()

    assert args.stage in ['beginning', 'finetune']

    print("Weights: ", args.weights)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = HeartConfig(args.stage.lower())
    else:
        class InferenceConfig(HeartConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0.7
            DETECTION_MAX_INSTANCES = 1
        config = InferenceConfig(args.stage.lower())

    config.display()

    # Create model
    if args.command == "train":
        model = model.MaskRCNN(config=config, model_dir=args.logs, test_flag=False)
    else:
        model = model.MaskRCNN(config=config, model_dir=args.logs, test_flag=True)

    if config.GPU_COUNT:
        model = model.cuda()

    if args.weights.lower() != "none":
        # Select weights file to load
        weights_path = args.weights
        # Load weights
        print("Loading weights ", weights_path)
        model.load_weights(weights_path)

    # Train or evaluate
    if args.command == "train":
        print("Training...")
        train(model)
    elif args.command == "test":
        print("Testing...")
        test(model, int(args.limit.lower()), args.save.lower(), args.bbox.lower())
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'test'".format(args.command))
