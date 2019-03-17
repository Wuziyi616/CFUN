"""
CFUN

Main configurations, train and test functions for the LiTS dataset.

"""


import os
import time
import numpy as np
import nibabel as nib
from skimage.transform import resize

from config import Config
import model
import utils
import warnings
warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

############################################################
#  Configurations
############################################################


class LiTSConfig(Config):
    """Configuration for training on the LiTS dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "LiTS"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # Background + liver + tumor

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Number of validation steps to run at the end of every training epoch
    VALIDATION_STEPS = 20

    # Number of epoch to save the model weights
    SAVE_EPOCH = 5

    # Number of workers for train data generation
    TRAIN_NUM_WORKERS = 15

    # Number of workers for validation data generation
    VAL_NUM_WORKERS = 10

    # Backbone network architecture
    # Supported values are: P3D63, P3D131, P3D199.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    BACKBONE = "P3D35"

    # The strides of each layer of the FPN Pyramid. These values
    # are based on the P3D-Resnet backbone.
    BACKBONE_STRIDES = [8, 16]

    # Channel numbers in backbone.
    BACKBONE_CHANNELS = [24, 48]

    # Size of the fully-connected layers in the classification graph
    FPN_CLASSIFY_FC_LAYERS_SIZE = 320  # 1024

    # Channels in U-net of the mrcnn mask branch
    UNET_MASK_BRANCH_CHANNEL = 32

    # Size of the top-down layers used to build the feature pyramid
    TOP_DOWN_PYRAMID_SIZE = 160  # 256

    # Channels of the conv layer in RPN
    RPN_CONV_CHANNELS = 320  # 512

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
    POST_NMS_ROIS_INFERENCE = 50  # 1000

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = False  # True

    # Input image resizing
    # self:   Self-designed resize strategy.
    #         Resize the image to [IMAGE_MAX_DIM, IMAGE_MAX_DIM, IMAGE_MIN_DIM, 1]
    # the average image shape is [512, 512, 448]
    IMAGE_RESIZE_MODE = "self"
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 320

    # Pad all images to this shape and then resize
    PAD_IMAGE_SHAPE = [646, 646, 536]

    # Mean spacing of all the training image
    MEAN_SPACING = np.array([0.79272507, 0.79272507, 1.50625819])

    # Whether to apply augmentation to images in the train data generator
    AUGMENTATION = True

    # Whether to shuffle the data in the generator
    SHUFFLE_DATASET = True

    # Minimum scaling ratio. Checked after MIN_IMAGE_DIM and can force further
    # up scaling. For example, if set to 2 then images are scaled up to double
    # the width and height, or more, even if MIN_IMAGE_DIM doesn't require it.
    # However, in 'square' mode, it can be overruled by IMAGE_MAX_DIM.
    IMAGE_MIN_SCALE = 0
    # Number of color channels per image. grey-scale = 1, RGB = 3, RGB-D = 4
    IMAGE_CHANNEL_COUNT = 1

    # Pooled ROIs
    POOL_SIZE = [12, 12, 12]
    MASK_POOL_SIZE = [32, 80, 80]

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection
    # TODO: adjust this threshold bigger to get more pred_bbox?
    DETECTION_NMS_THRESHOLD = 0.7

    # The iou of pred_bbox and gt_bbox > this threshold is regarded as positive rois
    DETECTION_TARGET_IOU_THRESHOLD = 0.5

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 32  # 100

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 32  # 100

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 50.,  # correct detection in rpn is proved to be very important!
        "rpn_bbox_loss": 5.,
        "mrcnn_class_loss": 50.,
        "mrcnn_bbox_loss": 5.,
        "mrcnn_mask_loss": 2.,
        "mrcnn_mask_edge_loss": 0.25
    }

    # Train or freeze batch normalization layers
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when predicting
    TRAIN_BN = False  # Defaulting to False since batch size is often small


############################################################
#  Dataset
############################################################

class LiTSDataset(utils.Dataset):

    def load_LiTS(self, subset):
        """Load a subset of the heart dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have seven classes to add.
        self.add_class("LiTS", 1, "liver")
        self.add_class("LiTS", 2, "tumor")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        all_image_path = []
        all_label_path = []
        if subset == "train":
            for i in range(111):
                all_image_path.append(args.data + "image_np/liver_" + str(i) + ".npy")
                all_label_path.append(args.data + "label_np/liver_label_" + str(i) + ".npy")
        else:
            for i in range(111, 131):
                all_image_path.append(args.data + "image_np/liver_" + str(i) + ".npy")
                all_label_path.append(args.data + "label_np/liver_label_" + str(i) + ".npy")

        # Add images and masks
        for image_path, label_path in zip(all_image_path, all_label_path):
            # image = np.load(image_path).astype(np.float32)
            # height, width, depth = image.shape
            self.add_image(
                "LiTS",
                image_id=image_path,  # use its path as a unique image id
                path=image_path,
                # width=width, height=height, depth=depth,
                mask=label_path)  # save the path of the corresponding mask

    def load_image(self, image_id):
        """Load the specified image and return a [H, W, D] Numpy array."""
        # Load image
        image = np.load(self.image_info[image_id]['path']).astype(np.float32)

        return image

    def process_mask(self, mask):
        """Given the [depth, height, width] mask that may contains many classes of annotations.
        Generate instance masks and class_ids.
        Returns:
        masks: A np.int32 array of shape [num_classes, depth, height, width] with one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        masks = np.zeros((self.num_classes, mask.shape[0], mask.shape[1], mask.shape[2]))
        for i in range(self.num_classes):
            masks[i][mask == i] = 1

        # Return masks and array of class IDs of each instance.
        return masks.astype(np.float32), np.arange(1, self.num_classes, 1, dtype=np.int32)

    def load_mask(self, image_id):
        """Load the specified mask and return a [H, W, D] Numpy array.
       Returns:
        masks: A np.int32 array of shape [height, width, depth].
        """
        # If not a LiTS dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "LiTS":
            return super(self.__class__, self).load_mask(image_id)

        # Load mask
        mask = np.load(self.image_info[image_id]['mask']).astype(np.int32)

        return mask

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "LiTS":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model"""
    # Training dataset
    dataset_train = LiTSDataset()
    dataset_train.load_LiTS("train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = LiTSDataset()
    dataset_val.load_LiTS("val")
    dataset_val.prepare()

    print("Train all layers")
    model.train_model(dataset_train, dataset_val,
                      learning_rate=config.LEARNING_RATE,
                      epochs=1000)


############################################################
#  Detection
############################################################

def test(model):
    """Test the model."""
    save_path = "./results/" + args.weights[27:] + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    per_class_ious = []
    box_iou = []
    detect_time = 0
    for i in range(args.limit, 131):
        image = np.load(args.data + "image_np/liver_" + str(i) + ".npy").astype(np.float32)  # [H, W, D]
        label = np.load(args.data + "label_np/liver_label_" + str(i) + ".npy").astype(np.int32)  # [H, W, D, num_class]
        gt_bbox = utils.extract_bboxes(label)
        gt_bbox = utils.extend_bbox(gt_bbox, label.shape)
        nib_label = nib.load("/media/disk1/LiTS/labelTr/segmentation-" + str(i) + ".nii.gz")
        affine = nib_label.affine  # prepared to save the predicted mask later
        ori_shape = nib_label.shape
        try:
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
            if args.stage == 'beginning':
                mask = np.zeros(mask.shape).astype(np.int32)
            rois = rois.clip(min=0)
            rois[:, 3] = rois[:, 3].clip(max=mask.shape[0] - 1)
            rois[:, 4] = rois[:, 4].clip(max=mask.shape[1] - 1)
            rois[:, 5] = rois[:, 5].clip(max=mask.shape[2] - 1)
            rois = rois.astype(np.int32)
            # import pdb
            # pdb.set_trace()
            # compute bbox iou
            print("gt_bbox:", gt_bbox, "pred_bbox:", rois)
            box_iou.append(utils.compute_overlaps(np.array([gt_bbox]), rois)[0, 0])
            if args.stage != 'beginning':
                # Prepare the gt-masks and pred-masks to calculate the ious. [H, W, D, num_classes - 1]
                gt_masks = np.zeros(label.shape[:3] + (model.config.NUM_CLASSES - 1,))
                pred_masks = np.zeros(image.shape + (model.config.NUM_CLASSES - 1,))
                for j in range(model.config.NUM_CLASSES - 1):
                    gt_masks[:, :, :, j][label == j + 1] = 1
                    pred_masks[:, :, :, j][mask == j + 1] = 1
                # calculate different kind of ious
                per_class_iou = utils.compute_per_class_mask_iou(gt_masks, pred_masks)
                per_class_ious.append(per_class_iou)
            # Save the results
            if args.save:
                # Draw bboxes
                if args.bbox:
                    for j in range(rois.shape[0]):
                        y1, x1, z1, y2, x2, z2 = rois[j, :]
                        mask[y1:y2, x1:x2, z1:z2] = 100
                mask = resize(mask, ori_shape, order=0, mode='constant', preserve_range=True, anti_aliasing=False)
                vol = nib.Nifti1Image(mask.astype(np.uint8), affine)
                if args.stage != 'beginning':
                    nib.save(vol, save_path + str(per_class_iou.mean()) + "_liver_" + str(i) + ".nii.gz")
                    print("liver_" + str(i) + " detected done. iou = " + str(per_class_iou))
                else:
                    nib.save(vol, save_path + str(box_iou[-1]) + "_liver_" + str(i) + ".nii.gz")
                    print("liver_" + str(i) + " detected done. box_iou = " + str(box_iou[-1]))
        except:
            print("detect error!")
            pass
    print("Test completed.")
    # Print the iou results.
    box_iou = np.array(box_iou)
    print("box iou:", box_iou)
    print("mean:", box_iou.mean())
    if args.stage != 'beginning':
        per_class_ious = np.array(per_class_ious)
        print("per class iou mean:", np.mean(per_class_ious, axis=0))
        print("std:", np.std(per_class_ious, axis=0))
        print("Total ious mean:", per_class_ious.mean())
        print("Total detect time:", detect_time)


def predict_for_submit(model):
    """Predict on the test set for final submission.
    model: the model to test.
    limit: the images to be used.
    save: whether to save the masks.
    """
    for i in range(args.limit, 70):
        image = np.load(args.data + "image_test_np/liver_" + str(i) + ".npy").astype(np.float32)  # [512, 512, D]
        nib_image = nib.load(args.data + "imagesTs/test-volume-" + str(i) + ".nii.gz")
        affine = nib_image.affine
        ori_shape = nib_image.shape
        start_time = time.time()
        try:
            result = model.detect([image])[0]
            print("processing", i, "detect_time:", time.time() - start_time)
            mask = result["mask"]
            mask = resize(mask, ori_shape, order=0, preserve_range=True, anti_aliasing=False)
            vol = nib.Nifti1Image(mask.astype(np.uint8), affine)
            if not os.path.exists("./results/submissions/"):
                os.makedirs("./results/submissions/")
            nib.save(vol, "./results/submissions/test-segmentation-" + str(i) + ".nii")
        except:
            pass

    print("prediction completed")


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train the CFUN model to apply whole heart segmentation.'
    )
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'test'")
    parser.add_argument('--weights', required=False, default="none",
                        metavar="/path/to/weights.pth",
                        help="Path to weights .pth file")
    parser.add_argument('--stage', required=False, type=str, default='beginning',
                        help="The training_stages now, 'beginning' or 'finetune'")
    parser.add_argument('--logs', required=False,
                        default="./logs/",
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--data', required=False, default="../",
                        metavar="/path/to/data/",
                        help='Dataset directory (default=../data/)')
    parser.add_argument('--limit', required=False, type=int,
                        default=5,
                        help='The number of images used for testing (default=4)')
    parser.add_argument('--save', required=False, type=bool,
                        default=True,
                        help='Whether to save the detected masks (default=True)')
    parser.add_argument('--bbox', required=False, type=bool,
                        default=False,
                        help='Whether to draw the bbox (default=False)')

    args = parser.parse_args()

    assert args.stage in ['beginning', 'together', 'finetune']

    print("Weights: ", args.weights)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = LiTSConfig(args.stage.lower())
    else:
        class InferenceConfig(LiTSConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0.7
            DETECTION_MAX_INSTANCES = 10
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
        for p in model.parameters():
            p.require_grad = False
        test(model)
    elif args.command == "submit":
        print("Predicting...")
        for p in model.parameters():
            p.require_grad = False
        predict_for_submit(model)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'test'".format(args.command))
