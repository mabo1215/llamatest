import numpy as np
import struct
import matplotlib.pyplot as plt
import os, requests, torch, math, cv2, yaml
from PIL import Image
import logging
import shutil
from typing import List, Optional




def set_logging(name=None):
    rank = int(os.getenv('RANK', -1))
    logging.basicConfig(format="%(message)s", level=logging.INFO if (rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)

LOGGER = set_logging(__name__)
def load_yaml(file_path):
    """Load data from yaml file."""
    if isinstance(file_path, str):
        with open(file_path, errors='ignore') as f:
            data_dict = yaml.safe_load(f)
    return data_dict


def save_yaml(data_dict, save_path):
    """Save data to yaml file"""
    with open(save_path, 'w') as f:
        yaml.safe_dump(data_dict, f, sort_keys=False)

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    '''Resize and pad image while meeting stride-multiple constraints.'''
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    elif isinstance(new_shape, list) and len(new_shape) == 1:
       new_shape = (new_shape[0], new_shape[0])

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return im, r, (left, top)

def read_tensor_from_binary(binary_file_path, tensor_shape=(1, 1, 3549, 85)):
    """
    Read the YOLOv6 inference tensor from a binary file as raw bytes without converting to np.float32.

    :param binary_file_path: Path to the binary file containing the inference results
    :param tensor_shape: The expected shape of the tensor (default is (1, 1, 3549, 85))
    :return: Raw binary data as bytes
    """
    total_elements = np.prod(tensor_shape)
    # Open the binary file and read raw bytes
    with open(binary_file_path, 'rb') as f:
        raw_data = f.read()  # Read all bytes in the file
        binary_data = np.frombuffer(raw_data, dtype=np.float32)

        # Unpack the binary data into a list of float32 values
        unpacked_data = struct.unpack(f'{total_elements}f', binary_data)

        # Convert the unpacked data into a numpy array and reshape it into the tensor shape
        tensor = np.array(unpacked_data, dtype=np.float32).reshape(tensor_shape)
        print(unpacked_data)
        return tensor
    return None

def read_tensor_from_binary3(binary_file_path, tensor_shape=(1, 1, 10647, 85)):
    """
    Read the YOLOv6 inference tensor from a binary file.

    :param binary_file_path: Path to the binary file containing the inference results
    :param tensor_shape: The expected shape of the tensor (default is (1, 1, 3549, 85))
    :return: Tensor as a NumPy array
    """
    # Calculate the total number of elements in the tensor
    total_elements = np.prod(tensor_shape)

    # Open the binary file
    with open(binary_file_path, 'rb') as f:
        # Read the entire file and unpack it into a numpy array
        tensor_data = np.fromfile(f, dtype=np.float32, count=total_elements)
    # Reshape the data into the expected tensor shape
    tensor = tensor_data.reshape(tensor_shape)
    return tensor


def parse_yolov6_output(tensor, confidence_threshold=0.25, num_classes=80):
    """
    Parse YOLOv6 inference results into label format, including bounding boxes, class ID, and confidence.

    :param tensor: YOLOv6 inference tensor (1, 1, grid_size, 85)
    :param confidence_threshold: Confidence threshold
    :param num_classes: Total number of classes
    :return: Parsed label results, formatted as [(x1, y1, x2, y2, class_id, confidence), ...]
    """
    # Remove batch and channel dimensions
    predictions = tensor[0, 0, :, :]  # Shape: (3549, 85)

    # Separate bounding boxes, confidence, and class probabilities
    bboxes = predictions[:, :4]  # [x, y, w, h]
    object_confidences = predictions[:, 4:5]  # Object confidence
    class_probabilities = predictions[:, 5:]  # Class probabilities

    # Calculate total confidence = object_confidence * class_probability
    scores = object_confidences * class_probabilities  # Shape: (3549, num_classes)
    class_ids = np.argmax(scores, axis=1)  # Best class index for each box
    confidences = np.max(scores, axis=1)  # Max confidence for each box

    # Filter out predictions with low confidence
    keep = confidences > confidence_threshold
    bboxes = bboxes[keep]
    class_ids = class_ids[keep]
    confidences = confidences[keep]

    # Format results into a list
    results = []
    for bbox, class_id, confidence in zip(bboxes, class_ids, confidences):
        x, y, w, h = bbox
        results.append((int(class_id),int(x), int(y), int(w), int(h), float(confidence)))
    return results


def visualize_matrix(matrix):
    """
    Visualize matrix data.

    :param matrix: Matrix
    """
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title("Matrix Visualization")
    plt.show()


def convert_bbox_to_pixel_coords(bbox, image_width=416, image_height=416):
    """
    Convert normalized bounding box coordinates to pixel coordinates.

    :param bbox: Normalized bounding box [x1, y1, x2, y2]
    :param image_width: Input image width (default is 416)
    :param image_height: Input image height (default is 416)
    :return: Bounding box in pixel coordinates [x1_pixel, y1_pixel, x2_pixel, y2_pixel]
    """
    x1, y1, x2, y2 = bbox
    x1_pixel = x1 * image_width
    y1_pixel = y1 * image_height
    x2_pixel = x2 * image_width
    y2_pixel = y2 * image_height
    return [x1_pixel, y1_pixel, x2_pixel, y2_pixel]


def convert_all_bboxes(labels, image_width=416, image_height=416):
    """
    Convert all normalized bounding boxes to pixel coordinates.

    :param labels: NMS filtered detection results [(x1, y1, x2, y2, class_id, confidence), ...]
    :param image_width: Input image width (default is 416)
    :param image_height: Input image height (default is 416)
    :return: Bounding boxes in pixel coordinates
    """
    converted_labels = []
    for label in labels:
        bbox = label[:4]  # Extract bbox
        # print("bbox",bbox)
        # rounded_bbox = tuple(round(coord) for coord in bbox)
        rounded_bbox = ",".join(str(int(coord)) for coord in bbox)
        class_id = label[4]
        confidence = label[5]
        pixel_bbox = convert_bbox_to_pixel_coords(rounded_bbox, image_width, image_height)
        converted_labels.append((*pixel_bbox, class_id, confidence))
    return converted_labels

def yolo_to_bbox(yolo_label, img_width, img_height):
    """
    Convert YOLO format to bounding box format.
    Args:
        yolo_label: A tuple (class_id, x_center, y_center, width, height)
        img_width: Width of the image
        img_height: Height of the image

    Returns:
        A tuple (class_id, x_min, y_min, x_max, y_max)
    """
    class_id, x_center, y_center, width, height, confidence = yolo_label
    x_min = (x_center - width / 2)
    y_min = (y_center - height / 2)
    x_max = (x_center + width / 2)
    y_max = (y_center + height / 2)

    return (class_id, x_min, y_min, x_max, y_max,confidence)

def iou(bbox1, bbox2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Args:
        bbox1: A tuple (x_min, y_min, x_max, y_max)
        bbox2: A tuple (x_min, y_min, x_max, y_max)

    Returns:
        IoU value
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    union = bbox1_area + bbox2_area - intersection
    return intersection / union if union > 0 else 0


def nms(bboxes, iou_threshold):
    """
    Perform Non-Maximum Suppression (NMS) on bounding boxes.
    Args:
        bboxes: A list of tuples [(class_id, x_min, y_min, x_max, y_max, confidence), ...]
        iou_threshold: IoU threshold for NMS

    Returns:
        List of filtered bounding boxes
    """
    # Sort by confidence score
    bboxes = sorted(bboxes, key=lambda x: x[-1], reverse=True)

    filtered_bboxes = []
    while bboxes:
        # Pick the box with the highest confidence
        chosen_box = bboxes.pop(0)
        filtered_bboxes.append(chosen_box)

        # Remove boxes with high IoU
        bboxes = [
            box for box in bboxes
            if iou(chosen_box[1:5], box[1:5]) < iou_threshold
        ]

    return filtered_bboxes


def convert_label_to_nms_result(label,img_width = 416, img_height = 416,iou_threshold = 0.5):
    """
    Convert formatted label information into NMS result list.

    :param label: Tuple containing bounding box, class ID, and confidence (x1, y1, x2, y2, class_id, confidence)
    :return: NMS result list [(x1, y1, x2, y2, class_id, confidence)]
    """
    # Convert bbox to np.float32 tuple
    bboxes = [
        yolo_to_bbox(label, img_width, img_height) # Add confidence score
        for label in label
    ]
    # print(bboxes)
    filtered_bboxes = nms(bboxes, iou_threshold)
    # Return NMS result
    return filtered_bboxes

def process_image(path, stride=3, half=True):
  '''Process image before image inference.'''
  try:
    from PIL import Image
    img_src = np.asarray(Image.open(path))
    img_size = img_src.shape
    assert img_src is not None, f'Invalid image: {path}'
  except Exception as e:
    LOGGER.Warning(e)
  image = letterbox(img_src, img_size, stride=stride)[0]

  # Convert
  image = image.transpose((2, 0, 1))  # HWC to CHW
  image = torch.from_numpy(np.ascontiguousarray(image))
  image = image.half() if half else image.float()  # uint8 to fp16/32
  image /= 255  # 0 - 255 to 0.0 - 1.0

  return image, img_src, img_size

# Main function to input image and draw bounding boxes and labels
if __name__ == "__main__":
    image_path = "C:/work/datasets/test/img/mt416.jpg"
    binary_file_path = "C:/work/datasets/test/newoutputs/mt416xout.bin"
    device = 'cpu'
    v10_tensor_shape = (1,1, 3549, 85)
    # v5_tensor_shape = (1,1, 10647, 85)

    # Confidence threshold
    conf_thres: float = .45  # @param {type:"number"}
    iou_thres: float = .45  # @param {type:"number"}
    max_det: int = 1000  # @param {type:"integer"}
    agnostic_nms: bool = False  # @param {type:"boolean"}
    coco_classes = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "TV", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    ]

    # Read the tensor from the binary file
    tensor = read_tensor_from_binary(binary_file_path, tensor_shape=v10_tensor_shape)

    # Load the image
    image = cv2.imread(image_path)

    # Parse the YOLOv6 output
    labels = parse_yolov6_output(tensor, confidence_threshold=conf_thres)
    classes: Optional[List[int]] = None  # the classes to keep

    # Convert bounding boxes to pixel coordinates
    converted_labels = convert_label_to_nms_result(labels,img_width = 416, img_height = 416,iou_threshold = 0.5)

    # Print the labels (class ID, bbox coordinates,  confidence)
    # Draw bounding boxes and labels on the image
    for label in converted_labels:
        class_id, x1, y1, x2, y2, confidence = label
        name = coco_classes[class_id]
        print(
            f"BBox: ({int(x1)}, {int(y1)}), ({int(x2)}, {int(y2)}), Class ID: {name}, Confidence: {confidence:.2f}")
        # Draw the bounding box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # Draw the label
        label_text = f"{name}: {confidence:.2f}"
        cv2.putText(image, label_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Show the image with bounding boxes
    cv2.imshow("NNCTRL Inference", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()