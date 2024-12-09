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

def read_2(binary_file_path, tensor_shape=(1, 1, 3549, 85)):
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

def read3(binary_file_path, tensor_shape=(1, 1, 10647, 85)):
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


def parse_yolov6_output(tensor, confidence_threshold=0.25, imgsz=640,num_classes=80):
    """
    Parse YOLOv6 inference results into label format, including bounding boxes, class ID, and confidence.

    :param tensor: YOLOv6 inference tensor (1, 1, grid_size, 85)
    :param confidence_threshold: Confidence threshold
    :param num_classes: Total number of classes
    :return: Parsed label results, formatted as [(x1, y1, x2, y2, class_id, confidence), ...]
    """
    # Remove batch and channel dimensions
    predictions = tensor[0, 0, :, :]  # Shape: (1,1, 8400, 85) -> (8400, 85)

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
        x, y, w, h = convert_bboxes_to_xyminmax(bbox,imgsz)[0]
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
def convert_bboxes_to_xyminmax(bboxes,imgsz=640):
    """
    转换 YOLO 输出的 x_center, y_center, width, height 到 x_min, y_min, x_max, y_max 格式。
    Args:
        bboxes (numpy.ndarray): 输入边界框数组，形状为 (N, 4)，每行为 [x_center, y_center, width, height]

    Returns:
        numpy.ndarray: 转换后的边界框坐标，格式为 [x_min, y_min, x_max, y_max]
    """
    # 如果是 1D 数据重塑为 2D
    if bboxes.ndim == 1:
        bboxes = bboxes.reshape(-1, 4)

    # 提取参数
    x_center = bboxes[:, 0]
    y_center = bboxes[:, 1]
    width = bboxes[:, 2]
    height = bboxes[:, 3]

    # # 转换为实际的 x_min, y_min, x_max, y_max
    # x_min = x_center - width / 2
    # y_min = y_center - height / 2
    # x_max = x_center + width / 2
    # y_max = y_center + height / 2

    # 堆叠成最终格式
    converted_bboxes = np.stack((x_center, y_center, width, height), axis=1)

    return converted_bboxes


def convert_label_to_nms_result(label,img_width = 416, img_height = 416,iou_threshold = 0.5):
    """
    Convert formatted label information into NMS result list.

    :param label: Tuple containing bounding box, class ID, and confidence (x1, y1, x2, y2, class_id, confidence)
    :return: NMS result list [(x1, y1, x2, y2, class_id, confidence)]
    """
    # Convert bbox to np.float32 tuple
    # bboxes = [
    #     yolo_to_bbox(label, img_width, img_height) # Add confidence score
    #     for label in label
    # ]
    # print(bboxes)
    filtered_bboxes = nms(label, iou_threshold)
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


def demo_postprocess(outputs, img_size, p6=False):
    """
    用于处理 YOLO 输出。
    修正 grids 和 expanded_strides 使其与 outputs 匹配。
    """
    grids = []
    expanded_strides = []
    strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        # 生成每个尺度的网格
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), axis=-1).reshape(1, -1, 2)  # 重塑维度
        grids.append(grid)

        # 为每个网格生成步幅
        expanded_stride = np.full(grid.shape, stride)  # 创建步幅
        expanded_strides.append(expanded_stride)

    # 合并所有尺度的网格与步幅
    grids = np.concatenate(grids, axis=1)  # 合并所有尺度的网格
    expanded_strides = np.concatenate(expanded_strides, axis=1)  # 合并所有尺度的步幅

    # 确保 grids 和 expanded_strides 的维度与 outputs 匹配
    # 输出尺寸应与 (N, 2) 保持一致
    if outputs.shape[1] != grids.shape[1]:
        grids = grids[:, :outputs.shape[1], :]
        expanded_strides = expanded_strides[:, :outputs.shape[1], :]

    # 修正计算 (确保输出匹配网格步幅和偏移量)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs

def demo_2(outputs, img_size, p6=False):
    grids = []
    expanded_strides = []
    strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)

    if outputs.shape[1] != grids.shape[1]:
        grids = grids[:, :outputs.shape[1], :]
        expanded_strides = expanded_strides[:, :outputs.shape[1], :]

    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs


def normalize_to_pixel_coords(coord, imgsz=[640, 640]):

    x1_pixel = int(coord[0] * imgsz[0])/100
    y1_pixel = int(coord[1] * imgsz[1])/100
    x2_pixel = int(coord[2] * imgsz[0])/100
    y2_pixel = int(coord[3] * imgsz[1])/100

    pixel_coords = [x1_pixel, y1_pixel, x2_pixel, y2_pixel]

    return pixel_coords

def display(image_path,tensor,imgsz,conf_thres,iou_thres):
    # Load the image
    image = cv2.imread(image_path)

    # Parse the YOLOv6 output
    labels = parse_yolov6_output(tensor, confidence_threshold=conf_thres,imgsz=imgsz)

    # Convert bounding boxes to pixel coordinates
    converted_labels = convert_label_to_nms_result(labels,img_width = imgsz, img_height = imgsz,iou_threshold = iou_thres)

    # Print the labels (class ID, bbox coordinates,  confidence)
    # Draw bounding boxes and labels on the image
    for label in converted_labels:
        class_id, x1, y1, x2, y2, confidence = label
        normalized_coords = label[1:5]
        print(normalized_coords)
        name = coco_classes[class_id]
        # print(f"BBox1: ({int(x1)}, {int(y1)}), ({int(x2)}, {int(y2)}), Class: {name}, Confidence: {confidence:.2f}")
        pixel_coords = normalize_to_pixel_coords(normalized_coords, [imgsz,imgsz])
        x_min, y_min,x_max,y_max = pixel_coords
        print(f"BBox: ({int(x_min)}, {int(y_min)}), ({int(x_max)}, {int(y_max)}), Class: {name}, Confidence: {confidence:.2f}")
        # Draw the bounding box
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        # Draw the label
        label_text = f"{name}: {confidence:.2f}"
        cv2.putText(image, label_text, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Show the image with bounding boxes
    cv2.imshow("NNCTRL Inference", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def read_tensor_from_binary(binary_file_path, tensor_shape=(1, 1, 8400, 85)):
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

        return tensor
    return None

def transfer_bin(tensor):
    output = tensor.squeeze(0).squeeze(0)  # 变成 (3549, 85)

    x_center = output[:, 0]
    y_center = output[:, 1]
    width = output[:, 2]
    height = output[:, 3]


    # 转换为 [x_min, y_min, x_max, y_max]
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2

    # 确保 x_min, y_min, x_max, y_max 是 PyTorch 张量
    x_min = torch.tensor(x_min) if isinstance(x_min, np.ndarray) else x_min
    y_min = torch.tensor(y_min) if isinstance(y_min, np.ndarray) else y_min
    x_max = torch.tensor(x_max) if isinstance(x_max, np.ndarray) else x_max
    y_max = torch.tensor(y_max) if isinstance(y_max, np.ndarray) else y_max

    # 使用 torch.stack
    bboxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)

    # 提取 object confidence 和 class confidence
    object_conf = output[:, 4]
    class_conf = output[:, 5:]  # 后80列是每个类别的分数

    # 计算最终的置信度和类别ID
    def ensure_tensor(x):
        """确保输入是 PyTorch 张量"""
        if isinstance(x, np.ndarray):
            return torch.tensor(x)
        return x

    # 确保 class_conf 是 PyTorch 张量
    class_conf = ensure_tensor(class_conf)

    # 计算类别 ID
    class_id = torch.argmax(class_conf, dim=1)
    if isinstance(object_conf, np.ndarray):
        object_conf = torch.tensor(object_conf)

    # 计算置信度
    confidence = object_conf * class_conf[torch.arange(len(class_conf)), class_id]  # 置信度是object_conf乘以最大class_conf

    # 拼接结果
    final_output = torch.cat([bboxes, confidence.unsqueeze(1), class_id.unsqueeze(1)], dim=1)

    # 转换为 NumPy 格式（如果需要）
    final_output_np = final_output.numpy()

    return final_output_np

def draw_detections(image_path, detections):
    """
    在原图上绘制检测结果
    Args:
        image_path (str): 原图路径
        detections (numpy.ndarray): 检测结果，形状为 (N, 6)，包含 [x_min, y_min, x_max, y_max, confidence, class_id]
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("无法加载图像:", image_path)
        return

    # 遍历检测结果
    for det in detections:
        print(det)
        x_min, y_min, x_max, y_max, confidence, class_id = det
        # print(x_min, y_min, x_max, y_max, confidence, class_id)

        # 将坐标转换为整数
        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

        # 绘制边界框
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # 显示类别 ID 和置信度
        label = f"ID: {int(class_id)}, Conf: {confidence:.2f}"
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow("Detections", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 可选择保存结果
    result_path = "detection_result.jpg"
    cv2.imwrite(result_path, image)
    print(f"检测结果已保存到: {result_path}")

# Main function to input image and draw bounding boxes and labels
if __name__ == "__main__":
    image_path = "F:/source/llamatest/bininference/img/l640.jpg"
    binary_file_path = "F:/source/llamatest/bininference/out/l640xxout.bin"
    device = 'cpu'
    tensor_shape = (1,1, 8400, 85)  # For YOLOX_X or YOLOX_Large
    imgsz = 640

    # tensor_shape = (1, 1, 3549, 85)  # For YOLOX_TINY or YOLOX_NANO
    # imgsz = 416


    # Confidence threshold
    conf_thres: float = .75  # @param {type:"number"}
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
    tensor = read_tensor_from_binary(binary_file_path, tensor_shape=tensor_shape)

    # final_output_np = transfer_bin(tensor)
    # print(final_output_np)

    postres = demo_postprocess(tensor, img_size=[imgsz, imgsz],p6=False)

    # draw_detections(image_path, postres)


    # Load the image
    display(image_path,postres,imgsz,conf_thres,iou_thres)

