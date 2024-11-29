import cv2
import numpy as np
import onnxruntime as ort
import time
import torch


# Initialize ONNX model for YOLOv10
def initialize_yolo_model(model_path):
    return ort.InferenceSession(model_path)


# Preprocess the frame for YOLOv10
def preprocess_yolo_input(frame, input_shape):
    h, w = input_shape
    frame_resized = cv2.resize(frame, (w, h))
    frame_normalized = frame_resized / 255.0  # Normalize pixel values to [0, 1]
    frame_transposed = frame_normalized.transpose(2, 0, 1)  # HWC to CHW
    frame_input = np.expand_dims(frame_transposed, axis=0).astype(np.float32)
    return frame_input


def v10postprocess(preds, max_det, nc=80):
    assert(4 + nc == preds.shape[-1])
    boxes, scores = preds.split([4, nc], dim=-1)
    max_scores = scores.amax(dim=-1)
    max_scores, index = torch.topk(max_scores, max_det, axis=-1)
    index = index.unsqueeze(-1)
    boxes = torch.gather(boxes, dim=1, index=index.repeat(1, 1, boxes.shape[-1]))
    scores = torch.gather(scores, dim=1, index=index.repeat(1, 1, scores.shape[-1]))

    scores, index = torch.topk(scores.flatten(1), max_det, axis=-1)
    labels = index % nc
    index = index // nc
    boxes = boxes.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))
    return boxes, scores, labels


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    dw = x[..., 2] / 2  # half-width
    dh = x[..., 3] / 2  # half-height
    y[..., 0] = x[..., 0] - dw  # top left x
    y[..., 1] = x[..., 1] - dh  # top left y
    y[..., 2] = x[..., 0] + dw  # bottom right x
    y[..., 3] = x[..., 1] + dh  # bottom right y
    return y

def postprocess_yolo_testoutput(output, frame_shape, input_shape, conf_threshold=0.25, iou_threshold=0.45,nc=80):
    """
    Process YOLOv10 model output for bounding boxes, classes, and scores.
    """
    detections = []
    h_frame, w_frame = frame_shape[:2]
    h_input, w_input = input_shape
    x_factor = w_frame / w_input
    y_factor = h_frame / h_input

    # Ensure output is a 2D array
    output = np.squeeze(output)  # Remove extra dimensions, if any

    boxes = []
    scores = []
    class_ids = []

    for det in output:
        # Now you can safely use split
        class_probs = det[5:]
        class_id = np.argmax(class_probs)
        confidence = class_probs[class_id]


        if confidence > conf_threshold:
            # Extract bbox in whxy format (center_x, center_y, width, height)
            x, y, w, h = det[:4]

            # Convert from (center_x, center_y, width, height) to (x1, y1, x2, y2)
            left = int((x - w / 2) * x_factor)
            top = int((y - h / 2) * y_factor)
            right = int((x + w / 2) * x_factor)
            bottom = int((y + h / 2) * y_factor)

            # Clip coordinates to be within image boundaries (to avoid drawing boxes outside image)
            left = max(0, min(w_frame, left))
            top = max(0, min(h_frame, top))
            right = max(0, min(w_frame, right))
            bottom = max(0, min(h_frame, bottom))

            boxes.append((left, top, right, bottom))
            scores.append(confidence)
            class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS)
    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)

        # Extract final detections after NMS
        for i in indices.flatten():
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            # Clip coordinates to ensure they are within frame bounds
            x1, y1, x2, y2 = box
            x1 = max(0, min(w_frame, x1))
            y1 = max(0, min(h_frame, y1))
            x2 = max(0, min(w_frame, x2))
            y2 = max(0, min(h_frame, y2))

            detections.append((class_id, score, (x1, y1, x2, y2)))

    return detections

def postprocess_yolo_output3(output, frame_shape, input_shape, conf_threshold=0.25, iou_threshold=0.45):
    """
    Process YOLOv10 model output for bounding boxes, classes, and scores.
    """
    detections = []
    h_frame, w_frame = frame_shape[:2]
    h_input, w_input = input_shape

    # Assume `output` is a numpy array of shape (N, 85), where:
    # - N: Number of detected objects
    # - 85: [x_center, y_center, width, height, obj_confidence, class_probs...]

    # Ensure output is a 2D array
    output = np.squeeze(output)  # Remove extra dimensions, if any

    for det in output:
        # Object confidence is at index 4
        obj_confidence = det[4]
        if obj_confidence < conf_threshold:
            continue  # Skip low-confidence detections

        # Class probabilities start at index 5
        class_probs = det[5:]
        class_id = np.argmax(class_probs)
        confidence = class_probs[class_id]

        if confidence > conf_threshold:
            # Rescale bounding box coordinates to original frame size
            x_center, y_center, width, height = det[:4]
            x_center *= w_frame / w_input
            y_center *= h_frame / h_input
            width *= w_frame / w_input
            height *= h_frame / h_input

            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            detections.append((class_id, confidence, (x1, y1, x2, y2)))

    return detections

def postprocess_yolo_output(output, frame_shape, input_shape, conf_threshold=0.25, iou_threshold=0.45):
    """
    Process YOLOv10 model output for bounding boxes, classes, and scores.
    """
    detections = []
    h_frame, w_frame = frame_shape[:2]
    h_input, w_input = input_shape
    # scale = min(w_input / w_frame, h_input / h_frame)
    scale = min(w_frame / w_input, h_frame / h_input)
    new_width = int(w_frame * scale)
    new_height = int(h_frame * scale)
    pad_x = (w_input - new_width) // 2
    pad_y = (h_input - new_height) // 2

    # Assume `output` is a numpy array of shape (N, 85), where:
    # - N: Number of detected objects
    # - 85: [x_center, y_center, width, height, obj_confidence, class_probs...]

    # Ensure output is a 2D array
    output = np.squeeze(output)  # Remove extra dimensions, if any

    for det in output:
        # Object confidence is at index 4
        obj_confidence = det[4]
        if obj_confidence < conf_threshold:
            continue  # Skip low-confidence detections

        # Class probabilities start at index 5
        class_probs = det[5:]
        class_id = np.argmax(class_probs)
        confidence = class_probs[class_id]
        left, top, right, bottom = det[:4]

        if confidence > conf_threshold:
            # Rescale bounding box coordinates to original frame size
            left = (left - pad_x) / scale
            top = (top - pad_y) / scale
            right = (right - pad_x) / scale
            bottom = (bottom - pad_y) / scale

            x = int(left)
            y = int(top)
            width = int(right - left)
            height = int(bottom - top)
            x1 = x
            y1 = y
            x2 = x + width
            y2 = y + height
            detections.append((class_id, confidence, (x1, y1, x2, y2)))

    return detections

# Draw detections on the frame
def draw_detections(frame, detections, labels, query_id):
    count = 0
    for class_id, confidence, bbox in detections:
        if query_id == -1 or class_id == query_id:  # If query_id is -1, include all classes
            count += 1
            x1, y1, x2, y2 = bbox
            label = f"{labels[class_id]}: {confidence:.2f}"
            print(f"Class ID: {class_id}, Confidence: {confidence}, BBox: {bbox}")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame, count


# Main loop for processing RTSP stream
def process_rtsp_stream(rtsp_url, session, input_shape, labels, query_id):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Unable to connect to the RTSP stream")
        return

    last_processed_time = 0  # Tracks the last frame's processing time
    process_interval = 1  # Process one frame every 5 seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Unable to read frame or stream ended")
            break

        curr_time = time.time()

        # Skip frames unless the interval has passed
        if curr_time - last_processed_time < process_interval:
            # Optionally, show the current frame without processing
            cv2.imshow('RTSP Stream', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        last_processed_time = curr_time

        # Preprocess input and run inference
        input_tensor = preprocess_yolo_input(frame, input_shape)
        output = session.run(None, {"images": input_tensor})[0]  # Assuming the input name is 'images'

        # Postprocess output
        detections = postprocess_yolo_output(output, frame.shape, input_shape, conf_threshold=0.25, iou_threshold=0.45)

        # Draw results and count detections
        frame, count = draw_detections(frame, detections, labels, query_id)

        # Display query, count, and FPS on the frame
        query_text = f"Query: {'All' if query_id == -1 else labels[query_id]}"
        count_text = f"Count: {count}"
        timestamp_text = f"Timestamp: {time.strftime('%H:%M:%S')}"
        cv2.putText(frame, query_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, count_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, timestamp_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Show the processed frame
        cv2.imshow('RTSP Stream', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model_path = "F:/source/yolov10/yolov10n.onnx"
    rtsp_url = "rtsp://192.168.31.120:8554/oclea-stream1"
    input_shape = (640, 640)  # YOLOv10 expects 640x640 input
    labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light"]  # Replace with your model's labels
    query_id = -1  # Use -1 to include all classes, or specify a class ID

    yolo_session = initialize_yolo_model(model_path)
    process_rtsp_stream(rtsp_url, yolo_session, input_shape, labels, query_id)