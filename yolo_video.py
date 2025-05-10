import numpy as np
import cv2
import os

def get_output_layers(net):
    layer_names = net.getLayerNames()
    return [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

def process_video(input_path, output_path, confidence=0.2, threshold=0.1):
    # Kiểm tra đường dẫn file YOLO
    labelsPath = os.path.sep.join(['yolo-coco', "coco.names"])
    weightsPath = os.path.sep.join(['yolo-coco', "yolov3.weights"])
    configPath = os.path.sep.join(['yolo-coco', "yolov3.cfg"])

    for path in [labelsPath, weightsPath, configPath]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File không tồn tại: {path}")

    # Load COCO class labels
    LABELS = open(labelsPath).read().strip().split("\n")

    # Initialize colors
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    # Load YOLO
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # Initialize video stream
    vs = cv2.VideoCapture(input_path)
    if not vs.isOpened():
        raise ValueError(f"Không thể mở video: {input_path}")

    # Lấy thông số video đầu vào
    fps = int(vs.get(cv2.CAP_PROP_FPS)) or 30
    width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Cố định kích thước đầu ra
    output_width, output_height = 640, 480

    writer = None
    detections = []

    print(f"Processing video: {input_path}, FPS: {fps}, Input Size: {width}x{height}, Output Size: {output_width}x{output_height}")

    # Loop over frames
    frame_count = 0
    while True:
        (grabbed, frame) = vs.read()
        if not grabbed:
            break
        frame_count += 1

        # Create blob and perform forward pass
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (608, 608), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(get_output_layers(net))

        # Initialize lists
        boxes = []
        confidences = []
        classIDs = []

        # Process detections
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                conf = scores[classID]
                if conf > confidence:
                    box = detection[0:4] * np.array([width, height, width, height])
                    (centerX, centerY, w, h) = box.astype("int")
                    x = int(centerX - (w / 2))
                    y = int(centerY - (h / 2))
                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(conf))
                    classIDs.append(classID)

        # Apply non-maxima suppression
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

        # Draw bounding boxes and collect detections
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                detection = {"label": LABELS[classIDs[i]], "confidence": float(confidences[i])}
                detections.append(detection)
                print(f"Frame {frame_count}: Added detection: {detection}")

        # Resize frame to output size
        frame = cv2.resize(frame, (output_width, output_height))

        # Initialize video writer
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            writer = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
            if not writer.isOpened():
                raise ValueError(f"Không thể tạo file video đầu ra: {output_path}")

        # Write frame
        writer.write(frame)

    # Clean up
    writer.release()
    vs.release()

    print(f"Total frames processed: {frame_count}")
    print(f"Raw detections (count: {len(detections)}): {detections}")

    # Loại bỏ trùng lặp dựa trên label và confidence
    unique_detections = []
    seen = set()
    for d in detections:
        key = (d["label"], round(d["confidence"], 4))
        if key not in seen:
            unique_detections.append(d)
            seen.add(key)

    print(f"Final unique detections (count: {len(unique_detections)}): {unique_detections}")
    return unique_detections