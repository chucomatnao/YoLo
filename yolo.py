import numpy as np
import cv2
import os

def process_image(image_path, output_path, confidence=0.5, threshold=0.3):
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

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")
    (H, W) = image.shape[:2]

    # Get YOLO output layers
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    # Create blob and perform forward pass
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

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
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(conf))
                classIDs.append(classID)

    # Apply non-maxima suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

    # Initialize detections list
    detections = []

    # Draw bounding boxes and collect detections
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            detections.append({"label": LABELS[classIDs[i]], "confidence": float(confidences[i])})

    # Save output image
    if not cv2.imwrite(output_path, image):
        raise ValueError(f"Không thể lưu ảnh đầu ra: {output_path}")

    print(f"Image detections (count: {len(detections)}): {detections}")  # Debug
    return detections