import cv2
import numpy as np

# Load YOLO
# The 'yolov3.weights' file contains the pre-trained weights
# The 'yolov3.cfg' file contains the configuration for the YOLO model
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Get the names of all the layers in the network
layer_names = net.getLayerNames()

# Get the output layer names that we need from YOLO
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load the COCO class names, which includes 'car'
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Load the video file
cap = cv2.VideoCapture('video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Create a 4D blob from a frame
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    
    # Run the forward pass to get output of the output layers
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    # For each detection from each output layer, get the confidence, class id, bounding box params
    # If the confidence is greater than a threshold, save it as a detected object
    for out in outs:
        for detection in out:
            scores = detection[5:]  # Skip the first 5 elements and get the rest
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == 'car':  # Only detect cars
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression to remove overlapping bounding boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes around detected objects
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)  # Green bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)

            # Basic cut-in detection: check if the car is in the lower half of the frame
            # This is a simplified example and might need to be more sophisticated
            if y + h > height / 2:
                cv2.putText(frame, "Cut-in Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the output frame
    cv2.imshow('Image', frame)
    
    # Exit when 'ESC' key is pressed
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()# Vehicle-cut-in-detection-
It will detect the vehicle when it's taken diversion, it will prevent the accident and traffic in roads 
