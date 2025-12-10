import cv2
import numpy as np
import onnxruntime as ort
import time

# --- Configuration ---
# Your specific class names, as defined in your training.
# This list MUST be in the correct order (matching the model's output indices 0, 1, 2, 3).
CLASS_NAMES = [
    'Book',           
    'Newspaper',       
    'Old School Bag',  
    'Zip-top can'     
]
# Filepath to your ONNX model
MODEL_PATH = "best.onnx"

# Input size used during ONNX export. The model expects 416x416.
INPUT_WIDTH = 416
INPUT_HEIGHT = 416

# Confidence threshold for detections (0.25 is standard)
CONF_THRESHOLD = 0.25
# Non-Max Suppression threshold (0.45 is standard)
NMS_THRESHOLD = 0.45
# The corrected provider name
PROVIDER = 'CPUExecutionProvider'

# --- Initialization Functions ---

def load_onnx_session(model_path, provider):
    """Initializes the ONNX Runtime Inference Session."""
    print(f"Loading ONNX model from: {model_path} with provider: {provider}...")
    try:
        # We specify the correct provider name here
        session = ort.InferenceSession(
            model_path, 
            providers=[provider]
        )
        return session
    except Exception as e:
        print(f"FATAL ERROR: Could not load ONNX session. Ensure {model_path} is correct.")
        print(f"Details: {e}")
        return None

def process_frame(session, frame):
    """Pre-processes the frame and runs inference."""
    
    # 1. Pre-process: Resize, BGR to RGB, Normalize, Transpose
    input_frame = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
    
    # Normalize (0-255) to (0.0-1.0) and then transpose to NCHW format (1, 3, 416, 416)
    input_tensor = input_frame.astype(np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0) # Add batch dimension

    # 2. Run Inference
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    
    # Onnx Runtime returns a list of outputs
    outputs = session.run(output_names, {input_name: input_tensor})
    
    # 3. Process Output (Post-processing)
    # The output shape is usually (1, 84, N) where N is the number of predictions. 
    # We transpose it to (N, 84) for easier handling.
    predictions = np.transpose(np.squeeze(outputs[0]))
    
    # Filter by confidence
    scores = np.max(predictions[:, 4:], axis=1) # Max score across all classes
    mask = scores > CONF_THRESHOLD
    predictions = predictions[mask]
    scores = scores[mask]

    # Handle case with no detections (returns three empty lists)
    if len(predictions) == 0:
        return [], [], [] 

    # Get boxes and class IDs
    class_ids = np.argmax(predictions[:, 4:], axis=1)
    
    # Convert center coordinates (cx, cy, w, h) to (x1, y1, x2, y2)
    boxes = predictions[:, :4]
    
    # Scale boxes from 416x416 back to original frame size
    scale_x = frame.shape[1] / INPUT_WIDTH
    scale_y = frame.shape[0] / INPUT_HEIGHT
    
    # Convert cx, cy, w, h to x1, y1, x2, y2
    x_center, y_center, width, height = boxes.T
    x_center *= scale_x
    y_center *= scale_y
    width *= scale_x
    height *= scale_y

    x1 = (x_center - width / 2).astype(int)
    y1 = (y_center - height / 2).astype(int)
    # Calculate box coordinates for NMS (needs x, y, width, height format)
    x = x1.astype(int)
    y = y1.astype(int)
    w = width.astype(int)
    h = height.astype(int)
    
    # Calculate x2, y2 for final output drawing
    x2 = (x1 + w).astype(int)
    y2 = (y1 + h).astype(int)
    
    # Prepare boxes for NMS
    nms_boxes = list(zip(x, y, w, h))
    
    # Apply Non-Max Suppression (NMS)
    # The NMSBoxes function expects the boxes as a list of (x, y, w, h) tuples
    indices = cv2.dnn.NMSBoxes(
        nms_boxes, 
        scores, 
        CONF_THRESHOLD, 
        NMS_THRESHOLD
    )
    
    final_boxes = []
    final_class_ids = []
    final_scores = []
    
    # Format the results
    # indices structure might vary slightly, ensuring it's iterable
    if isinstance(indices, tuple):
        indices = []
    elif indices.ndim == 2:
        indices = indices.flatten()

    for i in indices:
        idx = int(i) # Ensure index is an integer
        final_boxes.append((x1[idx], y1[idx], x2[idx], y2[idx]))
        final_class_ids.append(class_ids[idx])
        final_scores.append(scores[idx])

    return final_boxes, final_class_ids, final_scores

def draw_boxes(frame, boxes, class_ids, scores):
    """Draws bounding boxes and labels on the frame."""
    for box, class_id, score in zip(boxes, class_ids, scores):
        x1, y1, x2, y2 = box
        # Handle potential index error if CLASS_NAMES is shorter than expected
        try:
            label_name = CLASS_NAMES[class_id]
        except IndexError:
            label_name = f"Class {class_id}"
            
        label = f"{label_name}: {score:.2f}"
        
        # Draw box
        color = (0, 255, 0) # Green BGR
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        
        # Draw text
        cv2.putText(frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return frame

# --- Main Application Loop ---

def main():
    # 1. Load the ONNX Session
    session = load_onnx_session(MODEL_PATH, PROVIDER)
    if not session:
        return

    # 2. Initialize Camera
    # Note: The warning about AVCaptureDeviceTypeExternal is specific to 
    # macOS and relates to deprecated system APIs for camera access, but 
    # it usually doesn't prevent OpenCV from working.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # 3. Main Detection Loop
    start_time = time.time()
    frame_count = 0
    
    print("\nStarting live detection. Press 'q' to quit...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Run inference
        # Expects 3 outputs: boxes, class_ids, scores
        boxes, class_ids, scores = process_frame(session, frame)
        
        # Draw results
        frame = draw_boxes(frame, boxes, class_ids, scores)

        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Display FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show the frame
        cv2.imshow("YOLOv8 Live Detector", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 4. Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nDetection closed successfully.")

if __name__ == "__main__":
    main()