from ultralytics import YOLO
import cv2
import os
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_fps():
    """Calculate FPS using a moving average"""
    return getattr(get_fps, 'fps', 0)

def update_fps(frame_time):
    """Update FPS calculation"""
    if not hasattr(get_fps, 'times'):
        get_fps.times = []
    
    get_fps.times.append(frame_time)
    if len(get_fps.times) > 30:  # Keep only last 30 frames for moving average
        get_fps.times.pop(0)
    
    get_fps.fps = len(get_fps.times) / sum(get_fps.times)

def process_input(input_path=None, input_type='video', save_output=True, conf_thresh=0.45, model_path=None):
    """
    Process different types of inputs for drone detection
    Args:
        input_path: Path to input file or folder
        input_type: 'video', 'image', or 'webcam'
        save_output: Whether to save detection results
        conf_thresh: Confidence threshold for detections (0-1)
        model_path: Custom path to YOLO model weights
    """
    # Load model
    if model_path is None:
        model_path = os.path.join('training_output', 'drone_detector', 'weights', 'best.pt')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    
    logging.info(f"Loading model from {model_path}")
    model = YOLO(model_path)
    
    # Initialize detection statistics
    stats = {
        'total_frames': 0,
        'detected_objects': 0,
        'avg_confidence': 0.0
    }
    
    # Set up output paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = os.path.join('output')
    ensure_dir(output_base)

    if input_type == 'image':
        # Process single image
        if input_path is None:
            raise ValueError("Please provide input image path")
        
        # Read image
        frame = cv2.imread(input_path)
        if frame is None:
            raise ValueError(f"Could not read image: {input_path}")

        # Run inference
        results = model.predict(
            source=frame,
            conf=0.45,
            show=False
        )

        # Process and save results
        for r in results:
            annotated_frame = r.plot()
            if save_output:
                output_path = os.path.join(output_base, 'images', f'detection_{timestamp}.jpg')
                cv2.imwrite(output_path, annotated_frame)
                print(f"Saved detection result to: {output_path}")

            # Display the frame
            cv2.imshow('Drone Detection', annotated_frame)
            cv2.waitKey(0)

        cv2.destroyAllWindows()

    else:  # video or webcam
        # Set up video capture
        cap = cv2.VideoCapture(0 if input_type == 'webcam' else input_path)
        
        # Check if video opened successfully
        if not cap.isOpened():
            print("Error: Could not open video source")
            return

        # Get video properties for output
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Set up video writer if saving output
        out = None
        if save_output and input_type != 'webcam':
            output_path = os.path.join(output_base, 'videos', f'detection_{timestamp}.mp4')
            out = cv2.VideoWriter(output_path, 
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                fps, (frame_width, frame_height))
            print(f"Saving video output to: {output_path}")

        cv2.namedWindow('Drone Detection', cv2.WINDOW_NORMAL)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            # Run inference on frame
            results = model.predict(
                source=frame,
                conf=0.45,
                show=False,  # Don't show automatically
                stream=True  # Enable streaming mode
            )
            
            # Process results
            for r in results:
                annotated_frame = r.plot()
                
                # Save frame if video output is enabled
                if out is not None:
                    out.write(annotated_frame)

                # Display the frame
                cv2.imshow('Drone Detection', annotated_frame)
                
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        if out is not None:
            out.write(annotated_frame)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage:
    
    # For webcam
    # process_input(input_type='webcam')
    
    # For video file
    process_input(
        input_path="/Users/parvashah/Desktop/DS Projects/Drone Detection System/Testing Videos/27363-363524166_medium.mp4",
        input_type='video',
        save_output=True
    )
    
    # For image file
    # process_input(
    #     input_path="path_to_image.jpg",
    #     input_type='image',
    #     save_output=True
    # )