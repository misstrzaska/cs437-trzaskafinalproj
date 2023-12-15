# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run the object detection routine."""
import argparse
import sys
import time
import csv

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils


def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    enable_edgetpu: True/False whether the model is a EdgeTPU model.
  """

  # Variables to calculate FPS
  counter, fps = 0, 0
  cellcounter = 0
  start_time = time.time()

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  # Initialize the object detection model
  base_options = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
  detection_options = processor.DetectionOptions(
      max_results=3, score_threshold=0.3)
  options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)

  #Generate new csv file (differs by date and time) each time running the program
  timestamp_str = time.strftime("%Y%m%d_%H%M%S")
  csv_file_path = f'obj_detect{timestamp_str}.csv'
    
  with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Timestamp', 'Object Detected'])

  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

    counter += 1
    image = cv2.flip(image, 1)

    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a TensorImage object from the RGB image.
    input_tensor = vision.TensorImage.create_from_array(rgb_image)

    # Run object detection estimation using the model.
    detection_result = detector.detect(input_tensor)
    
    for detection in detection_result.detections:
          #Output the predicted object and probability to the terminal
          category = detection.categories[0]
          category_name = category.category_name
          probability = round(category.score, 2)

        # Draw keypoints and edges on input image
        #image = utils.visualize(image, detection_result)

          # Calculate the FPS
          if counter % fps_avg_frame_count == 0:
            end_time = time.time()
            fps = fps_avg_frame_count / (end_time - start_time)
            start_time = time.time()

          # Show the FPS
          fps_text = 'FPS = {:.1f}'.format(fps)
          #check if category name is cell phone and percentage is greater than 30
          percentage = probability * 100
          if category_name == "cell phone" and percentage > 30:
            print(f"Detected Object: {category_name}, Probability: {probability}, FPS: {fps_text}")
            text_location = (left_margin, row_size)
            cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                      font_size, text_color, font_thickness)
            #Get current timestamp when object is detected in format of month, day, year, hours, min, sec
            timestamp = time.strftime("%m/%d/%Y %H:%M:%S")
            #increase cellcounter by 1 when cell phone detected
            cellcounter += 1
            #Write to csv file, append on a new line
            with open(csv_file_path, 'a', newline='') as csvfile:
              csv_writer = csv.writer(csvfile)
              #timestamp written, as well as 1 for cellphone
              csv_writer.writerow([timestamp, 1])
            #print that a cell phone was detected, at what time stamp, and what the total cell phone count is at the time
            print(f"Detected a cell phone at time: {timestamp} Total cell phone count: {cellcounter}")
          #in the case that a cell phone was NOT detected  
          else:
            #Get current timestamp when object is detected in format of month, day, year, hours, min, sec
            timestamp = time.strftime("%m/%d/%Y %H:%M:%S")  
            #Write to csv file, append on a new line
            with open(csv_file_path, 'a', newline='') as csvfile:
              csv_writer = csv.writer(csvfile)
              #for better visualization, have timestamp written, as well as 0 for not cell phone detected
              csv_writer.writerow([timestamp, 0])
            #print that a cell phone was detected, at what time stamp, and what the total cell phone count is at the time
            print(f"Did not detect a cell phone at time: {timestamp}")
          #delay to achieve the desired fps
          time.sleep(0.8)
    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    #actually show the image of object detection
    #cv2.imshow('object_detector', image)
    
  cap.release()
  cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='efficientdet_lite0.tflite')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  args = parser.parse_args()

  run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU))


if __name__ == '__main__':
  main()
