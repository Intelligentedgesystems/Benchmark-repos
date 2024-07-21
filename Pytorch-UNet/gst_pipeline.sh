#!/bin/bash

# Set the model path, input image path, and output image path
MODEL_PATH="model.tflite"
INPUT_IMAGE="input.jpg"
OUTPUT_IMAGE="output.png"

# Define the desired dimensions for the resized image
NEW_WIDTH=256
NEW_HEIGHT=256

# Run the GStreamer pipeline
gst-launch-1.0 \
  filesrc location=$INPUT_IMAGE ! \
  decodebin ! \
  videoconvert ! \
  videoscale ! \
  video/x-raw,width=$NEW_WIDTH,height=$NEW_HEIGHT ! \
  videoconvert ! \
  tensor_converter ! \
  tensor_filter framework=tensorflow-lite model=$MODEL_PATH ! \
  videoconvert ! \
  pngenc ! \
  filesink location=$OUTPUT_IMAGE
