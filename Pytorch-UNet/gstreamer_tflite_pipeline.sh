#!/bin/bash

# Check if required arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <model.tflite> <input video> [<output video>]"
    exit 1
fi

MODEL=$1
INPUT_VIDEO=$2
OUTPUT_VIDEO=${3:-output.mp4}

# Set the environment variable for the TFLite model
export GST_PLUGIN_PATH=/usr/lib/gstreamer-1.0

# GStreamer pipeline to read video, apply the TFLite model for segmentation, and write the output video
gst-launch-1.0 \
    filesrc location=$INPUT_VIDEO ! \
    decodebin ! \
    videoconvert ! \
    videoscale ! \
    video/x-raw,format=RGB ! \
    tensor_converter ! \
    tensor_filter framework=tensorflow-lite model=$MODEL ! \
    tensor_sink name=tensor_result ! \
    queue ! \
    tensor_converter ! \
    videoscale ! \
    videoconvert ! \
    x264enc ! \
    mp4mux ! \
    filesink location=$OUTPUT_VIDEO
