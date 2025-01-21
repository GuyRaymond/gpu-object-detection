# Use a base image with OpenCV and CUDA support
FROM guyraymond/opencv-cuda-cudnn:latest

# Set working directory
WORKDIR /workspace

# Copy project files to the container
COPY . .

# Create models directory and download YOLOv4 model files
RUN mkdir -p models && \
    wget -q --show-progress https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg?raw=true -O models/yolov4.cfg && \
    wget -q --show-progress https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true -O models/coco.names && \
    wget -q --show-progress https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4.weights -O models/yolov4.weights


# Create build directory and compile the C++ project
RUN mkdir -p build && \
    cmake -S . -B build && \
    cmake --build build --config Release

# Ensure the executable exists
RUN test -f build/gpu_object_detection || { echo "Build failed or gpu_object_detection executable not found"; exit 1; }

# Command to run the executable
CMD ["./build/gpu_object_detection"]
