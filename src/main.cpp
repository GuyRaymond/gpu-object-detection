#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream> // Add this line to include the fstream header

// Function to draw bounding boxes and labels on the image
void drawPrediction(cv::Mat& frame, int classId, float confidence, int left, int top, int right, int bottom, const std::vector<std::string>& classes) {
    // Draw a rectangle around the detected object
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 255, 0), 2);

    // Get the label for the class name and its confidence
    std::string label = cv::format("%s: %.2f", classes[classId].c_str(), confidence);

    // Display the label at the top of the bounding box
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = std::max(top, labelSize.height);
    cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
}

int main() {
    // Load the input image
    cv::Mat inputImage = cv::imread("images/input.jpg");
    if (inputImage.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return -1;
    }

    // Load the YOLOv3 model
    std::string modelConfiguration = "models/yolov4.cfg";
    std::string modelWeights = "models/yolov4.weights";
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    // Load class names
    std::vector<std::string> classes;
    std::string classesFile = "models/coco.names";
    std::ifstream ifs(classesFile.c_str()); // This line requires <fstream>
    std::string line;
    while (std::getline(ifs, line)) {
        classes.push_back(line);
    }

    // Prepare the image for YOLO
    cv::Mat blob;
    cv::dnn::blobFromImage(inputImage, blob, 1 / 255.0, cv::Size(416, 416), cv::Scalar(0, 0, 0), true, false);
    net.setInput(blob);

    // Forward pass to get detections
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    // Parameters for filtering detections
    float confidenceThreshold = 0.6; // Confidence threshold
    float nmsThreshold = 0.4;       // Non-maximum suppression threshold
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // Process detections
    for (const auto& output : outputs) {
        for (int i = 0; i < output.rows; i++) {
            const float* detection = output.ptr<float>(i);
            cv::Mat scores = output.row(i).colRange(5, output.cols);
            cv::Point classIdPoint;
            double confidence;
            cv::minMaxLoc(scores, nullptr, &confidence, nullptr, &classIdPoint);

            if (confidence > confidenceThreshold) {
                int centerX = static_cast<int>(detection[0] * inputImage.cols);
                int centerY = static_cast<int>(detection[1] * inputImage.rows);
                int width = static_cast<int>(detection[2] * inputImage.cols);
                int height = static_cast<int>(detection[3] * inputImage.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back(static_cast<float>(confidence));
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }

    // Apply non-maximum suppression to remove overlapping boxes
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confidenceThreshold, nmsThreshold, indices);

    // Draw the final detections
    for (int idx : indices) {
        cv::Rect box = boxes[idx];
        drawPrediction(inputImage, classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, classes);
    }

    // Save the output image
    std::string outputPath = "images/output.jpg";
    if (!cv::imwrite(outputPath, inputImage)) {
        std::cerr << "Failed to save the output image!" << std::endl;
        return -1;
    }

    std::cout << "Object detection completed and output saved to " << outputPath << std::endl;

    return 0;
}
