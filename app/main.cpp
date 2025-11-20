#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <chrono>
#include <ctime>

using namespace cv;
using namespace std;

// Generate timestamp-based filename
string timeStamp() {
    auto now = chrono::system_clock::now();
    time_t t = chrono::system_clock::to_time_t(now);
    tm *lt = localtime(&t);

    char buffer[80];
    strftime(buffer, 80, "image_%Y-%m-%d_%H-%M-%S.jpg", lt);
    return string(buffer);
}

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Cannot open camera!" << endl;
        return -1;
    }

    Mat frame, gray, blurImg, edgeImg;
    CascadeClassifier faceCascade;

    // ⭐ Crash-Proof Loading
    bool cascadeLoaded = faceCascade.load("haarcascade_frontalface_default.xml");
    if (!cascadeLoaded) {
        cerr << "Error: Could not load Haar Cascade file!" << endl;
        cerr << "Make sure haarcascade_frontalface_default.xml is in the SAME folder." << endl;
    }

    cout << "--------------- CAMERA APPLICATION ---------------\n";
    cout << "c : Capture Image\n";
    cout << "f : Toggle Face Detection ON/OFF\n";
    cout << "1 : Grayscale Filter\n";
    cout << "2 : Blur Filter\n";
    cout << "3 : Edge Detection\n";
    cout << "o : Original Frame\n";
    cout << "q : Quit\n";
    cout << "--------------------------------------------------\n";

    bool faceDetectionEnabled = false;
    int filterMode = 0; // 0=original, 1=gray, 2=blur, 3=edge

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Start timer for FPS
        auto start = chrono::high_resolution_clock::now();

        Mat display = frame.clone();

        // -------------------- FILTERS ---------------------
        if (filterMode == 1) {
            cvtColor(display, display, COLOR_BGR2GRAY);
            cvtColor(display, display, COLOR_GRAY2BGR);
        } 
        else if (filterMode == 2) {
            GaussianBlur(display, display, Size(15, 15), 0);
        } 
        else if (filterMode == 3) {
            Canny(display, edgeImg, 50, 150);
            cvtColor(edgeImg, display, COLOR_GRAY2BGR);
        }

        // ---------------- FACE DETECTION ON/OFF -----------------
        if (faceDetectionEnabled) {
            if (!cascadeLoaded || faceCascade.empty()) {
                putText(display, "Face Detection: ERROR (Cascade not loaded)",
                        Point(10, 80), FONT_HERSHEY_SIMPLEX, 0.6,
                        Scalar(0, 0, 255), 2);
            } 
            else {
                vector<Rect> faces;
                Mat grayFace;
                cvtColor(frame, grayFace, COLOR_BGR2GRAY);

                faceCascade.detectMultiScale(grayFace, faces);

                for (auto &face : faces) {
                    rectangle(display, face, Scalar(0, 255, 0), 2);
                }
            }
        }

        // --------------------- STATUS TEXT ------------------------
        auto end = chrono::high_resolution_clock::now();
        double fps = 1000.0 / chrono::duration_cast<chrono::milliseconds>(end - start).count();

        putText(display, "FPS: " + to_string((int)fps),
                Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8,
                Scalar(0, 255, 255), 2);

        // status of face detection (On / Off)
        string fdStatus = faceDetectionEnabled ? "Face Detection: ON" : "Face Detection: OFF";
        Scalar fdColor = faceDetectionEnabled ? Scalar(0, 255, 0) : Scalar(0, 0, 255);

        putText(display, fdStatus, Point(10, 60),
                FONT_HERSHEY_SIMPLEX, 0.7, fdColor, 2);

        // --------------------- DISPLAY ------------------------
        imshow("Camera Application", display);

        char key = (char)waitKey(1);

        if (key == 'q') break;

        if (key == 'c') {
            string filename = timeStamp();
            imwrite(filename, frame);
            cout << "Saved: " << filename << endl;
        }

        // ⭐ Toggle ON/OFF face detection
        if (key == 'f') {
            faceDetectionEnabled = !faceDetectionEnabled;
        }

        // Filters
        if (key == '1') filterMode = 1;
        if (key == '2') filterMode = 2;
        if (key == '3') filterMode = 3;
        if (key == 'o') filterMode = 0;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
