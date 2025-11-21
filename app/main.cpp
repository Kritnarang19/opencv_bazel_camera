#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <chrono>
#include <ctime>
#include <string>

using namespace cv;
using namespace std;

// Generate timestamp filenames
string timeStamp() {
    auto now = chrono::system_clock::now();
    time_t t = chrono::system_clock::to_time_t(now);
    tm *lt = localtime(&t);

    char buffer[80];
    strftime(buffer, 80, "image_%Y-%m-%d_%H-%M-%S", lt);
    return string(buffer);
}

// Generate video filename
string videoStamp() {
    auto now = chrono::system_clock::now();
    time_t t = chrono::system_clock::to_time_t(now);
    tm *lt = localtime(&t);

    char buffer[80];
    strftime(buffer, 80, "video_%Y-%m-%d_%H-%M-%S.mp4", lt);
    return string(buffer);
}

int main() {

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Cannot open camera!" << endl;
        return -1;
    }

    Mat frame, gray, blurImg, edgeImg;

    // ============= LOAD ALL CASCADES ===================
    CascadeClassifier faceCascade, eyeCascade, smileCascade, fullBodyCascade;

    bool faceLoaded  = faceCascade.load("haarcascade_frontalface_default.xml");
    bool eyeLoaded   = eyeCascade.load("haarcascade_eye.xml");
    bool smileLoaded = smileCascade.load("haarcascade_smile.xml");
    bool bodyLoaded  = fullBodyCascade.load("haarcascade_fullbody.xml");

    if (!faceLoaded)  cerr << "Face cascade missing!" << endl;
    if (!eyeLoaded)   cerr << "Eye cascade missing!" << endl;
    if (!smileLoaded) cerr << "Smile cascade missing!" << endl;
    if (!bodyLoaded)  cerr << "Full-body cascade missing!" << endl;

    // ============= UI INSTRUCTIONS =====================
    cout << "--------------- CAMERA APPLICATION ---------------\n";
    cout << "c : Capture Image\n";
    cout << "v : Start/Stop Video Recording\n";
    cout << "f : Toggle Face Detection\n";
    cout << "e : Toggle Eye Detection\n";
    cout << "s : Toggle Smile Detection\n";
    cout << "b : Toggle Full Body Detection\n";
    cout << "1 : Grayscale Filter\n";
    cout << "2 : Blur Filter\n";
    cout << "3 : Edge Detection\n";
    cout << "o : Original Frame\n";
    cout << "q : Quit\n";
    cout << "--------------------------------------------------\n";

    // Toggle states
    bool faceOn = false;
    bool eyeOn  = false;
    bool smileOn = false;
    bool bodyOn = false;

    int filterMode = 0;

    // ============= VIDEO RECORDING VARIABLES =============
    bool recording = false;
    VideoWriter writer;
    int fourcc = VideoWriter::fourcc('m','p','4','v');  // codec for mp4
    double fps = 20.0;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        auto start = chrono::high_resolution_clock::now();
        Mat display = frame.clone();

        // ============= FILTERS ===================
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

        // Convert to gray for all detectors
        Mat grayFrame;
        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

        // ============= FULL BODY DETECTION ===================
        if (bodyOn && bodyLoaded) {
            vector<Rect> bodies;
            fullBodyCascade.detectMultiScale(grayFrame, bodies, 1.1, 3);

            for (auto &body : bodies) {
                rectangle(display, body, Scalar(255, 0, 0), 2);  // Blue
            }
        }

        // ============= FACE DETECTION ===================
        vector<Rect> faces;
        if (faceOn && faceLoaded) {
            faceCascade.detectMultiScale(
                grayFrame, faces,
                1.1, 5, 0,
                Size(60, 60),
                Size()
            );

            for (auto &face : faces) {
                rectangle(display, face, Scalar(0, 255, 0), 2); // Green
            }
        }

        // ============= EYE DETECTION ===================
        if (eyeOn && eyeLoaded) {
            for (auto &face : faces) {
                Mat faceROI = grayFrame(face);
                vector<Rect> eyes;
                eyeCascade.detectMultiScale(faceROI, eyes, 1.1, 4);

                for (auto &eye : eyes) {
                    Rect eyeBox(face.x + eye.x, face.y + eye.y, eye.width, eye.height);
                    rectangle(display, eyeBox, Scalar(0, 255, 255), 2); // Yellow
                }
            }
        }

        // ============= SMILE DETECTION ===================
        if (smileOn && smileLoaded) {
            for (auto &face : faces) {
                Mat faceROI = grayFrame(face);
                vector<Rect> smiles;
                smileCascade.detectMultiScale(faceROI, smiles, 1.2, 40);

                for (auto &smile : smiles) {
                    Rect smileBox(face.x + smile.x, face.y + smile.y, smile.width, smile.height);
                    rectangle(display, smileBox, Scalar(255, 0, 255), 2); // Pink
                }
            }
        }

        // ============= FPS DISPLAY ===================
        auto end = chrono::high_resolution_clock::now();
        double fpsCounter = 1000.0 / chrono::duration_cast<chrono::milliseconds>(end - start).count();

        putText(display, "FPS: " + to_string((int)fpsCounter),
                Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8,
                Scalar(0, 255, 255), 2);

        // ========== RECORDING STATUS =========
        string text = recording ? "RECORDING..." : "REC OFF";
        Scalar color = recording ? Scalar(0, 0, 255) : Scalar(0, 255, 0);

        putText(display, text, Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.7, color, 2);

        imshow("Camera Application", display);

        // ============= WRITE VIDEO IF RECORDING ============
        if (recording && writer.isOpened()) {
            writer.write(frame);
        }

        // ============= KEY CONTROLS ===================
        char key = (char)waitKey(1);

        if (key == 'q') break;

        if (key == 'c')
            imwrite(timeStamp() + ".jpg", frame);

        // ==== Start/Stop video recording ====
        if (key == 'v') {
            if (!recording) {
                string filename = videoStamp();
                cout << "Recording started: " << filename << endl;

                writer.open(filename, fourcc, fps,
                            Size(frame.cols, frame.rows), true);

                if (!writer.isOpened()) {
                    cerr << "Error: Could not open VideoWriter!" << endl;
                } else {
                    recording = true;
                }
            } else {
                cout << "Recording stopped.\n";
                recording = false;
                writer.release();
            }
        }

        if (key == 'f') faceOn = !faceOn;
        if (key == 'e') eyeOn = !eyeOn;
        if (key == 's') smileOn = !smileOn;
        if (key == 'b') bodyOn = !bodyOn;

        if (key == '1') filterMode = 1;
        if (key == '2') filterMode = 2;
        if (key == '3') filterMode = 3;
        if (key == 'o') filterMode = 0;
    }

    cap.release();
    writer.release();
    destroyAllWindows();
    return 0;
}
