✨ Face Detection & Crowd Management System

AI-powered application that detects faces, counts people, and estimates crowd levels using YOLOv8 and your custom-trained Faster R-CNN model.

📌 Project Description

The Face Detection & Crowd Management System is a complete end-to-end AI solution that identifies faces in images or webcam snapshots and determines overall crowd density. It blends real-time performance from YOLOv8 with the precision of your own Faster R-CNN model trained on FDDB.

This tool is ideal for:

Public surveillance

Event management

School/college monitoring

Smart attendance systems

Safety & occupancy control

Built using Streamlit, it offers a clean, interactive UI that works on any device.

🚀 Features
🔍 Face Detection

Detect faces using:

YOLOv8 (Fast) → real-time, best for large crowds

Faster R-CNN (Your Model) → custom-trained, accurate

👥 People Counting

Automatically counts the number of detected faces.

🚦 Crowd Level Estimation

Classifies the crowd into:

Empty

Low

Medium

High

(using adjustable thresholds)

🎥 Webcam Snapshot Support

Capture an image directly from your webcam and run detection.

📂 Image Upload Support

Upload photos for detection and crowd analysis.

💾 Download Results

Save the processed image with bounding boxes.

🎨 Streamlit UI

Dark-themed, clean, and responsive interface.

🖼️ Sample Output

Real detection with face count and crowd level.

(Your screenshot: 8 faces detected → Medium crowd)

🛠️ Tech Stack

Frontend: Streamlit

Backend: Python 3.11

Models: YOLOv8 + Faster R-CNN

Libraries:

PyTorch / TorchVision

Ultralytics YOLO

NumPy

OpenCV

Pillow

📦 Installation Guide
Step 1: Clone the Repository
git clone https://github.com/<your-username>/<your-repo>.git
cd face-detection-project

Step 2: Create Virtual Environment

Windows

python -m venv .venv
.venv\Scripts\activate


macOS/Linux

python3 -m venv .venv
source .venv/bin/activate

Step 3: Install Required Packages
pip install -r requirements.txt


Your requirements.txt includes:

torch
torchvision
ultralytics
opencv-python
numpy
pillow
streamlit
pycocotools
albumentations
matplotlib
tqdm

Step 4: Set Up the Dataset (FDDB)

Download FDDB images

Place them in:

data/fddb/originalPics/


Ensure ellipseList.txt exists

Convert annotations:

python src/fddb_to_coco.py

Step 5: Train Faster R-CNN (Optional)
python -m src.train


Model checkpoint saved at:

outputs/checkpoints/facercnn.pth

Step 6: Run the App
streamlit run src/face_detection_app.py

🧠 How It Works

Upload or capture an image

Select model:

YOLOv8 → Fast

Faster R-CNN → Trained model

App detects faces

Counts total people

Classifies the crowd level

Displays results + download option

📈 Future Enhancements

Live video feed with FPS

Real-time crowd alerts

Logging crowd history to CSV

Line charts for occupancy monitoring

Integration with CCTV systems

Age/Gender prediction

Deployment on cloud (Streamlit Cloud / AWS / Hugging Face)

👤 Author

Anirudh Nair
AI Developer | ML Engineer
https://www.linkedin.com/in/anirudh071/
