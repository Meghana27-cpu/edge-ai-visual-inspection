# 🏭 EdgeInspect AI
### Real-Time Edge-AI Visual Inspection System for Automated Industrial Defect Detection

**EdgeInspect AI** is a computer vision application that uses YOLO26n and Streamlit to detect surface defects in steel images. It provides an interactive interface for uploading images, visualizing predicted defects with bounding boxes, reviewing inspection history, and evaluating model performance.

🔗 **Live Demo:** [Open EdgeInspect AI](https://edge-ai-visual-inspection-s6ccmdbpgfcmutaonm7v8g.streamlit.app/)  
💻 **GitHub Repository:** [edge-ai-visual-inspection](https://github.com/Meghana27-cpu/edge-ai-visual-inspection)

---

## ✨ Features

- **AI-Based Defect Detection:** Detect steel surface defects using a trained YOLO26n model.
- **Image Upload:** Upload steel surface images in JPG, JPEG, PNG, or BMP format.
- **Visual Predictions:** Display detected defects with bounding boxes, class labels, and confidence scores.
- **Adjustable Confidence Threshold:** Control the detection confidence threshold.
- **Inspection Dashboard:** View inspection statistics, recent inspections, and inference time.
- **Inspection History:** Store inspection records in SQLite.
- **Detection Details:** Review available detection information for individual inspections.
- **Export Results:** Download annotated images, inspection history as CSV, and supported inspection reports.
- **Model Evaluation:** Display precision, recall, mAP@50, mAP@50–95, and confidence-threshold experiment results.
- **Performance Monitoring:** Compare CPU and GPU inference performance where supported.

## 🧠 Model and Dataset

**Model:** YOLO26n (Ultralytics)

**Dataset:** NEU-DET — a steel surface defect detection dataset.

The dataset contains six defect categories:

| Class | Description |
|---|---|
| `crazing` | Fine cracks on the steel surface |
| `inclusion` | Foreign material or inclusions |
| `patches` | Patch-like surface defects |
| `pitted_surface` | Small pits or depressions |
| `rolled-in_scale` | Scale pressed into the steel surface |
| `scratches` | Scratch-like surface defects |

The dataset was prepared for object detection by converting annotations to YOLO format and creating training and validation splits.

## 📊 Model Evaluation

The application displays the following previously reported validation metrics:

| Metric | Value |
|---|---:|
| Precision | 74.3% |
| Recall | 68.5% |
| mAP@50 | 77.0% |
| mAP@50–95 | 42.1% |

A separate confidence-threshold experiment is also displayed in the application.

**Note:** The threshold experiment uses the validation dataset and is not an independent test-set evaluation. Performance can vary depending on the confidence threshold and input image.

## 🛠️ Technology Stack

- **Programming Language:** Python
- **Deep Learning:** PyTorch
- **Object Detection:** Ultralytics YOLO26n
- **Computer Vision:** OpenCV
- **Web Interface:** Streamlit
- **Data Processing:** NumPy, Pandas
- **Database:** SQLite
- **Reports:** ReportLab
- **Version Control:** Git and GitHub

## 📁 Project Structure

```text
edge-ai-visual-inspection/
├── app.py
├── train.py
├── benchmark.py
├── analyze_thresholds.py
├── check_weak_classes.py
├── prepare_dataset.py
├── validate_labels.py
├── requirements.txt
├── models/
│   └── best.pt
├── .gitignore
└── README.md
```

The dataset, generated training runs, virtual environments, and temporary outputs are excluded from the Git repository.

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Meghana27-cpu/edge-ai-visual-inspection.git
cd edge-ai-visual-inspection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the application

```bash
streamlit run app.py
```

Streamlit will display a local URL, usually `http://localhost:8501`.

**Note:** GPU acceleration requires compatible hardware, drivers, and PyTorch/CUDA dependencies. On environments without a supported GPU, inference can run on the CPU.

## 🌐 Live Deployment

The application is deployed using Streamlit Community Cloud.

**Live application:**  
https://edge-ai-visual-inspection-s6ccmdbpgfcmutaonm7v8g.streamlit.app/

The cloud deployment uses CPU inference, so performance may differ from the local GPU-enabled environment.

## ⚠️ Limitations

- Predictions depend on the quality of the trained model and input images.
- The model is trained on NEU-DET and may not generalize to all real-world steel surfaces, lighting conditions, or manufacturing environments.
- CPU inference performance differs from GPU inference.
- The application detects defects in uploaded images; it does not independently verify product quality or guarantee defect-free steel.
- Multi-view uploads do not establish that detections across different views represent the same physical defect unless additional tracking or fusion logic is implemented.

## 🔮 Future Improvements

- Improve detection accuracy and recall through further training and data augmentation.
- Evaluate performance on a separate, independent test dataset.
- Optimize inference for edge devices.
- Add camera-based real-time inspection.
- Enhance multi-view defect association and product-level reporting.
- Add more detailed evaluation visualizations and class-wise metrics.

## 👩‍💻 Author

**Meghana Sattaru**  
B.Tech — Computer Science and Engineering (Artificial Intelligence and Machine Learning)

Interested in Artificial Intelligence, Machine Learning, Computer Vision, and Edge AI.


## 📄 License

No open-source license has been added to this repository yet.