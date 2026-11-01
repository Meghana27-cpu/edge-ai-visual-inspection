
import io
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
import streamlit as st
import torch
from PIL import Image

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as ReportImage,
)

from ultralytics import YOLO
from datetime import datetime
from zoneinfo import ZoneInfo


# =========================================================
# 1. CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="EdgeInspect AI",
    page_icon="🏭",
    layout="wide",
)
st.markdown("""
<style>
    /* ---------- Brand and workspace ---------- */
    :root {
        --ei-navy: #10233f;
        --ei-blue: #2563eb;
        --ei-blue-hover: #1d4ed8;
        --ei-bg: #f5f7fb;
        --ei-border: #e2e8f0;
        --ei-text: #172b4d;
        --ei-muted: #64748b;
    }
    
    
    /* Hide Streamlit radio indicators */
    [data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {
            display: none !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radio"] > div {
            display: none !important;
    }

    /* Make navigation items look like clean menu links */
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
            padding: 10px 12px !important;
            border-radius: 8px !important;
            margin-bottom: 4px !important;
            cursor: pointer !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            background-color: #1d3556 !important;
    }


    .stApp {
        background: var(--ei-bg);
        color: var(--ei-text);
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* ---------- Typography ---------- */
    h1, h2, h3 {
        color: var(--ei-text);
        font-weight: 650;
        letter-spacing: -0.4px;
    }

    p, label {
        line-height: 1.55;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: var(--ei-navy);
        border-right: 1px solid #243a58;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #f1f5f9;
    }

    [data-testid="stSidebar"] hr {
        border-color: #344761;
    }

    /* ---------- Sidebar navigation ---------- */
    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 8px;
    }

    [data-testid="stSidebar"] [role="radio"] {
        background: transparent;
        border-radius: 8px;
        padding: 9px 10px;
    }

    [data-testid="stSidebar"] [role="radio"]:hover {
        background: #1d3556;
    }

    /* ---------- Metric cards ---------- */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--ei-border);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 3px 10px rgba(16, 35, 63, 0.04);
    }

    [data-testid="stMetricLabel"] {
        color: var(--ei-muted);
        font-size: 0.88rem;
    }

    [data-testid="stMetricValue"] {
        color: var(--ei-navy);
        font-weight: 700;
    }

    /* ---------- Buttons ---------- */
    .stButton > button,
    .stDownloadButton > button {
        border-radius: 8px;
        min-height: 40px;
        font-weight: 600;
        transition: border-color 0.15s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: var(--ei-blue);
        color: var(--ei-blue);
    }

    /* ---------- Upload area ---------- */
    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px dashed #b7c6d9;
        border-radius: 12px;
        padding: 12px;
    }

    /* ---------- Tables and charts ---------- */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--ei-border);
        border-radius: 10px;
        overflow: hidden;
    }

    [data-testid="stTable"] {
        border-radius: 10px;
    }

    /* ---------- Expanders and status messages ---------- */
    [data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid var(--ei-border);
        border-radius: 10px;
    }

    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    /* ---------- Footer ---------- */
    .ei-footer {
        text-align: center;
        color: var(--ei-muted);
        font-size: 0.85rem;
        padding: 20px 0 8px;
        border-top: 1px solid var(--ei-border);
        margin-top: 32px;
    }
</style>
""", unsafe_allow_html=True)
    

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"
DB_PATH = BASE_DIR / "database" / "inspections.db"

IMAGE_SIZE = 640

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

# Previously recorded CPU/GPU benchmark results.
BENCHMARK_RESULTS = {
    "CPU": {
        "avg_latency_ms": 70.40,
        "median_latency_ms": 67.38,
        "p95_latency_ms": 98.20,
        "throughput_images_per_sec": 14.20,
    },
    "GPU": {
        "avg_latency_ms": 28.36,
        "median_latency_ms": 28.40,
        "p95_latency_ms": 43.84,
        "throughput_images_per_sec": 35.26,
    },
}

GPU_AVAILABLE = torch.cuda.is_available()

DEVICE = "0" if GPU_AVAILABLE else "cpu"

DEVICE_NAME = (
    torch.cuda.get_device_name(0)
    if GPU_AVAILABLE
    else "CPU"
)


def synchronize_gpu():
    """Wait for pending GPU operations before timing."""
    if GPU_AVAILABLE:
        torch.cuda.synchronize()


# =========================================================
# 2. DATABASE
# =========================================================

def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def initialize_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                filename TEXT NOT NULL,
                defects INTEGER NOT NULL,
                classes TEXT,
                inference_ms REAL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS detection_details (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                inspection_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                bbox TEXT,
                FOREIGN KEY (inspection_id)
                    REFERENCES inspections(id)
            )
        """)

        # Add the bounding-box column to older databases.
        columns = [
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(detection_details)"
            ).fetchall()
        ]

        if "bbox" not in columns:
            conn.execute(
                "ALTER TABLE detection_details ADD COLUMN bbox TEXT"
            )

        conn.commit()


def save_inspection(filename, detections, inference_ms):
    
    timestamp = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
    

    class_names = [
        detection["class"]
        for detection in detections
    ]

    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO inspections
                (timestamp, filename, defects, classes, inference_ms)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                filename,
                len(detections),
                ", ".join(class_names),
                inference_ms,
            ),
        )

        inspection_id = cursor.lastrowid

        for detection in detections:
            conn.execute(
                """
                INSERT INTO detection_details
                    (inspection_id, class_name, confidence, bbox)
                VALUES (?, ?, ?, ?)
                """,
                (
                    inspection_id,
                    detection["class"],
                    detection["confidence"],
                    str(detection["bbox"]),
                ),
            )

        conn.commit()

    return timestamp


def load_history():
    with get_connection() as conn:
        return pd.read_sql_query(
            """
            SELECT id, timestamp, filename, defects,
                   classes, inference_ms
            FROM inspections
            ORDER BY id DESC
            """,
            conn,
        )



def load_detection_details(inspection_id=None):
    """
    Load detection details from SQLite.

    If inspection_id is provided, return details for that inspection.
    Otherwise, return all saved detection details.
    """

    try:
        conn = sqlite3.connect(DB_PATH)

        if inspection_id is not None:
            query = """
                SELECT *
                FROM detection_details
                WHERE inspection_id = ?
            """
            df = pd.read_sql_query(
                query,
                conn,
                params=(inspection_id,)
            )
        else:
            query = "SELECT * FROM detection_details"
            df = pd.read_sql_query(query, conn)

        conn.close()
        return df

    except sqlite3.Error as error:
        st.error(f"Could not load detection details: {error}")
        return pd.DataFrame()



initialize_database()


# =========================================================
# 3. MODEL LOADING
# =========================================================

@st.cache_resource
def load_model(model_path):
    return YOLO(str(model_path))


# =========================================================
# 4. PDF REPORT GENERATION
# =========================================================

def generate_inspection_pdf(
    filename,
    timestamp,
    detections,
    inference_ms,
    device_name,
    annotated_image_bytes,
    image_size,
):
    pdf_buffer = io.BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        textColor=colors.HexColor("#15324b"),
        alignment=1,
        spaceAfter=12,
    )

    story.append(
        Paragraph("EdgeInspect AI", title_style)
    )

    story.append(
        Paragraph(
            "Industrial Steel Surface Inspection Report",
            styles["Heading2"],
        )
    )

    story.append(Spacer(1, 12))

    image_width, image_height = image_size

    metadata = [
        ["Image filename:", escape(str(filename))],
        ["Inspection date:", escape(str(timestamp))],
        [
            "Image dimensions:",
            f"{image_width} x {image_height}",
        ],
        ["Inference device:", escape(str(device_name))],
        ["Inference time:", f"{inference_ms:.2f} ms"],
        ["Detection candidates:", str(len(detections))],
    ]

    metadata_table = Table(
        metadata,
        colWidths=[145, 350],
    )

    metadata_table.setStyle(
        TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )

    story.append(metadata_table)
    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "Detection Details",
            styles["Heading2"],
        )
    )

    story.append(Spacer(1, 8))

    table_data = [[
        "Class",
        "Confidence (%)",
        "Bounding box (x1, y1, x2, y2)",
    ]]

    for detection in detections:
        bbox_text = ", ".join(
            f"{value:.1f}"
            for value in detection["bbox"]
        )

        table_data.append([
            escape(detection["class"]),
            f'{detection["confidence"] * 100:.2f}',
            bbox_text,
        ])

    if not detections:
        table_data.append([
            "No candidates",
            "-",
            "-",
        ])

    detection_table = Table(
        table_data,
        colWidths=[115, 100, 280],
        repeatRows=1,
    )

    detection_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#243447"),
            ),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ])
    )

    story.append(detection_table)
    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "Annotated Inspection Image",
            styles["Heading2"],
        )
    )

    story.append(Spacer(1, 8))

    image_buffer = io.BytesIO(annotated_image_bytes)

    report_image = ReportImage(
        image_buffer,
        width=6 * inch,
        height=4 * inch,
        kind="proportional",
    )

    story.append(report_image)
    story.append(Spacer(1, 16))

    disclaimer = (
        "<i>Important: The model reports predicted defect candidates. "
        "These predictions are not independent confirmation of physical "
        "defects and should be validated before industrial quality or "
        "safety decisions.</i>"
    )

    story.append(
        Paragraph(disclaimer, styles["Italic"])
    )

    document.build(story)

    pdf_buffer.seek(0)

    return pdf_buffer.getvalue()


    
def check_image_quality(image):
    """
    Check image quality before defect inspection.
    Returns advisory warnings; it does not reject images.
    """
    import numpy as np

    warnings = []

    width, height = image.size
    gray = np.asarray(image.convert("L"), dtype=np.float32)

    # 1. Image resolution
    if width < 200 or height < 200:
        warnings.append(
            f"Low resolution: {width} × {height}. "
            "Use an image at least 200 × 200 pixels if possible."
        )

    # 2. Brightness
    brightness = float(np.mean(gray))

    if brightness < 40:
        warnings.append(
            "Image may be too dark. Improve the lighting and retake it if possible."
        )
    elif brightness > 215:
        warnings.append(
            "Image may be overexposed. Reduce glare or lighting and retake it if possible."
        )

    # 3. Contrast
    contrast = float(np.std(gray))

    if contrast < 25:
        warnings.append(
            "Low contrast: surface details may be difficult to distinguish."
        )

    # 4. Blur estimate using the variance of the Laplacian
    padded = np.pad(gray, 1, mode="edge")

    laplacian = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4 * gray
    )

    blur_score = float(np.var(laplacian))

    if blur_score < 50:
        warnings.append(
            "Image may be blurry. Keep the camera steady and refocus if possible."
        )

    return {
        "width": width,
        "height": height,
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "blur_score": round(blur_score, 2),
        "warnings": warnings,
    }

# =========================================================
# 5. SIDEBAR
# =========================================================

st.sidebar.title("🏭 EdgeInspect AI")
st.sidebar.caption("AI-powered steel surface inspection")
selected_page = st.sidebar.radio(
    "Navigation",
    [
        "🔍 Inspect Image",
        "📊 Dashboard & History",
        "⚡ CPU vs GPU Performance",
        "🎯 Model Evaluation",
        "ℹ️ About",
    ],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.subheader("System Status")

if MODEL_PATH.exists():
    st.sidebar.success("Model file found")
else:
    st.sidebar.error(f"Model not found: {MODEL_PATH}")

if GPU_AVAILABLE:
    st.sidebar.success("CUDA GPU available")
    st.sidebar.caption(DEVICE_NAME)
else:
    st.sidebar.warning("CUDA unavailable — using CPU")

st.sidebar.markdown("---")

confidence_threshold = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.10,
    max_value=0.95,
    value=0.30,
    step=0.05,
    help="Only predictions at or above this confidence are displayed.",
)

st.sidebar.caption(f"Image size: {IMAGE_SIZE} × {IMAGE_SIZE}")
st.sidebar.caption(f"Active device: {DEVICE_NAME}")
st.sidebar.divider()

with st.sidebar.expander("ℹ️ About the Project"):
    st.markdown("""
    **EdgeInspect AI**

    Real-Time Edge-AI Visual Inspection System
    for Automated Industrial Defect Detection.

    **Model:** YOLO26n

    **Dataset:** NEU-DET

    **Image Size:** 640 × 640

    **Defect Classes:**
    - Crazing
    - Inclusion
    - Patches
    - Pitted Surface
    - Rolled-in Scale
    - Scratches

    **Technologies:**
    - Python
    - Streamlit
    - PyTorch
    - OpenCV
    - SQLite

    **Disclaimer:** This is a project prototype.
    Predictions should be verified before
    production quality-control decisions.
    """)

# =========================================================
# 6. MAIN HEADER AND TABS
# =========================================================

st.title("🏭 EdgeInspect AI")

st.write(
    "Real-Time Edge-AI Visual Inspection System "
    "for Automated Industrial Defect Detection"
)



# =========================================================
# 7. IMAGE INSPECTION - MULTI-IMAGE / PER-VIEW RESULTS
# =========================================================

if selected_page == "🔍 Inspect Image":

    st.subheader("Steel Surface Defect Inspection")
    st.write(
        "Upload or capture images of the steel products. "
        "Assign a view to image and inspect them."
    )

    if not MODEL_PATH.exists():
        st.error(
            "Model file is missing. Place your trained best.pt "
            "file inside the project's models folder."
        )

    else:
        
        
        # Initialize camera session state
        if "camera_images" not in st.session_state:
            st.session_state.camera_images = []

        if "camera_slot" not in st.session_state:
            st.session_state.camera_slot = 0

        if "show_camera" not in st.session_state:
            st.session_state.show_camera = False
            
        # Display captured images
        if st.session_state.camera_images:
            st.markdown("### Captured Images")

            for image_data in st.session_state.camera_images:
                st.image(image_data["bytes"], caption=image_data["name"])
        # Create two columns
        upload_col, camera_col = st.columns(2)
        with upload_col:
            st.markdown("### 📁 Upload Image")

            uploaded_files = st.file_uploader(
                "Choose images from your device",
                type=["jpg", "jpeg", "png", "bmp"],
                accept_multiple_files=True,
                key="multi_inspection_uploader"
            )

        with camera_col:
            st.markdown("### 📷 Take Photo")

            if st.button("📸 Open Camera", use_container_width=True):
                st.session_state.show_camera = True
                st.rerun()

        # Camera opens only after clicking the button
        if st.session_state.show_camera:
            st.markdown("### Capture Steel Surface")

            camera_photo = st.camera_input(
                "Take a photo",
                key=f"camera_capture_{st.session_state.camera_slot}"
            )

            if camera_photo is not None:
                if st.button("➕ Add Photo to Batch"):
                    st.session_state.camera_images.append({
                        "name": f"captured_image_{len(st.session_state.camera_images) + 1}.jpg",
                        "bytes": camera_photo.getvalue()
                    })

                    st.session_state.camera_slot += 1
                    st.session_state.show_camera = False
                    st.rerun()

            if st.button("✖ Close Camera"):
                st.session_state.show_camera = False
                st.rerun()

            for i, image_data in enumerate(st.session_state.camera_images):
                st.image(
                    image_data["bytes"],
                    caption=image_data["name"],
                    width=200
                )

            if st.button("🗑️ Clear Captured Images"):
                st.session_state.camera_images = []
                st.session_state.camera_slot = 0
                st.rerun()


        
        # Combine uploaded and captured images
        inspection_files = list(uploaded_files or [])

        for image_data in st.session_state.camera_images:
            image_file = io.BytesIO(image_data["bytes"])
            image_file.name = image_data["name"]
            inspection_files.append(image_file)

        view_options = [
            "Front", "Back", "Left", "Right",
            "Top", "Bottom", "Other"
        ]

        if inspection_files:
            st.markdown("### Uploaded Images")
            st.caption(
                "Choose the camera view for each image before inspection."
            )

            view_by_file = {}

            for index, uploaded_file in enumerate(inspection_files):
                file_key = f"{uploaded_file.name}_{index}"

                col_image, col_details = st.columns([1, 2])

                with col_image:
                    preview = Image.open(uploaded_file).convert("RGB")
                    
                    quality = check_image_quality(preview)

                    st.caption(
                        f"Image size: {quality['width']} × {quality['height']} pixels"
                    )

                with st.expander("📷 Image Quality Check", expanded=True):
                    if quality["warnings"]:
                        for warning in quality["warnings"]:
                            st.warning(warning)
                    else:
                            st.success("No obvious image-quality issues detected.")

                            st.caption(
                                    f"Brightness: {quality['brightness']} | "
                                    f"Contrast: {quality['contrast']} | "
                                    f"Sharpness score: {quality['blur_score']}"
                            )

                    st.image(
                        preview,
                        caption=uploaded_file.name,
                        use_container_width=True
                    )

                with col_details:
                    selected_view = st.selectbox(
                        f"View for {uploaded_file.name}",
                        view_options,
                        key=f"view_{file_key}"
                    )

                    view_by_file[file_key] = selected_view
                    st.caption(
                        f"Image dimensions: {preview.width} × {preview.height}"
                    )

            
            button_label = (
                "🔍 Inspect Image"
                if len(inspection_files) == 1
                else "🔍 Inspect All Images"
            )

            if st.button(
                button_label,
                type="primary",
                key="inspect_all_images"
            ):
                try:
                    model = load_model(str(MODEL_PATH))
                    all_results = []

                    progress = st.progress(0)
                    status = st.empty()

                    for index, uploaded_file in enumerate(inspection_files):
                        file_key = f"{uploaded_file.name}_{index}"
                        selected_view = view_by_file[file_key]

                        status.write(
                            f"Inspecting image {index + 1} of "
                            f"{len(inspection_files)}: {uploaded_file.name}"
                        )

                        original_image = Image.open(
                            uploaded_file
                        ).convert("RGB")
                        # Save original image for comparison
                        original_buffer = io.BytesIO()

                        original_image.save(
                            original_buffer,
                            format="PNG"
                        )

                        original_bytes = original_buffer.getvalue()
                        # Warm up YOLO once per session/device.
                        warmup_key = f"warmed_{DEVICE}"
                        if not st.session_state.get(warmup_key, False):
                            model.predict(
                                source=original_image,
                                imgsz=IMAGE_SIZE,
                                conf=confidence_threshold,
                                device=DEVICE,
                                half=GPU_AVAILABLE,
                                verbose=False
                            )
                            synchronize_gpu()
                            st.session_state[warmup_key] = True

                        synchronize_gpu()
                        start_time = time.perf_counter()

                        results = model.predict(
                            source=original_image,
                            imgsz=IMAGE_SIZE,
                            conf=confidence_threshold,
                            device=DEVICE,
                            half=GPU_AVAILABLE,
                            verbose=False
                        )

                        synchronize_gpu()
                        inference_ms = (
                            time.perf_counter() - start_time
                        ) * 1000

                        result = results[0]
                        annotated_array = result.plot()
                        annotated_image = Image.fromarray(
                            annotated_array[:, :, ::-1]
                        )

                        detections = []

                        if result.boxes is not None:
                            for box in result.boxes:
                                class_id = int(box.cls[0].item())
                                confidence = float(box.conf[0].item())
                                bbox = [
                                    round(float(value), 2)
                                    for value in box.xyxy[0].tolist()
                                ]

                                detections.append({
                                    "class": result.names.get(
                                        class_id, f"class_{class_id}"
                                    ),
                                    "confidence": confidence,
                                    "bbox": bbox
                                })

                        image_buffer = io.BytesIO()
                        annotated_image.save(image_buffer, format="PNG")
                        annotated_bytes = image_buffer.getvalue()

                        # Preserve the existing SQLite history format.
                        timestamp = save_inspection(
                            uploaded_file.name,
                            detections,
                            inference_ms
                        )

                        all_results.append({
                            "filename": uploaded_file.name,
                            "view": selected_view,
                            "timestamp": timestamp,
                            "detections": detections,
                            "inference_ms": inference_ms,
                            "device_name": DEVICE_NAME,

                            # Original and detected images
                            "original_bytes": original_bytes,
                            "annotated_bytes": annotated_bytes,

                            "image_size": original_image.size
                        })

                        progress.progress(
                            (index + 1) / len(inspection_files)
                        )

                    st.session_state["multi_inspection_results"] = all_results
                    st.session_state["multi_inspection_signature"] = [
                        (f.name, view_by_file[f"{f.name}_{i}"])
                        for i, f in enumerate(inspection_files)
                    ]

                    status.success(
                        f"Inspection completed for {len(all_results)} image(s)."
                    )

                except Exception as error:
                    st.error(f"Inspection failed: {error}")
                    st.exception(error)

            # Show results only when they correspond to the current uploads/views.
            current_signature = [
                (f.name, view_by_file[f"{f.name}_{i}"])
                for i, f in enumerate(inspection_files)
            ]

            saved_results = st.session_state.get(
                "multi_inspection_results", []
            )
            saved_signature = st.session_state.get(
                "multi_inspection_signature", []
            )

            if saved_results and saved_signature == current_signature:
                st.markdown("---")
                st.subheader("Per-View Inspection Results")

                total_detections = sum(
                    len(item["detections"]) for item in saved_results
                )
                total_inference_ms = sum(
                    item["inference_ms"] for item in saved_results
                )

                col1, col2, col3 = st.columns(3)
                col1.metric("Images Inspected", len(saved_results))
                col2.metric("Detection Candidates", total_detections)
                col3.metric(
                    "Total Inference Time",
                    f"{total_inference_ms:.2f} ms"
                )

                for index, item in enumerate(saved_results):
                    with st.expander(
                        f'{item["view"]} View — {item["filename"]}',
                        expanded=(index == 0)
                    ):
                        col_image, col_details = st.columns([3, 2])

                        with col_image:

                             original_col, detected_col = st.columns(2)
                        with original_col:
                            st.markdown("**Original Image**")

                            st.image(
                                item["original_bytes"],
                                use_container_width=True
                            )

                        with detected_col:
                            st.markdown("**Detected Image**")

                            st.image(
                                item["annotated_bytes"],
                                use_container_width=True
                            )    

                        with col_details:
                            metric1, metric2 = st.columns(2)
                            metric1.metric(
                                "Candidates",
                                len(item["detections"])
                            )
                            metric2.metric(
                                "Inference",
                                f'{item["inference_ms"]:.2f} ms'
                            )

                            st.caption(
                                f'Device: {item["device_name"]} · '
                                f'Time: {item["timestamp"]}'
                            )

                            if item["detections"]:
                                detection_df = pd.DataFrame([
                                    {
                                        "Class": detection["class"],
                                        "Confidence (%)": round(
                                            detection["confidence"] * 100, 2
                                        ),
                                        "Bounding box": str(detection["bbox"])
                                    }
                                    for detection in item["detections"]
                                ])
                                st.dataframe(
                                    detection_df,
                                    use_container_width=True,
                                    hide_index=True
                                )
                                st.warning(
                                    "Defect candidates detected in this image."
                                )
                            else:
                                st.success(
                                     "✅ Product Quality: GOOD"
                                )
                                st.write(
                                     "No defects detected. The product appears "
                                     "to be in good condition based on this inspection."
                                )

                                st.caption(
                                     "Automated inspection result. Final quality "
                                     "approval should follow applicable inspection standards."
                                )

                        st.download_button(
                            "📥 Download Annotated Image",
                            data=item["annotated_bytes"],
                            file_name=(
                                f'edgeinspect_{item["view"].lower()}_'
                                f'{Path(item["filename"]).stem}.png'
                            ),
                            mime="image/png",
                            key=f"download_multi_image_{index}"
                        )

                        # Keep the existing individual PDF report feature.
                        try:
                            pdf_bytes = generate_inspection_pdf(
                                filename=item["filename"],
                                timestamp=item["timestamp"],
                                detections=item["detections"],
                                inference_ms=item["inference_ms"],
                                device_name=item["device_name"],
                                annotated_image_bytes=item["annotated_bytes"],
                                image_size=item["image_size"]
                            )

                            st.download_button(
                                "📄 Download PDF Report",
                                data=pdf_bytes,
                                file_name=(
                                    f'edgeinspect_{item["view"].lower()}_'
                                    f'{Path(item["filename"]).stem}.pdf'
                                ),
                                mime="application/pdf",
                                key=f"download_multi_pdf_{index}"
                            )

                        except Exception as error:
                            st.error(
                                f"Could not generate PDF report: {error}"
                            )

                st.caption(
                    "Each image is processed independently. The view label "
                    "organizes the results; it does not identify the same "
                    "physical defect across different views."
                )



# ============================================================
# TAB 2: DASHBOARD & INSPECTION HISTORY
# ============================================================

if selected_page == "📊 Dashboard & History":

    st.header("📊 Inspection Dashboard")

    history_df = load_history()

    if history_df.empty:

        st.info(
            "No inspection history available yet. "
            "Inspect an image first to populate the dashboard."
        )

    else:

        # ----------------------------------------------------
        # SUMMARY METRICS
        # ----------------------------------------------------

        total_inspections = len(history_df)

        total_detections = int(history_df["defects"].sum())

        average_inference = float(
            history_df["inference_ms"].mean()
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Inspections",
            total_inspections
        )

        col2.metric(
            "Total Detected Objects",
            total_detections
        )

        col3.metric(
            "Average Inference Time",
            f"{average_inference:.2f} ms"
        )

        st.markdown("---")

        # ----------------------------------------------------
        # INSPECTION HISTORY TABLE
        # ----------------------------------------------------

        st.subheader("📋 Recent Inspections")

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # DETECTION HISTORY DETAILS
        # ----------------------------------------------------

        st.subheader("🔎 View Detection Details")

        if "id" in history_df.columns:

            inspection_ids = history_df["id"].tolist()

            selected_id = st.selectbox(
                "Select an inspection ID",
                inspection_ids,
                key="history_inspection_id"
            )

            details_df = load_detection_details(selected_id)

            if details_df is not None and not details_df.empty:

                st.dataframe(
                    details_df,
                    use_container_width=True,
                    hide_index=True
                )

            else:
                st.info(
                    "No individual detection details are available "
                    "for this inspection."
                )

        # ----------------------------------------------------
        # EXPORT HISTORY AS CSV
        # ----------------------------------------------------

        st.subheader("📥 Export Inspection History")

        csv_data = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download History as CSV",
            data=csv_data,
            file_name="inspection_history.csv",
            mime="text/csv"
        )

        # ----------------------------------------------------
        # DEFECT FREQUENCY CHART
        # ----------------------------------------------------

        st.subheader("📈 Defect Frequency")

        defect_counts = {}

        for _, inspection in history_df.iterrows():

            inspection_id = inspection.get("id")

            if inspection_id is None:
                continue

            details = load_detection_details(inspection_id)

            if details is None or details.empty:
                continue

            if "class_name" not in details.columns:
                continue

            for defect_name in details["class_name"].dropna():
                defect_counts[defect_name] = (
                    defect_counts.get(defect_name, 0) + 1
                )

        if defect_counts:

            defect_df = pd.DataFrame(
                list(defect_counts.items()),
                columns=["Defect", "Count"]
            ).set_index("Defect")

            st.bar_chart(defect_df)

        else:
            st.info(
                "Defect frequency data will appear after inspections "
                "with saved detection details."
            )

# ============================================================
# TAB 3: CPU VS GPU PERFORMANCE
# ============================================================

if selected_page == "⚡ CPU vs GPU Performance":

    st.header("⚡ CPU vs GPU Performance Comparison")

    st.write(
        "Compare inference speed and throughput when running "
        "the defect detection model on the CPU and GPU."
    )

    # --------------------------------------------------------
    # BENCHMARK RESULTS
    # --------------------------------------------------------

    gpu_results = BENCHMARK_RESULTS.get("GPU", {})
    cpu_results = BENCHMARK_RESULTS.get("CPU", {})

    if gpu_results and cpu_results:

        gpu_latency = float(gpu_results.get("avg_latency_ms", 0))
        cpu_latency = float(cpu_results.get("avg_latency_ms", 0))

        gpu_throughput = float(
            gpu_results.get("throughput_images_per_sec", 0)
        )
        cpu_throughput = float(
            cpu_results.get("throughput_images_per_sec", 0)
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "GPU Average Latency",
            f"{gpu_latency:.2f} ms"
        )

        col2.metric(
            "CPU Average Latency",
            f"{cpu_latency:.2f} ms"
        )

        if gpu_latency > 0:
            speedup = cpu_latency / gpu_latency

            col3.metric(
                "GPU Speedup",
                f"{speedup:.2f}×"
            )

        st.markdown("---")

        # ----------------------------------------------------
        # LATENCY COMPARISON
        # ----------------------------------------------------

        st.subheader("📊 Average Inference Latency")

        latency_df = pd.DataFrame(
            {
                "Device": ["CPU", "GPU"],
                "Latency (ms)": [cpu_latency, gpu_latency]
            }
        ).set_index("Device")

        st.bar_chart(latency_df)

        # ----------------------------------------------------
        # THROUGHPUT COMPARISON
        # ----------------------------------------------------

        st.subheader("🚀 Throughput Comparison")

        throughput_df = pd.DataFrame(
            {
                "Device": ["CPU", "GPU"],
                "Images per Second": [
                    cpu_throughput,
                    gpu_throughput
                ]
            }
        ).set_index("Device")

        st.bar_chart(throughput_df)

        # ----------------------------------------------------
        # ADDITIONAL BENCHMARK METRICS
        # --------------------------------------------------------

        st.subheader("📋 Detailed Benchmark Results")

        detailed_rows = []

        for device_name, result in [
            ("CPU", cpu_results),
            ("GPU", gpu_results)
        ]:

            detailed_rows.append(
                {
                    "Device": device_name,
                    "Average Latency (ms)": result.get(
                        "avg_latency_ms", "N/A"
                    ),
                    "Median Latency (ms)": result.get(
                        "median_latency_ms", "N/A"
                    ),
                    "P95 Latency (ms)": result.get(
                        "p95_latency_ms", "N/A"
                    ),
                    "Throughput (images/sec)": result.get(
                        "throughput_images_per_sec", "N/A"
                    )
                }
            )

        benchmark_df = pd.DataFrame(detailed_rows)

        st.dataframe(
            benchmark_df,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # PERFORMANCE INTERPRETATION
        # ----------------------------------------------------

        st.subheader("💡 Performance Analysis")

        if gpu_latency > 0 and cpu_latency > 0:

            latency_reduction = (
                (cpu_latency - gpu_latency) / cpu_latency
            ) * 100

            st.write(
                f"- GPU speedup: **{cpu_latency / gpu_latency:.2f}×**"
            )

            st.write(
                f"- Latency reduction with GPU: "
                f"**{latency_reduction:.2f}%**"
            )

            if gpu_throughput > cpu_throughput:

                st.success(
                    "The GPU processes more images per second "
                    "than the CPU in these benchmark results."
                )

            elif gpu_throughput < cpu_throughput:

                st.info(
                    "The CPU shows higher throughput in these "
                    "benchmark results."
                )

            else:

                st.info(
                    "CPU and GPU throughput are equal in these results."
                )

    else:

        st.warning(
            "Benchmark results are unavailable. Run your benchmark "
            "script and update BENCHMARK_RESULTS with its output."
        )

# ============================================================
# TAB 4: MODEL EVALUATION
# ============================================================

if selected_page == "🎯 Model Evaluation":

    st.header("🎯 Model Evaluation")

    st.write(
        "Evaluate the trained YOLO26 model using validation "
        "metrics from the NEU-DET dataset."
    )

    # --------------------------------------------------------
    # OVERALL VALIDATION METRICS
    # --------------------------------------------------------

    st.subheader("📊 Overall Performance")

    # Metrics obtained from your previous validation results.
    precision = 0.743
    recall = 0.685
    map50 = 0.770
    map50_95 = 0.421

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Precision", f"{precision:.3f}")
    col2.metric("Recall", f"{recall:.3f}")
    col3.metric("mAP@50", f"{map50:.3f}")
    col4.metric("mAP@50–95", f"{map50_95:.3f}")

    st.caption(
        "These are the previously reported validation results. "
        "Update them if you retrain or re-evaluate the model."
    )
    st.markdown("---")
    st.subheader("Confidence Threshold Experiment (conf=0.55)")

    col1, col2, col3 = st.columns(3)

    col1.metric("Precision", "90.9%")
    col2.metric("Recall", "50.2%")
    col3.metric("mAP@50", "48.8%")

    st.caption(
        "Threshold experiment results on the 360-image validation set. "
        "Higher confidence improves precision but reduces recall. "
        "These results are not from an independent test set."
    )
    st.markdown("---")

    # --------------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------------

    st.subheader("📁 Validation Dataset")

    col1, col2, col3 = st.columns(3)

    col1.metric("Training Images", "1,440")
    col2.metric("Validation Images", "360")
    col3.metric("Validation Instances", "854")

    st.markdown("---")

    # --------------------------------------------------------
    # PER-CLASS METRICS
    # --------------------------------------------------------

    st.subheader("🔬 Per-Class Performance")

    class_metrics = pd.DataFrame(
        {
            "Defect Class": [
                "crazing",
                "inclusion",
                "patches",
                "pitted_surface",
                "rolled-in_scale",
                "scratches"
            ],
            "Precision": [
                0.707,
                0.807,
                0.861,
                0.846,
                0.623,
                0.615
            ],
            "Recall": [
                0.389,
                0.767,
                0.834,
                0.701,
                0.508,
                0.909
            ],
            "mAP@50": [
                0.561,
                0.857,
                0.921,
                0.797,
                0.640,
                0.844
            ],
            "mAP@50–95": [
                0.236,
                0.489,
                0.592,
                0.438,
                0.281,
                0.489
            ]
        }
    )

    st.dataframe(
        class_metrics.style.format(
            {
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "mAP@50": "{:.3f}",
                "mAP@50–95": "{:.3f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # --------------------------------------------------------
    # PRECISION AND RECALL CHART
    # --------------------------------------------------------

    st.subheader("📈 Precision vs Recall by Class")

    precision_recall_df = class_metrics.set_index(
        "Defect Class"
    )[["Precision", "Recall"]]

    st.bar_chart(precision_recall_df)

    # --------------------------------------------------------
    # MAP COMPARISON CHART
    # --------------------------------------------------------

    st.subheader("📊 Mean Average Precision by Class")

    map_df = class_metrics.set_index(
        "Defect Class"
    )[["mAP@50", "mAP@50–95"]]

    st.bar_chart(map_df)

    # --------------------------------------------------------
    # BEST AND WEAKEST CLASSES
    # --------------------------------------------------------

    st.subheader("💡 Evaluation Insights")

    best_map_row = class_metrics.loc[
        class_metrics["mAP@50"].idxmax()
    ]

    weakest_map_row = class_metrics.loc[
        class_metrics["mAP@50"].idxmin()
    ]

    best_recall_row = class_metrics.loc[
        class_metrics["Recall"].idxmax()
    ]

    st.success(
        f"Best mAP@50: **{best_map_row['Defect Class']}** "
        f"({best_map_row['mAP@50']:.3f})"
    )

    st.warning(
        f"Lowest mAP@50: **{weakest_map_row['Defect Class']}** "
        f"({weakest_map_row['mAP@50']:.3f}). "
        "This class may need further model improvement."
    )

    st.info(
        f"Highest recall: **{best_recall_row['Defect Class']}** "
        f"({best_recall_row['Recall']:.3f})"
    )

    # --------------------------------------------------------
    # DOWNLOAD EVALUATION RESULTS
    # --------------------------------------------------------

    st.subheader("📥 Export Evaluation Results")

    evaluation_csv = class_metrics.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Evaluation Metrics (CSV)",
        data=evaluation_csv,
        file_name="model_evaluation_metrics.csv",
        mime="text/csv"
    )

# ============================================================
# TAB 5: ABOUT THE PROJECT
# ============================================================

if selected_page == "ℹ️ About":

    st.header("ℹ️ About the Project")

    st.subheader(
        "Real-Time Edge-AI Visual Inspection System "
        "for Automated Industrial Defect Detection"
    )

    st.write(
        """
        This project uses deep learning and computer vision to
        detect surface defects in industrial steel images.

        The system uses a trained YOLO26 object detection model
        to identify defects and display their locations with
        bounding boxes, class labels, and confidence scores.
        """
    )

    st.markdown("---")

    # --------------------------------------------------------
    # PROJECT OBJECTIVES
    # --------------------------------------------------------

    st.subheader("🎯 Project Objectives")

    st.markdown(
        """
        - Automate the inspection of industrial steel surfaces.
        - Detect and locate surface defects using object detection.
        - Display predictions through an interactive Streamlit GUI.
        - Store inspection results in an SQLite database.
        - Generate downloadable inspection reports.
        - Compare CPU and GPU inference performance.
        - Evaluate the model using precision, recall, and mAP.
        """
    )

    st.markdown("---")

    # --------------------------------------------------------
    # TECHNOLOGIES USED
    # --------------------------------------------------------

    st.subheader("🛠️ Technologies Used")

    technologies = pd.DataFrame(
        {
            "Technology": [
                "Python",
                "YOLO26",
                "Ultralytics",
                "PyTorch",
                "Streamlit",
                "OpenCV",
                "Pandas",
                "SQLite",
                "ReportLab"
            ],
            "Purpose": [
                "Main programming language",
                "Object detection model",
                "Model training and inference",
                "Deep learning framework",
                "Interactive graphical interface",
                "Image processing",
                "Data handling and analysis",
                "Inspection history storage",
                "PDF report generation"
            ]
        }
    )

    st.dataframe(
        technologies,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # --------------------------------------------------------
    # DETECTABLE DEFECT CLASSES
    # --------------------------------------------------------

    st.subheader("🔍 Supported Defect Classes")

    defect_classes = [
        ("Crazing", "Fine crack-like patterns on the surface."),
        ("Inclusion", "Foreign material inclusions."),
        ("Patches", "Patch-like surface irregularities."),
        ("Pitted Surface", "Small pits or cavities on the surface."),
        ("Rolled-in Scale", "Scale-related surface imperfections."),
        ("Scratches", "Linear marks or scratches.")
    ]

    for defect_name, description in defect_classes:

        with st.expander(defect_name):
            st.write(description)

    st.markdown("---")

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    st.write("**Detection Model:** YOLO26n")
    st.write("**Dataset:** NEU-DET")
    st.write(f"**Input Image Size:** {IMAGE_SIZE} × {IMAGE_SIZE}")
    st.write(f"**Supported Defect Classes:** {len(CLASS_NAMES)}")

    st.markdown("---")

    # --------------------------------------------------------
    # PROJECT DISCLAIMER
    # --------------------------------------------------------

    st.subheader("⚠️ Disclaimer")

    st.info(
        "This system is a project prototype for industrial "
        "surface defect detection. Predictions depend on the "
        "training data and model performance. Results should "
        "be verified before using them for real production "
        "quality-control decisions."
    )

    st.caption(
        "Edge-AI Visual Inspection System | "
        "YOLO26 + NEU-DET + Streamlit"
    )


# ============================================================
# END OF APPLICATION
# ============================================================

st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.caption("🏭 **EdgeInspect AI**")
    st.caption("AI-powered steel defect detection")

with footer_col2:
    if st.button("ℹ️ About", key="footer_about"):
        st.session_state["footer_about_clicked"] = True

    if st.session_state.get("footer_about_clicked", False):
        st.info(
            "EdgeInspect AI uses YOLO26n and the NEU-DET "
            "dataset to detect steel surface defect candidates."
        )

with footer_col3:
    st.caption("Built with Python, PyTorch & Streamlit")