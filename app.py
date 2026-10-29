
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


# =========================================================
# 1. CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="EdgeInspect AI",
    page_icon="🏭",
    layout="wide",
)

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
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

# =========================================================
# 5. SIDEBAR
# =========================================================

st.sidebar.title("🏭 EdgeInspect AI")
st.sidebar.caption("AI-powered steel surface inspection")

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
    value=0.45,
    step=0.05,
    help="Only predictions at or above this confidence are displayed.",
)

st.sidebar.caption(f"Image size: {IMAGE_SIZE} × {IMAGE_SIZE}")
st.sidebar.caption(f"Active device: {DEVICE_NAME}")


# =========================================================
# 6. MAIN HEADER AND TABS
# =========================================================

st.title("🏭 EdgeInspect AI")

st.write(
    "Real-Time Edge-AI Visual Inspection System "
    "for Automated Industrial Defect Detection"
)

(
    inspect_tab,
    dashboard_tab,
    performance_tab,
    evaluation_tab,
    about_tab,
) = st.tabs([
    "🔍 Inspect Image",
    "📊 Dashboard & History",
    "⚡ CPU vs GPU Performance",
    "🎯 Model Evaluation",
    "ℹ️ About",
])


# =========================================================
# 7. IMAGE INSPECTION
# =========================================================

with inspect_tab:

    st.subheader("Steel Surface Defect Inspection")

    if not MODEL_PATH.exists():

        st.error(
            "Model file is missing. Place your trained best.pt file "
            "inside the project's models folder."
        )

    else:

        uploaded_file = st.file_uploader(
            "Upload a steel surface image",
            type=["jpg", "jpeg", "png", "bmp"],
            key="inspection_uploader",
        )

        if uploaded_file is not None:

            original_image = Image.open(
                uploaded_file
            ).convert("RGB")

            st.image(
                original_image,
                caption="Uploaded steel surface image",
                width=500,
            )

            if st.button(
                "🔍 Inspect Image",
                type="primary",
                key="inspect_button",
            ):

                try:

                    with st.spinner("Running YOLO26 inspection..."):

                        model = load_model(str(MODEL_PATH))

                        # Warm up the model once per session/device.
                        warmup_key = f"warmed_{DEVICE}"

                        if not st.session_state.get(
                            warmup_key, False
                        ):

                            model.predict(
                                source=original_image,
                                imgsz=IMAGE_SIZE,
                                conf=confidence_threshold,
                                device=DEVICE,
                                half=GPU_AVAILABLE,
                                verbose=False,
                            )

                            synchronize_gpu()

                            st.session_state[warmup_key] = True

                        # Measure the prediction call.
                        synchronize_gpu()

                        start_time = time.perf_counter()

                        results = model.predict(
                            source=original_image,
                            imgsz=IMAGE_SIZE,
                            conf=confidence_threshold,
                            device=DEVICE,
                            half=GPU_AVAILABLE,
                            verbose=False,
                        )

                        synchronize_gpu()

                        inference_ms = (
                            time.perf_counter() - start_time
                        ) * 1000

                        result = results[0]

                        # Convert prediction image from BGR to RGB.
                        annotated_array = result.plot()

                        annotated_image = Image.fromarray(
                            annotated_array[:, :, ::-1]
                        )

                        detections = []

                        if result.boxes is not None:

                            for box in result.boxes:

                                class_id = int(
                                    box.cls[0].item()
                                )

                                confidence = float(
                                    box.conf[0].item()
                                )

                                bbox = [
                                    round(float(value), 2)
                                    for value in box.xyxy[0].tolist()
                                ]

                                class_name = result.names.get(
                                    class_id,
                                    f"class_{class_id}",
                                )

                                detections.append({
                                    "class": class_name,
                                    "confidence": confidence,
                                    "bbox": bbox,
                                })

                        # Save the annotated image in memory.
                        image_buffer = io.BytesIO()

                        annotated_image.save(
                            image_buffer,
                            format="PNG",
                        )

                        annotated_bytes = image_buffer.getvalue()

                        # Save inspection and detection details to SQLite.
                        timestamp = save_inspection(
                            uploaded_file.name,
                            detections,
                            inference_ms,
                        )

                        # Save the latest result so downloads remain
                        # available after Streamlit reruns.
                        st.session_state["last_inspection"] = {
                            "filename": uploaded_file.name,
                            "timestamp": timestamp,
                            "detections": detections,
                            "inference_ms": inference_ms,
                            "device_name": DEVICE_NAME,
                            "annotated_bytes": annotated_bytes,
                            "image_size": original_image.size,
                        }

                except Exception as error:

                    st.error(f"Inspection failed: {error}")
                    st.exception(error)

        # Display the last result outside the button block.
        saved_result = st.session_state.get(
            "last_inspection"
        )

        if (
            saved_result
            and uploaded_file is not None
            and saved_result["filename"] == uploaded_file.name
        ):

            st.markdown("---")
            st.subheader("Inspection Results")

            detections = saved_result["detections"]

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Objects detected",
                len(detections),
            )

            col2.metric(
                "Inference time",
                f'{saved_result["inference_ms"]:.2f} ms',
            )

            col3.metric(
                "Inference device",
                saved_result["device_name"],
            )

            st.caption(
                "Inference time measures the warmed-up prediction call. "
                "It excludes model loading, annotation drawing, and "
                "database saving."
            )

            annotated_image = Image.open(
                io.BytesIO(saved_result["annotated_bytes"])
            )

            st.image(
                annotated_image,
                caption="YOLO26 predictions",
                width=700,
            )

            if detections:

                detection_df = pd.DataFrame([
                    {
                        "Class": item["class"],
                        "Confidence (%)": round(
                            item["confidence"] * 100, 2
                        ),
                        "Bounding box": str(item["bbox"]),
                    }
                    for item in detections
                ])

                st.dataframe(
                    detection_df,
                    use_container_width=True,
                    hide_index=True,
                )

                st.warning("Defect candidates detected.")

            else:

                st.success(
                    "No defect candidates detected above the selected "
                    "confidence threshold. This does not guarantee "
                    "that the surface is defect-free."
                )

            st.success("Inspection saved to history.")

            # Download annotated image.
            st.download_button(
                "📥 Download Annotated Image",
                data=saved_result["annotated_bytes"],
                file_name="edgeinspect_annotated.png",
                mime="image/png",
                key="download_annotated_image",
            )

            # Generate and download PDF report.
            try:

                pdf_bytes = generate_inspection_pdf(
                    filename=saved_result["filename"],
                    timestamp=saved_result["timestamp"],
                    detections=detections,
                    inference_ms=saved_result["inference_ms"],
                    device_name=saved_result["device_name"],
                    annotated_image_bytes=saved_result[
                        "annotated_bytes"
                    ],
                    image_size=saved_result["image_size"],
                )

                st.download_button(
                    "📄 Download PDF Inspection Report",
                    data=pdf_bytes,
                    file_name="edgeinspect_inspection_report.pdf",
                    mime="application/pdf",
                    key="download_pdf_report",
                )

            except Exception as error:

                st.error(
                    f"Could not generate PDF report: {error}"
                )

# ============================================================
# TAB 2: DASHBOARD & INSPECTION HISTORY
# ============================================================

with dashboard_tab:

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

with performance_tab:

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

with evaluation_tab:

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

with about_tab:

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

    st.subheader("🧠 Model Information")

    st.write(f"**Model path:** `{MODEL_PATH}`")
    st.write(f"**Dataset:** NEU-DET")
    st.write(f"**Image size:** {IMAGE_SIZE} × {IMAGE_SIZE}")
    st.write(f"**Number of classes:** {len(CLASS_NAMES)}")

    if MODEL_PATH.exists():
        st.success("Model checkpoint file exists.")
    else:
        st.error(
            "Model checkpoint was not found. "
            "Check the models/best.pt path."
        )

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

st.caption(
    "Industrial Defect Detection | "
    "Powered by Python, PyTorch, YOLO26 and Streamlit"
)
