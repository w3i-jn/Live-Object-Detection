import tarfile
import streamlit as st
import PIL
import cv2
import numpy as np
import io
import time
from pathlib import Path
import collections
from openvino.tools import mo
import openvino as ov
import notebook_utils as notebook_utils
from openvino.tools.mo.front import tf as ov_tf_front

# OpenVINO Object Detection Model
def download_and_convert_model():
    base_model_dir = Path("model")
    model_name = "ssdlite_mobilenet_v2"
    archive_name = Path(f"{model_name}_coco_2018_05_09.tar.gz")
    model_url = f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/{model_name}/{archive_name}"

    # Download the model
    downloaded_model_path = base_model_dir / archive_name
    if not downloaded_model_path.exists():
        notebook_utils.download_file(model_url, downloaded_model_path.name, downloaded_model_path.parent)

    # Unpack the model
    tf_model_path = base_model_dir / archive_name.with_suffix("").stem / "frozen_inference_graph.pb"
    if not tf_model_path.exists():
        with tarfile.open(downloaded_model_path) as file:
            file.extractall(base_model_dir)

    precision = "FP16"
    converted_model_path = Path("model") / f"{model_name}_{precision.lower()}.xml"

    # Convert model to OpenVINO IR format
    if not converted_model_path.exists():
        trans_config_path = Path(ov_tf_front.__file__).parent / "ssd_v2_support.json"
        ov_model = mo.convert_model(
            tf_model_path,
            compress_to_fp16=(precision == "FP16"),
            transformations_config=trans_config_path,
            tensorflow_object_detection_api_pipeline_config=tf_model_path.parent / "pipeline.config",
            reverse_input_channels=True,
        )
        ov.save_model(ov_model, converted_model_path)

    return converted_model_path

# Function to process object detection results
def process_results(frame, results, thresh=0.6):
    h, w = frame.shape[:2]
    results = results.squeeze()
    boxes, labels, scores = [], [], []
    for _, label, score, xmin, ymin, xmax, ymax in results:
        boxes.append(tuple(map(int, (xmin * w, ymin * h, (xmax - xmin) * w, (ymax - ymin) * h))))
        labels.append(int(label))
        scores.append(float(score))
    
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=thresh, nms_threshold=0.6)
    if len(indices) == 0:
        return []
    return [(labels[idx], scores[idx], boxes[idx]) for idx in indices.flatten()]

# Drawing bounding boxes on detected objects
def draw_boxes(frame, boxes):
    colors = cv2.applyColorMap(src=np.arange(0, 255, 255 / 80, dtype=np.float32).astype(np.uint8), colormap=cv2.COLORMAP_RAINBOW).squeeze()
    for label, score, box in boxes:
        color = tuple(map(int, colors[label]))
        x2 = box[0] + box[2]
        y2 = box[1] + box[3]
        cv2.rectangle(img=frame, pt1=box[:2], pt2=(x2, y2), color=color, thickness=3)
        cv2.putText(img=frame, text=f"{label} {score:.2f}", org=(box[0] + 10, box[1] + 30), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=frame.shape[1] / 1000, color=color, thickness=1, lineType=cv2.LINE_AA)
    return frame

# Run object detection on video or webcam input
def run_object_detection(video_source, conf_threshold):
    core = ov.Core()
    device = "CPU"
    model_path = download_and_convert_model()
    model = core.read_model(model=model_path)
    compiled_model = core.compile_model(model=model, device_name=device)
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    height, width = list(input_layer.shape)[1:3]
    
    camera = cv2.VideoCapture(video_source)
    st_frame = st.empty()
    
    while camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            break

        input_img = cv2.resize(frame, (width, height))
        input_img = input_img[np.newaxis, ...]
        results = compiled_model([input_img])[output_layer]
        boxes = process_results(frame, results, conf_threshold)
        frame = draw_boxes(frame, boxes)
        st_frame.image(frame, channels="BGR")

    camera.release()

# Streamlit Interface
st.set_page_config(page_title="Facial & Object Detection", page_icon=":sun_with_face:", layout="centered", initial_sidebar_state="expanded")

st.title("Facial & Object Detection :sun_with_face:")
st.sidebar.header("Type")
source_radio = st.sidebar.radio("Select Source", ["IMAGE", "VIDEO", "WEBCAM"])

st.sidebar.header("Confidence")
conf_threshold = float(st.sidebar.slider("Select the Confidence Threshold", 10, 100, 20)) / 100

input_file = None
temporary_location = None

# Image Processing Section
if source_radio == "IMAGE":
    st.sidebar.header("Upload")
    input_file = st.sidebar.file_uploader("Choose an image.", type=("jpg", "png"))

    if input_file is not None:
        uploaded_image = PIL.Image.open(input_file)
        uploaded_image_cv = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)
        visualized_image = utils.predict_image(uploaded_image_cv, conf_threshold=conf_threshold)
        st.image(visualized_image, channels="BGR")
    else:
        st.image("assets/sample_image.jpg")
        st.write("Click on 'Browse Files' in the sidebar to run inference on an image.")

# Video or Webcam Processing Section
elif source_radio in ["VIDEO", "WEBCAM"]:
    if source_radio == "VIDEO":
        st.sidebar.header("Upload")
        input_file = st.sidebar.file_uploader("Choose a video.", type=("mp4"))

        if input_file is not None:
            g = io.BytesIO(input_file.read())
            temporary_location = "upload.mp4"
            with open(temporary_location, "wb") as out:
                out.write(g.read())

            run_object_detection(temporary_location, conf_threshold)
        else:
            st.video("assets/sample_video.mp4")
            st.write("Click on 'Browse Files' to run inference on a video.")

    elif source_radio == "WEBCAM":
        run_object_detection(0, conf_threshold)
