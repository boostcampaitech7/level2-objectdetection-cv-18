import argparse
import csv
import json
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, LoadImages
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode

@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",  # model path
    source=ROOT / "test",  # directory containing test images
    json_file=ROOT / "test.json",  # path to the JSON file
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    save_csv=True,  # save results in CSV format
    project=ROOT / "runs/detect",  # save results to project/name
    name="predictions",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
):
    source = str(source)
    
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    # Define the path for the CSV file
    csv_path = save_dir / "predictions.csv"

    # Create or append to the CSV file
    def write_to_csv(image_id, prediction_string):
        """Writes prediction data for an image to a CSV file, appending if the file exists."""
        data = {"image_id": image_id, "PredictionString": prediction_string}
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if not csv_path.is_file():
                writer.writeheader()
            writer.writerow(data)

    # Run inference
    for path, img, im0s, vid_cap, s in dataset:
        img = torch.from_numpy(img).to(model.device)
        img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        pred = model(img)

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic_nms=False, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            # Prepare prediction string
            prediction_string = ""
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0s.shape).round()
                # Generate prediction string
                for *xyxy, conf, cls in det:
                    class_name = names[int(cls)]
                    prediction_string += f"{class_name} {conf:.2f} "

            # Extract image ID from path (you can modify this if needed)
            image_id = Path(path).name

            # Write predictions to CSV
            write_to_csv(image_id, prediction_string.strip())

if __name__ == "__main__":
    run()
