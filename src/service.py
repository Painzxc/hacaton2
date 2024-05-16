import io
import json
import logging
import pydantic
import numpy as np
from typing import List
from fastapi import FastAPI, File, UploadFile, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
from src.datacontract.service_config import ServiceConfig
from src.datacontract.service_output import ServiceOutput
import torchvision.models as models
from scipy.spatial import distance as dist
from ultralytics import YOLO

app = FastAPI()

service_config_path = "./src/configs/service_config.json"
with open(service_config_path, "r") as service_config:
    service_config_json = json.load(service_config)

detector = YOLO(service_config_json["path_to_detector"])


@app.get(
    "/health",
    tags=["healthcheck"],
    summary="Perform health check",
    response_description="Return HTTP status code 200 (OK)",
    status_code=status.HTTP_200_OK,
)
def health_check() -> str:
    return '{"Status" : "OK"}'


@app.post("/file/", response_model=List[ServiceOutput])
async def inference(image: UploadFile = File(...)):

    image_content = await image.read()
    image = Image.open(io.BytesIO(image_content))
    image = image.convert("RGB")
    output_list = []

    # Perform object detection
    detector_outputs = detector(image)

    # Process each detected object
    for box in detector_outputs[0].boxes.xyxy:
        box_data = box.tolist()
        xtl, ytl, xbr, ybr = box_data

        output_list.append(
            ServiceOutput(
                objectid="0",
                classname="0",
                xtl=int(xtl),
                xbr=int(xbr),
                ytl=int(ytl),
                ybr=int(ybr),
            )
        )

    return output_list
