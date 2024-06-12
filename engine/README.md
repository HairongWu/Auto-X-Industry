# Auto-X Engine

The Auto-X Engine is a deep learning inference engine designed for MCUs/CPUs. It is written entirely in C and runs on Auto-X private model format.

This engine only supports the model structures described in this repo at this time.

This repo also demonstrates how to connect ESP32 based cameras/PX4 based drones/Android based Robots to Auto-X Studio via MQTT using Auto-X Engine.

## Model Pool

> **Note** Because image classification models are rarely used in practical scenarios, we do not privide such kind of models.
> We also provide guidelines and running code to customize and retrain the following models using your own data.

### Models for MCU (such as ESP32 and Arm Cortex-M)

| Model Name   | Input size | Latency(s) |  Weight  | Config | Demo Firmware | Target Device |
| :-------- | :--------: | :---------------------: | :----------------: | :----------------: | :---------------: | :-----------------------------: |
| [PicoDet-XS](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.7/configs/picodet) |  192*192   |                     |               |                |               |    ESP32-S3-EYE           |
| [PP-Tinypose](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.7/configs/keypoint/tiny_pose) |  256*192   |                     |               |                |               |    ESP32-S3-EYE           |
| [PaddleOCR(mobile)](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/models_list_en.md) |     |                     |               |                |               |    ESP32-S3-EYE           |

### Models for CPU (such as Arm Cortex-A and X86)

| Model Name   | Input size | Latency(s) |  Weight  | Config | Demo Firmware | Target Device |
| :-------- | :--------: | :---------------------: | :----------------: | :----------------: | :---------------: | :-----------------------------: |
| [YOLO-World](https://github.com/AILab-CVC/YOLO-World) |     |                     |               |                |               |    Raspberry Pi 4          |
| [UniPose](https://github.com/IDEA-Research/UniPose) |     |                     |               |                |               |    Raspberry Pi 4          |
| [PP-Structure](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppstructure/docs/models_list_en.md) |    |                     |               |                |               |    Raspberry Pi 4          |
| [SAM](https://github.com/ggerganov/ggml) |     |                     |               |                |               |    Raspberry Pi 4           |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) |     |                     |               |                |               |    Raspberry Pi 4           |
| [ChatGLM](https://github.com/ggerganov/ggml) |     |                     |               |                |               |    Raspberry Pi 4           |
| [llama2](https://github.com/karpathy/llama2.c) |     |                     |               |                |               |    Raspberry Pi 4           |
| [Chronos](https://github.com/amazon-science/chronos-forecasting) |     |                     |               |                |               |    Raspberry Pi 4           |
| [StableTTS](https://github.com/KdaiP/StableTTS) |     |                     |               |                |               |    Raspberry Pi 4           |
| [GeneFace++](https://github.com/yerfor/GeneFacePlusPlus) |     |                     |               |                |               |    Raspberry Pi 4           |

## Model Converter

This tool can convert the pre-trained models generated from Pytorch, PaddlePaddle, ggml, and onnx to Auto-X format.
At this time, we only support models described in the previous section.


## Demos

### Timer Camera X

Get more information about the hardware at [Timer Camera X](https://docs.m5stack.com/en/unit/timercam_x)

This demo simulates the following industrial scenarios:
1. Fix the camera in front of the target object
2. run the Object Detection(COCO)/OCR(number, english, and so on)/Gauge Transcription model on device side
3. The camera sends the device info. and the recognition results to Auto-X Studio via Wi-Fi at a fixed interval


###  Raspberry Pi 4 Navio2 Autopilot

Get more information about the hardware at [Navio2](https://navio2.hipi.io/)

This demo simulates the following industrial scenarios:
1. Schedule the drone to autonomously collect images of specified objects at some specified places.
2. Exchange drone status with Auto-X Studio during the flight
3. Upload the collected images to Auto-X Studio when drone goes home


## References

- [ggml](https://github.com/ggerganov/ggml)
- [Eclipse Ditto :: Examples](https://github.com/eclipse-ditto/ditto-examples)
- [PX4 Drone Autopilot](https://github.com/PX4/PX4-Autopilot)
- [M5_Camera_Examples](https://github.com/m5stack/M5_Camera_Examples/tree/main)