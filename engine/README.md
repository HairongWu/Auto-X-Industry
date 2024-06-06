# Auto-X Engine

The Auto-X Engine is a deep learning inference engine designed for MCUs/CPUs. It is written entirely in C and runs on Auto-X private model format.

This engine only supports the model structures described in this repo at this time.

This repo also demonstrates how to connect ESP32 based cameras/PX4 based drones/Android based Robots to Eclipse Ditto via MQTT using Auto-X Engine.

## Model Pool

> **Note** Because image classification models are rarely used in practical scenarios, we do not privide such kind of models.
> We also provide guidelines and running code to customize and retrain the following models using your own private data.

### Models for MCU (such as ESP32 and Arm Cortex-M)

- [PicoDet-XS](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.7/configs/picodet)
- [PP-Tinypose](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.7/configs/keypoint/tiny_pose)
- [PP-MobileSeg-Tiny](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.9/configs/pp_mobileseg)
- [PaddleOCR(mobile)](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/models_list_en.md)

### Models for CPU (such as Arm Cortex-A and X86)

- [Recognize Anything Model](https://github.com/xinyu1205/recognize-anything)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [UniPose](https://github.com/IDEA-Research/UniPose)
- [PaddleOCR(server)](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/models_list_en.md)
- [PP-Structure](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppstructure/docs/models_list_en.md)
- [SAM](https://github.com/ggerganov/ggml)
- [whisper](https://github.com/ggerganov/ggml)
- [ChatGLM](https://github.com/ggerganov/ggml)
- [llama2](https://github.com/karpathy/llama2.c)
- [Chronos](https://github.com/amazon-science/chronos-forecasting)
- [StableTTS](https://github.com/KdaiP/StableTTS)
- [GeneFace++](https://github.com/yerfor/GeneFacePlusPlus)

## Demos

### ESP32-S3-EYE

The ESP32-S3-EYE is a small-sized AI development board produced by Espressif. It is based on the ESP32-S3 SoC and ESP-WHO, Espressif’s AI development framework. It features a 2-Megapixel camera, an LCD display, and a microphone, which are used for image recognition and audio processing. ESP32-S3-EYE offers plenty of storage, with an 8 MB Octal PSRAM and a 8 MB flash. It also supports image transmission via Wi-Fi and debugging through a Micro-USB port. With ESP-WHO, you can develop a variety of AIoT applications, such as smart doorbell, surveillance systems, facial recognition time clock, etc.

Get more information about the hardware at [ESP32-S3-EYE](https://github.com/espressif/esp-who/blob/master/docs/en/get-started/ESP32-S3-EYE_Getting_Started_Guide.md)

This demo simulates the following industrial scenarios:
1. Fix the camera in front of the target objects in COCO dataset categories
2. The camera sends the camera status info. and the recognition results to Auto-X Studio at a fixed interval

###  Raspberry Pi 4 Navio2 Autopilot

Get more information about the hardware at [Navio2](https://navio2.hipi.io/)

This demo simulates the following industrial scenarios:
1. Schedule the drone to autonomously collect images of specified objects at some specified places.
2. Exchange drone status with Auto-X Studio during the flight
3. Upload the collected images to Auto-X Studio when drone goes home

###  Diesease Diagnosis with virtual human agent (Android)


## References

- [Eclipse Ditto :: Examples](https://github.com/eclipse-ditto/ditto-examples)
- [PX4 Drone Autopilot](https://github.com/PX4/PX4-Autopilot)