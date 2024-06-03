# Auto-X Engine

The Auto-X Engine is a deep learning inference engine designed for MCUs/CPUs. It is written entirely in C and runs on Auto-X private model format.

This engine only supports the model structures described in this repo at this time.

This repo also demonstrates how to connect ESP32 based cameras/PX4 based drones/Android based Robots to Eclipse Ditto via MQTT using Auto-X Engine.

## Model Pool

> **Note** Because image classification models are rarely used in practical scenarios, we do not privide such kind of models.

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
- [SAM](https://github.com/ggerganov/ggml)
- [whisper](https://github.com/ggerganov/ggml)
- [ChatGLM](https://github.com/ggerganov/ggml)
- [llama2](https://github.com/karpathy/llama2.c)


## Demos

### Timer Camera X

Timer Camera X is a camera module based on ESP32, integrated with ESP32 chip and 8M-PSRAM. The camera (ov3660) with 3 million pixels , DFOV 66.5 ° and shoot 2048x1536 resolution photo, built-in 140mAh battery and LED status indicator, featuring ultra-low power consumption design. There is a reset button under the LED. Through RTC (BM8563), timing sleep and wake-up can be realized. The standby current is only 2μA. After timing photo taking function(one photo per hour) is turned on, the battery can work continuously for more than one month. The module supports Wi-Fi image transmission and USB port debugging. The bottom HY2.0-4P port output can be connected to other peripherals.

Get more information about [Timer Camera X](https://docs.m5stack.com/en/unit/timercam_x).

This demo simulates the following industrial scenarios:
1. Fix the camera in front of the target objects in COCO dataset categories
2. The camera sends the camera status info. and recognition results to Eclipse Ditto at a fixed interval

###  Pixhawk 4 and RP Pi 4B

This demo simulates the following industrial scenarios:
1. Schedule the drone to autonomously collect images of specified objects at some specified places.
2. Exchange drone status with Eclipse Ditto during the flight
3. Upload the collected images to Eclipse Ditto when drone goes home

###  Android Robot


## References

- [Eclipse Ditto :: Examples](https://github.com/eclipse-ditto/ditto-examples)
- [M5_Camera_Examples](https://github.com/m5stack/M5_Camera_Examples)
- [PX4 Drone Autopilot](https://github.com/PX4/PX4-Autopilot)