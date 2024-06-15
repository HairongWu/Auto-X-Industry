# Auto-X Engine

The Auto-X Engine is a deep learning inference engine designed for MCUs/CPUs. It is written entirely in C and runs on Auto-X private model format. Most of the codes are adapted from [ggml](https://github.com/ggerganov/ggml), [Paddle Lite](https://github.com/PaddlePaddle/Paddle-Lite),
[OpenCV](https://github.com/opencv/opencv), [darknet](https://github.com/pjreddie/darknet) and [llama2.c](https://github.com/karpathy/llama2.c).

This engine only supports the model structures described in this repo at this time. And these models are needed in the built-in solutions.

This repo also demonstrates how to connect ESP32 based cameras/PX4 based drones/Android based Robots to Auto-X Studio via MQTT using Auto-X Engine.

## Model Pool

> **Note** The following models could be modified from the originial ones.
> We also provide guidelines and running code to customize and retrain the following models using your own data.

### Models for MCU (such as ESP32 and Arm Cortex-M)


### Models for CPU (such as Arm Cortex-A and X86)

1. Llama2
   The demo resides in the 'demos' folder with a MSVS project. As to the model downloading and other details, please refer to [here](https://github.com/karpathy/llama2.c).
   
2. Whisper

## Demos

### Timer Camera X

Get more information about the hardware at [Timer Camera X](https://docs.m5stack.com/en/unit/timercam_x)

This demo simulates the following industrial scenarios:
1. Fix the camera in front of the target object
2. run the Object Detection(COCO)/OCR(number, english, and so on)/Gauge Transcription model on device side
3. The camera sends the device info. and the recognition results to Auto-X Studio via Wi-Fi at a fixed interval

To connect a ESP32 device to Eclipse Ditto, please refer to [here](https://github.com/eclipse-ditto/ditto-examples/tree/master/mqtt-bidirectional).

###  Raspberry Pi 4 Navio2 Autopilot

Get more information about the hardware at [Navio2](https://navio2.hipi.io/)

This demo simulates the following industrial scenarios:
1. Schedule the drone to autonomously collect images of specified objects at some specified places.
2. Exchange drone status with Auto-X Studio during the flight
3. Upload the collected images to Auto-X Studio when drone goes home


## References

- [Eclipse Ditto :: Examples](https://github.com/eclipse-ditto/ditto-examples)
- [PX4 Drone Autopilot](https://github.com/PX4/PX4-Autopilot)
- [M5_Camera_Examples](https://github.com/m5stack/M5_Camera_Examples/tree/main)
- [ESP32-S3-EYE](https://github.com/W00ng/ESP32-S3-EYE)
- [FreeRTOS](https://github.com/FreeRTOS/FreeRTOS)