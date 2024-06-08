# Auto-X Demos

## ESP32-S3-EYE

The ESP32-S3-EYE is a small-sized AI development board produced by Espressif. It is based on the ESP32-S3 SoC and ESP-WHO, Espressif’s AI development framework. It features a 2-Megapixel camera, an LCD display, and a microphone, which are used for image recognition and audio processing. ESP32-S3-EYE offers plenty of storage, with an 8 MB Octal PSRAM and a 8 MB flash. It also supports image transmission via Wi-Fi and debugging through a Micro-USB port. With ESP-WHO, you can develop a variety of AIoT applications, such as smart doorbell, surveillance systems, facial recognition time clock, etc.

Get more information about the hardware at [ESP32-S3-EYE](https://github.com/espressif/esp-who/blob/master/docs/en/get-started/ESP32-S3-EYE_Getting_Started_Guide.md)

This demo simulates the following industrial scenarios:
1. Fix the camera in front of the target object
2. run the Object Detection(COCO)/OCR(number, english, and so on)/Gauge Transcription model on device side
3. The camera sends the device info. and the recognition results to Auto-X Studio via Wi-Fi at a fixed interval

### Validate the Object Detection Model


### Retrain the Object Detection Model

Please follow [here](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.7/configs/picodet/README_en.md) to retrain the model.

But use this [configurations](./ESP32-S3-EYE/PaddleDetection) instead.


### Run the whole demo


##  Raspberry Pi 4 Navio2 Autopilot

Get more information about the hardware at [Navio2](https://navio2.hipi.io/)

This demo simulates the following industrial scenarios:
1. Schedule the drone to autonomously collect images of specified objects at some specified places.
2. Exchange drone status with Auto-X Studio during the flight
3. Upload the collected images to Auto-X Studio when drone goes home

##  Virtual doctor with virtual human agent (Android)

This demo simulates the following industrial scenarios on Android Devices:
1. Ask questions about your symptoms via virtual huamn agent.
2. Anaylze video and speech data.
3. Generate diagnosis report and explanations


## References

- [Eclipse Ditto :: Examples](https://github.com/eclipse-ditto/ditto-examples)
- [PX4 Drone Autopilot](https://github.com/PX4/PX4-Autopilot)