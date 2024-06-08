# Auto-X Engine

The Auto-X Engine is a deep learning inference engine designed for MCUs/CPUs. It is written entirely in C and runs on Auto-X private model format.

This engine only supports the model structures described in this repo at this time.

This repo also demonstrates how to connect ESP32 based cameras/PX4 based drones/Android based Robots to Auto-X Studio via MQTT using Auto-X Engine.

## Model Pool

> **Note** Because image classification models are rarely used in practical scenarios, we do not privide such kind of models.
> We also provide guidelines and running code to customize and retrain the following models using your own data.

### Models for MCU (such as ESP32 and Arm Cortex-M)

- [PicoDet-XS](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.7/configs/picodet)
- [PP-Tinypose](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.7/configs/keypoint/tiny_pose)
- [PP-MobileSeg-Tiny](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.9/configs/pp_mobileseg)
- [PaddleOCR(mobile)](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/models_list_en.md)

### Models for CPU (such as Arm Cortex-A and X86)

- [YOLO-World](https://github.com/AILab-CVC/YOLO-World)
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

## Model Converter

This tool can convert pre-trained models to Auto-X format.
At this time, we only support models described in the previous section.


## References

- [ggml](https://github.com/ggerganov/ggml)