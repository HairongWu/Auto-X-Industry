# Auto-X AI Server


This projects is originally based on label-studio-ml-backend, so it is compatible with label-studio interfaces.
Auto-X AI Server supports three types of tasks: prediction tasks, training tasks, LLM-based web scraping tasks and agent tasks.
The prediction tasks, training tasks, and LLM-based web scraping tasks can be used in Auto-X Studio, and agent tasks can be used in Auto-X Service.

## Environment Setup

1. Install [PyTorch](https://pytorch.org/) according to you hardwares

2. Install [PaddlePaddle](https://www.paddlepaddle.org.cn/documentation/docs/en/2.4/install/index_en.html) according to you hardwares

3. Install [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) and download the [model](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth)

4. Install [UniPose](https://github.com/IDEA-Research/UniPose) and download the [model](https://drive.google.com/file/d/13gANvGWyWApMFTAtC3ntrMgx0fOocjIa/view)

5. Install other dependencies

```bash
pip install -r requirements.txt
```

6. Set the model paths and start server for test purposes

```bash
python _wsgi.py --log-level DEBUG --groundingdino_model xxx --pose_model xxx
```

## References

- [label-studio-ml-backend](https://github.com/HumanSignal/label-studio-ml-backend)
- [RAGFlow](https://github.com/infiniflow/ragflow)
- [XAgent](https://github.com/OpenBMB/XAgent)