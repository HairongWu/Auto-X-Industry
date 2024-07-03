# Auto-X Agents


This projects is originally based on label-studio-ml-backend, so it is compatible with label-studio interfaces.
Auto-X AI Server supports three types of tasks: prediction tasks, training tasks, and agent tasks.
The prediction tasks, training tasks, and LLM-based web scraping tasks can be used in Auto-X Studio, and agent tasks can be used in Auto-X Service.

## Environment Setup

1. Install [PyTorch](https://pytorch.org/) according to you hardwares

2. Install [PaddlePaddle](https://www.paddlepaddle.org.cn/documentation/docs/en/2.4/install/index_en.html) according to you hardwares

3. Install Deep Lake
```bash
pip install "deeplake[all]"
```
4. Install [ArangoDB](https://github.com/arangodb/arangodb) according to you hardwares
```bash
pip install pyarango --user
```
5. Install other dependencies

```bash
pip install -r requirements.txt
```

6. Download models and set the 'model_pool' paths in '_wsgi.py' and start server for test purposes

```bash
python _wsgi.py --log-level DEBUG  --ls-access-token xxxxxxx --ls-url http://127.0.0.1:8080/
```

## References

- [label-studio-ml-backend](https://github.com/HumanSignal/label-studio-ml-backend)

- [RAGFlow](https://github.com/infiniflow/ragflow)
- [XAgent](https://github.com/OpenBMB/XAgent)
- [crewAI](https://github.com/joaomdmoura/crewAI)

- [VideoLLaMA 2](https://github.com/DAMO-NLP-SG/VideoLLaMA2)
- [anomalib](https://github.com/openvinotoolkit/anomalib)
- [EmoLLM](https://github.com/SmartFlowAI/EmoLLM)

- [PP-Structure](https://github.com/PaddlePaddle/PaddleOCR/tree/main/ppstructure)

- [Chronos](https://github.com/amazon-science/chronos-forecasting)

- [V-Express](https://github.com/tencent-ailab/V-Express)

- [ScrapeGraphAI](https://github.com/VinciGit00/Scrapegraph-ai)
- [crawl4ai](https://github.com/unclecode/crawl4ai)

- [Docs2KG](https://github.com/AI4WA/Docs2KG)
- [Extract, Define, Canonicalize](https://github.com/clear-nus/edc)


- [MetaGPT](https://github.com/geekan/MetaGPT)
- [aiXcoder](https://github.com/aixcoder-plugin/aiXcoder-7B)

- [Deep Lake](https://github.com/activeloopai/deeplake)