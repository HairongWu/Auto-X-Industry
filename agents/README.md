# Auto-X Agents


This projects is originally based on label-studio-ml-backend, so it is compatible with label-studio interfaces.
Auto-X AI Agents supports two types of tasks: training task and agent task.
The agent task is intefrated with the Auto-X Engine Serving, and the training task with the Auto-X Engine Training.

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
