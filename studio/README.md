# Auto-X Studio

Auto-X Studio is based on Label Studio and provides full workflows to build or finetune your own AI solution models.
Auto-X Studio contains three types of interfaces to build solution models: document table building tool, 
knowledge graph building tool, and general purpose building tool.


## Run studio

You can run the latest Auto-X Studio locally without installing the package. 
User docker for production purposes.

```bash
# Install all package dependencies
pip install poetry
poetry install
# Run database migrations
python label_studio/manage.py migrate
python label_studio/manage.py collectstatic
# Start the server in development mode at http://localhost:8080
python label_studio/manage.py runserver
```

## Create Detect Anything Dataset

1. Create a project and select the 'Detect Anything' labeling template. Click 'Save'.
2. Start [Auto-X Agents Server](../agents) and Connect to Auto-X Agents
3. Go to the Data Manager, and import the image data
4. Select 'Actions > Retrieve predictions'.
5. Select 'Actions > Create Annotations from Predictions'.
6. Confirm the pre-annotated data manually

> **Note** You can also using existing Vector Database for pre-labeling.

## Create Document Recognition Dataset

## Create LLM Dataset

## Create Video Captioning Dataset


## Create Document Table Recognition Dataset


## Create Vector Database for Image Recognition


## Create Knowledge Graph Database



## Training (Not available for now)

1. Start the [Auto-X Training server](https://github.com/HairongWu/Auto-X-Training-Server)
2. Click 'Start Training' of dropdown menu of project settings.

