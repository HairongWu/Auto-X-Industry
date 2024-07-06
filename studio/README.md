# Auto-X Studio

Auto-X Studio provides full workflows to build or finetune your own AI solution models.


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

## Workflow

1. Create a project and select the appropriate labeling template. Click 'Save'.
<div  align="center">
  <img src="../assets/lspr/select_template.png" width="500"/>
</div>

2. Start [Auto-X Agents](../agents) and Connect to Auto-X Agents
<div  align="center">
  <img src="../assets/lspr/autox_agents.png" width="500"/>
</div>

3. Go to the Data Manager, and import the data
4. Select 'Actions > Retrieve predictions'.
5. Select 'Actions > Create Annotations from Predictions'.
6. Confirm the pre-annotated data manually and modify if necessary
7. Start the [Auto-X Engine Training Server](https://github.com/HairongWu/Auto-X-Engine)(Not available for now)
8. Click 'Start Training' of dropdown menu of project settings.(Not available for now)


## Demos

### Create Detect Anything Dataset


### Create Document Recognition Dataset (Layout Analysis and OCR)

### Create LLM Dataset

### Create Video Captioning Dataset


### Create Document Table Recognition Dataset


### Create Vector Database


### Create Knowledge Graph Database



