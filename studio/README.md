# Auto-X Studio

Auto-X Studio is based on Label Studio and provides full workflows to build or finetune your own AI solution models.

## Install for local development

You can run the latest Auto-X Studio locally without installing the package. 

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

## Connect to Auto-X Server

1. Open the project settings and select 'Model'. Click 'Connect Model'
2. Set your Auto-X Server configurations

## Pre-labeling

1. Select a template from the available templates.
2. Click 'Save'.
3. Go to the Data Manager, select the tasks you want to get predictions for, and then select 'Actions > Retrieve predictions'.

## Training

1. For a specific project, open 'Settings > Model'.
2. Select 'Start Training' option of top right dropdown menu in the connnected Auto-X Server Card.
3. The pre-labeling model will be updated automatiocally when finished. And you check the training status in Auto-X Server side.

## Examples

