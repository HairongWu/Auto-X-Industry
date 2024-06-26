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

## Setup storages

1. For a specific project, open 'Settings > Storage'.
2. Click 'Add Source Storage' or 'Add Target Storage'. And fill the information needed.
3. (Optional) Toggle 'Treat every bucket object as a source file'. Enable this option if you want to create Label Studio tasks from media files automatically, such as JPG, MP3, or similar file types. Use this option for labeling configurations with one source tag. Disable this option if you want to import tasks in Label Studio JSON format directly from your storage. Use this option for complex labeling configurations with HyperText or multiple source tags.
4. Click 'Add Storage'.
5. After adding the storage, click 'Sync Storage'. If you configure target storage, annotations are sent to target storage after you click Sync for the configured target storage connection. The target storage receives a JSON-formatted export of each annotation. 

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

