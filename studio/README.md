# Auto-X Studio

Auto-X Studio is based on Label Studio and provides full functions to build AI solutions from the start.

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
