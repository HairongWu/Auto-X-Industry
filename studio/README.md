# Auto-X Studio

Auto-X Studio is based on Label Studio and provides full functions to build AI solutions from the start.

It has the following features:
- exchange data with Eclipse Ditto™ or data server
- device mangement
- full workflow to build models: pre-labeling, labeling manually, fine-tuning/training and deploy
- built-in solution templates
- support knowledge graph creation tasks
- schedule tasks with edge devices or Auto-X Server

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

## References

- [Eclipse Ditto™ Client SDKs](https://github.com/eclipse-ditto/ditto-clients)