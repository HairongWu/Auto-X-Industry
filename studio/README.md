# Auto-X Studio

Auto-X Studio is based on Label Studio with the following features:
- basic labeling functions of Label Studio (with modifications)
- exchange data with Eclipse Ditto™
- device mangement under projects
- modify the pre-labeling, fine-tuning and training behaviors
- import PDFs directly from user interfaces
- auto-generated label templates
- support knowledge graph creation tasks
- schedule tasks with edge devices or Auto-X Server

## Install for local development

You can run the latest Label Studio version locally without installing the package from pypi. 

```bash
# Install all package dependencies
pip install poetry
poetry install
# Run database migrations
python label_studio/manage.py migrate
python label_studio/manage.py collectstatic
# Start the server in development mode at http://localhost:8080
python label_studio/manage.py runserver


## References

- [Eclipse Ditto™ Client SDKs](https://github.com/eclipse-ditto/ditto-clients)