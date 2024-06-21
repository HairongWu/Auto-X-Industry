from .Docs2KG.parser.pdf.pdf2blocks import PDF2Blocks
from .Docs2KG.parser.pdf.pdf2metadata import PDF_TYPE_SCANNED, get_scanned_or_exported
from .Docs2KG.parser.pdf.pdf2tables import PDF2Tables
from .Docs2KG.parser.pdf.pdf2text import PDF2Text
from .Docs2KG.modules.llm.markdown2json import LLMMarkdown2Json
from .Docs2KG.parser.pdf.pdf2image import PDF2Image
from .Docs2KG.modules.llm.sheet2metadata import Sheet2Metadata
from .Docs2KG.parser.excel.excel2image import Excel2Image
from .Docs2KG.parser.excel.excel2markdown import Excel2Markdown
from .Docs2KG.parser.excel.excel2table import Excel2Table
from .Docs2KG.parser.email.utils.email_connector import EmailConnector

from scrapegraphai.graphs import SmartScraperGraph

from pymilvus import MilvusClient

class KnowledgePipeline(): 
    def __init__(self):
        super().__init__()

    def exported_pdfs():
        pass
    def scanned_pdfs():
        pass
    def web_pages():
        pass
    def excels():
        pass
    def emails():
        pass

    def from_docs(self):
        pass
    
    def from_scraping(self, source):
        graph_config = {
            "llm": {
                "model": "ollama/mistral",
                "temperature": 0,
                "format": "json",  # Ollama needs the format to be specified explicitly
                "base_url": "http://localhost:11434",  # set Ollama URL
            },
            "embeddings": {
                "model": "ollama/nomic-embed-text",
                "base_url": "http://localhost:11434",  # set Ollama URL
            },
            "verbose": True,
        }

        smart_scraper_graph = SmartScraperGraph(
            prompt="List me all the projects with their descriptions",
            # also accepts a string with the already downloaded HTML code
            source=source,
            config=graph_config
        )

        result = smart_scraper_graph.run()
        print(result)

