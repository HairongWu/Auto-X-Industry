import magic 

from Docs2KG.kg.pdf_layout_kg import PDFLayoutKG
from Docs2KG.kg.semantic_kg import SemanticKG
from Docs2KG.kg.utils.json2triplets import JSON2Triplets
from Docs2KG.kg.utils.neo4j_connector import Neo4jLoader
from Docs2KG.modules.llm.markdown2json import LLMMarkdown2Json
from Docs2KG.parser.pdf.pdf2blocks import PDF2Blocks
from Docs2KG.parser.pdf.pdf2metadata import PDF_TYPE_SCANNED, get_scanned_or_exported
from Docs2KG.parser.pdf.pdf2tables import PDF2Tables
from Docs2KG.parser.pdf.pdf2text import PDF2Text

from Docs2KG.kg.web_layout_kg import WebLayoutKG

from Docs2KG.kg.email_layout_kg import EmailLayoutKG
from Docs2KG.parser.email.email_compose import EmailDecompose

from Docs2KG.modules.llm.sheet2metadata import Sheet2Metadata
from Docs2KG.parser.excel.excel2image import Excel2Image
from Docs2KG.parser.excel.excel2markdown import Excel2Markdown
from Docs2KG.parser.excel.excel2table import Excel2Table
from Docs2KG.utils.constants import DATA_INPUT_DIR

from .pipeline import *

class GraphPipeline(Pipeline): 
    def __init__(self):
        super().__init__()

    def pdf2kg(self, pdf_file):
        scanned_or_exported = get_scanned_or_exported(pdf_file)
        if scanned_or_exported == PDF_TYPE_SCANNED:
                    # This will extract the text, images.
            #
            # Output images/text with bounding boxes into a df
            pdf_2_blocks = PDF2Blocks(pdf_file)
            blocks_dict = pdf_2_blocks.extract_df(output_csv=True)
            logger.info(blocks_dict)

            # Processing the text from the pdf file
            # For each page, we will have a markdown and text content,
            # Output will be in a csv

            pdf_to_text = PDF2Text(pdf_file)
            text = pdf_to_text.extract2text(output_csv=True)
            md_text = pdf_to_text.extract2markdown(output_csv=True)

            # Until now, your output folder should be some like this
            # .
            # ├── 4.pdf
            # ├── images
            # │         ├── blocks_images.csv
            # │         ├── page_0_block_1.jpeg
            # │         ├── page_0_block_4.jpeg
            # │         ├── ....
            # ├── metadata.json
            # └── texts
            #     ├── blocks_texts.csv
            #     ├── md.csv
            #     └── text.csv
            # under the image folder, are not valid image, we need better models for that.
            input_md_file = output_folder / "texts" / "md.csv"

            markdown2json = LLMMarkdown2Json(
                input_md_file,
                llm_model_name="gpt-3.5-turbo",
            )

            markdown2json.clean_markdown()
            markdown2json.markdown_file = output_folder / "texts" / "md.cleaned.csv"

            markdown2json.extract2json()

            pdf_2_image = PDF2Image(pdf_file)
            pdf_2_image.extract_page_2_image_df()
            # after this we will have a added `md.json.csv` in the `texts` folder

            # next we will start to extract the layout knowledge graph first

            layout_kg = PDFLayoutKG(output_folder, scanned_pdf=True)
            layout_kg.create_kg()

            # After this, you will have the layout.json in the `kg` folder

            # then we add the semantic knowledge graph
            semantic_kg = SemanticKG(
                output_folder, llm_enabled=True, input_format="pdf_scanned"
            )
            semantic_kg.add_semantic_kg()

            # After this, the layout_kg.json will be augmented with the semantic connections
            # in the `kg` folder

            # then we do the triplets extraction
            json_2_triplets = JSON2Triplets(output_folder)
            json_2_triplets.transform()

            # After this, you will have the triplets_kg.json in the `kg` folder
            # You can take it from here, load it into your graph db, or handle it in any way you want

            
        else:
            # This will extract the text, images.
            #
            # Output images/text with bounding boxes into a df
            pdf_2_blocks = PDF2Blocks(pdf_file)
            blocks_dict = pdf_2_blocks.extract_df(output_csv=True)
            logger.info(blocks_dict)

            # This will extract the tables from the pdf file
            # Output also will be the csv of the summary and each individual table csv

            pdf2tables = PDF2Tables(pdf_file)
            pdf2tables.extract2tables(output_csv=True)

            # Processing the text from the pdf file
            # For each page, we will have a markdown and text content,
            # Output will be in a csv

            pdf_to_text = PDF2Text(pdf_file)
            text = pdf_to_text.extract2text(output_csv=True)
            md_text = pdf_to_text.extract2markdown(output_csv=True)

            # Until now, your output folder should be some like this
            # .
            # ├── 4.pdf
            # ├── images
            # │         ├── blocks_images.csv
            # │         ├── page_0_block_1.jpeg
            # │         ├── page_0_block_4.jpeg
            # │         ├── ....
            # ├── metadata.json
            # ├── tables
            # │         ├── page_16-table_1.csv
            # │         ├── ....
            # │         └── tables.csv
            # └── texts
            #     ├── blocks_texts.csv
            #     ├── md.csv
            #     └── text.csv

            input_md_file = output_folder / "texts" / "md.csv"

            markdown2json = LLMMarkdown2Json(
                input_md_file,
                llm_model_name="gpt-3.5-turbo",
            )
            markdown2json.extract2json()

            # after this we will have a added `md.json.csv` in the `texts` folder

            # next we will start to extract the layout knowledge graph first

            layout_kg = PDFLayoutKG(output_folder)
            layout_kg.create_kg()
            # After this, you will have the layout.json in the `kg` folder

            # then we add the semantic knowledge graph
            semantic_kg = SemanticKG(output_folder, llm_enabled=True)
            semantic_kg.add_semantic_kg()

            # After this, the layout_kg.json will be augmented with the semantic connections
            # in the `kg` folder

            # then we do the triplets extraction
            json_2_triplets = JSON2Triplets(output_folder)
            json_2_triplets.transform()

            # After this, you will have the triplets_kg.json in the `kg` folder
            # You can take it from here, load it into your graph db, or handle it in any way you want

    def excel2kg(self, filename):
        excel2table = Excel2Table(excel_file=excel_file)
        excel2table.extract_tables_from_excel()

        excel2image = Excel2Image(excel_file=excel_file)
        excel2image.excel2image_and_pdf()

        excel2markdown = Excel2Markdown(excel_file=excel_file)
        excel2markdown.extract2markdown()

        sheet_2_metadata = Sheet2Metadata(
            excel2markdown.md_csv,
            llm_model_name="gpt-3.5-turbo",
        )
        sheet_2_metadata.extract_metadata()

    def webpages2kg(self, filename):
        """
        Extract the HTML file to images, markdown, tables, and urls and save it to the output directory

        1. Get html, images, markdown, tables, and urls from the given URL
        """
        url = "https://abs.gov.au/census/find-census-data/quickstats/2021/LGA57080"

        web_layout_kg = WebLayoutKG(url=url)
        web_layout_kg.create_kg()

        semantic_kg = SemanticKG(
            folder_path=web_layout_kg.output_dir, input_format="html", llm_enabled=True
        )
        semantic_kg.add_semantic_kg()

        json_2_triplets = JSON2Triplets(web_layout_kg.output_dir)
        json_2_triplets.transform()

    def email2kg(self, filename):
        email_decomposer = EmailDecompose(email_file=email_filename)
        email_decomposer.decompose_email()

        email_layout_kg = EmailLayoutKG(output_dir=email_decomposer.output_dir)
        email_layout_kg.create_kg()

        semantic_kg = SemanticKG(
            email_decomposer.output_dir, llm_enabled=True, input_format="email"
        )
        semantic_kg.add_semantic_kg()

        json_2_triplets = JSON2Triplets(email_decomposer.output_dir)
        json_2_triplets.transform()

    def predict(self, paths):
        results = []
        for file in paths:
            mime = magic.from_file(file, mime = True)
            if 'application/vnd.ms-excel' in mime:
                self.excel2kg(file)

            elif 'message/rfc822' in mime:
                self.email2kg(file)
            elif mime == 'application/pdf':
                self.pdf2kg(file)
            elif mime == 'text/html':
                self.webpages2kg(file)

        # If you want to load it into Neo4j, you can refer to the `examples/kg/utils/neo4j_connector.py`
        # to get it quickly loaded into Neo4j
        # You can do is run the `docker compose -f examples/compose/docker-compose.yml up`
        # So we will have a Neo4j instance running, then you can run the `neo4j_connector.py` to load the data
        uri = "bolt://localhost:7687"  # if it is a remote graph db, you can change it to the remote uri
        username = "neo4j"
        password = "testpassword"
        json_file_path = output_folder / "kg" / "triplets_kg.json"

        neo4j_loader = Neo4jLoader(uri, username, password, json_file_path, clean=True)
        neo4j_loader.load_data()
        neo4j_loader.close()
        return results
