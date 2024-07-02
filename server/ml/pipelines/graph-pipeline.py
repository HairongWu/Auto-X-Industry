import magic 

from Docs2KG.parser.pdf.pdf2blocks import PDF2Blocks
from Docs2KG.parser.pdf.pdf2metadata import PDF_TYPE_SCANNED, get_scanned_or_exported
from Docs2KG.parser.pdf.pdf2tables import PDF2Tables
from Docs2KG.parser.pdf.pdf2text import PDF2Text
from Docs2KG.modules.llm.markdown2json import LLMMarkdown2Json
from Docs2KG.parser.pdf.pdf2image import PDF2Image

from Docs2KG.modules.llm.sheet2metadata import Sheet2Metadata
from Docs2KG.parser.excel.excel2image import Excel2Image
from Docs2KG.parser.excel.excel2markdown import Excel2Markdown
from Docs2KG.parser.excel.excel2table import Excel2Table

from Docs2KG.parser.email.utils.email_connector import EmailConnector
from Docs2KG.parser.email.email_compose import EmailDecompose

from .pipeline import *

class VideoPipeline(Pipeline): 
    def __init__(self):
        super().__init__()

       

    def predict(self, paths):
        results = []
        for file in paths:
            mime = magic.from_file(file, mime = True)
            if 'application/vnd.ms-excel' in mime:
                excel2table = Excel2Table(excel_file=file)
                excel2table.extract_tables_from_excel()

                excel2image = Excel2Image(excel_file=file)
                excel2image.excel2image_and_pdf()

                excel2markdown = Excel2Markdown(excel_file=file)
                excel2markdown.extract2markdown()

            elif 'message/rfc822' in mime:
                email_decomposer = EmailDecompose(email_file=file)
                email_decomposer.decompose_email()
            elif mime == 'application/pdf':
                scanned_or_exported = get_scanned_or_exported(file)
                if scanned_or_exported == PDF_TYPE_SCANNED:
                    # This will extract the text, images.
                    #
                    # Output images/text with bounding boxes into a df
                    pdf_2_blocks = PDF2Blocks(file)
                    blocks_dict = pdf_2_blocks.extract_df(output_csv=True)

                    # Processing the text from the pdf file
                    # For each page, we will have a markdown and text content,
                    # Output will be in a csv

                    pdf_to_text = PDF2Text(file)
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
                    pdf_2_image = PDF2Image(file)
                    pdf_2_image.extract_page_2_image_df()
                    # after this we will have a added `md.json.csv` in the `texts` folder
                else:
                    # This will extract the text, images.
                    #
                    # Output images/text with bounding boxes into a df
                    pdf_2_blocks = PDF2Blocks(file)
                    blocks_dict = pdf_2_blocks.extract_df(output_csv=True)

                    # This will extract the tables from the pdf file
                    # Output also will be the csv of the summary and each individual table csv

                    pdf2tables = PDF2Tables(file)
                    pdf2tables.extract2tables(output_csv=True)

                    # Processing the text from the pdf file
                    # For each page, we will have a markdown and text content,
                    # Output will be in a csv

                    pdf_to_text = PDF2Text(file)
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
            elif mime == 'text/':
                pass
        return results

    def train(self):
        pass

    def finetune(self):
        pass