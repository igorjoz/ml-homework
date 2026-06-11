import asyncio
import sys
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
import nbformat
from nbconvert import WebPDFExporter

with open('skin_lesion_mlp_pytorch.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)
exporter = WebPDFExporter(allow_chromium_download=True)
pdf_data, _ = exporter.from_notebook_node(nb)
with open('skin_lesion_mlp_pytorch.pdf', 'wb') as f:
    f.write(pdf_data)