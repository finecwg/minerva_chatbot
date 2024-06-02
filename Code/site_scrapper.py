__author__ = 'finecwg'

import requests
from bs4 import BeautifulSoup
import pdfkit
import os


def fetch_sitemap(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return [loc.text for loc in soup.find_all('loc')]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching sitemap: {e}")
        return []

class SitemapToHtml:
    def __init__(self, sitemap_url, output_dir):
        self.sitemap_url = sitemap_url
        self.output_dir = output_dir

    def save_webpage_as_html(self, url):
        # Fetch the webpage content from the given URL and save it as an HTML file in the output directory.

        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            html_file = os.path.join(self.output_dir, f"{url.split('//')[-1].replace('/', '_')}.html")

            with open(html_file, 'w', encoding='utf-8') as file:
                file.write(soup.prettify())

            print(f"Saved HTML: {html_file}")
        except Exception as e:
            print(f"Failed to process {url}: {e}")

    def process_sitemap(self):
        # Process the sitemap by fetching all URLs and saving their webpage content as HTML files.
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        pages = fetch_sitemap(self.sitemap_url)
        for page in pages:
            print(f"Processing: {page}")
            self.save_webpage_as_html(page)


class SitemapToPDF:
    def __init__(self, sitemap_url, output_dir, wkhtmltopdf_path='/usr/bin/wkhtmltopdf'):
        self.sitemap_url = sitemap_url
        self.output_dir = output_dir
        self.config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)

    def save_webpage_as_pdf(self, url):
        """
        Fetch the webpage content from the given URL and save it as a PDF file in the output directory.
        """
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            html_file = os.path.join(self.output_dir, 'temp.html')
            pdf_file = os.path.join(self.output_dir, f"{url.split('//')[-1].replace('/', '_')}.pdf")

            with open(html_file, 'w', encoding='utf-8') as file:
                file.write(soup.prettify())

            pdfkit.from_file(html_file, pdf_file, configuration=self.config)
            print(f"Saved PDF: {pdf_file}")
            os.remove(html_file)
        except Exception as e:
            print(f"Failed to process {url}: {e}")

    def process_sitemap(self):
        """
        Process the sitemap by fetching all URLs and saving their webpage content as PDF files.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        pages = fetch_sitemap(self.sitemap_url)
        for page in pages:
            print(f"Processing: {page}")
            self.save_webpage_as_pdf(page)

def main(args):

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    pages = fetch_sitemap(args.sitemap_url)

    if args.format == 'html':
        processor = SitemapToHtml(args.sitemap_url, args.output_dir)
    elif args.format == 'pdf':
        processor = SitemapToPDF(args.sitemap_url, args.output_dir, wkhtmltopdf_path=args.wkhtmltopdf_path)
    else:
        raise ValueError("Invalid format. Use 'html' or 'pdf'.")

    processor.process_sitemap()

