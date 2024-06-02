__author__ = 'finecwg'

import argparse
import site_scrapper

def parse_args():
    parser = argparse.ArgumentParser(description="Web scrapper")

    parser.add_argument("--sitemap_url", type=str, required=True, help="web sitemap url")
    parser.add_argument('--output_dir', type=str, help="Directory to save the output files")
    parser.add_argument('--format', type=str, choices=['html', 'pdf'], default='html', help="Output format: 'html' or 'pdf'")
    parser.add_argument('--wkhtmltopdf_path', type=str, default='/usr/bin/wkhtmltopdf', help="Path to the wkhtmltopdf executable (required for PDF format)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    site_scrapper.main(args)