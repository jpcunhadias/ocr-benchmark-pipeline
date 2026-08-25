"""
Generates a small set of synthetic "scanned document" PDFs so the OCR
pipeline can be run end-to-end without any real input data.

Usage:
    python scripts/data_prep/generate_sample_data.py
    python scripts/data_prep/generate_sample_data.py --output_dir data/pdf/2025-04/sample-report
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PAGE_SIZE = (1700, 2200)  # ~200 DPI letter-ish page
MARGIN = 140

_FONT_CANDIDATES = {
    "regular": [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ],
    "bold": [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ],
}


def _load_font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES[kind]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def _blank_page() -> Image.Image:
    return Image.new("L", PAGE_SIZE, color=255)


def _draw_lines(draw: ImageDraw.ImageDraw, x: int, y: int, lines: list[tuple], line_gap: int = 18) -> int:
    """Draw (text, font) pairs stacked vertically; returns the final y position."""
    for text, font in lines:
        draw.text((x, y), text, fill=0, font=font)
        bbox = draw.textbbox((x, y), text, font=font)
        y = bbox[3] + line_gap
    return y


def build_report_page(title_font, label_font, body_font, report_id: str, page_no: int) -> Image.Image:
    img = _blank_page()
    draw = ImageDraw.Draw(img)
    x = MARGIN
    y = MARGIN

    y = _draw_lines(
        draw,
        x,
        y,
        [
            ("ACME LOGISTICS", title_font),
            ("Delivery Inspection Report", label_font),
        ],
        line_gap=30,
    )
    draw.line((x, y, PAGE_SIZE[0] - MARGIN, y), fill=0, width=3)
    y += 50

    fields = [
        ("Report ID:", report_id),
        ("Date:", "2025-04-14"),
        ("Route:", "Warehouse 4 -> Distribution Center B"),
        ("Inspector:", "J. Alvarez"),
        ("Status:", "PASSED"),
    ]
    for label, value in fields:
        draw.text((x, y), label, fill=0, font=label_font)
        draw.text((x + 420, y), value, fill=0, font=body_font)
        y += 70

    y += 30
    draw.text((x, y), "Notes:", fill=0, font=label_font)
    y += 60
    notes = (
        "All packages were inspected on arrival and no damage was found. "
        "Seals matched the manifest and temperature logs were within the "
        "expected range for the full duration of transport. Cleared for "
        "unloading and inventory intake."
    )
    words = notes.split()
    line, cur_len = [], 0
    max_chars = 62
    for word in words:
        if cur_len + len(word) + 1 > max_chars:
            draw.text((x, y), " ".join(line), fill=0, font=body_font)
            y += 55
            line, cur_len = [], 0
        line.append(word)
        cur_len += len(word) + 1
    if line:
        draw.text((x, y), " ".join(line), fill=0, font=body_font)
        y += 55

    footer = f"Page {page_no}"
    draw.text((PAGE_SIZE[0] - MARGIN - 120, PAGE_SIZE[1] - MARGIN), footer, fill=0, font=body_font)
    return img


def generate_sample_pdf(output_path: Path, num_pages: int = 2) -> None:
    title_font = _load_font("bold", 54)
    label_font = _load_font("bold", 34)
    body_font = _load_font("regular", 34)

    pages = [
        build_report_page(title_font, label_font, body_font, report_id=f"RPT-{1000 + i}", page_no=i + 1)
        for i in range(num_pages)
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # PAGE_SIZE is 1700x2200px, sized for a 200 DPI US Letter page (8.5x11in).
    # Without explicit resolution, PIL assumes 72 DPI and the PDF page balloons
    # to ~23x30in, which then gets upscaled/blurred badly when re-rasterized.
    pages[0].save(
        output_path, save_all=True, append_images=pages[1:], resolution=200.0
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic sample PDFs for an end-to-end pipeline run"
    )
    parser.add_argument(
        "--output_dir",
        default="data/pdf/2025-04/sample-report",
        help="Directory to write the sample PDF(s) into",
    )
    parser.add_argument(
        "--num_docs", type=int, default=1, help="How many sample PDFs to generate"
    )
    parser.add_argument(
        "--num_pages", type=int, default=2, help="Pages per sample PDF"
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    for i in range(args.num_docs):
        suffix = "" if args.num_docs == 1 else f"_{i + 1}"
        pdf_path = out_dir / f"sample_delivery_report{suffix}.pdf"
        generate_sample_pdf(pdf_path, num_pages=args.num_pages)
        print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
