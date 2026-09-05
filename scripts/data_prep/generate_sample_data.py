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


def _load_font(kind: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_CANDIDATES[kind]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def _blank_page() -> Image.Image:
    return Image.new("L", PAGE_SIZE, color=255)


def _draw_lines(
    draw: ImageDraw.ImageDraw, x: int, y: int, lines: list[tuple], line_gap: int = 18
) -> int:
    """Draw (text, font) pairs stacked vertically; returns the final y position."""
    for text, font in lines:
        draw.text((x, y), text, fill=0, font=font)
        bbox = draw.textbbox((x, y), text, font=font)
        y = bbox[3] + line_gap
    return y


def build_report_page(
    title_font, label_font, body_font, report_id: str, page_no: int
) -> tuple[Image.Image, list[str]]:
    """Draw one report page and return (image, ground_truth_lines).

    ``ground_truth_lines`` holds the same text in reading order, built from
    the exact strings drawn onto the page so the label can never drift from
    what the image actually shows.
    """
    img = _blank_page()
    draw = ImageDraw.Draw(img)
    x = MARGIN
    y = MARGIN

    header_lines = ["ACME LOGISTICS", "Delivery Inspection Report"]
    y = _draw_lines(
        draw,
        x,
        y,
        [
            (header_lines[0], title_font),
            (header_lines[1], label_font),
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
    field_lines = []
    for label, value in fields:
        draw.text((x, y), label, fill=0, font=label_font)
        draw.text((x + 420, y), value, fill=0, font=body_font)
        field_lines.append(f"{label} {value}")
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
    line: list[str] = []
    cur_len = 0
    max_chars = 62
    notes_lines = []
    for word in words:
        if cur_len + len(word) + 1 > max_chars:
            drawn = " ".join(line)
            draw.text((x, y), drawn, fill=0, font=body_font)
            notes_lines.append(drawn)
            y += 55
            line, cur_len = [], 0
        line.append(word)
        cur_len += len(word) + 1
    if line:
        drawn = " ".join(line)
        draw.text((x, y), drawn, fill=0, font=body_font)
        notes_lines.append(drawn)
        y += 55

    footer = f"Page {page_no}"
    draw.text(
        (PAGE_SIZE[0] - MARGIN - 120, PAGE_SIZE[1] - MARGIN),
        footer,
        fill=0,
        font=body_font,
    )

    ground_truth_lines = (
        header_lines + field_lines + ["Notes:"] + notes_lines + [footer]
    )
    return img, ground_truth_lines


def write_ground_truth(
    pdf_path: Path, page_lines: list[list[str]], labels_root: Path = Path("data/labels")
) -> None:
    """Write one label file per page, matching the <doc>_<page>.png naming
    that convert_pdfs_to_images.py gives the corresponding page image."""
    doc_dir = labels_root / pdf_path.stem
    doc_dir.mkdir(parents=True, exist_ok=True)
    for idx, lines in enumerate(page_lines, start=1):
        label_path = doc_dir / f"{pdf_path.stem}_{idx:03}.txt"
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_sample_pdf(output_path: Path, num_pages: int = 2) -> None:
    title_font = _load_font("bold", 54)
    label_font = _load_font("bold", 34)
    body_font = _load_font("regular", 34)

    built = [
        build_report_page(
            title_font,
            label_font,
            body_font,
            report_id=f"RPT-{1000 + i}",
            page_no=i + 1,
        )
        for i in range(num_pages)
    ]
    pages = [img for img, _ in built]
    page_lines = [lines for _, lines in built]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # PAGE_SIZE is 1700x2200px, sized for a 200 DPI US Letter page (8.5x11in).
    # Without explicit resolution, PIL assumes 72 DPI and the PDF page balloons
    # to ~23x30in, which then gets upscaled/blurred badly when re-rasterized.
    pages[0].save(output_path, save_all=True, append_images=pages[1:], resolution=200.0)
    write_ground_truth(output_path, page_lines)


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
    parser.add_argument("--num_pages", type=int, default=2, help="Pages per sample PDF")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    for i in range(args.num_docs):
        suffix = "" if args.num_docs == 1 else f"_{i + 1}"
        pdf_path = out_dir / f"sample_delivery_report{suffix}.pdf"
        generate_sample_pdf(pdf_path, num_pages=args.num_pages)
        print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
