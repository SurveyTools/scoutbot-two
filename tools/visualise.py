#!/usr/bin/env python3
"""
Standalone visualisation tool for ScoutBot detection results.

Usage:
    uv run python tools/visualise.py IMAGE.jpg DETECTIONS.json [OPTIONS]

Renders bounding boxes and confidence scores onto the source image.
Supports single-image and multi-image JSON output from the ScoutBot CLI.
"""

import argparse
import atexit
import json
import sys
import tempfile
import zlib
from pathlib import Path

import cv2

# Fixed BGR palette for per-class box colours. zlib.crc32 gives a stable hash
# across runs (unlike hash(), which is salted per-process since Python 3.3).
_PALETTE = [
    (255, 100, 0),
    (0, 200, 60),
    (0, 60, 255),
    (0, 200, 255),
    (200, 0, 200),
    (0, 200, 200),
]


def _box_color(label: str) -> tuple[int, int, int]:
    return _PALETTE[zlib.crc32(label.encode()) % len(_PALETTE)]


def _draw_label(img, text: str, x: int, y: int, box_h: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.5, 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 4

    if y < th + baseline + pad:
        bg_y1, bg_y2 = y + box_h, y + box_h + th + baseline + pad
        text_y = y + box_h + th + 2
    else:
        bg_y1, bg_y2 = y - th - baseline - pad, y
        text_y = y - baseline - 2

    cv2.rectangle(img, (x, bg_y1), (x + tw + pad, bg_y2), (255, 255, 255), -1)
    cv2.putText(img, text, (x + 2, text_y), font, scale, (0, 0, 0), thickness)


def _open_for_display(path: Path) -> None:
    import webbrowser

    # Hands a file:// URI to the OS default handler. On macOS this is typically
    # Preview; on Linux it delegates to xdg-open. Headless environments may no-op.
    webbrowser.open(path.as_uri())


def load_detections(json_path: Path, image_key: str | None = None) -> dict:
    data = json.loads(json_path.read_text())
    if not data:
        raise ValueError(f'Empty JSON file: {json_path}')

    if len(data) > 1 and image_key is None:
        keys = '\n  '.join(data)
        raise ValueError(
            f'JSON contains {len(data)} images. Specify one with --image-key:\n  {keys}'
        )

    if image_key is not None:
        if image_key not in data:
            available = ', '.join(repr(k) for k in data)
            raise ValueError(f'Image key {image_key!r} not found. Available: {available}')
        return data[image_key]

    return next(iter(data.values()))


def visualize_detections(
    image_path: Path,
    json_path: Path,
    threshold: float = 0.0,
    output_path: Path | None = None,
    display: bool = True,
    image_key: str | None = None,
) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f'Could not load image: {image_path}')

    result = load_detections(json_path, image_key=image_key)
    detections = result.get('loc', [])
    wic_score = result.get('wic', 0.0)

    print(f'Image: {image_path}')
    print(f'WIC score: {wic_score:.4f}')

    visible = [d for d in detections if d['c'] >= threshold]
    print(
        f'Detections: {len(visible)} above threshold {threshold} (total {len(detections)})'
    )

    for det in visible:
        label, conf = det['l'], det['c']
        x, y, w, h = round(det['x']), round(det['y']), round(det['w']), round(det['h'])
        cv2.rectangle(img, (x, y), (x + w, y + h), _box_color(label), 2)
        _draw_label(img, f'{label}: {conf:.3f}', x, y, h)

    if output_path is not None:
        cv2.imwrite(str(output_path), img)
        print(f'Saved: {output_path}')
        if display:
            _open_for_display(output_path)
    elif display:
        # atexit covers normal exit and unhandled exceptions but not SIGKILL or
        # interpreter crashes — temp files can survive those. Acceptable for a
        # developer utility; the OS reclaims /tmp on reboot regardless.
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            tmp = Path(f.name)
        atexit.register(tmp.unlink, missing_ok=True)
        cv2.imwrite(str(tmp), img)
        _open_for_display(tmp)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Render ScoutBot detections as bounding boxes on the source image.'
    )
    parser.add_argument('image', help='Source image file')
    parser.add_argument('json', help='ScoutBot JSON output file')
    parser.add_argument(
        '-t',
        '--threshold',
        type=float,
        default=0.0,
        help='Minimum confidence to display (default: 0.0)',
    )
    parser.add_argument('-o', '--output', help='Save annotated image to this path')
    parser.add_argument(
        '-k',
        '--image-key',
        help='Image key to use when JSON contains multiple images',
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Save output without opening a viewer (requires --output)',
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    json_path = Path(args.json)

    if not image_path.exists():
        print(f'Error: image not found: {image_path}', file=sys.stderr)
        sys.exit(1)
    if not json_path.exists():
        print(f'Error: JSON not found: {json_path}', file=sys.stderr)
        sys.exit(1)
    if args.no_display and args.output is None:
        print('Error: --no-display requires --output', file=sys.stderr)
        sys.exit(1)

    try:
        visualize_detections(
            image_path,
            json_path,
            threshold=args.threshold,
            output_path=Path(args.output) if args.output else None,
            display=not args.no_display,
            image_key=args.image_key,
        )
    except (ValueError, OSError) as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
