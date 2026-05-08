#!/usr/bin/env python
"""
Standalone visualization script for ScoutBot detection results.

Usage:
    python visualise.py IMAGE.jpg IMAGE.json [--threshold THRESHOLD]

Displays the image with bounding boxes and confidence scores for each detection.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def load_detections(json_path):
    """Load detections from ScoutBot JSON output file."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # JSON structure: {image_path: {wic: float, loc: [detections]}}
    # Get the first (and typically only) image entry
    if not data:
        raise ValueError(f"Empty JSON file: {json_path}")

    image_key = list(data.keys())[0]
    return data[image_key]


def visualize_detections(image_path, json_path, threshold=0.0, output_path=None):
    """
    Visualize bounding boxes on an image with confidence scores.

    Args:
        image_path: Path to the input image
        json_path: Path to the ScoutBot JSON output
        threshold: Minimum confidence threshold to display (default: 0.0)
        output_path: Optional path to save the output image
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Load detections
    result = load_detections(json_path)
    detections = result.get('loc', [])
    wic_score = result.get('wic', 0.0)

    print(f"Image: {image_path}")
    print(f"Whole Image Confidence: {wic_score:.4f}")
    print(f"Total detections: {len(detections)}")

    # Filter by threshold
    filtered_detections = [d for d in detections if d['c'] >= threshold]
    print(f"Detections above threshold ({threshold}): {len(filtered_detections)}")

    # Draw bounding boxes
    for detect in filtered_detections:
        label = detect['l']
        conf = detect['c']

        # Calculate box coordinates
        x = int(np.around(detect['x']))
        y = int(np.around(detect['y']))
        w = int(np.around(detect['w']))
        h = int(np.around(detect['h']))

        point1 = (x, y)
        point2 = (x + w, y + h)

        # Draw rectangle (BGR color: blue)
        color = (255, 0, 0)
        thickness = 2
        cv2.rectangle(img, point1, point2, color, thickness)

        # Add label with confidence score
        text = f"{label}: {conf:.3f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1

        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, font_thickness
        )

        # Draw background rectangle for text (white with some transparency)
        text_bg_point1 = (x, y - text_height - baseline - 4)
        text_bg_point2 = (x + text_width + 4, y)
        cv2.rectangle(img, text_bg_point1, text_bg_point2, (255, 255, 255), -1)

        # Draw text
        text_position = (x + 2, y - baseline - 2)
        cv2.putText(img, text, text_position, font, font_scale, (0, 0, 0), font_thickness)

    # Save output if requested
    if output_path:
        cv2.imwrite(str(output_path), img)
        print(f"Saved output to: {output_path}")

    # Display image
    # Create window and display
    window_name = f"ScoutBot Detections - {Path(image_path).name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)

    print("\nPress any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize ScoutBot detection results with bounding boxes and confidence scores.'
    )
    parser.add_argument(
        'image',
        type=str,
        help='Path to the input image file'
    )
    parser.add_argument(
        'json',
        type=str,
        help='Path to the ScoutBot JSON output file'
    )
    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=0.0,
        help='Minimum confidence threshold to display (default: 0.0)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Optional path to save the output image'
    )

    args = parser.parse_args()

    # Validate inputs
    image_path = Path(args.image)
    json_path = Path(args.json)

    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else None

    # Run visualization
    try:
        visualize_detections(image_path, json_path, args.threshold, output_path)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
