#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLI for ScoutBot v3

A simplified command-line interface for running YOLOv8-based wildlife detection.
"""

import json
import os
from os.path import exists

import click

import scoutbot
from scoutbot import loc, log


def pipeline_filepath_validator(ctx, param, value):
    if not exists(value):
        log.error(f'Input filepath does not exist: {value}')
        ctx.exit(1)
    return value


model_option = [
    click.option(
        '--config',
        help='Which ML model to use for inference',
        default='v3',
        type=click.Choice([k for k in loc.CONFIGS if k is not None]),
    ),
]

output_option = [
    click.option(
        '--output',
        help='Path to output JSON (if unspecified, results are printed to screen)',
        default=None,
        type=str,
    ),
]

v3_options = [
    click.option(
        '--device',
        help='Specifies the device for inference (see YOLO and PyTorch documentation)',
        default=os.getenv('SCOUTBOT_DEVICE', 'cuda:0'),
        type=click.Choice(['cuda:0', 'cuda', 'mps', 'cpu']),
    ),
    click.option(
        '--loc-thresh',
        help='Detection confidence threshold (0-100)',
        default=45,
        type=click.IntRange(0, 100, clamp=True),
    ),
]


def add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func

    return _add_options


@click.command('pipeline')
@click.argument(
    'filepath',
    nargs=1,
    type=str,
    callback=pipeline_filepath_validator,
)
@add_options(model_option)
@add_options(v3_options)
@add_options(output_option)
def pipeline(
    filepath,
    config,
    output,
    device,
    loc_thresh,
):
    """
    Run the ScoutBot pipeline on an input image filepath.

    Example output JSON:

    \b
    {
        '/path/to/image.ext': {
            'wic': 0.85,
            'loc': [
                {
                    'l': 'elephant',
                    'c': 0.9,
                    'x': 100,
                    'y': 100,
                    'w': 50,
                    'h': 10
                },
                ...
            ],
        }
    }
    """
    loc_thresh_float = loc_thresh / 100.0

    wic_, detects = scoutbot.pipeline(
        filepath,
        config=config,
        backend_device=device,
        loc_thresh=loc_thresh_float,
        slice_height=loc.CONFIGS[config]['slice_height'],
        slice_width=loc.CONFIGS[config]['slice_width'],
        overlap_height_ratio=loc.CONFIGS[config]['overlap_height_ratio'],
        overlap_width_ratio=loc.CONFIGS[config]['overlap_width_ratio'],
        perform_standard_pred=loc.CONFIGS[config]['perform_standard_pred'],
        postprocess_class_agnostic=loc.CONFIGS[config]['postprocess_class_agnostic'],
    )

    data = {
        filepath: {
            'wic': wic_,
            'loc': detects,
        }
    }

    log.debug('Outputting results...')
    if output:
        with open(output, 'w') as outfile:
            json.dump(data, outfile)
    else:
        print(json.dumps(data, indent=2))


@click.command('batch')
@click.argument(
    'filepaths',
    nargs=-1,
    type=str,
)
@add_options(model_option)
@add_options(v3_options)
@add_options(output_option)
@click.option(
    '--no-progress',
    is_flag=True,
    help='Disable progress bar',
)
def batch(
    filepaths,
    config,
    output,
    device,
    loc_thresh,
    no_progress,
):
    """
    Run the ScoutBot pipeline in batch on a list of input image filepaths.

    Example output JSON:

    \b
    {
        '/path/to/image1.ext': {
            'wic': 0.85,
            'loc': [ ... ],
        },
        '/path/to/image2.ext': {
            'wic': 0.12,
            'loc': [],
        },
        ...
    }
    """
    loc_thresh_float = loc_thresh / 100.0

    log.debug(f'Running batch on {len(filepaths)} files...')

    wic_list, detects_list = scoutbot.batch(
        filepaths,
        config=config,
        backend_device=device,
        loc_thresh=loc_thresh_float,
        slice_height=loc.CONFIGS[config]['slice_height'],
        slice_width=loc.CONFIGS[config]['slice_width'],
        overlap_height_ratio=loc.CONFIGS[config]['overlap_height_ratio'],
        overlap_width_ratio=loc.CONFIGS[config]['overlap_width_ratio'],
        perform_standard_pred=loc.CONFIGS[config]['perform_standard_pred'],
        postprocess_class_agnostic=loc.CONFIGS[config]['postprocess_class_agnostic'],
        show_progress=not no_progress,
    )
    results = zip(filepaths, wic_list, detects_list)

    data = {}
    for filepath, wic_, detects in results:
        data[filepath] = {
            'wic': wic_,
            'loc': detects,
        }

    log.debug('Outputting results...')
    if output:
        with open(output, 'w') as outfile:
            json.dump(data, outfile)
    else:
        print(json.dumps(data, indent=2))


@click.command('batch-resumable')
@click.argument(
    'filepaths',
    nargs=-1,
    type=str,
)
@click.option(
    '--output',
    '-o',
    required=True,
    help='Output JSONL file path (results saved incrementally)',
    type=str,
)
@add_options(model_option)
@add_options(v3_options)
@click.option(
    '--no-progress',
    is_flag=True,
    help='Disable progress bar',
)
def batch_resumable(
    filepaths,
    output,
    config,
    device,
    loc_thresh,
    no_progress,
):
    """
    Process images in batch with automatic resume capability.

    Results are saved incrementally to a JSONL file. If interrupted,
    run the same command again to resume from where it stopped.

    Example output (each line is a JSON object):

    \b
    {"filepath": "/path/to/img1.jpg", "wic": 0.85, "detections": [...], ...}
    {"filepath": "/path/to/img2.jpg", "wic": 0.12, "detections": [], ...}
    """
    loc_thresh_float = loc_thresh / 100.0

    log.debug(f'Running resumable batch on {len(filepaths)} files...')
    log.debug(f'Output file: {output}')

    summary = scoutbot.batch_resumable(
        filepaths=list(filepaths),
        output_path=output,
        config=config,
        backend_device=device,
        loc_thresh=loc_thresh_float,
        slice_height=loc.CONFIGS[config]['slice_height'],
        slice_width=loc.CONFIGS[config]['slice_width'],
        overlap_height_ratio=loc.CONFIGS[config]['overlap_height_ratio'],
        overlap_width_ratio=loc.CONFIGS[config]['overlap_width_ratio'],
        perform_standard_pred=loc.CONFIGS[config]['perform_standard_pred'],
        postprocess_class_agnostic=loc.CONFIGS[config]['postprocess_class_agnostic'],
        show_progress=not no_progress,
    )

    print('\nSummary:')
    print(f'  Processed: {summary["processed"]}')
    print(f'  Skipped (already done): {summary["skipped"]}')
    print(f'  Errors: {summary["errors"]}')
    print(f'  Total: {summary["total"]}')
    print(f'\nResults saved to: {output}')


@click.command('fetch')
@add_options(model_option)
def fetch(config):
    """
    Fetch the required YOLOv8 model for inference.
    """
    scoutbot.fetch(config=config)


@click.command('export-coreml')
@click.option(
    '--config',
    default=None,
    type=click.Choice(
        [
            k
            for k, v in loc.CONFIGS.items()
            if k is not None and isinstance(v, dict) and v.get('backend') == 'coreml'
        ],
    ),
    help='Core ML config to export (default: all coreml configs).',
)
@click.option('--force', is_flag=True, help='Re-export even if the bundle exists.')
def export_coreml(config, force):
    """
    Export Core ML (.mlpackage) models from their source .pt models.
    """
    if config:
        configs = [config]
    else:
        configs = [
            name
            for name, cfg in loc.CONFIGS.items()
            if name is not None
            and isinstance(cfg, dict)
            and cfg.get('backend') == 'coreml'
        ]
    for name in configs:
        print(loc.export_coreml(name, force=force))


@click.command('example')
def example():
    """
    Run a test of the pipeline on an example image with the default configuration.
    """
    scoutbot.example()


@click.command('get_classes')
@add_options(model_option)
@add_options(output_option)
def get_classes(config, output):
    """
    Get the list of class labels supported by the model.
    """
    classes = scoutbot.get_classes(config=config)
    log.debug('Outputting classes list...')
    if output:
        with open(output, 'w') as outfile:
            json.dump(classes, outfile)
    else:
        print(json.dumps(classes, indent=2))


@click.group()
def cli():
    """
    ScoutBot CLI - YOLOv8-based wildlife detection in aerial imagery.
    """
    pass


cli.add_command(pipeline)
cli.add_command(batch)
cli.add_command(batch_resumable)
cli.add_command(fetch)
cli.add_command(export_coreml)
cli.add_command(example)
cli.add_command(get_classes)


if __name__ == '__main__':
    cli()
