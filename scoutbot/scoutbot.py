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
    click.option(
        '--model',
        help='Model name from registry (mutually exclusive with --config)',
        default=None,
        type=str,
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
    model,
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
    ctx = click.get_current_context()
    if (
        model is not None
        and ctx.get_parameter_source('config') != click.core.ParameterSource.DEFAULT
    ):
        raise click.UsageError('--model and --config are mutually exclusive.')

    loc_thresh_float = loc_thresh / 100.0
    if model is None:
        run_cfg = loc.CONFIGS[config]
    else:
        _, run_cfg, _ = scoutbot._resolve_model(
            config=None,
            model_name=model,
            confidence_threshold=loc_thresh_float,
        )

    wic_, detects = scoutbot.pipeline(
        filepath,
        config=config if model is None else None,
        model_name=model,
        backend_device=device,
        loc_thresh=loc_thresh_float,
        slice_height=run_cfg['slice_height'],
        slice_width=run_cfg['slice_width'],
        overlap_height_ratio=run_cfg['overlap_height_ratio'],
        overlap_width_ratio=run_cfg['overlap_width_ratio'],
        perform_standard_pred=run_cfg['perform_standard_pred'],
        postprocess_class_agnostic=run_cfg['postprocess_class_agnostic'],
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
    model,
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
    ctx = click.get_current_context()
    if (
        model is not None
        and ctx.get_parameter_source('config') != click.core.ParameterSource.DEFAULT
    ):
        raise click.UsageError('--model and --config are mutually exclusive.')

    loc_thresh_float = loc_thresh / 100.0
    if model is None:
        run_cfg = loc.CONFIGS[config]
    else:
        _, run_cfg, _ = scoutbot._resolve_model(
            config=None,
            model_name=model,
            confidence_threshold=loc_thresh_float,
        )

    log.debug(f'Running batch on {len(filepaths)} files...')

    wic_list, detects_list = scoutbot.batch(
        filepaths,
        config=config if model is None else None,
        model_name=model,
        backend_device=device,
        loc_thresh=loc_thresh_float,
        slice_height=run_cfg['slice_height'],
        slice_width=run_cfg['slice_width'],
        overlap_height_ratio=run_cfg['overlap_height_ratio'],
        overlap_width_ratio=run_cfg['overlap_width_ratio'],
        perform_standard_pred=run_cfg['perform_standard_pred'],
        postprocess_class_agnostic=run_cfg['postprocess_class_agnostic'],
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
@click.option(
    '--processes',
    default=1,
    type=click.IntRange(min=1),  # reject 0/negatives at the CLI boundary
    help='Number of worker processes (Linux/CUDA). 1 = sequential. Each process '
    'loads its own model copy; ensure N copies fit in VRAM.',
)
@click.option(
    '--summary-json',
    default=None,
    type=str,
    help='Optional path for a machine-readable run summary.',
)
def batch_resumable(
    filepaths,
    output,
    config,
    device,
    loc_thresh,
    model,
    no_progress,
    processes,
    summary_json,
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
    ctx = click.get_current_context()
    if (
        model is not None
        and ctx.get_parameter_source('config') != click.core.ParameterSource.DEFAULT
    ):
        raise click.UsageError('--model and --config are mutually exclusive.')

    loc_thresh_float = loc_thresh / 100.0
    if model is None:
        run_cfg = loc.CONFIGS[config]
    else:
        _, run_cfg, _ = scoutbot._resolve_model(
            config=None,
            model_name=model,
            confidence_threshold=loc_thresh_float,
        )

    log.debug(f'Running resumable batch on {len(filepaths)} files...')
    log.debug(f'Output file: {output}')

    summary = scoutbot.batch_resumable(
        filepaths=list(filepaths),
        output_path=output,
        config=config if model is None else None,
        model_name=model,
        backend_device=device,
        loc_thresh=loc_thresh_float,
        slice_height=run_cfg['slice_height'],
        slice_width=run_cfg['slice_width'],
        overlap_height_ratio=run_cfg['overlap_height_ratio'],
        overlap_width_ratio=run_cfg['overlap_width_ratio'],
        perform_standard_pred=run_cfg['perform_standard_pred'],
        postprocess_class_agnostic=run_cfg['postprocess_class_agnostic'],
        show_progress=not no_progress,
        processes=processes,
        summary_json=summary_json,
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


@click.command('register')
@click.option('--name', required=True, help='Registry name for this model')
@click.option('--path', required=True, help='Absolute path to model file on disk')
@click.option(
    '--arch', required=True, help='Framework/architecture (e.g. yolov8, herdnet, owl)'
)
@click.option('--description', default='', help='Human-readable description')
@click.option(
    '--threshold', default=0.45, type=float, help='Default confidence threshold'
)
@click.option('--slice-height', default=512, type=int, help='SAHI slice height in pixels')
@click.option('--slice-width', default=512, type=int, help='SAHI slice width in pixels')
@click.option(
    '--overlap', default=0.25, type=float, help='SAHI overlap ratio (both axes)'
)
def register_model(
    name, path, arch, description, threshold, slice_height, slice_width, overlap
):
    """Register a local model file in the Rebel Workflow model registry."""
    import sys
    from datetime import datetime, timezone
    from pathlib import Path as _Path

    from scoutbot import model_store
    from scoutbot import registry as _registry
    from scoutbot.registry.schema import ModelEntry, RegistrySource

    model_path = _Path(path).resolve()
    if not model_path.exists():
        log.error(f'File not found: {model_path}')
        sys.exit(1)

    log.info(f'Computing MD5 for {model_path.name}...')
    md5 = model_store._md5(model_path)

    entry = ModelEntry(
        name=name,
        path=str(model_path),
        md5=md5,
        framework=arch,
        registered=datetime.now(timezone.utc).isoformat(),
        source=RegistrySource(type='local', original_path=str(model_path)),
        default_run_config={
            'confidence_threshold': threshold,
            'slice_height': slice_height,
            'slice_width': slice_width,
            'overlap_height_ratio': overlap,
            'overlap_width_ratio': overlap,
        },
        description=description,
    )
    _registry.register(entry)
    print(f'Registered {name!r} ({arch}) — {model_path.name}')


@click.group('models', invoke_without_command=True)
@click.pass_context
def models_group(ctx):
    """List and inspect registered models. Defaults to listing all models."""
    if ctx.invoked_subcommand is None:
        from pathlib import Path as _Path

        from scoutbot import registry as _registry

        entries = _registry.list_models()
        if not entries:
            print('No models registered. Run: uv run scoutbot fetch --config v3')
            return
        col = '{:<25} {:<12} {:<35} {:<10} {}'
        print(col.format('NAME', 'FRAMEWORK', 'FILE', 'MD5', 'REGISTERED'))
        print('-' * 90)
        for e in entries:
            fname = _Path(e.path).name
            print(col.format(e.name, e.framework, fname, e.md5[:8], e.registered[:10]))


@models_group.command('info')
@click.argument('name')
def models_info(name):
    """Print the full registry entry for a named model as JSON."""
    import sys

    from scoutbot import registry as _registry

    entry = _registry.lookup(name)
    if entry is None:
        log.error(f'Model not found in registry: {name}')
        sys.exit(1)
    print(json.dumps(entry.to_dict(), indent=2))


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
cli.add_command(register_model, name='register')
cli.add_command(models_group, name='models')


if __name__ == '__main__':
    cli()
