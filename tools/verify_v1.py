"""Compare V0 and V1 stage dumps, homographies and PNGs on identical inputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image

from benchmark import ROOT, SITES
from compare_runs import projection_error, read_h


EXACT_FILES = ('blurred_A.pgm', 'blurred_B.pgm', 'keypoints_A.csv', 'keypoints_B.csv',
               'descriptors_A.txt', 'descriptors_B.txt', 'matches.csv', 'inliers.csv', 'seam_x.csv')
BLEND_FIELDS = ('canvas_w', 'canvas_h', 'offset_x', 'offset_y', 'exposure_applied', 'seam_used')
ENV = {**os.environ, 'NO_COLOR': '1'}


def compare_case(binary, left, right, output, ground_truth=None, reference=0, candidate=1):
    ref, actual = output / f'v{reference}', output / f'v{candidate}'
    for version, folder in ((reference, ref), (candidate, actual)):
        folder.mkdir(parents=True)
        run = subprocess.run([str(binary), str(left), str(right), str(folder / 'panorama.png'),
                              '--version', str(version), '--runs', '1', '--dump-dir', str(folder)],
                             env=ENV, capture_output=True, text=True)
        if run.returncode:
            return {'passed': False, 'error': f'V{version}: {run.stderr.strip() or run.stdout.strip()}'}

    mismatches = [name for name in EXACT_FILES if (ref / name).read_bytes() != (actual / name).read_bytes()]
    a_blend, b_blend = [json.loads((folder / 'blend.json').read_text()) for folder in (ref, actual)]
    mismatches += [name for name in BLEND_FIELDS if a_blend[name] != b_blend[name]]
    for name in ('gain', 'bias', 'overlap_mae'):
        if not np.allclose(a_blend[name], b_blend[name], rtol=0, atol=1e-8):
            mismatches.append(name)

    with Image.open(right) as img:
        w, h = img.size
    homography = read_h(actual / 'H.txt')
    proj_mean, proj_max = projection_error(read_h(ref / 'H.txt'), homography, w, h)
    with Image.open(ref / 'panorama.png') as a_img, Image.open(actual / 'panorama.png') as b_img:
        same_canvas = a_img.size == b_img.size
        pixel_max = None
        if same_canvas:
            # Limit scratch RAM for large full panoramas.
            pixel_max = 0
            for y in range(0, a_img.height, 128):
                box = (0, y, a_img.width, min(y + 128, a_img.height))
                delta = np.asarray(a_img.crop(box), dtype=np.int16) - np.asarray(b_img.crop(box), dtype=np.int16)
                pixel_max = max(pixel_max, int(np.abs(delta).max()))

    result = dict(mismatches=mismatches, projection_mean_px=proj_mean, projection_max_px=proj_max,
                  same_canvas=same_canvas, max_pixel_delta=pixel_max)
    result['passed'] = bool(not mismatches and proj_mean <= 0.01 and proj_max <= 0.1
                            and same_canvas and pixel_max <= 1)
    if ground_truth is not None:
        mean, peak = projection_error(read_h(ground_truth), homography, w, h)
        result.update(ground_truth_mean_px=mean, ground_truth_max_px=peak)
        result['passed'] &= mean <= 1.0
    if reference == 1 and candidate == 2:
        result['passed'] &= pixel_max == 0 and (ref / 'H.txt').read_bytes() == (actual / 'H.txt').read_bytes()
    result['seam_used'] = b_blend['seam_used']
    return result


def verify(binary, reference=0, candidate=1):
    cases = []
    for resolution in ('small', 'medium', 'large'):
        folder = ROOT / 'dataset/prepared' / resolution
        for site in SITES:
            cases.append((f'{resolution}/{site}', folder / f'{site}_left.png',
                          folder / f'{site}_right.png', None))
    synth = ROOT / 'dataset/synthetic'
    cases.append(('synthetic', synth / 'synthetic_left.png', synth / 'synthetic_right.png',
                  synth / 'H_ground_truth.txt'))
    results = {}
    for label, left, right, ground_truth in cases:
        with tempfile.TemporaryDirectory(prefix='stitch-verify-') as tmp:
            results[label] = compare_case(binary, left, right, Path(tmp), ground_truth, reference, candidate)
        print(f'{label}: {json.dumps(results[label], ensure_ascii=False)}', flush=True)

    with tempfile.TemporaryDirectory(prefix='stitch-empty-') as tmp:
        folder = Path(tmp)
        blank, panorama = folder / 'blank.png', folder / 'panorama.png'
        Image.new('RGB', (64, 64)).save(blank)
        run = subprocess.run([str(binary), str(blank), str(blank), str(panorama),
                              '--version', str(candidate), '--runs', '1'], env=ENV, capture_output=True, text=True)
        results['empty_descriptors'] = {
            'passed': (run.returncode != 0 and 'Insufficient feature matches: 0' in run.stderr
                       and not panorama.exists()),
            'diagnostic': run.stderr.strip(),
        }
        print(f'empty_descriptors: {json.dumps(results["empty_descriptors"])}', flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT / ('stitch.exe' if os.name == 'nt' else 'stitch'))
    parser.add_argument('--report', type=Path)
    parser.add_argument('--reference', type=int, choices=(0, 1), default=0)
    parser.add_argument('--candidate', type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if args.reference >= args.candidate:
        parser.error('candidate must be newer than reference')
    results = verify(args.binary.resolve(), args.reference, args.candidate)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    raise SystemExit(0 if all(result['passed'] for result in results.values()) else 1)
