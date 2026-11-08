from console import log
import argparse
import math
import re
import sys
from pathlib import Path

ANSI = re.compile(r'\x1b\[[0-9;]*m')
LAST_FACTS = {
    'keypoints': re.compile(r'Keypoints: A=(\d+), B=(\d+)'),
    'matches': re.compile(r'Mutual ratio matches: (\d+)'),
    'inliers': re.compile(r'RANSAC inliers: (\d+)/(\d+)'),
    'blend': re.compile(r'Blend: exposure=(\w+), overlap MAE=([^,\s]+), seam=(\w+)'),
    'saved': re.compile(r'Saved .* \((\d+)x(\d+)\)'),
}
ERRORS = re.compile(r'\[ERROR\]: (.*)')
SIZES = re.compile(r'V\d+: A=(\d+)x(\d+), B=(\d+)x(\d+)')


def log_facts(path):
    if not path.exists():
        return None, None
    text = ANSI.sub('', path.read_text(encoding='utf-8', errors='replace'))
    facts = {}
    for key, pattern in LAST_FACTS.items():
        found = pattern.findall(text)
        if found:
            facts[key] = found[-1]
    errors = ERRORS.findall(text)
    if errors:
        facts['errors'] = tuple(line.strip() for line in errors)
    size = SIZES.search(text)
    return facts, (tuple(int(v) for v in size.groups()) if size else None)


def read_h(path):
    values = [float(v) for v in path.read_text(encoding='utf-8').split()]
    if len(values) != 9:
        raise ValueError(f'{path}: expected 9 values, found {len(values)}')
    return [values[0:3], values[3:6], values[6:9]]


def project(h, x, y):
    w = h[2][0] * x + h[2][1] * y + h[2][2]
    return ((h[0][0] * x + h[0][1] * y + h[0][2]) / w,
            (h[1][0] * x + h[1][1] * y + h[1][2]) / w)


def projection_error(h1, h2, width, height):
    errors = []
    for j in range(12):
        for i in range(16):
            x = 20 + (width - 41) * i / 15
            y = 20 + (height - 41) * j / 11
            a, b = project(h1, x, y), project(h2, x, y)
            errors.append(math.hypot(a[0] - b[0], a[1] - b[1]))
    return sum(errors) / len(errors), max(errors)


def compare_h(base, cand, size):
    if not base.exists() and not cand.exists():
        return True, 'không có H (cả hai)'
    if not base.exists() or not cand.exists():
        return False, 'chỉ một bên có H'
    if base.read_bytes() == cand.read_bytes():
        return True, 'trùng từng byte'
    if size is None:
        return False, 'khác byte; không rõ kích thước ảnh B để đo sai số chiếu'
    mean, peak = projection_error(read_h(base), read_h(cand), size[2], size[3])
    return False, f'khác byte; sai số chiếu mean={mean:.3e} px, max={peak:.3e} px'


def compare_png(base, cand):
    if base.read_bytes() == cand.read_bytes():
        return True, 'trùng từng byte'
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return False, 'khác byte; cần Pillow + numpy để thống kê pixel'
    with Image.open(base) as a, Image.open(cand) as b:
        x = np.asarray(a.convert('RGB'), dtype=np.int16)
        y = np.asarray(b.convert('RGB'), dtype=np.int16)
    if x.shape != y.shape:
        return False, f'khác kích thước {x.shape[1]}x{x.shape[0]} so với {y.shape[1]}x{y.shape[0]}'
    diff = np.abs(x - y).max(axis=2)
    if not diff.any():
        return True, 'khác byte nhưng pixel trùng'
    return False, (f'max |diff|={int(diff.max())}, '
                   f'pixel lệch={np.count_nonzero(diff) / diff.size:.4%}, '
                   f'pixel lệch >1={np.count_nonzero(diff > 1) / diff.size:.4%}')


def describe(facts):
    return '; '.join(f'{k}={v}' for k, v in sorted(facts.items()))


def compare(baseline, candidate):
    ok = True
    cases = sorted({p.stem for d in (baseline, candidate) for p in d.glob('*.log')})
    print('| Case | Log | H |\n|---|---|---|')
    for case in cases:
        base_facts, base_size = log_facts(baseline / f'{case}.log')
        cand_facts, cand_size = log_facts(candidate / f'{case}.log')
        if base_facts is None or cand_facts is None:
            log_text, log_ok = 'chỉ một bên có log', False
        elif base_facts == cand_facts:
            log_text, log_ok = 'trùng', True
        else:
            changed = sorted(k for k in set(base_facts) | set(cand_facts)
                             if base_facts.get(k) != cand_facts.get(k))
            log_text = 'khác: ' + ', '.join(
                f'{k} {base_facts.get(k)} → {cand_facts.get(k)}' for k in changed)
            log_ok = False
        h_ok, h_text = compare_h(baseline / f'{case}.H.txt', candidate / f'{case}.H.txt',
                                 base_size or cand_size)
        ok &= log_ok and h_ok
        print(f'| {case} | {log_text} | {h_text} |')
    images = sorted({p.name for d in (baseline, candidate) for p in d.glob('*.png')})
    print('\n| Ảnh | Kết quả |\n|---|---|')
    for name in images:
        base, cand = baseline / name, candidate / name
        if not base.exists() or not cand.exists():
            same, text = False, 'chỉ một bên có ảnh'
        else:
            same, text = compare_png(base, cand)
        ok &= same
        print(f'| {name} | {text} |')
    return ok


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    args = parser.parse_args()
    for directory in (args.baseline, args.candidate):
        if not directory.is_dir():
            log('ERROR', f'Không tìm thấy thư mục {directory}')
            sys.exit(2)
    identical = compare(args.baseline, args.candidate)
    log('SUCCESS' if identical else 'WARN',
        'Hai lần chạy trùng nhau' if identical else 'Có khác biệt giữa hai lần chạy')
    sys.exit(0 if identical else 1)
