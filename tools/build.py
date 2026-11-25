# Author: Nguyen Huu Khanh Hung - 23120271 (Member 3). Original portable build helper.
from console import log
import argparse
import platform
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def build(root=ROOT, cuda=None, arch='sm_75'):
    sources = ['main.cu', 'stb_impl.cpp', 'v0_frontend.cpp', 'v0_backend.cpp', 'v0_stitching.cpp']
    binary = root / ('stitch.exe' if platform.system() == 'Windows' else 'stitch')
    nvcc = shutil.which('nvcc')
    use_cuda = bool(nvcc) if cuda is None else cuda
    compiler = nvcc if use_cuda else shutil.which('g++')
    if not compiler:
        raise RuntimeError('Install CUDA nvcc for V1' if use_cuda else 'Install g++ for the CPU build')
    command = [compiler, '-O3']
    if use_cuda:
        sources += ['gpu_frontend.cu', 'gpu_backend.cu', 'gpu_pipeline.cu']
        command += ['-std=c++17', '--fmad=false', f'-arch={arch}', '-DSTITCH_CUDA']
    else:
        command += ['-std=c++14', '-x', 'c++']
    command += [str(root / 'src' / name) for name in sources] + ['-o', str(binary)]
    log('PROCESS', ' '.join(command))
    subprocess.run(command, check=True, cwd=root)
    return binary, command

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--cuda', dest='cuda', action='store_true')
    mode.add_argument('--cpu', dest='cuda', action='store_false')
    parser.set_defaults(cuda=None)
    parser.add_argument('--arch', default='sm_75', help='CUDA architecture (default: sm_75 for Colab T4)')
    args = parser.parse_args()
    build(cuda=args.cuda, arch=args.arch)
