import os
import argparse
import shutil
import subprocess
import zipfile
import torchvision.datasets as datasets

def download_cifar10(data_dir):
    print(f"Downloading CIFAR-10 to {data_dir}...")
    datasets.CIFAR10(root=data_dir, train=True, download=True)
    datasets.CIFAR10(root=data_dir, train=False, download=True)
    print("Download complete.")

def download_cifar100(data_dir):
    print(f"Downloading CIFAR-100 to {data_dir}...")
    datasets.CIFAR100(root=data_dir, train=True, download=True)
    datasets.CIFAR100(root=data_dir, train=False, download=True)
    print("Download complete.")


def _extract_all_zips(folder):
    zip_files = [f for f in os.listdir(folder) if f.endswith('.zip')]
    if not zip_files:
        print("No zip files found to extract.")
        return

    for name in zip_files:
        path = os.path.join(folder, name)
        out_dir = os.path.join(folder, os.path.splitext(name)[0])
        print(f"Extracting {path} -> {out_dir}")
        os.makedirs(out_dir, exist_ok=True)
        with zipfile.ZipFile(path, 'r') as archive:
            archive.extractall(out_dir)


def download_imagenet1k(data_dir, competition='imagenet-object-localization-challenge', extract=False):
    """
    Downloads ImageNet-1k files from Kaggle competition API.

    Requirements:
    - Kaggle CLI installed and configured (`kaggle.json` credentials).
    - Access granted to the competition dataset.
    """
    kaggle_bin = shutil.which('kaggle')
    if kaggle_bin is None:
        raise RuntimeError(
            "Kaggle CLI not found. Install it with `pip install kaggle` and configure credentials."
        )

    target_dir = os.path.join(data_dir, 'imagenet1k')
    os.makedirs(target_dir, exist_ok=True)

    print(f"Downloading ImageNet-1k from Kaggle competition '{competition}' to {target_dir}...")
    cmd = [
        kaggle_bin,
        'competitions',
        'download',
        '-c',
        competition,
        '-p',
        target_dir,
    ]
    subprocess.run(cmd, check=True)
    print("Download complete.")

    if extract:
        _extract_all_zips(target_dir)
        print("Extraction complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory to save data')
    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar10',
        choices=['cifar10', 'cifar100', 'imagenet1k'],
        help='Dataset to download'
    )
    parser.add_argument(
        '--kaggle_competition',
        type=str,
        default='imagenet-object-localization-challenge',
        help='Kaggle competition slug for ImageNet download'
    )
    parser.add_argument(
        '--extract',
        action='store_true',
        help='Extract downloaded zip files (ImageNet only)'
    )
    args = parser.parse_args()
    
    os.makedirs(args.data_dir, exist_ok=True)
    
    if args.dataset == 'cifar10':
        download_cifar10(args.data_dir)
    elif args.dataset == 'cifar100':
        download_cifar100(args.data_dir)
    elif args.dataset == 'imagenet1k':
        download_imagenet1k(
            data_dir=args.data_dir,
            competition=args.kaggle_competition,
            extract=args.extract,
        )
    else:
        print(f"Dataset {args.dataset} not supported for automatic download.")
