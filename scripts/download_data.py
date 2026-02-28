import os
import argparse
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory to save data')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='Dataset to download')
    args = parser.parse_args()
    
    os.makedirs(args.data_dir, exist_ok=True)
    
    if args.dataset == 'cifar10':
        download_cifar10(args.data_dir)
    elif args.dataset == 'cifar100':
        download_cifar100(args.data_dir)
    else:
        print(f"Dataset {args.dataset} not supported for automatic download.")
