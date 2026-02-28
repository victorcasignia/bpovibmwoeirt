import os
import yaml
import torch
import argparse
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import min_s

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_dummy_dataloader(batch_size, image_size, num_classes, num_samples=50):
    x = torch.randn(num_samples, 3, image_size, image_size)
    y = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def test(config_path, checkpoint_path):
    config = load_config(config_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model
    model = min_s(
        num_classes=config['model']['num_classes'],
        in_channels=config['model']['in_channels'],
        stem_channels=config['model']['stem_channels'],
        depths=config['model']['depths'],
        dims=config['model']['dims'],
        infusion_stages=config['model']['infusion_stages'],
        attention_type=config['model'].get('attention_type', 'area'),
        num_heads=config['model'].get('num_heads', 8),
        area_size=config['model'].get('area_size', 2),
        window_size=config['model'].get('window_size', 4),
        num_kv_heads=config['model'].get('num_kv_heads', 2),
        num_landmarks=config['model'].get('num_landmarks', 64),
        segment_size=config['model'].get('segment_size', 16),
        topk_landmarks=config['model'].get('topk_landmarks', 4),
    ).to(device)
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Testing with random weights.")
        
    model.eval()
    
    # Data
    if config['data']['dataset'] == 'dummy':
        test_loader = get_dummy_dataloader(config['training']['batch_size'], config['data']['image_size'], config['model']['num_classes'], num_samples=50)
    elif config['data']['dataset'] == 'cifar10':
        import torchvision.transforms as transforms
        import torchvision.datasets as datasets
        
        transform_test = transforms.Compose([
            transforms.Resize(config['data']['image_size']),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        testset = datasets.CIFAR10(root=config['data']['data_dir'], train=False, download=True, transform=transform_test)
        test_loader = DataLoader(testset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'])
    elif config['data']['dataset'] == 'cifar100':
        import torchvision.transforms as transforms
        import torchvision.datasets as datasets
        
        transform_test = transforms.Compose([
            transforms.Resize(config['data']['image_size']),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        testset = datasets.CIFAR100(root=config['data']['data_dir'], train=False, download=True, transform=transform_test)
        test_loader = DataLoader(testset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'])
    else:
        raise NotImplementedError(f"Dataset {config['data']['dataset']} is not implemented")
        
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # Inference noise is 0
            outputs = model(inputs, noise_scale=0.0)
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    test_acc = 100. * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/min_s.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/min_s_final.pth', help='Path to checkpoint')
    args = parser.parse_args()
    test(args.config, args.checkpoint)
