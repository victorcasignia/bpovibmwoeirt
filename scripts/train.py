import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import wandb
import argparse
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import min_s
from optim import MuSGD

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_dummy_dataloader(batch_size, image_size, num_classes, num_samples=100):
    x = torch.randn(num_samples, 3, image_size, image_size)
    y = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train(config_path):
    config = load_config(config_path)
    scheduler_cfg = config['training'].get('scheduler', {'type': 'none'})
    
    # Initialize wandb
    wandb.init(project=config['logging']['project'], name=config['logging']['run_name'], config=config)
    
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
    print(
        f"Attention backend: {config['model'].get('attention_type', 'area')} "
        f"(num_heads={config['model'].get('num_heads', 8)}, "
        f"area_size={config['model'].get('area_size', 2)}, "
        f"window_size={config['model'].get('window_size', 4)}, "
        f"num_kv_heads={config['model'].get('num_kv_heads', 2)}, "
        f"num_landmarks={config['model'].get('num_landmarks', 64)}, "
        f"segment_size={config['model'].get('segment_size', 16)}, "
        f"topk_landmarks={config['model'].get('topk_landmarks', 4)})"
    )
    
    # Optimizer
    if config['training']['optimizer'] == 'musgd':
        optimizer = MuSGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=config['training']['sgd_momentum'],
            muon_momentum=config['training']['muon_momentum'],
            weight_decay=config['training']['weight_decay']
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])

    scheduler_type = scheduler_cfg.get('type', 'none').lower()
    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'],
            eta_min=scheduler_cfg.get('min_lr', 1e-6),
        )
    elif scheduler_type == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}. Use 'none' or 'cosine'.")
        
    criterion = nn.CrossEntropyLoss()
    
    # Data
    print(f"Loading dataset: {config['data']['dataset']}...")
    if config['data']['dataset'] == 'dummy':
        train_loader = get_dummy_dataloader(config['training']['batch_size'], config['data']['image_size'], config['model']['num_classes'], num_samples=100)
        val_loader = get_dummy_dataloader(config['training']['batch_size'], config['data']['image_size'], config['model']['num_classes'], num_samples=20)
    elif config['data']['dataset'] == 'cifar10':
        import torchvision.transforms as transforms
        import torchvision.datasets as datasets
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(config['data']['image_size']),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(config['data']['image_size']),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = datasets.CIFAR10(root=config['data']['data_dir'], train=True, download=True, transform=transform_train)
        train_loader = DataLoader(trainset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['data']['num_workers'])

        valset = datasets.CIFAR10(root=config['data']['data_dir'], train=False, download=True, transform=transform_test)
        val_loader = DataLoader(valset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'])
    elif config['data']['dataset'] == 'cifar100':
        import torchvision.transforms as transforms
        import torchvision.datasets as datasets
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(config['data']['image_size']),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(config['data']['image_size']),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        trainset = datasets.CIFAR100(root=config['data']['data_dir'], train=True, download=True, transform=transform_train)
        train_loader = DataLoader(trainset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['data']['num_workers'])

        valset = datasets.CIFAR100(root=config['data']['data_dir'], train=False, download=True, transform=transform_test)
        val_loader = DataLoader(valset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'])
    else:
        raise NotImplementedError(f"Dataset {config['data']['dataset']} is not implemented")
    print(f"Dataset loaded. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
    epochs = config['training']['epochs']
    noise_anneal_epochs = config['training']['noise_anneal_epochs']
    
    print("Starting training loop...")
    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]['lr']
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        # Noise scheduling
        if epoch >= epochs - noise_anneal_epochs:
            noise_scale = max(0.0, 1.0 - (epoch - (epochs - noise_anneal_epochs)) / noise_anneal_epochs)
        else:
            noise_scale = 1.0
            
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs, noise_scale=noise_scale)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total, 'noise': noise_scale})
            
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{epochs}")
            for inputs, targets in val_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                # Inference noise is 0
                outputs = model(inputs, noise_scale=0.0)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                val_pbar.set_postfix({'val_loss': loss.item(), 'val_acc': 100.*correct/total})
                
        val_acc = 100. * correct / total
        
        print(f"Epoch {epoch+1} | LR: {current_lr:.8f} | Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.2f}%")
        
        wandb.log({
            'epoch': epoch + 1,
            'lr': current_lr,
            'train_loss': train_loss / len(train_loader),
            'train_acc': train_acc,
            'val_loss': val_loss / len(val_loader),
            'val_acc': val_acc,
            'noise_scale': noise_scale
        })

        if scheduler is not None:
            scheduler.step()
        
    wandb.finish()
    
    # Save model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/min_s_final.pth')
    print("Training complete. Model saved to checkpoints/min_s_final.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/min_s.yaml', help='Path to config file')
    args = parser.parse_args()
    train(args.config)
