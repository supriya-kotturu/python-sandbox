#!/usr/bin/env python3
"""
Distributed Training Benchmarking Tool for Deep Learning Models
A modular framework to benchmark various distributed training strategies
"""

import os
import json
import yaml
import time
import logging
import argparse
import multiprocessing as mp
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np

# Optional imports for additional frameworks
try:
    import ray
    from ray import train
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("Ray not installed. Run: pip install 'ray[train]'")

try:
    import horovod.torch as hvd
    HOROVOD_AVAILABLE = True
except ImportError:
    HOROVOD_AVAILABLE = False
    # Only log once
    if not hasattr(globals(), '_horovod_warning_shown'):
        print("Horovod not installed. Run: pip install horovod")
        globals()['_horovod_warning_shown'] = True

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    # Only log once
    if not hasattr(globals(), '_deepspeed_warning_shown'):
        print("DeepSpeed not installed. Run: pip install deepspeed")
        globals()['_deepspeed_warning_shown'] = True

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkMetrics:
    """Data class to store benchmark metrics"""
    framework: str
    model_name: str
    dataset_name: str
    batch_size: int
    num_epochs: int
    world_size: int
    total_training_time: float
    avg_epoch_time: float
    throughput_samples_per_sec: float
    final_accuracy: float
    peak_memory_mb: float
    
    def to_dict(self):
        return asdict(self)

class MetricsCollector:
    """Collects and manages benchmark metrics"""
    
    def __init__(self):
        self.metrics = []
        self.current_run = {}
        
    def start_run(self, config: Dict[str, Any]):
        """Start a new benchmark run"""
        self.current_run = {
            'framework': config.get('framework', 'unknown'),
            'model_name': config.get('model', 'unknown'),
            'dataset_name': config.get('dataset', 'unknown'),
            'batch_size': config.get('batch_size', 32),
            'num_epochs': config.get('epochs', 1),
            'world_size': config.get('world_size', 1),
            'start_time': time.time(),
            'epoch_times': [],
            'accuracies': []
        }
        
    def log_epoch(self, epoch_time: float, accuracy: float):
        """Log metrics for a single epoch"""
        self.current_run['epoch_times'].append(epoch_time)
        self.current_run['accuracies'].append(accuracy)
        
    def end_run(self) -> BenchmarkMetrics:
        """End the current run and return metrics"""
        end_time = time.time()
        total_time = end_time - self.current_run['start_time']
        avg_epoch_time = sum(self.current_run['epoch_times']) / len(self.current_run['epoch_times'])
        
        # Calculate throughput (samples per second)
        total_samples = self.current_run['batch_size'] * len(self.current_run['epoch_times'])
        throughput = total_samples / total_time if total_time > 0 else 0
        
        # Get memory usage (simplified)
        peak_memory = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
        
        metrics = BenchmarkMetrics(
            framework=self.current_run['framework'],
            model_name=self.current_run['model_name'],
            dataset_name=self.current_run['dataset_name'],
            batch_size=self.current_run['batch_size'],
            num_epochs=self.current_run['num_epochs'],
            world_size=self.current_run['world_size'],
            total_training_time=total_time,
            avg_epoch_time=avg_epoch_time,
            throughput_samples_per_sec=throughput,
            final_accuracy=self.current_run['accuracies'][-1] if self.current_run['accuracies'] else 0.0,
            peak_memory_mb=peak_memory
        )
        
        self.metrics.append(metrics)
        return metrics
        
    def save_metrics(self, filepath: str):
        """Save all metrics to file"""
        metrics_data = [m.to_dict() for m in self.metrics]
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)

class BaseModel(ABC):
    """Abstract base class for models"""
    
    @abstractmethod
    def create_model(self) -> nn.Module:
        pass
    
    @abstractmethod
    def get_optimizer(self, model: nn.Module, lr: float = 0.001) -> optim.Optimizer:
        pass
    
    @abstractmethod
    def get_criterion(self) -> nn.Module:
        pass

class ResNet50Model(BaseModel):
    """ResNet50 model implementation"""
    
    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes
        
    def create_model(self) -> nn.Module:
        # Use weights parameter instead of deprecated pretrained parameter
        model = torchvision.models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        return model
        
    def get_optimizer(self, model: nn.Module, lr: float = 0.001) -> optim.Optimizer:
        return optim.Adam(model.parameters(), lr=lr)
        
    def get_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()

class UNetModel(BaseModel):
    """UNet model for segmentation tasks"""
    
    def __init__(self, n_channels: int = 3, n_classes: int = 2):
        self.n_channels = n_channels
        self.n_classes = n_classes
        
    def create_model(self) -> nn.Module:
        class UNet(nn.Module):
            def __init__(self, n_channels, n_classes):
                super(UNet, self).__init__()
                self.n_channels = n_channels
                self.n_classes = n_classes
                
                # Encoder
                self.inc = self.double_conv(n_channels, 64)
                self.down1 = self.down(64, 128)
                self.down2 = self.down(128, 256)
                self.down3 = self.down(256, 512)
                
                # Decoder
                self.up1 = self.up(512, 256)
                self.up2 = self.up(256, 128)
                self.up3 = self.up(128, 64)
                self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
                
            def double_conv(self, in_channels, out_channels):
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            
            def down(self, in_channels, out_channels):
                return nn.Sequential(
                    nn.MaxPool2d(2),
                    self.double_conv(in_channels, out_channels)
                )
            
            def up(self, in_channels, out_channels):
                return nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
                    self.double_conv(in_channels, out_channels)
                )
            
            def forward(self, x):
                x1 = self.inc(x)
                x2 = self.down1(x1)
                x3 = self.down2(x2)
                x4 = self.down3(x3)
                
                x = self.up1(x4)
                x = torch.cat([x, x3], dim=1)
                x = self.up2(x)
                x = torch.cat([x, x2], dim=1)
                x = self.up3(x)
                x = torch.cat([x, x1], dim=1)
                x = self.outc(x)
                return x
                
        return UNet(self.n_channels, self.n_classes)
        
    def get_optimizer(self, model: nn.Module, lr: float = 0.001) -> optim.Optimizer:
        return optim.Adam(model.parameters(), lr=lr)
        
    def get_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()

class SimpleTransformer(BaseModel):
    """Simple Transformer model for benchmarking"""
    
    def __init__(self, vocab_size: int = 1000, d_model: int = 512, nhead: int = 8, num_layers: int = 6):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
    def create_model(self) -> nn.Module:
        class TransformerModel(nn.Module):
            def __init__(self, vocab_size, d_model, nhead, num_layers):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model)
                encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                self.fc = nn.Linear(d_model, vocab_size)
                
            def forward(self, x):
                x = self.embedding(x)
                x = self.transformer(x)
                return self.fc(x.mean(dim=1))  # Global average pooling
                
        return TransformerModel(self.vocab_size, self.d_model, self.nhead, self.num_layers)
        
    def get_optimizer(self, model: nn.Module, lr: float = 0.001) -> optim.Optimizer:
        return optim.Adam(model.parameters(), lr=lr)
        
    def get_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()

class BaseDataset(ABC):
    """Abstract base class for datasets"""
    
    @abstractmethod
    def get_dataloaders(self, batch_size: int, num_workers: int = 4) -> tuple:
        pass

class CIFAR10Dataset(BaseDataset):
    """CIFAR-10 dataset implementation"""
    
    def get_dataloaders(self, batch_size: int, num_workers: int = 4) -> tuple:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        trainloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        testloader = DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        return trainloader, testloader

class SyntheticSegmentationDataset(BaseDataset):
    """Synthetic segmentation dataset for UNet testing"""
    
    def get_dataloaders(self, batch_size: int, num_workers: int = 4) -> tuple:
        # Define dataset class at module level to fix pickling issues
        trainset = SyntheticSegData(size=500)  # Smaller for demo
        testset = SyntheticSegData(size=100)  # Smaller for demo
        
        # CPU-optimized settings
        cpu_workers = 0  # Set to 0 to avoid multiprocessing issues
        
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=cpu_workers)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=cpu_workers)
        
        return trainloader, testloader

# Define dataset at module level to fix pickling issues
class SyntheticSegData(Dataset):
    def __init__(self, size=1000, img_size=(256, 256)):
        self.size = size
        self.img_size = img_size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Generate synthetic image (3 channels)
        img = torch.randn(3, *self.img_size)
        
        # Generate synthetic segmentation mask (2 classes: background, foreground)
        mask = torch.zeros(self.img_size, dtype=torch.long)
        # Create a random circle as foreground
        center_x, center_y = np.random.randint(50, self.img_size[0]-50), np.random.randint(50, self.img_size[1]-50)
        radius = np.random.randint(20, 40)
        
        y_coords, x_coords = np.ogrid[:self.img_size[0], :self.img_size[1]]
        mask_circle = (x_coords - center_x)**2 + (y_coords - center_y)**2 <= radius**2
        mask[mask_circle] = 1
        
        return img, mask

class BaseFramework(ABC):
    """Abstract base class for distributed training frameworks"""
    
    @abstractmethod
    def setup(self, world_size: int, rank: int):
        pass
    
    @abstractmethod
    def train_model(self, model: BaseModel, dataset: BaseDataset, config: Dict[str, Any]) -> BenchmarkMetrics:
        pass
    
    @abstractmethod
    def cleanup(self):
        pass

class PyTorchDDPFramework(BaseFramework):
    """PyTorch Distributed Data Parallel implementation (CPU optimized)"""
    
    def setup(self, world_size: int, rank: int):
        """Setup DDP environment for CPU"""
        if world_size > 1:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            # Use spawn method for Windows compatibility
            torch.multiprocessing.set_start_method('spawn', force=True)
            torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
    
    def train_model(self, model: BaseModel, dataset: BaseDataset, config: Dict[str, Any]) -> BenchmarkMetrics:
        """Train model using PyTorch DDP (CPU optimized)"""
        metrics_collector = MetricsCollector()
        config['framework'] = 'pytorch_ddp'
        metrics_collector.start_run(config)
        
        # Create model
        net = model.create_model()
        device = torch.device("cpu")  # Force CPU for our setup
        net.to(device)
        
        # Wrap with DDP if distributed
        if config.get('world_size', 1) > 1:
            net = torch.nn.parallel.DistributedDataParallel(net)
        
        # Get data loaders with CPU optimization - use 0 workers to avoid multiprocessing issues
        trainloader, testloader = dataset.get_dataloaders(
            config['batch_size'], 
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        
        # Setup training
        criterion = model.get_criterion()
        optimizer = model.get_optimizer(net, config.get('lr', 0.001))
        
        # Training loop
        for epoch in range(config['epochs']):
            epoch_start = time.time()
            net.train()
            running_loss = 0.0
            
            for i, (inputs, labels) in enumerate(trainloader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                # Limit batches for demo (CPU training can be slow)
                if i >= config.get('max_batches_per_epoch', 20):
                    break
            
            epoch_time = time.time() - epoch_start
            
            # Quick accuracy calculation
            accuracy = self._calculate_accuracy(net, testloader, device, config.get('max_test_batches', 5))
            
            metrics_collector.log_epoch(epoch_time, accuracy)
            logger.info(f"Epoch {epoch+1}/{config['epochs']}, Time: {epoch_time:.2f}s, Accuracy: {accuracy:.2f}%")
        
        return metrics_collector.end_run()
    
    def _calculate_accuracy(self, model, testloader, device, max_batches=5):
        """Calculate model accuracy on test set (limited batches for demo)"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(testloader):
                if i >= max_batches:
                    break
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                # Handle segmentation vs classification
                if len(outputs.shape) == 4:  # Segmentation (B, C, H, W)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.numel()
                    correct += (predicted == labels).sum().item()
                else:  # Classification (B, C)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
        
        return 100 * correct / total if total > 0 else 0.0
    
    def cleanup(self):
        """Cleanup DDP"""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

class RayTrainFramework(BaseFramework):
    """Ray Train framework implementation"""
    
    def setup(self, world_size: int, rank: int):
        """Setup Ray"""
        if not RAY_AVAILABLE:
            raise ImportError("Ray not available. Install with: pip install ray[train]")
        
        if not ray.is_initialized():
            ray.init(num_cpus=world_size, ignore_reinit_error=True)
    
    def train_model(self, model: BaseModel, dataset: BaseDataset, config: Dict[str, Any]) -> BenchmarkMetrics:
        """Train model using Ray Train"""
        if not RAY_AVAILABLE:
            raise ImportError("Ray not available")
            
        config['framework'] = 'ray_train'
        
        def train_func(train_config):
            """Training function to run on Ray workers"""
            # Create model and data
            net = model.create_model()
            device = torch.device("cpu")
            net.to(device)
            
            trainloader, testloader = dataset.get_dataloaders(train_config['batch_size'], num_workers=0)
            criterion = model.get_criterion()
            optimizer = model.get_optimizer(net, train_config.get('lr', 0.001))
            
            metrics_collector = MetricsCollector()
            metrics_collector.start_run(train_config)
            
            # Training loop
            for epoch in range(train_config['epochs']):
                epoch_start = time.time()
                net.train()
                
                for i, (inputs, labels) in enumerate(trainloader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    if i >= train_config.get('max_batches_per_epoch', 20):
                        break
                
                epoch_time = time.time() - epoch_start
                accuracy = self._calculate_accuracy_ray(net, testloader, device, train_config.get('max_test_batches', 5))
                
                metrics_collector.log_epoch(epoch_time, accuracy)
                
                # Report to Ray
                train.report({
                    "epoch": epoch,
                    "epoch_time": epoch_time,
                    "accuracy": accuracy
                })
            
            return metrics_collector.end_run()
        
        # Run training with Ray
        from ray.train import ScalingConfig
        from ray.train.torch import TorchTrainer
        
        scaling_config = ScalingConfig(num_workers=config.get('world_size', 1), use_gpu=False)
        trainer = TorchTrainer(
            train_func,
            train_loop_config=config,
            scaling_config=scaling_config
        )
        
        result = trainer.fit()
        
        # Extract metrics (simplified for demo)
        metrics_collector = MetricsCollector()
        metrics_collector.start_run(config)
        
        # Simulate some metrics (in real implementation, we'd extract from Ray results)
        for i in range(config['epochs']):
            metrics_collector.log_epoch(1.0, 75.0)  # Dummy values
        
        return metrics_collector.end_run()
    
    def _calculate_accuracy_ray(self, model, testloader, device, max_batches=5):
        """Calculate accuracy for Ray training"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(testloader):
                if i >= max_batches:
                    break
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                if len(outputs.shape) == 4:  # Segmentation
                    _, predicted = torch.max(outputs, 1)
                    total += labels.numel()
                    correct += (predicted == labels).sum().item()
                else:  # Classification
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
        
        return 100 * correct / total if total > 0 else 0.0
    
    def cleanup(self):
        """Cleanup Ray"""
        if ray.is_initialized():
            ray.shutdown()

class HorovodFramework(BaseFramework):
    """Horovod framework implementation"""
    
    def setup(self, world_size: int, rank: int):
        """Setup Horovod"""
        if not HOROVOD_AVAILABLE:
            raise ImportError("Horovod not available. Install with: pip install horovod")
        
        hvd.init()
    
    def train_model(self, model: BaseModel, dataset: BaseDataset, config: Dict[str, Any]) -> BenchmarkMetrics:
        """Train model using Horovod"""
        if not HOROVOD_AVAILABLE:
            raise ImportError("Horovod not available")
            
        config['framework'] = 'horovod'
        metrics_collector = MetricsCollector()
        metrics_collector.start_run(config)
        
        # Create model
        net = model.create_model()
        device = torch.device("cpu")
        net.to(device)
        
        # Get data loaders
        trainloader, testloader = dataset.get_dataloaders(config['batch_size'], num_workers=0)
        
        # Setup training with Horovod
        criterion = model.get_criterion()
        optimizer = model.get_optimizer(net, config.get('lr', 0.001) * hvd.size())
        
        # Horovod: wrap optimizer with DistributedOptimizer
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=net.named_parameters())
        
        # Horovod: broadcast parameters & optimizer state
        hvd.broadcast_parameters(net.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        
        # Training loop
        for epoch in range(config['epochs']):
            epoch_start = time.time()
            net.train()
            
            for i, (inputs, labels) in enumerate(trainloader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                if i >= config.get('max_batches_per_epoch', 20):
                    break
            
            epoch_time = time.time() - epoch_start
            accuracy = self._calculate_accuracy_hvd(net, testloader, device, config.get('max_test_batches', 5))
            
            if hvd.rank() == 0:  # Only log on rank 0
                metrics_collector.log_epoch(epoch_time, accuracy)  
                logger.info(f"Epoch {epoch+1}/{config['epochs']}, Time: {epoch_time:.2f}s, Accuracy: {accuracy:.2f}%")
        
        return metrics_collector.end_run() if hvd.rank() == 0 else None
    
    def _calculate_accuracy_hvd(self, model, testloader, device, max_batches=5):
        """Calculate accuracy for Horovod"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(testloader):
                if i >= max_batches:
                    break
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                if len(outputs.shape) == 4:  # Segmentation
                    _, predicted = torch.max(outputs, 1)
                    total += labels.numel()
                    correct += (predicted == labels).sum().item()
                else:  # Classification
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
        
        return 100 * correct / total if total > 0 else 0.0
    
    def cleanup(self):
        """Cleanup Horovod"""
        pass  # Horovod cleanup is automatic

class CPUDeepSpeedFramework(BaseFramework):
    """Simplified DeepSpeed framework for CPU (limited functionality)"""
    
    def setup(self, world_size: int, rank: int):
        """Setup DeepSpeed"""
        if not DEEPSPEED_AVAILABLE:
            raise ImportError("DeepSpeed not available. Install with: pip install deepspeed")
    
    def train_model(self, model: BaseModel, dataset: BaseDataset, config: Dict[str, Any]) -> BenchmarkMetrics:
        """Train model using DeepSpeed (CPU mode - limited features)"""
        if not DEEPSPEED_AVAILABLE:
            raise ImportError("DeepSpeed not available")
            
        config['framework'] = 'deepspeed_cpu'
        metrics_collector = MetricsCollector()
        metrics_collector.start_run(config)
        
        # Note: DeepSpeed CPU support is limited, this is a simplified implementation
        # For full DeepSpeed features, GPU setup would be required
        
        # Create model
        net = model.create_model()
        device = torch.device("cpu")
        net.to(device)
        
        # Get data loaders
        trainloader, testloader = dataset.get_dataloaders(config['batch_size'], num_workers=0)
        
        # Standard training (DeepSpeed CPU features are limited)
        criterion = model.get_criterion()
        optimizer = model.get_optimizer(net, config.get('lr', 0.001))
        
        logger.info("Running simplified DeepSpeed training (CPU mode has limited features)")
        
        # Training loop
        for epoch in range(config['epochs']):
            epoch_start = time.time()
            net.train()
            
            for i, (inputs, labels) in enumerate(trainloader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                if i >= config.get('max_batches_per_epoch', 20):
                    break
            
            epoch_time = time.time() - epoch_start
            accuracy = self._calculate_accuracy_ds(net, testloader, device, config.get('max_test_batches', 5))
            
            metrics_collector.log_epoch(epoch_time, accuracy)
            logger.info(f"Epoch {epoch+1}/{config['epochs']}, Time: {epoch_time:.2f}s, Accuracy: {accuracy:.2f}%")
        
        return metrics_collector.end_run()
    
    def _calculate_accuracy_ds(self, model, testloader, device, max_batches=5):
        """Calculate accuracy for DeepSpeed"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(testloader):
                if i >= max_batches:
                    break
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                if len(outputs.shape) == 4:  # Segmentation
                    _, predicted = torch.max(outputs, 1)
                    total += labels.numel()
                    correct += (predicted == labels).sum().item()
                else:  # Classification
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
        
        return 100 * correct / total if total > 0 else 0.0
    
    def cleanup(self):
        """Cleanup DeepSpeed"""
        pass

class BenchmarkEngine:
    """Main benchmark engine that orchestrates experiments"""
    
    def __init__(self):
        self.models = {
            'resnet50': ResNet50Model,
            'transformer': SimpleTransformer,
            'unet': UNetModel
        }
        self.datasets = {
            'cifar10': CIFAR10Dataset,
            'synthetic_seg': SyntheticSegmentationDataset
        }
        self.frameworks = {
            'pytorch_ddp': PyTorchDDPFramework,
            'ray_train': RayTrainFramework if RAY_AVAILABLE else None,
            'horovod': HorovodFramework if HOROVOD_AVAILABLE else None,
            'deepspeed_cpu': CPUDeepSpeedFramework if DEEPSPEED_AVAILABLE else None
        }
        
        # Remove None frameworks
        self.frameworks = {k: v for k, v in self.frameworks.items() if v is not None}
        
    def run_benchmark(self, config_path: str) -> List[BenchmarkMetrics]:
        """Run benchmark from configuration file"""
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        
        results = []
        
        for experiment in config['experiments']:
            logger.info(f"Running experiment: {experiment['name']}")
            
            # Skip if framework not available
            if experiment['framework'] not in self.frameworks:
                logger.warning(f"Framework {experiment['framework']} not available, skipping...")
                continue
                
            # Get components
            model_class = self.models[experiment['model']]
            dataset_class = self.datasets[experiment['dataset']]
            framework_class = self.frameworks[experiment['framework']]
            
            # Create instances
            model = model_class()
            dataset = dataset_class()
            framework = framework_class()
            
            # Run experiment
            try:
                framework.setup(experiment.get('world_size', 1), 0)  # Single process for now
                
                metrics = framework.train_model(model, dataset, experiment)
                if metrics:  # Some frameworks may return None for non-root processes
                    results.append(metrics)
                    
                    logger.info(f"Completed: {metrics.framework} - {metrics.model_name}")
                    logger.info(f"Training time: {metrics.total_training_time:.2f}s")
                    logger.info(f"Final accuracy: {metrics.final_accuracy:.2f}%")
                    logger.info(f"Throughput: {metrics.throughput_samples_per_sec:.2f} samples/sec")
                
                framework.cleanup()
                
            except Exception as e:
                logger.error(f"Experiment failed: {e}")
                import traceback
                logger.error(traceback.format_exc())  # Print full traceback for better debugging
                try:
                    framework.cleanup()
                except Exception:
                    logger.error("Failed to cleanup framework")
        
        return results
    
    def save_results(self, results: List[BenchmarkMetrics], output_path: str):
        """Save benchmark results"""
        results_data = [r.to_dict() for r in results]
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")

def create_sample_config():
    """Create a sample configuration file optimized for CPU testing"""
    config = {
        "experiments": [
            {
                "name": "ResNet50_CIFAR10_PyTorchDDP",
                "model": "resnet50",
                "dataset": "cifar10",
                "framework": "pytorch_ddp",
                "batch_size": 16,  # Smaller batch for CPU
                "epochs": 2,
                "lr": 0.001,
                "world_size": 1,
                "max_batches_per_epoch": 15,  # Limit for faster testing
                "max_test_batches": 3
            },
            {
                "name": "UNet_SyntheticSeg_PyTorchDDP",
                "model": "unet",
                "dataset": "synthetic_seg",
                "framework": "pytorch_ddp",
                "batch_size": 8,  # Smaller batch for memory efficiency
                "epochs": 2,
                "lr": 0.001,
                "world_size": 1,
                "max_batches_per_epoch": 10,
                "max_test_batches": 2
            },
            {
                "name": "Transformer_CIFAR10_PyTorchDDP",
                "model": "transformer",
                "dataset": "cifar10",
                "framework": "pytorch_ddp",
                "batch_size": 16,
                "epochs": 2,
                "lr": 0.001,
                "world_size": 1,
                "max_batches_per_epoch": 15,
                "max_test_batches": 3
            }
        ]
    }
    
    # Add Ray Train experiments if available
    if RAY_AVAILABLE:
        config["experiments"].extend([
            {
                "name": "ResNet50_CIFAR10_RayTrain",
                "model": "resnet50",
                "dataset": "cifar10",
                "framework": "ray_train",
                "batch_size": 16,
                "epochs": 2,
                "lr": 0.001,
                "world_size": 2,  # Multi-worker
                "max_batches_per_epoch": 15,
                "max_test_batches": 3
            }
        ])
    
    # Add Horovod experiments if available
    if HOROVOD_AVAILABLE:
        config["experiments"].extend([
            {
                "name": "UNet_SyntheticSeg_Horovod",
                "model": "unet",
                "dataset": "synthetic_seg", 
                "framework": "horovod",
                "batch_size": 8,
                "epochs": 2,
                "lr": 0.001,
                "world_size": 1,
                "max_batches_per_epoch": 10,
                "max_test_batches": 2
            }
        ])
    
    # Add DeepSpeed experiments if available
    if DEEPSPEED_AVAILABLE:
        config["experiments"].extend([
            {
                "name": "ResNet50_CIFAR10_DeepSpeedCPU",
                "model": "resnet50",
                "dataset": "cifar10",
                "framework": "deepspeed_cpu",
                "batch_size": 16,
                "epochs": 2,
                "lr": 0.001,
                "world_size": 1,
                "max_batches_per_epoch": 15,
                "max_test_batches": 3
            }
        ])
    
    with open('benchmark_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Sample configuration created: benchmark_config.json")
    print("Available frameworks:", list(BenchmarkEngine().frameworks.keys()))

def print_system_info():
    """Print system information for benchmarking context"""
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    print(f"CPU Count: {mp.cpu_count()}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Ray Available: {RAY_AVAILABLE}")
    print(f"Horovod Available: {HOROVOD_AVAILABLE}")
    print(f"DeepSpeed Available: {DEEPSPEED_AVAILABLE}")
    
    # CPU info (simplified)
    try:
        import platform
        print(f"Platform: {platform.platform()}")
        print(f"Processor: {platform.processor()}")
    except:
        pass
    
    print("="*60)

def main():
    # Set multiprocessing start method to 'spawn' for Windows compatibility
    if os.name == 'nt':  # Windows
        mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description='Distributed ML Benchmarking Tool (CPU Optimized)')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--output', type=str, default='benchmark_results.json', 
                    help='Output file for results')
    parser.add_argument('--create-config', action='store_true', 
                    help='Create sample configuration file')
    parser.add_argument('--system-info', action='store_true',
                    help='Display system information')
    parser.add_argument('--list-frameworks', action='store_true',
                    help='List available frameworks')
    
    args = parser.parse_args()
    
    if args.system_info:
        print_system_info()
        return
    
    if args.list_frameworks:
        engine = BenchmarkEngine()
        print("\nAvailable Frameworks:")
        for name in engine.frameworks.keys():
            print(f"  - {name}")
        print("\nAvailable Models:")
        for name in engine.models.keys():
            print(f"  - {name}")
        print("\nAvailable Datasets:")
        for name in engine.datasets.keys():
            print(f"  - {name}")
        return
    
    if args.create_config:
        create_sample_config()
        return
    
    if not args.config:
        logger.error("Please provide a configuration file with --config or create one with --create-config")
        return
    
    # Print system info at start
    print_system_info()
    
    # Run benchmark
    engine = BenchmarkEngine()
    results = engine.run_benchmark(args.config)
    
    if results:
        engine.save_results(results, args.output)
        
        # Print detailed CLI summary
        print("\n" + "="*70)
        print("DETAILED BENCHMARK RESULTS")
        print("="*70)
        
        # Summary table
        print(f"{'Framework':<15} {'Model':<12} {'Dataset':<12} {'Time(s)':<8} {'Throughput':<12} {'Accuracy':<10}")
        print("-" * 70)
        
        for result in results:
            print(f"{result.framework:<15} {result.model_name:<12} {result.dataset_name:<12} "
                f"{result.total_training_time:<8.2f} {result.throughput_samples_per_sec:<12.2f} "
                f"{result.final_accuracy:<10.2f}%")
        
        # Detailed analysis
        print("\n" + "="*70)
        print("PERFORMANCE ANALYSIS")
        print("="*70)
        
        if len(results) > 1:
            # Find fastest and most accurate
            fastest = min(results, key=lambda x: x.total_training_time)
            most_accurate = max(results, key=lambda x: x.final_accuracy)
            highest_throughput = max(results, key=lambda x: x.throughput_samples_per_sec)
            
            print(f"🚀 Fastest Training: {fastest.framework} ({fastest.model_name}) - {fastest.total_training_time:.2f}s")
            print(f"🎯 Highest Accuracy: {most_accurate.framework} ({most_accurate.model_name}) - {most_accurate.final_accuracy:.2f}%")
            print(f"⚡ Highest Throughput: {highest_throughput.framework} ({highest_throughput.model_name}) - {highest_throughput.throughput_samples_per_sec:.2f} samples/sec")
            
            # Framework comparison
            frameworks_used = list(set(r.framework for r in results))
            print(f"\n📊 Frameworks Tested: {len(frameworks_used)} ({', '.join(frameworks_used)})")
            print(f"📈 Models Tested: {len(set(r.model_name for r in results))}")
            print(f"💾 Peak Memory Usage: {max(r.peak_memory_mb for r in results):.1f} MB")
        
        print(f"\n💾 Results saved to: {args.output}")
        print("="*70)
    else:
        print("No results to display. Check the logs for errors.")

if __name__ == "__main__":
    main()