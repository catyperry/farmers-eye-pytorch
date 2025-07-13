from abc import ABC, abstractmethod
from typing import Dict, Any, Iterator, Literal
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
import torch
import math

class WAdam(optim.Optimizer):
    """
    WAdam (Weighted Adam) optimizer
    Implements weighted Adam algorithm with adaptive weight decay.
    
    Args:
        params: iterable of parameters to optimize
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing running averages (default: (0.9, 0.999))
        eps: term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay: weight decay (L2 penalty) (default: 1e-2)
        amsgrad: whether to use the AMSGrad variant (default: False)
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=1e-2, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super(WAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data).float()
                    state['exp_avg_sq'] = torch.zeros_like(p.data).float()
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data).float()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Weighted decay
                if group['weight_decay'] != 0:
                    # Adaptive weight decay based on gradient magnitude
                    grad_norm = grad.norm()
                    adaptive_weight_decay = group['weight_decay'] / (1 + grad_norm)
                    p.data.mul_(1 - group['lr'] * adaptive_weight_decay)

                # Exponential moving average of gradient values
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Exponential moving average of squared gradient values
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if group['amsgrad']:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

class ModelConfig(ABC):
    @abstractmethod
    def create_model(self, num_classes: int) -> nn.Module:
        pass
    
    @abstractmethod
    def create_optimizer(self, model: nn.Module, lr: float) -> optim.Optimizer:
        pass
    
    @abstractmethod
    def get_default_hyperparams(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_trainable_params(self, model: nn.Module) -> Iterator[Parameter]:
        """Return the parameters that should be trained"""
        pass

class MobileNetV2Config(ModelConfig):
    def create_model(self, num_classes: int) -> nn.Module:
        from torchvision import models
        from torchvision.models import MobileNet_V2_Weights
        
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False
        # Replace classifier
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        return model
    
    def get_trainable_params(self, model: nn.Module) -> Iterator[Parameter]:
        classifier_layer = model.classifier[1]  # type: ignore
        if not isinstance(classifier_layer, nn.Linear):
            raise TypeError("Expected classifier layer to be nn.Linear")
        return classifier_layer.parameters()
    
    def create_optimizer(self, model: nn.Module, optimizer_type: str = "sgd", lr: float = 0.001, momentum: float = 0.0, weight_decay: float = 0.01) -> optim.Optimizer:
        trainable_params = self.get_trainable_params(model)
        if optimizer_type.lower() == "sgd":
            print(f"Using optimizer: SGD with momentum {momentum}")
            return optim.SGD(trainable_params, lr=lr, momentum=momentum)
        elif optimizer_type.lower() == "adam":
            print(f"Using optimizer: Adam")
            return optim.Adam(trainable_params, lr=lr)
        elif optimizer_type.lower() == "adamw":
            print(f"Using optimizer: AdamW (Adam with weight decay {weight_decay})")
            return optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}. Supported: 'sgd', 'adam', 'adamw'")
    
    def get_default_hyperparams(self) -> Dict[str, Any]:
        return {
            'learning_rate': 0.0035148759,
            'batch_size': 1000,
            'num_epochs': 10,
            'test_every_x_epochs': 20
        }

class ViTHugeConfig(ModelConfig):
    def create_model(self, num_classes: int) -> nn.Module:
        import timm
        
        model = timm.create_model("vit_huge_patch14_224", pretrained=True, num_classes=num_classes)
        # Freeze all layers except head
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True
        return model
    
    def get_trainable_params(self, model: nn.Module) -> Iterator[Parameter]:
        # Type cast to help Pylance understand the structure
        vit_model = model  # type: ignore
        return vit_model.head.parameters()  # type: ignore
    
    def create_optimizer(self, model: nn.Module, optimizer_type: str = "sgd", lr: float = 0.001, momentum: float = 0.0, weight_decay: float = 0.01) -> optim.Optimizer:
        trainable_params = self.get_trainable_params(model)
        if optimizer_type.lower() == "sgd":
            print(f"Using optimizer: SGD with momentum {momentum}")
            return optim.SGD(trainable_params, lr=lr, momentum=momentum)
        elif optimizer_type.lower() == "adam":
            print(f"Using optimizer: Adam")
            return optim.Adam(trainable_params, lr=lr)
        elif optimizer_type.lower() == "adamw":
            print(f"Using optimizer: AdamW (Adam with weight decay {weight_decay})")
            return optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}. Supported: 'sgd', 'adam', 'adamw'")
    
    def get_default_hyperparams(self) -> Dict[str, Any]:
        return {
            'learning_rate': 0.0035148759,
            'batch_size': 64,  # Smaller batch size for huge model
            'num_epochs': 10,
            'test_every_x_epochs': 20
        }
    
class ViTBaseConfig(ModelConfig):
    def create_model(self, num_classes: int) -> nn.Module:
        import timm

        model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
        # Freeze all layers except head
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True
        return model
    
    def get_trainable_params(self, model: nn.Module) -> Iterator[Parameter]:
        # Type cast to help Pylance understand the structure
        vit_model = model  # type: ignore
        return vit_model.head.parameters()  # type: ignore
    
    def create_optimizer(self, model: nn.Module, optimizer_type: str = "adam", lr: float = 0.001, momentum: float = 0.0, weight_decay: float = 0.01) -> optim.Optimizer:
        trainable_params = self.get_trainable_params(model)
        if optimizer_type.lower() == "sgd":
            print(f"Using optimizer: SGD with momentum {momentum}")
            return optim.SGD(trainable_params, lr=lr, momentum=momentum)
        elif optimizer_type.lower() == "adam":
            print(f"Using optimizer: Adam")
            return optim.Adam(trainable_params, lr=lr)
        elif optimizer_type.lower() == "adamw":
            print(f"Using optimizer: AdamW (Adam with weight decay {weight_decay})")
            return optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}. Supported: 'sgd', 'adam', 'adamw'")
        
        
    def get_default_hyperparams(self) -> Dict[str, Any]:
        return {
            'learning_rate': 0.0035148759,
            'batch_size': 1000,
            'num_epochs': 10,
            'test_every_x_epochs': 20
        }

class ResNet50Config(ModelConfig):
    def create_model(self, num_classes: int) -> nn.Module:
        from torchvision import models
        from torchvision.models import ResNet50_Weights

        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False
        # Replace the final fully connected layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    def get_trainable_params(self, model: nn.Module) -> Iterator[Parameter]:
        classifier_layer = model.fc  # type: ignore
        if not isinstance(classifier_layer, nn.Linear):
            raise TypeError("Expected final layer to be nn.Linear")
        return classifier_layer.parameters()

    def create_optimizer(self, model: nn.Module, optimizer_type: str = "adam", lr: float = 0.001, momentum: float = 0.0, weight_decay: float = 0.01) -> optim.Optimizer:
        trainable_params = self.get_trainable_params(model)
        if optimizer_type.lower() == "sgd":
            print(f"Using optimizer: SGD with momentum {momentum}")
            return optim.SGD(trainable_params, lr=lr, momentum=momentum)
        elif optimizer_type.lower() == "adam":
            print(f"Using optimizer: Adam")
            return optim.Adam(trainable_params, lr=lr)
        elif optimizer_type.lower() == "adamw":
            print(f"Using optimizer: AdamW (Adam with weight decay {weight_decay})")
            return optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}. Supported: 'sgd', 'adam', 'adamw'")
        

    def get_default_hyperparams(self) -> Dict[str, Any]:
        return {
            'learning_rate': 0.0035148759,
            'batch_size': 240,
            'num_epochs': 3000,
            'test_every_x_epochs': 20
        }

# Registry
MODEL_REGISTRY = {
    'mobilenet_v2': MobileNetV2Config(),
    'vit_huge_patch14_224': ViTHugeConfig(),
    'vit_base_patch16_224': ViTBaseConfig(),
    'resnet50': ResNet50Config(),
}

MODEL_NAME = Literal['mobilenet_v2', 'vit_huge_patch14_224', 'vit_base_patch16_224', 'resnet50']
