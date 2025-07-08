from abc import ABC, abstractmethod
from typing import Dict, Any, Iterator, Literal
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter

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
    
    def create_optimizer(self, model: nn.Module, lr: float) -> optim.Optimizer:
        return optim.SGD(self.get_trainable_params(model), lr=lr, momentum=0.0)
    
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
    
    def create_optimizer(self, model: nn.Module, lr: float) -> optim.Optimizer:
        return optim.SGD(self.get_trainable_params(model), lr=lr, momentum=0.0)
    
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
    
    def create_optimizer(self, model: nn.Module, optimizer_type: str = "adam", lr: float = 0.001, momentum: float = 0.0) -> optim.Optimizer:
        trainable_params = self.get_trainable_params(model)
        if optimizer_type.lower() == "sgd":
            print(f"Using optimizer: SGD with momentum {momentum}")
            return optim.SGD(trainable_params, lr=lr, momentum=momentum)
        elif optimizer_type.lower() == "adam":
            print(f"Using optimizer: Adam")
            return optim.Adam(trainable_params, lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        
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

    def create_optimizer(self, model: nn.Module, lr: float) -> optim.Optimizer:
        return optim.SGD(self.get_trainable_params(model), lr=lr, momentum=0.0)

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
