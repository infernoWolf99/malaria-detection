import torchvision
from torchvision.models.detection.retinanet import RetinaNet_ResNet50_FPN_Weights


class RetinaNet:
    """
    A wrapper class for initializing RetinaNet models for malaria detection.
    """

    def __init__(self, num_classes: int, pre_trained: bool = False):
        self.num_classes = num_classes
        self.pre_trained = pre_trained
        self.model = self._build_model()

    def _build_model(self):
        weights = None
        
        if self.pre_trained:
            weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT

        model = torchvision.models.detection.retinanet_resnet50_fpn(
            weights=weights, num_classes=self.num_classes
        )
        return model

    def get_model(self):
        """Returns the raw PyTorch model execution graph."""
        return self.model
