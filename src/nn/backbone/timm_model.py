"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.

https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055#0583
"""

import torch
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

from ...core import register
from .utils import IntermediateLayerGetter


@register()
class TimmModel(torch.nn.Module):
    def __init__(
        self, name, return_layers, pretrained=False, exportable=True, features_only=True, **kwargs
    ) -> None:
        super().__init__()

        import timm

        model = timm.create_model(
            name,
            pretrained=pretrained,
            exportable=exportable,
            features_only=features_only,
            **kwargs,
        )
        # nodes, _ = get_graph_node_names(model)
        # print(nodes)
        # features = {'': ''}
        # model = create_feature_extractor(model, return_nodes=features)

        assert set(return_layers).issubset(
            model.feature_info.module_name()
        ), f"return_layers should be a subset of {model.feature_info.module_name()}"

        # model.feature_info uses dotted names (e.g. 'stages.1'),
        # but actual module children use underscores (e.g. 'stages_1').
        # Map to actual child names for IntermediateLayerGetter.
        fixed_return_layers = [name.replace('.', '_') for name in return_layers]
        self.model = IntermediateLayerGetter(model, fixed_return_layers)

        return_idx = [model.feature_info.module_name().index(name) for name in return_layers]
        self.strides = [model.feature_info.reduction()[i] for i in return_idx]
        self.channels = [model.feature_info.channels()[i] for i in return_idx]
        self.return_idx = return_idx
        self.return_layers = return_layers

    def forward(self, x: torch.Tensor):
        outputs = self.model(x)
        # outputs = [outputs[i] for i in self.return_idx]
        return outputs


if __name__ == "__main__":
    model = TimmModel(name="resnet34", return_layers=["layer2", "layer3"])
    data = torch.rand(1, 3, 640, 640)
    outputs = model(data)

    for output in outputs:
        print(output.shape)

    """
    model:
        type: TimmModel
        name: resnet34
        return_layers: ['layer2', 'layer4']
    """
