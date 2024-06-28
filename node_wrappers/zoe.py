from ..utils import common_annotator_call, create_node_input_types
import comfy.model_management as model_management
from controlnet_aux.zoe import ZoeDetector

class Zoe_Depth_Map_Preprocessor:
    def __init__(self) -> None:
        self.model = ZoeDetector.from_pretrained().to(model_management.get_torch_device())

    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types()

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Normal and Depth Estimators"

    def execute(self, image, resolution=512, **kwargs):
        out = common_annotator_call(self.model, image, resolution=resolution)
        # del model
        return (out, )

NODE_CLASS_MAPPINGS = {
    "Zoe-DepthMapPreprocessor": Zoe_Depth_Map_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Zoe-DepthMapPreprocessor": "Zoe Depth Map"
}