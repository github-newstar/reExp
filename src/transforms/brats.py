import torch
from monai.transforms import MapTransform


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert BraTS labels to multi-channel format.

    Input labels:
    - 0: background
    - 1: NCR
    - 2: ED
    - 3: ET (BraTS23 convention)

    Output channels:
    - channel 0: TC = NCR + ET
    - channel 1: WT = NCR + ED + ET
    - channel 2: ET
    """

    def __call__(self, data):
        output = dict(data)
        for key in self.keys:
            label = torch.as_tensor(output[key])
            if label.ndim == 4 and label.shape[0] == 1:
                label = label.squeeze(0)

            tc = (label == 1) | (label == 3)
            wt = (label == 1) | (label == 2) | (label == 3)
            et = label == 3
            output[key] = torch.stack([tc, wt, et], dim=0).float()
        return output
