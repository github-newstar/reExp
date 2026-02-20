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


class RandETFocusedCropd(MapTransform):
    """
    Random crop with ET-prioritized center sampling for BraTS labels.

    The crop center is sampled from:
    1) ET voxels (channel 2) with probability `et_focus_prob`, if available.
    2) Otherwise WT voxels (channel 1), if available.
    3) Otherwise uniform random center.

    This helps increase ET-positive patches in training.
    """

    def __init__(
        self,
        keys,
        roi_size,
        label_key="label",
        et_channel=2,
        wt_channel=1,
        et_focus_prob=0.7,
        random_prob=0.33,
    ):
        super().__init__(keys)
        self.roi_size = tuple(int(x) for x in roi_size)
        self.label_key = label_key
        self.et_channel = int(et_channel)
        self.wt_channel = int(wt_channel)
        self.et_focus_prob = float(et_focus_prob)
        self.random_prob = float(random_prob)

    def _sample_center(self, label_tensor: torch.Tensor):
        # label_tensor: [C, D, H, W]
        spatial_shape = tuple(int(x) for x in label_tensor.shape[-3:])
        # Support both:
        # 1) multi-channel BraTS labels [TC, WT, ET]
        # 2) scalar single-channel BraTS labels [1, D, H, W] in {0,1,2,3}
        if label_tensor.shape[0] == 1:
            scalar = label_tensor[0]
            et_mask = scalar == 3
            wt_mask = scalar > 0
        else:
            if self.et_channel >= label_tensor.shape[0] or self.wt_channel >= label_tensor.shape[0]:
                raise ValueError(
                    f"Label channels={label_tensor.shape[0]} are insufficient for "
                    f"et_channel={self.et_channel}, wt_channel={self.wt_channel}"
                )
            et_mask = label_tensor[self.et_channel] > 0
            wt_mask = label_tensor[self.wt_channel] > 0

        center = None
        
        if torch.rand(1).item() > self.random_prob:
            if torch.rand(1).item() < self.et_focus_prob:
                et_coords = torch.nonzero(et_mask, as_tuple=False)
                if et_coords.numel() > 0:
                    center = et_coords[torch.randint(et_coords.shape[0], (1,)).item()]

            if center is None:
                wt_coords = torch.nonzero(wt_mask, as_tuple=False)
                if wt_coords.numel() > 0:
                    center = wt_coords[torch.randint(wt_coords.shape[0], (1,)).item()]

        if center is None:
            center = torch.tensor(
                [
                    torch.randint(spatial_shape[0], (1,)).item(),
                    torch.randint(spatial_shape[1], (1,)).item(),
                    torch.randint(spatial_shape[2], (1,)).item(),
                ],
                dtype=torch.long,
            )
        return center, spatial_shape

    def _compute_slices(self, center, spatial_shape):
        starts = []
        ends = []
        for dim_idx in range(3):
            dim = int(spatial_shape[dim_idx])
            size = int(self.roi_size[dim_idx])
            start = int(center[dim_idx]) - size // 2
            start = max(0, min(start, dim - size))
            end = start + size
            starts.append(start)
            ends.append(end)
        return tuple(starts), tuple(ends)

    def __call__(self, data):
        output = dict(data)
        label = torch.as_tensor(output[self.label_key])
        if label.ndim != 4:
            raise ValueError(
                f"Expected {self.label_key} shape [C,D,H,W], got {tuple(label.shape)}"
            )

        center, spatial_shape = self._sample_center(label)
        starts, ends = self._compute_slices(center=center, spatial_shape=spatial_shape)
        z0, y0, x0 = starts
        z1, y1, x1 = ends

        for key in self.keys:
            tensor = torch.as_tensor(output[key])
            if tensor.ndim == 4:
                output[key] = tensor[:, z0:z1, y0:y1, x0:x1]
            elif tensor.ndim == 3:
                output[key] = tensor[z0:z1, y0:y1, x0:x1]
            else:
                raise ValueError(
                    f"Unsupported tensor rank for key={key}: shape={tuple(tensor.shape)}"
                )
        return output
