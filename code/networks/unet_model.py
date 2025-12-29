""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from channel_mask import AdaptiveChannelMask


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, channel_mask: AdaptiveChannelMask = None):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.channel_mask = channel_mask if channel_mask is not None else AdaptiveChannelMask(enabled=False)
        self.mask_layers = nn.ModuleDict()
        if self.channel_mask.enabled:
            self.mask_layers = nn.ModuleDict({
                "enc1": AdaptiveChannelMask(
                    mask_percent=self.channel_mask.mask_percent,
                    rank_percent=self.channel_mask.rank_percent,
                    mask_value=self.channel_mask.mask_value,
                    ema_momentum=self.channel_mask.ema_momentum,
                    enabled=True,
                ),
                "enc2": AdaptiveChannelMask(
                    mask_percent=self.channel_mask.mask_percent,
                    rank_percent=self.channel_mask.rank_percent,
                    mask_value=self.channel_mask.mask_value,
                    ema_momentum=self.channel_mask.ema_momentum,
                    enabled=True,
                ),
                "enc3": AdaptiveChannelMask(
                    mask_percent=self.channel_mask.mask_percent,
                    rank_percent=self.channel_mask.rank_percent,
                    mask_value=self.channel_mask.mask_value,
                    ema_momentum=self.channel_mask.ema_momentum,
                    enabled=True,
                ),
                "enc4": AdaptiveChannelMask(
                    mask_percent=self.channel_mask.mask_percent,
                    rank_percent=self.channel_mask.rank_percent,
                    mask_value=self.channel_mask.mask_value,
                    ema_momentum=self.channel_mask.ema_momentum,
                    enabled=True,
                ),
                "bottleneck": AdaptiveChannelMask(
                    mask_percent=self.channel_mask.mask_percent,
                    rank_percent=self.channel_mask.rank_percent,
                    mask_value=self.channel_mask.mask_value,
                    ema_momentum=self.channel_mask.ema_momentum,
                    enabled=True,
                ),
                "dec1": AdaptiveChannelMask(
                    mask_percent=self.channel_mask.mask_percent,
                    rank_percent=self.channel_mask.rank_percent,
                    mask_value=self.channel_mask.mask_value,
                    ema_momentum=self.channel_mask.ema_momentum,
                    enabled=True,
                ),
                "dec2": AdaptiveChannelMask(
                    mask_percent=self.channel_mask.mask_percent,
                    rank_percent=self.channel_mask.rank_percent,
                    mask_value=self.channel_mask.mask_value,
                    ema_momentum=self.channel_mask.ema_momentum,
                    enabled=True,
                ),
                "dec3": AdaptiveChannelMask(
                    mask_percent=self.channel_mask.mask_percent,
                    rank_percent=self.channel_mask.rank_percent,
                    mask_value=self.channel_mask.mask_value,
                    ema_momentum=self.channel_mask.ema_momentum,
                    enabled=True,
                ),
                "dec4": AdaptiveChannelMask(
                    mask_percent=self.channel_mask.mask_percent,
                    rank_percent=self.channel_mask.rank_percent,
                    mask_value=self.channel_mask.mask_value,
                    ema_momentum=self.channel_mask.ema_momentum,
                    enabled=True,
                ),
            })

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def _resolve_mask_layers(self, mask_at):
        if mask_at is None:
            return ["bottleneck"]
        if isinstance(mask_at, str):
            return [mask_at]
        return list(mask_at)

    def _update_layer_masks(self, x, layer_name, mask_idx_out):
        if layer_name not in self.mask_layers:
            raise ValueError(f"Unsupported mask layer: {layer_name}")
        layer_mask = self.mask_layers[layer_name]
        layer_mask.update_from_labeled(x)
        mask_idx_out[layer_name] = layer_mask.sample_mask_indices(x.size(1), x.device)

    def _apply_layer_mask(self, x, layer_name, mask_idx):
        if layer_name not in self.mask_layers:
            raise ValueError(f"Unsupported mask layer: {layer_name}")
        return self.mask_layers[layer_name].apply_mask(x, mask_idx)

    def forward(self, x, feature=False, is_labeled=False, mask_idx=None, return_mask=False, mask_at=None):
        mask_layers = self._resolve_mask_layers(mask_at)
        x1 = self.inc(x)
        if is_labeled and self.mask_layers and "enc1" in mask_layers:
            mask_idx_out = {}
            self._update_layer_masks(x1, "enc1", mask_idx_out)
        elif mask_idx is not None and "enc1" in mask_layers:
            x1 = self._apply_layer_mask(x1, "enc1", mask_idx["enc1"])
        x2 = self.down1(x1)
        if is_labeled and self.mask_layers and "enc2" in mask_layers:
            mask_idx_out = mask_idx_out if "mask_idx_out" in locals() else {}
            self._update_layer_masks(x2, "enc2", mask_idx_out)
        elif mask_idx is not None and "enc2" in mask_layers:
            x2 = self._apply_layer_mask(x2, "enc2", mask_idx["enc2"])
        x3 = self.down2(x2)
        if is_labeled and self.mask_layers and "enc3" in mask_layers:
            mask_idx_out = mask_idx_out if "mask_idx_out" in locals() else {}
            self._update_layer_masks(x3, "enc3", mask_idx_out)
        elif mask_idx is not None and "enc3" in mask_layers:
            x3 = self._apply_layer_mask(x3, "enc3", mask_idx["enc3"])
        x4 = self.down3(x3)
        if is_labeled and self.mask_layers and "enc4" in mask_layers:
            mask_idx_out = mask_idx_out if "mask_idx_out" in locals() else {}
            self._update_layer_masks(x4, "enc4", mask_idx_out)
        elif mask_idx is not None and "enc4" in mask_layers:
            x4 = self._apply_layer_mask(x4, "enc4", mask_idx["enc4"])
        x5 = self.down4(x4)
        if is_labeled and self.mask_layers and "bottleneck" in mask_layers:
            mask_idx_out = mask_idx_out if "mask_idx_out" in locals() else {}
            self._update_layer_masks(x5, "bottleneck", mask_idx_out)
        elif mask_idx is not None and "bottleneck" in mask_layers:
            x5 = self._apply_layer_mask(x5, "bottleneck", mask_idx["bottleneck"])
        x = self.up1(x5, x4)
        if is_labeled and self.mask_layers and "dec1" in mask_layers:
            mask_idx_out = mask_idx_out if "mask_idx_out" in locals() else {}
            self._update_layer_masks(x, "dec1", mask_idx_out)
        elif mask_idx is not None and "dec1" in mask_layers:
            x = self._apply_layer_mask(x, "dec1", mask_idx["dec1"])
        x = self.up2(x, x3)
        if is_labeled and self.mask_layers and "dec2" in mask_layers:
            mask_idx_out = mask_idx_out if "mask_idx_out" in locals() else {}
            self._update_layer_masks(x, "dec2", mask_idx_out)
        elif mask_idx is not None and "dec2" in mask_layers:
            x = self._apply_layer_mask(x, "dec2", mask_idx["dec2"])
        x = self.up3(x, x2)
        if is_labeled and self.mask_layers and "dec3" in mask_layers:
            mask_idx_out = mask_idx_out if "mask_idx_out" in locals() else {}
            self._update_layer_masks(x, "dec3", mask_idx_out)
        elif mask_idx is not None and "dec3" in mask_layers:
            x = self._apply_layer_mask(x, "dec3", mask_idx["dec3"])
        x = self.up4(x, x1)
        if is_labeled and self.mask_layers and "dec4" in mask_layers:
            mask_idx_out = mask_idx_out if "mask_idx_out" in locals() else {}
            self._update_layer_masks(x, "dec4", mask_idx_out)
        elif mask_idx is not None and "dec4" in mask_layers:
            x = self._apply_layer_mask(x, "dec4", mask_idx["dec4"])
        logits = self.outc(x)
        if feature and return_mask:
            return logits, x, mask_idx_out
        if feature:
            return logits, x
        if return_mask:
            return logits, mask_idx_out
        return logits
