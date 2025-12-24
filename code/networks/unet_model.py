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

    def forward(self, x, feature=False, is_labeled=False, mask_idx=None, return_mask=False):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        mask_idx_out = None
        if is_labeled:
            if self.channel_mask.enabled:
                self.channel_mask.update_from_labeled(x5)
                mask_idx_out = self.channel_mask.sample_mask_indices(x5.size(1), x5.device)
        elif mask_idx is not None and self.channel_mask.enabled:
            x5 = self.channel_mask.apply_mask(x5, mask_idx)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        if feature and return_mask:
            return logits, x, mask_idx_out
        if feature:
            return logits, x
        if return_mask:
            return logits, mask_idx_out
        return logits
