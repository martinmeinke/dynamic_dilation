import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicDilation(nn.Module):

    def __init__(self, in_ch, out_ch, min_dil_range, max_dil_range, smallest_dil, largest_dil):
        """[summary]

        Args:
            in_ch (int): number of input channels
            out_ch (int): number of output channels
            min_dil_range ([type]): range at which the minimal dilation applies
            max_dil_range ([type]): range at which the maximal dilation applies
            smallest_dil ([type]): smallest dilation size (applies at min_dil_range)
            largest_dil ([type]): largest dilation size (applies at max_dil_range)
        """
        super(DynamicDilation, self).__init__()

        self.n_dilation_levels = largest_dil - smallest_dil + 1

        self.in_ch = in_ch
        self.out_ch = out_ch

        self.min_dil_range = min_dil_range
        self.max_dil_range = max_dil_range
        self.smallest_dil = smallest_dil
        self.largest_dil = largest_dil

        # create convolution layer for each dilation
        self.conv_layers = nn.ModuleList([nn.Conv2d(in_ch, out_ch, (3, 3), stride=(1, 1), dilation=(
            d, d), padding=(d, d)) for d in range(smallest_dil, largest_dil+1)])

    def forward(self, x, range_image):
        conv_features = [c(x) for c in self.conv_layers]

        # interpolate range image to match input feature size
        range_image_interp = F.interpolate(range_image, x.shape[2:])
        dilstack = torch.stack(conv_features)

        # select best dilation by looking into the range_image_interp
        range_bin_size = (self.min_dil_range - self.max_dil_range) / \
            self.n_dilation_levels

        # clamp range image to the range in which we intend to scale dilation
        range_image_interp_clamp = torch.clamp(
            range_image_interp, self.max_dil_range, self.min_dil_range)

        dilations = ((self.min_dil_range - range_image_interp_clamp) /
                     range_bin_size + 1).long()

        dilations = torch.clamp(
            dilations, self.smallest_dil, self.largest_dil) - 1
        dilations = dilations.repeat(1, self.out_ch, 1, 1)
        dilations = dilations.unsqueeze(0)

        # gather features from proper dilation
        thatsit = torch.gather(dilstack, 0, dilations)

        return thatsit.squeeze(0)
