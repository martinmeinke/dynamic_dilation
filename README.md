# Dynamic Dilation Layer for Pytorch
Custom pytorch layer to promote learing of scale invariant features.

The Layer applies the same convolution kernels using different dilation factors, depending on the location within the image for which a corresponding range measurement is available. Locations with measurements corresponding to close ranges apply the kernel using a high dilation, far away locations with lower dilations.


![Image](dynamic_dilation.png)

```python
# Interpolate dilation factors between 1 and 5, for ranges between 50 and 0 (clip dilation factor outside range)
dd = DynamicDilation(in_channels, out_channels, min_dil_range=50, max_dil_range=0, smallest_dil=1, largest_dil=5)
# pass features from prev layer (or input) along with the range image
out_features = dd(in_features, in_range_image)
```
