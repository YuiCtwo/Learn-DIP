# An excellent Python implementation of SeamCarving in https://github.com/li-plus/seam-carving
# Using it as Lib: pip install seam-carving

# A demo code from README
import numpy as np
from PIL import Image
import seam_carving

src = np.array(Image.open("./castle.jpg"))
src_h, src_w, _ = src.shape
dst = seam_carving.resize(
    src,  # source image (rgb or gray)
    size=(src_w - 200, src_h),  # target size
    energy_mode="backward",  # choose from {backward, forward}
    order="width-first",  # choose from {width-first, height-first}
    keep_mask=None,  # object mask to protect from removal
)
Image.fromarray(dst).show()