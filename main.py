from image_process import Segmenter, get_image_from_url, display_image
from id_to_class import ID_TO_CLASS
import numpy as np

segmenter = Segmenter()
##modify image url
img_url = "https://upload.wikimedia.org/wikipedia/commons/5/5a/Batik_Fashion_01.jpg"
img = get_image_from_url(img_url)
segmap, id_to_class = segmenter.predict_on_url(img_url)
#display_image(img, segmap)
print(id_to_class)
segmap = np.array(segmap)
mask = (segmap == 1) | (segmap == 9)
segmap *= mask
#display_image(img, segmap)