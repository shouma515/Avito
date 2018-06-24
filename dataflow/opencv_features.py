import time
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import itemfreq
import io
from PIL import Image

import cv2


def _get_dominant_color(img):
#     path = images_path + img 
#     img = cv2.imread(path)
    img = np.array(img) 
    # Convert RGB to BGR 
    img = img[:, :, ::-1].copy() 
    arr = np.float32(img)
    pixels = arr.reshape((-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)

    palette = np.uint8(centroids)
    quantized = palette[labels.flatten()]
    quantized = quantized.reshape(img.shape)

    dominant_color = palette[np.argmax(itemfreq(labels)[:, -1])]
    return dominant_color


def _get_blurrness_score(image):
  image = np.array(image) 
  # Convert RGB to BGR 
  image = image[:, :, ::-1].copy() 
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  fm = cv2.Laplacian(image, cv2.CV_64F).var()
  return fm

def main():
    archive = zipfile.ZipFile('data/train_jpg.zip', 'r')
    # filenames = archive.namelist()[1:]
    train_df = pd.read_pickle('pickles/df_train')
    f_out = open('image_features/opencv_features.csv', 'w')
    t_start = time.time()
    for idx, row in train_df.iterrows():
        if idx >= 1000:
            break
        item_id = row['item_id']
        image_id = row['image']
        if (image_id is None) or ('nan' in str(image_id)):
            # print('id none')
            continue
        image_file = 'data/competition_files/train_jpg/%s.jpg' %image_id
        try:
            with archive.open(image_file) as f:
                # image_bytes = f.read()
                # img = Image.open(io.BytesIO(image_bytes))
                img = Image.open(f)
                dominant_r, dominant_g, dominant_b = _get_dominant_color(img)
                blurrness = '%.3f' %_get_blurrness_score(img)
                # image_id = image_file.split('/')[-1].split('.')[0]
                features = [item_id, image_id, str(dominant_r), str(dominant_g), str(dominant_b), blurrness]
                # print('write features')
                f_out.write(','.join(features) + '\n')
        except IOError:  # pylint: disable=broad-except
            print('cannot find image', image_file)
            continue
    t_finish = time.time()
    print('Total running time: ', (t_finish - t_start) / 60)
    f_out.close()

if __name__ == '__main__':
    main()
