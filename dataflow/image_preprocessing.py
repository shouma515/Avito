"""A image preprocessing workflow."""

from __future__ import absolute_import

import argparse
# import zipfile
import csv
import io
import logging
import operator
from collections import defaultdict

import numpy as np
import scipy
from PIL import Image, ImageStat
from skimage import feature

import apache_beam as beam
# import cv2
from apache_beam.io import ReadFromText, WriteToText
from apache_beam.metrics import Metrics
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io

# # Inorder to use Keras and other cnn packages, need to provide a requirement
# # file so the Dataflow workers can have the proper packages installed. Method
# # see: https://beam.apache.org/documentation/sdks/python-pipeline-dependencies/
# from keras import applications
# from keras.applications.vgg16 import VGG16

# # import warnings
# # warnings.filterwarnings("always")

# # Increase allowed recursion max, so that the VGG16 module can be import
# # globally, and then pickled and send to workers.
# # See https://cloud.google.com/dataflow/faq and search --save_main_session.
# import sys
# sys.setrecursionlimit(5000)

class ProcessImageDoFn(beam.DoFn):
  """Parse each line of input text into words."""

  def __init__(self):
    self.image_counter = Metrics.counter(self.__class__, 'image')
    self.missing_image_counter = Metrics.counter(self.__class__, 'missing_image')
    self.error_processing_image_counter = Metrics.counter(self.__class__, 'error_processing_image')

  def _color_analysis(self, img):
    # obtain the color palatte of the image 
    palatte = defaultdict(int)
    for pixel in img.getdata():
        palatte[pixel] += 1
    
    # sort the colors present in the image 
    sorted_x = sorted(palatte.items(), key=operator.itemgetter(1), reverse = True)
    light_shade, dark_shade, shade_count, pixel_limit = 0, 0, 0, 25
    for i, x in enumerate(sorted_x[:pixel_limit]):
        if all(xx <= 20 for xx in x[0][:3]): ## dull : too much darkness 
            dark_shade += x[1]
        if all(xx >= 240 for xx in x[0][:3]): ## bright : too much whiteness 
            light_shade += x[1]
        shade_count += x[1]
        
    light_percent = round((float(light_shade)/shade_count)*100, 2)
    dark_percent = round((float(dark_shade)/shade_count)*100, 2)
    return light_percent, dark_percent

  def _perform_color_analysis(self, im, flag):
#     path = images_path + img 
#     im = IMG.open(path) #.convert("RGB")
    
    # cut the images into two halves as complete average may give bias results
    size = im.size
    halves = (size[0]/2, size[1]/2)
    im1 = im.crop((0, 0, size[0], halves[1]))
    im2 = im.crop((0, halves[1], size[0], size[1]))

    try:
        light_percent1, dark_percent1 = self._color_analysis(im1)
        light_percent2, dark_percent2 = self._color_analysis(im2)
    except Exception:
        return None

    light_percent = (light_percent1 + light_percent2)/2 
    dark_percent = (dark_percent1 + dark_percent2)/2 
    
    if flag == 'black':
        return dark_percent
    elif flag == 'white':
        return light_percent
    else:
        return None
    
  def _average_pixel_width(self, im): 
    im_array = np.asarray(im.convert(mode='L'))
    edges_sigma1 = feature.canny(im_array, sigma=3)
    apw = (float(np.sum(edges_sigma1)) / (im.size[0]*im.size[1]))
    return apw*100
  
#   def _get_dominant_color(self, img):
# #     path = images_path + img 
# #     img = cv2.imread(path)
#     img = np.array(img) 
#     # Convert RGB to BGR 
#     img = img[:, :, ::-1].copy() 
#     arr = np.float32(img)
#     pixels = arr.reshape((-1, 3))

#     n_colors = 5
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
#     flags = cv2.KMEANS_RANDOM_CENTERS
#     _, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)

#     palette = np.uint8(centroids)
#     quantized = palette[labels.flatten()]
#     quantized = quantized.reshape(img.shape)

#     dominant_color = palette[np.argmax(scipy.stats.itemfreq(labels)[:, -1])]
#     return dominant_color
  
  def _get_average_color(self, img):
    img = np.array(img) 
    # Convert RGB to BGR 
    img = img[:, :, ::-1].copy() 
    average_color = [img[:, :, i].mean() for i in range(img.shape[-1])]
    return average_color
  
  # def _get_blurrness_score(self, image):
  #   image = np.array(image) 
  #   # Convert RGB to BGR 
  #   image = image[:, :, ::-1].copy() 
  #   image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  #   fm = cv2.Laplacian(image, cv2.CV_64F).var()
  #   return fm

  def process(self, element):
    """Process an image. Use the image id to read the image from Google cloud
       storage, process the image, then return item_id and the generated image
       features.
    Args:
      element: the element being processed
    Returns:
      The processed element.
    """
    item_id, image_id = element[0], element[1]
    if not image_id:
      self.missing_image_counter.inc()
      return

    # image_uri = "gs://avito-kaggle/train_jpg/%s.jpg" %image_id
    image_uri = "gs://avito-kaggle/test_jpg/%s.jpg" %image_id

    try:
      with file_io.FileIO(image_uri, mode='rb') as f:
      # with gcs.open(image_uri) as f:
        image_bytes = f.read()
        img = Image.open(io.BytesIO(image_bytes))
    # A variety of different calling libraries throw different exceptions here.
    # They all correspond to an unreadable file so we treat them equivalently.
    except Exception as e:  # pylint: disable=broad-except
      logging.exception('Error processing image %s: %s', image_uri, str(e))
      self.error_processing_image_counter.inc()
      return

    # local test.
    # archive = zipfile.ZipFile('data/train_jpg.zip', 'r')
    # image_file = 'data/competition_files/train_jpg/%s.jpg' %image_id
    # try:
    #   with archive.open(image_file) as f:
    #     image_bytes = f.read()
    #     img = Image.open(io.BytesIO(image_bytes))
    # except Exception as e:  # pylint: disable=broad-except
    #   logging.exception('Error processing image %s: %s', image_id, str(e))
    #   return

    # Count number of image processed.
    self.image_counter.inc()

    # Simple pixel size feature
    width, height = img.size

    img_yuv = img.convert('YCbCr')
    #min, max, avg, std of luminance
    luminance_min, luminance_max = ImageStat.Stat(img_yuv).extrema[0]
    luminance_avg = ImageStat.Stat(img_yuv).mean[0]
    luminance_std = ImageStat.Stat(img_yuv).stddev[0]

    img_hsv = img.convert('HSV')
    #min, max, avg, std of saturation
    saturation_min, saturation_max = ImageStat.Stat(img_hsv).extrema[0]
    saturation_avg = ImageStat.Stat(img_hsv).mean[0]
    saturation_std = ImageStat.Stat(img_hsv).stddev[0]

    pix = np.array(img)/255
    rg = (pix[:,:,0]-pix[:,:,1]).flatten()
    yb = ((pix[:,:,0]+pix[:,:,1])/2 - pix[:,:,2]).flatten()
    s_rg = rg.std()
    s_yb = yb.std()
    u_rg = rg.mean()
    u_yb = yb.mean()
    # Colorfulness
    colorfulness = np.sqrt(s_rg**2+s_yb**2)+0.3*np.sqrt(u_rg**2+u_yb**2)

    # Lightness:
    # L = (Cmax + Cmin) / 2
    lightness = (np.max(pix[:,:,0:2], axis=2)+np.min(pix[:,:,0:2], axis=2))/2
    lightness_max = lightness.max()
    lightness_min = lightness.min()
    lightness_std = lightness.std()
    lightness_avg = lightness.mean()

    # Dullness and whiteness
    dullness = self._perform_color_analysis(img, 'black')
    whiteness = self._perform_color_analysis(img, 'white')

    # Avg pixel width
    avg_pixel_width = self._average_pixel_width(img)

    # Dominant color
    # dominant_r, dominant_g, dominant_b = self._get_dominant_color(img)

    # Average color
    avg_r, avg_g, avg_b = self._get_average_color(img)

    # Blurrness
    # blurrness = self._get_blurrness_score(img)

    w_h_ratio = width / (height + 1),
    yield item_id, image_id, [width, height, width * height, w_h_ratio, avg_pixel_width,
                    luminance_min, luminance_max, luminance_avg, luminance_std,
                    saturation_min, saturation_max, saturation_avg, saturation_std,
                    colorfulness,
                    lightness_min, lightness_max, lightness_avg, lightness_std,
                    dullness, whiteness,
                    avg_r, avg_g, avg_b]

# VGG16 example
# def process_image(img, model, input_size, top_n=5):
#   assert(top_n <= 1000)
#   # convert PIL Image object to nd-array.
#   resized_img = img.resize(input_size, Image.ANTIALIAS)
#   img_array = np.array(resized_img) # the dimension is w * h * 3 (channels)
#   # Keras model takes batch input, thus need to add a dimension even though we
#   # process image one-by-one
#   w, h = input_size
#   img_array = img_array.reshape((1, w, h, 3))
#   # Classification probability on 1000 ImageNet classes.
#   predicts = model.predict(img_array, batch_size=1)[0]
#   # Returns the top n score as features.
#   return list(predicts[np.argsort(predicts)[top_n:]])

def format_result(row):
    item_id, image_id, features = row
    features_str = ",".join(['%.5f' %f for f in features])
    return '%s,%s,%s' % (item_id, image_id, features_str)

def run(argv=None):
  """Main entry point; defines and runs the wordcount pipeline."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--input',
                      dest='input',
                      required=True,
                      help='Input file to process.')
  parser.add_argument('--output',
                      dest='output',
                      required=True,
                      help='Output file to write results to.')
  known_args, pipeline_args = parser.parse_known_args(argv)

  # We use the save_main_session option because one or more DoFn's in this
  # workflow rely on global context (e.g., a module imported at module level).
  pipeline_options = PipelineOptions(pipeline_args)
  pipeline_options.view_as(SetupOptions).save_main_session = True
  p = beam.Pipeline(options=pipeline_options)
  # # Create a pre-trained VGG16 model to use.
  # vgg16 = VGG16(weights='imagenet')

  # Read the text file[pattern] into a PCollection.
  _ = (p | 'read' >> ReadFromText(known_args.input)
            | 'parse input' >> beam.Map(lambda line: csv.reader([line]).next())
            | 'process image' >> beam.ParDo(ProcessImageDoFn())
            | 'format to csv' >> beam.Map(format_result)
            | 'write to file' >> WriteToText(known_args.output)
      )

  result = p.run()
  # result.wait_until_finish()

if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  run()
