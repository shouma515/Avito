"""A image preprocessing workflow."""

from __future__ import absolute_import

import argparse
# import zipfile
import csv
import io
import logging
import numpy as np

import apache_beam as beam
from apache_beam.io import ReadFromText, WriteToText
from apache_beam.metrics import Metrics
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions
from PIL import Image, ImageStat
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

    yield item_id, image_id, [width, height, width * height,
                    luminance_min, luminance_max, luminance_avg, luminance_std,
                    saturation_min, saturation_max, saturation_avg, saturation_std,
                    colorfulness,
                    lightness_min, lightness_max, lightness_avg, lightness_std]

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
