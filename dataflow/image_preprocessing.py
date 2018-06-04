"""A image preprocessing workflow."""

from __future__ import absolute_import

import argparse
import logging
import zipfile
import csv
import io

from PIL import Image

import apache_beam as beam
from apache_beam.io import ReadFromText, WriteToText
from apache_beam.metrics import Metrics
# from apache_beam.metrics.metric import MetricsFilter
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions

# import re

# import six


class ProcessImageDoFn(beam.DoFn):
  """Parse each line of input text into words."""

  def __init__(self):
    self.image_counter = Metrics.counter(self.__class__, 'image')
    self.missing_image_counter = Metrics.counter(self.__class__, 'missing_image')

  def process(self, element):
    """Returns an iterator over the words of this element.
    The element is a line of text.  If the line is blank, note that, too.
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

    # def _open_file_read_binary(uri):
    #   try:
    #     return file_io.FileIO(uri, mode='rb')
    #   except errors.InvalidArgumentError:
    #     return file_io.FileIO(uri, mode='r')

    # try:
    #   with _open_file_read_binary(image_uri) as f:
    #     image_bytes = f.read()
    #     img = Image.open(io.BytesIO(image_bytes))
    # # A variety of different calling libraries throw different exceptions here.
    # # They all correspond to an unreadable file so we treat them equivalently.
    # except Exception as e:  # pylint: disable=broad-except
    #   logging.exception('Error processing image %s: %s', image_uri, str(e))
    #   return

    archive = zipfile.ZipFile('data/train_jpg.zip', 'r')
    image_file = 'data/competition_files/train_jpg/%s.jpg' %image_id
    try:
      with archive.open(image_file) as f:
        image_bytes = f.read()
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:  # pylint: disable=broad-except
      logging.exception('Error processing image %s: %s', image_id, str(e))
      return

    width, height = img.size
    yield item_id, width * height


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

  def format_result(row):
      item_id, image_pixels = row
      return '%s, %d' % (item_id, image_pixels)

  # Read the text file[pattern] into a PCollection.
  _ = (p | 'read' >> ReadFromText(known_args.input)
            | 'parse input' >> beam.Map(lambda line: csv.reader([line]).next())
            | 'process image' >> beam.ParDo(ProcessImageDoFn())
            | 'format to csv' >> beam.Map(format_result)
            | 'write to file' >> WriteToText(known_args.output)
      )

  result = p.run()
  result.wait_until_finish()

if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  run()
