"""A image preprocessing workflow."""

from __future__ import absolute_import

import argparse
import csv
import logging

import apache_beam as beam
from apache_beam.io import ReadFromText, WriteToText
from apache_beam.metrics import Metrics
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions

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

class Extract(beam.DoFn):
  """Parse each line of input text into words."""

  def __init__(self, key_cols, val_col):
    # Count the row with missing values.
    self.null_row_count = Metrics.counter(self.__class__, 'null_row')
    self.key_cols = key_cols
    self.val_col = val_col

  def process(self, element):
    """Extract values from the corresponding column of the row
    Args:
      element: the element being processed
    Returns:
      The processed element.
    """
    print element
    keys = []
    for index in self.key_cols:
        keys.append(element[index])
    # If any key missing, we do not include the row.
    if any([not key for key in keys]):
      self.null_row_count.inc()
      return
    
    # TODO(hzn): figure out support multi value
    value = element[self.val_col]
    # If any value missing, we do not include the row.
    if not value:
      self.null_row_count.inc()
      return
    value = float(value)
    
    yield tuple(keys), value


def format_result(row):
    keys, value = row
    keys_str = ",".join(keys)
    return '%s,%s' % (keys_str, '%.5f' %value)


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

    # Schema of the train data: title, description, image, image_top_1 excluded.
    schema = ['item_id', 'user_id', 'region', 'city', 'parent_category_name',
            'category_name', 'param_1', 'param_2', 'param_3', 'price',
            'item_seq_number', 'activation_date', 'user_type']

    keys = ['region', 'category_name']
    value = 'price'

    key_cols = []
    for key in keys:
        key_cols.append(schema.index(key))
    
    val_col = schema.index(value)
    
    # Read the text file[pattern] into a PCollection.
    _ = (p | 'read' >> ReadFromText(known_args.input)
    # TODO: need to correctly parse csv, cannot use split by ',' here, not
    # complete enough
                | 'parse input' >> beam.Map(lambda line: line.split(','))
                | 'Extrace keys and value' >> beam.ParDo(Extract(key_cols, val_col))
                | 'Combine' >> beam.CombinePerKey(beam.combiners.MeanCombineFn())
                | 'format to csv' >> beam.Map(format_result)
                | 'write to file' >> WriteToText(known_args.output)
        )

    result = p.run()
    # result.wait_until_finish()

if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  run()
