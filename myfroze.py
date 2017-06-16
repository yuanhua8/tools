""" 
Example code to convert the tp3 model's variables to constants
(1) pass correct path
(2) pass output node names.
"""
import numpy as np

import tensorflow as tf
from tensorflow.python.platform import gfile
# from tensorflow.python.training.training_util import write_graph
# from tensorflow.python.tools.freeze_graph import freeze_graph
from tensorflow.python.framework.graph_util import convert_variables_to_constants


def convert_graph(path):
    print(path)
    saver = tf.train.import_meta_graph(path + ".meta", import_scope=None)
    with tf.Session() as sess:
        saver.restore(sess, path)
        freeze_graph(sess)

def freeze_graph(sess):
    # convert_variables_to_constants(sess, input_graph_def, output_node_names, variable_names_whitelist=None)
    graph_def = sess.graph.as_graph_def()
    print(graph_def)
       
    frozen_graph_def = convert_variables_to_constants(sess, graph_def, ["polymath1_model_Polymath1_Model_1_0_1/Softmax_1", "polymath1_model_Polymath1_Model_1_0_1/Softmax"])

    with tf.gfile.GFile("./" + "frozen.pb", "wb") as f:
        f.write(frozen_graph_def.SerializeToString())

    return frozen_graph_def


# for example we pass in path1 as g:\export_polymath\00000001\export
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 2:
        _, path1 = sys.argv
   
    convert_graph(path1)
