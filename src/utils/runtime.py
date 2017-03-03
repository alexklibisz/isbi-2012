from keras.backend.tensorflow_backend import set_session
import os, psutil
import tensorflow as tf
import sys

def funcname():
    return sys._getframe(1).f_code.co_name

def memory_usage():
	proc = psutil.Process(os.getpid())
	return float(proc.memory_info().rss) / (10**9)

def gpu_selection(visible_devices="1", memory_fraction=0.95):
	os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = memory_fraction
	set_session(tf.Session(config=config))