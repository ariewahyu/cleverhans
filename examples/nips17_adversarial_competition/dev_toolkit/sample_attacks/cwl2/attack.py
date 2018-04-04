"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import SaliencyMapMethod
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.attacks import DeepFool
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.attacks import VirtualAdversarialMethod
from cleverhans.attacks import ElasticNetMethod
from cleverhans.attacks import MadryEtAl
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib.slim.nets import alexnet
import mobilenet_v1 as mobilenet_v1
# from tensorflow.contrib.slim.nets import mobilenet_v1

slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    # 'checkpoint_path', '../checkpoint/inception/inception_v3.ckpt', 'Path to checkpoint for inception network.')
    # 'checkpoint_path', '../checkpoint/mobilenet/mobilenet_v1_1.0_224.ckpt', 'Path to checkpoint for inception network.')
    # 'checkpoint_path', '../checkpoint/squeezenet/squeezenet_v1_1.npy', 'Path to checkpoint for inception network.')
    # 'checkpoint_path', '../checkpoint/deepcompression/deepcompression.npy', 'Path to checkpoint for inception network.')
    # 'checkpoint_path', '../checkpoint/inq/inq.npy', 'Path to checkpoint for inception network.')
    # 'checkpoint_path', '../checkpoint/proposed/proposed.npy', 'Path to checkpoint for inception network.')
    'checkpoint_path', '../checkpoint/alexnet/bvlc_alexnet.npy', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    # 'input_dir', '/mnt/data/imagenet/ilsvrc12_val_299', 'Input directory with images')
    'input_dir', 'ilsvrc12_val_299', 'Input directory with images.')
    # 'input_dir', 'input', 'Input directory with images.')
    # 'input_dir', '/mnt/data/imagenet/ILSVRC2012_img_train_299/*/', 'Input directory with images.')

tf.flags.DEFINE_string(
    # 'output_dir', 'mobilenet_bim_val', 'Output directory with images.')
    # 'output_dir', 'output', 'Output directory with images.')
    'output_dir', '/mnt/data/imagenet/adversarial/BIM_Val_Results/alexnet_bim_val', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'image_width', 227, 'Width of each input images.') #default Inception v3: 299, MobileNets: 224, AlexNet: 227, AlexNet v2: 224, SqueezeNet: 227

tf.flags.DEFINE_integer(
    'image_height', 227, 'Height of each input images.')  #default Inception v3: 299, MobileNets: 224, AlexNet: 227, AlexNet v2: 224, SqueezeNet: 227

tf.flags.DEFINE_integer(
    'batch_size', 100, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS


def load_images(input_dir, batch_shape):
  """Read png images from input directory in batches.

  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  print('input #1')
  images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  print('input #2')
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.JPEG')):
    with tf.gfile.Open(filepath) as f:
      image = np.array(Image.open(f).resize([FLAGS.image_height, FLAGS.image_width]).convert('RGB')).astype(np.float) / 255.0
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    print('input #3')
    images[idx, :, :, :] = image * 2.0 - 1.0
    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      yield filenames, images
      filenames = []
      images = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    yield filenames, images


def save_images(images, filenames, output_dir):
  """Saves images to the output directory.

  Args:
    images: array with minibatch of images
    filenames: list of filenames without path
      If number of file names in this list less than number of images in
      the minibatch then only first len(filenames) images will be saved.
    output_dir: directory where to save images
  """
  for i, filename in enumerate(filenames):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # so rescale them back to [0, 1].
    # with tf.gfile.Open(os.path.join(output_dir, filename[:9], filename), 'w') as f:
    with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
      img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
      Image.fromarray(img).save(f, format='JPEG')


class MobileNetModel(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False

  def __call__(self, x_input):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
      _, end_points = mobilenet_v1.mobilenet_v1(
          x_input, num_classes=self.num_classes, is_training=False,
          reuse=reuse)
    self.built = True
    output = end_points['Predictions']
    # Strip off the extra reshape op at the output
    probs = output.op.inputs[0]
    return probs


class InceptionModel(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False

  def __call__(self, x_input):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    with slim.arg_scope(inception.inception_v3_arg_scope()):
      _, end_points = inception.inception_v3(
          x_input, num_classes=self.num_classes, is_training=False,
          reuse=reuse)
    self.built = True
    output = end_points['Predictions']
    print('end_points', end_points)
    print('output', output)
    # Strip off the extra reshape op at the output
    probs = output.op.inputs[0]
    return probs


def load_kaffe_model(name,  x_input, reuse=False):
    '''Creates and returns an instance of the model given its class name.
    The created model has a single placeholder node for feeding images.
    '''
    from caffe_tensorflow import models

    # Find the model class from its name
    all_models = models.get_models()
    lut = {model.__name__: model for model in all_models}
    if name not in lut:
        print('Invalid model index. Options are:')
        # Display a list of valid model names
        for model in all_models:
            print('\t* {}'.format(model.__name__))
        return None
    NetClass = lut[name]

    # Create a placeholder for the input image
    spec = models.get_data_spec(model_class=NetClass)
    data_node = tf.placeholder(tf.float32,
                               shape=(None, spec.crop_size, spec.crop_size, spec.channels))

    # Construct and return the model
    return NetClass({'data': x_input}, reuse=reuse, trainable=False)

class KaffeModel(object):

  def __init__(self, num_classes, model_name):
    self.num_classes = num_classes
    self.model_name = model_name
    self.built = False
    self.net = None

  def __call__(self, x_input):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    net = load_kaffe_model(self.model_name, x_input, reuse=reuse)
    self.built = True
    self.net = net
    #output = end_points['alexnet_v2/fc8']
    # Strip off the extra reshape op at the output
    output = self.net.get_output()
    probs = output.op.inputs[0]
    return probs

  def load_model(self, model_path, session):
    self.net.load(data_path=model_path, session=session)



def main(_):
  # Images for inception classifier are normalized to be in [-1, 1] interval,
  # eps is a difference between pixels so it should be in [0, 2] interval.
  # Renormalizing epsilon from [0, 255] to [0, 2].
  print('this line #1')
  eps = 2.0 * FLAGS.max_epsilon / 255.0
  slim_model = False        # For AlexNet and SqueezeNet: False, For Inception and MobileNet: True
  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
  num_classes = 1001

  tf.logging.set_verbosity(tf.logging.INFO)
  print('this line #2')
  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)

    # model = InceptionModel(num_classes)
    # model = MobileNetModel(num_classes)
    model = KaffeModel(num_classes, 'AlexNet') # Use AlexNet model for DeepCompression, INQ, and Proposed
    # model = KaffeModel(num_classes, 'Squeezenet')
    print('this line #3')

    # # FGSM 
    # fgsm = FastGradientMethod(model)
    # x_adv = fgsm.generate(x_input, eps=eps, clip_min=-1., clip_max=1.)

    # # BIM 
    bim = BasicIterativeMethod(model)
    x_adv = bim.generate(x_input, eps=eps, clip_min=-1., clip_max=1.)

    # # VAT
    # vat = VirtualAdversarialMethod(model)
    # x_adv = vat.generate(x_input, eps=eps, clip_min=-1., clip_max=1.)

    # # PGD 
    # pgd = MadryEtAl(model)
    # pgd_params = {'clip_min':-1., 'clip_max':1.}
    # x_adv = pgd.generate(x_input, clip_min=-1., clip_max=1.)#, **pgd_params)
    
    # # JSMA
    # sess = tf.Session()
    # jsma = SaliencyMapMethod(model)
    # # target = np.zeros((1,1000),dtype=np.float32)
    # # target[0,50] = 1                    #here, we suppose that the target label is 50
    # # jsma_params = {'theta': 1., 'gamma': 0.1, 'clip_min': 0., 'clip_max': 1., 'y_target': target}
    # jsma_params = {'theta': 1., 'gamma': 0.1, 'clip_min': 0., 'clip_max': 1.}
    # x_adv = jsma.generate(x_input,**jsma_params)
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())

    # jsma = SaliencyMapMethod(model, back='tf', sess=sess)
    # jsma_params = {'theta': 1., 'gamma': 0.1, 'clip_min': 0., 'clip_max': 1.}
    # x_adv = jsma.generate(x_input,**jsma_params)


    # # DeepFool
    # deepfool = DeepFool(model, back='tf', sess=sess)
    # # # deepfool_params =  {'over_shoot', 'max_iter':1000, 'clip_max':1., 'clip_min':0., 'nb_candidate}
    # deepfool_params =  {'max_iter':1000, 'clip_max':1., 'clip_min':0.}
    # x_adv = deepfool.generate(x_input,**deepfool_params)

    # # sess = tf.Session()
    # # with tf.Session() as sess:
    # # # # # CarliniWagner L2
    # # # # sess = tf.train.MonitoredSession()
    # # # sess = tf.Session()
    # cwl2 = CarliniWagnerL2(model, back='tf', sess=sess)
    # # cwl2 = CarliniWagnerL2(model, back='tf')
    # # # cwl2_params = {'batch_size':9, 'confidence':0, 'max_iterations':1000, 'clip_min':0., 'clip_max':1.}
    # cwl2_params = {'clip_min':-1.0, 'clip_max':1.0}
    # # cwl2_params = {'batch_size':9, 'confidence':0,'learning_rate':1e-2,'binary_search_steps':9, 'max_iterations':1000,'abort_early':True, 'initial_const': 1e-3,'clip_min': 0.0, 'clip_max': 1.0}
    # x_adv = cwl2.generate(x_input,**cwl2_params)
    # with tf.Session() as sess:
    #   cwl2 = CarliniWagnerL2(sess, model, batch_size=1, confidence=0, targeted=True, learning_rate=5e-3, binary_search_steps=5, max_iterations=1000, abort_early=True, initial_const=1e-2, clip_min=0, clip_max=1, num_labels=3, shape=x_input.get_shape().as_list()[1:])
    #   # x_adv = cwl2.
    #   def cw_wrap(x_val, y_val):
    #       return np.array(cwl2.attack(x_val, y_val), dtype=np.float32)
    #   x_adv = tf.py_func(cw_wrap, [x, labels], tf.float32)
    # (self, sess, model, batch_size, confidence,
    #              targeted, learning_rate,
    #              binary_search_steps, max_iterations,
    #              abort_early, initial_const,
    #              clip_min, clip_max, num_labels, shape)

        # attack = CWL2(self.sess, self.model, self.batch_size,
        #               self.confidence, 'y_target' in kwargs,
        #               self.learning_rate, self.binary_search_steps,
        #               self.max_iterations, self.abort_early,
        #               self.initial_const, self.clip_min, self.clip_max,
        #               nb_classes, x_input.get_shape().as_list()[1:])
    # sess = tf.Session()
    # cw = CarliniWagnerL2(model, back='tf', sess=sess)
    # cw_params = {'binary_search_steps': 1, 'y': None, 'max_iterations': 1000, 'learning_rate': 5e-3, 'batch_size': 1, 'initial_const': 1e-2}
    # # x_adv = cw.generate_np(x_input,**cw_params)
    # x_adv = cw.generate(x_input,**cw_params)

    # (self, model, back='tf', sess=None)
    
    print('this line #4')

    # Run computation
    if slim_model:
        saver = tf.train.Saver(slim.get_model_variables())
        session_creator = tf.train.ChiefSessionCreator(
            scaffold=tf.train.Scaffold(saver=saver),
            checkpoint_filename_with_path=FLAGS.checkpoint_path,
            master=FLAGS.master)

        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
          for filenames, images in load_images(FLAGS.input_dir, batch_shape):
            adv_images = sess.run(x_adv, feed_dict={x_input: images})
            save_images(adv_images, filenames, FLAGS.output_dir)

    else:
        with tf.Session() as sess:
          model.load_model(model_path=FLAGS.checkpoint_path, session=sess)
          for filenames, images in load_images(FLAGS.input_dir, batch_shape):
            adv_images = sess.run(x_adv, feed_dict={x_input: images})
            save_images(adv_images, filenames, FLAGS.output_dir)

if __name__ == '__main__':
  tf.app.run()
