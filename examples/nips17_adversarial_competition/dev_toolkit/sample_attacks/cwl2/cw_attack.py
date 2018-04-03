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
from cleverhans.attacks import FastFeatureAdversaries
from cleverhans.attacks import MomentumIterativeMethod
from cleverhans.attacks import MadryEtAl
from cleverhans.attacks import LBFGS
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception

slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    # 'checkpoint_path', '/home/arie/crest/cleverhans/examples/nips17_adversarial_competition/sample_attacks/inception_v3.ckpt', 'Path to checkpoint for inception network.')
    # 'checkpoint_path', '/home/arie/Desktop/cleverhans/examples/nips17_adversarial_competition/sample_attacks/inception_v3.ckpt', 'Path to checkpoint for inception network.')
    'checkpoint_path', '../checkpoint/inception/inception_v3.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', 'input', 'Input directory with images')
    # 'input_dir', '/mnt/data/imagenet/adversarial/val/squeezenet_fgsm/', 'Input directory with images.')
    # 'input_dir', 'input/ILSVRC2012_img_train_299/*/', 'Input directory with images.')

tf.flags.DEFINE_string(
    # 'output_dir', '/mnt/data/imagenet/Training/BIM/', 'Output directory with images.')
    'output_dir', 'ENM', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 2, 'How many images process at one time.')

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
      image = np.array(Image.open(f).convert('RGB')).astype(np.float) / 255.0
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
    # Strip off the extra reshape op at the output
    probs = output.op.inputs[0]
    return probs


def main(_):
  # Images for inception classifier are normalized to be in [-1, 1] interval,
  # eps is a difference between pixels so it should be in [0, 2] interval.
  # Renormalizing epsilon from [0, 255] to [0, 2].
  print('this line #1')
  eps = 2.0 * FLAGS.max_epsilon / 255.0
  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
  num_classes = 1001

  tf.logging.set_verbosity(tf.logging.INFO)
  print('this line #2')
  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)

    model = InceptionModel(num_classes)
    print('this line #3')

    # # FGSM 
    # fgsm = FastGradientMethod(model)
    # x_adv = fgsm.generate(x_input, eps=eps, clip_min=-1., clip_max=1.)

    # fgsm = FastGradientMethod(model, sess=sess)
    # fgsm_params = {'eps': 0.3,'clip_min': 0.,'clip_max': 1.}
    # adv_x = fgsm.generate(x, **fgsm_params)

    # (train_start=0, train_end=60000, test_start=0, test_end=10000, viz_enabled=True, nb_epochs=6, batch_size=128, nb_classes=10, source_samples=10, learning_rate=0.001, attack_iterations=100, model_path=os.path.join("models", "mnist"), targeted=True)

    # source_samples=10
    # viz_enabled=True
    # targeted=True
    # X_test=x_input
    # if viz_enabled:
    #     assert source_samples == nb_classes
    #     idxs = [np.where(np.argmax(Y_test, axis=1) == i)[0][0]
    #             for i in range(nb_classes)]
    # if targeted:
    #     if viz_enabled:
    #         # Initialize our array for grid visualization
    #         grid_shape = (nb_classes, nb_classes, img_rows, img_cols, channels)
    #         grid_viz_data = np.zeros(grid_shape, dtype='f')

    #         adv_inputs = np.array(
    #             [[instance] * nb_classes for instance in X_test[idxs]],
    #             dtype=np.float32)
    #     else:
    #         adv_inputs = np.array(
    #             [[instance] * nb_classes for
    #              instance in X_test[:source_samples]], dtype=np.float32)

    #     one_hot = np.zeros((nb_classes, nb_classes))
    #     one_hot[np.arange(nb_classes), np.arange(nb_classes)] = 1

    #     adv_inputs = adv_inputs.reshape(
    #         (source_samples * nb_classes, img_rows, img_cols, 1))
    #     adv_ys = np.array([one_hot] * source_samples,
    #                       dtype=np.float32).reshape((source_samples *
    #                                                  nb_classes, nb_classes))
    #     yname = "y_target"
    # else:
    #     if viz_enabled:
    #         # Initialize our array for grid visualization
    #         grid_shape = (nb_classes, 2, img_rows, img_cols, channels)
    #         grid_viz_data = np.zeros(grid_shape, dtype='f')

    #         adv_inputs = X_test[idxs]
    #     else:
    #         adv_inputs = X_test[:source_samples]

    #     adv_ys = None
    #     yname = "y"
    # # adv_ys = None
    # # yname = "y"
    # sess = tf.Session()
    # cw = CarliniWagnerL2(model, back='tf', sess=sess)
    # cw_params = {'binary_search_steps': 1, yname: adv_ys, 'max_iterations': 1000, 'learning_rate': 5e-3, 'batch_size': 1, 'initial_const': 1e-2}
    # # cw_params = {'binary_search_steps': 1, 'y': None, 'max_iterations': 1000, 'learning_rate': 5e-3, 'batch_size': 1, 'initial_const': 1e-2}
    # # cw_params = {'binary_search_steps': 1, 'y': None, 'max_iterations': 1000, 'learning_rate': 5e-3, 'batch_size': 1, 'initial_const': 1e-2}
    # # batch_size=1, confidence=0, targeted=True, learning_rate=5e-3, binary_search_steps=5, max_iterations=1000, abort_early=True, initial_const=1e-2, clip_min=0, clip_max=1, num_labels=3, shape=x_input.get_shape().as_list()[1:])
    # # x_adv = cw.generate_np(x_input,**cw_params)
    # x_adv = cw.generate(x_input,**cw_params)




    # # BIM 
    # bim = BasicIterativeMethod(model)
    # x_adv = bim.generate(x_input, eps=eps, clip_min=-1., clip_max=1.)

    # # VAT 
    # vat = VirtualAdversarialMethod(model)
    # x_adv = vat.generate(x_input, eps=eps, clip_min=-1., clip_max=1.)

    # # PGD 
    # madry = MadryEtAl(model)
    # elastic_params = {'clip_min':-1., 'clip_max':1.}
    # x_adv = madry.generate(x_input, clip_min=-1., clip_max=1.)#, **elastic_params)

    # FFA
    # ffa = FastFeatureAdversaries(model)
    # x_adv = ffa.generate(x_input, eps=eps, clip_min=-1., clip_max=1.)

    
    # # JSMA
    # sess = tf.Session()
    jsma = SaliencyMapMethod(model)
    # # target = np.zeros((1,1000),dtype=np.float32)
    # # target[0,50] = 1                    #here, we suppose that the target label is 50
    # # jsma_params = {'theta': 1., 'gamma': 0.1, 'clip_min': 0., 'clip_max': 1., 'y_target': target}
    jsma_params = {'theta': 1., 'gamma': 0.1, 'clip_min': -1., 'clip_max': 1.}
    x_adv = jsma.generate(x_input, **jsma_params)
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())

    # jsma = SaliencyMapMethod(model, back='tf', sess=sess)
    # jsma_params = {'theta': 1., 'gamma': 0.1, 'clip_min': 0., 'clip_max': 1.}
    # x_adv = jsma.generate(x_input,**jsma_params)

# self.structural_kwargs = ['over_shoot', 'max_iter', 'clip_max',
#                                   'clip_min', 'nb_candidate']

#                                   self.feedable_kwargs = {'eps': np.float32,
#                                 'eps_iter': np.float32,
#                                 'y': np.float32,
#                                 'y_target': np.float32,
#                                 'clip_min': np.float32,
#                                 'clip_max': np.float32}
#         self.structural_kwargs = ['ord', 'nb_iter']

    # # DeepFool
    # deepfool = DeepFool(model)
    # # # deepfool_params =  {'over_shoot', 'max_iter':1000, 'clip_max':1., 'clip_min':0., 'nb_candidate}
    # deepfool_params =  {'max_iter':10, 'clip_max':1., 'clip_min':-1.}
    # x_adv = deepfool.generate(x_input,**deepfool_params)

    # LBFGS
    # lbfgs = LBFGS(model)
    # lbfgs_params = {'clip_max':1., 'clip_min':-1.}
    # x_adv = lbfgs.generate(x_input, y_target=None, **lbfgs_params)

    # ENM
    # enm = ElasticNetMethod(model)
    # enm_params = {'clip_max':1., 'clip_min':-1.}
    # x_adv = enm.generate(x_input, **enm_params)

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


    # (self, model, back='tf', sess=None)
    
    print('this line #4')

     # Run computation
    saver = tf.train.Saver(slim.get_model_variables())
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=tf.train.Scaffold(saver=saver),
        checkpoint_filename_with_path=FLAGS.checkpoint_path,
        master=FLAGS.master)

    print('this line #5')

    with tf.train.MonitoredSession(session_creator=session_creator) as sess:
    # with tf.Session() as sess:

      # cwl2 = CarliniWagnerL2(model, back='tf', sess=sess)
      # cwl2_params = {'clip_min':-1.0, 'clip_max':1.0}
      # x_adv = cwl2.generate(x_input,**cwl2_params)


      for filenames, images in load_images(FLAGS.input_dir, batch_shape):
        print('this line #6')
        adv_images = sess.run(x_adv, feed_dict={x_input: images})
        print('this line #7')
        save_images(adv_images, filenames, FLAGS.output_dir)
        print('this line #8')

    # # sess = tf.Session()

    # # Run computation
    # saver = tf.train.Saver(slim.get_model_variables())
    
    # ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
    # if ckpt and ckpt.model_checkpoint_path:
    #   saver.restore(sess, ckpt.model_checkpoint_path)

    # sess.run(tf.global_variables_initializer())

    # session_creator = tf.train.ChiefSessionCreator(
    #     scaffold=tf.train.Scaffold(saver=saver),
    #     checkpoint_filename_with_path=FLAGS.checkpoint_path,
    #     master=FLAGS.master)
    # print('this line #5')

    # # with tf.train.MonitoredSession(session_creator=session_creator) as sess:
    # #   for filenames, images in load_images(FLAGS.input_dir, batch_shape):
    # #     print('this line #6')
    # #     adv_images = sess.run(x_adv, feed_dict={x_input: images})
    # #     save_images(adv_images, filenames, FLAGS.output_dir)
    # #     print('this line #7')
    
    # # with tf.train.MonitoredSession() as sess:
    # # with tf.Session(FLAGS.master) as sess:
    # with tf.train.MonitoredSession(session_creator=session_creator) as sess:
    #   for filenames, images in load_images(FLAGS.input_dir, batch_shape):
    #     print('this line #6')
    #     adv_images = sess.run(x_adv, feed_dict={x_input: images})
    #     print('this line #7')
    #     save_images(adv_images, filenames, FLAGS.output_dir)
    #     print('this line #8')

if __name__ == '__main__':
  tf.app.run()
