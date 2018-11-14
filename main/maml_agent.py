import numpy as np
import tensorflow as tf
from maml.maml import MAML
from maml.data_generator import DataGenerator
from tensorflow.python.platform import flags
from main.datagen import *
from main.dataViz import *
import yaml
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS
def register_flags():
    flags.DEFINE_string('f', '', 'kernel')

    ## Dataset/method options
    flags.DEFINE_string('datasource', 'sinusoid', 'sinusoid or omniglot or miniimagenet')
    flags.DEFINE_integer('num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')
    # oracle means task id is input (only suitable for sinusoid)
    flags.DEFINE_string('baseline', None, 'oracle, or None')

    ## Training options
    flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
    flags.DEFINE_integer('metatrain_iterations', 15000, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
    flags.DEFINE_integer('meta_batch_size', 25, 'number of tasks sampled per meta-update')
    flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
    flags.DEFINE_integer('update_batch_size', 5, 'number of examples used for inner gradient update (K for K-shot learning).')
    flags.DEFINE_float('update_lr', 1e-3, 'step size alpha for inner gradient update.') # 0.1 for omniglot
    flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')

    ## Model options
    flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
    flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
    flags.DEFINE_bool('conv', True, 'whether or not to use a convolutional network, only applicable in some cases')
    flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
    flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')

    ## Logging, saving, and testing options
    flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
    flags.DEFINE_string('logdir', '/tmp/data', 'directory for summaries and checkpoints.')
    flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
    flags.DEFINE_bool('train', True, 'True to train, False to test.')
    flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
    flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
    flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
    flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)') # 0.1 for omniglot


class MAMLAgent:
    def __init__(self, config, sess, graph=None, exp_string="maml_test"):
        FLAGS.norm = 'None'
        FLAGS.update_batch_size = 10
        FLAGS.datasource = 'sinusoid' # just to make sure maml sets up arch right   
        self.config = config
        self.sess = sess
        self.graph = graph if graph is not None else tf.get_default_graph()
        self.logdir = "summaries"
        self.exp_string = exp_string
        self.test_num_updates = 5

        
    def construct_model(self, load_model=True):
        with self.graph.as_default():
            self.model = MAML(self.config['x_dim'], self.config['y_dim'], self.test_num_updates)
            self.model.construct_model(input_tensors=None, prefix='metatrain_')
            self.model.summ_op = tf.summary.merge_all()

            self.saver = self.loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)

            tf.global_variables_initializer().run(session=self.sess)

            self.resume_itr = 0
            if load_model:
                model_file = tf.train.latest_checkpoint(self.logdir + '/' + self.exp_string)
                if FLAGS.test_iter > 0:
                    model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
                if model_file:
                    ind1 = model_file.index('model')
                    self.resume_itr = int(model_file[ind1+5:])
                    print("Restoring model weights from " + model_file)
                    self.saver.restore(self.sess, model_file)

    
    def train(self, dataset, metatrain_iterations):
        SUMMARY_INTERVAL = 100
        SAVE_INTERVAL = 1000
        PRINT_INTERVAL = 1000
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5
        
        pretrain_iterations = 10000
        update_batch_size = FLAGS.update_batch_size #config['data_horizon']
        meta_batch_size = FLAGS.meta_batch_size #config['meta_batch_size']

        train_writer = tf.summary.FileWriter(self.logdir + '/' + self.exp_string, self.graph)
        print('Done initializing, starting training.')
        prelosses, postlosses = [], []
        for itr in range(self.resume_itr, pretrain_iterations+metatrain_iterations):
            feed_dict = {}
            X,Y = dataset.sample(meta_batch_size, update_batch_size*2)

            inputa = X[:, :update_batch_size, :]
            labela = Y[:, :update_batch_size, :]
            inputb = X[:, update_batch_size:, :]
            labelb = Y[:, update_batch_size:, :]

            feed_dict = {self.model.inputa: inputa, self.model.inputb: inputb,  self.model.labela: labela, self.model.labelb: labelb}
            if itr < pretrain_iterations:
                input_tensors = [self.model.pretrain_op]
            else:
                input_tensors = [self.model.metatrain_op]

            if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
                input_tensors.extend([self.model.summ_op, self.model.total_loss1, self.model.total_losses2[FLAGS.num_updates-1]])

            result = self.sess.run(input_tensors, feed_dict)

            if itr % SUMMARY_INTERVAL == 0:
                prelosses.append(result[-2])
                train_writer.add_summary(result[1], itr)
                postlosses.append(result[-1])

            if (itr!=0) and itr % PRINT_INTERVAL == 0:
                if itr < pretrain_iterations:
                    print_str = 'Pretrain Iteration ' + str(itr)
                else:
                    print_str = 'Iteration ' + str(itr - pretrain_iterations)
                print_str += ': ' + str(np.mean(prelosses)) + ', ' + str(np.mean(postlosses))
                print(print_str)
                prelosses, postlosses = [], []

            if (itr!=0) and itr % SAVE_INTERVAL == 0:
                self.saver.save(self.sess, self.logdir + '/' + self.exp_string + '/model' + str(itr))

        self.saver.save(self.sess, self.logdir + '/' + self.exp_string +  '/model' + str(itr))
    
    def test(self, ux, uy, x, num_updates=1):
        feed_dict = {
            self.model.inputa: ux,
            self.model.labela: uy,
            self.model.inputb: x,
        }
        
        y, = self.sess.run([self.model.outputbs], feed_dict=feed_dict)
        
        return y[num_updates-1], None
    
class MAMLDynamics(MAMLAgent):
    def __init__(self, config, sess, graph=None, exp_string="maml_test"):
        super(MAMLDynamics, self).__init__(config, sess, graph, exp_string)
        self.ux = []
        self.uy = []
        
    def sample_rollout(self, x0, actions):
        T, a_dim = actions.shape
        mult_sample = False
        if x0.ndim == 1:
            N_samples = 1
            x_dim = x0.shape[0]
            
            x0 = np.expand_dims(x0, axis=1)
        elif x0.ndim == 2:
            mult_sample = True
            N_samples = x0.shape[0]
            x_dim = x0.shape[1]
            
        actions = np.tile(np.expand_dims(actions,axis=0), (N_samples, 1, 1))
            
        x_pred = np.zeros( (N_samples, T+1, x_dim) )
        x_pred[:,0,:] = x0
        
        if len(self.ux) > 0:
            UX = np.concatenate(self.ux, axis=1)
            UY = np.concatenate(self.uy, axis=1)
        else:
            UX = np.zeros([1,0,self.config['x_dim']])
            UY = np.zeros([1,0,self.config['y_dim']])
        for t in range(0, T):
            x_inp = np.concatenate( (x_pred[0:1,t:t+1,:], actions[0:1,t:t+1,:]), axis=2 )
            y, s = self.test(UX, UY, x_inp, num_updates=5)
            x_pred[:,t+1,:] =  y + x_pred[:,t,:]
        
        if mult_sample:
            return x_pred[:,1:,:]
        else:
            return x_pred[0,1:,:]
        
    def reset_to_prior(self):
        self.ux = []
        self.uy = []
        
    def incorporate_transition(self, x, u, xp):
        x_inp = np.reshape( np.concatenate( (x,u), axis=0 ), (1,1,-1) )
        y = np.reshape(xp - x, (1,1,-1))
        
        self.ux.append(x_inp)
        self.uy.append(y)