import os, cv2, sys
from utils import *
from ops import *
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from PIL import ImageTk, Image
import time

# to change anime/celebA::
#change output size
# changee dataset


## anime:64*64
## final4:128*128

class Cgan(object):

    def __init__(self, mode=None):
        self.y_features = {"Bald": 2000, "Blond_Hair": 3000, "Eyeglasses": 1000, "Wearing_Hat": 1000, "Smiling": 3000
            ,"Male": 5000
                           }

        self.sample_size = 10000
        self.output_size = 128
        self.crop = True
        self.filter = True
        self.channel = 3
        self.learning_rate = 0.0002
        self.batch_size = 64
        self.max_epochs = 500
        self.d_itters = 1
        self.g_itters = 1
        self.save_mode = 2  # 1 = every epoch    2 = every 5 batches
        self.save_model = 5

        self.data_set = CelebA(self.output_size, self.channel, self.sample_size, self.batch_size, self.crop, self.filter,
                             y_features=self.y_features)
        if mode=="anime":
            self.data_set = Anime(output_size=64, channel=3, sample_size=2e4, batch_size=64, crop=True, filter=True,
                     ignore_image_description=True,
                     data_dir=r'W:\Projects\2DO\anime-faces\data')
            self.output_size = 64

        self.z_dim = 100
        self.y_dim = self.data_set.y_dim
        self.version = 'anime1'
        # self.version = 'face_gen_per_batch_filtered_41'
        self.log_dir = '/tmp/tensorflow_cgan/' + self.version
        self.model_dir = 'model/'
        self.sample_dir = 'samples/'
        self.test_dir = 'test/'
        self.sequence_dir = 'image_sequence/'

        self.real_images = tf.placeholder('float',
                                          shape=[self.batch_size, self.output_size, self.output_size, self.channel],
                                          name='real_images')
        self.z = tf.placeholder('float', shape=[self.batch_size, self.z_dim], name='noise_vec')
        self.y = tf.placeholder('float', shape=[self.batch_size, self.y_dim], name='condition_vec')


        self.is_training = tf.placeholder(tf.bool)

        self.loaded_test_weights = None

    def build_model(self):

        self.fake_images, self.rec_prediction = self.generator(self.z, self.y)

        self.gen_sampler = self.sampler(self.z, self.y)

        real_result, real_logits = self.discriminator(self.real_images, self.y)

        fake_result, fake_logits = self.discriminator(self.fake_images, self.y, reuse=True)

        d_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_result), logits=fake_logits))

        d_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_result), logits=real_logits))

        self.d_loss = d_real_loss + d_fake_loss

        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_result), logits=fake_logits))

        self.z_loss = tf.reduce_mean(tf.square(tf.concat([self.z, self.y], 1) - self.rec_prediction),
                                     name='z_prediction_loss')

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'dis' in var.name]
        self.g_vars = [var for var in t_vars if 'gen' in var.name]

        self.saver = tf.train.Saver()

    def train(self, path=None):

        trainer_d = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5).minimize(self.d_loss,
                                                                                                 var_list=self.d_vars)
        trainer_g = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5).minimize(self.g_loss,
                                                                                                 var_list=self.g_vars)
        trainer_z = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5).minimize(self.z_loss,
                                                                                                 var_list=self.g_vars)

        batch_num = self.sample_size // self.batch_size

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            start_epoch = 1
            sample_noise = None
            sample_labels = None

            if os.path.exists(self.model_dir + self.version):
                if path is None:
                    self.saver.restore(sess, self.model_dir + self.version + '/' + self.version + '.ckpt')
                else:
                    self.saver.restore(sess, path[:path.find("ckpt") + 4].replace('//', '/'))
                with open(self.model_dir + self.version + '/epoch.txt', 'r') as ep:
                    start_epoch = int(ep.read()) + 1
                self.data_set.load(self.model_dir + self.version)
                sample_noise = np.load(self.model_dir + self.version + '/sample_noise.npy')
                sample_labels = np.load(self.model_dir + self.version + '/sample_labels.npy')
                print('\n===CHECKPOINT RESTORED===')
            else:
                os.makedirs(self.model_dir + self.version)
                self.data_set.load_data()
                self.data_set.save(self.model_dir + self.version)
            sample_noise = np.random.uniform(-1, 1, size=[self.batch_size, self.z_dim]).astype(np.float32)
            np.save(self.model_dir + self.version + '/sample_noise.npy', sample_noise)
            _, sample_labels = self.data_set.get_next_batch(0)
            np.save(self.model_dir + self.version + '/sample_labels.npy', sample_labels)

            # print(sample_labels)
            print('\n===HYPER PARAMS===')
            print('Version: {}'.format(self.version))
            print('Crop: {}'.format(self.crop))
            print('Filter: {}'.format(self.filter))
            print('Sample Size: {}'.format(self.sample_size))
            print('Max Epochs: {}'.format(self.max_epochs))
            print('Batch Size: {}'.format(self.batch_size))
            print('Batches per Epoch: {}'.format(batch_num))
            print('Starting training...\n')

            for epoch in range(start_epoch, self.max_epochs + 1):

                dLoss_avg = []
                gLoss_avg = []

                epoch_start_time = time.time()

                for batch in range(batch_num):
                    batch_start_time = time.time()
                    train_noise = np.random.uniform(-1, 1, size=[self.batch_size, self.z_dim]).astype(np.float32)
                    train_images, real_labels = self.data_set.get_next_batch(batch)
                    for d in range(self.d_itters):
                        _, dLoss = sess.run([trainer_d, self.d_loss],
                                            feed_dict={self.z: train_noise, self.real_images: train_images,
                                                       self.y: real_labels, self.is_training: True})
                        dLoss_avg.append(dLoss)

                    for g in range(self.g_itters):
                        _, gLoss = sess.run([trainer_g, self.g_loss],
                                            feed_dict={self.z: train_noise, self.y: real_labels,
                                                       self.is_training: True})
                        sess.run([trainer_z],
                                 feed_dict={self.z: train_noise, self.y: real_labels, self.is_training: True})
                        gLoss_avg.append(gLoss)

                    print(
                        '\rEpoch {}/{} - Batch {}/{} - D_loss {:.3f} - G_loss {:.3f}'.format(
                            epoch, self.max_epochs,
                            batch + 1, batch_num,
                            avg(dLoss_avg),
                            avg(gLoss_avg)),
                        # time.time() - batch_start_time, time.time() - epoch_start_time,
                        end='')

                    if self.save_mode == 2 and batch % 200 == 0:
                        if not os.path.exists(self.sample_dir + self.version):
                            os.makedirs(self.sample_dir + self.version)
                        imgtest = sess.run(self.gen_sampler, feed_dict={self.z: sample_noise, self.y: sample_labels,
                                                                        self.is_training: False})
                        imgtest = imgtest * 255.0
                        save_images(imgtest, [8, 8],
                                    self.sample_dir + self.version + '/epoch_' + str(epoch) + '_batch_' + str(
                                        batch) + '.jpg')

                print('')
                if epoch % self.save_model == 0:
                    self.saver.save(sess,
                                    self.model_dir + self.version + '/' + self.version + "_" + str(epoch) + '.ckpt')
                    with open(self.model_dir + self.version + '/epoch.txt', 'w') as ep:
                        ep.write(str(epoch))
                    print('Model Saved | Epoch:[{}] | D_loss:[{:.2f}] | G_loss:[{:.2f}]'.format(epoch,
                                                                                                              avg(
                                                                                                                  dLoss_avg),
                                                                                                              avg(
                                                                                                                  gLoss_avg)
                                                                                                # ,
                                                                                                              # time.time() - epoch_start_time
                                                                                                ))

                if self.save_mode == 1 and epoch % 1 == 0:
                    if not os.path.exists(self.sample_dir + self.version):
                        os.makedirs(self.sample_dir + self.version)
                    imgtest = sess.run(self.gen_sampler,
                                       feed_dict={self.z: sample_noise, self.y: sample_labels, self.is_training: False})
                    imgtest = imgtest * 255.0
                    save_images(imgtest, [8, 8], self.sample_dir + self.version + '/epoch_' + str(epoch) + '.jpg')

                    print('Sample Saved [epoch_{}.jpg]'.format(epoch))

    def test(self, noise=None, desc=None, alt_path=None, is_training=False):
        test_start_time = time.time()
        path = self.model_dir + self.version
        # print("loading weights: ", alt_path)

        if alt_path is not None or os.path.exists(path):

            with tf.Session() as sess:
                if True or self.loaded_test_weights != alt_path:
                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.local_variables_initializer())
                    if alt_path is not None:
                        self.saver.restore(sess, alt_path)
                    else:
                        self.saver.restore(sess, path + '/' + self.version + "" + '.ckpt')
                    self.loaded_test_weights = alt_path

                description = desc.lower()
                sample_z = np.ndarray(shape=(64, 100))
                description_vec = np.ndarray((64,5))
                if noise is None:
                    # print(sample_z[0].shape)
                    # sample_z[0] = np.random.uniform(-1, 1, size=(100))
                    # np.append(sample_z, np.random.uniform(-1, 1, size=(100))).astype(np.float32)
                    sample_z = np.random.uniform(-1, 1, size=[self.batch_size, self.z_dim]).astype(np.float32)
                else:
                    # sample_z[0] = np.random.uniform(-1, 1, size=(100))
                    # np.append(sample_z, np.random.uniform(-1, 1, size=(100))).astype(np.float32)
                    sample_z = np.random.uniform(-1, 1, size=[self.batch_size, self.z_dim]).astype(np.float32)

                    # print(get_truncated_normal(mean=int(noise), sd=0.2, low=-1, upp=1).shape)
                description_vec = self.data_set.text_to_vector(description)

                # description_vec[0] = self.data_set.text_to_vector(description)[0]
                # np.append(description_vec,self.data_set.text_to_vector(description))

                output = sess.run(self.gen_sampler,
                                  feed_dict={self.z: sample_z, self.y: description_vec, self.is_training: is_training})[
                         0:64]

                output = output * 255.0
                if not os.path.exists(self.test_dir + self.version):
                    os.makedirs(self.test_dir + self.version)

                if desc == '':
                    filename = "test"
                else:
                    filename = description.replace(' ', '_')
                save_images(output, [8, 8],
                            self.test_dir + self.version + '/{}.jpg'.format(filename))
                print("-------", self.test_dir + self.version + '/{}.jpg'.format(filename))
                print("generated image: -{}- after {:.2f} seconds".format(filename+".jpg", time.time() - test_start_time))
                return True

        else:
            print('ERROR - [Model {} not found] - Path {}'.format(self.version, path))
        return False

    def discriminator(self, image, y, reuse=False):

        with tf.variable_scope('dis') as scope:
            k = 64

            if reuse == True:
                scope.reuse_variables()

            # Data shape is (128, 128, 3)
            #yb = tf.reshape(y, shape=[self.batch_size, 1, 1, self.y_dim])

            conv1 = conv2d(image, k, name='d_conv1')
            conv1 = batch_norm(conv1, scope='d_conv1_bn', train=self.is_training)
            conv1 = tf.nn.leaky_relu(conv1, name='d_conv1_act')

            conv2 = conv2d(conv1, k * 2, name='d_conv2')
            conv2 = batch_norm(conv2, scope='d_conv2_bn', train=self.is_training)
            conv2 = tf.nn.leaky_relu(conv2, name='d_conv2_act')

            conv3 = conv2d(conv2, k * 4, name='')
            conv3 = batch_norm(conv3, scope='d_conv3_bn', train=self.is_training)
            conv3 = tf.nn.leaky_relu(conv3, name='d_conv3_act')

            conv4 = conv2d(conv3, k * 8, name='d_conv4')
            conv4 = batch_norm(conv4, scope='d_conv4_bn', train=self.is_training)
            conv4 = tf.nn.leaky_relu(conv4, name='d_conv4_act')

            if(self.output_size == 128):
                conv4 = conv2d(conv4, k * 16, name='d_conv')
                conv4 = batch_norm(conv4, scope='d_conv_bn', train=self.is_training)
                conv4 = tf.nn.leaky_relu(conv4, name='d_conv_act')

            flat = tf.reshape(conv4, [self.batch_size, -1])
            flat = tf.concat([flat, y], 1)

            full1 = fully_connected(flat, 1024, 'd_full1')
            full1 = tf.nn.relu(full1, name='d_full1_act')

            full2 = fully_connected(full1, 1, 'd_full2')

            return tf.nn.sigmoid(full2, name='d_full2_act'), full2

    def generator(self, z, y):

        with tf.variable_scope('gen') as scope:
            k = 64
            s_h, s_w = self.output_size, self.output_size
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
            s_h32, s_w32 = conv_out_size_same(s_h16, 2), conv_out_size_same(s_w16, 2)

            #yb = tf.reshape(y, shape=[self.batch_size, 1, 1, self.y_dim])
            z = tf.concat([z, y], 1)

            if self.output_size == 128:
                full1 = fully_connected(z, k * 16 * s_h32 * s_w32, 'g_full1')
            else:
                full1 = fully_connected(z, k * 8 * s_h16 * s_w16, 'g_full1')

            full1 = tf.nn.relu(full1, name='g_full1_act1')
            full1 = batch_norm(full1, scope='g_full1_bn', train=self.is_training)
            full1_act = tf.nn.leaky_relu(full1, name='g_full1_act2')

            if self.output_size == 128:

                conv1 = tf.reshape(full1_act, shape=[self.batch_size, s_h32, s_h32, k * 16], name='g_conv1')

                conv1 = deconv2d(conv1, [self.batch_size, s_h16, s_w16, k * 8], name='g_conv2')
                conv1 = batch_norm(conv1, scope='g_conv2_bn', train=self.is_training)
                conv1 = tf.nn.leaky_relu(conv1, name='g_conv2_act')
            else:
                conv1 = tf.reshape(full1_act, shape=[self.batch_size, s_h16, s_h16, k * 8], name='g_conv1')

            conv2 = deconv2d(conv1, [self.batch_size, s_h8, s_w8, k * 4], name='g_conv')
            conv2 = batch_norm(conv2, scope='g_conv_bn', train=self.is_training)
            conv2 = tf.nn.leaky_relu(conv2, name='g_conv_act')

            conv3 = deconv2d(conv2, [self.batch_size, s_h4, s_w4, k * 2], name='g_conv3')
            conv3 = batch_norm(conv3, scope='g_conv3_bn', train=self.is_training)
            conv3 = tf.nn.leaky_relu(conv3, name='g_conv3_act')

            conv4 = deconv2d(conv3, [self.batch_size, s_h2, s_w2, k], name='g_conv4')
            conv4 = batch_norm(conv4, scope='g_conv4_bn', train=self.is_training)
            conv4 = tf.nn.leaky_relu(conv4, name='g_conv4_act')

            conv5 = deconv2d(conv4, [self.batch_size, s_h, s_w, self.channel], name='g_conv5')

            conv5 = tf.nn.tanh(conv5, name='g_conv5_act')

            # Auto encoder to predict noise
            z_pred = fully_connected(full1, self.z_dim + self.y_dim, 'z_full')
            z_pred = tf.nn.tanh(z_pred, name='z_full_act')

            return conv5, z_pred

    def sampler(self, z, y):

        with tf.variable_scope('gen') as scope:
            scope.reuse_variables()

            k = 64
            s_h, s_w = self.output_size, self.output_size
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
            s_h32, s_w32 = conv_out_size_same(s_h16, 2), conv_out_size_same(s_w16, 2)

            #yb = tf.reshape(y, shape=[self.batch_size, 1, 1, self.y_dim])
            z = tf.concat([z, y], 1)

            if self.output_size == 128:
                full1 = fully_connected(z, k * 16 * s_h32 * s_w32, 'g_full1')
            else:
                full1 = fully_connected(z, k * 8 * s_h16 * s_w16, 'g_full1')

            full1 = tf.nn.relu(full1, name='g_full1_act1')
            full1 = batch_norm(full1, scope='g_full1_bn', train=self.is_training)
            full1_act = tf.nn.leaky_relu(full1, name='g_full1_act2')

            if self.output_size == 128:

                conv1 = tf.reshape(full1_act, shape=[self.batch_size, s_h32, s_h32, k * 16], name='g_conv1')

                conv1 = deconv2d(conv1, [self.batch_size, s_h16, s_w16, k * 8], name='g_conv2')
                conv1 = batch_norm(conv1, scope='g_conv2_bn', train=self.is_training)
                conv1 = tf.nn.leaky_relu(conv1, name='g_conv2_act')
            else:
                conv1 = tf.reshape(full1_act, shape=[self.batch_size, s_h16, s_h16, k * 8], name='g_conv1')

            conv2 = deconv2d(conv1, [self.batch_size, s_h8, s_w8, k * 4], name='g_conv')
            conv2 = batch_norm(conv2, scope='g_conv_bn', train=self.is_training)
            conv2 = tf.nn.leaky_relu(conv2, name='g_conv_act')

            conv3 = deconv2d(conv2, [self.batch_size, s_h4, s_w4, k * 2], name='g_conv3')
            conv3 = batch_norm(conv3, scope='g_conv3_bn', train=self.is_training)
            conv3 = tf.nn.leaky_relu(conv3, name='g_conv3_act')

            conv4 = deconv2d(conv3, [self.batch_size, s_h2, s_w2, k], name='g_conv4')
            conv4 = batch_norm(conv4, scope='g_conv4_bn', train=self.is_training)
            conv4 = tf.nn.leaky_relu(conv4, name='g_conv4_act')

            conv5 = deconv2d(conv4, [self.batch_size, s_h, s_w, self.channel], name='g_conv5')

            conv5 = tf.nn.tanh(conv5, name='g_conv5_act')
            return conv5

    def to_image_sequence(self):
        samples_dir = self.sample_dir + self.version
        dir = self.sequence_dir + self.version
        if not os.path.exists(dir):
            os.makedirs(dir)
        count = 0
        for i in range(1, 101):
            for j in range(0, 311, 5):
                img_name = 'epoch_{}_batch_{}.jpg'.format(i, j)
                print(img_name)
                img = cv2.imread(os.path.join(samples_dir, img_name))
                cv2.imwrite(os.path.join(dir, "frame_{:05d}.jpg".format(count)), img)
                count += 1

    def make_movie(self):
        import cv2
        import os

        image_folder = self.sample_dir + self.version
        video_name = 'video.avi'

        images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, 0, 1, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()


from scipy.stats import truncnorm


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


if __name__ == "__main__":
    path = input("Enter saved model path: ")
    if path in ['', " "]:
        path=None
    cgan = Cgan()
    cgan.build_model()
    # cgan.make_movie()
    cgan.train(path=path)
