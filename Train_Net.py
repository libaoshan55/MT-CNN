import tensorflow as tf
import numpy as np
from project_mtcnn.sample import sampling_12_train
from project_mtcnn.sample import sampling_24_train
from project_mtcnn.sample import sampling_48_train
from project_mtcnn.sample import sampling_train
from project_mtcnn.sample import sampling_test
import matplotlib.image as img
import matplotlib.pyplot as plt
import PIL.Image as image
import PIL.ImageDraw as imagedraw
import os
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

p_pb_file_path = r"E:\AI\python_project\project_mtcnn\protocolbuffer\p_net\p_model.pb"
r_pb_file_path = r"E:\AI\python_project\project_mtcnn\protocolbuffer\r_net\r_model.pb"
o_pb_file_path = r"E:\AI\python_project\project_mtcnn\protocolbuffer\o_net\o_model.pb"

p_ckpt_file_path = r"E:\AI\python_project\project_mtcnn\checkpoint\p_parameter\p_param.ckpt"
r_ckpt_file_path = r"E:\AI\python_project\project_mtcnn\checkpoint\r_parameter\r_param.ckpt"
o_ckpt_file_path = r"E:\AI\python_project\project_mtcnn\checkpoint\o_parameter\o_param.ckpt"

class P_Net:

    def __init__(self):

        self.x = tf.placeholder(dtype=tf.float32,shape=[None,None,None,3],name="p_x")
        self.y = tf.placeholder(dtype=tf.float32,shape=[None,5])

        self.conv_w1 = tf.Variable(tf.truncated_normal(shape=[3,3,3,10],stddev=tf.sqrt(1/10),dtype=tf.float32))
        self.conv_b1 = tf.Variable(tf.zeros(shape=[10],dtype=tf.float32))#[5,5,10]

        self.conv_w2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 10, 16], stddev=tf.sqrt(1 / 16), dtype=tf.float32))
        self.conv_b2 = tf.Variable(tf.zeros(shape=[16], dtype=tf.float32))#[3,3,16]

        self.conv_w3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 16, 32], stddev=tf.sqrt(1 / 32), dtype=tf.float32))
        self.conv_b3 = tf.Variable(tf.zeros(shape=[32], dtype=tf.float32))#[1,1,32]

        self.conv_w4 = tf.Variable(tf.truncated_normal(shape=[1, 1, 32, 5], stddev=tf.sqrt(1 / 5), dtype=tf.float32))
        self.conv_b4 = tf.Variable(tf.zeros(shape=[5], dtype=tf.float32))#[1,1,5]

    def forward(self):
        self.conv_y1 = tf.nn.leaky_relu(tf.layers.batch_normalization
        (tf.nn.conv2d(input=self.x,filter=self.conv_w1,strides=[1,1,1,1],padding="VALID")+self.conv_b1))#[10,10,10]
        self.maxpool = tf.nn.max_pool(value=self.conv_y1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")#[5,5,10]

        self.conv_y2 = tf.nn.leaky_relu(tf.layers.batch_normalization
        (tf.nn.conv2d(input=self.maxpool,filter=self.conv_w2,strides=[1,1,1,1],padding="VALID")+self.conv_b2))#[3,3,16]

        self.conv_y3 = tf.nn.leaky_relu(tf.layers.batch_normalization
        (tf.nn.conv2d(input=self.conv_y2, filter=self.conv_w3, strides=[1, 1, 1, 1],padding="VALID") + self.conv_b3))#[1,1,32]

        self.conv_y4 = tf.nn.conv2d(input=self.conv_y3, filter=self.conv_w4, strides=[1, 1, 1, 1],padding="VALID") + self.conv_b4 #[1,1,5]

        self.label_cls =self.y[:,:1]
        self.label_offset = self.y[:,1:]


        self.output_cls = tf.sigmoid(self.conv_y4[...,:1])
        self.output_cls = tf.multiply(self.output_cls,1,name="p_cls_output")
        self.output_offset = tf.layers.batch_normalization(self.conv_y4[...,1:])
        self.output_offset = tf.multiply(self.output_offset,1,name="p_offset_output")

    def transpose(self):
        mask_cls_index = tf.where(self.label_cls[:, 0] < 2)  # [NHW,1],正样本和负样本，用来计算分类,自信度
        self._cls_output = tf.gather(self.output_cls, mask_cls_index)[0]# 获取索引对应的值，降维度,去掉批次
        self._cls_label = tf.gather(self.label_cls, mask_cls_index)[0]

        mask_offset_index = tf.where(self.label_cls[:, 0] > 0)  # 获取索引，[NHW,4],正样本和部分样本，用来计算回归
        self._offset_output = tf.gather( self.output_offset, mask_offset_index)[0]# 获取索引对应的值，降维度,去掉批次
        self._offset_label = tf.gather(self.label_offset, mask_offset_index)[0]

    def backward(self):
        # self.y1_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self._cls_label,logits=self._cls_output))
        self.y1_loss = tf.reduce_mean(tf.square(self._cls_label-self._cls_output))

        self.y2_loss = tf.reduce_mean(tf.abs(self._offset_label - self._offset_output))

        self.all_loss = tf.add(self.y1_loss ,self.y2_loss)

        self.optimizer = tf.train.AdamOptimizer(0.01).minimize(loss=self.all_loss)

    def validate(self):
        self.cls_accuracy = tf.multiply((1 - tf.reduce_sum((self.y[:,:1] - self._cls_output) ** 2 / tf.reduce_sum(self.y))), 100)
        self.coord_accuracy = tf.multiply((1 - tf.reduce_sum((self.y[:,1:] - self._offset_output)**2 / tf.reduce_sum(self.y))),100)

class R_Net:

    def __init__(self):

        self.x = tf.placeholder(dtype=tf.float32,shape=[None,None,None,3],name="r_x")
        self.y = tf.placeholder(dtype=tf.float32,shape=[None,5])

        self.conv_w1 = tf.Variable(tf.truncated_normal(shape=[3,3,3,28],stddev=tf.sqrt(1/28),dtype=tf.float32))
        self.conv_b1 = tf.Variable(tf.zeros(shape=[28],dtype=tf.float32))#[11,11,28]

        self.conv_w2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 28, 48], stddev=tf.sqrt(1 / 48), dtype=tf.float32))
        self.conv_b2 = tf.Variable(tf.zeros(shape=[48], dtype=tf.float32))#[4,4,48]

        self.conv_w3 = tf.Variable(tf.truncated_normal(shape=[2, 2, 48, 64], stddev=tf.sqrt(1 / 64), dtype=tf.float32))
        self.conv_b3 = tf.Variable(tf.zeros(shape=[64], dtype=tf.float32))#[3,3,64]

        self.fcn_w1 = tf.Variable(tf.truncated_normal(shape=[3*3*64,128],stddev=tf.sqrt(1/128),dtype=tf.float32))
        self.fcn_b1 = tf.Variable(tf.zeros(shape=[128],dtype=tf.float32))#[128]

        self.fcn_w2 = tf.Variable(tf.truncated_normal(shape=[128, 5], stddev=tf.sqrt(1 / 5), dtype=tf.float32))
        self.fcn_b2 = tf.Variable(tf.zeros(shape=[5], dtype=tf.float32))#[5]

    def forward(self):
        self.conv_y1 = tf.nn.leaky_relu(tf.layers.batch_normalization
        (tf.nn.conv2d(input=self.x,filter=self.conv_w1,strides=[1,1,1,1],padding="VALID")+self.conv_b1))#[22,22,28]
        self.maxpool_y1 = tf.nn.max_pool(value=self.conv_y1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")#[11,11,28]

        self.conv_y2 = tf.nn.leaky_relu(tf.layers.batch_normalization
        (tf.nn.conv2d(input=self.maxpool_y1,filter=self.conv_w2,strides=[1,1,1,1],padding="VALID")+self.conv_b2))#[9,9,48]
        self.maxpool_y2 = tf.nn.max_pool(value=self.conv_y2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding="VALID") # [4,4,48]

        self.conv_y3 = tf.nn.leaky_relu(tf.layers.batch_normalization
        (tf.nn.conv2d(input=self.maxpool_y2, filter=self.conv_w3, strides=[1, 1, 1, 1],padding="VALID") + self.conv_b3))#[3,3,64]

        self.y_fcn = tf.reshape(self.conv_y3,[-1,3*3*64])

        self.fcn_y1 = tf.nn.leaky_relu(tf.layers.batch_normalization(tf.matmul(self.y_fcn,self.fcn_w1)+self.fcn_b1))#[128]

        self.fcn_y2 = tf.matmul(self.fcn_y1,self.fcn_w2)+self.fcn_b2#[5]

        self.label_cls = self.y[:, :1]
        self.label_offset = self.y[:, 1:]
        #
        self.output_cls = tf.nn.sigmoid(self.fcn_y2[:, :1])
        self.output_cls = tf.multiply(self.output_cls,1,name="r_cls_output")
        self.output_offset = tf.layers.batch_normalization(self.fcn_y2[:, 1:])
        self.output_offset = tf.multiply(self.output_offset,1,name="r_offset_output")

    def transpose(self):
        mask_cls_index = tf.where(self.label_cls[:, 0] < 2)  # [NHW,1],正样本和负样本，用来计算分类,自信度
        self._cls_output = tf.gather(self.output_cls, mask_cls_index)[0]  # 获取索引对应的值，降维度,去掉批次
        self._cls_label = tf.gather(self.label_cls, mask_cls_index)[0]

        mask_offset_index = tf.where(self.label_cls[:, 0] > 0)  # 获取索引，[NHW,4],正样本和部分样本，用来计算回归
        self._offset_output = tf.gather(self.output_offset, mask_offset_index)[0]  # 获取索引对应的值，降维度,去掉批次
        self._offset_label = tf.gather(self.label_offset, mask_offset_index)[0]

    def backward(self):
        self.y1_loss = tf.reduce_mean(tf.square(self._cls_label-self._cls_output))
        self.y2_loss = tf.reduce_mean((self._offset_label - self._offset_output)**2)
        self.all_loss = tf.add(self.y1_loss ,self.y2_loss)

        self.optimizer = tf.train.AdamOptimizer().minimize(loss=self.all_loss)

    def validate(self):
        self.cls_accuracy = tf.multiply(
            (1 - tf.reduce_sum((self.y[:, :1] - self._cls_output) ** 2 / tf.reduce_sum(self.y))), 100)
        self.coord_accuracy = tf.multiply(
            (1 - tf.reduce_sum((self.y[:, 1:] - self._offset_output) ** 2 / tf.reduce_sum(self.y))), 100)

class O_Net:
    def __init__(self):

        self.x = tf.placeholder(dtype=tf.float32,shape=[None,None,None,3],name="o_x")
        self.y = tf.placeholder(dtype=tf.float32,shape=[None,15])

        self.conv_w1 = tf.Variable(tf.truncated_normal(shape=[3,3,3,32],stddev=tf.sqrt(1/32),dtype=tf.float32))
        self.conv_b1 = tf.Variable(tf.zeros(shape=[32],dtype=tf.float32))#[23,23,32]

        self.conv_w2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 64], stddev=tf.sqrt(1 / 64), dtype=tf.float32))
        self.conv_b2 = tf.Variable(tf.zeros(shape=[64], dtype=tf.float32))#[10,10,64]

        self.conv_w3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], stddev=tf.sqrt(1 / 64), dtype=tf.float32))
        self.conv_b3 = tf.Variable(tf.zeros(shape=[64], dtype=tf.float32))#[4,4,64]

        self.conv_w4 = tf.Variable(tf.truncated_normal(shape=[2, 2, 64,128], stddev=tf.sqrt(1 / 128), dtype=tf.float32))
        self.conv_b4 = tf.Variable(tf.zeros(shape=[128], dtype=tf.float32))  # [3,3,128]

        self.fcn_w1 = tf.Variable(tf.truncated_normal(shape=[3*3*128,256],stddev=tf.sqrt(1/256),dtype=tf.float32))
        self.fcn_b1 = tf.Variable(tf.zeros(shape=[256],dtype=tf.float32))#[256]

        self.fcn_w2 = tf.Variable(tf.truncated_normal(shape=[256, 15], stddev=tf.sqrt(1 / 15), dtype=tf.float32))
        self.fcn_b2 = tf.Variable(tf.zeros(shape=[15], dtype=tf.float32))#[15]

    def forward(self):
        self.conv_y1 = tf.nn.leaky_relu(tf.layers.batch_normalization
        (tf.nn.conv2d(input=self.x,filter=self.conv_w1,strides=[1,1,1,1],padding="VALID")+self.conv_b1))#[46,46,32]
        self.maxpool_y1 = tf.nn.max_pool(value=self.conv_y1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")#[23,23,32]

        self.conv_y2 = tf.nn.leaky_relu(tf.layers.batch_normalization
        (tf.nn.conv2d(input=self.maxpool_y1,filter=self.conv_w2,strides=[1,1,1,1],padding="VALID")+self.conv_b2))#[21,21,64]
        self.maxpool_y2 = tf.nn.max_pool(value=self.conv_y2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding="VALID")  # [10,10,64]

        self.conv_y3 = tf.nn.leaky_relu(tf.layers.batch_normalization
        (tf.nn.conv2d(input=self.maxpool_y2, filter=self.conv_w3, strides=[1, 1, 1, 1],padding="VALID") + self.conv_b3))#[8,8,64]
        self.maxpool_y3 = tf.nn.max_pool(value=self.conv_y3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding="VALID")  # [4,4,64]

        self.conv_y4 = tf.nn.leaky_relu(tf.layers.batch_normalization
        (tf.nn.conv2d(input=self.maxpool_y3, filter=self.conv_w4, strides=[1, 1, 1, 1],padding="VALID") + self.conv_b4))  # [3,3,128]

        self.y_fcn = tf.reshape(self.conv_y4,[-1,3*3*128])

        self.fcn_y1 = tf.nn.leaky_relu(tf.layers.batch_normalization(tf.matmul(self.y_fcn,self.fcn_w1)+self.fcn_b1))#[256]

        self.fcn_y2 =tf.matmul(self.fcn_y1,self.fcn_w2)+self.fcn_b2#[15]

        self.label_cls = self.y[:, :1]
        self.label_offset = self.y[:, 1:5]
        self.label_markoffset = self.y[:, 5:15]
        #
        self.output_cls = tf.nn.sigmoid(self.fcn_y2[:, :1])
        self.output_cls = tf.multiply(self.output_cls,1,name="o_cls_output")
        self.output_offset = tf.layers.batch_normalization(self.fcn_y2[:, 1:5])
        self.output_offset = tf.multiply(self.output_offset,1,name="o_offset_output")
        self.output_markoffset = tf.layers.batch_normalization(self.fcn_y2[:,5:15])
        self.output_markoffset = tf.multiply(self.output_markoffset,1,name="o_markoffset_output")

    def transpose(self):
        mask_cls_index = tf.where(self.label_cls[:, 0] < 2)  # [NHW,1],正样本和负样本，用来计算分类,自信度
        self._cls_output = tf.gather(self.output_cls, mask_cls_index)[0]  # 获取索引对应的值，降维度,去掉批次
        self._cls_label = tf.gather(self.label_cls, mask_cls_index)[0]

        mask_offset_index = tf.where(self.label_cls[:, 0] > 0)  # 获取索引，[NHW,4],正样本和部分样本，用来计算回归
        self._offset_output = tf.gather(self.output_offset, mask_offset_index)[0]  # 获取索引对应的值，降维度,去掉批次
        self._offset_label = tf.gather(self.label_offset, mask_offset_index)[0]
        self._offset_markoutput = tf.gather(self.output_markoffset, mask_offset_index)[0]  # 获取索引对应的值，降维度,去掉批次
        self._offset_marklabel = tf.gather(self.label_markoffset, mask_offset_index)[0]

    def backward(self):
        self.y1_loss = tf.reduce_mean(tf.square(self._cls_label-self._cls_output))
        self.y2_loss = tf.reduce_mean((self._offset_label - self._offset_output)**2)
        self.y3_loss = tf.reduce_mean((self._offset_marklabel - self._offset_markoutput) ** 2)

        self.all_loss = self.y1_loss+self.y2_loss+self.y3_loss

        self.optimizer = tf.train.AdamOptimizer().minimize(loss=self.all_loss)

    def validate(self):
        self.cls_accuracy = tf.multiply(
            (1 - tf.reduce_sum((self.y[:, :1] - self._cls_output) ** 2 / tf.reduce_sum(self.y))), 100)
        self.coord_accuracy = tf.multiply(
            (1 - tf.reduce_sum((self.y[:, 1:5] - self._offset_output) ** 2 / tf.reduce_sum(self.y))), 100)
        self.mark_accuracy = tf.multiply(
            (1 - tf.reduce_sum((self.y[:, 5:15] - self._offset_markoutput) ** 2 / tf.reduce_sum(self.y))), 100)

class Net:

    def pnet(self):
        self.p_net = P_Net()
        self.p_net.forward()
        self.p_net.transpose()
        self.p_net.backward()
        self.p_net.validate()

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        a = []
        b = []
        plt.ion()

        with tf.Session() as sess:
            # sess.run(init)
            saver.restore(sess,p_ckpt_file_path)
            img,label= sampling_12_train.sample_train.get_batch(5000)
            for i in range(20000):
                indices = np.random.randint(0,len(img),500)  # shape[0]表示第0轴的长度，通常是训练数据的数量
                xs = img[indices]
                ys= label[indices]

                cls,coord,cls_error,coord_error,error,_ = sess.run(fetches=[self.p_net.output_cls,self.p_net.output_offset,self.p_net.y1_loss,self.p_net.y2_loss,self.p_net.all_loss,self.p_net.optimizer],
                feed_dict={self.p_net.x:xs,self.p_net.y:ys})

                if i% 100 == 0:
                    out_cls = cls[0][...,0]
                    _offset_outx1 = coord[0][...,0]
                    _offset_outy1 = coord[0][...,1]
                    _offset_outx2 = coord[0][...,2]
                    _offset_outy2 = coord[0][...,3]

                    out_x1 = _offset_outx1 * 12 + 0
                    out_y1 = _offset_outy1 * 12 + 0
                    out_x2 = _offset_outx2 * 12 + 12
                    out_y2 = _offset_outy2 * 12 + 12

                    label_cls = int(ys[0][0])
                    _offset_labelx1 = float(ys[0][1])
                    _offset_labely1 = float(ys[0][2])
                    _offset_labelx2 = float(ys[0][3])
                    _offset_labely2 = float(ys[0][4])

                    label_x1 = _offset_labelx1 * 12 + 0
                    label_y1 = _offset_labely1 * 12 + 0
                    label_x2 = _offset_labelx2 * 12 + 12
                    label_y2 = _offset_labely2 * 12 + 12

                    print("p_net:",i)
                    print(cls_error, coord_error, error)

                    print("label_x1:", label_x1, ",label_y1:", label_y1, ",label_x2:", label_x2, ",out_y2:", label_y2)
                    print("label_cls:", label_cls)
                    print("out_x1:", out_x1, ",out_y1:", out_y1, ",out_x2:", out_x2, ",out_y2:", out_y2)
                    print("out_cls:", out_cls)

                    # a.append(i)
                    # b.append(error)
                    # plt.clf()
                    # plt.plot(a,b)
                    # plt.pause(0.001)
                    saver.save(sess,p_ckpt_file_path)
                    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,["p_x","p_cls_output","p_offset_output"])
                    with tf.gfile.FastGFile(p_pb_file_path, mode='wb') as f: f.write(constant_graph.SerializeToString())

    def rnet(self):
        self.r_net = R_Net()
        self.r_net.forward()
        self.r_net.transpose()
        self.r_net.backward()
        self.r_net.validate()

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        a = []
        b = []
        plt.ion()

        with tf.Session() as sess:
            # sess.run(init)
            saver.restore(sess,r_ckpt_file_path)

            img, label = sampling_24_train.sample_train.get_batch(5000)
            for i in range(40000):
                indices = np.random.randint(0, len(img), 500)  # shape[0]表示第0轴的长度，通常是训练数据的数量
                xs = img[indices]
                ys = label[indices]

                cls,coord,cls_error,coord_error,error,_ = sess.run(fetches=[self.r_net.output_cls,self.r_net.output_offset,self.r_net.y1_loss,self.r_net.y2_loss,self.r_net.all_loss,self.r_net.optimizer],
                feed_dict={self.r_net.x:xs,self.r_net.y:ys})

                if i% 50 == 0:
                    print("r_net:",i)
                    print(cls_error, coord_error, error)

                    # img = image.fromarray(np.uint8(imgs))
                    # w,h = img.size
                    # img = img.resize((int(w*scale[0]),int(h*scale[0])))

                    out_cls = cls[0][0]
                    _offset_outx1 = coord[0][0]
                    _offset_outy1 = coord[0][1]
                    _offset_outx2 = coord[0][2]
                    _offset_outy2 = coord[0][3]


                    out_x1 = _offset_outx1 * 24 + 0
                    out_y1 = _offset_outy1 * 24 + 0
                    out_x2 = _offset_outx2 * 24 + 24
                    out_y2 = _offset_outy2 * 24 + 24

                    label_cls = int(ys[0][0])
                    _offset_labelx1 = float(ys[0][1])
                    _offset_labely1 = float(ys[0][2])
                    _offset_labelx2 = float(ys[0][3])
                    _offset_labely2 = float(ys[0][4])


                    label_x1 = _offset_labelx1 * 24 + 0
                    label_y1 = _offset_labely1 * 24 + 0
                    label_x2 = _offset_labelx2 * 24 + 24
                    label_y2 = _offset_labely2 * 24 + 24
                    #
                    print("label_x1:", label_x1, ",label_y1:", label_y1, ",label_x2:", label_x2, ",out_y2:", label_y2)
                    print("label_cls:", label_cls)
                    print("out_x1:", out_x1, ",out_y1:", out_y1, ",out_x2:", out_x2, ",out_y2:", out_y2)
                    print("out_cls:", out_cls)
                    # imgdraw = imagedraw.Draw(img)
                    # imgdraw.rectangle((label_x1, label_y1, label_x2, label_y2), outline="blue")
                    # imgdraw.rectangle((out_x1, out_y1, out_x2, out_y2), outline="red")
                    # img.show()

                    # a.append(i)
                    # b.append(error)
                    # plt.clf()
                    # plt.plot(a,b)
                    # plt.pause(0.001)
                    saver.save(sess,r_ckpt_file_path)
                    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,["r_x","r_cls_output","r_offset_output"])
                    with tf.gfile.FastGFile(r_pb_file_path, mode='wb') as f:f.write(constant_graph.SerializeToString())

    def onet(self):
        self.o_net = O_Net()
        self.o_net.forward()
        self.o_net.transpose()
        self.o_net.backward()
        self.o_net.validate()

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        a = []
        b = []
        plt.ion()

        with tf.Session() as sess:
            sess.run(init)
            # saver.restore(sess,o_ckpt_file_path)

            img, label = sampling_48_train.sample_train.get_batch(5000)
            for i in range(80000):
                indices = np.random.randint(0, len(img), 500)  # shape[0]表示第0轴的长度，通常是训练数据的数量
                xs = img[indices]
                ys = label[indices]

                cls,coord,mark_coord,cls_error,coord_error,mark_error,error,_ = sess.run\
                (fetches=[self.o_net.output_cls,self.o_net.output_offset,self.o_net.output_markoffset,self.o_net.y1_loss,self.o_net.y2_loss,self.o_net.y3_loss,self.o_net.all_loss,self.o_net.optimizer],
                feed_dict={self.o_net.x:xs,self.o_net.y:ys})

                if i % 10 == 0:
                    print(i)
                    print(cls_error, coord_error, error)

                    # img = image.fromarray(np.uint8(imgs))
                    # w,h = img.size
                    # img = img.resize((int(w*scale[0]),int(h*scale[0])))

                    out_cls = cls[0][0]
                    _offset_outx1 = coord[0][0]
                    _offset_outy1 = coord[0][1]
                    _offset_outx2 = coord[0][2]
                    _offset_outy2 = coord[0][3]

                    out_x1 = _offset_outx1 * 48 + 0
                    out_y1 = _offset_outy1 * 48 + 0
                    out_x2 = _offset_outx2 * 48 + 48
                    out_y2 = _offset_outy2 * 48 + 48

                    label_cls = int(ys[0][0])
                    _offset_labelx1 = float(ys[0][1])
                    _offset_labely1 = float(ys[0][2])
                    _offset_labelx2 = float(ys[0][3])
                    _offset_labely2 = float(ys[0][4])

                    label_x1 = _offset_labelx1 * 48 + 0
                    label_y1 = _offset_labely1 * 48 + 0
                    label_x2 = _offset_labelx2 * 48 + 48
                    label_y2 = _offset_labely2 * 48 + 48
                    #
                    print("label_x1:", label_x1, ",label_y1:", label_y1, ",label_x2:", label_x2, ",out_y2:",label_y2)
                    print("label_cls:", label_cls)
                    print("out_x1:", out_x1, ",out_y1:", out_y1, ",out_x2:", out_x2, ",out_y2:", out_y2)
                    print("out_cls:", out_cls)
                    # imgdraw = imagedraw.Draw(img)
                    # imgdraw.rectangle((label_x1, label_y1, label_x2, label_y2), outline="blue")
                    # imgdraw.rectangle((out_x1, out_y1, out_x2, out_y2), outline="red")
                    # img.show()



                    # a.append(i)
                    # b.append(error)
                    # plt.clf()
                    # plt.plot(a, b)
                    # plt.pause(0.001)
                    saver.save(sess,o_ckpt_file_path)
                    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,["o_x","o_cls_output", "o_offset_output","o_markoffset_output"])
                    with tf.gfile.FastGFile(o_pb_file_path, mode='wb') as f:f.write(constant_graph.SerializeToString())
if __name__ == "__main__":
    net = Net()
    # print(net.pnet())
    print(net.rnet())
    # print(net.onet())
