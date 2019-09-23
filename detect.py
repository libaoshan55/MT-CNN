import tensorflow as tf
import numpy as np
import PIL.Image as image
import PIL.ImageDraw as draw
import matplotlib.image as img
import matplotlib.pyplot as plt
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from skimage import io,transform
from project_mtcnn.tools import until

p_pb_file_path = r"E:\AI\python_project\project_mtcnn\protocolbuffer\p_net\p_model.pb"
r_pb_file_path = r"E:\AI\python_project\project_mtcnn\protocolbuffer\r_net\r_model.pb"
o_pb_file_path = r"E:\AI\python_project\project_mtcnn\protocolbuffer\o_net\o_model.pb"


from project_mtcnn.sample import sampling_test

class Detector:

    def pnet(self):
        sess = tf.Session()
        with gfile.FastGFile(p_pb_file_path,mode="rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')  # 导入计算图

            sess.run(tf.global_variables_initializer())
            input_x = sess.graph.get_tensor_by_name('x:0')
            output_cls = sess.graph.get_tensor_by_name("cls_output:0")
            output_offset = sess.graph.get_tensor_by_name("offset_output:0")

            pyramid = list(transform.pyramid_gaussian(img_array, downscale=1.5))
            img_array_w = img_array.shape[1]
            img_array_h = img_array.shape[0]
            # print(img_array_w,img_array_h)

            image_num = len(pyramid)
            box = np.zeros([0,5],dtype=np.float32)
            boxes =[]
            for i in range(image_num):
                # print(pyramid[i].shape[0],pyramid[i].shape[1])
                if min(pyramid[i].shape[0], pyramid[i].shape[1]) < 12:
                    break
                img_resized = pyramid[i]
                img_resized_h = img_resized.shape[0]
                img_resized_w = img_resized.shape[1]

                w_resize = img_array_w / img_resized_w
                h_resize = img_array_h / img_resized_h

                img_resized = np.array([img_resized])
                _cls, _offset = sess.run([output_cls, output_offset], feed_dict={input_x: img_resized})

                # 对应上每个偏移框上的值
                _offset[..., 0] = _offset[..., 0] * 12 + 0
                _offset[..., 1] = _offset[..., 1] * 12 + 0
                _offset[..., 2] = _offset[..., 2] * 12 + 12
                _offset[..., 3] = _offset[..., 3] * 12 + 12

                # 对应到每个金字塔图上的坐标值,然后在还原到测试图上
                for i in range(_offset.shape[1]):#遍历h索引
                    _offset[:, i, :, 1] = _offset[:, i, :, 1] + i * 2#还原h到缩放后的图
                    _offset[:, i, :, 3] = _offset[:, i, :, 3] + i * 2

                _offset[..., 1] = _offset[..., 1] * h_resize#把h还原到原图上
                _offset[..., 3] = _offset[..., 3] * h_resize

                for i in range(_offset.shape[2]):#遍历w索引
                    _offset[:, :, i, 0] = _offset[:, :, i, 0] + i * 2#还原w到缩放后的图
                    _offset[:, :, i, 2] = _offset[:, :, i, 2] + i * 2

                _offset[..., 0] = _offset[..., 0] * w_resize#把w还原到原图上
                _offset[..., 2] = _offset[..., 2] * w_resize

                isobj_result = np.concatenate((_cls, _offset), axis=3)#把一张图的cls和offset连接起来

                index = np.where(isobj_result[..., 0] > 0.7)
                isobj_result = isobj_result[index]
                one_img_box = until.nms(isobj_result, 0.3)

                if one_img_box.shape[0]>0:
                    box = np.concatenate((box,one_img_box),axis=0)
                    # print(box.shape, "******45656")

            #     img = draw.Draw(img_test)
            #     for i in range(one_img_box.shape[0]):
            #         img.rectangle(one_img_box[i, 1:], outline='red')
            # img_test.show()
            return box

    def rnet(self,img_box):
        sess = tf.Session()
        with gfile.FastGFile(r_pb_file_path, mode="rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')  # 导入计算图

            sess.run(tf.global_variables_initializer())
            input_x = sess.graph.get_tensor_by_name('r_x:0')
            output_cls = sess.graph.get_tensor_by_name("r_cls_output:0")
            output_offset = sess.graph.get_tensor_by_name("r_offset_output:0")

            convert_img = until.convert_to_square(img_box)
            boxes = []
            for box in convert_img:
                convert_x1 = box[1]
                convert_y1 = box[2]
                convert_x2 = box[3]
                convert_y2 = box[4]
                convert_w = convert_x2 -convert_x1
                convert_h = convert_y2 - convert_y1
                img_crop = img_test.crop([convert_x1,convert_y1,convert_x2,convert_y2])
                img_size = img_crop.resize((24,24))
                img_array = [np.array(img_size)/255]
                cls,offset = sess.run([output_cls,output_offset],feed_dict={input_x:img_array})
                # print(cls,"r_cls")
                #返回到原图的坐标
                x1 = offset[:, 0] * convert_w + convert_x1
                y1 = offset[:, 1] * convert_h + convert_y1
                x2 = offset[:, 2] * convert_w + convert_x2
                y2 = offset[:, 3] * convert_h + convert_y2

                x1 = np.expand_dims(x1,axis=1)
                y1 = np.expand_dims(y1, axis=1)
                x2 = np.expand_dims(x2, axis=1)
                y2 = np.expand_dims(y2, axis=1)
                box = np.concatenate((cls,x1,y1,x2,y2),axis=1)
                boxes.append(box[0])
            boxes = np.stack(boxes)
            # print(boxes.shape)
            index = np.where(boxes[:, 0]>0.8)
            isobj_result = boxes[index]
            # print(isobj_result.shape)
            rone_img_box = until.nms(isobj_result, 0.3)
            # print(rone_img_box.shape)
        #     img = draw.Draw(img_test)
        #     for i in range(rone_img_box.shape[0]):
        #         img.rectangle(rone_img_box[i, 1:], outline='red')
        # img_test.show()
        return rone_img_box

    def onet(self,rimg_box):
        sess = tf.Session()
        with gfile.FastGFile(o_pb_file_path, mode="rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')  # 导入计算图
            sess.run(tf.global_variables_initializer())
            input_x = sess.graph.get_tensor_by_name('o_x:0')
            output_cls = sess.graph.get_tensor_by_name("o_cls_output:0")
            output_offset = sess.graph.get_tensor_by_name("o_offset_output:0")

            convert_img = until.convert_to_square(rimg_box)
            boxes = []
            for box in convert_img:
                markoffset_output = sess.graph.get_tensor_by_name("o_markoffset_output:0")
                convert_x1 = box[1]
                convert_y1 = box[2]
                convert_x2 = box[3]
                convert_y2 = box[4]
                convert_w = convert_x2 -convert_x1
                convert_h = convert_y2 - convert_y1

                img_crop = img_test.crop([convert_x1,convert_y1,convert_x2,convert_y2])
                img_size = img_crop.resize((48,48))
                img_array = np.array(img_size)/255
                img_array = np.expand_dims(img_array, 0)

                cls,offset,markoffset_output = sess.run([output_cls, output_offset,markoffset_output],feed_dict={input_x:img_array})

                #返回到原图的坐标

                x1 = offset[:, 0] * convert_w + convert_x1
                y1 = offset[:, 1] * convert_h + convert_y1
                x2 = offset[:, 2] * convert_w + convert_x2
                y2 = offset[:, 3] * convert_h + convert_y2

                lefteye_x = markoffset_output[:,0] * convert_w + convert_x1
                lefteye_y = markoffset_output[:,1] * convert_h + convert_y1
                righteye_x = markoffset_output[:,2] * convert_w + convert_x1
                righteye_y = markoffset_output[:,3] * convert_h + convert_y1
                nose_x = markoffset_output[:,4] * convert_w + convert_x1
                nose_y = markoffset_output[:,5] * convert_h + convert_y1
                leftmouth_x = markoffset_output[:,6] * convert_w + convert_x1
                leftmouth_y = markoffset_output[:,7] * convert_h + convert_y1
                rightmouth_x = markoffset_output[:,8] * convert_w + convert_x1
                rightmouth_y = markoffset_output[:,9] * convert_h + convert_y1

                x1 = np.expand_dims(x1,axis=1)
                y1 = np.expand_dims(y1, axis=1)
                x2 = np.expand_dims(x2, axis=1)
                y2 = np.expand_dims(y2, axis=1)

                lefteye_x = np.expand_dims(lefteye_x,axis=1)
                lefteye_y = np.expand_dims(lefteye_y,axis=1)
                righteye_x = np.expand_dims(righteye_x,axis=1)
                righteye_y = np.expand_dims(righteye_y,axis=1)
                nose_x = np.expand_dims(nose_x,axis=1)
                nose_y = np.expand_dims(nose_y,axis=1)
                leftmouth_x = np.expand_dims(leftmouth_x,axis=1)
                leftmouth_y = np.expand_dims(leftmouth_y,axis=1)
                rightmouth_x = np.expand_dims(rightmouth_x,axis=1)
                rightmouth_y = np.expand_dims(rightmouth_y,axis=1)

                box = np.concatenate((cls,x1,y1,x2,y2,lefteye_x,lefteye_y,righteye_x,righteye_y,
                nose_x,nose_y,leftmouth_x,leftmouth_y,rightmouth_x,rightmouth_y),axis=1)
                boxes.append(box[0])

            boxes = np.stack(boxes)
            print(boxes.shape)
            index = np.where(boxes[:, 0]>0.85)
            isobj_result = boxes[index]
            print(isobj_result.shape)
            rone_img_box = until.nms(isobj_result, 0.3, isMin = True)
            print(rone_img_box.shape)
            img = draw.Draw(img_test)
            o = 1
            x = 2
            for i in range(rone_img_box.shape[0]):
                img.rectangle(rone_img_box[i, 1:5], outline='red')
                img.rectangle((rone_img_box[i, 1]-o,rone_img_box[i, 2]-o,rone_img_box[i, 3]+o,rone_img_box[i, 4]+o), outline='red')
                # img.point(rone_img_box[i,5:],fill="blue")
                img.rectangle((rone_img_box[i, 5]-x,rone_img_box[i, 6]-x,rone_img_box[i, 5]+x,rone_img_box[i, 6]+x), fill="blue")
                img.rectangle((rone_img_box[i, 7]-x,rone_img_box[i, 8]-x,rone_img_box[i, 7]+x,rone_img_box[i, 8]+x), fill="blue")
                img.rectangle((rone_img_box[i, 9]-x,rone_img_box[i, 10]-x,rone_img_box[i, 9]+x,rone_img_box[i, 10]+x), fill="blue")
                img.rectangle((rone_img_box[i, 11]-x,rone_img_box[i, 12]-x,rone_img_box[i, 11]+x,rone_img_box[i, 12]+x), fill="blue")
                img.rectangle((rone_img_box[i, 13]-x,rone_img_box[i, 14]-x,rone_img_box[i, 13]+x,rone_img_box[i, 14]+x), fill="blue")
        img_test.show()

detect = Detector()
img_path = r"E:\AI\python_project\project_mtcnn\dataset\test_image\6.jpg"
img_array = io.imread(img_path)
img_test = image.open(img_path)
img_box = detect.pnet()
rimg_box = detect.rnet(img_box)
oimg_box = detect.onet(rimg_box)
# print(img_box)
# print(rimg_box)
print(oimg_box)

