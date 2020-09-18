import time
import os
import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel


parser = argparse.ArgumentParser()
parser.add_argument(
    '--flist', default='', type=str,
    help='The filenames of image to be processed: input, mask, output.')
parser.add_argument(
    '--image_height', default=-1, type=int,
    help='The height of images should be defined, otherwise batch mode is not'
    ' supported.')
parser.add_argument(
    '--image_width', default=-1, type=int,
    help='The width of images should be defined, otherwise batch mode is not'
    ' supported.')
parser.add_argument(
    '--checkpoint_dir', default='', type=str,
    help='The directory of tensorflow checkpoint.')

def resize_with_pad(image, height, width):

    def get_padding_size(image, height, width):
        # h, w, _ = image.shape
        h = image.shape[0]
        w = image.shape[1]
        # print("h=",h,"w=",w)
        top, bottom, left, right = (0, 0, 0, 0)
        if h < height:
            dh = height - h
            top = dh // 2
            bottom = dh - top
        if w < width:
            dw = width - w
            left = dw // 2
            right = dw - left
        else:
            pass
        return top, bottom, left, right

    top, bottom, left, right = get_padding_size(image, height, width)
    BLACK = [0, 0, 0]
    constant = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    return constant

def resizeToOrgImg(bgOriginalshape, new):
    if(bgOriginalshape[0] < new.shape[0]):
        diffH = int((new.shape[0] - bgOriginalshape[0]) / 2)
        new = new[diffH:bgOriginalshape[0] + diffH, :, :]

    if(bgOriginalshape[1] < new.shape[1]):
        diffW = int((new.shape[1] - bgOriginalshape[1]) / 2)
        new = new[:, diffW:bgOriginalshape[1] + diffW, :]

    return new

if __name__ == "__main__":
    FLAGS = ng.Config('inpaint.yml')
    # ng.get_gpus(1)
    # os.environ['CUDA_VISIBLE_DEVICES'] ='0'
    args = parser.parse_args()
    kernel = np.ones((11, 11), np.uint8)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    model = InpaintCAModel()
    # input_image_ph = tf.placeholder(
    #     tf.float32, shape=(1, args.image_height/4, args.image_width/2, 3))
    input_image_ph = tf.placeholder(
        tf.float32, shape=(1, args.image_height, args.image_width*2, 3))
    output = model.build_server_graph(FLAGS, input_image_ph)
    output = (output + 1.) * 127.5
    output = tf.reverse(output, [-1])
    output = tf.saturate_cast(output, tf.uint8)
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = []
    for var in vars_list:
        vname = var.name
        from_name = vname
        var_value = tf.contrib.framework.load_variable(
            args.checkpoint_dir, from_name)
        assign_ops.append(tf.assign(var, var_value))
    sess.run(assign_ops)
    print('Model loaded.')

    with open(args.flist, 'r') as f:
        lines = f.read().splitlines()
    t = time.time()
    for line in lines:
    # for i in range(100):
        image, mask, out = line.split()
        base = os.path.basename(mask)

        image = cv2.imread(image)
        mask = cv2.imread(mask)

        # image = cv2.resize(image, (args.image_width, args.image_height))
        # mask = cv2.resize(mask, (args.image_width, args.image_height))

        o_shape = image.shape
        o_image = image
        o_mask = mask

        image = resize_with_pad(image, args.image_height, args.image_width)
        mask = resize_with_pad(mask, args.image_height, args.image_width)
        # image = cv2.resize(image, (int(args.image_width/4), int(args.image_height/4)))
        # mask = cv2.resize(mask, (int(args.image_width/4), int(args.image_height/4)))

        # cv2.imwrite(out, image*(1-mask/255.) + mask)
        # # continue
        # image = np.zeros((128, 256, 3))
        # mask = np.zeros((128, 256, 3))

        ######Slice the image to 128x128 parts
        # outputImage = np.zeros((4,(int(args.image_height/4),(int(args.image_width/4)))
        # grid_h = int(args.image_height/4)
        # grid_w = int(args.image_width/4)
        # outputImage = np.zeros(( int(args.image_height),int(args.image_width),3 ))
        # outputImageOuts = [0.0]*4
        # for i in range(0,2):
        #     for j in range(0,2):
        #         assert image.shape == mask.shape

        #         # print( (i*grid_h),(i+1)*grid_h,(j*grid_w),(j+1)*grid_w )
        #         image_tmp = image[(i*grid_h):(i+1)*grid_h,(j*grid_w):(j+1)*grid_w,:]
        #         mask_tmp = mask[(i*grid_h):(i+1)*grid_h,(j*grid_w):(j+1)*grid_w,:]
        #         mask_tmp = cv2.dilate(mask_tmp, kernel, iterations=1)

        #         h, w, _ = image_tmp.shape
        #         # grid = 4
        #         grid = 8
        #         image_tmp = image_tmp[:h//grid*grid, :w//grid*grid, :]
        #         mask_tmp = mask_tmp[:h//grid*grid, :w//grid*grid, :]
        #         print('Shape of image: {}'.format(image_tmp.shape))

        #         image_tmp = np.expand_dims(image_tmp, 0)
        #         mask_tmp = np.expand_dims(mask_tmp, 0)
        #         input_image = np.concatenate([image_tmp, mask_tmp], axis=2)

        #         # load pretrained model
        #         result = sess.run(output, feed_dict={input_image_ph: input_image})
        #         print('Processed: {}'.format(out))

        #         tmp = result[0][:, :, ::-1]
        #         # outputImageOuts[(i*2)+j] = np.copy(tmp)
        #         outputImage[(i*grid_h):(i+1)*grid_h,(j*grid_w):(j+1)*grid_w,:] = np.copy(tmp)
        #         # cv2.imwrite(out+str((i*2)+j)+".jpg", tmp)
        ######Join the output

        assert image.shape == mask.shape

        # print( (i*grid_h),(i+1)*grid_h,(j*grid_w),(j+1)*grid_w )
        # image_tmp = image[(i*grid_h):(i+1)*grid_h,(j*grid_w):(j+1)*grid_w,:]
        # mask_tmp = mask[(i*grid_h):(i+1)*grid_h,(j*grid_w):(j+1)*grid_w,:]
        mask = cv2.dilate(mask, kernel, iterations=1)

        h, w, _ = image.shape
        # grid = 4
        grid = 8
        # grid = 16
        image = image[:h//grid*grid, :w//grid*grid, :]
        mask = mask[:h//grid*grid, :w//grid*grid, :]
        # print('Shape of image: {}'.format(image.shape))

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image = np.concatenate([image, mask], axis=2)

        # load pretrained model
        result = sess.run(output, feed_dict={input_image_ph: input_image})
        # print('Processed: {}'.format(out))

        outputImage = result[0][:, :, ::-1]
        # outputImageOuts[(i*2)+j] = np.copy(tmp)
        # outputImage[(i*grid_h):(i+1)*grid_h,(j*grid_w):(j+1)*grid_w,:] = np.copy(tmp)
        # cv2.imwrite(out+str((i*2)+j)+".jpg", tmp)

        # exit(1)
        # outputImage = cv2.resize(outputImage, (args.image_width, args.image_height))
        # for i in range(0,2):
        #     for j in range(0,2):
        #         outputImage[(i*grid_h):(i+1)*grid_h,(j*grid_w):(j+1)*grid_w,:]=outputImageOuts[(i*2)+j]
        outputImage = resizeToOrgImg(o_shape, outputImage)

        # extend from 1 channel to 3
        # mask3d = np.tile(o_mask[:, :, None], [1, 1, 3])

        # dilate mask to process additional border
        mask3d = cv2.dilate(o_mask, kernel, iterations=1)
        mask3d = cv2.blur(mask3d, (11, 11))
        mask3d = mask3d / 255.0  # convert to float
        inv_mask3d = 1.0 - mask3d  # need to invert mask due to framework

        outputImage = (mask3d*outputImage)+(inv_mask3d*o_image)

        cv2.imwrite(out, outputImage)

    print('Time total: {}'.format(time.time() - t))
