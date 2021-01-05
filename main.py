from collections import namedtuple
import tensorflow as tf
import os
import model as m
import glob as gb
import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--channels', dest='channels', type=int, default=3, help='image channel size')
parser.add_argument('--checkpoint', dest='ckpt_dir', default='./checkpoints', help='models are saved here')
parser.add_argument('--out_dir', dest='out_dir', default='out', help='out dir for testing')
parser.add_argument('--imgdir', dest='imgdir', default='', help='')
parser.add_argument('--image')
args = parser.parse_args()

def main():
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    images = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
    model = m.IRCNN(images)
    model.build_graph()
    saver = tf.train.Saver()

    try:
        ckpt_state = tf.train.get_checkpoint_state(args.ckpt_dir)
    except tf.errors.OutOfRangeError as e:
        tf.logging.error('Cannot restore checkpoint: %s', e)

    swts = ['*.jpg','*.png','*.jpeg','*.bmp','*.JPEG','*.PNG','*.JPG','*.BMP']
    path_lists = []
    if args.image:
        path_lists.append(args.image)
    else:
        for swt in swts:
            path_lists.extend(gb.glob(os.path.join(args.imgdir, swt)))

    with tf.Session() as sess:
        saver.restore(sess, ckpt_state.model_checkpoint_path)
        for path in path_lists:
            gt = cv2.imread(path)
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

            gt_img = np.expand_dims(gt, axis=0)
            gt = gt_img.astype(np.float32) / 255.
            img= sess.run(model.clear, feed_dict={images:gt})

            image_name = os.path.basename(path)

            img = np.clip(img*255, 0, 255).astype('uint8')

            cv2.imwrite(os.path.join(args.out_dir, image_name), cv2.cvtColor(np.squeeze(img), cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    main()
