import os
import numpy as np
import skimage
import tensorflow as tf
from pydensecrf import densecrf

from _rmi import _init_paths
from LSTM_model import LSTM_model
from util.refvg_loader import RMIRefVGLoader
from util.processing_tools import *
from util import im_processing

the_mu = np.array((104.00698793, 116.66876762, 122.67891434))


def lstm_refvg_predictor(split='val', eval_img_count=-1, out_path='output/eval_refvg/lstm', model_iter=750000,
                        dcrf=True, mu=the_mu):
    pretrained_model = './_rmi/refvg/tfmodel/refvg_resnet_LSTM_iter_' + str(model_iter) + '.tfmodel'

    data_loader = RMIRefVGLoader(split=split)
    vocab_size = len(data_loader.vocab_dict)

    score_thresh = 1e-9
    H, W = 320, 320

    model = LSTM_model(H=H, W=W, mode='eval', vocab_size=vocab_size, weights='resnet')

    # Load pretrained model
    snapshot_restorer = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    snapshot_restorer.restore(sess, pretrained_model)

    predictions = dict()

    while not data_loader.is_end:
        img_id, task_id, im, mask, sent, text = data_loader.get_img_data(rand=False, is_train=False)
        mask = mask.astype(np.float32)

        proc_im = skimage.img_as_ubyte(im_processing.resize_and_pad(im, H, W))
        proc_im_ = proc_im.astype(np.float32)
        proc_im_ = proc_im_[:, :, ::-1]
        proc_im_ -= mu

        scores_val, up_val, sigm_val = sess.run([model.pred, model.up, model.sigm],
                                                feed_dict={
                                                    model.words: np.expand_dims(text, axis=0),
                                                    model.im: np.expand_dims(proc_im_, axis=0)
                                                })

        # scores_val = np.squeeze(scores_val)
        # pred_raw = (scores_val >= score_thresh).astype(np.float32)
        up_val = np.squeeze(up_val)
        pred_raw = (up_val >= score_thresh).astype(np.float32)
        predicts = im_processing.resize_and_crop(pred_raw, mask.shape[0], mask.shape[1])
        pred_mask = predicts
        if dcrf:
            # Dense CRF post-processing
            sigm_val = np.squeeze(sigm_val)
            d = densecrf.DenseCRF2D(W, H, 2)
            U = np.expand_dims(-np.log(sigm_val), axis=0)
            U_ = np.expand_dims(-np.log(1 - sigm_val), axis=0)
            unary = np.concatenate((U_, U), axis=0)
            unary = unary.reshape((2, -1))
            d.setUnaryEnergy(unary)
            d.addPairwiseGaussian(sxy=3, compat=3)
            d.addPairwiseBilateral(sxy=20, srgb=3, rgbim=proc_im, compat=10)
            Q = d.inference(5)
            pred_raw_dcrf = np.argmax(Q, axis=0).reshape((H, W)).astype(np.float32)
            predicts_dcrf = im_processing.resize_and_crop(pred_raw_dcrf, mask.shape[0], mask.shape[1])
            pred_mask = predicts_dcrf

        if img_id not in predictions.keys():
            predictions[img_id] = dict()
        pred_mask = np.packbits(pred_mask.astype(np.bool))
        predictions[img_id][task_id] = {'pred_mask': pred_mask}
        print data_loader.img_idx, img_id, task_id

    if out_path is not None:
        print('lstm_refvg_predictor: saving predictions to %s ...' % out_path)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        fname = split
        if eval_img_count > 0:
            fname += '_%d' % eval_img_count
        fname += '.npy'
        f_path = os.path.join(out_path, fname)
        np.save(f_path, predictions)
    print('LSTM refvg predictor done!')
    return predictions


