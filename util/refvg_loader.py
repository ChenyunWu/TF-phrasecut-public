import numpy as np
import skimage

from _dataset.utils.word_embed import WordEmbed
from _dataset.utils.refvg_loader import RefVGLoader
from _dataset.utils.data_transfer import polygons_to_mask

from _rmi.util import im_processing, text_processing


class RMIRefVGLoader(RefVGLoader):
    def __init__(self, split=None, input_W=320, input_H=320, phrase_len=20):
        RefVGLoader.__init__(self, split=split)
        self.input_W = input_W
        self.input_H = input_H
        self.phrase_len = phrase_len
        word_embed = WordEmbed()
        self.vocab_dict = word_embed.word_to_ix
        for k, v in self.vocab_dict.items():
            if k.lower() != k:
                self.vocab_dict[k.lower()] = v

        self.img_idx = 0
        self.img_task_idx = 0
        self.is_end = False
        return

    def get_img_data(self, rand=True, is_train=True):
        if rand:
            img_id = np.random.choice(self.ref_img_ids)
            ref_data = self.get_img_ref_data(img_id)
            task_idx = np.random.choice(range(len(ref_data['phrases'])))
            task_id = ref_data['task_ids'][task_idx]
        else:
            img_id = self.ref_img_ids[self.img_idx]
            ref_data = self.get_img_ref_data(img_id)
            task_idx = self.img_task_idx
            task_id = ref_data['task_ids'][task_idx]
            if task_idx == len(ref_data['phrases']) - 1:
                self.img_idx += 1
                self.img_task_idx = 0
                if self.img_idx == len(self.ref_img_ids):
                    self.is_end = True
            else:
                self.img_task_idx += 1

        img = skimage.io.imread('data/refvg/images/%d.jpg' % img_id)
        phrase = ref_data['phrases'][task_idx]
        Polygons = ref_data['gt_Polygons'][task_idx]
        polygons = []
        for ps in Polygons:
            polygons += ps
        mask = polygons_to_mask(polygons, ref_data['width'], ref_data['height'])

        if is_train:
            img = skimage.img_as_ubyte(im_processing.resize_and_pad(img, self.input_H, self.input_W))
            mask = im_processing.resize_and_pad(mask, self.input_H, self.input_W)
        if img.ndim == 2:
            img = np.tile(img[:, :, np.newaxis], (1, 1, 3))

        phrase_encoded = text_processing.preprocess_sentence(phrase, self.vocab_dict, self.phrase_len)

        return img_id, task_id, img, mask, phrase, phrase_encoded

