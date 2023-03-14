from logging import getLogger
import numpy as np
import torch
import os
import pickle


from .dataset import ParallelDataset, Dataset


logger = getLogger()


def load_images(sentence_ids, feat_path, img_names, n_regions):
    img_scores, img_boxes, img_feats, img_labels = [], [], [], []

    has_vision_feats = [True for _ in sentence_ids]
    for i, idx in enumerate(sentence_ids):
        # Everything should be loadable. If features do not exist
        # use the dummy empty_feats.pkl
        f_name = os.path.join(feat_path, img_names[idx])
        if os.path.exists(f_name):
            with open(f_name, "rb") as f:
                x = pickle.load(f)
                assert len(x) != 0 and len(x["detection_scores"]) == 36

                # reduce to requested # of regions
                img_scores.append(x['detection_scores'][:n_regions].squeeze().astype(np.float32))
                img_boxes.append(x['detection_boxes'][:n_regions].squeeze().astype(np.float32))
                img_feats.append(x['detection_features'][:n_regions].squeeze().astype(np.float32))
                img_labels.append(x['detection_classes'][:n_regions].squeeze())
        else:
            # missing images; pad an empty feature
            img_scores.append(np.zeros([36, 1601], dtype=np.float32))
            img_boxes.append(np.zeros([36, 4], dtype=np.float32))
            img_feats.append(np.zeros([36, 2048], dtype=np.float32))
            img_labels.append(np.zeros([36], dtype=np.int64))
            has_vision_feats[i] = False


    # convert to numpy arrays
    # detection_scores is not used anywhere so we don't return it
    img_boxes = torch.from_numpy(
        np.array(img_boxes, dtype=img_boxes[0].dtype))
    img_feats = torch.from_numpy(
        np.array(img_feats, dtype=img_feats[0].dtype))
    img_labels = torch.from_numpy(
        np.array(img_labels, dtype='int64'))
    has_vision_feats = torch.from_numpy(
        np.array(has_vision_feats, dtype='bool'))

    return img_boxes, img_feats, img_labels, has_vision_feats


class DatasetWithRegions(Dataset):
    def __init__(self, sent, pos, image_names, params):
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.batch_size = params.batch_size
        self.tokens_per_batch = params.tokens_per_batch
        self.max_batch_size = params.max_batch_size

        self.sent = sent
        self.pos = pos
        self.lengths = self.pos[:, 1] - self.pos[:, 0]

        self.num_of_regions = params.num_of_regions
        self.region_features_path = params.region_feats_path
        self.image_names = np.array(image_names)

        # check number of sentences
        assert len(self.pos) == (self.sent == self.eos_index).sum()

        # Set RNG
        self._rng = np.random.RandomState(seed=params.iter_seed)

        # sanity checks
        self.check()

    def remove_empty_sentences(self):
        """
        Remove empty sentences.
        """
        init_size = len(self.pos)
        indices = np.arange(len(self.pos))
        indices = indices[self.lengths[indices] > 0]
        self.pos = self.pos[indices]
        self.lengths = self.pos[:, 1] - self.pos[:, 0]
        logger.info("Removed %i empty sentences." % (init_size - len(indices)))
        self.check()
        return indices

    def remove_long_sentences(self, max_len):
        indices = super().remove_long_sentences(max_len)
        self.image_names = self.image_names[indices]
        self.check()

    def select_data(self, a, b):
        super().select_data(a, b)
        self.image_names = self.image_names[a:b]

    def load_images(self, sentence_ids):
        return load_images(
            sentence_ids, self.region_features_path, self.image_names,
            self.num_of_regions)

    def get_batches_iterator(self, batches):
        for sentence_ids in batches:
            if 0 < self.max_batch_size < len(sentence_ids):
                self._rng.shuffle(sentence_ids)
                sentence_ids = sentence_ids[:self.max_batch_size]

            pos = self.pos[sentence_ids]
            sent = self.batch_sentences([self.sent[a:b] for a, b in pos])

            # Visual features dictionary
            img_boxes, img_feats, img_labels = self.load_images(sentence_ids)

            yield (sent, (img_boxes, img_feats, img_labels), sentence_ids)


class ParallelDatasetWithRegions(ParallelDataset):
    def __init__(self, sent1, pos1, sent2, pos2, image_names, params):
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.batch_size = params.batch_size
        self.tokens_per_batch = params.tokens_per_batch
        self.max_batch_size = params.max_batch_size
        self.sent1 = sent1
        self.sent2 = sent2
        self.pos1 = pos1
        self.pos2 = pos2
        self.image_names = np.array(image_names)
        self.region_features_path = params.region_feats_path
        self.num_of_regions = params.num_of_regions
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]

        # Set RNG
        self._rng = np.random.RandomState(seed=params.iter_seed)

        # sanity checks
        self.check()

    def remove_empty_sentences(self):
        """
        Remove empty sentences.
        """
        init_size = len(self.pos1)
        indices = np.arange(len(self.pos1))
        indices = indices[self.lengths1[indices] > 0]
        indices = indices[self.lengths2[indices] > 0]
        self.pos1 = self.pos1[indices]
        self.pos2 = self.pos2[indices]
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]
        logger.info("Removed %i empty sentences." % (init_size - len(indices)))
        self.check()
        return indices

    def remove_long_sentences(self, max_len):
        indices = super().remove_long_sentences(max_len)
        self.image_names = self.image_names[indices]

    def select_data(self, a, b):
        super().select_data(a, b)
        self.image_names = self.image_names[a:b]

    def load_images(self, sentence_ids):
        return load_images(
            sentence_ids, self.region_features_path, self.image_names,
            self.num_of_regions)

    def get_batches_iterator(self, batches):
        for sentence_ids in batches:
            if 0 < self.max_batch_size < len(sentence_ids):
                self._rng.shuffle(sentence_ids)
                sentence_ids = sentence_ids[:self.max_batch_size]

            # Textual stream
            pos1 = self.pos1[sentence_ids]
            pos2 = self.pos2[sentence_ids]
            sent1 = self.batch_sentences([self.sent1[a:b] for a, b in pos1])
            sent2 = self.batch_sentences([self.sent2[a:b] for a, b in pos2])

            # Visual features as separate tensors
            img_boxes, img_feats, img_labels, has_img_feats = self.load_images(sentence_ids)

            # remove samples having no image features
            if (~has_img_feats).sum() > 0:
                sent1 = (
                    sent1[0].transpose(0, 1)[has_img_feats].transpose(0,1),
                    sent1[1][has_img_feats]
                )
                sent2 = (
                    sent2[0].transpose(0, 1)[has_img_feats].transpose(0,1),
                    sent2[1][has_img_feats]
                )
                img_boxes = img_boxes[has_img_feats]
                img_feats = img_feats[has_img_feats]
                img_labels = img_labels[has_img_feats]
                sentence_ids = sentence_ids[has_img_feats]

                #TODO: shrink batch to fit the constrain between batch size and lengths

            yield (sent1, sent2, (img_boxes, img_feats, img_labels), sentence_ids)
