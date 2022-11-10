# Copyright (c)  Westone Information Industry Inc.
# --------------------------------------------------------
# References:
# deep-text-recognition-benchmark: https://github.com/clovaai/deep-text-recognition-benchmark
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import os
import sys
import re
import six
import lmdb
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms as transforms
from util.transforms_ABINet import CVColorJitter, CVDeterioration, CVGeometry


class Batch_Balanced_Dataset(object):

    def __init__(self, opt, num_tasks, global_rank):
        """
        Modulate the data ratio in the batch.
        For example, when select_data is "MJ-ST" and batch_ratio is "0.5-0.5",
        the 50% of the batch is filled with MJ and the other 50% of the batch is filled with ST.
        """
        if global_rank == 0:
            log = open(f'{opt.output_dir}/log_dataset.txt', 'a')
            dashed_line = '-' * 80
            print(dashed_line)
            log.write(dashed_line + '\n')
            print(
                f'dataset_root: {opt.data_path}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}')
            log.write(
                f'dataset_root: {opt.data_path}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}\n')

        opt.select_data = opt.select_data.split('-')
        opt.batch_ratio = opt.batch_ratio.split('-')
        assert len(opt.select_data) == len(opt.batch_ratio)
        self.opt = opt
        _AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, opt=opt)
        self.data_loader_list = []
        self.dataset_name_list = []
        self.dataloader_iter_list = []
        self.epoch_list = []
        batch_size_list = []
        Total_batch_size = 0
        for selected_d, batch_ratio_d in zip(opt.select_data, opt.batch_ratio):
            _batch_size = max(round(opt.batch_size * float(batch_ratio_d)), 1)
            _dataset, _dataset_log = hierarchical_dataset(root=opt.data_path, opt=opt, select_data=[
                                                          selected_d], data_filtering_off=opt.data_filtering_off)

            if global_rank == 0:
                print(dashed_line)
                log.write(dashed_line + '\n')
                log.write(_dataset_log)
            total_number_dataset = len(_dataset)

            """
            The total number of data can be modified with opt.total_data_usage_ratio.
            ex) opt.total_data_usage_ratio = 1 indicates 100% usage, and 0.2 indicates 20% usage.
            See 4.2 section in our paper.
            """
            """
            number_dataset = int(total_number_dataset * float(opt.total_data_usage_ratio))
            dataset_split = [number_dataset, total_number_dataset - number_dataset]
            indices = range(total_number_dataset)
            _dataset, _ = [Subset(_dataset, indices[offset - length:offset])
                           for offset, length in zip(_accumulate(dataset_split), dataset_split)]
            """
            if global_rank == 0:
                selected_d_log = f'num total samples of {selected_d}: {total_number_dataset} (total_data_usage_ratio) = {len(_dataset)}\n'
                selected_d_log += f'num samples of {selected_d} per batch: {opt.batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}'
                print(selected_d_log)
                log.write(selected_d_log + '\n')
            batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            sampler_train = torch.utils.data.DistributedSampler(
                _dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            print("Sampler_train = %s" % str(sampler_train))

            _data_loader = torch.utils.data.DataLoader(
                _dataset, sampler=sampler_train,
                batch_size=_batch_size,
                num_workers=opt.num_workers,
                collate_fn=_AlignCollate,
                pin_memory=opt.pin_mem,
                drop_last=True,
            )
            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))
            self.dataset_name_list.append(selected_d)
            self.epoch_list.append(0)

        if global_rank == 0:
            Total_batch_size_log = f'{dashed_line}\n'
            batch_size_sum = '+'.join(batch_size_list)
            Total_batch_size_log += f'Total_batch_size: {batch_size_sum} = {Total_batch_size}\n'
            Total_batch_size_log += f'{dashed_line}'

            print(Total_batch_size_log)
            log.write(Total_batch_size_log + '\n')
            log.close()
        opt.batch_size = Total_batch_size

    def get_batch(self):

        balanced_batch_images = []

        if self.opt.pretrain:
            balanced_batch_w_masks = []
            balanced_batch_texts = []

            for i, data_loader_iter in enumerate(self.dataloader_iter_list):
                try:
                    image, text, w_mask = data_loader_iter.next()
                    if self.dataset_name_list[i] == 'RealUnlabel' or self.dataset_name_list[i] == 'RealLabel':
                        unlabel_text = [
                            'unlabeleddata' for i in range(len(text))]
                        text = unlabel_text
                        w_mask = torch.ones_like(w_mask)

                    balanced_batch_images.append(image)
                    balanced_batch_w_masks.append(w_mask)
                    balanced_batch_texts += text

                except StopIteration:
                    self.data_loader_list[i].sampler.set_epoch(
                        self.epoch_list[i])
                    self.epoch_list[i] += 1
                    self.dataloader_iter_list[i] = iter(
                        self.data_loader_list[i])
                    image, text, w_mask = self.dataloader_iter_list[i].next()

                    if self.dataset_name_list[i] == 'RealUnlabel' or self.dataset_name_list[i] == 'RealLabel':
                        unlabel_text = [
                            'unlabeleddata' for i in range(len(text))]
                        text = unlabel_text
                        w_mask = torch.ones_like(w_mask)

                    balanced_batch_images.append(image)
                    balanced_batch_w_masks.append(w_mask)
                    balanced_batch_texts += text
                except ValueError:
                    pass

            balanced_batch_images = torch.cat(balanced_batch_images, 0)
            balanced_batch_w_masks = torch.cat(balanced_batch_w_masks, 0)

            return balanced_batch_images, balanced_batch_texts, balanced_batch_w_masks

        else:
            balanced_batch_texts = []

            for i, data_loader_iter in enumerate(self.dataloader_iter_list):
                try:
                    image, text, _ = data_loader_iter.next()
                    balanced_batch_images.append(image)
                    balanced_batch_texts += text
                except StopIteration:
                    self.data_loader_list[i].sampler.set_epoch(
                        self.epoch_list[i])
                    self.epoch_list[i] += 1
                    self.dataloader_iter_list[i] = iter(
                        self.data_loader_list[i])
                    image, text, _ = self.dataloader_iter_list[i].next()
                    balanced_batch_images.append(image)
                    balanced_batch_texts += text
                except ValueError:
                    pass

            balanced_batch_images = torch.cat(balanced_batch_images, 0)

            return balanced_batch_images, balanced_batch_texts


def hierarchical_dataset(root, opt, select_data='/', data_filtering_off=False, global_rank=0):
    """ select_data='/' contains all sub-directory of root directory """
    dataset_list = []
    if global_rank == 0:
        dataset_log = f'dataset_root:    {root}\t dataset: {select_data[0]}'
        print(dataset_log)
        dataset_log += '\n'
    else:
        dataset_log = None
    for dirpath, dirnames, filenames in os.walk(root+'/'):
        if not dirnames:
            select_flag = False
            for selected_d in select_data:
                if selected_d in dirpath.split('/')+['/']:
                    select_flag = True
                    break

            if select_flag:
                dataset = LmdbDataset(dirpath, opt, data_filtering_off)
                if global_rank == 0:
                    sub_dataset_log = f'sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}'
                    print(sub_dataset_log)
                    dataset_log += f'{sub_dataset_log}\n'
                dataset_list.append(dataset)

    concatenated_dataset = ConcatDataset(dataset_list)

    return concatenated_dataset, dataset_log


class LmdbDataset(Dataset):

    def __init__(self, root, opt, data_filtering_off=False):

        self.root = root
        self.env = lmdb.open(root, max_readers=32, readonly=True,
                             lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)
        self.is_eval = opt.eval
        self.max_length = opt.label_max_length
        self.sensitive = opt.sensitive
        self.data_filtering_off = data_filtering_off

        self.transform = DataAugment(opt, self.is_eval)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

            if self.data_filtering_off:
                # for fast check or benchmark evaluation with no filtering
                self.filtered_index_list = [
                    index + 1 for index in range(self.nSamples)]
            else:
                """ Filtering part
                If you want to evaluate IC15-2077 & CUTE datasets which have special character labels,
                use --data_filtering_off and only evaluate on alphabets and digits.
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L190-L192

                And if you want to evaluate them with the model trained with --sensitive option,
                use --sensitive and --data_filtering_off,
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/dff844874dbe9e0ec8c5a52a7bd08c7f20afe704/test.py#L137-L144
                """
                self.filtered_index_list = []
                for index in range(self.nSamples):
                    index += 1  # lmdb starts with 1
                    label_key = 'label-%09d'.encode() % index
                    label = txn.get(label_key).decode('utf-8')

                    if len(label) > self.max_length:
                        # print(f'The length of the label is longer than max_length: length
                        # {len(label)}, {label} in dataset {self.root}')
                        continue

                    # By default, images containing characters which are not in opt.character are filtered.
                    # You can add [UNK] token to `opt.character` in utils.py instead of this filtering.
                    '''
                    out_of_char = f'[^{self.opt.character}]'
                    if re.search(out_of_char, label.lower()):
                        continue
                    '''

                    self.filtered_index_list.append(index)

                self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('RGB')  # for color image

            except IOError:
                print(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                label = '[dummy_label]'

            img = self.transform(img)

            if not self.sensitive:
                label = label.lower()

            # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
            #out_of_char = f'[^{self.opt.character}]'
            #label = re.sub(out_of_char, '', label)
            label = re.sub('[^0-9a-zA-Z]+', '', label)

        return (img, label)


class DataAugment(object):
    '''
    Supports with and without data augmentation
    '''

    def __init__(self, opt, is_eval):

        self.opt = opt
        self.is_eval = is_eval

        if not is_eval:
            if self.opt.random_crop:
                self.random_crop = transforms.RandomApply([
                    transforms.RandomResizedCrop((self.opt.imgW, self.opt.imgH), scale=(0.85, 1.0), ratio=(3.5 / 1.0, 5 / 1.0))],
                    p=0.75)
            elif self.opt.abinet_augment:
                self.augment_abinet = transforms.Compose([
                    CVGeometry(degrees=45, translate=(0.0, 0.0), scale=(
                        0.5, 2.), shear=(45, 15), distortion=0.5, p=0.5),
                    CVDeterioration(var=20, degrees=6, factor=4, p=0.25),
                    CVColorJitter(brightness=0.5, contrast=0.5,
                                  saturation=0.5, hue=0.1, p=0.25)
                ])

    def __call__(self, img):

        img = img.resize((self.opt.imgW, self.opt.imgH), Image.BICUBIC)

        if not self.is_eval:

            if self.opt.random_crop:

                img = self.random_crop(img)
                img = img.resize((self.opt.imgW, self.opt.imgH), Image.BICUBIC)

            elif self.opt.abinet_augment:

                img = self.augment_abinet(img)
                img = img.resize((self.opt.imgW, self.opt.imgH), Image.BICUBIC)

        img = transforms.ToTensor()(img)
        img.sub_(0.5).div_(0.5)
        return img


class RandomWordMaskingGenerator:
    def __init__(self, mask_prob=0.2, max_length=27):

        self.mask_prob = mask_prob
        self.max_length = max_length

    def __call__(self, length):

        num_mask = int(
            # add a random number for probabilistic rounding
            self.mask_prob * length
            + np.random.rand()
        )
        mask_idc = np.random.choice(length, num_mask, replace=False)
        mask_idc = mask_idc[mask_idc < length]
        mask_idc = mask_idc + 1
        mask = np.zeros(self.max_length)
        mask[mask_idc] = 1
        return mask


class AlignCollate(object):

    def __init__(self, imgH=112, imgW=448, opt=None, is_eval=False):
        self.imgH = imgH
        self.imgW = imgW
        self.opt = opt
        self.is_eval = is_eval
        if not is_eval:
            self.word_masked_position_generator = RandomWordMaskingGenerator()

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        image_tensors = images
        image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        if self.is_eval:
            return image_tensors, labels, None
        else:
            word_mask = torch.cat([torch.from_numpy(self.word_masked_position_generator(
                len(label)+1)).unsqueeze(0) for label in labels], 0)
            return image_tensors, labels, word_mask
