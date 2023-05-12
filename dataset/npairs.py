from itertools import chain
import torch
import math
import random


class BaseSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, sz_batch, sampler, **sampler_args):
        self.sampler = sampler
        self.dataset = dataset
        self.sz_batch = sz_batch
        self.sampler_args = sampler_args
    def __iter__(self):
        yield from chain.from_iterable(
            self.sampler(self.dataset, self.sz_batch, **self.sampler_args)
        )
    def __len__(self):
        return len(self.dataset.ys)


class NPairs(BaseSampler):
    def __init__(self, dataset, batch_size, num_samples_per_class = 4):
        BaseSampler.__init__(self, dataset, batch_size, _npairs, K = num_samples_per_class)
        print(f'Npairs sz_batch={batch_size}, k={num_samples_per_class}')


def index_dataset(dataset):
    return {
        c : [
            example_idx for example_idx, (
                image_file_name, class_label_ind
            ) in enumerate(
                zip(dataset.im_paths, dataset.ys)
            ) if class_label_ind == c
        ] for c in set(dataset.ys)
    }


def sample_from_class(images_by_class, class_label_ind):
    return images_by_class[class_label_ind][
        random.randrange(len(images_by_class[class_label_ind]))
    ]


def _npairs(dataset, sz_batch, K = 4):
    images_by_class = index_dataset(dataset)
    for _ in range(
        int(
            math.ceil(
                len(dataset) * 1.0 / sz_batch
            )
        )
    ):
        example_indices = [
            sample_from_class(
                images_by_class,
                class_label_ind
            ) for k in range(
                int(
                    math.ceil(
                        sz_batch * 1.0 / K
                    )
                )
            ) for class_label_ind in [
                random.choice(
                    list(images_by_class.keys()))
            ] for i in range(K)
        ]
        yield example_indices[:sz_batch]


