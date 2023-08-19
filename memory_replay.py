import torch
import numpy as np
import random

def unbatch(half_batch):
    """
    Unbatches a batch into list of examples.

    Args:
        batch: A batch of examples with the structure :
        [torch.Tensor, torch.Tensor, torch.Tensor]

    Returns:
        list of unbatched examples: [[torch.Tensor, torch.Tensor, torch.Tensor], [torch.Tensor, torch.Tensor, torch.Tensor], [torch.Tensor, torch.Tensor, torch.Tensor]]

    """
    list_of_examples = []

    num_examples = len(half_batch[0])

    for idx in range(num_examples):
        list_of_examples.append([half_batch[0][idx], half_batch[1][idx], half_batch[2][idx]])

    return list_of_examples


def batch(list_of_examples):
    """
    Batches unbatched examples into one

    Args:
        list_of_examples: list of unbatched examples: [[torch.Tensor, torch.Tensor, torch.Tensor], [torch.Tensor, torch.Tensor, torch.Tensor], [torch.Tensor, torch.Tensor, torch.Tensor]]

    Returns:
        A batch of examples with the structure :
        [torch.Tensor, torch.Tensor, torch.Tensor]
    """
    img_feats = []
    q_feats = []
    labels = []
    for example in list_of_examples:
        img_feats.append(example[0])
        q_feats.append(example[1])
        labels.append(example[2])

    return torch.concat(img_feats), torch.concat(q_feats), torch.concat(labels)


def combine_batch_and_list(half_batch, list_of_examples):
    for example in list_of_examples:
        half_batch[0] = torch.concat([half_batch[0], example[0].unsqueeze(0)], dim=0)
        half_batch[1] = torch.concat([half_batch[1], example[1].unsqueeze(0)], dim=0)
        half_batch[2] = torch.concat([half_batch[2], example[2].unsqueeze(0)], dim=0)
    return half_batch

class ExperienceReplay:
    def __init__(self, samples_per_class=10, num_classes=20, half_batch_size=8):
        self.samples_per_class = samples_per_class
        self.num_classes = num_classes
        self.half_batch_size = half_batch_size

        self.memory_size = self.samples_per_class * self.num_classes

        self.memory = []

    def update_memory(self, current_batch, elapsed_examples=0):
        list_of_examples = unbatch(current_batch)

        counter = 0

        for example in list_of_examples:
            if len(self.memory) < self.memory_size:
                self.memory.append(example)
            else:
                idx = random.randint(0, elapsed_examples + counter)
                if idx < self.memory_size:
                    self.memory[idx] = example

            counter += 1
        return None

    def get_from_memory(self, num_samples):
        return random.choices(self.memory, k=num_samples)