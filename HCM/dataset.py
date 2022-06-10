import torch
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self,sequence, args,):
        self.args = args
        self.sequence = self.process_sequence(sequence)

    def uniq_words(self):
        return np.unique(self.sequence)


    def process_sequence(self,sequence):
        sequence = sequence.astype(int)
        sequence = sequence.flatten()
        seq = list(sequence)
        return seq

    def __len__(self):
        return len(self.sequence) - self.args.sequence_length # how long the sequence is with the sliding window

    def __getitem__(self, index): # a sliding window on sequence length
        return (
            torch.tensor(self.sequence[index:index+self.args.sequence_length]),
            torch.tensor(self.sequence[index+1:index+self.args.sequence_length+1]),
        )
