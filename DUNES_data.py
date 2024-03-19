## Provides implementation of DUNES data set in PyTorch.
## Randall Reese. 
## Based on work from preamble_data.ipynb
## March 4, 2024.


'''
We need to be able to read in very large data sets that are several GBs in size a piece. We use an IterableDataset from Pytorch. 

A few things to note:
* As far as I understand, a PyTorch DataLoader on an IterableDataset calls the __iter__() function as many times is needed to build a batch the size of batch_size parameter.
* There are ways of having multiple workers in the DataLoader with an IterableDataset....it's complicated and as long as the data loading is not TOO slow it will be easiest to not have to implement this.
* To "shuffle" the data, just shuffle the index list. This should be the default unless explicitly set to False.
* To create a test and train set, split the index list and create two data *sets* explicitly. This means that we do not do the split after creating the full data set. 
'''

import sys
import random
import numpy as np
from pathlib import Path
from scipy.fft import fft, fftfreq, dct

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset


## -----------------------------------------------------------------------------
## Some macro definitions. 

dir_paths = {"bt": "Bluetooth",
              "ble": "Bluetooth_LE",
              "wlan": "WLAN",
              "whe":"WLAN_HE",
              "zig":"zigbee"}

data_paths = {"bt": "Bluetooth_BR_EDR_data_",
              "ble": "Bluetooth_LE_data_",
              "wlan": "WLAN_data_",
              "whe":"WLAN_HE_data_",
              "zig":"zigbee_data_"}

labels_to_int = {"bt": 0,
              "ble": 1,
              "wlan": 2,
              "whe":3,
              "zig":4}

labels_to_int_closed = {"bt": 0,
              "ble": 1,
              "wlan": 2,
              "whe":3}

int_to_labels=["bt", "ble", "wlan", "whe", "zig"]

int_to_labels_closed=["bt", "ble", "wlan", "whe"]

complex_dtype = np.complex128 


## -----------------------------------------------------------------------------

def get_start_indices(input_file, shuffle=True):
    # Read start and end indices from the input text file
    with open(input_file, 'r') as file:
        indices_data = [line.strip().split(',') for line in file]

    # Convert indices to integers
    start_indices = [int(start) for start, end in indices_data]

    ## Note that random.shuffle() shuffles in place. No return value. 
    if shuffle:
        random.shuffle(start_indices)
    
    return start_indices

## -----------------------------------------------------------------------------

def get_rand_data_order(num_classes=5, num_samp = 10000):

    # Initialize a list to store the sampled integers
    sampled_labels = []
    
    # Repeat the sampling process for each integer
    for i in range(num_classes):
        samples =  [int_to_labels[i]]*num_samp
        sampled_labels.extend(samples)
    
    # Shuffle the list to randomize the order. Remember that random.shuffle occurs in place and does not have a return value.
    random.shuffle(sampled_labels)
    return sampled_labels
'''
## This can act as a unit test for the reader to implement if desired. 
## Demonstration. For notes purposes only. 
# Print the first 10 sampled labels as an example
samp_order=get_rand_data_order()
print(samp_order[:20])

# Verify that each label occurs exactly 10000 times
print("Counts of each label:")
for i in int_to_labels:
    print(f"{i}: {samp_order.count(i)}")
'''


## -----------------------------------------------------------------------------
## chunk_sz is the packet header size we want to sample. 
## train_ratio is the amount of data to keep for training. 

def make_dataset(data_key="ble", snr=25, chunk_sz = 256, train_ratio=0.8, complex_dtype = np.complex128  ):

    base_path="/projects/dunes/large_data"
    main_file_path = Path(base_path) / dir_paths[data_key] / f"{data_paths[data_key]}{snr}dB_10000packets.dat"
    pos_file_path = Path(base_path) / dir_paths[data_key] / f"{data_paths[data_key]}pos_10000packets.txt"

    ## This open statement can be pushed to the LargeIterDataset class. 
    file = open(main_file_path, 'rb')

            
    ## Define start indices. These will be shuffled randomly by default.
    ## Split this list based on the proportion of train ratio.
    ## Obvioulsy if you just want a single dataset, use 1.0 as the ratio.
    start_indices = get_start_indices(pos_file_path)
    
    # Calculate the number of elements to select
    num_to_select = int(len(start_indices) * train_ratio)
    
    ## Shuffle the list and take the first num_to_select entries.
    ## This means that we do not need to shuffle the data again.
    ## Test data will also be shuffled, but that is fine. 
    random.shuffle(start_indices)
    train_idx = start_indices[:num_to_select]
    test_idx=start_indices[num_to_select:]
    
    
    iterds_train = LargeIterDataset(file, train_idx, data_type=complex_dtype, label = data_key, labels_dict = labels_to_int,chunk_size=chunk_sz)
    iterds_test = LargeIterDataset(file, test_idx, data_type=complex_dtype, label = data_key, labels_dict = labels_to_int,chunk_size=chunk_sz)
    #print("How many times is make_dataset() being called?")
    return iterds_train, iterds_test


## -----------------------------------------------------------------------------

## These data dictionaries must be created all at once since we do a shuffling of the data in make_dataset()
## The train and test data sets are complements of one another, so they need to be created off of the same shuffle. 
def make_data_dicts(snr, chunk_sz, train_ratio = 0.8, int_to_labels=int_to_labels, explain=True):
    train_data_dict={}
    test_data_dict={}

    for s in int_to_labels:
        train_ds,test_ds = make_dataset(data_key=s, snr=snr, chunk_sz=chunk_sz, train_ratio=train_ratio)
        train_data_dict[s] = train_ds
        test_data_dict[s] = test_ds
    
    if explain:    
        print("The data labels are as follows:\n\n\t0: Bluetooth\n\t1: Bluetooth LE\n\t2: WLAN\n\t3: WLAN HE\n\t4: Zigbee")
    
    return train_data_dict, test_data_dict

## -----------------------------------------------------------------------------

def batch_iter(data_iterator, batch_sz):
    batch = []
    for sample in data_iterator:
        batch.append(sample)
        if len(batch) == batch_sz:
            yield batch
            batch = []
    # If there are remaining samples, yield a final batch
    if batch:
        yield batch

## -----------------------------------------------------------------------------       
        
def rip_repack(batch, batch_sz):
    #print(f"This is the batch dimension: {len(batch)}")
    t_list=[]
    f_list=[]
    d_list=[]
    l_list=[]
    
    for d_0 in range(len(batch)):
        #print(f"We are printing through d_0 {d_0}")
        #print(f'Printing each piece of the batch {batch[d_0]}')
        
        #print(f'This is the time domain \n{batch[d_0][0]} of batch piece {d_0}')
        t_list.append(batch[d_0][0])
        
        #print(f'This is the freq domain \n{batch[d_0][1]} of batch piece {d_0}')
        f_list.append(batch[d_0][1])
        
        #print(f'This is the dct domain \n{batch[d_0][2]} of batch piece {d_0}')
        d_list.append(batch[d_0][2])
        
        #print(f'This is the label: {batch[d_0][3]} of batch piece {d_0}')
        l_list.append(batch[d_0][3])
        #print(l_list)
        
    t_batch = torch.stack(t_list)
    f_batch = torch.stack(f_list)
    d_batch = torch.stack(d_list)
    l_batch = torch.IntTensor(l_list)
    
    return t_batch, f_batch, d_batch, l_batch
        
## -----------------------------------------------------------------------------

## -----------------------------------------------------------------------------

## -----------------------------------------------------------------------------

## -----------------------------------------------------------------------------

## -----------------------------------------------------------------------------

## -----------------------------------------------------------------------------

## -----------------------------------------------------------------------------
class PacketIterator:
    def __init__(self,
                 data_bytes,
                 start_indices,
                 data_type,
                 label,
                 offset_mult=16,
                 chunk_size=256):
    
        self.data_bytes = data_bytes
        self.start_indices = start_indices
        self.chunk_size = chunk_size
        self.index = 0  # Index to keep track of the current position in start_indices
        self.dtype = data_type
        self.off_mult = offset_mult
        self.label = label
        ##print(f'Initiating a PacketIterator for {self.data_bytes}')
     
    def __iter__(self):
        return self
    
    
    def __transform__(self, complex_data, trans_type):
        
        ## Start out with a default assignment. 
        transform_result=complex_data
        
        if trans_type == 'fft':
            # Perform Fast Fourier Transform (FFT)
            transform_result = fft(complex_data)
        
        elif trans_type == 'dct':
            # Perform Discrete Cosine Transform (DCT)
            transform_result_real = dct(complex_data.real, type=2, norm='ortho')
            transform_result_imag = dct(complex_data.imag, type=2, norm='ortho')
            transform_result = np.vectorize(complex)(transform_result_real, transform_result_imag)
            
        elif trans_type=='time':##Do nothing. No need to even call this. Just here for completeness.
            transform_result=complex_data
            
        else:
            raise ValueError("Invalid transform type. Use 'fft' for FFT or 'dct' for DCT or 'time' for time domain.")
        
        return np.array([transform_result.real, transform_result.imag])
        
    def __collate__(self, complex_data):
        
        stack_data_time = self.__transform__(complex_data, trans_type='time')
        stack_data_freq = self.__transform__(complex_data, trans_type='fft')
        stack_data_dct = self.__transform__(complex_data, trans_type='dct')
        
        
        return stack_data_time, stack_data_freq, stack_data_dct
    
    def __reset__(self):
        #print(f"resetting {self} in PacketIterator")
        self.index = 0
        
    ## Unless otherwise needed, it is assumed the data is complex128. So offset_mult=16
    def __next__(self):
        if self.index < len(self.start_indices):
            start_index = self.start_indices[self.index]
            end_index = start_index + self.chunk_size
            # Move the file pointer to the desired position
            self.data_bytes.seek(start_index*self.off_mult)
    
            # Read 256 consecutive bytes from the stream
            chunk_data = self.data_bytes.read(self.chunk_size*self.off_mult)
            complex_data = np.frombuffer(chunk_data, dtype=complex_dtype)
         
            self.index += 1
            
            ## sd: stacked_data
            ## t,f,dct: time, frequency, dct.
            sd_t, sd_f, sd_dct = self.__collate__(complex_data)
           
            return torch.tensor(sd_t), torch.tensor(sd_f), torch.tensor(sd_dct), self.label
        
        else:
            raise StopIteration
            
## -----------------------------------------------------------------------------
## This is the Dataset class. No need to import this directly.  
class LargeIterDataset(IterableDataset):
    def __init__(self, data_bytes, start_indices, data_type, label, labels_dict, chunk_size=256):
        super(LargeIterDataset, self).__init__()
        self.data_bytes = data_bytes
        self.start_indices = start_indices
        self.chunk_size = chunk_size
        self.dtype=data_type
        self.int_label = labels_dict[label]
        self.iterator = PacketIterator(self.data_bytes, self.start_indices, self.dtype, self.int_label, chunk_size =self.chunk_size)
    
    def __reset__(self):
        #print(f"resetting in LargeIterDataset")
        self.iterator.__reset__()
        
    def __iter__(self):
        return self.iterator

## -----------------------------------------------------------------------------
## This is the iterator that you need to import if you only want one sample at a time. 
## Define these iterators by creating the necessary data dictionary. 
## Note that each class (protocol) will be represented equally in the train and test iterators respectively.
## This means that the training iterator will be balanced for all classes. Same with test iterator. 
class MacroIterator:
    def __init__(self,
                 data_dict,
                 num_samp=100):
        
        if num_samp > 10000:
            print("Max samples avaliable is 10000.")
            raise ValueError
            
        self.order=get_rand_data_order(num_samp=num_samp)
        self.data_dict = data_dict
        self.index = 0  # Index to keep track of the current position in order
        
        
    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.order):
            s=self.order[self.index]
            next_samp = self.data_dict[s].__iter__().__next__()
            self.index += 1
            ##print(f"Pulling data from {s}")
            return next_samp
        else:
            raise StopIteration    
            
## -----------------------------------------------------------------------------
##This is the iterator you want if you want to do batching.
## This returns the Batch.
## Do something where we go through and "reset" every data set each time a new batch iterator is constructed. 


class BatchIterator:
    def __init__(self, data_dict, batch_sz=16, num_samp=100):
        self.__reset__(data_dict)
        self.single_iter = MacroIterator(data_dict, num_samp)
        self.b_sz = batch_sz
        self.batch_iter = batch_iter(self.single_iter, self.b_sz)
        ##print(f"Creating a batch iterator with this many total samples {num_samp} and a batch size of {batch_sz}.")
    
    def __reset__(self, dd):
        ##lid is LargeIterDataset
        for _, lid in dd.items():
            #print(f"resetting {lid} in BatchIterator")
            lid.__reset__()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return rip_repack(next(self.batch_iter), self.b_sz)
        
## -----------------------------------------------------------------------------
class SimpleConvAE(nn.Module):
    def __init__(self):
        super(SimpleConvAE, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(4, 8, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Conv1d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            #nn.Flatten()
        )
        self.flatten = nn.Flatten()
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.ConvTranspose1d(8, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.ConvTranspose1d(4, 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() ## We use sigmoid activation to output values between 0 and 1
            #nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        ## We need to swap dimensions to match Conv1d input format:
            ## [batch_size, channels, length]
        #x = x.permute(1,0)
        if len(x.shape)<3:
            x = x.unsqueeze(dim=0)
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def features(self,x):
        x =self.encoder(x)
        x = self.flatten(x)
        return x
    
## -----------------------------------------------------------------------------

class SimpleLinearAE(nn.Module):
    def __init__(self, input_sz, factor=2):
        super(SimpleLinearAE, self).__init__()
        print(f"\n-----------------\nIt is assumed that input data has separated real and imaginary parts.\n")
        print(f"Data will be flattened to an input size of {2*input_sz} unless a value for factor arg has been set.\n-----------------")
        self.in_sz = factor*input_sz
        self.factor = factor
        
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.in_sz, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16)  ## Latent space size is 16
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.in_sz),
            nn.Sigmoid()  # Sigmoid activation to output values between 0 and 1
        )

    def forward(self, x):
       
        x = self.encoder(x)
        x = self.decoder(x)
        
        ## Reshape output to match original input size
        x = x.view(-1, 2, self.in_sz//self.factor)
        return x


## -----------------------------------------------------------------------------

## -----------------------------------------------------------------------------
    
## -----------------------------------------------------------------------------

## -----------------------------------------------------------------------------

## -----------------------------------------------------------------------------

## -----------------------------------------------------------------------------

## -----------------------------------------------------------------------------

## -----------------------------------------------------------------------------
    
    
    