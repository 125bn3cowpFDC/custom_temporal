from torch.utils.data import Dataset, DataLoader
import json, os
import numpy as np
import torch
class SoobDataset(Dataset):
    def __init__(self, datapath, mode=None):
        #data load
        self.datapath = datapath 
        if mode=="train":
            self.data_path = datapath + '/hanchoom_train' #dir
            self.label_path = datapath + '/hanchoom_train_label.json' #file
        elif mode=="val":
            self.data_path = datapath + '/hanchoom_val' #dir
            self.label_path = datapath + '/hanchoom_val_label.json' #file
        elif mode=="test":
            self.data_path = 'C:/Users/godma/OneDrive/Desktop/project/handmade_choom4/data_hanchoom_final/hanchoom_test'
            self.label_path = 'C:/Users/godma/OneDrive/Desktop/project/handmade_choom4/data_hanchoom_final/hanchoom_test_label.json' #file
        self.data_names = os.listdir(self.data_path) #datafiles 숨쉬기~발사위 등

        with open(self.label_path) as f:
            label_info = json.load(f)

        data_id = [name.split('.')[0] for name in self.data_names] # ex) 'breathing_1
        self.label = np.array([label_info[id]['label_index'] for id in data_id]) # ex) 'breathing_1 -> 0

    
    def __len__(self):
        return len(self.data_names)
    
    def __getitem__(self, idx):
        data_name = self.data_names[idx]
        data_file_path = self.data_path+'/'+data_name

        with open(data_file_path, 'r') as f:
            data_info = json.load(f)

        data = np.zeros([100,18,2]) #feature (frame,joint,coordinate)
        for data_details in data_info['data']:
            frame_index = data_details['frame_index']-1  #1,2,3...100  index -> 0,1,2...99
            data[frame_index,:,0] = data_details['skeleton'][0]['pose'][0::2]
            data[frame_index,:,1] = data_details['skeleton'][0]['pose'][1::2]
        
        data =torch.from_numpy(data).float()

        label = data_info['label_index']
        assert (self.label[idx] == label)

        return data, label
    
if __name__ == "__main__":
    path = 'C:/Users/godma/OneDrive/Desktop/project/handmade_choom5/data_hanchoom_final'
    train_dataset = SoobDataset(datapath=path
                          , mode="train")
    
    train_dataloader = DataLoader(dataset=train_dataset,
                        batch_size=16,
                        shuffle=True,
                        drop_last=False,
                        num_workers=2)
    
    val_dataset = SoobDataset(datapath=path
                          , mode="val")
    
    val_dataloader = DataLoader(dataset=val_dataset,
                        batch_size=16,
                        shuffle=False,
                        drop_last=False,
                        num_workers=2)
    
    test_dataset = SoobDataset(datapath=path
                          , mode="test")
    
    test_dataloader = DataLoader(dataset=test_dataset,
                        batch_size=8,
                        shuffle=False,
                        drop_last=False,
                        num_workers=2)
    print(next(iter(test_dataloader)))

    for epoch in range(2):
        print(f"epoch : {epoch} ")
        for i, batch in enumerate(test_dataloader):
            data, label = batch
            print("afafffa",i)
            print(data.shape)
            