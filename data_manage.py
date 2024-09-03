import os
import pickle
from matplotlib import pyplot as plt
from client import Client
import torch.nn as nn
import copy
import csv
import numpy as np




class SharedClass:


    def save_backdoor_dataset(self ,clients ,helper,dir):

        if not os.path.exists(dir):
            os.makedirs(dir)

        for client in clients:
            client_id =client.id
            path =os.path.join(dir,f'{client_id}.pth')
            print(f'当前写入文件名称:{path}')
            with open(path ,'wb') as f:
                pickle.dump(client.poisoned_dataset,f)
            if helper.params['append_poisoned_data']:
                dataset_poison_type='replacement'
            else:
                dataset_poison_type='append'
            path_png=os.path.join(dir,f'{client_id}_{dataset_poison_type}')

            os.makedirs(path_png, exist_ok=True)
            self.save_dataset_as_png(client.poisoned_dataset,path_png)
            class_dict = helper.build_classes_dict(client.poisoned_dataset)
            target_label = helper.params['target_label']
            samples = helper.get_samples_by_label(client.poisoned_dataset, class_dict, target_label)

            print(f'目标标签为{target_label}的样本编号：{class_dict[target_label]}')
            print(f'当前保存的客户端{client.id}的标签为{target_label}的样本数目：{len(samples)}')



    def save_benign_dataset(self ,clients ,dir):

        if not os.path.exists(dir):
            os.makedirs(dir)

        for client in clients:
            client_id =client.id
            path =os.path.join(dir,f'{client_id}.pth')
            with open(path ,'wb') as f:
                pickle.dump(client.dataset ,f)
    def load_clients_data(self,helper,benign_dir, adversarial_dir):
        benign_clients_name_list = []
        adversaria_clients_name_list = []

        for file_name in os.listdir(adversarial_dir):
            if file_name.endswith('.pth'):
                client_id = int(file_name.replace('.pth', ''))
                adversaria_clients_name_list.append(client_id)

        clients = []

        criterion = nn.CrossEntropyLoss()
        for file_name in os.listdir(benign_dir):
            if file_name.endswith('.pth'):
                client_id = int(file_name.replace('.pth', ''))
                if client_id not in adversaria_clients_name_list:
                    benign_clients_name_list.append(client_id)

                benign_loader_path = os.path.join(benign_dir, file_name)
                with open(benign_loader_path, 'rb') as f:
                    dataset = pickle.load(f)

                client = Client(helper.global_model, dataset, criterion, helper, client_id)
                clients.append(client)
        for client_id in adversaria_clients_name_list:
            adversarial_loader_path = os.path.join(adversarial_dir, f'{client_id}.pth')

            with open(adversarial_loader_path, 'rb') as f:
                backdoor_dataset = pickle.load(f)

            backdoor_dataset.change_data_shape()
            print(f"样本形状：{backdoor_dataset[0][0].shape}，样本标签{backdoor_dataset[0][1]}")

            for client in clients:
                if client.id==client_id:
                    client.set_backdoor_dataset(backdoor_dataset,helper.params['batch_size'])


        return clients, benign_clients_name_list, adversaria_clients_name_list


    def save_dataset_as_png(self,dataset,save_path):
        for idx,(image,label) in enumerate(dataset):
            if image.ndimension()==2:
                cmap='gray'
                image_np = image.numpy()
            elif image.ndimension()==3:
                if image.shape[0]==1:
                    cmap='gray'
                    image_np = np.squeeze(image.numpy())

                elif image.shape[0]==3:
                    cmap=None
                    image_np=np.transpose(image.numpy(),(1,2,0))

            if image_np.max() > 1.0:
                image_np = image_np / 255.0
            image_path = os.path.join(save_path, f'image_{idx}_label_{label}.png')
            plt.imsave(image_path, image_np, cmap=cmap)


