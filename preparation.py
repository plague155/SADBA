import copy
import gc
import os
from collections import defaultdict
import random
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, ImageFolder,FashionMNIST,CIFAR10
from torch.utils.data import DataLoader, ConcatDataset, Subset, random_split, Dataset
from common import logger
import config
from helper import Helper, CustomDataset
from models.MNIST import SimpleNet
from config import device
import numpy as np

random.seed(1)

def convert_subset_to_custom_dataset(subset, original_dataset):
    indices = subset.indices
    data_list = []
    labels_list = []
    for idx in indices:
        data, label = original_dataset[idx]
        data_list.append(data)
        labels_list.append(label)

    data_tensor = torch.stack(data_list)
    labels_tensor = torch.tensor(labels_list)

    return CustomDataset(data_tensor, labels_tensor)

class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train

        # 读取标签
        self.label_map = self.create_label_map()

        if is_train:
            self.image_paths, self.labels = self.load_train_data()
        else:
            self.image_paths, self.labels = self.load_val_data()

    def create_label_map(self):
        classes = [d.name for d in os.scandir(os.path.join(self.root_dir, 'train')) if d.is_dir()]
        return {cls: idx for idx, cls in enumerate(classes)}

    def load_train_data(self):
        image_paths = []
        labels = []
        for cls, label in self.label_map.items():
            cls_dir = os.path.join(self.root_dir, 'train', cls, 'images')
            for img_name in os.listdir(cls_dir):
                if img_name.endswith('.JPEG'):
                    image_paths.append(os.path.join(cls_dir, img_name))
                    labels.append(label)
        return image_paths, labels

    def load_val_data(self):
        image_paths = []
        labels = []
        val_annotations_path = os.path.join(self.root_dir, 'val', 'val_annotations.txt')
        with open(val_annotations_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                img_name = parts[0]
                cls = parts[1]
                if cls in self.label_map:
                    image_paths.append(os.path.join(self.root_dir, 'val', 'images', img_name))
                    labels.append(self.label_map[cls])
        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
class ImageHelper(Helper):

    def load_test_data(self):
        transform = transforms.Compose([transforms.ToTensor()])

        if self.params['type'] == config.TYPE_MNIST:
            test_dataset = MNIST(root='./data', train=False, transform=transform, download=True)

        elif self.params['type'] == config.TYPE_CIFAR10:
            test_dataset = CIFAR10(root='./data', train=False, transform=transform, download=True)

        elif self.params['type'] == config.TYPE_FashionMNIST:
            test_dataset = FashionMNIST(root='./data', train=False, transform=transform, download=True)

        test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

        self.check_data(test_loader)
        self.test_data = test_loader
        self.info(test_dataset)
        self.m = self.params['trigger_pattern']  # trigger pattern
        setting = self.params['setting']
        position = self.params['test_pos']
        global_trigger = self.global_trigger_generate(setting)
        test_indices = self.params['test_trigger_indices']
        trigger = self.get_trigger_by_indices(global_trigger, test_indices)

        # for sub_trigger in trigger:
        #     print(sub_trigger)
        datas = []
        labels = []
        count = 0
        for batch_data, batch_labels in test_loader:
            for i in range(len(batch_data)):
                data = batch_data[i]

                poison_data = self.insert_several_triggers(data, trigger, position)
                datas.append(poison_data)
                labels.append(self.params['target_label'])

                count += 1
        test_poisoned_dataset = CustomDataset(datas, labels)
        self.test_data_poison = DataLoader(dataset=test_poisoned_dataset, batch_size=64, shuffle=False)
        del datas, labels
        gc.collect()

    def check_data(self, loader):
        for inputs, labels in loader:
            if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                print("Data contains NaN or Inf!")
                break


    def latin_hypercube_sampling(self,combinations, T):
        """
        Input:
            Number of participants and alpha (param for distribution)
        Output:
            A list of indices denoting data in training set.
        Parameters:
            combinations (list): List of all possible combinations.
            T (int): The number of combinations to be selected.
        Returns:
            selected_combinations (list): A list containing the selected combinations based on the LHS method.
        """
        num_combinations = len(combinations)
        indices = np.linspace(0, num_combinations - 1, T, dtype=int)
        selected_combinations = [combinations[i] for i in indices]
        return selected_combinations


    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        """
            Input: Number of participants and alpha (param for distribution)
            Output: A list of indices denoting data in CIFAR training set.
            Requires: cifar_classes, a preprocessed class-indice dictionary.
            Sample Method: take a uniformly sampled 10-dimension vector as parameters for
            dirichlet distribution to sample number of images in each class.
        """

        classes = self.classes_dict
        class_size = len(classes[0])
        per_participant_list = defaultdict(list)
        no_classes = len(classes.keys())

        image_nums = []
        for n in range(no_classes):
            image_num = []
            random.shuffle(classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = classes[n][:min(len(classes[n]), no_imgs)]
                image_num.append(len(sampled_list))
                per_participant_list[user].extend(sampled_list)
                classes[n] = classes[n][min(len(classes[n]), no_imgs):]
            image_nums.append(image_num)

        min_samples=510 if self.params['trigger_pattern']==9 else 14 if self.params['trigger_pattern']==4 else 0
        for user, samples in per_participant_list.items():
            if len(samples) < min_samples:
                deficit = min_samples - len(samples)
                for other_user, other_samples in per_participant_list.items():
                    if len(other_samples) > min_samples + deficit:
                        per_participant_list[user].extend(other_samples[:deficit])
                        del per_participant_list[other_user][:deficit]
                        break
        return per_participant_list


    def load_data(self):
        logger.info('Loading data')

        # datapath='./data'
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        if self.params['type'] == config.TYPE_MNIST:
            train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)

        elif self.params['type'] == config.TYPE_FashionMNIST:
            print("正在导入Fashion——MNIST数据集")
            train_dataset=FashionMNIST(root='./data', train=True, transform=transform, download=True)

        elif self.params['type'] == config.TYPE_CIFAR10:
            train_dataset = CIFAR10(root='./data', train=True, transform=transform, download=True)


        self.info(train_dataset)
        self.target_label=self.params['target_label']
        print(f"Target label:{self.target_label}")

        train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)

        if self.params['debug']:
            num_temp_samples = self.params['num_temp_samples']
        else:
            num_temp_samples = len(train_loader)
        count = 0
        all_data = []
        all_labels = []
        for data, label in train_loader:
            if count < num_temp_samples:
                all_data.append(data)
                all_labels.append(label)
                count += 1
            else:
                break

        all_data = torch.cat(all_data).squeeze()
        all_label = torch.cat(all_labels).squeeze()
        train_shuffle_dataset = self.initialize_dataset(all_data, all_label)
        print("打乱后的数据集大小:",len(train_shuffle_dataset))
        if self.params['sampling_dirichlet']:

            self.classes_dict=self.build_classes_dict(train_dataset)
            indices_per_client = self.sample_dirichlet_train_data(self.params['total_clients'], self.params['dirichlet_alpha'])

            client_datasets = []

            for client_id, (client_id_actual, indices) in enumerate(indices_per_client.items()):
                print(f"客户端{client_id}的所含样本数目为{len(indices)}")
                client_data = [train_dataset[i][0] for i in indices]#获取该客户端的数据样本
                client_labels = [train_dataset[i][1] for i in indices]
                client_datasets.append(CustomDataset(client_data, client_labels))


        else:
            num_clients = self.params['total_clients']
            client_datasets = []
            total_size = len(train_shuffle_dataset)
            client_size = total_size // num_clients
            remainder = total_size % num_clients

            lengths = [client_size] * num_clients
            for i in range(remainder):
                lengths[i] += 1

            dataset_split = random_split(train_shuffle_dataset, lengths)

            for dataset in dataset_split:
                custom_dataset = convert_subset_to_custom_dataset(dataset,train_shuffle_dataset)
                client_datasets.append(custom_dataset)

        print("train loaders done")
        print('train_shuffle_dataset:')
        print(self.info(train_shuffle_dataset))
        self.train_data=client_datasets
        self.shuffle_train_dataset=train_shuffle_dataset

        if self.params['is_random_adversary']:
            temp_num_adversarial=self.params['MPC'] * self.params['total_clients']
            if temp_num_adversarial<1 and temp_num_adversarial>0:
                temp_num_adversarial=1
            num_adversarial = int(temp_num_adversarial)

            self.adversarial_namelist = random.sample(range(self.params['total_clients']), num_adversarial)
        else:
            self.adversarial_namelist=self.params['total_list']
        print('恶意客户端为：',self.adversarial_namelist)
        if self.params['is_random_namelist'] == False:
            self.participants_list = self.params['participants_namelist']
        else:
            self.participants_list = list(range(self.params['total_clients']))
        self.benign_namelist=list(set(self.participants_list) - set(self.adversarial_namelist))


    def initialize_name_list(self,adversarial_clients_name_list,benign_clients_name_list):
        self.adversarial_namelist=adversarial_clients_name_list
        self.benign_namelist=benign_clients_name_list
        self.total_namelist=sorted(adversarial_clients_name_list+benign_clients_name_list)


    def build_classes_dict(self,dataset):
        classes = {}
        for ind, x in enumerate(dataset):
            _, label = x
            if isinstance(label,int):
                label=label
            elif isinstance(label,torch.Tensor):
                label=label.item()
            else:
                print("labels数据类型有误！")
            if label in classes:
                classes[label].append(ind)
            else:
                classes[label] = [ind]
        return classes

    def get_samples_by_label(self,dataset, class_dict, label):
        indices = class_dict.get(label, [])
        samples = [dataset[ind] for ind in indices]
        return samples

    def poison_dataset(self,clients):
        print('Start poison data...')

        appendix_dataset=ConcatDataset(self.train_data[i] for i in self.benign_namelist)
        adversarial_clients_dataset=ConcatDataset(client.dataset for client in clients)
        total_dataset=ConcatDataset([appendix_dataset,adversarial_clients_dataset])

        print('total_dataset:')
        self.info(total_dataset)
        total_classes=self.build_classes_dict(total_dataset)


        for label,ind in total_classes.items():
            if label==self.target_label:
                target_samples_ind=total_classes[label]
                print(f'Label {self.target_label} 的数目:{len(target_samples_ind)}')
                break

        target_samples=[total_dataset[i] for i in target_samples_ind]

        self.m = self.params['trigger_pattern']  # trigger pattern
        setting=self.params['setting']
        position=[0,0]
        global_trigger=self.global_trigger_generate(setting)
        print(f'global_trigger形状：')
        for shape in global_trigger:
            print(f'{shape}')

        if self.params['best_trigger_position']:
            # image=self.insert_several_triggers(total_dataset[0][0],global_trigger,position)
            # self.img_show(image)
            # best_pos=self.best_trigger_position(total_dataset[0],target_samples,global_trigger,setting)
            # image = self.insert_several_triggers(total_dataset[0][0], global_trigger, best_pos)
            # self.img_show(image)
            # image = self.insert_several_triggers(total_dataset[0][0], global_trigger_four, best_pos)
            # self.img_show(image)

            best_trigger_position=[]
            poison_dataset = []
            indices=list(range(len(total_dataset)))
            temp_data= Subset(total_dataset, indices)
            temp_data_indices = temp_data.indices

            with torch.no_grad():
                if self.params['unify_trigger_pos']:

                    temp_best_position=self.best_unify_trigger_position(temp_data,target_samples,global_trigger,setting)
                    print("总最佳植入位置为：",temp_best_position)
                    # batch_size = 100
                    #
                    # num_batches = len(temp_data) // batch_size + (1 if len(temp_data) % batch_size != 0 else 0)
                    #
                    # for batch_idx in range(num_batches):
                    #     batch_start = batch_idx * batch_size
                    #     batch_end = min((batch_idx + 1) * batch_size, len(temp_data))
                    #     batch_indices = temp_data_indices[batch_start:batch_end]
                    #     batch_data=[total_dataset_copy[idx] for idx in batch_indices]
                    #     for sample in batch_data:
                    #         best_trigger_position.append(temp_best_position)
                    #         sample_triggerd = self.insert_several_triggers_reducec_memory(sample[0], global_trigger,
                    #                                                        temp_best_position)
                    #         poison_dataset.append((sample_triggerd.cpu(), self.target_label))
                    #         del sample_triggerd
                    #         gc.collect()
                    #
                    #     del batch_data,batch_indices
                    #     gc.collect()
                    #     torch.cuda.empty_cache()

                    count=0
                    for sample in temp_data:
                        # data=data.to(device)
                        # if len(data.shape)==3:
                        #     data=data.unsqueeze(1)
                        best_trigger_position.append(temp_best_position)
                        sample_triggerd=self.insert_several_triggers(sample[0],global_trigger,temp_best_position)
                        poison_dataset.append((sample_triggerd.cpu(), self.target_label))
                        # if count % 6000 ==0:
                        #     del sample_triggerd
                        #     gc.collect()
                        count+=1

                        # del trigger_applied
                        # gc.collect()

                        # for i in range(len(data)):
                        #     sample_data = data[i]
                        #     best_trigger_position.append(temp_best_position)
                        #     trigger_applied = self.insert_several_triggers_into_batch_samples(sample_data, global_trigger, temp_best_position)
                        #     poison_dataset.append((trigger_applied.cpu(), self.target_label))


                else:
                    for data,labels in temp_data:
                        data = data.to(device)
                        labels = labels.to(device)
                        for i in range(len(data)):
                            sample_data=data[i]
                            sample_label=labels[i]
                            temp_best_position = self.best_trigger_position([sample_data,sample_label], target_samples, global_trigger,
                                                                            setting)
                            best_trigger_position.append(temp_best_position)

                            trigger_applied = self.insert_several_triggers(sample_data, global_trigger, temp_best_position)
                            poison_dataset.append((trigger_applied.cpu(), self.target_label))

            # for i in range(1):
            #     self.img_show(total_dataset[i][0])
            #     self.img_show(subset[i][0])
            #     self.img_show(self.insert_several_triggers(total_dataset[i][0],global_trigger,best_trigger_position[i]))
        else:
            best_trigger_position = [[0,0] for _ in range(len(total_dataset))]
            temp_best_position=[0,0]
            poison_dataset = []
            indices = list(range(len(total_dataset)))

            # temp_data_loader=DataLoader(Subset(total_dataset, indices),batch_size=1,shuffle=False)
            # with torch.no_grad():
            #
            #     for data,labels in temp_data_loader:
            #         # data = data.to(device)
            #         # labels = labels.to(device)
            #         for i in range(len(data)):
            #             sample_data=data[i]
            #             trigger_applied = self.insert_several_triggers(sample_data, global_trigger, [0, 0])
            #             poison_dataset.append((trigger_applied.cpu(), self.target_label))

            temp_data_loader = DataLoader(Subset(total_dataset, indices), batch_size=1, shuffle=False)
            with torch.no_grad():
                for data, labels in temp_data_loader:
                    data = data.to(device)
                    labels = labels.to(device)
                    sample_data = data[0]
                    trigger_applied = self.insert_several_triggers(sample_data, global_trigger, [0, 0])
                    poison_dataset.append((trigger_applied.cpu(), self.target_label))
        #optimal sample selection
        poison_rate=self.params['poison_rate']
        adversarial_total_size=sum(len(client.dataset) for client in clients)
        R=int(adversarial_total_size*poison_rate)
        print(f'total_samples_size:{len(total_dataset)};total_adversarial_size:{adversarial_total_size};R:{R}')
        if self.params['optimal_sample_selection']:
            optimal_backdoor_index=self.optimal_sample_selection(total_dataset,poison_dataset,R)
            print(optimal_backdoor_index)
        else:
            optimal_backdoor_index = random.sample(range(len(total_dataset)),R)
            print(optimal_backdoor_index)

        backdoor_parts = []

        self.M=len(self.adversarial_namelist)

        parts=self.split_list(optimal_backdoor_index,self.M)
        if self.params['full_combination_trigger']:
            if self.params['SADBA']:
                full_index=self.generate_full_combination_trigger(list(range(len(global_trigger))))
                T=self.params['MPC']*self.params['total_clients']
                index=self.latin_hypercube_sampling(full_index,int(T))
                print(f'local_trigger类型：SADBA;m:{self.m};排列顺序：{index}')
            else:
                index = self.generate_full_combination_trigger(list(range(len(global_trigger))))
                print(f'local_trigger类型:full_combination_trigger;m:{self.m};排列顺序：{index}')
        else:
            index = list(range(len(global_trigger)))
            print(f'local_trigger类型:dba;m:{self.m};排列顺序：{index}')

        append_poisoned_data=self.params['append_poisoned_data']
        local_triggers = self.generate_local_trigger(global_trigger, index)
        # local_triggers = [item for sublist in local_triggers for item in sublist]
        for trigger in local_triggers:
            print(trigger)



        count=0
        for i in range(len(parts)):

            print(f'part{i+1}:待植入恶意客户端序号：{clients[i].id};植入trigger类型:{local_triggers[i]}')

            backdoor_samples=self.get_samples(parts[i],total_dataset,best_trigger_position)


            if self.params['trigger_distribution_strategy']==1:
                # print(index)
                uniformly_poisoned_samples=self.trigger_uniformly_distribution(backdoor_samples,global_trigger,index)
                if count==0 and self.params['show_first_malicious_client_backdoor_samples']:
                    temp_samples=[]
                    for j in parts[count]:
                        temp_samples.append(total_dataset[j])
                    self.show_samples_by_batch(temp_samples)
                    self.show_samples_by_batch(uniformly_poisoned_samples)
                    count+=1
                clients[i].get_backdoor_dataset(uniformly_poisoned_samples,append_poisoned_data)



            elif self.params['trigger_distribution_strategy']==2:
                # print("1-to-1 mapping")
                static_poisoned_samples=self.trigger_static_distribution(backdoor_samples,local_triggers[i])
                # self.show_samples_by_batch(static_poisoned_samples)
                clients[i].get_backdoor_dataset(static_poisoned_samples,append_poisoned_data)
                count+=1

            else:
                print('请输入正确trigger植入类型')


        del best_trigger_position
        return temp_best_position





