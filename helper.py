import copy
import gc
import math
import os
import random
import torch.nn as nn
import numpy as np
import torch
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
import torchvision.models as models
from models.MNIST import SimpleNet
# from models.
import torch.optim as optim
import config
from config import device
from itertools import combinations
from shutil import copyfile
import torch.nn.functional as F


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def change_data_shape(self):
        new_data = []
        for data in self.data:
            if data.dim() == 2:
                data = data.unsqueeze(0)
            new_data.append(data)
        self.data = torch.stack(new_data)

    def add_data(self, new_samples):
        new_datas=[sample[0] for sample in new_samples]
        new_labels=[sample[1] for sample in new_samples]
        if isinstance(self.data,list) and isinstance(self.labels,list):

            self.data=self.data+new_datas
            self.labels=self.labels+new_labels
        elif isinstance(self.data,torch.Tensor) and isinstance(self.labels,torch.Tensor):
            tensor_datas=torch.stack([data.clone().detach() for data in new_datas])
            tensor_labels=torch.tensor(new_labels)
            self.data=torch.cat((self.data,tensor_datas),dim=0)
            self.labels = torch.cat((self.labels, tensor_labels), dim=0)
        else:
            print("自定义数据集对象类型错误！")

    def replace_data(self, indices,new_samples):
        for idx, (new_data, new_label) in zip(indices, new_samples):
            self.data[idx] = new_data
            self.labels[idx] = new_label

    def sample_replacement(self,idx,sample):
        data,label=sample
        self.data[idx]=data
        self.labels[idx]=label

    def copy(self):
        if isinstance(self.data,torch.Tensor) and isinstance(self.labels,torch.Tensor):
            copied_data = self.data.clone()
            copied_labels = self.labels.clone()
        elif isinstance(self.data,list) and isinstance(self.labels,list):
            copied_data = copy.deepcopy(self.data)
            copied_labels = copy.deepcopy(self.labels)
        else:
            print("self.data or self.labels type error!")

        return CustomDataset(copied_data, copied_labels)

class PoissonDiskSampling:
    def __init__(self, width, height, min_dist, new_points_count=30):
        self.width = width
        self.height = height
        self.min_dist = min_dist
        self.cell_size = min_dist / math.sqrt(2)
        self.grid_width = int(math.ceil(width / self.cell_size))
        self.grid_height = int(math.ceil(height / self.cell_size))
        self.grid = [[None for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        self.active_list = []
        self.points = []
        self.new_points_count = new_points_count

    def generate_points(self):
        initial_point = (random.uniform(0, self.width), random.uniform(0, self.height))
        self.points.append(initial_point)
        self.active_list.append(initial_point)
        grid_x = int(initial_point[0] / self.cell_size)
        grid_y = int(initial_point[1] / self.cell_size)
        self.grid[grid_y][grid_x] = initial_point

        while self.active_list:
            idx = random.randint(0, len(self.active_list) - 1)
            point = self.active_list[idx]
            found = False
            for _ in range(self.new_points_count):
                new_point = self._generate_random_point_around(point)
                if self._is_valid_point(new_point):
                    self.points.append(new_point)
                    self.active_list.append(new_point)
                    grid_x = int(new_point[0] / self.cell_size)
                    grid_y = int(new_point[1] / self.cell_size)
                    self.grid[grid_y][grid_x] = new_point
                    found = True
                    break
            if not found:
                self.active_list.pop(idx)
        return self.points

    def _generate_random_point_around(self, point):
        r = random.uniform(self.min_dist, 2 * self.min_dist)
        theta = random.uniform(0, 2 * math.pi)
        new_x = point[0] + r * math.cos(theta)
        new_y = point[1] + r * math.sin(theta)
        return (new_x, new_y)

    def _is_valid_point(self, point):
        if not (0 <= point[0] < self.width and 0 <= point[1] < self.height):
            return False
        grid_x = int(point[0] / self.cell_size)
        grid_y = int(point[1] / self.cell_size)
        for i in range(max(0, grid_x - 2), min(self.grid_width, grid_x + 3)):
            for j in range(max(0, grid_y - 2), min(self.grid_height, grid_y + 3)):
                neighbor = self.grid[j][i]
                if neighbor is not None:
                    dist = math.sqrt((point[0] - neighbor[0]) ** 2 + (point[1] - neighbor[1]) ** 2)
                    if dist < self.min_dist:
                        return False
        return True

class Helper:
    def __init__(self,current_time,params,name):
        self.current_time=current_time
        self.global_model=None
        self.local_model=None
        self.train_data=None
        self.test_data=None
        self.poisoned_data=None
        self.params=params
        self.name=name
        self.best_loss = math.inf
        self.color_map = {
            0: [0.8, 0.8, 0],  # yellow
            1: [1, 0, 0],  # red
            2: [0, 1, 0],  # green
            3: [0, 0, 1],  # blue
            4: [1, 0.5, 0],  # orange
            5: [0.5, 0, 0.5],  # purple
            6: [0, 1, 1],  # cyan
            7: [1, 0, 1],  # magenta
            8: [0.5, 0.5, 0.5],  # grey
            9: [0.8, 0.8, 0],  # faint yellow

        }

    def save_model(self,attack_type, model, epoch=0, val_loss=0):
        if self.params['save_model']:
            # save_model
            print("saving model")
            folder_path = os.path.join(self.params['folder_path'], f'{attack_type}')
            model_name = os.path.join(folder_path, 'model_last.pt.tar')
            saved_dict = {'state_dict': model.state_dict(), 'epoch': epoch,
                          'lr': self.params['lr']}
            self.save_checkpoint(saved_dict, False, model_name)
            if epoch in self.params['save_on_epochs']:
                print(f'Saving model on epoch {epoch}')
                self.save_checkpoint(saved_dict, False, filename=f'{model_name}.epoch_{epoch}')
            if val_loss < self.best_loss:
                self.save_checkpoint(saved_dict, False, f'{model_name}.best')
                self.best_loss = val_loss

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if not self.params['save_model']:
            return False
        torch.save(state, filename)

        if is_best:
            copyfile(filename, 'model_best.pth.tar')

    def get_trigger_by_indices(self,global_trigger,indices):
        trigger=[]
        for i in indices:
            trigger.append(global_trigger[i])
        return trigger

    def get_all_data_from_dataset(self,dataset):
        return [dataset[i][0] for i in range(len(dataset))]

    def get_all_sample_from_dataset(self,dataset):
        return [dataset[i] for i in range(len(dataset))]

    def initialize_adversarial_clients(self,clients,adversarial_clients_name_list):
        adversarial_clients = []
        count = 0
        for i in adversarial_clients_name_list:
            adversarial_client = self.get_client_by_id(clients, i)
            # 初始化恶意客户端的poison_epoch
            poison_epoch_list = self.params[str(count) + '_poison_epochs']
            adversarial_client.set_poison_epoch_list(poison_epoch_list)
            print(f"当前第{count}号恶意客户端：{adversarial_client.id}的poison_epoch:{adversarial_client.get_poison_epoch_list()}")
            count += 1
            adversarial_clients.append(adversarial_client)
        self.adversarial_clients = adversarial_clients

    def get_adversarial_clients(self):
        return self.adversarial_clients

    @staticmethod
    def model_dist_norm_var(model, target_params_variables, norm=2):
        size = 0
        for name, layer in model.named_parameters():
            size += layer.view(-1).shape[0]
        sum_var = torch.FloatTensor(size).fill_(0)
        sum_var = sum_var.to(config.device)
        size = 0
        for name, layer in model.named_parameters():
            sum_var[size:size + layer.view(-1).shape[0]] = (
                    layer - target_params_variables[name]).view(-1)
            size += layer.view(-1).shape[0]

        return torch.norm(sum_var, norm)

    def malicious_express_backdoor_loss(self,adversarial_clients):
        avg_backdoor_loss=0.0
        for adversarial_client in adversarial_clients:
            avg_backdoor_loss+=adversarial_client.latest_backdoor_loss
        return (avg_backdoor_loss/len(adversarial_clients))

    def get_client_by_id(self,clients,id):
        for client in clients:
            if client.id==id:
                return client
        print('未找到该id编号的客户端！')
        return None

    @staticmethod
    def model_dist_norm(model, target_params):
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data - target_params[name].data, 2))
        return math.sqrt(squared_sum)

    @staticmethod
    def model_global_norm(model):
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data, 2))
        return math.sqrt(squared_sum)

    @staticmethod
    def model_previous_global_norm(model):
        squared_sum = 0
        for name, layer in model.items():
            squared_sum += torch.sum(torch.pow(layer.data, 2))
        return math.sqrt(squared_sum)

    def creat_model(self):
        local_model = None
        global_model = None
        if self.params['type'] == config.TYPE_MNIST:
            local_model = SimpleNet(name='local', created_time=self.current_time).to(device)
            global_model = SimpleNet(name='global', created_time=self.current_time).to(device)


        elif self.params['type'] == config.TYPE_FashionMNIST:
            local_model = SimpleNet(name='local', created_time=self.current_time).to(device)
            global_model = SimpleNet(name='global', created_time=self.current_time).to(device)

        elif self.params['type'] == config.TYPE_CIFAR10:
            local_model = models.resnet18(pretrained=True).to(device)
            global_model = models.resnet18(pretrained=True).to(device)

        if self.params['resumed_model']:
            if torch.cuda.is_available():
                loaded_params = torch.load(
                    f"saved_models/{self.attack_type}/{self.params['resumed_model_name']}")
            else:
                loaded_params = torch.load(self.params['resumed_model_path'], map_location='cpu')

            global_model.load_state_dict(loaded_params['state_dict'])
            if self.params['reset_epoch']:
                self.start_epoch=1
            else:
                self.start_epoch = loaded_params['epoch'] + 1
            self.params['lr'] = loaded_params.get('lr', self.params['lr'])
            print(f"Loaded parameters from saved model: LR is"
                  f" {self.params['lr']} and current epoch is {self.start_epoch}")
        else:
            self.start_epoch = 1

        self.local_model = local_model
        self.global_model = global_model

    def initialize_dataset(self, data, labels):
        self.dataset = CustomDataset(data, labels)
        return self.dataset

    def init_weight_accumulator(self, global_model):
        weight_accumulator = dict()
        for name, data in global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(data)

        return weight_accumulator

    def get_batch(self, train_data, bptt, evaluation=False):
        data, target = bptt
        data = data.to(device)
        target = target.to(device)
        if evaluation:
            data.requires_grad_(False)
            target.requires_grad_(False)
        return data, target

    def get_poison_batch(self, bptt, adversarial_index=-1, evaluation=False, test_type=None):

        images, targets = bptt

        poison_count = 0
        new_images = images
        new_targets = targets

        for index in range(0, len(images)):

            if evaluation:
                new_targets[index] = self.params['target_label']
                new_images[index] = self.insert_several_triggers(images[index], self.global_model,position=[0,0])
                poison_count += 1

            else:  # poison part of data when training
                if index < self.params['poisoning_per_batch']:
                    new_targets[index] = self.params['target_label']
                    new_images[index] = self.add_pixel_pattern(images[index], adversarial_index)
                    poison_count += 1
                else:
                    new_images[index] = images[index]
                    new_targets[index] = targets[index]

        new_images = new_images.to(device)
        new_targets = new_targets.to(device).long()
        if evaluation:
            new_images.requires_grad_(False)
            new_targets.requires_grad_(False)
        return new_images, new_targets, poison_count

        # if config.WHETHER_STORE:
        # self.folder_path = f'saved_models/model_{self.name}_{current_time}'
        # try:
        #     os.mkdir(self.folder_path)
        # except FileExistsError:
        #     logger.info('Folder already exists')
        # logger.addHandler(logging.FileHandler(filename=f'{self.folder_path}/log.txt'))
        # logger.addHandler(logging.StreamHandler())
        # logger.setLevel(logging.DEBUG)
        # logger.info(f'current path: {self.folder_path}')
        # if not self.params.get('environment_name', False):
        #     self.params['environment_name'] = self.name
        # self.params['current_time'] = self.current_time
        # self.params['folder_path'] = self.folder_path
        # self.fg = FoolsGold(use_memory=self.params['fg_use_memory'])

    def info(self,dataset):
        print(f'Dataset size:{len(dataset)}')
        sample_image, sample_label=dataset[0]
        print(f'Sample image shape:{sample_image.shape}')
        # self.img_show(sample_image)

        if len(sample_image.shape) == 3:
            self.image_channel, self.image_height, self.image_width = sample_image.shape
            print(
                f'Image_channel: {self.image_channel}; Image_height: {self.image_height}; Image_width: {self.image_width}')
        elif len(sample_image.shape) == 2:
            self.image_height, self.image_width = sample_image.shape
            self.image_channel=1
            print(f'Image_height: {self.image_height}; Image_width: {self.image_width}')
        else:
            print("Unexpected sample image shape.")

        class_labels=set()
        for _,label in dataset:
            if isinstance(label,torch.Tensor):
                class_labels.add(label.item())
            else:
                class_labels.add(label)
        print(f'Number of labels in dataset:{sorted(class_labels)}')

    def img_show(self,img):
        plt.imshow(transforms.ToPILImage()(img))
        # plt.title(f'Lable:{label}')
        plt.show()

    def generate_full_combination_trigger(self,global_trigger_index):
        m=len(global_trigger_index)
        full_combination_index=[]

        for r in range(1,m):
            for subset in combinations(global_trigger_index,r):
                full_combination_index.append(list(subset))

        return full_combination_index

    def generate_local_trigger(self,global_trigger,indices):
        local_list=[]
        # print(indices)
        length= 1 if type(indices)== int else len(indices)
        if self.params['full_combination_trigger']:
            for i in range(length):
                combination=[]
                for j in indices[i]:
                    combination.append(global_trigger[j])
                local_list.append(combination)

        else:
            for i in range(length):
                combination=global_trigger[i]
                local_list.append(combination)
        return local_list

    def trigger_uniformly_distribution(self,samples_list,global_trigger,index1):

        backdoored_samples=[]

        local_trigger=self.generate_local_trigger(global_trigger,index1)

        num_trigger_type=len(local_trigger)
        num_samples=len(samples_list)

        batch_size=num_samples // num_trigger_type

        extra_count=num_samples % num_trigger_type
        print(f'样本数目:{num_samples},trigger类型数目:{num_trigger_type}')

        start_index=0

        for i in range(num_trigger_type):

            group_size=batch_size+(1 if extra_count>0 else 0)

            for j in range(group_size):
                ori_img=samples_list[start_index+j][0][0]
                position=samples_list[start_index+j][1]
                # self.img_show(ori_img)
                if self.params['full_combination_trigger']:
                    data = self.insert_several_triggers(ori_img, local_trigger[i], position)
                else:
                    data=self.insert_single_trigger(ori_img,local_trigger[i],position)
                label=self.params['target_label']
                backdoored_samples.append([data,label])
            extra_count-=1
            start_index+=group_size

        return backdoored_samples

    def trigger_static_distribution(self,samples_list,local_trigger):

        backdoored_samples = []


        num_samples = len(samples_list)

        for j in range(num_samples):
            ori_img = samples_list[j][0][0]
            position = samples_list[j][1]
            if self.params['full_combination_trigger']:

                data = self.insert_several_triggers(ori_img, local_trigger, position)
            else:
                data = self.insert_single_trigger(ori_img, local_trigger, position)
            label = self.params['target_label']
            backdoored_samples.append([data, label])

        return backdoored_samples

    def show_samples(self,samples):
        for sample in samples:
            data,label=sample
            self.img_show(data)

    def show_samples_by_batch(self, samples, images_per_batch=14):
        """
        Args:
            samples (list of tuples): A list of samples containing (image data, labels).
            images_per_batch (int):The default number of images displayed per batch is 14.
        """
        num_samples = len(samples)
        num_batches = (num_samples + images_per_batch - 1) // images_per_batch

        for batch_index in range(num_batches):
            start_idx = batch_index * images_per_batch
            end_idx = min((batch_index + 1) * images_per_batch, num_samples)
            batch_samples = samples[start_idx:end_idx]

            num_cols = 7
            num_rows = 2

            plt.figure(figsize=(num_cols * 2, num_rows * 2))

            for i, (data, label) in enumerate(batch_samples):
                plt.subplot(num_rows, num_cols, i + 1)

                if len(data.shape) == 2:
                    plt.imshow(data, cmap='gray')
                elif len(data.shape) == 3:
                    plt.imshow(np.transpose(data, (1, 2, 0)))

                plt.title(f'Label: {label}')
                plt.axis('off')

            plt.tight_layout()
            plt.show()

    def global_trigger_generate(self, setting):
        """

         Args:
         size (int): The length of each sub-pixel block (assuming the width of each sub-pixel block is 1).
        gapi (int): The gap between two adjacent sub-pixel blocks in the row direction.
        gapj (int): The gap between two adjacent sub-pixel blocks in the column direction.
        m (int): The number of local trigger sub-blocks.

        Returns:
        list: A list containing the positions of the triggers for each sub-pixel block.
         """
        patterns = []
        size = setting[0]
        gapi = setting[1]
        gapj = setting[2]
        m = setting[3]  # trigger_pattern
        l=int(math.sqrt(m))
        i=0
        j=0

        for _ in range(l):
            j=0
            for _ in range(l):
                pattern = []
                for x in range(size):
                    pattern.append([i,j+x])
                j+=size+gapj
                patterns.append(pattern)

            i+=gapi


        return patterns

    def frobenius_norm(self,a,b):
        # print(a.shape)
        # print(b.shape)
        return torch.norm(a-b,'fro')

    def best_unify_trigger_position(self, samples_dataset, target_samples, global_trigger, setting):

        img_height = self.image_height
        img_width = self.image_width
        best_position = None
        # max_similarity = -float('inf')
        min_distance = float('inf')
        trigger_size = setting[0]
        gapi = setting[1]
        gapj = setting[2]
        times = math.sqrt(setting[3]) - 1
        width = times * gapi + math.sqrt(setting[3]) * trigger_size
        height = times * gapj + math.sqrt(setting[3]) * 1
        pds = PoissonDiskSampling(img_width - width, img_height - height, max(width, height))
        candidate_points = pds.generate_points()

        # self.show_samples_by_batch(target_samples)
        # print(type(target_samples))
        # copied_target_samples = copy.copy(target_samples)
        target_samples_tensor = torch.stack([ts[0] for ts in target_samples]).to(device)
        target_samples_flat = target_samples_tensor.view(len(target_samples_tensor), -1)




        for point in candidate_points:
            i, j = int(point[1]), int(point[0])
            min_distance_for_point = float('inf')
            current_distance = 0

            for o in range(2):
                indices = list(range(len(samples_dataset)))
                random.shuffle(indices)
                subset_indices=indices[:5000]
                temp_dataset=[samples_dataset[l] for l in subset_indices]
                samples_insert_trigger = []

                for k in range(len(temp_dataset)):
                    data = temp_dataset[k][0]
                    samples_insert_trigger.append(self.insert_several_triggers(data, global_trigger, [i, j]))
                samples_insert_trigger_tensor = torch.stack([data for data in samples_insert_trigger]).to(device)
                samples_insert_trigger_flat = samples_insert_trigger_tensor.view(len(samples_insert_trigger_tensor),
                                                                                 -1)

                # current_similarity = 0
                for sample_flat in samples_insert_trigger_flat:
                    sample_flat = sample_flat.unsqueeze(0)
                    distances = torch.norm(sample_flat.unsqueeze(1) - target_samples_flat.unsqueeze(0), p='fro',
                                           dim=2)
                    current_distance += distances.min().item()

                current_distance /= len(samples_insert_trigger_flat)
                if current_distance < min_distance_for_point:
                    min_distance_for_point = current_distance
                del samples_insert_trigger_tensor
                del samples_insert_trigger_flat
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if min_distance_for_point < min_distance:
                min_distance = min_distance_for_point
                best_position = [i, j]

                # for i in range(len(data)):
                #     sample_data = data[i]

                    # triggerd_sample = self.insert_several_triggers(sample_data, global_trigger, [i, j]).to(device)
                    # distances = torch.stack(
                    #     [self.frobenius_norm(triggerd_sample, target_sample[0]) for target_sample in target_samples])
                    # current_distance += distances.sum().item()
            # if current_distance < min_distance:
            #     min_distance = current_distance
            #     best_position = [i, j]
            #
            # if len(data.shape) == 3:
            #     data = data.unsqueeze(1)
            #
            # # print(data.shape)
            # triggered_samples = self.insert_several_triggers_into_batch_samples(data, global_trigger, [i, j])
            # triggered_samples_flat = triggered_samples.view(len(triggered_samples), -1).to(device)
            # similarities = F.cosine_similarity(triggered_samples_flat.unsqueeze(1),
            #                                    target_samples_flat.unsqueeze(0), dim=2)
            #
            # current_similarity += similarities.sum().item()
        # del samples_insert_trigger_tensor
        # del samples_insert_trigger_flat
        # gc.collect()

        print(best_position)
        return best_position

    def best_trigger_position(self,sample,target_samples,global_trigger,setting):
        img=sample[0].to(device)
        img_height=self.image_height
        img_width=self.image_width
        best_position=None
        min_distance=float('inf')
        trigger_size=setting[0]
        gapi = setting[1]
        gapj = setting[2]

        times=math.sqrt(setting[3])-1
        width=times*gapi+math.sqrt(setting[3])*trigger_size
        height=times*gapj+math.sqrt(setting[3])*1
        pds=PoissonDiskSampling(img_width-width,img_height-height,max(width,height))
        candidate_points=pds.generate_points()

        for point in candidate_points:
            i,j = int(point[1]),int(point[0])
            triggerd_sample=self.insert_several_triggers(img,global_trigger,[i,j]).to(device)
            distances=torch.stack([self.frobenius_norm(triggerd_sample,target_sample[0]) for target_sample in target_samples])
            current_distance=distances.sum().item()

            if current_distance < min_distance:
                min_distance = current_distance
                best_position=[i,j]

        # for i in range(img_height-2*1-gapi+1):
        #     for j in range(img_width-2*trigger_size-gapj+1):
        #         triggered_sample=self.insert_several_triggers(img,global_trigger,[i,j])
        #         current_distance=sum(self.frobenius_norm(triggered_sample,target_sample[0]) for target_sample in target_samples)
        #         if current_distance<min_distance:
        #             min_distance=current_distance
        #             best_position=[i,j]
                # del triggered_sample
        # gc.collect()

        print(best_position)
        return best_position

    def insert_single_trigger(self, ori_image, trigger, position):
        image = copy.deepcopy(ori_image)
        if len(image.shape) == 2:
            height, width = image.shape
            image = image.unsqueeze(0)
            channels = 1
        elif len(image.shape) == 3:
            channels, height, width = image.shape
        else:
            print("图像维度异常！")
        for pos in trigger:
            x, y = pos[0] + position[0], pos[1] + position[1]
            if 0 <= x < height and 0 <= y < width:
                if channels == 1:
                    image[0, x, y] = 1
                elif channels == 2 or channels == 3:
                    for c in range(self.image_channel):
                        image[c, x, y] = 1
        return image

    def insert_several_triggers_into_batch_samples(self, ori_images, trigger, position):
        images = copy.deepcopy(ori_images)

        if len(images.shape) == 3:
            images = images.unsqueeze(1)
        batch_size, channels, height, width = images.shape
        for sub_trigger in trigger:
            for pos in sub_trigger:
                x, y = pos[0] + position[0], pos[1] + position[1]
                if 0 <= x < height and 0 <= y < width:
                    if channels == 1:
                        images[:,:,x,y] = 1
                    elif channels == 2 or channels == 3:
                            images[:,:,x,y] = 1

        return images

    def insert_several_triggers_reducec_memory(self, ori_image, trigger, position):
        # image = copy.copy(ori_image)
        image=ori_image.detach()
        # image = copy.copy(ori_image)
        if len(image.shape) == 2:
            height, width = image.shape
            image = image.unsqueeze(0)
            channels = 1
        elif len(image.shape) == 3:
            channels, height, width = image.shape
        else:
            print("图像维度异常！")
        with torch.no_grad():
            for i, sub_trigger in enumerate(trigger):

                if channels == 3:
                    color = self.color_map.get(i, [1, 1, 0])
                for pos in sub_trigger:
                    x, y = pos[0] + position[0], pos[1] + position[1]
                    if 0 <= x < height and 0 <= y < width:
                        if channels == 1:
                            image[0, x, y] = 1
                        elif channels == 3:
                            image[0, x, y] = color[0]
                            image[1, x, y] = color[1]
                            image[2, x, y] = color[2]
        return image

    def insert_several_triggers(self, ori_image, trigger, position,thickness=0):
        image = copy.deepcopy(ori_image)
        # image = copy.copy(ori_image)
        if len(image.shape) == 2:
            height, width = image.shape
            image = image.unsqueeze(0)
            channels = 1
        elif len(image.shape) == 3:
             channels, height, width = image.shape
        else:
            print("图像维度异常！")
        for i, sub_trigger in enumerate(trigger):
            if channels == 3:
                color = self.color_map.get(i, [1, 1, 0])
            for pos in sub_trigger:
                x, y = pos[0] + position[0], pos[1] + position[1]
                # if channels == 1:
                #     for dx in range(max(0,x-thickness//2),min(height,x+thickness//2+1)):
                #         image[0,dx,y]=1
                #
                # elif channels==3:
                #     for dx in range(max(0, x - thickness // 2), min(height, x + thickness // 2 + 1)):
                #         image[0, dx, y] = color[0]
                #         image[1, dx, y] = color[1]
                #         image[2, dx, y] = color[2]
                if 0 <= x < height and 0 <= y < width:
                    if channels ==1:
                        image[0, x, y] = 1
                        # self.img_show(image)
                    elif channels == 3:
                        image[0, x, y] = color[0]
                        image[1, x, y] = color[1]
                        image[2, x, y] = color[2]
        return image

    def list_substraction(self,a,b):
        a_set=set(a)
        b_set=set(b)
        a_set-=b_set
        return list(a_set)

    def random_namekeys(self, total_clients_name_list, adversarial_clients_name_list):
        agent_name_keys = []
        adversarial_name_keys = []
        agent_name_keys = random.sample(total_clients_name_list, self.params['num_select_clients'])
        for _name_keys in agent_name_keys:
            if _name_keys in adversarial_clients_name_list:
                adversarial_name_keys.append(_name_keys)
        return agent_name_keys, adversarial_name_keys

    def fixed_poison_epoch_namekeys(self, epoch, adversarial_clients_name_list, benign_clients_name_list):
        count = 0
        # agent_name_keys = []
        adversarial_name_keys = []
        for adversarial_client in self.adversarial_clients:
            poison_epoch = adversarial_client.get_poison_epoch_list()
            if epoch in poison_epoch and count < self.params['num_select_clients']:
                adversarial_name_keys.append(adversarial_client.id)
                count += 1
        nonattacker = []
        for adv in adversarial_clients_name_list:
            if adv not in adversarial_name_keys:
                nonattacker.append(copy.deepcopy(adv))
        print(adversarial_name_keys)
        if self.params['is_random_choice_in_fixed_mode']:
            benign_num = self.params['num_select_clients'] - len(adversarial_name_keys)
            random_agent_name_keys = random.sample(benign_clients_name_list + nonattacker,
                                                   benign_num)
            agent_name_keys = adversarial_name_keys + random_agent_name_keys
            for _name_keys in agent_name_keys:
                if _name_keys in adversarial_clients_name_list:
                    adversarial_name_keys.append(_name_keys)
            return agent_name_keys, adversarial_name_keys
        else:
            random_agent_name_keys = random.sample(benign_clients_name_list, self.params['num_select_clients'] - len(
                adversarial_name_keys))
            agent_name_keys = adversarial_name_keys + random_agent_name_keys
            return agent_name_keys, adversarial_name_keys
    def optimal_sample_selection(self,total_dataset,poison_dataset,R):
        all_data = []
        all_labels = []
        length=len(total_dataset) if len(total_dataset)<30000 else 6000
        for i in range(length):
            data,label=total_dataset[i]
            # poison_data,poison_label=poison_dataset[i]
            # copy_total_dataset.append((data,label))
            all_data.append(data)
            all_labels.append(label)
        copy_total_dataset=self.initialize_dataset(all_data,all_labels)
        if self.params['type'] == config.TYPE_MNIST:
            agent_model = SimpleNet(name=None, created_time=None)
        elif self.params['type'] == config.TYPE_CIFAR10:
            agent_model = models.resnet18(pretrained=True)
        elif self.params['type']==config.TYPE_FashionMNIST:
            agent_model = SimpleNet(name=None, created_time=None)

        optimizer= optim.Adam(agent_model.parameters(), lr=self.params['lr'])
        criterion = nn.CrossEntropyLoss()
        backdoor_list=[]
        benign_list=list(range(len(copy_total_dataset)))

        num_samples=len(copy_total_dataset)
        batch_size=5000
        num_batches=math.ceil(num_samples/batch_size)

        M = self.params['update_rounds']

        T = int(R) #
        R=int(batch_size)
        print(f'T:{T};R:{R};total_dataset_size:{len(copy_total_dataset)}')

        for epoch in range(M):
            if epoch==0:
                backdoor_list=random.sample(benign_list,R)
                benign_list=self.list_substraction(benign_list,backdoor_list)
                for idx in backdoor_list:
                    copy_total_dataset.sample_replacement(idx, poison_dataset[idx])
            else:
                append_backdoor_list=random.sample(benign_list,R-T)
                for idx in append_backdoor_list:
                    copy_total_dataset.sample_replacement(idx, poison_dataset[idx])
                backdoor_list+=append_backdoor_list
                benign_list=self.list_substraction(benign_list,append_backdoor_list)
            # print(f'epoch:{epoch},backdoor_list:{backdoor_list},benign_list_size:{len(benign_list)}')
            dataloader=DataLoader(copy_total_dataset,batch_size=1,shuffle=False)
            backdoor_grads=[]
            mean_backdoor_grad=None
            projections=[]
            for i , (data,label) in enumerate(dataloader):
                optimizer.zero_grad()
                pred=agent_model(data)
                loss=criterion(pred,label)
                loss.backward()

                if i in backdoor_list:
                    grads=[]
                    for param in agent_model.parameters():
                        if param.grad is not None:
                            grads.append(param.grad.clone().view(-1))
                            param.grad,data.zero_()
                    grads=torch.cat(grads)
                    score=torch.linalg.norm(grads)
                    backdoor_grads.append((i,grads))
                    if mean_backdoor_grad is None:
                        mean_backdoor_grad=1/R*grads.clone()/score
                    else:
                        mean_backdoor_grad+=1/R*grads/score
            # print(mean_backdoor_grad.shape)
            for i in range(len(backdoor_grads)):
                value=torch.matmul(backdoor_grads[i][1], mean_backdoor_grad).item()
                projections.append((backdoor_grads[i][0],value))
            projections.sort(key=lambda x:x[1],reverse=True)
            reverse_backdoor_list=[item[0] for item in projections[T:R]]
            backdoor_list=self.list_substraction(backdoor_list,reverse_backdoor_list)
            benign_list+=reverse_backdoor_list
            for idx in reverse_backdoor_list:
                copy_total_dataset.sample_replacement(idx, total_dataset[idx])
        # del all_data, all_labels,backdoor_grads,projections,mean_backdoor_grad,reverse_backdoor_list
        # gc.collect()
        del all_data,all_labels,backdoor_grads
        gc.collect()
        return backdoor_list

    def split_list(self,lst,M):
        n = len(lst)
        part_size = n // M
        remainder = n % M

        parts = []
        start_index = 0

        for i in range(M):
            end_index = start_index + part_size + (1 if i < remainder else 0)
            parts.append(lst[start_index:end_index])
            start_index = end_index

        return parts

    def get_samples(self,index,dataset,best_trigger_position):
        samples=[]
        for i in index:
            #position的顺序与total_dataset保持一致
            samples.append([dataset[i],best_trigger_position[i]])
        return samples

