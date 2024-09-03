
import copy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import helper
import test
from common import vis
import numpy as np
import matplotlib.pyplot as plt
from SADBA import config


class Client:
    def __init__(self,model, dataset,criterion,helper,id):
        self.helper=helper
        self.model=copy.deepcopy(model)
        self.previous_parameters = copy.deepcopy(self.model.state_dict())
        self.client_grad=[]
        self.dataset=dataset
        self.benign_dataloader=DataLoader(dataset=self.dataset,batch_size=64,shuffle=True)
        self.criterion=criterion
        self.diff_update={}
        self.id=id
        self.latest_backdoor_loss=float('inf')

    def set_poison_epoch_list(self,poison_epoch_list):
        self.poison_epoch_list=poison_epoch_list

    def get_poison_epoch_list(self):
        return self.poison_epoch_list

    def set_backdoor_dataset(self,dataset,batch_size):
        self.backdoor_dataset=dataset
        self.backdoor_dataloader=DataLoader(dataset=dataset,batch_size=64,shuffle=True)

        # data_iter=iter(self.backdoor_dataloader)
        # backdoored_first_batch=next(data_iter)
        # for img,label in backdoored_first_batch:
        #     if label ==7:
        #             img = img.numpy().transpose((1, 2, 0))
        #             plt.imshow(img)
        #             plt.show()


    def get_backdoor_dataset(self,backdoor_samples,append_poisoned_data):

        copy_dataset=self.dataset.copy()
        if append_poisoned_data:
            copy_dataset.add_data(backdoor_samples)
        else:
            indices_to_replace = np.random.choice(len(copy_dataset), len(backdoor_samples), replace=False)
            copy_dataset.replace_data(indices_to_replace,backdoor_samples)
        self.poisoned_dataset=copy_dataset


    def train(self,common,epochs_submit_update_dict,helper,num_samples_dict,current_number_of_adversaries,is_poison,start_epoch):
        epochs_local_update_list = []
        client_grad = []

        dataset_size = 0
        optimizer = torch.optim.SGD(self.model.parameters(), lr=helper.params['lr'],
                                    momentum=helper.params['momentum'],
                                    weight_decay=helper.params['decay'])

        optimizer = optim.Adam(self.model.parameters(), lr=self.helper.params['lr'])
        optimizer.zero_grad()
        self.model.train()
        if is_poison:
            if helper.params['is_ASR_t']:
                wether_train_in_backdoor_mode=True if (self.id in helper.adversarial_namelist and ((start_epoch in helper.params['poison_epochs']) or (start_epoch in self.get_poison_epoch_list()) or start_epoch<150)) else False
            else:
                wether_train_in_backdoor_mode=True if self.id in self.helper.adversarial_namelist else False
        else:
            wether_train_in_backdoor_mode=False

        for epoch in range(start_epoch, start_epoch + helper.params['aggr_epoch_interval']):

            if wether_train_in_backdoor_mode :
                print('当前轮次将注入后门')

                step_lr = helper.params['poison_step_lr']
                poison_milestones=helper.params['poison_milestones']

                poison_lr=helper.params['poison_lr']
                poison_internal_epoch_num = helper.params['poison_internal_poison_epochs']

                if helper.params['optimize_adversarial_local_training']:
                    avg_backdoor_loss=helper.malicious_express_backdoor_loss(helper.get_adversarial_clients())
                    if self.latest_backdoor_loss!=float('inf') and self.latest_backdoor_loss<=avg_backdoor_loss:
                        gama=0.1
                        poison_lr = helper.params['poison_lr']  # lr'
                        poison_internal_epoch_num = helper.params['poison_internal_poison_epochs']
                    else:
                        gama=1
                        poison_lr = helper.params['poison_lr']*helper.params['u1']
                        poison_internal_epoch_num = helper.params['poison_internal_poison_epochs']*helper.params['u2']
                poison_optimizer = torch.optim.SGD(self.model.parameters(),
                                                   lr=poison_lr,
                                               momentum=helper.params['momentum'],
                                               weight_decay=helper.params['decay'])

                poison_optimizer.zero_grad()
                scheduler = torch.optim.lr_scheduler.MultiStepLR(poison_optimizer, milestones=poison_milestones,
                                                             gamma=0.1)
                temp_local_epoch=self.backdoor_local_train(common,helper,epoch,num_samples_dict,poison_lr,poison_internal_epoch_num,step_lr,poison_optimizer,scheduler,client_grad)

                print(f'Global model norm: {helper.model_previous_global_norm(self.previous_parameters)}.')
                print(f'Norm before scaling: {helper.model_global_norm(self.model)}. '
                                 f'Distance: {helper.model_dist_norm(self.model, self.previous_parameters)}')
                if not helper.params['baseline']:
                    clip_rate = helper.params['scale_weights_poison']
                    print(f"Scaling by  {clip_rate}")

                    for key, value in self.model.state_dict().items():
                        target_value = self.previous_parameters[key]
                        new_value = target_value + (value - target_value) * clip_rate  *(1/current_number_of_adversaries)
                        self.model.state_dict()[key].copy_(new_value)
                    distance = helper.model_dist_norm(self.model, self.previous_parameters) #（Lt+1-Gt）^2
                    print(
                        f'注入后门后的距离参数: '
                        f'{helper.model_global_norm(self.model)}, distance: {distance}')

                    common.backdoor_localmodel_distance.append([self.id,epoch,round(distance, 4)])

                    if helper.params["batch_track_distance"]:
                        temp_data_len = len(self.backdoor_dataloader)
                        self.model.track_distance_batch_vis(vis=vis, epoch=temp_local_epoch,
                                                       data_len=temp_data_len,
                                                       batch=temp_data_len - 1,
                                                       distance_to_global_model=distance,
                                                       eid=helper.params['type'],
                                                       name=str(self.id), is_poisoned=True)

                if helper.params['diff_privacy']:
                    model_norm = helper.model_dist_norm(self.model, self.previous_parameters)
                    if model_norm > helper.params['s_norm']:
                        norm_scale = helper.params['s_norm'] / (model_norm)
                        for name, layer in self.model.named_parameters():
                            #### don't scale tied weights:
                            if helper.params.get('tied', False) and name == 'decoder.weight' or '__' in name:
                                continue
                            clipped_difference = norm_scale * (
                                    layer.data - self.previous_parameters[name])
                            layer.data.copy_(self.previous_parameters[name] + clipped_difference)
                distance = helper.model_dist_norm(self.model, self.previous_parameters)
                print(f"Total norm for {current_number_of_adversaries} "
                                 f"adversaries is: {helper.model_global_norm(self.model)}. distance: {distance}")

            else:
                lr=helper.params['lr']
                internal_epoch_num = helper.params['local_epoch']
                print(f'当前客户端本地训练设置：学习率：{lr}；本地训练轮数：{internal_epoch_num}')
                self.benign_local_train(common,helper,epoch,num_samples_dict,lr,internal_epoch_num,optimizer,client_grad,self.previous_parameters)

                # test local model after internal epoch finishing
            epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest(helper=helper, epoch=epoch,
                                                                           model=self.model, is_poison=False, visualize=True,
                                                                           agent_name_key=self.id)
            common.local_testMA_result.append([self.id, epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])

            if is_poison:
                if self.id in helper.adversarial_namelist:
                    epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest_poison(helper=helper,
                                                                                          epoch=epoch,
                                                                                          model=self.model,
                                                                                          is_poison=True,
                                                                                          visualize=True,
                                                                                          agent_name_key=self.id)
                    self.latest_backdoor_loss=epoch_loss
                    common.local_triggerASR_result.append([self.id,helper.params['test_trigger_indices'],epoch,epoch_loss,epoch_acc,epoch_corret,epoch_total])


                #  test on local triggers   测试单个子trigger_partern的性能
                # if self.id in helper.params['adversary_list_1']:self.tr
                #     if helper.params['vis_trigger_split_test']:
                #         model.trigger_agent_test_vis(vis=main.vis, epoch=epoch, acc=epoch_acc, loss=None,
                #                                      eid=helper.params['type'],
                #                                      name=str(agent_name_key)  + "_combine")
                #
                #     epoch_loss, epoch_acc, epoch_corret, epoch_total = \
                #         test.Mytest_poison_agent_trigger(helper=helper, model=model, agent_name_key=agent_name_key)
                #     print(f'误触率： 当前轮次:{epoch} 本轮的参与方 : {agent_name_key}   平均损失 : {epoch_loss}  正确率 : {epoch_acc}.')
                #     csv_record.poisontriggertest_result.append(
                #         [agent_name_key, str(agent_name_key) + "_trigger", "", epoch, epoch_loss,
                #          epoch_acc, epoch_corret, epoch_total])
                #     if helper.params['vis_trigger_split_test']:
                #         model.trigger_agent_test_vis(vis=main.vis, epoch=epoch, acc=epoch_acc, loss=None,
                #                                      eid=helper.params['type'],
                #                                      name=str(agent_name_key) + "_trigger")


            current_parameter = self.get_parameters()
            for key in current_parameter.keys():
                self.diff_update[key]=current_parameter[key]-self.previous_parameters[key]
            epochs_local_update_list.append(self.get_diff_update())
        epochs_submit_update_dict[self.id] = epochs_local_update_list




    def backdoor_local_train(self,common,helper,epoch,num_samples_dict,poison_lr,internal_epoch_num,step_lr,poison_optimizer,scheduler,client_grad):

        temp_local_epoch = (epoch - 1) * internal_epoch_num
        target_params_variables = dict()
        for name, param in self.model.named_parameters():
            target_params_variables[name] = self.previous_parameters[name].clone().detach().requires_grad_(False)
        for internal_epoch in range(1, internal_epoch_num + 1):
            temp_local_epoch += 1
            total_loss = 0.
            correct = 0
            dataset_size = 0
            dis2global_list = []
            for batch_id, batch in enumerate(self.backdoor_dataloader):

                data, targets = batch
                poison_optimizer.zero_grad()
                for param in self.model.parameters():
                    if param.grad is not None:
                        assert torch.all(param.grad == 0), "Gradient is not zeroed out!"
                dataset_size += len(data)


                output = self.model(data)
                class_loss = nn.functional.cross_entropy(output, targets)



                distance_loss = helper.model_dist_norm_var(self.model, target_params_variables)
                # Lmodel = αLclass + (1 − α)Lano; alpha_loss =1 fixed
                loss = helper.params['alpha_loss'] * class_loss + (1 - helper.params['alpha_loss']) * distance_loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                poison_optimizer.step()
                total_loss += loss.item()
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

                if helper.params["batch_track_distance"]:
                    # we can calculate distance to this model now.
                    temp_data_len = len(self.backdoor_dataloader)
                    distance_to_global_model = helper.model_dist_norm(self.model, target_params_variables)
                    dis2global_list.append(distance_to_global_model)
                    self.model.track_distance_batch_vis(vis=vis, epoch=temp_local_epoch,
                                                   data_len=temp_data_len,
                                                   batch=batch_id, distance_to_global_model=distance_to_global_model,
                                                   eid=helper.params['type'],
                                                   name=str(self.id), is_poisoned=True)

            if step_lr:
                scheduler.step()
                print(f'当前学习率: {scheduler.get_last_lr()}')

            acc = 100.0 * (float(correct) / float(dataset_size))
            total_l = total_loss / dataset_size
            print(
                '当前全局轮次： {:3d}, 客户端id： {}, 目前处于多少轮 {:3d},  平均损失: {:.4f}, '
                '准确率: {}/{} ({:.4f}%)'.format( epoch, self.id,
                                                             internal_epoch,
                                                             total_l, correct, dataset_size,
                                                             acc))

            # common.train_result.append(
            #     [self.id, temp_local_epoch,
            #      epoch, internal_epoch, total_l, acc, correct, dataset_size])

            if helper.params['vis_train']:
                self.model.train_vis(vis, temp_local_epoch,
                                acc, loss=total_l, eid=helper.params['type'], is_poisoned=True,
                                name=str(self.id))
            num_samples_dict[self.id]=dataset_size

            if helper.params["batch_track_distance"]:
                print(
                    f'MODEL {self.id}. P-norm is {helper.model_global_norm(self.model):.4f}. '
                    f'Distance to the global model: {dis2global_list}. ')

        return temp_local_epoch

    def benign_local_train(self,common,helper,epoch,num_samples_dict,lr,internal_epoch_num,optimizer,client_grad,target_params_variables):
        temp_local_epoch = (epoch - 1) * internal_epoch_num
        for internal_epoch in range(1, internal_epoch_num + 1):
            temp_local_epoch+=1

            total_loss = 0.
            correct = 0
            dataset_size = 0
            dis2global_list = []

            for batch_id, batch in enumerate(self.benign_dataloader):
                optimizer.zero_grad()
                for param in self.model.parameters():
                    if param.grad is not None:
                        assert torch.all(param.grad == 0), "Gradient is not zeroed out!"

                data, targets = helper.get_batch(self.benign_dataloader, batch, evaluation=False)

                if data.dim() == 3:
                    data = data.unsqueeze(1)


                dataset_size += len(data)
                output = self.model(data)

                loss = nn.functional.cross_entropy(output, targets)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                if helper.params['diff_privacy']:
                    optimizer.step()
                    model_norm = helper.model_dist_norm(self.model, target_params_variables)
                    print(f'Test_model_dist_norm: {model_norm}')
                    if model_norm > helper.params['s_norm']:
                        norm_scale = helper.params['s_norm'] / (model_norm)
                        for name, layer in self.model.named_parameters():
                            #### don't scale tied weights:
                            if helper.params.get('tied',
                                                 False) and name == 'decoder.weight' or '__' in name:
                                continue
                            clipped_difference = norm_scale * (
                                    layer.data - helper.target_model.state_dict()[name])
                            layer.data.copy_(
                                helper.target_model.state_dict()[name] + clipped_difference)
                else:
                    optimizer.step()
                total_loss += loss.item()
                pred = output.data.max(1)[1]
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
                if helper.params["vis_train_batch_loss"]:
                    cur_loss = loss.data
                    temp_data_len = len(self.benign_dataloader)
                    self.model.train_batch_vis(vis=vis,
                                          epoch=temp_local_epoch,
                                          data_len=temp_data_len,
                                          batch=batch_id,
                                          loss=cur_loss,
                                          eid=helper.params['type'],
                                          name=str(self.id), win='train_batch_loss', is_poisoned=False)
                # if helper.params["batch_track_distance"]:
                #     # we can calculate distance to this model now
                #     temp_data_len = len(self.benign_dataloader)
                #     distance_to_global_model = helper.model_dist_norm(self.model, target_params_variables)
                #     dis2global_list.append(distance_to_global_model)
                #     self.model.track_distance_batch_vis(vis=vis, epoch=temp_local_epoch,
                #                                    data_len=temp_data_len,
                #                                    batch=batch_id, distance_to_global_model=distance_to_global_model,
                #                                    eid=helper.params['type'],
                #                                    name=str(self.id), is_poisoned=False)

            acc = 100.0 * (float(correct) / float(dataset_size))
            total_l = total_loss / dataset_size
            print('___Train ,  当前全局轮次： {:3d}, 客户端名称： {}, 当前本地训练epoch {:3d},  平均损失: {:.4f}, '
                             '准确率: {}/{} ({:.4f}%)'.format( epoch, self.id, internal_epoch,
                                                           total_l, correct, dataset_size, acc)
                             )

            # common.train_result.append([self.id, temp_local_epoch,
            #                                 epoch, internal_epoch, total_l, acc, correct, dataset_size])
            if helper.params['vis_train']:
                self.model.train_vis(vis, temp_local_epoch,
                                acc, loss=total_l, eid=helper.params['type'], is_poisoned=False,
                                name=str(self.id))
            num_samples_dict[self.id] = dataset_size

            if helper.params["batch_track_distance"]:
                print(
                    f'MODEL {self.id}. P-norm is {helper.model_global_norm(self.model):.4f}. '
                    f'Distance to the global model: {dis2global_list}. ')

    def get_parameters(self):
        return copy.deepcopy(self.model.state_dict())

    def set_parameters(self, parameters):
        self.model.load_state_dict(parameters)
        self.previous_parameters = copy.deepcopy(self.model.state_dict())

    def get_diff_update(self):
        return self.diff_update
