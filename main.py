import argparse
import copy
import datetime
import time
from torch.utils.data import Subset
import config
import yaml
import common
from SADBA import test
from common import logger,vis
import torch.nn as nn
import torch
from client import Client
from server import Server
import os
import numpy as np
import random
from preparation import ImageHelper
from data_manage import SharedClass


def generate_dataset_save_path(home_dir,best_trigger_position,optimal_sample_selection,full_combination_trigger,trigger_distribution_strategy):
    trigger_position_select_strategy=('best_trigger_position' if best_trigger_position else 'random_trigger_position')
    sample_choice_strategy=('optimal_sample_selection' if optimal_sample_selection else 'random_samples_selection')
    trigger_type=('full_combination' if full_combination_trigger else 'DBA,m=4')
    trigger_distribution_strategy_str = (
        'uniform_distribution' if trigger_distribution_strategy == 1
        else 'static_1-to-1_mapping' if trigger_distribution_strategy == 2
        else None
    )
    path=os.path.join(home_dir,trigger_position_select_strategy,sample_choice_strategy,trigger_type,trigger_distribution_strategy_str)
    return path

def dict_html(dict_obj, current_time):
    out = ''
    for key, value in dict_obj.items():

        #filter out not needed parts:
        if key in ['poisoning_test', 'test_batch_size', 'discount_size', 'folder_path', 'log_interval',
                   'coefficient_transfer', 'grad_threshold' ]:
            continue

        out += f'<tr><td>{key}</td><td>{value}</td></tr>'
    output = f'<h4>Params for model: {current_time}:</h4><table>{out}</table>'
    return output

def load_yaml(file_path):
    with open(file_path,'r') as f:
        return yaml.safe_load(f)


if __name__=='__main__':

    time_strat_load_everything = time.time()
    # parser = argparse.ArgumentParser(description='SADBA')
    # parser.add_argument('--params', dest='params')
    # args = parser.parse_args()
    # with open(f'./{args.params}', 'r') as f:
    #     params_loaded = yaml.safe_load(f)
    task_params_path='./task_params/total_config.yaml'
    with open(task_params_path, 'r') as f:
        params_loaded = yaml.safe_load(f)
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    subtask_params_path=params_loaded.get('subtask_dir_path')
    attack_type= 'SADBA'
    data_saved_path=params_loaded.get('saved_path')
    print(f"task_params_dir：{subtask_params_path}")
    subtask_paths=[]
    for root,dirs,files in os.walk(subtask_params_path):
        for file in files:
            if file.endswith('.yaml'):
                subtask_paths.append(os.path.join(root,file))
    for path in subtask_paths:
        print(path)
        params_loaded1=load_yaml(path)
        helper = ImageHelper(current_time=current_time, params=params_loaded1, name='SADBA')
        print(f"NC:{helper.params['total_clients']};m:{helper.params['trigger_pattern']};m_shape:{helper.params['setting']};MPC:{helper.params['MPC']}")
        if helper.params['training_mode']:
            random.seed(1)
            helper.creat_model()
            print('create model done')
            helper.load_test_data()
            print('test data load done')
            shared = SharedClass()
            home_path = helper.params['home_path']
            benign_dir = os.path.join(home_path, 'dataset/benign_dataset')
            adversarial_dir = os.path.join(home_path, 'dataset/adversarial_dataset')
            # adversarial_dir += str(helper.params['test_pos'])
            print(f"恶意数据集打开路径{adversarial_dir}")
            clients, benign_clients_name_list, adversarial_clients_name_list = shared.load_clients_data(helper,
                                                                                                        benign_dir,
                                                                                                        adversarial_dir)
            server = Server(helper.global_model, helper)
            print('load data done')
            print(f'恶意客户端名单：{adversarial_clients_name_list}')
            print(f'良性客户端名单：{benign_clients_name_list}')
            total_clients_name_list = sorted(adversarial_clients_name_list + benign_clients_name_list)
            print(f'总客户端名单：{total_clients_name_list}')
            helper.initialize_name_list(adversarial_clients_name_list, benign_clients_name_list)
            helper.initialize_adversarial_clients(clients, adversarial_clients_name_list)
            print('数据加载完成')
            best_loss = float('inf')
            num_selected_clients = helper.params['num_select_clients']
            num_clients = helper.params['total_clients']
            weight_accumulator = helper.init_weight_accumulator(helper.global_model)
            is_poison = helper.params['is_poison']
            folder_path = helper.params['folder_path']


            MPC = f"MPC_{helper.params['MPC']} NC_{helper.params['total_clients']}"
            epoch0 = f"_epoch_{helper.start_epoch}-{helper.params['epochs']} "
            m = f"m_{helper.params['trigger_pattern']} "
            info_dir = MPC + epoch0+ m
            params = helper.params
            common.initialize_csv(folder_path, attack_type, is_poison, info_dir, params)
            early_poison_epochs = helper.params['poison_epochs']

            for epoch in range(helper.start_epoch, helper.params['epochs'] + 1, helper.params['aggr_epoch_interval']):
                start_time = time.time()
                t = time.time()
                agent_name_keys = []
                adversarial_name_keys = []
                if helper.params['is_ASR_t']:
                    if epoch < 100:
                        if epoch in early_poison_epochs:
                            count = len(adversarial_clients_name_list)
                            if count < helper.params['num_select_clients']:
                                adversarial_name_keys = adversarial_clients_name_list
                            else:
                                adversarial_name_keys = random.sample(adversarial_clients_name_list,
                                                                      helper.params['num_select_clients'])
                            nonattacker = []
                            for adv in adversarial_clients_name_list:
                                if adv not in adversarial_name_keys:
                                    nonattacker.append(copy.deepcopy(adv))
                            if count < helper.params['num_select_clients']:
                                benign_num = helper.params['num_select_clients'] - len(adversarial_name_keys)
                                random_agent_name_keys = random.sample(benign_clients_name_list + nonattacker,
                                                                       benign_num)
                                agent_name_keys = adversarial_name_keys + random_agent_name_keys
                            else:
                                agent_name_keys = adversarial_name_keys
                        else:
                            agent_name_keys, adversarial_name_keys = helper.fixed_poison_epoch_namekeys(epoch,
                                                                                                        adversarial_clients_name_list,
                                                                                                        benign_clients_name_list)
                    else:
                        agent_name_keys, adversarial_name_keys = helper.fixed_poison_epoch_namekeys(epoch,
                                                                                                    adversarial_clients_name_list,
                                                                                                    benign_clients_name_list)
                else:
                    if helper.params['is_random_adversary']:
                        agent_name_keys, adversarial_name_keys=helper.random_namekeys(total_clients_name_list,adversarial_clients_name_list)
                    else:
                        agent_name_keys,adversarial_name_keys=helper.fixed_poison_epoch_namekeys(epoch,adversarial_clients_name_list,benign_clients_name_list)

                print(f'当前轮次:{epoch}')


                selected_clients = [helper.get_client_by_id(clients, i) for i in
                                    agent_name_keys]

                server.send_to_clients(clients)
                num_samples_dict = dict()
                epochs_submit_update_dict = dict()

                for i, client in enumerate(selected_clients):
                    client.train(common, epochs_submit_update_dict, helper, num_samples_dict,
                                 len(adversarial_name_keys), is_poison=is_poison, start_epoch=epoch)

                weight_accumulator, updates = server.accumulate_weight(helper, weight_accumulator,
                                                                       epochs_submit_update_dict,

                                                                       agent_name_keys, num_samples_dict)

                server.update_global_model(weight_accumulator, updates, helper, epoch, vis, adversarial_name_keys)


                weight_accumulator = helper.init_weight_accumulator(server.model)
                print("Global model performance")
                temp_global_epoch = epoch + helper.params['aggr_epoch_interval'] - 1
                epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest(helper=helper, epoch=temp_global_epoch,

                                                                               model=server.model, is_poison=False,

                                                                               visualize=True, agent_name_key="global")
                common.global_testMA_result.append(
                    [temp_global_epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])
                if helper.params['is_poison']:
                    epoch_loss, epoch_acc_p, epoch_corret, epoch_total = test.Mytest_poison(helper=helper,

                                                                                            epoch=temp_global_epoch,

                                                                                            model=server.model,

                                                                                            is_poison=True,

                                                                                            visualize=False,

                                                                                            agent_name_key="global")

                    common.global_triggerASR_result.append(
                        [temp_global_epoch, helper.params['test_trigger_indices'], epoch_loss, epoch_acc_p,
                         epoch_corret, epoch_total])
                if helper.params['vis_trigger_split_test']:
                    server.model.trigger_agent_test_vis(vis=vis, epoch=epoch, acc=epoch_acc_p, loss=None,

                                                        eid=helper.params['environment_name'],

                                                        name="global_combine")
                helper.save_model(attack_type, server.model, epoch=epoch, val_loss=epoch_loss)
            print('Saving all the graphs.')
            common.save_csv()
            common.clear_global_parameter()
        else:
            helper.load_data()
            print('dataset:', helper.params['type'])
            print('load data done')
            helper.creat_model()
            print('create model done')
            print(f'malicious clients：{helper.adversarial_namelist}')
            num_clients = helper.params['total_clients']
            criterion = nn.CrossEntropyLoss()
            clients = [Client(helper.global_model, helper.train_data[i], criterion, helper, i) for i in
                       range(num_clients)]
            is_poison = helper.params['is_poison']
            adversarial_clients = []
            for client in clients:
                if client.id in helper.adversarial_namelist:
                    adversarial_clients.append(client)

            # home_dir = os.path.expanduser('~')  # 用于服务器获取权限，注意服务器路径和本地路径存在差异
            # home_path=os.path.join(home_dir, 'LYZ/MR/SADBA')
            # home_path=generate_dataset_save_path(home_path,helper.params['best_trigger_position'],helper.params['optimal_sample_selection'],helper.params['full_combination_trigger'],helper.params['trigger_distribution_strategy'])
            home_path = params_loaded.get('saved_path')
            print(f"poison_data_saving_path：{data_saved_path}")
            subtask_type = f"NC={helper.params['total_clients']};m_shape={helper.params['setting']};MPC={helper.params['MPC']}"
            subtask_type_path = os.path.join(home_path, str(helper.params['attack']))
            final_path = os.path.join(subtask_type_path, subtask_type)
            print("saving path：", final_path)
            benign_dir = os.path.join(final_path, 'dataset/benign_dataset')
            adversarial_dir = os.path.join(final_path, 'dataset/adversarial_dataset')

            shared = SharedClass()
            if is_poison:
                best_pos=helper.poison_dataset(adversarial_clients)
                adversarial_dir += str(best_pos)
                shared.save_backdoor_dataset(adversarial_clients, helper, dir=adversarial_dir)
            shared.save_benign_dataset(clients, dir=benign_dir)







