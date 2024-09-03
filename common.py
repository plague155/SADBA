import logging

import json
import os
import csv

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)
vis=None
train_result = []
local_testMA_result = []
global_testMA_result = []
local_triggerASR_result = []
global_triggerASR_result = []
backdoor_localmodel_distance = []

def clear_global_parameter():
        global train_result
        global local_testMA_result
        global global_testMA_result
        global local_triggerASR_result
        global global_triggerASR_result
        global backdoor_localmodel_distance

        # Clear the lists
        train_result.clear()
        local_testMA_result.clear()
        global_testMA_result.clear()
        local_triggerASR_result.clear()
        global_triggerASR_result.clear()
        backdoor_localmodel_distance.clear()
class Path:
    def __init__(self):
        self.train_path = None
        self.local_testMA_path = None
        self.global_testMA_path = None
        self.local_triggerASR_path = None
        self.global_triggerASR_path = None
        self.backdoor_localmodel_distance_path = None
path = Path()

def initialize_one_csv(file_path,header):
    with open(file_path,'w',newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

def save_result_csv(file_path,train_result):
    with open(file_path,'a',newline='') as file:
        writer=csv.writer(file)
        writer.writerows(train_result)

def save_csv():
    print(f"train_results保存路径：{path.train_path}")
    save_result_csv(str(path.train_path),train_result)
    save_result_csv(str(path.local_testMA_path),local_testMA_result)
    save_result_csv(str(path.global_testMA_path),global_testMA_result)
    save_result_csv(str(path.local_triggerASR_path),local_triggerASR_result)
    save_result_csv(str(path.global_triggerASR_path),global_triggerASR_result)
    save_result_csv(str(path.backdoor_localmodel_distance_path),backdoor_localmodel_distance)



def initialize_csv(folder_path,attack_type,is_poison,info_dir,params):

    home_path=os.path.join(folder_path, f'{attack_type}/{info_dir}')
    os.makedirs(home_path,exist_ok=True)

    info_path=os.path.join(home_path,"training_params.txt")
    params_str=json.dumps(params,indent=4)
    with open(info_path,'w') as file:
        file.write(params_str)

    train_fileHeader = ["client_id", "rounds", "epoch", "local_epoch", "avg_loss", "acc", "correct_data",
                        "total_data"]

    local_testMA_fileHeader = ["client_id", "epoch", "avg_loss", "acc", "correct_data", "total_data"]

    global_testMA_fileHeader = ["epoch", "epoch_loss", "epoch_acc", "epoch_correct", "epoch_total"]

    path.train_path = os.path.join(folder_path, f'{attack_type}/{info_dir}/train_result.csv')
    path.local_testMA_path = os.path.join(folder_path, f'{attack_type}/{info_dir}/local_testMA.csv')
    path.global_testMA_path = os.path.join(folder_path, f'{attack_type}/{info_dir}/global_testMA.csv')
    initialize_one_csv(path.train_path, train_fileHeader)
    initialize_one_csv(path.local_testMA_path, local_testMA_fileHeader)
    initialize_one_csv(path.global_testMA_path, global_testMA_fileHeader)
    if is_poison:
        local_triggerASR_fileHeader = ["malicious_client_id", "test_trigger_type", "epoch", "avg_loss", "acc",
                                       "correct_data", "total_data"]
        global_triggerASR_fileHeader = ["epoch", "test_trigger_type", "epoch_loss", "epoch_acc", "epoch_correct",
                                        "epoch_total"]
        backdoor_localmodel_distance_fileHeader=["malicious_client_id","epoch","distance"]

        path.local_triggerASR_path = os.path.join(folder_path, f'{attack_type}/{info_dir}/local_triggerASR.csv')
        path.global_triggerASR_path = os.path.join(folder_path, f'{attack_type}/{info_dir}/global_triggerASR.csv')
        path.backdoor_localmodel_distance_path = os.path.join(folder_path, f'{attack_type}/{info_dir}/backdoor_localmodel_distance.csv')
        initialize_one_csv(path.local_triggerASR_path, local_triggerASR_fileHeader)
        initialize_one_csv(path.global_triggerASR_path, global_triggerASR_fileHeader)
        initialize_one_csv(path.backdoor_localmodel_distance_path, backdoor_localmodel_distance_fileHeader)

