
import torch
import torch.nn as nn
import config
from common import vis

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def Mytest(helper, epoch,
           model, is_poison=False, visualize=False, agent_name_key=""):
    model.eval()
    total_loss = 0
    correct = 0
    dataset_size = 0

    data_iterator = helper.test_data
    for batch_id, batch in enumerate(data_iterator):
            data, targets = helper.get_batch(data_iterator, batch, evaluation=True)
            dataset_size += len(data)
            output = model(data)
            total_loss += nn.functional.cross_entropy(output, targets,
                                                      reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(dataset_size))  if dataset_size!=0 else 0
    total_l = total_loss / dataset_size if dataset_size!=0 else 0

    print('___Test  是否注入后门: {}, 当前全局轮次: {}: 平均损失: {:.4f}, '
                     '准确率: {}/{} ({:.4f}%)'.format( is_poison, epoch,
                                                        total_l, correct, dataset_size,
                                                        acc))

    model.train()
    return (total_l, acc, correct, dataset_size)


def Mytest_poison(helper, epoch,
                  model, is_poison=False, visualize=True, agent_name_key="",test_type=None):
    model.eval()
    total_loss = 0.0
    correct = 0

    poison_data_count = 0

    data_iterator = helper.test_data_poison
    for batch_id, batch in enumerate(data_iterator):

            data, labels= batch


            poison_data_count += len(data)
            output = model(data)
            total_loss += nn.functional.cross_entropy(output, labels,
                                                      reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()
    print('{}  {}'.format(float(correct),poison_data_count))
    acc = 100.0 * (float(correct) / float(poison_data_count))  if poison_data_count!=0 else 0
    total_l = total_loss / poison_data_count if poison_data_count!=0 else 0
    print('___Test  是否注入后门: {}, 当前全局轮次: {}: 平均损失: {:.4f}, '
                     '准确率: {}/{} ({:.4f}%)'.format( is_poison, epoch,
                                                        total_l, correct, poison_data_count,
                                                        acc))

    model.train()
    return total_l, acc, correct, poison_data_count


def Mytest_poison_trigger(helper, model, adver_trigger_index,test_type=None):
    model.eval()
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    poison_data_count = 0

    data_iterator = helper.test_data_poison
    adv_index = adver_trigger_index
    for batch_id, batch in enumerate(data_iterator):
            data, targets, poison_num = helper.get_poison_batch(batch, adversarial_index=adv_index, evaluation=True,test_type=test_type)

            poison_data_count += poison_num
            dataset_size += len(data)
            output = model(data)
            total_loss += nn.functional.cross_entropy(output, targets,
                                                      reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(poison_data_count)) if poison_data_count!=0 else 0
    total_l = total_loss / poison_data_count if poison_data_count!=0 else 0

    model.train()
    return total_l, acc, correct, poison_data_count


def Mytest_poison_agent_trigger(helper, model, agent_name_key):
    model.eval()
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    poison_data_count = 0
    data_iterator = helper.test_data_poison
    adv_index = -1
    for temp_index in range(0, len(helper.params['total_list'])):
            if int(agent_name_key) == helper.params['total_list'][temp_index]:
                adv_index = temp_index
                break
    for batch_id, batch in enumerate(data_iterator):
            data, targets, poison_num = helper.get_poison_batch(batch, adversarial_index=adv_index, evaluation=True)

            poison_data_count += poison_num
            dataset_size += len(data)
            output = model(data)
            total_loss += nn.functional.cross_entropy(output, targets,
                                                      reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(poison_data_count)) if poison_data_count!=0 else 0
    total_l = total_loss / poison_data_count if poison_data_count!=0 else 0

    model.train()
    return total_l, acc, correct, poison_data_count
