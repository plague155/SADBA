import copy
import math

import visualization
import torch
import config
import numpy as np

class Server:
    def __init__(self,model,helper):
        self.model=model
        self.learning_rate=helper.params['gloabl_learning_rate']
        self.helper=helper
        # self.global_parameters=model.state_dict()

    def accumulate_weight(self,helper, weight_accumulator, epochs_submit_update_dict, state_keys,num_samples_dict):
        """
         return Args:
             updates: dict of (num_samples, update), where num_samples is the
                 number of training samples corresponding to the update, and update
                 is a list of variable weights
         """

        updates = dict()
        for i in range(0, len(state_keys)):
            local_model_update_list = epochs_submit_update_dict[state_keys[i]]
            update= dict()
            num_samples=num_samples_dict[state_keys[i]]

            for name, data in local_model_update_list[0].items():
                update[name] = torch.zeros_like(data)

            for j in range(0, len(local_model_update_list)):
                local_model_update_dict= local_model_update_list[j]
                for name, data in local_model_update_dict.items():
                    weight_accumulator[name].add_(local_model_update_dict[name])
                    update[name].add_(local_model_update_dict[name])
                    detached_data= data.cpu().detach().numpy()
                    # print(detached_data.shape)
                    detached_data=detached_data.tolist()
                    # print(detached_data)
                    local_model_update_dict[name]=detached_data # from gpu to cpu

            updates[state_keys[i]]=(num_samples,update)

        return weight_accumulator,updates

    def update_global_model(self,weight_accumulator, updates,helper,epoch,vis,adversarial_name_keys):
            # Average the models
            is_updated = self.average_shrink_models(weight_accumulator=weight_accumulator,

                                                      epoch_interval=helper.params['aggr_epoch_interval'])
            num_oracle_calls = 1




    @staticmethod
    def dp_noise(param, sigma):

        noised_layer = torch.cuda.FloatTensor(param.shape).normal_(mean=0, std=sigma)

        return noised_layer

    def average_shrink_models(self, weight_accumulator,  epoch_interval):
        """
        Perform FedAvg algorithm and perform some clustering on top of it.

        """

        for name, data in self.model.state_dict().items():
            if self.helper.params.get('tied', False) and name == 'decoder.weight':
                continue
            if 'num_batches_tracked' in name:
                continue

            update_per_layer = weight_accumulator[name] * (self.helper.params["gloabl_learning_rate"] / self.helper.params["total_clients"])
            # update_per_layer = weight_accumulator[name] * (self.params["eta"] / self.params["number_of_total_participants"])

            # update_per_layer = update_per_layer * 1.0 / epoch_interval
            if self.helper.params['diff_privacy']:
                update_per_layer.add_(self.dp_noise(data, self.helper.params['sigma']))

            data.add_(update_per_layer)

        return True

    def foolsgold_update(self, target_model, updates):
        client_grads = []
        alphas = []
        names = []
        for name, data in updates.items():
            client_grads.append(data[1])  # gradient
            alphas.append(data[0])  # num_samples
            names.append(name)

        adver_ratio = 0
        for i in range(0, len(names)):
            _name = names[i]
            if _name in self.helper.params['adversary_list']:
                adver_ratio += alphas[i]
        adver_ratio = adver_ratio / sum(alphas)
        poison_fraction = adver_ratio * self.helper.params['poisoning_per_batch'] / self.helper.params['batch_size']
        print(f'[foolsgold agg] training data poison_ratio: {adver_ratio}  data num: {alphas}')
        print(f'[foolsgold agg] considering poison per batch poison_fraction: {poison_fraction}')

        target_model.train()
        # train and update
        optimizer = torch.optim.SGD(target_model.parameters(), lr=self.helper.params['lr'],
                                    momentum=self.helper.params['momentum'],
                                    weight_decay=self.helper.params['decay'])

        optimizer.zero_grad()
        agg_grads, wv, alpha = self.fg.aggregate_gradients(client_grads, names)
        for i, (name, params) in enumerate(target_model.named_parameters()):
            agg_grads[i] = agg_grads[i] * self.helper.params["eta"]
            if params.requires_grad:
                params.grad = agg_grads[i].to(config.device)
        optimizer.step()
        wv = wv.tolist()
        return True, names, wv, alpha
    def geometric_median_update(self, target_model, updates, maxiter=4, eps=1e-5, verbose=False, ftol=1e-6,
                                max_update_norm=None):
        """Computes geometric median of atoms with weights alphas using Weiszfeld's Algorithm
               """
        points = []
        alphas = []
        names = []
        for name, data in updates.items():
            points.append(data[1])  # update
            alphas.append(data[0])  # num_samples
            names.append(name)

        adver_ratio = 0
        for i in range(0, len(names)):
            _name = names[i]
            if _name in self.helper.params['adversary_list']:
                adver_ratio += alphas[i]
        adver_ratio = adver_ratio / sum(alphas)
        poison_fraction = adver_ratio * self.helper.params['poisoning_per_batch'] / self.helper.params['batch_size']
        print(f'[rfa agg] training data poison_ratio: {adver_ratio}  data num: {alphas}')
        print(f'[rfa agg] considering poison per batch poison_fraction: {poison_fraction}')

        alphas = np.asarray(alphas, dtype=np.float64) / sum(alphas)
        alphas = torch.from_numpy(alphas).float()

        # alphas.float().to(task_params.device)
        median = self.weighted_average_oracle(points, alphas)
        num_oracle_calls = 1

        # logging
        obj_val = self.geometric_median_objective(median, points, alphas)
        logs = []
        log_entry = [0, obj_val, 0, 0]
        logs.append(log_entry)
        if verbose:
            print('Starting Weiszfeld algorithm')
            print(log_entry)
        print(f'[rfa agg] init. name: {names}, weight: {alphas}')
        # start
        wv = None
        for i in range(maxiter):
            prev_median, prev_obj_val = median, obj_val
            weights = torch.tensor([alpha / max(eps, self.l2dist(median, p)) for alpha, p in zip(alphas, points)],
                                   dtype=alphas.dtype)
            weights = weights / weights.sum()
            median = self.weighted_average_oracle(points, weights)
            num_oracle_calls += 1
            obj_val = self.geometric_median_objective(median, points, alphas)
            log_entry = [i + 1, obj_val,
                         (prev_obj_val - obj_val) / obj_val,
                         self.l2dist(median, prev_median)]
            logs.append(log_entry)
            if verbose:
                print(log_entry)
            if abs(prev_obj_val - obj_val) < ftol * obj_val:
                break
            print(
                f'[rfa agg] iter:  {i}, prev_obj_val: {prev_obj_val}, obj_val: {obj_val}, abs dis: {abs(prev_obj_val - obj_val)}')
            print(f'[rfa agg] iter:  {i}, weight: {weights}')
            wv = copy.deepcopy(weights)
        alphas = [self.l2dist(median, p) for p in points]

        update_norm = 0
        for name, data in median.items():
            update_norm += torch.sum(torch.pow(data, 2))
        update_norm = math.sqrt(update_norm)

        if max_update_norm is None or update_norm < max_update_norm:
            for name, data in target_model.state_dict().items():
                update_per_layer = median[name] * (self.helper["eta"])
                if self.helper.params['diff_privacy']:
                    update_per_layer.add_(self.dp_noise(data, self.helper.params['sigma']))
                data.add_(update_per_layer)
            is_updated = True
        else:
            print('\t\t\tUpdate norm = {} is too large. Update rejected'.format(update_norm))
            is_updated = False

        # utils.csv_record.add_weight_result(names, wv.cpu().numpy().tolist(), alphas)

        return num_oracle_calls, is_updated, names, wv.cpu().numpy().tolist(), alphas

    def send_to_clients(self,clients):
        for client in clients:
            client.set_parameters(self.model.state_dict())

    @staticmethod
    def l2dist(p1, p2):
        """L2 distance between p1, p2, each of which is a list of nd-arrays"""
        squared_sum = 0
        for name, data in p1.items():
            squared_sum += torch.sum(torch.pow(p1[name] - p2[name], 2))
        return math.sqrt(squared_sum)


    def geometric_median_objective(self,median, points, alphas):
        """Compute geometric median objective."""
        temp_sum = 0
        for alpha, p in zip(alphas, points):
            temp_sum += alpha * self.l2dist(median, p)
        return temp_sum

    @staticmethod
    def weighted_average_oracle(points, weights):
        """Computes weighted average of atoms with specified weights

        Args:
            points: list, whose weighted average we wish to calculate
                Each element is a list_of_np.ndarray
            weights: list of weights of the same length as atoms
        """
        tot_weights = torch.sum(weights)

        weighted_updates = dict()

        for name, data in points[0].items():
            weighted_updates[name] = torch.zeros_like(data)
        for w, p in zip(weights, points):  # 对每一个agent
            for name, data in weighted_updates.items():
                temp = (w / tot_weights).float().to(config.device)
                temp = temp * (p[name].float())
                # temp = w / tot_weights * p[name]
                if temp.dtype != data.dtype:
                    temp = temp.type_as(data)
                data.add_(temp)

        return weighted_updates


