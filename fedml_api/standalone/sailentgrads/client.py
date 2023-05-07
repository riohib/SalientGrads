import copy
import logging
import math

import numpy as np
import pdb
import torch
import torch.utils.data as data
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import Subset

class Client:

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device,
                 model_trainer, logger):
        self.logger = logger
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.logger.info("self.local_sample_number = " + str(self.local_sample_number))
        self.args = args
        self.device = device
        self.model_trainer = model_trainer

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
    
    def get_mean_sailency_scores(self, final_sailency_list):
        #model = self.model
        size = len(final_sailency_list)
        sailency_average={}

        # Loop through each dictionary in the list
        for grad_abs in final_sailency_list:
            # Loop through each key-value pair in the dictionary
            for k, v in dict(grad_abs).items():
                # If the key already exists in the averaged dictionary, add the value to the existing sum
                if k in sailency_average:
                    sailency_average[k] += v
                # Otherwise, create a new key in the averaged dictionary and initialize it with the value
                else:
                    sailency_average[k] = v.clone().detach()

        # Divide the summed values by the number of dictionaries in the list to get the average
        num_dicts = size
        for k in sailency_average.keys():
            sailency_average[k] /= num_dicts

        return sailency_average
    
    def generate_sailency_scores_from_each_client(self, itersnip_iteration=20, stratified_sampling=False):
        local_training_data = self.local_training_data
        final_sailency_list=[]
        if stratified_sampling:
            y = np.concatenate(np.array([label for _, label in local_training_data]))
            X = np.concatenate([image for image, _ in local_training_data])
            #splitter = StratifiedShuffleSplit(n_splits=itersnip_iteration, test_size=0.1, random_state=42)
            splitter = StratifiedKFold(n_splits=25, shuffle=True, random_state=42)
            for train_index, _ in splitter.split(X, y):
                train_mini_batch = [local_training_data.dataset[i] for i in train_index]
                train_X = torch.cat([torch.Tensor(x).unsqueeze(0) for x, _ in train_mini_batch], dim=0)
                train_y = torch.Tensor([y for _, y in train_mini_batch]).long()
                local_abs_grads = self.model_trainer.get_snip_scores((train_X, train_y))
                final_sailency_list.append(local_abs_grads)
        else:
            for i in range(itersnip_iteration):
                train_mini_batch = next(iter(local_training_data))
                local_abs_grads = self.model_trainer.get_snip_scores(train_mini_batch)
                final_sailency_list.append(local_abs_grads)
            
        mean_sailency_score = self.get_mean_sailency_scores(final_sailency_list)
        return mean_sailency_score

    def get_sample_number(self):
        return self.local_sample_number

    # def train(self, w, masks,round):
    #     # downlink params
    #     # num_comm_params = self.model_trainer.count_communication_params(w)
    #     # self.model_trainer.set_model_params(w)
    #     # self.model_trainer.set_masks(masks)
    #     # self.model_trainer.set_id(self.client_idx)
    #     # tst_results = self.model_trainer.test(self.local_test_data, self.device, self.args)
    #     # self.logger.info("test acc on this client before {} / {} : {:.2f}".format(tst_results['test_correct'], tst_results['test_total'], tst_results['test_acc']))

    #     self.model_trainer.train(self.local_training_data, self.device, self.args, round)
    #     #weights = self.model_trainer.get_model_params()
    #     weights = w
    #     self.model_trainer.set_model_params(weights)
    #     tst_results = self.model_trainer.test(self.local_test_data, self.device, self.args)
    #     self.logger.info("test acc on this client after {} / {} : {:.2f}".format(tst_results['test_correct'], tst_results['test_total'], tst_results['test_acc']))

    #     # update = {}
    #     # for name in weights:
    #     #     update[name] = weights[name] - w[name]

    #     #No Need to update the weights by subtracting old weights
    #     update=weights

    #     self.logger.info("-----------------------------------")
        
    #     sparse_flops_per_data = self.model_trainer.count_training_flops_per_sample()
    #     full_flops = self.model_trainer.count_full_flops_per_sample()
    #     self.logger.info("training flops per data {}".format(sparse_flops_per_data))
    #     self.logger.info("full flops for search {}".format(full_flops))
    #     # we train the data for `self.args.epochs` epochs, and forward one epoch of data with full density to screen gradient.
    #     training_flops = self.args.epochs*self.local_sample_number*sparse_flops_per_data+\
    #                      self.args.batch_size* full_flops

    #     # uplink params
    #     num_comm_params += self.model_trainer.count_communication_params(update)
    #     return masks,  weights, update, training_flops, num_comm_params, tst_results

    def train(self, w_global,round, masks):
        # self.logger.info(sum([torch.sum(w_per[name]) for name in w_per]))
        num_comm_params = self.model_trainer.count_communication_params(w_global)
        self.model_trainer.set_model_params(w_global)

        model_sparsity = self.model_trainer.get_model_sps()
        print("Sparsity before training is : " + str(model_sparsity))
        self.model_trainer.set_id(self.client_idx)
        self.model_trainer.train(self.local_training_data, self.device, self.args, round, masks)
        weights = self.model_trainer.get_model_params()
        # self.logger.info( "training_flops{}".format( self.model_trainer.count_training_flops_per_sample()))
        # self.logger.info("full{}".format(self.model_trainer.count_full_flops_per_sample()))
        training_flops = self.args.epochs * self.local_sample_number * self.model_trainer.count_training_flops_per_sample()
        sparse_flops_per_data = self.model_trainer.count_training_flops_per_sample()
        full_flops = self.model_trainer.count_full_flops_per_sample()
        self.logger.info("training flops per data {}".format(sparse_flops_per_data))
        self.logger.info("full flops for search {}".format(full_flops))
        # we train the data for `self.args.epochs` epochs, and forward one epoch of data with full density to screen gradient.
        #training_flops = self.args.epochs*self.local_sample_number*sparse_flops_per_data+\
    #                      self.args.batch_size* full_flops
        num_comm_params += self.model_trainer.count_communication_params(weights)
        self.logger.info("communication parameters for search {}".format(num_comm_params))
        return  weights,training_flops,num_comm_params


    def fire_mask(self, masks, weights, round):
        drop_ratio = self.args.anneal_factor / 2 * (1 + np.cos((round * np.pi) / self.args.comm_round))
        new_masks = copy.deepcopy(masks)

        num_remove = {}
        for name in masks:
            num_non_zeros = torch.sum(masks[name])
            num_remove[name] = math.ceil(drop_ratio * num_non_zeros)
            temp_weights = torch.where(masks[name] > 0, torch.abs(weights[name]), 100000 * torch.ones_like(weights[name]))
            x, idx = torch.sort(temp_weights.view(-1).to(self.device))
            new_masks[name].view(-1)[idx[:num_remove[name]]] = 0
        return new_masks, num_remove


    # we only update the private components of client's mask
    def regrow_mask(self, masks,  num_remove, gradient=None):
        new_masks = copy.deepcopy(masks)
        for name in masks:
            # if name not in public_layers:
                # if "conv" in name:
                if not self.args.dis_gradient_check:
                    temp = torch.where(masks[name] == 0, torch.abs(gradient[name]), -100000 * torch.ones_like(gradient[name]))
                    sort_temp, idx = torch.sort(temp.view(-1).to(self.device), descending=True)
                    new_masks[name].view(-1)[idx[:num_remove[name]]] = 1
                else:
                    temp = torch.where(masks[name] == 0, torch.ones_like(masks[name]),torch.zeros_like(masks[name]) )
                    idx = torch.multinomial( temp.flatten().to(self.device),num_remove[name], replacement=False)
                    new_masks[name].view(-1)[idx]=1
        return new_masks


    def local_test(self, w_per, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        
        self.model_trainer.set_model_params(w_per)
        
        model_sparsity = self.model_trainer.get_model_sps()
        print("Sparsity before testing in local is : " + str(model_sparsity))
        
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
