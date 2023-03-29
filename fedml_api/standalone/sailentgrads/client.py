import copy
import logging
import math

import numpy as np
import pdb
import torch
import torch.utils.data as data

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
    
    def generate_sailency_scores_from_each_client(self):
        local_training_data = self.local_training_data
        train_mini_batch = next(iter(local_training_data))
        local_abs_grads = self.model_trainer.get_snip_scores(train_mini_batch)
        return local_abs_grads

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
        self.model_trainer.set_id(self.client_idx)
        self.model_trainer.train(self.local_training_data, self.device, self.args, round, masks)
        weights = self.model_trainer.get_model_params()
        # self.logger.info( "training_flops{}".format( self.model_trainer.count_training_flops_per_sample()))
        # self.logger.info("full{}".format(self.model_trainer.count_full_flops_per_sample()))
        training_flops = self.args.epochs * self.local_sample_number * self.model_trainer.count_training_flops_per_sample()
        num_comm_params += self.model_trainer.count_communication_params(weights)
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
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
