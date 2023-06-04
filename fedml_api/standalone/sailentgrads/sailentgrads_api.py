import copy
import logging
import math
import pickle
import random
import time

import pdb
import numpy as np
import torch
from collections import OrderedDict
#from fedml_api.standalone.sailentgrads import client
from fedml_api.standalone.sailentgrads.client import Client
from fedml_api.standalone.DisPFL.slim_util import model_difference
from fedml_api.standalone.DisPFL.slim_util import hamming_distance
from fedml_api.standalone.sailentgrads.snip import get_mask_from_grads, get_mean_snip_scores, get_snip_scores

class SailentGradsAPI(object):
    def __init__(self, dataset, device, args, model_trainer, logger):
        self.logger = logger
        self.device = device
        self.args = args
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_counts] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.class_counts = class_counts
        self.model_trainer = model_trainer
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)
        self.init_stat_info()

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        self.logger.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_in_total):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer, self.logger)
            self.client_list.append(c)
        self.logger.info("############setup_clients (END)#############")
    
    def generate_global_mask_snip(self):
        sparsity_ratio = self.args.dense_ratio
        all_client_snip_scores=[]

        for clnt_idx in range(self.args.client_num_in_total):
            client = self.client_list[clnt_idx]
            #Updates the mask, but not the case for SNIP
            #new_mask, w_local_mdl, updates_matrix[clnt_idx], training_flops, num_comm_params, tst_results = client.generate_global_mask_from_clients()
            client_local_scores = client.generate_sailency_scores_from_each_client(itersnip_iteration = self.args.itersnip_iteration, stratified_sampling=self.args.stratified_sampling)
            all_client_snip_scores.append(client_local_scores)
        

        self.logger.info("@@@@@Aggregate the local masks@@@@@@@@@@@@@@@@@@@")
        #Aggregate the masks
        averaged_scores = get_mean_snip_scores(all_client_snip_scores)
        params = self.model_trainer.get_trainable_params()
        
        #Generate the final mask
        keep_mask, keep_layers, final_weight_mask = get_mask_from_grads(self.model_trainer, averaged_scores, sparsity_ratio, params)
        return final_weight_mask
    
    def get_model_sps_for_weight(self, custom_weights):
            nonzero = total = 0
            for keys in custom_weights:
                param = custom_weights[keys]
                if 'mask' not in keys:
                    tensor = param.detach().clone()
                    # nz_count.append(torch.count_nonzero(tensor))
                    nz_count = torch.count_nonzero(tensor).item()
                    total_params = tensor.numel()
                    nonzero += nz_count
                    total += total_params
            
            tensor = None
            # print(f"TOTAL: {total}")
            abs_sps = 100 * (total-nonzero) / total
            return abs_sps


    def train(self):

        params = self.model_trainer.get_trainable_params() #Model is same, so get parameters

        #Generate sailency scores, and update masks

        if self.args.snip_mask:
            temp = self.generate_global_mask_snip()
            mask_pers_local = [copy.deepcopy(temp) for i in range(self.args.client_num_in_total)]
        
        w_global = self.model_trainer.get_model_params() #Make a copy of model parameters
        w_per_mdls = []  #To store masked model parameters
        
        for clnt in range(self.args.client_num_in_total):  #For each client
            w_per_mdls.append(copy.deepcopy(w_global))
            for name in mask_pers_local[clnt]:
                w_per_mdls[clnt][name] = w_global[name] #* mask_pers_local[clnt][name]
        

        for round_idx in range(self.args.comm_round):
            self.logger.info("################Communication round : {}".format(round_idx))
            w_locals = []
            weight_locals=[]
            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total,
                                                   self.args.client_num_per_round)
            client_indexes = np.sort(client_indexes)

            self.logger.info("client_indexes = " + str(client_indexes))

            for cur_clnt in client_indexes:
                self.logger.info('@@@@@@@@@@@@@@@@ Training Client CM({}): {}'.format(round_idx, cur_clnt))
                # update dataset
                client = self.client_list[cur_clnt]
                # update meta components in personal network
                w_per,training_flops,num_comm_params = client.train(copy.deepcopy(w_global), round_idx, mask_pers_local[cur_clnt])
                mask_sps = self.get_model_sps_for_weight(w_per)
                w_per_mdls[cur_clnt] = copy.deepcopy(w_per)
                # self.logger.info("local weights = " + str(w))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w_per)))
                weight_locals.append(copy.deepcopy(w_per))
                self.stat_info["sum_training_flops"] += training_flops
                self.stat_info["sum_comm_params"] += num_comm_params
            # update global meta weights
            w_global = self._aggregate(w_locals)
            #Just testing the sparsity
            mask_sps = self.get_model_sps_for_weight(w_global)
            self._test_on_all_clients(w_global, w_per_mdls, round_idx)

        self._test_on_all_clients(w_global, w_per_mdls, -1)


    #Sampling the client

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        self.logger.info("client_indexes = %s" % str(client_indexes))
        return client_indexes
    

    #Choosing the type of topology

    def _benefit_choose(self, round_idx, cur_clnt, client_num_in_total, client_num_per_round, dist_local, total_dist, cs = False, active_ths_rnd = None):
        if client_num_in_total == client_num_per_round:
            # If one can communicate with all others and there is no bandwidth limit
            client_indexes = [client_index for client_index in range(client_num_in_total)]
            return client_indexes

        if cs == "random":
            # Random selection of available clients
            num_clients = min(client_num_per_round, client_num_in_total)
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
            while cur_clnt in client_indexes:
                client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)

        elif cs == "ring":
            # Ring Topology in Decentralized setting
            left = (cur_clnt - 1 + client_num_in_total) % client_num_in_total
            right = (cur_clnt + 1) % client_num_in_total
            client_indexes = np.asarray([left, right])

        elif cs == "full":
            # Fully-connected Topology in Decentralized setting
            client_indexes = np.array(np.where(active_ths_rnd==1)).squeeze()
            client_indexes = np.delete(client_indexes, int(np.where(client_indexes==cur_clnt)[0]))
        return client_indexes
    

    #This is a custom aggregate implementation for sanity check - No need to use now
    def average_weights(self, weights_list):
         #create an empty OrderedDict to store the averaged weights
        averaged_weights = OrderedDict()

        # iterate over the layers in the first OrderedDict to determine the size of each weight tensor
        first_weights = weights_list[0]
        for key in first_weights:
            size = first_weights[key].size()
            dtype = first_weights[key].dtype
            averaged_weights[key] = torch.zeros(size, dtype=dtype)

        # iterate over the weights in each layer and accumulate the sum
        num_weights = len(weights_list)
        for weights in weights_list:
            for key in weights:
                averaged_weights[key] += weights[key] / num_weights
        
        return averaged_weights
    
    
    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, _) = w_locals[idx]
            training_num += sample_num
        w_global ={}
        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    w_global[k] = local_model_params[k] * w
                else:
                    w_global[k] += local_model_params[k] * w
        return w_global


    #Global test on all client and local test on all clients
    def _test_on_all_clients(self, w_global, w_per_mdls, round_idx):

        self.logger.info("################global_test_on_all_clients : {}".format(round_idx))

        g_test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        p_test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        for client_idx in range(self.args.client_num_in_total):
            # test data
            client = self.client_list[client_idx]
            g_test_local_metrics = client.local_test(w_global, True)
            g_test_metrics['num_samples'].append(copy.deepcopy(g_test_local_metrics['test_total']))
            g_test_metrics['num_correct'].append(copy.deepcopy(g_test_local_metrics['test_correct']))
            g_test_metrics['losses'].append(copy.deepcopy(g_test_local_metrics['test_loss']))

            p_test_local_metrics = client.local_test(w_per_mdls[client_idx], True)
            p_test_metrics['num_samples'].append(copy.deepcopy(p_test_local_metrics['test_total']))
            p_test_metrics['num_correct'].append(copy.deepcopy(p_test_local_metrics['test_correct']))
            p_test_metrics['losses'].append(copy.deepcopy(p_test_local_metrics['test_loss']))

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break
        # test on test dataset
        g_test_acc = sum([np.array(g_test_metrics['num_correct'][i]) / np.array(g_test_metrics['num_samples'][i]) for i in
                        range(self.args.client_num_in_total)]) / self.args.client_num_in_total
        g_test_loss = sum([np.array(g_test_metrics['losses'][i]) / np.array(g_test_metrics['num_samples'][i]) for i in
                         range(self.args.client_num_in_total)]) / self.args.client_num_in_total

        p_test_acc = sum(
            [np.array(p_test_metrics['num_correct'][i]) / np.array(p_test_metrics['num_samples'][i]) for i in
             range(self.args.client_num_in_total)]) / self.args.client_num_in_total
        p_test_loss = sum([np.array(p_test_metrics['losses'][i]) / np.array(p_test_metrics['num_samples'][i]) for i in
                           range(self.args.client_num_in_total)]) / self.args.client_num_in_total


        stats = {'global_test_acc': g_test_acc, 'global_test_loss': g_test_loss}
        self.stat_info["global_test_acc"].append(g_test_acc)
        self.logger.info(stats)

        stats = {'person_test_acc': p_test_acc, 'person_test_loss': p_test_loss}
        self.stat_info["person_test_acc"].append(p_test_acc)
        self.logger.info(stats)


    #Individual local tests
    def _local_test_on_all_clients(self, tst_results_ths_round, round_idx):
            self.logger.info("################local_test_on_all_clients after local training in communication round: {}".format(round_idx))
            test_metrics = {
                'num_samples': [],
                'num_correct': [],
                'losses': []
            }
            for client_idx in range(self.args.client_num_in_total):
                # test data
                test_metrics['num_samples'].append(copy.deepcopy(tst_results_ths_round[client_idx]['test_total']))
                test_metrics['num_correct'].append(copy.deepcopy(tst_results_ths_round[client_idx]['test_correct']))
                test_metrics['losses'].append(copy.deepcopy(tst_results_ths_round[client_idx]['test_loss']))

                """
                Note: CI environment is CPU-based computing. 
                The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
                """
                if self.args.ci == 1:
                    break

            # # test on test dataset
            test_acc = sum([test_metrics['num_correct'][i] / test_metrics['num_samples'][i] for i in range(self.args.client_num_in_total) ] )/self.args.client_num_in_total
            test_loss = sum([np.array(test_metrics['losses'][i]) / np.array(test_metrics['num_samples'][i]) for i in range(self.args.client_num_in_total)])/self.args.client_num_in_total

            stats = {'test_acc': test_acc, 'test_loss': test_loss}

            self.logger.info(stats)
            self.stat_info["old_mask_test_acc"].append(test_acc)

    #Calculating the FLOPS
    def record_avg_inference_flops(self, w_global, mask_pers=None):
        inference_flops=[]
        for client_idx in range(self.args.client_num_in_total):

            if mask_pers== None:
                inference_flops += [self.model_trainer.count_inference_flops(w_global)]
            else:
                w_per = {}
                for name in mask_pers[client_idx]:
                    w_per[name] = w_global[name] *mask_pers[client_idx][name]
                inference_flops+= [self.model_trainer.count_inference_flops(w_per)]
        avg_inference_flops = sum(inference_flops)/len(inference_flops)
        self.stat_info["avg_inference_flops"]=avg_inference_flops


    def init_stat_info(self, ):
        self.stat_info = {}
        self.stat_info["label_num"] =self.class_counts
        self.stat_info["sum_comm_params"] = 0
        self.stat_info["sum_training_flops"] = 0
        self.stat_info["avg_inference_flops"] = 0
        self.stat_info["old_mask_test_acc"] = []
        self.stat_info["new_mask_test_acc"] = []
        self.stat_info["final_masks"] = []
        self.stat_info["mask_dis_matrix"] = []

        self.stat_info["global_test_acc"] = []
        self.stat_info["person_test_acc"] = []


