import copy
import logging
import math
import pickle
import random
import time

import pdb
import numpy as np
import torch

#from fedml_api.standalone.sailentgrads import client
from fedml_api.standalone.sailentgrads.client import Client
from fedml_api.standalone.DisPFL.slim_util import model_difference
from fedml_api.standalone.DisPFL.slim_util import hamming_distance

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
    
    # def generate_global_mask_snip(self):
    #     sparsity_ratio = self.args.dense_ratio
    #     all_client_snip_masks=[]

    #     #Get Sailency Scores for all clients
    #     for round_idx in range(self.args.comm_round):
    #         self.logger.info("################ Masks generation round : {}".format(round_idx))

    #         active_ths_rnd = np.random.choice([0, 1], size = self.args.client_num_in_total, p = [1.0 - self.args.active, self.args.active])
            
    #         #w_per_mdls_lstrd = copy.deepcopy(w_per_mdls)
    #         #mask_pers_shared_lstrd = copy.deepcopy(mask_pers_shared)

    #         tst_results_ths_round = []
    #         final_tst_results_ths_round = []
    #         for clnt_idx in range(self.args.client_num_in_total):
    #             client = self.client_list[clnt_idx]

    #             if active_ths_rnd[clnt_idx] == 0:
    #                 self.logger.info('@@@@@@@@@@@@@@@@ Client Drop this round CM({}) with spasity : {}'.format(round_idx, clnt_idx))
    #             #Updates the mask, but not the case for SNIP
    #             #new_mask, w_local_mdl, updates_matrix[clnt_idx], training_flops, num_comm_params, tst_results = client.generate_global_mask_from_clients()
    #             with torch.no_grad():
    #                 client_local_mask = client.generate_local_mask_from_each_client()
    #                 all_client_snip_masks.append(client_local_mask)
        

    #     self.logger.info("@@@@@Aggregare the local masks@@@@@@@@@@@@@@@@@@@")
    #     #Aggregate the masks
    #     averaged_scores = self.model_trainer.get_mean_snip_scores(all_client_snip_masks)
    #     params = self.model_trainer.get_trainable_params()
        
    #     #Generate the final mask
    #     keep_mask, keep_layers, final_weight_mask = self.model_trainer.get_mask_from_grads(averaged_scores, sparsity_ratio, params)


                
    #     return final_weight_mask
    

    def generate_global_mask_snip(self):
        sparsity_ratio = self.args.dense_ratio
        all_client_snip_scores=[]

        for clnt_idx in range(self.args.client_num_in_total):
            client = self.client_list[clnt_idx]
            #Updates the mask, but not the case for SNIP
            #new_mask, w_local_mdl, updates_matrix[clnt_idx], training_flops, num_comm_params, tst_results = client.generate_global_mask_from_clients()
            client_local_scores = client.generate_sailency_scores_from_each_client()
            all_client_snip_scores.append(client_local_scores)
        

        self.logger.info("@@@@@Aggregate the local masks@@@@@@@@@@@@@@@@@@@")
        #Aggregate the masks
        averaged_scores = self.model_trainer.get_mean_snip_scores(all_client_snip_scores)
        params = self.model_trainer.get_trainable_params()
        
        #Generate the final mask
        keep_mask, keep_layers, final_weight_mask = self.model_trainer.get_mask_from_grads(averaged_scores, sparsity_ratio, params)


                
        return final_weight_mask
    

   

    def train_old(self):

        # for first initialization, all the weights and the masks are the same
        # 在加入decentralized training时，所有client公用一个personalized mask和一个global model
        # different_initial 控制初始的client personalized mask是否是相同的，默认是相同的 即different_initial=False
        # masks = self.model_trainer.init_masks()
        params = self.model_trainer.get_trainable_params() #Model is same, so get parameters

        w_spa = [self.args.dense_ratio for i in range(self.args.client_num_in_total)] #Set if sparsity is different for clients

        
        if self.args.uniform:
            sparsities = self.model_trainer.calculate_sparsities(params,distribution="uniform", sparse = self.args.dense_ratio)
        else:
            sparsities = self.model_trainer.calculate_sparsities(params,sparse = self.args.dense_ratio)
        
        if self.args.snip_mask:
            temp = self.generate_global_mask_snip()
            mask_pers_local = [copy.deepcopy(temp) for i in range(self.args.client_num_in_total)]
        else:
            if not self.args.different_initial: #Same mask to all the clients
                temp = self.model_trainer.init_masks(params, sparsities)
                mask_pers_local = [copy.deepcopy(temp) for i in range(self.args.client_num_in_total)]

            elif not self.args.diff_spa: #Different sparsity
                mask_pers_local = [copy.deepcopy(self.model_trainer.init_masks(params, sparsities)) for i in range(self.args.client_num_in_total)]
            else:
                #Not needed, same sparsity for all clients
                divide = 5
                p_divide = [0.2, 0.4, 0.6, 0.8, 1.0]
                mask_pers_local = []
                for i in range(self.args.client_num_in_total):
                    sparsities = self.model_trainer.calculate_sparsities(params, sparse = p_divide[i % divide])
                    temp = self.model_trainer.init_masks(params, sparsities)
                    mask_pers_local.append(temp)
                    w_spa[i] = p_divide[i % divide]
        
        w_global = self.model_trainer.get_model_params() #Make a copy of model parameters
        w_per_mdls = []  #To store masked model parameters
        updates_matrix = []  # To store masked model updates
        
        for clnt in range(self.args.client_num_in_total):  #For each client
            w_per_mdls.append(copy.deepcopy(w_global))
            updates_matrix.append(copy.deepcopy(w_global))

            for name in mask_pers_local[clnt]:
                w_per_mdls[clnt][name] = w_global[name] * mask_pers_local[clnt][name]
                updates_matrix[clnt][name] = updates_matrix[clnt][name] - updates_matrix[clnt][name]

        
        w_per_globals = [copy.deepcopy(w_global) for i in range(self.args.client_num_in_total)]

        
        mask_pers_shared = copy.deepcopy(mask_pers_local)
        
        dist_locals = np.zeros(shape=(self.args.client_num_in_total, self.args.client_num_in_total))

        #---------------------------------------------------------------------------

        for round_idx in range(self.args.comm_round):
            self.logger.info("################Communication round : {}".format(round_idx))

            active_ths_rnd = np.random.choice([0, 1], size = self.args.client_num_in_total, p = [1.0 - self.args.active, self.args.active])
            # 更新communication round时的所有personalized model
            w_per_mdls_lstrd = copy.deepcopy(w_per_mdls)
            mask_pers_shared_lstrd = copy.deepcopy(mask_pers_shared)

            # 在每一个communication rounds需要进行每个client的local training
            tst_results_ths_round = []
            final_tst_results_ths_round = []
            for clnt_idx in range(self.args.client_num_in_total):
                if active_ths_rnd[clnt_idx] == 0:
                    self.logger.info('@@@@@@@@@@@@@@@@ Client Drop this round CM({}) with spasity {}: {}'.format(round_idx, w_spa[clnt_idx], clnt_idx))

                self.logger.info('@@@@@@@@@@@@@@@@ Training Client CM({}) with spasity {}: {}'.format(round_idx, w_spa[clnt_idx], clnt_idx))
                # 记录当前mask变化了多少
                dist_locals[clnt_idx][clnt_idx], total_dis = hamming_distance(mask_pers_shared_lstrd[clnt_idx], mask_pers_local[clnt_idx])
                self.logger.info("local mask changes: {} / {}".format(dist_locals[clnt_idx][clnt_idx], total_dis))
                if active_ths_rnd[clnt_idx] == 0:
                    nei_indexs = np.array([])
                else:
                    nei_indexs = self._benefit_choose(round_idx, clnt_idx, self.args.client_num_in_total,
                                                  self.args.client_num_per_round, dist_locals[clnt_idx], total_dis, self.args.cs, active_ths_rnd)
                

                if self.args.client_num_in_total != self.args.client_num_per_round:
                    nei_indexs = np.append(nei_indexs, clnt_idx)

                nei_indexs = np.sort(nei_indexs)


                # # 更新dist_locals 矩阵
                # for tmp_idx in nei_indexs:
                #     if tmp_idx != clnt_idx:
                #         dist_locals[clnt_idx][tmp_idx],_ = hamming_distance(mask_pers_local[clnt_idx], mask_pers_shared_lstrd[tmp_idx])

                if self.args.cs!="full":
                    self.logger.info("choose client_indexes: {}, accoring to {}".format(str(nei_indexs), self.args.cs))
                else:
                    self.logger.info("choose client_indexes: {}, accoring to {}".format(str(nei_indexs), self.args.cs))
                if active_ths_rnd[clnt_idx] != 0:
                    nei_distances = [dist_locals[clnt_idx][i] for i in nei_indexs]
                    self.logger.info("choose mask diff: " + str(nei_distances))

                # # Update each client's local model and the so-called consensus model
                # if active_ths_rnd[clnt_idx] == 1:
                #     w_local_mdl, w_per_globals[clnt_idx] = self._aggregate_func(clnt_idx, self.args.client_num_in_total, self.args.client_num_per_round, nei_indexs,
                #                     w_per_mdls_lstrd, mask_pers_local, mask_pers_shared_lstrd)
                # else:
                #     w_local_mdl, w_per_globals[clnt_idx] = copy.deepcopy(w_per_mdls_lstrd[clnt_idx]), copy.deepcopy(w_per_mdls_lstrd[clnt_idx])

                # # 聚合好模型后，更新shared mask
                # mask_pers_shared[clnt_idx] = copy.deepcopy(mask_pers_local[clnt_idx])

                # # 设置client进行local training
                client = self.client_list[clnt_idx]

                test_local_metrics = client.local_test(w_local_mdl, True)
                final_tst_results_ths_round.append(test_local_metrics)

                # #Updates the mask, but not the case for SNIp

                new_mask, w_local_mdl, updates_matrix[clnt_idx], training_flops, num_comm_params, tst_results = client.train(copy.deepcopy(w_local_mdl), copy.deepcopy(mask_pers_local[clnt_idx]), round_idx)
                tst_results_ths_round.append(tst_results)

                # # 更新local model和local mask
                # w_per_mdls[clnt_idx] = copy.deepcopy(w_local_mdl)
                # mask_pers_local[clnt_idx] = copy.deepcopy(new_mask)

                # 更新w_per_globals w_per_globals里存储的是每个client的训练完的最后状态(dense models)
                for key in w_per_globals[clnt_idx]:
                    w_per_globals[clnt_idx][key] += updates_matrix[clnt_idx][key]

                self.stat_info["sum_training_flops"] += training_flops
                self.stat_info["sum_comm_params"] += num_comm_params

            self._local_test_on_all_clients(tst_results_ths_round, round_idx)
            self._local_test_on_all_clients_new_mask(final_tst_results_ths_round, round_idx)

        for index in range(self.args.client_num_in_total):
            tmp_dist = []
            for clnt in range(self.args.client_num_in_total):
                tmp, _ = hamming_distance(mask_pers_local[index], mask_pers_local[clnt])
                tmp_dist.append(tmp.item())
            self.stat_info["mask_dis_matrix"].append(tmp_dist)

        ## uncomment this if u like to save the final mask; Note masks for Resnet could be large, up to 1GB for 100 clients
        if self.args.save_masks:
            saved_masks = [{} for index in range(len(mask_pers_local))]
            for index, mask in enumerate(mask_pers_local):
                for name in mask:
                        saved_masks[index][name] = mask[name].data.bool()
            self.stat_info["final_masks"] =saved_masks
        return

    def train(self):

        params = self.model_trainer.get_trainable_params() #Model is same, so get parameters

        #Generate sailency scores, and update masks

        if self.args.snip_mask:
            temp = self.generate_global_mask_snip()
            mask_pers_local = [copy.deepcopy(temp) for i in range(self.args.client_num_in_total)]
        
        w_global = self.model_trainer.get_model_params() #Make a copy of model parameters
        w_per_mdls = []  #To store masked model parameters
        updates_matrix = []  # To store masked model updates
        
        for clnt in range(self.args.client_num_in_total):  #For each client
            w_per_mdls.append(copy.deepcopy(w_global))
            updates_matrix.append(copy.deepcopy(w_global))

            for name in mask_pers_local[clnt]:
                w_per_mdls[clnt][name] = w_global[name] * mask_pers_local[clnt][name]
                updates_matrix[clnt][name] = updates_matrix[clnt][name] - updates_matrix[clnt][name]

        #----------------------------------------------------------------------------------------------

        # w_per_globals = [copy.deepcopy(w_global) for i in range(self.args.client_num_in_total)]
        # mask_pers_shared = copy.deepcopy(mask_pers_local)
        

        for round_idx in range(self.args.comm_round):
            self.logger.info("################Communication round : {}".format(round_idx))
            w_locals = []
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
                w_per,training_flops,num_comm_params = client.train(copy.deepcopy(w_per_mdls[cur_clnt]), round_idx, mask_pers_local[cur_clnt])
                w_per_mdls[cur_clnt] = copy.deepcopy(w_per)
                # self.logger.info("local weights = " + str(w))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w_per)))
                self.stat_info["sum_training_flops"] += training_flops
                self.stat_info["sum_comm_params"] += num_comm_params
            # update global meta weights
            w_global = self._aggregate(w_locals)

            self._test_on_all_clients(w_global, w_per_mdls, round_idx)
            # self._local_test_on_all_clients(w_global, round_idx)
        # self.record_avg_inference_flops(w_global)

        # 为了查看finetune的结果，在global avged model上再进行一轮训练
        self.logger.info("################Communication round Last Fine Tune Round")
        for clnt_idx in range(self.args.client_num_in_total):
            self.logger.info('@@@@@@@@@@@@@@@@ Training Client: {}'.format(clnt_idx))
            w_local_mdl = copy.deepcopy(w_global)
            client = self.client_list[clnt_idx]
            w_local_mdl, training_flops, num_comm_params = client.train(copy.deepcopy(w_local_mdl), -1, mask_pers_local[cur_clnt])
            # 更新local model
            w_per_mdls[clnt_idx] = copy.deepcopy(w_local_mdl)

        self._test_on_all_clients(w_global, w_per_mdls, -1)

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        self.logger.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

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

    def _aggregate_func(self, cur_idx, client_num_in_total, client_num_per_round, nei_indexs, w_per_mdls_lstrd, mask_pers, mask_pers_shared_lstrd):
        self.logger.info('Doing local aggregation!')
        # Use the received models to infer the consensus model
        count_mask = copy.deepcopy(mask_pers_shared_lstrd[cur_idx])
        for k in count_mask.keys():
            count_mask[k] = count_mask[k] - count_mask[k]
            for clnt in nei_indexs:
                count_mask[k] += mask_pers_shared_lstrd[clnt][k]
        for k in count_mask.keys():
            count_mask[k] = np.divide(1, count_mask[k], out = np.zeros_like(count_mask[k]), where = count_mask[k] != 0)
        w_tmp = copy.deepcopy(w_per_mdls_lstrd[cur_idx])
        for k in w_tmp.keys():
            w_tmp[k] = w_tmp[k] - w_tmp[k]
            for clnt in nei_indexs:
                w_tmp[k] += torch.from_numpy(count_mask[k]) * w_per_mdls_lstrd[clnt][k]
        w_p_g = copy.deepcopy(w_tmp)
        for name in mask_pers[cur_idx]:
            w_tmp[name] = w_tmp[name] * mask_pers[cur_idx][name]
        return w_tmp, w_p_g
    
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

    def _local_test_on_all_clients_new_mask(self, tst_results_ths_round, round_idx):
        self.logger.info("################local_test_on_all_clients before local training in communication round: {}".format(round_idx))
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
        test_acc = sum([test_metrics['num_correct'][i] / test_metrics['num_samples'][i] for i in
                        range(self.args.client_num_in_total)]) / self.args.client_num_in_total
        test_loss = sum([np.array(test_metrics['losses'][i]) / np.array(test_metrics['num_samples'][i]) for i in
                         range(self.args.client_num_in_total)]) / self.args.client_num_in_total

        stats = {'test_acc': test_acc, 'test_loss': test_loss}

        self.logger.info(stats)
        self.stat_info["new_mask_test_acc"].append(test_acc)

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


