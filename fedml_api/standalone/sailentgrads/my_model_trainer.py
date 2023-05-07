import copy
import logging
import time

import numpy as np
import pdb
import torch
from torch import nn

from fedml_api.model.cv.cnn_meta import Meta_net
import torch.nn.functional as F
import types

from fedml_api.standalone.sailentgrads.SNIP.snip import snip_forward_conv2d, snip_forward_linear
try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from fedml_core.trainer.model_trainer import ModelTrainer


class MyModelTrainer(ModelTrainer):
    def __init__(self, model, args=None, logger = None):
        super().__init__(model, args)
        self.args=args
        self.logger = logger

    def set_masks(self, masks):
        self.masks=masks
        # self.model.set_masks(masks)

    def init_masks(self, params, sparsities):
        masks ={}
        for name in params:
            masks[name] = torch.zeros_like(params[name])
            dense_numel = int((1-sparsities[name])*torch.numel(masks[name]))
            if dense_numel > 0:
                temp = masks[name].view(-1)
                perm = torch.randperm(len(temp))
                perm = perm[:dense_numel]
                temp[perm] =1
        return masks
    
    def init_masks_using_snip(self, params, sparsities):
        masks ={}
        for name in params:
            masks[name] = torch.zeros_like(params[name])
            dense_numel = int((1-sparsities[name])*torch.numel(masks[name]))
            if dense_numel > 0:
                temp = masks[name].view(-1)
                perm = torch.randperm(len(temp))
                perm = perm[:dense_numel]
                temp[perm] =1
        return masks

    def calculate_sparsities(self, params, tabu=[], distribution="ERK", sparse = 0.5):
        spasities = {}
        if distribution == "uniform":
            for name in params:
                if name not in tabu:
                    spasities[name] = 1 - self.args.dense_ratio
                else:
                    spasities[name] = 0
        elif distribution == "ERK":
            self.logger.info('initialize by ERK')
            total_params = 0
            for name in params:
                total_params += params[name].numel()
            is_epsilon_valid = False
            # # The following loop will terminate worst case when all masks are in the
            # custom_sparsity_map. This should probably never happen though, since once
            # we have a single variable or more with the same constant, we have a valid
            # epsilon. Note that for each iteration we add at least one variable to the
            # custom_sparsity_map and therefore this while loop should terminate.
            dense_layers = set()

            density = sparse
            while not is_epsilon_valid:
                # We will start with all layers and try to find right epsilon. However if
                # any probablity exceeds 1, we will make that layer dense and repeat the
                # process (finding epsilon) with the non-dense layers.
                # We want the total number of connections to be the same. Let say we have
                # for layers with N_1, ..., N_4 parameters each. Let say after some
                # iterations probability of some dense layers (3, 4) exceeded 1 and
                # therefore we added them to the dense_layers set. Those layers will not
                # scale with erdos_renyi, however we need to count them so that target
                # paratemeter count is achieved. See below.
                # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
                #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
                # eps * (p_1 * N_1 + p_2 * N_2) =
                #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
                # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

                divisor = 0
                rhs = 0
                raw_probabilities = {}
                for name in params:
                    if name in tabu:
                        dense_layers.add(name)
                    n_param = np.prod(params[name].shape)
                    n_zeros = n_param * (1 - density)
                    n_ones = n_param * density

                    if name in dense_layers:
                        rhs -= n_zeros
                    else:
                        rhs += n_ones
                        raw_probabilities[name] = (
                                                          np.sum(params[name].shape) / np.prod(params[name].shape)
                                                  ) ** self.args.erk_power_scale
                        divisor += raw_probabilities[name] * n_param
                epsilon = rhs / divisor
                max_prob = np.max(list(raw_probabilities.values()))
                max_prob_one = max_prob * epsilon
                if max_prob_one > 1:
                    is_epsilon_valid = False
                    for mask_name, mask_raw_prob in raw_probabilities.items():
                        if mask_raw_prob == max_prob:
                            (f"Sparsity of var:{mask_name} had to be set to 0.")
                            dense_layers.add(mask_name)
                else:
                    is_epsilon_valid = True

            # With the valid epsilon, we can set sparsities of the remaning layers.
            for name in params:
                if name in dense_layers:
                    spasities[name] = 0
                else:
                    spasities[name] = (1 - epsilon * raw_probabilities[name])
        return spasities

    def get_model_params(self):
        return copy.deepcopy(self.model.cpu().state_dict())

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def get_trainable_params(self):
        dict= {}
        for name, param in self.model.named_parameters():
            dict[name] = param
        return dict
    
    def get_model_sps(self):
        nonzero = total = 0
        for name, param in self.model.named_parameters():
            if 'mask' not in name:
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

    
    def get_snip_scores(self, mini_batch, re_init=False):
        model = self.model
        device = next(iter(model.parameters())).device
        # Grab a single batch from the training dataset
        # inputs, targets = next(iter(train_dataloader))
        inputs, targets = mini_batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Let's create a fresh copy of the modelwork so that we're not worried about
        # affecting the actual training-phase
        cp_model = copy.deepcopy(model)

        # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
        # instead of the weights
        for layer in cp_model.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight)).to(device)
                if re_init:
                    print("Re-initializing within SNIP!")
                    nn.init.xavier_normal_(layer.weight)
                layer.weight.requires_grad = False

            # # Override the forward methods:
            if isinstance(layer, nn.Conv2d):
                layer.forward = types.MethodType(snip_forward_conv2d, layer)

            if isinstance(layer, nn.Linear):
                layer.forward = types.MethodType(snip_forward_linear, layer)

        cp_model.to(device)
        # Compute gradients (but don't apply them)
        
        cp_model.zero_grad()
        outputs = cp_model.forward(inputs)

        loss = F.nll_loss(outputs, targets)
        loss.backward()
        
        # print(f"Model Gradients: {cp_model.features[0].weight_mask.grad}")

        # grads_abs = dict()
        # for name, layer in cp_model.named_modules():
        #     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        #         grads_abs[name] = torch.abs(layer.weight_mask.grad)

        grads_abs = []
        for name, layer in cp_model.named_modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                grads_abs.append((name, torch.abs(layer.weight_mask.grad)))


        # # Gather all scores in a single vector and normalise
        # all_scores = torch.cat([torch.flatten(x) for x in grads_abs.values()])
        # norm_factor = torch.sum(all_scores)
        # all_scores.div_(norm_factor)
        del cp_model
        return grads_abs


    def get_mask_from_grads(self, grads_abs, keep_ratio, params):
        model = self.model
        # Copy the Model Again so not to overwrite actual model to be trained.
        cp_model = copy.deepcopy(model)

        # Gather all scores in a single vector and normalise
        all_scores = torch.cat([torch.flatten(x) for x in grads_abs.values()])
        norm_factor = torch.sum(all_scores)
        all_scores.div_(norm_factor)

        num_params_to_keep = int(len(all_scores) * keep_ratio)
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]

        keep_masks = dict()
        keep_masks_weight = dict()
        for name, values in grads_abs.items():
            keep_masks[name] = ((values / norm_factor) >= acceptable_score).float()
            keep_masks_weight[name+".weight"] = ((values / norm_factor) >= acceptable_score).float()
            # keep_masks.append(((g / norm_factor) >= acceptable_score).float())

        mask_key_layer = dict()
        for i, (name, layer) in enumerate(cp_model.named_modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                mask_key_layer[layer] = keep_masks[name]
        
        
        final_weight_mask= dict()
        for name, param in self.model.named_parameters():
            if name in keep_masks_weight:
                final_weight_mask[name] = keep_masks_weight[name]
            else:
                final_weight_mask[name] = torch.ones_like(param)

        del cp_model



        return keep_masks, mask_key_layer, final_weight_mask
    
    def get_mean_snip_scores(self, grads_gathered):
        model = self.model
        size = len(grads_gathered)
        grad_abs_average={}

        # Loop through each dictionary in the list
        for grad_abs in grads_gathered:
            # Loop through each key-value pair in the dictionary
            for k, v in dict(grad_abs).items():
                # If the key already exists in the averaged dictionary, add the value to the existing sum
                if k in grad_abs_average:
                    grad_abs_average[k] += v
                # Otherwise, create a new key in the averaged dictionary and initialize it with the value
                else:
                    grad_abs_average[k] = v.clone().detach()

        # Divide the summed values by the number of dictionaries in the list to get the average
        num_dicts = size
        for k in grad_abs_average.keys():
            grad_abs_average[k] /= num_dicts

        return grad_abs_average



    def screen_gradients(self, train_data, device):
        model = self.model
        model.to(device)
        model.eval()
        # # # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        # # sample one epoch  of data
        model.zero_grad()
        (x, labels) = next(iter(train_data))
        x, labels = x.to(device), labels.to(device)
        log_probs = model.forward(x)
        loss = criterion(log_probs, labels.long())
        loss.backward()
        gradient={}
        for name, param in model.named_parameters():
            gradient[name] = param.grad.to("cpu")
        return gradient


    def train(self, train_data,  device,  args, round, masks):
        # torch.manual_seed(0)
        model = self.model
        model.to(device)
        model.train()
        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr* (args.lr_decay**round), momentum=args.momentum,weight_decay=args.wd)
        for epoch in range(args.epochs):
            epoch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model.forward(x)
                loss = criterion(log_probs, labels.long())
                loss.backward()
                # to avoid nan loss
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                optimizer.step()
                epoch_loss.append(loss.item())
                for name, param in self.model.named_parameters():
                    if name in masks:
                        param.data *= masks[name].to(device)
            self.logger.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                self.id, epoch, sum(epoch_loss) / len(epoch_loss)))



    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_acc':0.0,
            'test_loss': 0,
            'test_total': 0
        }

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target.long())

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
                metrics['test_acc'] = metrics['test_correct'] / metrics['test_total']
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False

