import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import types


def snip_forward_conv2d(self, x):
        # print(f"x: {x.device} | w: {self.weight.device} | wm: {self.weight_mask.device}")
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


def snip_forward_linear(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)


#Grab scores using SNIP
    
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
        
        grads_abs = []
        for name, layer in cp_model.named_modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                grads_abs.append((name, torch.abs(layer.weight_mask.grad)))

        del cp_model
        return grads_abs



#Calculate masks from the gradients

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
    
#Get Mean SNIP Scores
    
def get_mean_snip_scores(grads_gathered):
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

def get_weighted_mean_snip_scores(grads_gathered, probabilities):
    size = len(grads_gathered)
    grad_abs_average = {}

    # Loop through each dictionary in the list
    for i, grad_abs in enumerate(grads_gathered):
        weight = probabilities[i]  # Get the weight for the current client
        # Loop through each key-value pair in the dictionary
        for k, v in dict(grad_abs).items():
            # If the key already exists in the averaged dictionary, add the weighted value to the existing sum
            if k in grad_abs_average:
                grad_abs_average[k] += weight * v
            # Otherwise, create a new key in the averaged dictionary and initialize it with the weighted value
            else:
                grad_abs_average[k] = weight * v.clone().detach()

    # Normalize the summed values by dividing by the sum of the weights
    total_weight = sum(probabilities)
    for k in grad_abs_average.keys():
        grad_abs_average[k] /= total_weight

    return grad_abs_average

#In case of IterSNIP, a function to get the mean saliency scores
def get_mean_sailency_scores(final_sailency_list):
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

