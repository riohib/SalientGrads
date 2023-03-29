import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import types


def apply_prune_mask(model, keep_masks):
    # Before I can zip() layers and pruning masks I need to make sure they match
    # one-to-one by removing all the irrelevant modules:
    prunable_layers = filter(
        lambda layer: isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear), 
        model.modules()
        )

    # for layer, keep_mask in zip(prunable_layers, keep_masks.values()):
    for layer, keep_mask in zip(prunable_layers, keep_masks.values()):
        assert (layer.weight.shape == keep_mask.shape)

        def hook_factory(keep_mask):
            """
            The hook function can't be defined directly here because of Python's
            late binding which would result in all hooks getting the very last
            mask! Getting it through another function forces early binding.
            """

            def hook(grads):
                return grads * keep_mask

            return hook
        # mask[i] == 0 --> Prune parameter
        # mask[i] == 1 --> Keep parameter

        # Step 1: Set the masked weights to zero (NB the biases are ignored)
        # Step 2: Make sure their gradients remain zero
        layer.weight.data[keep_mask == 0.] = 0.
        layer.weight.register_hook(hook_factory(keep_mask))



def snip_forward_conv2d(self, x):
        # print(f"x: {x.device} | w: {self.weight.device} | wm: {self.weight_mask.device}")
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


def snip_forward_linear(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)


def get_snip_scores(model, mini_batch, re_init=False):
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

        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(snip_forward_linear, layer)

    # print(f"Model Weights: {cp_model.features[0].weight}")
    # try:
    #     print(f"Model Gradients: {cp_model.features[0].weight_mask.grad} \n")
    # except:
    #     print("Model Grads Not Yet Generated!! \n")

    cp_model.to(device)
    # Compute gradients (but don't apply them)
    cp_model.zero_grad()
    outputs = cp_model.forward(inputs)
    loss = F.nll_loss(outputs, targets)
    loss.backward()
    
    # print(f"Model Gradients: {cp_model.features[0].weight_mask.grad}")

    grads_abs = dict()
    for name, layer in cp_model.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            grads_abs[name] = torch.abs(layer.weight_mask.grad)
    # # Gather all scores in a single vector and normalise
    # all_scores = torch.cat([torch.flatten(x) for x in grads_abs.values()])
    # norm_factor = torch.sum(all_scores)
    # all_scores.div_(norm_factor)
    del cp_model
    return grads_abs


def get_mask_from_grads(model, grads_abs, keep_ratio):
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
    for name, values in grads_abs.items():
        keep_masks[name] = ((values / norm_factor) >= acceptable_score).float()
        # keep_masks.append(((g / norm_factor) >= acceptable_score).float())

    mask_key_layer = dict()
    for i, (name, layer) in enumerate(cp_model.named_modules()):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            mask_key_layer[layer] = keep_masks[name]
    
    del cp_model
    return keep_masks, mask_key_layer

