import torch
from typing import List, Tuple
import math
import diffsort
import numpy as np
from scipy.stats import wasserstein_distance
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
from torchmetrics.audio import PermutationInvariantTraining

SORTING_NETWORK_TYPE = List[torch.tensor]


def s_best(x):
    return torch.clamp(x, -0.25, 0.25) + .5 + \
        ((x > 0.25).float() - (x < -0.25).float()) * (0.25 - 1/16/(x.abs()+1e-10))


class NormalCDF(torch.autograd.Function):
    def forward(ctx, x, sigma):
        ctx.save_for_backward(x, torch.tensor(sigma))
        return 0.5 + 0.5 * torch.erf(x / sigma / math.sqrt(2))

    def backward(ctx, grad_y):
        x, sigma = ctx.saved_tensors
        return grad_y * 1 / sigma / math.sqrt(math.pi * 2) * torch.exp(-0.5 * (x/sigma).pow(2)), None


def execute_sort(
        sorting_network,
        vectors,
        steepness=10.,
        art_lambda=0.25,
        distribution='cauchy'
):
    x = vectors
    X = torch.eye(vectors.shape[1], dtype=x.dtype, device=x.device).repeat(x.shape[0], 1, 1)

    for split_a, split_b, combine_min, combine_max in sorting_network:
        split_a = split_a.type(x.dtype)
        split_b = split_b.type(x.dtype)
        combine_min = combine_min.type(x.dtype)
        combine_max = combine_max.type(x.dtype)

        a, b = x @ split_a.T, x @ split_b.T

        # float conversion necessary as PyTorch doesn't support Half for sigmoid as of 25. August 2021
        new_type = torch.float32 if x.dtype == torch.float16 else x.dtype

        if distribution == 'logistic':
            alpha = torch.sigmoid((b-a).type(new_type) * steepness).type(x.dtype)

        elif distribution == 'logistic_phi':
            alpha = torch.sigmoid((b-a).type(new_type) * steepness / ((a-b).type(new_type).abs() + 1.e-10).pow(art_lambda)).type(x.dtype)

        elif distribution == 'gaussian':
            v = (b - a).type(new_type)
            alpha = NormalCDF.apply(v, 1 / steepness)
            alpha = alpha.type(x.dtype)

        elif distribution == 'reciprocal':
            v = steepness * (b - a).type(new_type)
            alpha = 0.5 * (v / (2 + v.abs()) + 1)
            alpha = alpha.type(x.dtype)

        elif distribution == 'cauchy':
            v = steepness * (b - a).type(new_type)
            alpha = 1 / math.pi * torch.atan(v) + .5
            alpha = alpha.type(x.dtype)

        elif distribution == 'optimal':
            v = steepness * (b - a).type(new_type)
            alpha = s_best(v)
            alpha = alpha.type(x.dtype)

        else:
            raise NotImplementedError('softmax method `{}` unknown'.format(distribution))

        aX = X @ split_a.T
        bX = X @ split_b.T
        w_min = alpha.unsqueeze(-2) * aX + (1-alpha).unsqueeze(-2) * bX
        w_max = (1-alpha).unsqueeze(-2) * aX + alpha.unsqueeze(-2) * bX
        X = (w_max @ combine_max.T.unsqueeze(-3)) + (w_min @ combine_min.T.unsqueeze(-3))
        x = (alpha * a + (1-alpha) * b) @ combine_min.T + ((1-alpha) * a + alpha * b) @ combine_max.T
    return x, X


def sort(
        sorting_network: SORTING_NETWORK_TYPE,
        vectors: torch.Tensor,
        steepness: float = 10.0,
        art_lambda: float = 0.25,
        distribution: str = 'cauchy'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sort a matrix along axis 1 using differentiable sorting networks. Return the permutation matrix.

    Positional arguments:
    sorting_network
    vectors -- the matrix to sort along axis 1; sorted in-place

    Keyword arguments:
    steepness -- relevant for sigmoid and sigmoid_phi interpolation (default 10.0)
    art_lambda -- relevant for logistic_phi interpolation (default 0.25)
    distribution -- how to interpolate when swapping two numbers; (default 'cauchy')
    """
    assert sorting_network[0][0].device == vectors.device, (
        f"The sorting network is on device {sorting_network[0][0].device} while the vectors are on device"
        f" {vectors.device}, but they both need to be on the same device."
    )
    return execute_sort(
        sorting_network=sorting_network,
        vectors=vectors,
        steepness=steepness,
        art_lambda=art_lambda,
        distribution=distribution
    )


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def is_sorted(walk):
    result = True
    for i in range(len(walk)-1):
        if walk[i]>walk[i+1]:
            result = False
    return result


def score_inversions(walk):
    score = 0.0
    walk = walk.view(-1)
    walk_len = len(walk)
    if is_sorted(walk):
        return 1.0
    else:
        # Unsortedp
        score+=1.0
    for i in range(walk_len):
        for j in range(i + 1, walk_len):
            # inversions
            if walk[i] > walk[j]:
                score += 1.0
        # adjacent inversions
        if i<walk_len-1:
            if walk[i] > walk[i+1]:
                score +=1.0
    # insertion index
    lengths = [1] * walk_len
    for i in range(1, walk_len):
        for j in range(i):
            if walk[i] > walk[j]:
                lengths[i] = max(lengths[i], lengths[j] + 1)
    score += walk_len
    score -= max(lengths)
    norm = walk_len*walk_len/2.0 + 5.0*walk_len/2.0-2.0
    return 1.0-(score/norm)

def create_weight_matrix(l):
    wm = np.zeros((l, l))
    for i in range(l):
        for j in range(l):
            wm[i,j] = np.abs(i-j)
    return wm

def inv_create_weight_matrix(l):
    wm = np.zeros((l, l))
    for i in range(l):
        for j in range(l):
            wm[i,j] = np.abs(i-j)
    wm = l - 1 - wm
    return wm

def score_emd(walk):
    walk = walk.view(-1)
    walk_len = len(walk)
    sorted_walk, _ = torch.sort(walk)
    walk_c = walk.cpu()
    walk_c = walk_c.detach().numpy()
    sorted_walk_c = sorted_walk.cpu()
    sorted_walk_c = sorted_walk_c.detach().numpy()
    return wasserstein_distance(sorted_walk_c, walk_c, create_weight_matrix(walk_len))

def score_si_snr(walk):
    walk = walk.view(-1)
    walk_len = len(walk)
    sorted_walk, _ = torch.sort(walk)
    walk_c = walk.cpu()
    sorted_walk_c = sorted_walk.cpu()
    return scale_invariant_signal_noise_ratio(sorted_walk_c, walk_c)

def score_pesq(walk):
    walk = walk.view(-1)
    walk_len = len(walk)
    sorted_walk, _ = torch.sort(walk)
    walk_c = walk.cpu()
    sorted_walk_c = sorted_walk.cpu()
    return perceptual_evaluation_speech_quality(sorted_walk_c, walk_c, walk_len, 'wb')

def score_pit(walk):
    walk = walk.view(-1)
    walk_len = len(walk)
    sorted_walk, _ = torch.sort(walk)
    walk = walk.view(1,-1,1)
    sorted_walk = sorted_walk.view(1,-1,1)
    walk_c = walk.cpu()
    sorted_walk_c = sorted_walk.cpu()
    metric = PermutationInvariantTraining(scale_invariant_signal_noise_ratio)
    score = metric(sorted_walk_c, walk_c)
    return score

def find_min_continuous_length(nums):
    if not nums:
        return 0
    min_length = float('inf')
    current_length = 1
    for i in range(1, len(nums)):
        if nums[i] == nums[i - 1]:
            current_length += 1
        else:
            min_length = min(min_length, current_length)
            current_length = 1
    min_length = min(min_length, current_length)
    return min_length


def choose_batch_idx(sq_len_list):
    min_length = 0
    idx = [False]*len(sq_len_list)
    pt = 0
    while pt < len(sq_len_list):
        if pt+15<len(sq_len_list):
            while sq_len_list[pt]==sq_len_list[pt+15]:
                idx[pt:pt+16] = [True]*16
                pt +=16
        pt+=1
    return idx


def temperature_scaled_softmax(logits, temperature=1.0):
    logits = logits / temperature
    return torch.softmax(logits, dim=1)