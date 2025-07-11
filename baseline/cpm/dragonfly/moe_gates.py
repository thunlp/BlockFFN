import bmtrain as bmt
import torch
import torch.nn.functional as F
from .activation import ActivationContext
from .layer_norm import LayerNorm


class TopKRouter(bmt.DistributedModule):
    """
    Select top_k expert each time, with a learnable gate_network controlling expert scores.
    https://arxiv.org/pdf/2101.03961.pdf
    """
    def __init__(self, 
        dim_model: int,
        num_experts: int,
        top_k: int,
        dtype: torch.dtype,
        init_mean: float = 0,
        init_std: float = 0.01
    ):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.weight = bmt.DistributedParameter(
            torch.empty((num_experts, dim_model), dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std)
        )

    def forward(self, x: torch.Tensor):
        # [bs, seq_len, hidden_size]
        x = x.view(-1, x.shape[-1])
        # [bs * seq_len, hidden_size]
        scores = F.linear(x, self.weight)
        # [bs * seq_len, num_experts]
        scores_prob = F.softmax(scores, dim=-1, dtype=torch.float32)

        # add router entropy
        router_entropy = torch.sum(-scores_prob * torch.log(scores_prob + 1e-5), dim=-1)
        router_entropy = router_entropy.mean()

        # [bs * seq_len, num_experts]
        expert_weights, expert_indices = torch.topk(scores_prob, self.top_k, dim=-1)
        # NOTE: mixtral use topk norm, deepseek-moe does't norm topk probs
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        # [bs * seq_len, topk], [bs * seq_len, topk]

        # calculate load balancing loss
        token_num = x.shape[0]
        load = expert_indices.view(-1).bincount(minlength=self.num_experts)
        load_mean = load / (token_num * self.top_k)
        importance_mean = scores_prob.mean(dim=0)
        balance_loss = self.num_experts * torch.sum(importance_mean * load_mean)

        return {
            "topk_indices": expert_indices,
            "topk_scores": expert_weights.to(x.dtype),
            "load": load_mean,
            "balance_loss": balance_loss,
            "router_entropy": router_entropy,
        }


class ReLURouter(bmt.DistributedModule):
    """
    Select top_k expert each time, with a learnable gate_network controlling expert scores.
    https://arxiv.org/pdf/2101.03961.pdf
    """
    def __init__(self,
        dim_model: int,
        num_experts: int,
        top_k: int,
        dtype: torch.dtype,
        init_mean: float = 0,
        init_std: float = 0.01,
        rms_norm: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.weight = bmt.DistributedParameter(
            torch.empty((num_experts, dim_model), dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std)
        )
        self.rms_norm = rms_norm
        if self.rms_norm:
            self.router_norm = LayerNorm(dim_norm=num_experts, dtype=dtype, eps=eps)

    def forward(self, x: torch.Tensor):
        # [bs, seq_len, hidden_size]
        x = x.view(-1, x.shape[-1])
        # [bs * seq_len, hidden_size]
        scores = F.linear(x, self.weight)
        # [bs * seq_len, num_experts]
        scores_prob = torch.relu(scores)
        routing_map = scores_prob > 0

        scores_prob, balance_loss, load = ActivationContext.l1_reg_load_balancing(scores_prob, routing_map, self.top_k)
        if self.rms_norm:
            scores_prob = self.router_norm(scores_prob, output_hidden=False)

        sorted_probs, sorted_indices = torch.sort(scores_prob, descending=True, dim=-1)
        sorted_map = sorted_probs <= 0
        # sorted_map[:, 0] = False
        sorted_indices = torch.where(sorted_map, -1, sorted_indices)
        max_valid_num = max(sorted_probs.size(-1) - torch.min(torch.sum(sorted_map, dim=-1)).item(), 1)
        assert torch.all(sorted_map[:, max_valid_num:])
        sorted_probs = sorted_probs[:, :max_valid_num]
        sorted_indices = sorted_indices[:, :max_valid_num]
        if not self.rms_norm:
            assert torch.sum(routing_map) == torch.sum(sorted_indices != -1)

        return {
            "topk_indices": sorted_indices,
            "topk_scores": sorted_probs,
            "load": load,
            "balance_loss": balance_loss,
            "router_entropy": torch.tensor(0.).to(balance_loss),
        }


class TopKExpertChoiceRouter(bmt.DistributedModule):
    def __init__(self, 
        dim_model: int,
        num_experts: int,
        top_k: int,
        dtype: torch.dtype,
        init_mean: float = 0,
        init_std: float = 0.01
    ):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.weight = bmt.DistributedParameter(
            torch.empty((num_experts, dim_model), dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std)
        )

    def forward(self, x):
        # [bs, seq_len, hidden_size]
        x = x.view(-1, x.shape[-1])
        # [bs * seq_len, hidden_size]
        scores = F.linear(x, self.weight)
        # [bs * seq_len, num_experts]
        scores_prob = F.softmax(scores, dim=-1, dtype=torch.float32)
        # [bs * seq_len, num_experts]
        scores_prob = scores_prob.transpose(0, 1)
        # [num_experts, bs * seq_len]
        expert_weights, expert_indices = torch.topk(scores_prob, self.top_k, dim=-1)
        # [num_experts, topk], [num_experts, topk]

        return {
            "topk_indices": expert_indices,
            "topk_scores": expert_weights.to(x.dtype),
            "load": None,
            "balance_loss": None,
            "router_entropy": None,
        }


class TopPRouter(bmt.DistributedModule):
    def __init__(self,
        dim_model: int,
        num_experts: int,
        top_p: float,
        dtype: torch.dtype,
        init_mean: float = 0,
        init_std: float = 0.01
    ):
        super().__init__()
        self.top_p = top_p
        self.num_experts = num_experts
        self.weight = bmt.DistributedParameter(
            torch.empty((num_experts, dim_model), dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std)
        )

    def forward(self, x):
        # [bs, seq_len, hidden_size]
        x = x.view(-1, x.shape[-1])
        # [bs * seq_len, hidden_size]
        scores = F.linear(x, self.weight)
        # [bs * seq_len, num_experts]
        scores_prob = F.softmax(scores, dim=-1, dtype=torch.float32)
        # [bs * seq_len, num_experts]

        # add router entropy
        router_entropy = torch.sum(-scores_prob * torch.log(scores_prob + 1e-5), dim=-1)
        router_entropy = router_entropy.mean()

        sorted_probs, sorted_indices = torch.sort(scores_prob, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs > self.top_p

        threshold_indices = mask.long().argmax(dim=-1)
        threshold_mask = F.one_hot(threshold_indices, num_classes=sorted_indices.size(-1)).bool()

        mask = mask & ~threshold_mask
        # mask[:, 0] = False
        sorted_indices = torch.where(mask, -1, sorted_indices)
        sorted_probs = torch.where(mask, 0.0, sorted_probs)

        max_valid_num = max(mask.size(-1) - torch.min(torch.sum(mask, dim=-1)).item(), 1)
        assert torch.all(mask[:, max_valid_num:])

        sorted_indices = sorted_indices[:, :max_valid_num]
        sorted_probs = sorted_probs[:, :max_valid_num]
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        # add load balancing
        count_indices = torch.where(torch.eq(sorted_indices, -1), self.num_experts, sorted_indices)
        load = count_indices.flatten().bincount(minlength=self.num_experts+1)[:-1]
        load_mean = load / (torch.sum(load) + 1e-5)
        importance_mean = scores_prob.mean(dim=0)
        balance_loss = self.num_experts * torch.sum(importance_mean * load_mean)

        return {
            "topk_indices": sorted_indices,
            "topk_scores": sorted_probs.to(x.dtype),
            "load": load_mean,
            "balance_loss": balance_loss,
            "router_entropy": router_entropy,
        }


class BaseLayer(bmt.DistributedModule):
    def __init__(self, 
        dim_model: int,
        num_experts: int,
        top_k: int,
        dtype: torch.dtype,
        init_mean: float = 0,
        init_std: float = 0.006
    ):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.weight = bmt.DistributedParameter(
            torch.empty((num_experts, dim_model), dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std)
        )

    def balanced_assignment(self, scores, max_iterations=100):
        scores = scores.transpose(1, 0)
        num_workers, num_jobs = scores.size()
        jobs_per_worker = num_jobs // num_workers
        value = scores.clone()

        epsilon = (scores.max() - scores.min()) / 50
        epsilon = torch.clamp(epsilon, min=1e-04).cuda()
        
        iterations = 0
        cost = scores.new_zeros(1, num_jobs)
        
        jobs_with_bids = torch.zeros(num_workers).bool()
        
        while not jobs_with_bids.all():
            top_values, top_index = torch.topk(value, k=jobs_per_worker + 1, dim=1)
            # Each worker bids the difference in value between a job and the k+1th job
            bid_increments = top_values[:, :-1] - top_values[:, -1:]
            bids = torch.scatter(torch.zeros((num_workers, num_jobs), device="cuda"), dim=1, index=top_index[:, :-1], src=bid_increments)
            if 0 < iterations < max_iterations:
                # If a worker won a job on the previous round, put in a minimal bid to retain
                # the job only if no other workers bid this round.
                bids[top_bidders, jobs_with_bids] = epsilon
                
            # Find the highest bidding worker per job
            top_bids, top_bidders = bids.max(dim=0)
            jobs_with_bids = top_bids > 0
            top_bidders = top_bidders[jobs_with_bids]
            
            # Make popular items more expensive
            cost += top_bids
            value = scores - cost
            if iterations < max_iterations:
                # If a worker won a job, make sure it appears in its top-k on the next round
                value[top_bidders, jobs_with_bids] = float("inf")
            else:
                value[top_bidders, jobs_with_bids] = scores[top_bidders, jobs_with_bids]
            iterations += 1

        return top_index[:,:-1]

    def forward(self, x: torch.Tensor):
        # [bs, seq_len, hidden_size]
        x = x.view(-1, x.shape[-1])
        # [bs * seq_len, hidden_size]
        scores = F.linear(x, self.weight)
        # [bs * seq_len, num_experts]
        scores_prob = F.softmax(scores, dim=-1, dtype=torch.float32)
        # [bs * seq_len, num_experts]

        if self.training:
            indices = self.balanced_assignment(scores_prob)
            print(indices.shape)
            exit(1)
        else:
            # greedy assignment
            pass
        return {
            "topk_indices": torch.tensor(-1),
            "topk_scores": torch.tensor(-1),
            "load": torch.tensor(-1),
            "balance_loss": torch.tensor(-1),
            "router_entropy": torch.tensor(-1),
        }