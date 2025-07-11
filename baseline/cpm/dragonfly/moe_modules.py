import bmtrain as bmt
import torch
import torch.nn.functional as F


class TopkRouter(bmt.DistributedModule):
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
            "balance_loss": balance_loss
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
            "balance_loss": torch.tensor(-1)
        }