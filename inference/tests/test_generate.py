import torch
from llamacu.llama import LLM
from llamacu.speculative import LLM_with_medusa, LLM_with_eagle
from transformers import AutoTokenizer, AutoConfig
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="thunlp/blockffn_3b_sft")
parser.add_argument("--eagle-path", type=str, default="thunlp/blockffn_3b_sft-eagle")
parser.add_argument("--dtype", type=str, default="bf16")
parser.add_argument("--model-type", type=str, choices=["base", "medusa", "eagle"], default="base")
parser.add_argument("--disable-cuda-graph", action="store_true")
parser.add_argument("--use-kernel", action="store_true")
parser.add_argument("--num-generate", type=int, default=100)
parser.add_argument("--V", type=int, default=73440)
args = parser.parse_args()

dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
path = args.model_path
eagle_path = args.eagle_path
cuda_graph = not args.disable_cuda_graph
use_kernel = args.use_kernel
num_generate = args.num_generate
model_type = args.model_type
V = args.V

prompt = "Beijing is the"
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
config = AutoConfig.from_pretrained(path, trust_remote_code=True)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda().int()
num_tokens = input_ids.numel()
num_generate = 100

position_ids = torch.arange(num_tokens, dtype=torch.int32, device="cuda").view(1, num_tokens)

if model_type == "eagle":
    llm = LLM_with_eagle(eagle_path, path, dtype=dtype, memory_limit=0.5, use_kernel=use_kernel, cuda_graph=cuda_graph, V=V)
    our_generate = lambda: llm.generate(input_ids, num_generate)
else:
    llm = LLM(path, dtype=dtype, memory_limit=0.5, use_kernel=use_kernel, cuda_graph=cuda_graph)
    our_generate = lambda: llm.generate(input_ids, num_generate)

llm.init_storage()
if model_type == "eagle":
    if V == 73440:
        token_id_remap = torch.arange(V, dtype=torch.int32, device="cpu")
        llm._load("token_id_remap", token_id_remap, cls="eagle")
    else:
        import pickle
        with open(f"{eagle_path}/freq_{V}.pkl", "rb") as f:
            freq = pickle.load(f)
        token_id_remap = torch.tensor(list(freq), dtype=torch.int32, device="cpu")
        llm._load("token_id_remap", token_id_remap, cls="eagle")
llm.load_from_hf()

for i in range(config.num_hidden_layers): # TODO do not make activate position > 64 now
    router_score = torch.zeros(1024, config.num_experts, dtype=dtype, device="cpu")
    router_score[:, :config.num_experts:4] = 1 # activate 1/4
    llm._load(f"model.layers.{i}.mlp.router_score", router_score)

if model_type == "eagle":
    tokens, accept_lengths, _, _ = our_generate()
    print(tokens)
    print(tokenizer.decode(tokens))
    accept_lengths = sum(accept_lengths) / len(accept_lengths)
    print("mean accept_lengths:", accept_lengths)
else:
    tokens, _ = our_generate()
    print(tokens)
    print(tokenizer.decode(tokens))
    accept_lengths = 1
