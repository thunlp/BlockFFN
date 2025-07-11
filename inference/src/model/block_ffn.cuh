#pragma once

#include "ffn.cuh"
#include "block_ffn_kernel.cuh"

template <typename T>
struct Router {
    int hidden_size;
    int num_blocks;

    Linear<T> *proj;
    RMSNorm<T> *norm;
    T* output;

    Router(int hidden_size, int num_blocks, float rms_norm_eps) {
        this->hidden_size = hidden_size;
        this->num_blocks = num_blocks;

        this->proj = new Linear<T>(hidden_size, num_blocks);
        this->norm = new RMSNorm<T>(num_blocks, rms_norm_eps);
    }

    void init_weight_ptr(Memory* memory) {
        this->proj->init_weight_ptr(memory);
        this->norm->init_weight_ptr(memory);
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        int64_t proj_end = this->proj->init_output_ptr(memory, num_tokens, offset);
        this->output = this->proj->output;
        // norm inplace
        return proj_end;
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.find("router_proj") != std::string::npos) {
            this->proj->load_to_storage(name, ptr);
        } else if (name.find("router_norm") != std::string::npos) {
            this->norm->load_to_storage(name, ptr);
        } else {
            throw std::invalid_argument("Unsupported name " + name);
        }
    }

    void prefill(const Stream& stream, int32_t num_tokens, T* input) {
        this->proj->prefill(stream, num_tokens, input);
        relu_inplace(stream, num_tokens, this->num_blocks, this->proj->output);
        this->norm->prefill(stream, num_tokens, this->proj->output, nullptr, this->proj->output);
    }
};

template <typename T>
struct BlockFFN : FFN<T> {
    int hidden_size;
    int intermediate_size;
    int num_blocks, block_size;
    float rms_norm_eps;
    bool use_kernel;
    Router<T> *router;
    int *nnz;
    T *nz_val; int *nz_idx;
    float norm_scale;
    T* router_score;

    RMSNorm<T> *ffn_norm;
    Linear<T> *up_proj;
    Linear<T> *down_proj;

    BlockFFN(int hidden_size, int intermediate_size, float rms_norm_eps, int block_size, float norm_scale = 1.0, bool use_kernel = false) {
        this->hidden_size = hidden_size;
        this->intermediate_size = intermediate_size;
        this->num_blocks = intermediate_size / block_size;
        this->block_size = block_size;
        this->rms_norm_eps = rms_norm_eps;
        this->use_kernel = use_kernel;
        this->norm_scale = norm_scale;

        this->router = new Router<T>(hidden_size, num_blocks, rms_norm_eps);

        this->ffn_norm = new RMSNorm<T>(hidden_size, rms_norm_eps);
        this->up_proj = new Linear<T>(hidden_size, intermediate_size);
        this->down_proj = new Linear<T>(intermediate_size, hidden_size);
    }

    void init_weight_ptr(Memory* memory) {
        this->router->init_weight_ptr(memory);
        this->ffn_norm->init_weight_ptr(memory);
        this->up_proj->init_weight_ptr(memory);
        this->down_proj->init_weight_ptr(memory);
        this->router_score = (T*)memory->allocate_for_model(1024 * num_blocks * sizeof(T));
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        offset = this->ffn_norm->init_output_ptr(memory, num_tokens, offset);
        offset = this->router->init_output_ptr(memory, num_tokens, offset);
        offset = this->up_proj->init_output_ptr(memory, num_tokens, offset);
        offset = this->down_proj->init_output_ptr(memory, num_tokens, offset);
        this->output = this->down_proj->output;
        offset = memory->allocate((void**)&nnz, offset, sizeof(int));
        offset = memory->allocate((void**)&nz_val, offset, num_tokens * this->num_blocks * sizeof(T));
        offset = memory->allocate((void**)&nz_idx, offset, this->num_blocks * sizeof(int));
        return offset;
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.find("router_score") != std::string::npos) {
            cudaMemcpy((void*)this->router_score, ptr, 1024 * num_blocks * sizeof(T), cudaMemcpyHostToDevice);
        } else if (name.find("router") != std::string::npos) {
            this->router->load_to_storage(name, ptr);
        } else if (name.find("up_proj") != std::string::npos) {
            this->up_proj->load_to_storage(name, ptr);
        } else if (name.find("down_proj") != std::string::npos) {
            this->down_proj->load_to_storage(name, ptr);
        } else if (name.find("post_attention_layernorm") != std::string::npos) {
            this->ffn_norm->load_to_storage(name, ptr);
        } else {
            throw std::invalid_argument("Unsupported name " + name);
        }
    }

    void prefill(const Stream& stream, int32_t num_tokens, T* input, T* prev_output) {
        if (prev_output != nullptr) {
            elementwise_scale(stream, num_tokens, this->hidden_size, prev_output, this->norm_scale);
        }
        this->ffn_norm->prefill(stream, num_tokens, input, prev_output);
        this->router->prefill(stream, num_tokens, this->ffn_norm->output);

        this->up_proj->prefill(stream, num_tokens, this->ffn_norm->output);
        silu_inplace(stream, num_tokens, this->intermediate_size, this->up_proj->output);
        batched_mul(stream, num_tokens * this->num_blocks, this->block_size, this->up_proj->output, this->router->output, this->up_proj->output);
        this->down_proj->prefill(stream, num_tokens, this->up_proj->output);
    }

    void decode(const Stream& stream, int32_t num_tokens, T* input, T* prev_output) {
        if (prev_output != nullptr) {
            elementwise_scale(stream, num_tokens, this->hidden_size, prev_output, this->norm_scale);
        }
        this->ffn_norm->prefill(stream, num_tokens, input, prev_output);
        this->router->prefill(stream, num_tokens, this->ffn_norm->output);
        T* rs = this->router->output;

        if (this->use_kernel) {
            if (num_tokens == 1) {
                nonzero(stream, num_tokens, this->num_blocks, rs, this->nnz, this->nz_val, this->nz_idx);
                sparse_up(stream, num_tokens, this->num_blocks, this->block_size, this->hidden_size, this->nnz, this->nz_val, this->nz_idx, this->ffn_norm->output, this->up_proj->weight, this->up_proj->output);
                sparse_down(stream, num_tokens, this->num_blocks, this->block_size, this->hidden_size, this->nnz, this->nz_idx, this->up_proj->output, this->down_proj->weight, this->down_proj->output);
            } else if (num_tokens == 32 || num_tokens == 64) {
                nonzero(stream, num_tokens, this->num_blocks, rs, this->nnz, this->nz_val, this->nz_idx);
                sparse_up(stream, num_tokens, this->num_blocks, this->block_size, this->hidden_size, this->nnz, this->nz_val, this->nz_idx, this->ffn_norm->output, this->up_proj->weight, this->up_proj->output);
                batched_mul(stream, num_tokens * this->num_blocks, this->block_size, this->up_proj->output, rs, this->up_proj->output);
                sparse_down(stream, num_tokens, this->num_blocks, this->block_size, this->hidden_size, this->nnz, this->nz_idx, this->up_proj->output, this->down_proj->weight, this->down_proj->output);
            } else {
                throw std::invalid_argument("block_ffn: Unsupported num_tokens " + std::to_string(num_tokens));
            }
        } else {
            this->up_proj->prefill(stream, num_tokens, this->ffn_norm->output);
            silu_inplace(stream, num_tokens, this->intermediate_size, this->up_proj->output);
            batched_mul(stream, num_tokens * this->num_blocks, this->block_size, this->up_proj->output, rs, this->up_proj->output);
            this->down_proj->prefill(stream, num_tokens, this->up_proj->output);
        }
    }
};
