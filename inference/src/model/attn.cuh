#pragma once
#include "../trait.cuh"
#include "../utils.cuh"
#include "../flash_attn/flash_api.hpp"
#include "norm.cuh"
#include "linear.cuh"
#include "rotary.cuh"
#include "kvcache.cuh"
#include "mask.cuh"
#include <cuda_runtime.h>

namespace {
__global__ void copy_to_kvcache_kernel(int num_tokens, int dim, int total, float4* k, float4* v, float4* k_cache, float4* v_cache, int* cache_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = idx + (cache_length[0] - num_tokens) * dim;
    if (idx < total) {
        k_cache[offset] = k[idx];
        v_cache[offset] = v[idx];
    }
}

__global__ void permute_kernel(int num_tokens, int a, int b, float4* input, float4* output) {
    int row = blockIdx.x;
    int input_offset = row * (a + b + b);
    int output_offset = row * a;
    for (int i = threadIdx.x; i < a; i += blockDim.x) {
        output[output_offset + i] = input[input_offset + i];
    }
    input_offset += a;
    output_offset = num_tokens * a + row * b;
    for (int i = threadIdx.x; i < b; i += blockDim.x) {
        output[output_offset + i] = input[input_offset + i];
    }
    input_offset += b;
    output_offset = num_tokens * (a + b) + row * b;
    for (int i = threadIdx.x; i < b; i += blockDim.x) {
        output[output_offset + i] = input[input_offset + i];
    }
}

template <typename T>
void copy_to_kvcache(const Stream& stream, int num_tokens, T* k, T* v, KVCache<T>* kv_cache, int* cache_length) {
    int dim = kv_cache->dim / (16/sizeof(T));
    int total = num_tokens * dim;
    copy_to_kvcache_kernel<<<CEIL_DIV(total, 256), 256, 0, stream.stream>>>(num_tokens, dim, total, (float4*)k, (float4*)v, (float4*)kv_cache->k_cache, (float4*)kv_cache->v_cache, cache_length);
}

template <typename T>
void permute(const Stream& stream, int num_tokens, int a, int b, T* input, T* output) {
    a = a / (16/sizeof(T));
    b = b / (16/sizeof(T));
    permute_kernel<<<num_tokens, 512, 0, stream.stream>>>(num_tokens, a, b, (float4*)input, (float4*)output);
}

}

template <typename T>
struct ATTN {
    T* output;
    virtual void init_weight_ptr(Memory* memory) = 0;
    virtual int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) = 0;
    virtual void load_to_storage(std::string name, void* ptr) = 0;
    virtual void prefill(const Stream& stream, int32_t num_tokens, int32_t num_history_tokens, T* input, T* prev_output, int32_t* position_ids, KVCache<T>* kv_cache) = 0;
    virtual void decode(const Stream& stream, int32_t num_tokens, int32_t padded_length, T* input, T* prev_output, int32_t* position_ids, int32_t* cache_length, const Mask& mask, KVCache<T>* kv_cache) = 0;
};

template <typename T>
struct Attention : ATTN<T> {
    int hidden_size;
    int num_attention_heads;
    int num_key_value_heads;
    int head_dim;
    float rms_norm_eps;
    float norm_scale;

    Norm<T> *attn_norm;
    Linear<T> *q_proj, *k_proj, *v_proj;
    Linear<T> *o_proj;

    T* attn_output;
    float *softmax_lse, *softmax_lse_accum, *oaccum;

    Attention(int hidden_size, int num_attention_heads, int num_key_value_heads, int head_dim, float rms_norm_eps, float norm_scale = 1.0) {
        this->hidden_size = hidden_size;
        this->num_attention_heads = num_attention_heads;
        this->num_key_value_heads = num_key_value_heads;
        this->head_dim = head_dim;
        this->rms_norm_eps = rms_norm_eps;
        this->norm_scale = norm_scale;

        this->attn_norm = new RMSNorm<T>(hidden_size, rms_norm_eps);
        this->q_proj = new Linear<T>(hidden_size, num_attention_heads * head_dim);
        this->k_proj = new Linear<T>(hidden_size, num_key_value_heads * head_dim);
        this->v_proj = new Linear<T>(hidden_size, num_key_value_heads * head_dim);
        this->o_proj = new Linear<T>(hidden_size, num_attention_heads * head_dim);
    }

    void init_weight_ptr(Memory* memory) {
        this->attn_norm->init_weight_ptr(memory);
        this->q_proj->init_weight_ptr(memory);
        this->k_proj->init_weight_ptr(memory);
        this->v_proj->init_weight_ptr(memory);
        this->o_proj->init_weight_ptr(memory);
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        int64_t last = this->attn_norm->init_output_ptr(memory, num_tokens, offset);
        last = this->q_proj->init_output_ptr(memory, num_tokens, last);
        last = this->k_proj->init_output_ptr(memory, num_tokens, last);
        last = this->v_proj->init_output_ptr(memory, num_tokens, last);
        
        memory->allocate((void**)&this->attn_output, offset);
        int64_t softmax_lse_end = memory->allocate((void**)&this->softmax_lse, last, num_tokens * this->num_attention_heads * sizeof(float));
        int64_t softmax_lse_accum_end = memory->allocate((void**)&this->softmax_lse_accum, softmax_lse_end, num_tokens * this->num_attention_heads * sizeof(float));
        int64_t oaccum_end = memory->allocate((void**)&this->oaccum, softmax_lse_accum_end, num_tokens * this->num_attention_heads * this->head_dim * sizeof(float));

        int64_t o_proj_end = this->o_proj->init_output_ptr(memory, num_tokens, last);
        this->output = this->o_proj->output;

        return std::max(oaccum_end, o_proj_end);
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.find("q_proj") != std::string::npos) {
            this->q_proj->load_to_storage(name, ptr);
        } else if (name.find("k_proj") != std::string::npos) {
            this->k_proj->load_to_storage(name, ptr);
        } else if (name.find("v_proj") != std::string::npos) {
            this->v_proj->load_to_storage(name, ptr);
        } else if (name.find("o_proj") != std::string::npos) {
            this->o_proj->load_to_storage(name, ptr);
        } else if (name.find("input_layernorm") != std::string::npos) {
            this->attn_norm->load_to_storage(name, ptr);
        } else {
            throw std::invalid_argument("Unsupported name " + name);
        }
    }

    void prefill(const Stream& stream, int32_t num_tokens, int32_t num_history_tokens, T* input, T* prev_output, int32_t* position_ids, KVCache<T>* kv_cache) {
        T* k_cache = kv_cache->offset_k(num_history_tokens);
        T* v_cache = kv_cache->offset_v(num_history_tokens);

        if (prev_output != nullptr) {
            elementwise_scale(stream, num_tokens, this->hidden_size, prev_output, this->norm_scale);
        }
        this->attn_norm->prefill(stream, num_tokens, input, prev_output);
        this->q_proj->prefill(stream, num_tokens, this->attn_norm->output);
        this->k_proj->prefill(stream, num_tokens, this->attn_norm->output, k_cache);
        this->v_proj->prefill(stream, num_tokens, this->attn_norm->output, v_cache);
        kv_cache->rotary_embedding->prefill(stream, num_tokens, this->num_attention_heads, this->num_key_value_heads, this->q_proj->output, k_cache, position_ids);

        mha_fwd_kvcache(
            TypeTraits<T>::type_code()==1,
            1,
            num_tokens,
            num_history_tokens+num_tokens,
            num_tokens,
            this->num_attention_heads,
            this->num_key_value_heads,
            this->head_dim,
            this->q_proj->output,
            kv_cache->k_cache,
            kv_cache->v_cache,
            nullptr,
            Mask(nullptr),
            this->attn_output,
            this->softmax_lse,
            this->softmax_lse_accum,
            this->oaccum,
            rsqrtf(float(this->head_dim)),
            true,
            -1,
            -1,
            0,
            stream.stream
        );

        // flash attention and put output to attn_norm->output
        this->o_proj->prefill(stream, num_tokens, this->attn_output);
    }

    void decode(const Stream& stream, int32_t num_tokens, int32_t padded_length, T* input, T* prev_output, int32_t* position_ids, int32_t* cache_length, const Mask& mask, KVCache<T>* kv_cache) {
        if (prev_output != nullptr) {
            elementwise_scale(stream, num_tokens, this->hidden_size, prev_output, this->norm_scale);
        }
        this->attn_norm->prefill(stream, num_tokens, input, prev_output);
        T *q, *k, *v;
        int merge_dim_out = (this->num_attention_heads + 2 * this->num_key_value_heads) * this->head_dim;
        if (num_tokens > 1) {
            linear<T>(stream, num_tokens, this->hidden_size, merge_dim_out, this->attn_norm->output, this->q_proj->weight, this->v_proj->output);
            permute(stream, num_tokens, this->num_attention_heads * this->head_dim, this->num_key_value_heads * this->head_dim, this->v_proj->output, this->q_proj->output);
        } else {
            linear<T>(stream, num_tokens, this->hidden_size, merge_dim_out, this->attn_norm->output, this->q_proj->weight, this->q_proj->output);
        }
        q = this->q_proj->output;
        k = q + num_tokens * this->num_attention_heads * this->head_dim;
        v = k + num_tokens * this->num_key_value_heads * this->head_dim;
        kv_cache->rotary_embedding->prefill(stream, num_tokens, this->num_attention_heads, this->num_key_value_heads, q, k, position_ids);

        copy_to_kvcache(stream, num_tokens, k, v, kv_cache, cache_length);

        mha_fwd_kvcache(
            TypeTraits<T>::type_code()==1,
            1,
            num_tokens,
            padded_length,
            num_tokens,
            this->num_attention_heads,
            this->num_key_value_heads,
            this->head_dim,
            q,
            kv_cache->k_cache,
            kv_cache->v_cache,
            cache_length,
            mask,
            this->attn_output,
            this->softmax_lse,
            this->softmax_lse_accum,
            this->oaccum,
            rsqrtf(float(this->head_dim)),
            true,
            -1,
            -1,
            0,
            stream.stream
        );

        // flash attention and put output to attn_norm->output
        this->o_proj->prefill(stream, num_tokens, this->attn_output);
    }
};