#pragma once
#include "../trait.cuh"
#include "../utils.cuh"
#include "../flash_attn/flash_api.hpp"
#include "norm.cuh"
#include "linear.cuh"
#include "rotary.cuh"
#include "kvcache.cuh"
#include "mask.cuh"
#include "attn.cuh"
#include <cuda_runtime.h>

namespace {
__global__ void split_kernel(float4* input, float4* left, float4* right, int left_dim, int right_dim, int whole_dim) {
    int row = (blockIdx.x * blockDim.y + threadIdx.y);
    int tid = threadIdx.x;
    for (int i = tid; i < left_dim; i += blockDim.x) {
        left[row * left_dim + i] = input[row * whole_dim + i];
    }
    if (right != nullptr) {
        for (int i = tid; i < right_dim; i += blockDim.x) {
            right[row * right_dim + i] = input[row * whole_dim + left_dim + i];
        }
    }
}

template<bool repeat_right>
__global__ void merge_kernel(float4* output, float4* left, float4* right, int left_dim, int right_dim, int whole_dim) {
    int row = (blockIdx.x * blockDim.y + threadIdx.y);
    int tid = threadIdx.x;
    for (int i = tid; i < left_dim; i += blockDim.x) {
        output[row * whole_dim + i] = left[row * left_dim + i];
    }
    if (right != nullptr) {
        int row_right;
        if constexpr (repeat_right) {
            row_right = blockIdx.x;
        } else {
            row_right = row;
        }
        for (int i = tid; i < right_dim; i += blockDim.x) {
            output[row * whole_dim + left_dim + i] = right[row_right * right_dim + i];
        }
    }
}

template<bool repeat_right>
__global__ void merge_offset_kernel(float4* _output, float4* left, float4* right, int left_dim, int right_dim, int whole_dim, int32_t* cache_length) {
    int row = (blockIdx.x * blockDim.y + threadIdx.y);
    float4* output = _output + (cache_length[0] - gridDim.x) * blockDim.y * whole_dim;
    int tid = threadIdx.x;
    for (int i = tid; i < left_dim; i += blockDim.x) {
        output[row * whole_dim + i] = left[row * left_dim + i];
    }
    if (right != nullptr) {
        int row_right;
        if constexpr (repeat_right) {
            row_right = blockIdx.x;
        } else {
            row_right = row;
        }
        for (int i = tid; i < right_dim; i += blockDim.x) {
            output[row * whole_dim + left_dim + i] = right[row_right * right_dim + i];
        }
    }
}
}

template<typename T>
void split(const Stream& stream, const T* input, T* left, T* right, int num_tokens, int num_heads, int left_dim, int right_dim) {
    int bl = 16 / sizeof(T);
    left_dim = left_dim / bl;
    right_dim = right_dim / bl;
    split_kernel<<<num_tokens, dim3(8, num_heads), 0, stream.stream>>>((float4*)input, (float4*)left, (float4*)right, left_dim, right_dim, left_dim + right_dim);
}

template<typename T>
void merge(const Stream& stream, const T* left, const T* right, T* output, int num_tokens, int num_heads, int left_dim, int right_dim, bool repeat_right = false) {
    int bl = 16 / sizeof(T);
    left_dim = left_dim / bl;
    right_dim = right_dim / bl;
    if (repeat_right) {
        merge_kernel<true><<<num_tokens, dim3(8, num_heads), 0, stream.stream>>>((float4*)output, (float4*)left, (float4*)right, left_dim, right_dim, left_dim + right_dim);
    } else {
        merge_kernel<false><<<num_tokens, dim3(8, num_heads), 0, stream.stream>>>((float4*)output, (float4*)left, (float4*)right, left_dim, right_dim, left_dim + right_dim);
    }
}

template<typename T>
void merge_offset(const Stream& stream, const T* left, const T* right, T* output, int32_t* cache_length, int num_tokens, int num_heads, int left_dim, int right_dim, bool repeat_right = false) {
    int bl = 16 / sizeof(T);
    left_dim = left_dim / bl;
    right_dim = right_dim / bl;
    if (repeat_right) {
        merge_offset_kernel<true><<<num_tokens, dim3(8, num_heads), 0, stream.stream>>>((float4*)output, (float4*)left, (float4*)right, left_dim, right_dim, left_dim + right_dim, cache_length);
    } else {
        merge_offset_kernel<false><<<num_tokens, dim3(8, num_heads), 0, stream.stream>>>((float4*)output, (float4*)left, (float4*)right, left_dim, right_dim, left_dim + right_dim, cache_length);
    }
}

template <typename T>
struct AttentionMLA : ATTN<T> {
    int hidden_size;
    int num_attention_heads;
    int num_key_value_heads;
    int qk_nope_head_dim;
    int qk_rope_head_dim;
    int qk_head_dim;
    int v_head_dim;
    int q_lora_rank;
    int kv_lora_rank;
    float rms_norm_eps;
    float norm_scale;

    Norm<T> *attn_norm, *q_a_layernorm, *kv_a_layernorm;
    Linear<T> *q_a_proj, *q_b_proj, *kv_a_proj_with_mqa, *kv_b_proj;
    Linear<T> *o_proj;
    T *q_nope, *q_pe, *compressed_kv, *k_nope, *k_pe, *v_nope, *q;

    T* attn_output;
    float *softmax_lse, *softmax_lse_accum, *oaccum;

    AttentionMLA(int hidden_size, int num_attention_heads, int num_key_value_heads, int head_dim, float rms_norm_eps, float norm_scale = 1.0) {
        this->hidden_size = hidden_size;
        this->num_attention_heads = num_attention_heads;
        this->num_key_value_heads = num_key_value_heads;
        this->qk_nope_head_dim = 128;
        this->qk_rope_head_dim = 64;
        this->qk_head_dim = 192;
        this->v_head_dim = 128;
        this->q_lora_rank = 768;
        this->kv_lora_rank = 256;
        this->rms_norm_eps = rms_norm_eps;
        this->norm_scale = norm_scale;

        this->attn_norm = new RMSNorm<T>(hidden_size, rms_norm_eps);

        this->q_a_proj = new Linear<T>(hidden_size, q_lora_rank);
        this->q_a_layernorm = new RMSNorm<T>(q_lora_rank, rms_norm_eps);
        this->q_b_proj = new Linear<T>(q_lora_rank, num_attention_heads * qk_head_dim);

        this->kv_a_proj_with_mqa = new Linear<T>(hidden_size, this->kv_lora_rank + this->qk_rope_head_dim);
        this->kv_a_layernorm = new RMSNorm<T>(this->kv_lora_rank, rms_norm_eps);
        this->kv_b_proj = new Linear<T>(this->kv_lora_rank, num_attention_heads * (this->qk_nope_head_dim + this->v_head_dim));

        this->o_proj = new Linear<T>(num_attention_heads * v_head_dim, hidden_size);
    }

    void init_weight_ptr(Memory* memory) {
        this->attn_norm->init_weight_ptr(memory);
        this->q_a_proj->init_weight_ptr(memory);
        this->q_a_layernorm->init_weight_ptr(memory);
        this->q_b_proj->init_weight_ptr(memory);
        this->kv_a_proj_with_mqa->init_weight_ptr(memory);
        this->kv_a_layernorm->init_weight_ptr(memory);
        this->kv_b_proj->init_weight_ptr(memory);
        this->o_proj->init_weight_ptr(memory);
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        int64_t last = this->attn_norm->init_output_ptr(memory, num_tokens, offset);
        last = this->q_a_proj->init_output_ptr(memory, num_tokens, last);
        last = this->q_a_layernorm->init_output_ptr(memory, num_tokens, last);
        last = this->q_b_proj->init_output_ptr(memory, num_tokens, last);
        last = this->kv_a_proj_with_mqa->init_output_ptr(memory, num_tokens, last);
        last = this->kv_a_layernorm->init_output_ptr(memory, num_tokens, last);
        last = this->kv_b_proj->init_output_ptr(memory, num_tokens, last);

        last = memory->allocate((void**)&this->q_nope, last, num_tokens * this->num_attention_heads * this->qk_nope_head_dim * sizeof(T));
        last = memory->allocate((void**)&this->q_pe, last, num_tokens * this->num_attention_heads * this->qk_rope_head_dim * sizeof(T));
        last = memory->allocate((void**)&this->compressed_kv, last, num_tokens * 1 * this->kv_lora_rank * sizeof(T));
        last = memory->allocate((void**)&this->k_pe, last, num_tokens * 1 * this->qk_rope_head_dim * sizeof(T));
        last = memory->allocate((void**)&this->k_nope, last, num_tokens * this->num_attention_heads * this->qk_nope_head_dim * sizeof(T));
        last = memory->allocate((void**)&this->v_nope, last, num_tokens * this->num_attention_heads * this->v_head_dim * sizeof(T));
        last = memory->allocate((void**)&this->q, last, num_tokens * this->num_attention_heads * this->qk_head_dim * sizeof(T));

        memory->allocate((void**)&this->attn_output, offset);
        int64_t softmax_lse_end = memory->allocate((void**)&this->softmax_lse, last, num_tokens * this->num_attention_heads * sizeof(float));
        int64_t softmax_lse_accum_end = memory->allocate((void**)&this->softmax_lse_accum, softmax_lse_end, num_tokens * this->num_attention_heads * sizeof(float));
        int64_t oaccum_end = memory->allocate((void**)&this->oaccum, softmax_lse_accum_end, num_tokens * this->num_attention_heads * this->qk_head_dim * sizeof(float));

        int64_t o_proj_end = this->o_proj->init_output_ptr(memory, num_tokens, last);
        this->output = this->o_proj->output;

        return std::max(oaccum_end, o_proj_end);
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.find("q_a_proj") != std::string::npos) {
            this->q_a_proj->load_to_storage(name, ptr);
        } else if (name.find("q_a_layernorm") != std::string::npos) {
            this->q_a_layernorm->load_to_storage(name, ptr);
        } else if (name.find("q_b_proj") != std::string::npos) {
            this->q_b_proj->load_to_storage(name, ptr);
        } else if (name.find("kv_a_proj_with_mqa") != std::string::npos) {
            this->kv_a_proj_with_mqa->load_to_storage(name, ptr);
        } else if (name.find("kv_a_layernorm") != std::string::npos) {
            this->kv_a_layernorm->load_to_storage(name, ptr);
        } else if (name.find("kv_b_proj") != std::string::npos) {
            this->kv_b_proj->load_to_storage(name, ptr);
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

        this->q_a_proj->prefill(stream, num_tokens, this->attn_norm->output);
        this->q_a_layernorm->prefill(stream, num_tokens, this->q_a_proj->output, nullptr);
        this->q_b_proj->prefill(stream, num_tokens, this->q_a_layernorm->output);
        split(stream, this->q_b_proj->output, this->q_nope, this->q_pe, num_tokens, this->num_attention_heads, this->qk_nope_head_dim, this->qk_rope_head_dim);

        this->kv_a_proj_with_mqa->prefill(stream, num_tokens, this->attn_norm->output);
        split(stream, this->kv_a_proj_with_mqa->output, this->compressed_kv, this->k_pe, num_tokens, 1, this->kv_lora_rank, this->qk_rope_head_dim);
        this->kv_a_layernorm->prefill(stream, num_tokens, this->compressed_kv, nullptr);
        this->kv_b_proj->prefill(stream, num_tokens, this->kv_a_layernorm->output);
        split(stream, this->kv_b_proj->output, this->k_nope, this->v_nope, num_tokens, this->num_attention_heads, this->qk_nope_head_dim, this->v_head_dim);

        kv_cache->rotary_embedding->prefill(stream, num_tokens, this->num_attention_heads, 1, this->q_pe, this->k_pe, position_ids);

        merge(stream, this->q_nope, this->q_pe,  this->q, num_tokens, this->num_attention_heads, this->qk_nope_head_dim, this->qk_rope_head_dim);
        merge(stream, this->k_nope, this->k_pe,  k_cache, num_tokens, this->num_attention_heads, this->qk_nope_head_dim, this->qk_rope_head_dim, true);
        merge(stream, this->v_nope, (T*)nullptr, v_cache, num_tokens, this->num_attention_heads, this->v_head_dim, this->qk_head_dim - this->v_head_dim);

        mha_fwd_kvcache(
            TypeTraits<T>::type_code()==1,
            1,
            num_tokens,
            num_history_tokens+num_tokens,
            num_tokens,
            this->num_attention_heads,
            this->num_key_value_heads,
            this->qk_head_dim,
            this->q,
            kv_cache->k_cache,
            kv_cache->v_cache,
            nullptr,
            Mask(nullptr),
            this->attn_output,
            this->softmax_lse,
            this->softmax_lse_accum,
            this->oaccum,
            rsqrtf(float(this->qk_head_dim)),
            true,
            -1,
            -1,
            0,
            stream.stream
        );

        split(stream, this->attn_output, this->v_nope, (T*)nullptr, num_tokens, this->num_attention_heads, this->v_head_dim, this->qk_head_dim - this->v_head_dim);

        this->o_proj->prefill(stream, num_tokens, this->v_nope);
    }

    void decode(const Stream& stream, int32_t num_tokens, int32_t padded_length, T* input, T* prev_output, int32_t* position_ids, int32_t* cache_length, const Mask& mask, KVCache<T>* kv_cache) {
        if (prev_output != nullptr) {
            elementwise_scale(stream, num_tokens, this->hidden_size, prev_output, this->norm_scale);
        }
        this->attn_norm->prefill(stream, num_tokens, input, prev_output);

        this->q_a_proj->prefill(stream, num_tokens, this->attn_norm->output);
        this->q_a_layernorm->prefill(stream, num_tokens, this->q_a_proj->output, nullptr);
        this->q_b_proj->prefill(stream, num_tokens, this->q_a_layernorm->output);
        split(stream, this->q_b_proj->output, this->q_nope, this->q_pe, num_tokens, this->num_attention_heads, this->qk_nope_head_dim, this->qk_rope_head_dim);

        this->kv_a_proj_with_mqa->prefill(stream, num_tokens, this->attn_norm->output);
        split(stream, this->kv_a_proj_with_mqa->output, this->compressed_kv, this->k_pe, num_tokens, 1, this->kv_lora_rank, this->qk_rope_head_dim);
        this->kv_a_layernorm->prefill(stream, num_tokens, this->compressed_kv, nullptr);
        this->kv_b_proj->prefill(stream, num_tokens, this->kv_a_layernorm->output);
        split(stream, this->kv_b_proj->output, this->k_nope, this->v_nope, num_tokens, this->num_attention_heads, this->qk_nope_head_dim, this->v_head_dim);

        kv_cache->rotary_embedding->prefill(stream, num_tokens, this->num_attention_heads, 1, this->q_pe, this->k_pe, position_ids);

        merge(stream, this->q_nope, this->q_pe,  this->q, num_tokens, this->num_attention_heads, this->qk_nope_head_dim, this->qk_rope_head_dim);

        merge_offset(stream, this->k_nope, this->k_pe,  kv_cache->k_cache, cache_length, num_tokens, this->num_attention_heads, this->qk_nope_head_dim, this->qk_rope_head_dim, true);
        merge_offset(stream, this->v_nope, (T*)nullptr, kv_cache->v_cache, cache_length, num_tokens, this->num_attention_heads, this->v_head_dim, this->qk_head_dim - this->v_head_dim);

        mha_fwd_kvcache(
            TypeTraits<T>::type_code()==1,
            1,
            num_tokens,
            padded_length,
            num_tokens,
            this->num_attention_heads,
            this->num_key_value_heads,
            this->qk_head_dim,
            this->q,
            kv_cache->k_cache,
            kv_cache->v_cache,
            cache_length,
            mask,
            this->attn_output,
            this->softmax_lse,
            this->softmax_lse_accum,
            this->oaccum,
            rsqrtf(float(this->qk_head_dim)),
            true,
            -1,
            -1,
            0,
            stream.stream
        );

        split(stream, this->attn_output, this->v_nope, (T*)nullptr, num_tokens, this->num_attention_heads, this->v_head_dim, this->qk_head_dim - this->v_head_dim);

        this->o_proj->prefill(stream, num_tokens, this->v_nope);
    }
};