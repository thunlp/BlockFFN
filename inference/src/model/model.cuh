#pragma once
#include "memory.cuh"
#include "embedding.cuh"
#include "norm.cuh"
#include "linear.cuh"
#include "layer.cuh"
#include "kvcache.cuh"
#include "mask.cuh"
#include <algorithm>
#include <cuda_runtime.h>
#include <vector>
#include <regex>

struct Model {
    virtual int init_storage() = 0;
    virtual void load_to_storage(std::string name, void* ptr) = 0;
    virtual void prefill(int32_t num_tokens, int32_t num_history_tokens, int32_t* input, int32_t* position_ids, void* output) = 0;
    virtual void decode(int32_t num_tokens, int32_t padded_length, int32_t* input, int32_t* position_ids, int32_t* cache_length, uint64_t* mask_2d, void* output) = 0;

    virtual void draft(int32_t *tree_draft_ids, int32_t *tree_position_ids, int32_t *cache_length, uint64_t* attn_mask, int32_t* tree_parent) = 0;
    virtual int verify(int32_t num_tokens, int32_t* pred, int32_t* gt, int32_t* position_ids, int32_t* cache_length, uint64_t* attn_mask, int32_t* tree_parent) = 0;
    /* verify should find max accept length (based on tree_parent and position_ids) and return, fix kvcache (based on position_ids), and make pred[:accept_length] the accept path (based on attn_mask and position_ids) */
};

template <typename T>
struct ModelImpl : Model {
    Memory* memory;

    int vocab_size;
    int num_hidden_layers;
    int hidden_size;
    int intermediate_size;
    int num_attention_heads;
    int num_key_value_heads;
    int head_dim;
    float rms_norm_eps;

    int chunk_length;

    KVCacheManager<T>* kv_caches;

    Embedding<T>* embedding;
    std::vector<Layer<T>*> layers;
    RMSNorm<T>* norm;
    LMHead<T>* lm_head;
    float norm_scale;

    ModelImpl(
        int64_t memory_limit,
        void* memory_pool,
        int vocab_size,
        int num_hidden_layers,
        int hidden_size,
        int intermediate_size,
        int num_attention_heads,
        int num_key_value_heads,
        int head_dim,
        float rms_norm_eps,
        int chunk_length,
        bool use_kernel
    ) {
        this->vocab_size = vocab_size;
        this->num_hidden_layers = num_hidden_layers;
        this->hidden_size = hidden_size;
        this->intermediate_size = intermediate_size;
        this->num_attention_heads = num_attention_heads;
        this->num_key_value_heads = num_key_value_heads;
        this->head_dim = head_dim;
        this->rms_norm_eps = rms_norm_eps;
        this->norm_scale = 1.4f / sqrtf(num_hidden_layers);

        this->chunk_length = chunk_length;
        
        memory = new Memory(memory_limit, memory_pool);

        kv_caches = new KVCacheManager<T>(num_hidden_layers, num_key_value_heads, head_dim);

        embedding = new Embedding<T>(vocab_size, hidden_size);
        for (int i = 0; i < num_hidden_layers; i++) {
            layers.push_back(new Layer<T, false>(hidden_size, intermediate_size, num_attention_heads, num_key_value_heads, head_dim, rms_norm_eps, norm_scale, use_kernel));
        }
        norm = new RMSNorm<T>(hidden_size, rms_norm_eps);
        lm_head = new LMHead<T>(hidden_size, vocab_size);
    }

    void init_weight_ptr(Memory* memory) {
        embedding->init_weight_ptr(memory);
        for (int i = 0; i < num_hidden_layers; i++) {
            layers[i]->init_weight_ptr(memory);
        }
        norm->init_weight_ptr(memory);
        lm_head->init_weight_ptr(memory);
        kv_caches->init_weight_ptr(memory);
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        int64_t embedding_end = embedding->init_output_ptr(memory, num_tokens, offset);
        int64_t layer_end = 0;
        for (int i = 0; i < num_hidden_layers; i++) {
            layer_end = layers[i]->init_output_ptr(memory, num_tokens, embedding_end);
        }
        // norm and lm_head are not used in prefill
        int64_t norm_end = norm->init_output_ptr(memory, num_tokens, layer_end);
        int64_t lm_head_end = lm_head->init_output_ptr(memory, 64, norm_end);
        return lm_head_end;
    }

    int init_storage() {
        init_weight_ptr(memory);
        int64_t kv_cache_offset = init_output_ptr(memory, chunk_length, memory->model_offset);
        kv_cache_offset = kv_caches->init_output_ptr(memory, kv_cache_offset);
        return this->kv_caches->budget;
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.substr(0, 18) == "model.embed_tokens") {
            embedding->load_to_storage(name, ptr);
        } else if (name.substr(0, 10) == "model.norm") {
            norm->load_to_storage(name, ptr);
        } else if (name.substr(0, 7) == "lm_head") {
            lm_head->load_to_storage(name, ptr);
        } else if (name.find("rotary_emb") != std::string::npos) {
            kv_caches->rotary_embedding->load_to_storage(name, ptr);
        } else if (name.substr(0, 12) == "model.layers") { // e.g. model.layers.20.attn.q_proj.weight
            std::regex layer_regex("model\\.layers\\.(\\d+)\\.(.*)");
            std::smatch matches;
            if (std::regex_search(name, matches, layer_regex)) {
                int layer_idx = std::stoi(matches[1]);
                layers[layer_idx]->load_to_storage(matches[2], ptr);
            } else {
                throw std::invalid_argument("Unsupported name (layer_idx not found): " + name);
            }
        } else {
            throw std::invalid_argument("Unsupported name " + name);
        }
    }

    void prefill_embed(int32_t num_tokens, int32_t num_history_tokens, T* embed, int32_t* position_ids, void* output) {
        T* layer_output = nullptr;
        for (int i = 0; i < num_hidden_layers; i++) {
            this->layers[i]->prefill(num_tokens, num_history_tokens, embed, layer_output, position_ids, this->kv_caches->caches[i]);
            layer_output = this->layers[i]->output;
        }
        elementwise_scale(calc_stream, num_tokens, this->hidden_size, layer_output, this->norm_scale);
        this->norm->prefill(calc_stream, num_tokens, embed, layer_output);
        this->lm_head->prefill(calc_stream, 1, this->norm->output + (num_tokens - 1) * hidden_size, (T*)output);
    }

    void prefill(int32_t num_tokens, int32_t num_history_tokens, int32_t* input, int32_t* position_ids, void* output) {
        this->embedding->prefill(calc_stream, num_tokens, input);
        prefill_embed(num_tokens, num_history_tokens, this->embedding->output, position_ids, output);
    }

    void decode_embed(int32_t num_tokens, int32_t padded_length, T* embed, int32_t* position_ids, int32_t* cache_length, uint64_t* mask_2d, void* output) {
        Mask mask(mask_2d, num_tokens, num_tokens);
        T* layer_output = nullptr;
        for (int i = 0; i < num_hidden_layers; i++) {
            this->layers[i]->decode(num_tokens, padded_length, this->embedding->output, layer_output, position_ids, cache_length, mask, this->kv_caches->caches[i]);
            layer_output = this->layers[i]->output;
        }
        elementwise_scale(calc_stream, num_tokens, this->hidden_size, layer_output, this->norm_scale);
        this->norm->prefill(calc_stream, num_tokens, this->embedding->output, layer_output);
        this->lm_head->prefill(calc_stream, num_tokens, this->norm->output, (T*)output);
    }

    void decode(int32_t num_tokens, int32_t padded_length, int32_t* input, int32_t* position_ids, int32_t* cache_length, uint64_t* mask_2d, void* output) {
        this->embedding->prefill(calc_stream, num_tokens, input);
        decode_embed(num_tokens, padded_length, this->embedding->output, position_ids, cache_length, mask_2d, output);
    }

    void draft(int32_t *tree_draft_ids, int32_t *tree_position_ids, int32_t *cache_length, uint64_t* attn_mask, int32_t* tree_parent) { throw std::runtime_error("Draft is not supported"); }
    int verify(int32_t num_tokens, int32_t* pred, int32_t* gt, int32_t* position_ids, int32_t* cache_length, uint64_t* attn_mask, int32_t* tree_parent) { throw std::runtime_error("Verify is not supported"); }
};