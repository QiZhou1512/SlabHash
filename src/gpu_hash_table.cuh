/*
 * Copyright 2019 Saman Ashkiani
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied. See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "slab_hash.cuh"

/*
 * This class acts as a helper class to simplify simulations around different
 * kinds of slab hash implementations
 */
template <typename KeyT,
          typename ValueT,
          SlabHashTypeT SlabHashT>
class gpu_hash_table {
 private:
  
  uint64_t tot_key_blocks_;
  uint64_t tot_whitelist_blocks_;
  uint64_t num_kmers_;
  uint32_t num_buckets_;
  int64_t seed_;
  bool req_values_;
  bool identity_hash_;

 public:
  // Slab hash invariant
  GpuSlabHash<KeyT, ValueT, SlabHashT>* slab_hash_;

  // the dynamic allocator that is being used for slab hash
  DynamicAllocatorT* dynamic_allocator_;

  uint32_t device_idx_;

  // main arrays to hold keys, values, queries, results, etc.
  uint32_t* d_key_blocks;
  uint32_t* d_whitelist_blocks;
  ValueT* d_value_;
  ValueT* d_result_;
  gpu_hash_table(
		 uint64_t tot_key_blocks,
		 uint64_t tot_whitelist_blocks,
		 uint64_t num_kmers,
                 uint32_t num_buckets,
                 const uint32_t device_idx, 
                 const int64_t seed,
                 const bool req_values = false,
                 const bool identity_hash = false,
                 const bool verbose = false
		 )
      :
	tot_key_blocks_(tot_key_blocks),
	tot_whitelist_blocks_(tot_whitelist_blocks),
	num_kmers_(num_kmers), 
        num_buckets_(num_buckets),
        seed_(seed),
        req_values_(req_values),
        slab_hash_(nullptr),
        identity_hash_(identity_hash),
        dynamic_allocator_(nullptr),
        device_idx_(device_idx) {
    int32_t devCount = 0;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&devCount));
    assert(device_idx_ < devCount);

    CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));

    // allocating key, value arrays:
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_key_blocks, sizeof(KeyT) * tot_key_blocks));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_whitelist_blocks, sizeof(ValueT) * tot_whitelist_blocks_)); 
    CHECK_CUDA_ERROR( cudaMalloc((void**)&d_result_, sizeof(ValueT) * 32*tot_whitelist_blocks));

    // allocate an initialize the allocator:
    dynamic_allocator_ = new DynamicAllocatorT();

    // slab hash:
    slab_hash_ = new GpuSlabHash<KeyT, ValueT, SlabHashT>(
        num_buckets_, dynamic_allocator_, device_idx_, seed_, identity_hash_);
    if (verbose) {
      std::cout << slab_hash_->to_string() << std::endl;
    }
  }

  ~gpu_hash_table() {
    CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));
    CHECK_CUDA_ERROR(cudaFree(d_key_blocks));
    if (req_values_) {
      CHECK_CUDA_ERROR(cudaFree(d_value_));
    }
    CHECK_CUDA_ERROR(cudaFree(d_whitelist_blocks));
   // CHECK_CUDA_ERROR(cudaFree(d_query_));
    CHECK_CUDA_ERROR(cudaFree(d_result_));

    // delete the dynamic allocator:
    delete dynamic_allocator_;

    // slab hash:
    delete (slab_hash_);
  }


  //insert bulk function        
  float hash_insert_on_reads (uint32_t* h_key_blocks,uint32_t* h_whitelist_blocks) {
    // moving key-values to the device:
    CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));
    CHECK_CUDA_ERROR(cudaMemcpy(d_key_blocks, h_key_blocks, sizeof(KeyT) * tot_key_blocks_, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_whitelist_blocks, h_whitelist_blocks, sizeof(ValueT) * tot_whitelist_blocks_, cudaMemcpyHostToDevice));
    float temp_time = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // calling slab-hash's bulk insert procedure:
    slab_hash_->insertUpdateOnReads(d_key_blocks,d_whitelist_blocks,tot_whitelist_blocks_);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&temp_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return temp_time;
}

  float hash_search_on_reads(ValueT* h_result) {
    CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));
    CHECK_CUDA_ERROR(cudaMemset(d_result_, 0xFF, sizeof(ValueT) * 32*tot_whitelist_blocks_));

    float temp_time = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // == calling slab hash's individual search:
    slab_hash_->searchIndividualOnReads(d_key_blocks, d_whitelist_blocks,d_result_,tot_whitelist_blocks_ );
    //==

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&temp_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    CHECK_CUDA_ERROR(cudaMemcpy(h_result, d_result_,
                                sizeof(ValueT) * 32*tot_whitelist_blocks_,
                                cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    return temp_time;
  }





  float measureLoadFactor(int flag = 0) {
    return slab_hash_->computeLoadFactor(flag);
  }
};
