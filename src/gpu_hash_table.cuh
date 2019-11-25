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
  
  uint32_t max_keys_;
  uint32_t num_buckets_;
  int kmer_len_;
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
  uint32_t* d_key_;
  uint32_t* d_index_;
  ValueT* d_value_;
  uint32_t* d_query_;
  ValueT* d_result_;
  gpu_hash_table(
		 uint32_t max_keys,
                 uint32_t num_buckets,
		 int kmer_len,
                 const uint32_t device_idx, 
                 const int64_t seed,
                 const bool req_values = false,
                 const bool identity_hash = false,
                 const bool verbose = false
		 )
      : 
	max_keys_(max_keys),
        num_buckets_(num_buckets),
	kmer_len_(kmer_len),
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
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_key_, sizeof(KeyT) * max_keys_));
    
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_index_, sizeof(ValueT) * max_keys_));
    
    //CHECK_CUDA_ERROR(cudaMalloc((void**)&d_query_, sizeof(KeyT) * max_keys_));
    CHECK_CUDA_ERROR( cudaMalloc((void**)&d_result_, sizeof(ValueT) * max_keys_));

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
    CHECK_CUDA_ERROR(cudaFree(d_key_));
    if (req_values_) {
      CHECK_CUDA_ERROR(cudaFree(d_value_));
    }
    CHECK_CUDA_ERROR(cudaFree(d_index_));
   // CHECK_CUDA_ERROR(cudaFree(d_query_));
    CHECK_CUDA_ERROR(cudaFree(d_result_));

    // delete the dynamic allocator:
    delete dynamic_allocator_;

    // slab hash:
    delete (slab_hash_);
  }

  //insert bulk function	
float hash_insert (uint32_t* h_key,uint32_t* h_index, uint32_t totkmers) {
    // moving key-values to the device:
    CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));
    CHECK_CUDA_ERROR(cudaMemcpy(d_key_, h_key, sizeof(KeyT) * totkmers, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_index_, h_index, sizeof(ValueT) * totkmers, cudaMemcpyHostToDevice));
    float temp_time = 0.0f;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, 0);
    
    // calling slab-hash's bulk insert procedure:
    slab_hash_->insertUpdate(d_key_,d_index_,totkmers, kmer_len_);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&temp_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return temp_time;
} 

float hash_insert_buffered(uint32_t *h_key, uint32_t* h_index, uint32_t totkmers){
	//totkmers = 100;
	//creating the cudahost memory allocation in order to transfer data in stream
	int k = totkmers/4;
	uint32_t *h_key_cudaHost, *h_index_cudaHost, *d_key_buffer_0, *d_index_buffer_0,*d_key_buffer_1, *d_index_buffer_1;
	float temp_time = 0.0f;
	
	CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));
	cudaStream_t stream_0, stream_1;
	cudaStreamCreate( &stream_0 );
	cudaStreamCreate( &stream_1 );
	cudaHostAlloc((void**) &h_key_cudaHost,totkmers*sizeof(uint32_t),cudaHostAllocDefault);
	cudaHostAlloc((void**) &h_index_cudaHost,totkmers*sizeof(uint32_t),cudaHostAllocDefault);
	
	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_key_buffer_0, sizeof(KeyT) * k));
   	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_index_buffer_0, sizeof(ValueT) * k));

	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_key_buffer_1, sizeof(KeyT) * k));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_index_buffer_1, sizeof(ValueT) * k));	
	
	printf("%d\n",h_key[0]);
		
	for(int i = 0; i<totkmers; i++){
		//printf("%d",h_key[i]);
		h_key_cudaHost[i] = h_key[i];
		h_index_cudaHost[i] = h_index[i];
	}

    	cudaEvent_t start, stop;
    	cudaEventCreate(&start);
    	cudaEventCreate(&stop);

    	cudaEventRecord(start, 0);		
	cudaMemcpyAsync( d_key_buffer_0, h_key_cudaHost, k * sizeof(uint32_t), cudaMemcpyHostToDevice, stream_0 );
	cudaMemcpyAsync( d_index_buffer_0, h_index_cudaHost, k * sizeof(uint32_t), cudaMemcpyHostToDevice, stream_0 );
	
	for(int i = 0; i<4; i++){
		printf("index buffer swap : %d",i);
		if(i%2 == 0){
			cudaMemcpyAsync( d_key_buffer_1, h_key_cudaHost+k*(i+1), k * sizeof(uint32_t), cudaMemcpyHostToDevice, stream_1 );
			cudaMemcpyAsync( d_index_buffer_1, h_index_cudaHost+k*(i+1), k * sizeof(uint32_t), cudaMemcpyHostToDevice, stream_1 );
			slab_hash_->insertUpdate(d_key_buffer_0,d_index_buffer_0,k, kmer_len_);
		}else{
			if(i != 3){
				cudaMemcpyAsync( d_key_buffer_0, h_key_cudaHost+k*(i+1), k * sizeof(uint32_t), cudaMemcpyHostToDevice, stream_0 );
        			cudaMemcpyAsync( d_index_buffer_0, h_index_cudaHost+k*(i+1), k * sizeof(uint32_t), cudaMemcpyHostToDevice, stream_0 );
			}
			slab_hash_->insertUpdate(d_key_buffer_1, d_index_buffer_1, k, kmer_len_);
		
		} 

	}
	cudaStreamSynchronize( stream_0 );
	cudaStreamSynchronize( stream_1 );

	cudaEventRecord(stop, 0);
    	cudaEventSynchronize(stop);
    	cudaEventElapsedTime(&temp_time, start, stop);

    	cudaEventDestroy(start);
    	cudaEventDestroy(stop);
    	return temp_time;	
}


  float hash_build(KeyT* h_key, ValueT* h_value, uint32_t num_keys) {
    // moving key-values to the device:
    CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));
    CHECK_CUDA_ERROR(cudaMemcpy(d_key_, h_key, sizeof(KeyT) * num_keys,
                               cudaMemcpyHostToDevice));
    if (req_values_) {
      CHECK_CUDA_ERROR(cudaMemcpy(d_value_, h_value, sizeof(ValueT) * num_keys,
                                  cudaMemcpyHostToDevice));
    }

    float temp_time = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // calling slab-hash's bulk build procedure:
    slab_hash_->buildBulk(d_key_, d_value_, num_keys);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&temp_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return temp_time;
  }

  float hash_search(uint32_t* h_query,uint32_t* h_index , ValueT* h_result,uint32_t totkmers) {
    CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));
   /* 
	                for(int i =0 ; i<totkmers;i++){
                        for (int k = 63; 0 <= k; k--) {
                                printf("%c", (h_query[i] & (1ULL << k)) ? '1' : '0');
                        }
                        printf("\n");
                }
*/
    CHECK_CUDA_ERROR(cudaMemcpy(d_key_, h_query, sizeof(KeyT)* totkmers, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_index_, h_index, sizeof(ValueT)* totkmers, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemset(d_result_, 0xFF, sizeof(ValueT) * totkmers));   

    float temp_time = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // == calling slab hash's individual search:
    slab_hash_->searchIndividual(d_key_, d_index_,d_result_, totkmers, kmer_len_);
    //==

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&temp_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    CHECK_CUDA_ERROR(cudaMemcpy(h_result, d_result_,
                                sizeof(ValueT) * totkmers,
                                cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    return temp_time;
  }
  float hash_search_bulk(KeyT* h_query,
                         ValueT* h_result,
                         uint32_t num_queries) {
    CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));
    CHECK_CUDA_ERROR(cudaMemcpy(d_query_, h_query, sizeof(KeyT) * num_queries,
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemset(d_result_, 0xFF, sizeof(ValueT) * num_queries));

    float temp_time = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    //== slab hash's bulk search:
    slab_hash_->searchBulk(d_query_, d_result_, num_queries);
    //==

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&temp_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    CHECK_CUDA_ERROR(cudaMemcpy(h_result, d_result_,
                                sizeof(ValueT) * num_queries,
                                cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    return temp_time;
  }

  float hash_delete(KeyT* h_key, uint32_t num_keys) {
    CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));
    CHECK_CUDA_ERROR(cudaMemcpy(d_key_, h_key, sizeof(KeyT) * num_keys,
                                cudaMemcpyHostToDevice));

    float temp_time = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    //=== slab hash's deletion:
    slab_hash_->deleteIndividual(d_key_, num_keys);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&temp_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return temp_time;
  }

  float batched_operations(uint32_t* h_batch_op,
                           uint32_t* h_results,
                           uint32_t batch_size,
                           uint32_t batch_id) {
    CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));
    CHECK_CUDA_ERROR(cudaMemcpy(d_key_ + batch_id * batch_size, h_batch_op,
                                sizeof(uint32_t) * batch_size,
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemset(d_result_ + batch_id * batch_size, 0xFF,
                                sizeof(uint32_t) * batch_size));

    float temp_time = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    slab_hash_->batchedOperation(d_key_ + batch_id * batch_size, d_result_,
                               batch_size);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&temp_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    CHECK_ERROR(cudaMemcpy(
        h_results + batch_id * batch_size, d_result_ + batch_id * batch_size,
        sizeof(uint32_t) * batch_size, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    return temp_time;
  }

  float measureLoadFactor(int flag = 0) {
    return slab_hash_->computeLoadFactor(flag);
  }
};
