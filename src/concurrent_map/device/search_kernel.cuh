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

//=== Individual search kernel:
template <typename KeyT, typename ValueT>
__global__ void search_table(
    KeyT* d_queries,
    KeyT* d_index,
    ValueT* d_results,
    uint32_t totkmers,
    GpuSlabHashContext<KeyT, ValueT, SlabHashTypeT::ConcurrentMap> slab_hash) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t laneId = threadIdx.x & 0x1F;

    if ((tid - laneId) >= totkmers) {
    	return;
    }

    // initializing the memory allocator on each warp:
    slab_hash.getAllocatorContext().initAllocator(tid, laneId);

  KeyT myQuery = 0;
  ValueT myResult = static_cast<ValueT>(SEARCH_NOT_FOUND);
  uint32_t myBucket = 0;
  bool to_search = false;
  uint32_t index = 0 ;
  int num_div =4;
/////////////////////////////////
	if(tid==0){
	 printf("start search\n");
	}
        if(tid<totkmers){

                        index = d_index[tid];
                        myQuery = d_queries[tid];
                        myBucket = slab_hash.computeBucketindex(myQuery,(int)index, num_div);
	//		myBucket = slab_hash.computeBucket(myQuery);
                        to_search = true;
                
        }
       // __syncthreads();

   slab_hash.searchKey(to_search, laneId, myQuery, myResult, myBucket);
  // writing back the results:
  if (tid < totkmers) {
    d_results[tid] = myResult;
    //printf("myResult :%d\n",myResult);
  }
}

//=== Individual search kernel:
template <typename KeyT, typename ValueT>
__global__ void search_table_on_reads(
    KeyT* d_key_blocks,
    KeyT* d_whitelist_blocks,
    ValueT* d_results,
    uint64_t tot_whitelist_blocks,
    GpuSlabHashContext<KeyT, ValueT, SlabHashTypeT::ConcurrentMap> slab_hash) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t laneId = threadIdx.x & 0x1F;

    if ((tid - laneId) >= 32*tot_whitelist_blocks) {
        return;
    }

    // initializing the memory allocator on each warp:
    slab_hash.getAllocatorContext().initAllocator(tid, laneId);

  	KeyT myQuery = 0;
  	ValueT myResult = static_cast<ValueT>(SEARCH_NOT_FOUND);
  	uint32_t myBucket = 0;
  	bool to_search = false;
  	uint32_t index = 0 ;
  	int num_div =4;
        uint32_t block_index = 0;
        uint32_t bit_index = 0;
        block_index = tid/32;
        bit_index = tid%32;

	//check the thread Id and the position on the whitelist	
	if(tid<32*tot_whitelist_blocks && (((uint32_t) (d_whitelist_blocks[tid/32]>>(31-tid%32))&(0x00000001))!=0x0)){

        //                myBucket = slab_hash.computeBucketindex(myQuery,(int)index, num_div);
        	block_index = tid/16;
                bit_index = (tid%16)*2;
		
               	myQuery = (d_key_blocks[block_index] << bit_index)|
                          (bit_index!=0
                          		?(d_key_blocks[block_index+1]>>(32-bit_index))
                                        :0x0);

                myBucket = slab_hash.computeBucket(myQuery);
                to_search = true;

        }

	slab_hash.searchKey(to_search, laneId, myQuery, myResult, myBucket);
  	// writing back the results:
 	 if (tid < 32*tot_whitelist_blocks&&(((uint32_t) (d_whitelist_blocks[tid/32]>>(31-tid%32))&(0x00000001))!=0x0)) {
    		d_results[tid] = myResult;
  	}
}

//=== Bulk search kernel:
template <typename KeyT, typename ValueT>
__global__ void search_table_bulk(
    KeyT* d_queries,
    ValueT* d_results,
    uint32_t num_queries,
    GpuSlabHashContext<KeyT, ValueT, SlabHashTypeT::ConcurrentMap> slab_hash) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t laneId = threadIdx.x & 0x1F;

  if ((tid - laneId) >= num_queries) {
    return;
  }

  // initializing the memory allocator on each warp:
  slab_hash.getAllocatorContext().initAllocator(tid, laneId);

  KeyT myQuery = 0;
  ValueT myResult = static_cast<ValueT>(SEARCH_NOT_FOUND);
  uint32_t myBucket = 0;
  if (tid < num_queries) {
    myQuery = d_queries[tid];
    myBucket = slab_hash.computeBucket(myQuery);
  }

  slab_hash.searchKeyBulk(laneId, myQuery, myResult, myBucket);

  // writing back the results:
  if (tid < num_queries) {
    d_results[tid] = myResult;
  }
}
