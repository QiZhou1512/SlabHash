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

/*
 * each thread inserts a key-value pair into the hash table
 * it is assumed all threads within a warp are present and collaborating with
 * each other with a warp-cooperative work sharing (WCWS) strategy.
 */
template <typename KeyT, typename ValueT>
__device__ __forceinline__ void
GpuSlabHashContext<KeyT, ValueT, SlabHashTypeT::ConcurrentMap>::insertPair(
    bool& to_be_inserted,
    const uint32_t& laneId,
    const KeyT& myKey,
    const ValueT& myValue,
    const uint32_t bucket_id) {
  using SlabHashT = ConcurrentMapT<KeyT, ValueT>;
  uint32_t work_queue = 0;
  uint32_t last_work_queue = 0;
  uint32_t next = SlabHashT::A_INDEX_POINTER;

  while ((work_queue = __ballot_sync(0xFFFFFFFF, to_be_inserted))) {
    // to know whether it is a base node, or a regular node
    next = (last_work_queue != work_queue)
               ? SlabHashT::A_INDEX_POINTER
               : next;  // a successfull insertion in the warp
    uint32_t src_lane = __ffs(work_queue) - 1;
    uint32_t src_bucket = __shfl_sync(0xFFFFFFFF, bucket_id, src_lane, 32);

    uint32_t src_unit_data = (next == SlabHashT::A_INDEX_POINTER)
                                 ? *(getPointerFromBucket(src_bucket, laneId))
                                 : *(getPointerFromSlab(next, laneId));
    uint64_t old_key_value_pair = 0;

    uint32_t isEmpty = (__ballot_sync(0xFFFFFFFF, src_unit_data == EMPTY_KEY)) &
                       SlabHashT::REGULAR_NODE_KEY_MASK;
    if (isEmpty == 0) {  // no empty slot available:
      uint32_t next_ptr = __shfl_sync(0xFFFFFFFF, src_unit_data, 31, 32);
      if (next_ptr == SlabHashT::EMPTY_INDEX_POINTER) {
        // allocate a new node:
        uint32_t new_node_ptr = allocateSlab(laneId);

        // TODO: experiment if it's better to use lane 0 instead
        if (laneId == 31) {
          const uint32_t* p = (next == SlabHashT::A_INDEX_POINTER)
                                  ? getPointerFromBucket(src_bucket, 31)
                                  : getPointerFromSlab(next, 31);

          uint32_t temp = atomicCAS(
              (unsigned int*)p, SlabHashT::EMPTY_INDEX_POINTER, new_node_ptr);
          // check whether it was successful, and
          // free the allocated memory otherwise
          if (temp != SlabHashT::EMPTY_INDEX_POINTER) {
            freeSlab(new_node_ptr);
          }
        }
      } else {
        next = next_ptr;
      }
    } else {  // there is an empty slot available
      int dest_lane = __ffs(isEmpty & SlabHashT::REGULAR_NODE_KEY_MASK) - 1;
      if (laneId == src_lane) {
        const uint32_t* p = (next == SlabHashT::A_INDEX_POINTER)
                                ? getPointerFromBucket(src_bucket, dest_lane)
                                : getPointerFromSlab(next, dest_lane);

        old_key_value_pair =
            atomicCAS((unsigned long long int*)p, EMPTY_PAIR_64,
                      ((uint64_t)(*reinterpret_cast<const uint32_t*>(
                           reinterpret_cast<const unsigned char*>(&myValue)))
                       << 32) |
                          *reinterpret_cast<const uint32_t*>(
                              reinterpret_cast<const unsigned char*>(&myKey)));
        if (old_key_value_pair == EMPTY_PAIR_64)
          to_be_inserted = false;  // succesfful insertion
      }
    }
    last_work_queue = work_queue;
  }
}
#pragma once
template <typename KeyT, typename ValueT>
__device__ __forceinline__ void
GpuSlabHashContext<KeyT,ValueT, SlabHashTypeT::ConcurrentMap>::insertKeyUnique(
	int& to_be_inserted,
	const uint32_t& laneId,
	const KeyT& myKey,
	const uint32_t bucket_id) {
	using SlabHashT = ConcurrentMapT<KeyT, ValueT>;
	uint32_t work_queue = 0;
	uint32_t last_work_queue = 0;
	uint32_t next = SlabHashT::A_INDEX_POINTER;

	while ((work_queue = __ballot_sync(0xFFFFFFFF, to_be_inserted==1?true:false))) {
    	// to know whether it is a base node, or a regular node
    		next = (last_work_queue != work_queue)
               		? SlabHashT::A_INDEX_POINTER
               		: next;  // a successfull insertion in the warp
    		uint32_t src_lane = __ffs(work_queue) - 1;
    		uint32_t src_bucket = __shfl_sync(0xFFFFFFFF, bucket_id, src_lane, 32);
    		uint32_t wanted_key =
        		__shfl_sync(0xFFFFFFFF,
                    	*reinterpret_cast<const uint32_t*>(
                        reinterpret_cast<const unsigned char*>(&myKey)),
			src_lane, 32);
		//get the value stored in the slab memory for every thread
    		uint32_t src_unit_data = (next == SlabHashT::A_INDEX_POINTER)
                                 ? *(getPointerFromBucket(src_bucket, laneId))
                                 : *(getPointerFromSlab(next, laneId));
    		uint64_t old_key_value_pair = 0;
		//__ballot_sync
    		uint32_t isEmpty = (__ballot_sync(0xFFFFFFFF, src_unit_data == EMPTY_KEY)) &
                       			SlabHashT::REGULAR_NODE_KEY_MASK;
    		uint32_t isFound = (__ballot_sync(0xFFFFFFFF,src_unit_data ==wanted_key)) &
                       			SlabHashT::REGULAR_NODE_KEY_MASK;
    		uint32_t needToChain = (__ballot_sync(0xFFFFFFFF,src_unit_data ==EMPTY_KEY||src_unit_data ==wanted_key)) &
                       			SlabHashT::REGULAR_NODE_KEY_MASK;
    		if (needToChain == 0) {  // no empty slot available:
      			uint32_t next_ptr = __shfl_sync(0xFFFFFFFF, src_unit_data, 31, 32);
      			if (next_ptr == SlabHashT::EMPTY_INDEX_POINTER) {
        		// allocate a new node:
        			uint32_t new_node_ptr = allocateSlab(laneId);

        		// TODO: experiment if it's better to use lane 0 instead
        			if (laneId == 31) {
          				const uint32_t* p = (next == SlabHashT::A_INDEX_POINTER)
                                  				? getPointerFromBucket(src_bucket, 31)
                                  				: getPointerFromSlab(next, 31);

          				uint32_t temp = atomicCAS(
              						(unsigned int*)p, SlabHashT::EMPTY_INDEX_POINTER, new_node_ptr);
          				// check whether it was successful, and
          				// free the allocated memory otherwise
          				if (temp != SlabHashT::EMPTY_INDEX_POINTER) {
            					freeSlab(new_node_ptr);
          				}
        			}		
      			} else {
        				next = next_ptr;
      				}
    		} else {  // there is a match or there is a empty spot available
			//there is a match
			if(isFound){
 				int dest_lane = __ffs(isFound & SlabHashT::REGULAR_NODE_KEY_MASK) - 1;
      				uint32_t found_value =
                			__shfl_sync(0xFFFFFFFF, src_unit_data, __ffs(isFound), 32);
      				if ((laneId == src_lane)&&(to_be_inserted==1)) {
        				uint32_t value = 0;
        				const uint32_t* p = (next == SlabHashT::A_INDEX_POINTER)
                        	        	? getPointerFromBucket(src_bucket, dest_lane)
                                		: getPointerFromSlab(next, dest_lane);

        				//printf("key :%d, lane_id: %d, old %d \n",wanted_key,laneId,old_key_value_pair);
        				uint32_t to_add = 1;
        				atomicAdd((unsigned long long int*)p,((uint64_t)(*reinterpret_cast<const uint32_t*>(
                           			reinterpret_cast<const unsigned char*>(&to_add)))
                       				<< 32)|0x0);
                			to_be_inserted = 0;
        			}
			}else{
      				int dest_lane = __ffs(isEmpty & SlabHashT::REGULAR_NODE_KEY_MASK) - 1;
      				if (laneId == src_lane) {
        				const uint32_t* p = (next == SlabHashT::A_INDEX_POINTER)
                                		? getPointerFromBucket(src_bucket, dest_lane)
                                		: getPointerFromSlab(next, dest_lane);
					uint32_t myValue = 1;
        			old_key_value_pair =
            				atomicCAS((unsigned long long int*)p, EMPTY_PAIR_64,
                      			((uint64_t)(*reinterpret_cast<const uint32_t*>(
                           		reinterpret_cast<const unsigned char*>(&myValue)))
                       			<< 32) |
                          		*reinterpret_cast<const uint32_t*>(
                              		reinterpret_cast<const unsigned char*>(&myKey)));
        			if (old_key_value_pair == EMPTY_PAIR_64)
          				to_be_inserted = 0;  // succesfful insertion
      				}
			}
		}
		last_work_queue = work_queue;
  	}
}
