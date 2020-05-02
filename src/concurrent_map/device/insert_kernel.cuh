
template <typename KeyT, typename ValueT>
__global__ void insert_table_kernel(
	KeyT* d_key,
	ValueT* d_index,
    	uint32_t totkmers,
    	GpuSlabHashContext<KeyT, ValueT, SlabHashTypeT::ConcurrentMap> slab_hash) {
  	uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  	uint32_t laneId = threadIdx.x & 0x1F;	
  	if ((tid - laneId) >= totkmers) {
    		return;
  	}
  	// initializing the memory allocator on each warp:
  	slab_hash.getAllocatorContext().initAllocator(tid, laneId);

  	KeyT myKey = 0;
  	uint32_t myBucket = 0;
  	int to_insert = 0;
	int num_div = 4;
	/*if(tid==0){
		printf("tid 0\n");
		printf("totkmers %"PRIu32"\n",totkmers);
		printf("d_key : %"PRIu32"\n",d_key[tid]);
		for(int i =0 ; i<totkmers; i++){
                        for (int k = 31; 0 <= k; k--) {
                                printf("%c", (d_key[i] & (1 << k)) ? '1' : '0');
                        }
                        printf("\n");
		}
	}*/
	if(tid<totkmers){
			
			myKey = d_key[tid];
                	myBucket = slab_hash.computeBucketindex(myKey,d_index[tid], num_div);
	       //         myBucket = slab_hash.computeBucket(myKey);
			to_insert = 1;
 	}
  	//__syncthreads();

  	slab_hash.insertKeyUnique(to_insert, laneId, myKey, myBucket);
}

template <typename KeyT, typename ValueT>
__global__ void insert_table_kernel_on_reads(
        KeyT* d_key_blocks,
        ValueT* d_whitelist_blocks,
        uint64_t tot_whitelist_blocks,
        GpuSlabHashContext<KeyT, ValueT, SlabHashTypeT::ConcurrentMap> slab_hash) {
        uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
        uint32_t laneId = threadIdx.x & 0x1F;
        if ((tid - laneId) >= 32*tot_whitelist_blocks) {
                return;
        }
        // initializing the memory allocator on each warp:
        slab_hash.getAllocatorContext().initAllocator(tid, laneId);

        KeyT myKey = 0;
        uint32_t myBucket = 0;
        int to_insert = 0;
        int num_div = 4;
	uint32_t block_index = 0;
        uint32_t bit_index = 0;
        int offset = 31;
        block_index = tid/32;
        bit_index = (tid%32);
/*	
	if(tid == 0){
                for(int l = 0; l<4; l++){
                        for (int k = 31; 0 <= k; k--) {
                                printf("%c", (d_key_blocks[l] & (1 << k)) ? '1' : '0');
                        }
                        printf("\n");
                }
                for(int j = 0; j<2; j++){
                        for (int k = 31; 0 <= k; k--) {
                                printf("%c", (d_whitelist_blocks[j] & (1 << k)) ? '1' : '0');
                        }
                        printf("\n");
                }
                for(int i = 0; i<31; i++){
                        block_index = i/16;
                        bit_index = (i%16)*2;
                        myKey = (d_key_blocks[block_index] << bit_index)|
                                (bit_index!=0
                                                ?(d_key_blocks[block_index+1]>>(32-bit_index))
                                                :0x0);
                        printf("tid : %d\n",i);
                        for (int k = 31; 0 <= k; k--) {
                                printf("%c", (myKey & (1 << k)) ? '1' : '0');
                        }
                        printf("\n");
                }
                
        }

*/
/*
	if(((uint32_t) (d_whitelist_blocks[tid/32]>>(31-tid%32))&(0x00000001))==0x0){
        	block_index = tid/32;
                bit_index = tid%32;
                return;
        }
*/
	
        if(tid<32*tot_whitelist_blocks && (((uint32_t) (d_whitelist_blocks[tid/32]>>(31-tid%32))&(0x00000001))!=0x0)){

        	
		block_index = tid/16;
                bit_index = (tid%16);

                myKey = (uint32_t)(d_key_blocks[block_index] << (bit_index)*2)|
                	(bit_index!=0
                                     ?(d_key_blocks[block_index+1]>>(32-(bit_index)*2))
                                     :0x0);
                //myBucket = slab_hash.computeBucketindex(myKey,0x1, num_div);
		to_insert = 1;
		if(myKey == 0xFFFFFFFF){
			to_insert = 0;
		}
		myBucket = slab_hash.computeBucket(myKey);
		//printf("%" PRIu32 " tid: %d block: %d bit: %d\n",myKey, tid, block_index,bit_index);
		if(myKey == 0){
			printf("%" PRIu32 "\n",myKey);
		}
 
	}
       // __syncthreads();
//	printf("start insert tid: %d\n",tid);
        slab_hash.insertKeyUnique(to_insert, laneId, myKey, myBucket);
//	printf("end insert tid: %d\n",tid);
}
