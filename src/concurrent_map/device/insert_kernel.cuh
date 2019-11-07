
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
