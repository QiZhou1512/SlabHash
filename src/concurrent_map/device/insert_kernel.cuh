
template <typename KeyT, typename ValueT>
__global__ void insert_table_kernel(
	KeyT* d_key,
    	ValueT* d_value,
    	uint32_t num_vec,
    	int* num_kmers_read,
    	int totkmers,
    	int num_of_reads,
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
  	int to_insert = false;
  	uint32_t index_vector=0;
  	uint32_t index_array=0;
  	int num_char = 16;
  	uint32_t first_piece;
  	uint32_t second_piece;
  	int prev = 0;
  	int post = 0;
  	int div = 0;
  	int start_index = 0;

	if(tid<totkmers){
    	for(int i=0;i<num_of_reads;i++){
        	post += num_kmers_read[i];
        	if(tid>=prev&&tid<post){
                	//calculate index of the block : tid is the id of the kmer, by subtracting prev it just start from zero
                	index_vector = start_index + (tid-prev)/16;
                	index_array = (tid-prev)%16;
                	myKey = ((d_key[index_vector]<<(2*index_array))|
                        	(index_array!=0
                                        ?(d_key[index_vector+1]>>(2*(16-index_array)))
                                        :0x0));
                	myBucket = slab_hash.computeBucket(myKey);
                	to_insert = 1;

                	break;
        	}
        //calculate where the next block starts
        	div = num_kmers_read[i]+num_char-1;
        	start_index += div/num_char + (div % num_char != 0);
        	prev+=num_kmers_read[i];
  	}
  	start_index = 0;
  	prev = 0;
  	post = 0;
 }
/*
  for(int i=0;i<num_of_reads;i++){
	post = num_kmers_read[i];
	//printf("%d\n",post);		
	if(tid>=prev&&tid<post){
		//calculate index of the block : tid is the id of the kmer, by subtracting prev it just start from zero
		index_vector = start_index + (tid-prev);
    		index_array = (tid-prev)%16;
    		myKey = ((d_key[index_vector]<<(2*index_array))|
                        (index_array!=0
                                        ?(d_key[index_vector+1]>>(2*(16-index_array)))
                                        :0x0));
		printf("tid : %d \n",tid);
		for (int j = 31; 0 <= j; j--) {
                        printf("%c", (myKey & (1 << j)) ? '1' : '0');
             	}
		printf("\n");
    		myBucket = slab_hash.computeBucket(myKey);
    		to_insert = 1;
		
		break;
	}
	//calculate where the next block starts
	div = num_kmers_read[i]+num_char-1;
        start_index +=  div/num_char + (div % num_char != 0);
	prev+=num_kmers_read[i];
  }
*/	

/*
  if (tid < totkmers) {
                 	
    index_vector = tid/16;
    index_array = tid%16;
    myKey = ((d_key[index_vector]<<(2*index_array))|
                        (index_array!=0
                                        ?(d_key[index_vector+1]>>(2*(16-index_array)))
                                        :0x0));
    myBucket = slab_hash.computeBucket(myKey);
	
    to_insert = 1;
  }*/
  __syncthreads();

  slab_hash.insertKeyUnique(to_insert, laneId, myKey, myBucket);
}
