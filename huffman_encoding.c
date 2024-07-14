#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define MAX_CHARS 128
#define MAX_CODE_LENGTH 32
#define TREE_SERIALIZED_SIZE (MAX_CHARS * (sizeof(unsigned long long int) + sizeof(char)) + 1)
#define INT_MAX 2147483647
 
// Define the structure for a node in the Huffman tree
typedef struct Node {
    unsigned char data;
    unsigned long long int frequency;
    struct Node *left;
    struct Node *right;
} Node;

// Define a structure to represent a Huffman code dictionary entry
typedef struct {
    unsigned char character;
    char* code;
} CodeEntry;

// Define a structure to represent encoded data
typedef struct {
    char* encoded_data;
    size_t total_bits;   // This will track the number of bits used
    size_t total_bytes;   // This will track the number of bytes used
} EncodedResult;

size_t total_allocated_mem = 0;

void print_used_memory(){
    printf("Used memory: %zu bytes (%.2f MB)\n", total_allocated_mem, total_allocated_mem / (double)(1024 * 1024));
}

void add_memeory_to_total(size_t new_mem){
    total_allocated_mem += new_mem;
}

Node* createNode(unsigned char data, unsigned long long int frequency){
    // Function to create a new node
    Node* newNode = (Node*)malloc(sizeof(Node));
    if(newNode==NULL){
        fprintf(stderr, "Memory allocation failed for a new node \n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    newNode -> data = data;
    newNode -> frequency = frequency; 
    newNode->left = newNode->right = NULL;
    return newNode;
}

Node* mergeNodes(Node* left, Node* right){
//Function to combine two nodes into one 
    Node* mergedNode = createNode('$', left->frequency + right->frequency);
    mergedNode -> left = left;
    mergedNode -> right = right;
    return mergedNode;
}

// Main function to build the Huffman tree
Node* buildHuffmanTree(unsigned long long int* frequency) {
    int queueSize = MAX_CHARS;
    Node** queue = (Node**)malloc(queueSize * sizeof(Node*));
    if (queue == NULL) {
        fprintf(stderr, "Memory allocation failed for the priority queue\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Populate the priority queue with leaf nodes
    int queueIndex = 0;
    int i;
    for (i = 0; i < MAX_CHARS; ++i) {
        if (frequency[i] != 0) {
            queue[queueIndex++] = createNode((unsigned char)i, frequency[i]);
        }
    }

    // Build the Huffman tree
    while (queueIndex > 1) {
        // Find the two nodes with the lowest frequencies
        int min1 = 0, min2 = 1;
        if (queue[min1]->frequency > queue[min2]->frequency) {
            min1 = 1;
            min2 = 0;
        }
        int i;
        for ( i = 2; i < queueIndex; ++i) {
            if (queue[i]->frequency < queue[min1]->frequency) {
                min2 = min1;
                min1 = i;
            } else if (queue[i]->frequency < queue[min2]->frequency) {
                min2 = i;
            }
        }

        // Merge the two nodes
        Node* mergedNode = mergeNodes(queue[min1], queue[min2]);

        // Remove the merged nodes (min1, min2) from the queue and add the new merged node
        queue[min1] = mergedNode;
        queue[min2] = queue[--queueIndex];
    }

    // The remaining node in the queue is the root of the Huffman tree
    Node* root = queue[0];
    free(queue);
    return root;
}

void printHuffmanCodes(Node* root, int code[], int top){
    if(root->left){
        code[top] = 0;
        printHuffmanCodes(root->left, code, top+1);
    }
    if (root->right) {
        code[top] = 1;
        printHuffmanCodes(root->right, code, top + 1);
    }
    if (!root->left && !root -> right){
        printf("'%c': ", root->data);
        int i ;
        for (i = 0; i < top; ++i) {
            printf("%d", code[i]);
        }
        printf("\n");
    }
}


int readDataFromFile(char* filename, char** data, size_t* length) {
    // Open file for reading 
    FILE* fp = fopen(filename, "r");
    if (fp == NULL) {
        // Error handling for file opening
        fprintf(stderr, "Error opening file: %s\n", filename);
        return -1;
    }
    // Move file pointer to the end to determine the file size
    fseek(fp, 0L, SEEK_END);
    *length = ftell(fp); //get file size 
    rewind(fp); // Reset file pointer to the beginning

    // Allocate memory for data buffer 
    *data = (char*)malloc((*length + 1) * sizeof(char));
    if (*data == NULL){
        fprintf(stderr, "Memory allocation for input data is failed, failed on data length:%zu\n", *length);
        fclose(fp); // Close the file before exiting
        return -1;
    }

    // Read file content into the data buffer
    size_t bytes_read = fread(*data, sizeof(char), *length, fp);
    if (bytes_read != *length) {
        fprintf(stderr, "Error reading file: %s\n", filename);
        fclose(fp);
        free(*data);
        return -1;
    }
    
    (*data)[*length] = '\0'; // Null-terminate the string 

    // Close file 
    fclose(fp);
    return 0;
}

void calculate_frequency(char *data, unsigned long long int *frequency, size_t length){
    size_t i;
    for (i = 0; i < length; i ++) {
        frequency[(unsigned char)data[i]] ++;
    }
}

void print_frequency(unsigned long long int *frequency) {
    printf("Frequency of characters:\n");
    int i;
    for (i = 0; i < MAX_CHARS; ++i) {
        if (frequency[i] != 0) {
            printf("'%c': %llu\n", (char)i, frequency[i]);
        }
    }
}

void serialize_tree(Node* root, char* serialized_tree, int* index) {
    if (root == NULL) return;

    // For internal nodes, write '0' followed by left and right subtrees
    if (root->left != NULL || root->right != NULL) {
        serialized_tree[(*index)++] = '0';
        serialize_tree(root->left, serialized_tree, index);
        serialize_tree(root->right, serialized_tree, index);
    } 
    // For leaf nodes, write '1' followed by the character
    else {
        serialized_tree[(*index)++] = '1';
        serialized_tree[(*index)++] = root->data;
    }
}
 
Node* deserialize_tree(char* serialized_tree, int* index) {
    if (serialized_tree[*index] == '\0' || *index < 0) {
        return NULL;
    }

    // If current character is '1', it's a leaf node
    if (serialized_tree[*index] == '1') {
        (*index)++;
        char data = serialized_tree[*index];
        (*index)++;
        return createNode(data, 0);
    }
    // If current character is '0', it's an internal node
    else if (serialized_tree[*index] == '0') {
        (*index)++;
        Node* left = deserialize_tree(serialized_tree, index);
        Node* right = deserialize_tree(serialized_tree, index);
        return mergeNodes(left, right);
    } else {
        fprintf(stderr, "Invalid serialized tree format\n");
        exit(1);
    }
} 

// Helper function to recursively traverse the Huffman tree and populate the dictionary
void buildCodeDictionary(Node* root, char* code, int depth, CodeEntry* dictionary, int* dictionary_index) {
    if (root == NULL) {
        return;
    }
    // If leaf node, add the character and its code to the dictionary
    if (root->left == NULL && root->right == NULL) {
        dictionary[*dictionary_index].character = root->data;
        dictionary[*dictionary_index].code = (char*)malloc((depth + 1) * sizeof(char));
        strncpy(dictionary[*dictionary_index].code, code, depth);
        dictionary[*dictionary_index].code[depth] = '\0'; // Null-terminate the code
        (*dictionary_index)++;
        return;
    }
    // Traverse left with '0'
    code[depth] = '0';
    buildCodeDictionary(root->left, code, depth + 1, dictionary, dictionary_index);
    // Traverse right with '1'
    code[depth] = '1';
    buildCodeDictionary(root->right, code, depth + 1, dictionary, dictionary_index);
}

// Function to build a dictionary of Huffman codes
CodeEntry* buildCodeDictionaryFromTree(Node* root, int* dictionary_size) {
    // Allocate memory for the dictionary
    CodeEntry* dictionary = (CodeEntry*)malloc(MAX_CHARS * sizeof(CodeEntry));
    int dictionary_index = 0; // Index to track the position in the dictionary
    char code[MAX_CODE_LENGTH]; // Temporary buffer for storing the Huffman code
    
    // Recursively traverse the Huffman tree to build the dictionary
    buildCodeDictionary(root, code, 0, dictionary, &dictionary_index);
    
    *dictionary_size = dictionary_index; // Update the size of the dictionary
    return dictionary;
}

// Function to free memory allocated for the code dictionary
void freeCodeDictionary(CodeEntry* dictionary, int dictionary_size) {
    int i;
    for (i = 0; i < dictionary_size; i++) {
        free(dictionary[i].code); // Free memory for each code
    }
    free(dictionary); // Free memory for the dictionary array
}

void print_byte_as_bits(unsigned char byte) {
    int i;
    for (i = 7; i >= 0; i--) {
        printf("%c", (byte & (1 << i)) ? '1' : '0');
    }
    printf(" ");
}

// Function to encode the data using Huffman codes
EncodedResult encodeData(char* local_buffer, size_t local_size, CodeEntry* dictionary, int dictionary_size) {

    size_t allocation_size = local_size;
    
    char* encoded_data = (char*) malloc(allocation_size * sizeof(char));
    if (encoded_data == NULL) {
        fprintf(stderr, "Memory allocation failed for encoded_data\n");
        fprintf(stderr, "for local size %zu, dictionary size %d\n", allocation_size, dictionary_size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t total_bits = 0;
    size_t encoded_index = 0; // Tracks the byte position in encoded_data
    unsigned char current_byte = 0; // Holds the current byte being constructed
    int bit_count = 0; // Tracks the number of bits filled in the current_byte

    // Precompute dictionary lookup
    char* lookup_table[256] = {NULL};
    int j;
    for (j = 0; j < dictionary_size; ++j) {
        lookup_table[(unsigned char)dictionary[j].character] = dictionary[j].code;
    }
    
    size_t i; 
    for (i = 0; i < local_size; ++i) {
        char current_char = local_buffer[i];

        // Find the corresponding Huffman code for current_char
        char* huffman_code = lookup_table[(unsigned char)current_char];

        // If no Huffman code found, handle error or ignore character
        if (huffman_code == NULL) {
            fprintf(stderr, "Character '%c' not found in dictionary.\n", current_char);
            continue; // Skip this character or handle error as needed
        }

        // Append Huffman code bits to encoded data
        int code_length = strlen(huffman_code);
        total_bits += code_length;

        int k;
        for (k= 0; k < code_length; ++k) {
            // Append bit to current_byte
            if (huffman_code[k] == '1') {
                current_byte |= (1 << (7 - bit_count));
            }

            bit_count++;
            if (bit_count == 8) {
                encoded_data[encoded_index++] = current_byte;
                current_byte = 0;
                bit_count = 0;
            }
        }
    }

    // If there are remaining bits in current_byte, flush it to encoded_data
    if (bit_count > 0) {
        encoded_data[encoded_index++] = current_byte;
    }

    EncodedResult result = {encoded_data, total_bits, encoded_index};
    return result;
}

void saveHuffmanEncodingToFile(const char* filename, char* out_buffer, size_t buffer_length) {
    // Save the gathered encoded data into a file with the new filename
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "Error opening file\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    fwrite(out_buffer, sizeof(char), buffer_length, file);
    fclose(file);

    printf("Gathered encoded data saved to: %s\n", filename);
}

void write_metadata(const char *filename, long long int *recvcounts_byte, long long int *recvcounts_bit, int num_processes, unsigned long long int *frequency) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        perror("Failed to open metadata file");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    fprintf(file, "data_lengths:\n");
    int i;
    for (i = 0; i < num_processes; i++) {
        fprintf(file, "%lld ", recvcounts_byte[i]);
    }
    fprintf(file, "\npadding_bits:\n");
    for (i = 0; i < num_processes; i++) {
        int padding_bits = recvcounts_byte[i] * 8 - recvcounts_bit[i];
        fprintf(file, "%d ", padding_bits);
    }
    fprintf(file, "\nfrequency_table:\n");
    for (i = 0; i < MAX_CHARS; i++) {
        if (frequency[i] > 0) {
            fprintf(file, "%d %llu\n", i, frequency[i]);
        }
    }

    fclose(file);
    printf("Metadata saved to: %s\n", filename);
}

int main(int argc, char** argv) {
    int rank, size;
    char *data = NULL;
    char *local_data = NULL;
    size_t local_data_length = 0;
    size_t length = 0;
    int dictionary_size = 0;
    unsigned long long int *frequency = NULL;
    unsigned long long int *local_frequency = NULL;
    long long int* recvcounts_bit = NULL;
    long long int* recvcounts_byte = NULL;
    char* gathered_encoded_data = NULL;
 
    Node* global_root = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int i, j;
    double runtimes[10]; // Array to store runtimes of different phases

    runtimes[0] = MPI_Wtime(); // Start timing

    // Check command line arguments [file, data]
    if (argc != 2) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    //############ 1. File read ############

    // Read data in rank 0
    if (rank == 0) {
        print_used_memory();

        // Read data from file
        if (readDataFromFile(argv[1], &data, &length) != 0){
            fprintf(stderr, "Error reading data from file %s on rank %d\n", argv[1], rank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        add_memeory_to_total(length);
        print_used_memory();

        printf("Size of data: %.2f MB\n", (double)length / (1024 * 1024));
        printf("Length of data: %zu \n", length);
        printf("Number of processes: %d \n", size); 
    }
    runtimes[1] = MPI_Wtime(); //File reading runtime 

    //############ 2. Data distribution ############ 

    // braodcast length to all processes 
    MPI_Bcast(&length, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

    // Synchronize all processes after file read
    MPI_Barrier(MPI_COMM_WORLD);

    // calculates total number of partitions to be sent
    int num_partitions = size;
    int num_send = 1; 
    if (length/size > INT_MAX) {
        num_send =  ((length / size) + INT_MAX - 1) / INT_MAX;
        num_partitions = size * num_send;
    }

    // keeps data size in each partition and corresponding offsets
    size_t partitionSizes[num_partitions];
    size_t partitionOffsets[num_partitions];
    
    // calculate partition sizes and offsets
    for (i = 0; i < num_partitions; ++i) {
        partitionSizes[i] = (length / num_partitions) + (i < length % num_partitions ? 1 : 0);
        partitionOffsets[i] = (i > 0 ? partitionOffsets[i - 1] + partitionSizes[i - 1] : 0);
    }
    if (rank == 0){
        printf("num_partitions: %d, split: %d\n", num_partitions, num_send);
    }

    // calculat total size for local data allocation
    for (i = rank * num_send; i < (rank+1) * num_send; ++i) {
        local_data_length += partitionSizes[i];
    }

    // allocate memory for local data 
    local_data = (char *)malloc(local_data_length * sizeof(char)); 
    if (local_data == NULL) {
        fprintf(stderr, "Memory allocation for local_data_buffer failed in rank: %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    if (rank == 0){
        add_memeory_to_total((local_data_length) * sizeof(char));
        print_used_memory();
    }

    // Data sending/receiving. Root process sends to other processes, other processes receives from the root 
    if (rank == 0){
        MPI_Request requests[(size - 1) * num_send];
        int request_count = 0;

        // Send partitions to other processes
        int dest;
        for(dest = 1; dest<size; ++dest){
            for(i = dest*num_send; i<(dest+1) * num_send; ++i){
                MPI_Isend(data + partitionOffsets[i], partitionSizes[i], MPI_CHAR, dest, 0, MPI_COMM_WORLD, &requests[request_count]);
                request_count++;
            }
        }

        // Copy its own partitions to local_data
        size_t offset = 0; // for local writer offset 
        for(i = 0; i<num_send; ++i){
            memcpy(local_data + offset, data + partitionOffsets[i], partitionSizes[i]*sizeof(char));
            offset += partitionSizes[i];
        }

        // Wait for all non-blocking sends to complete
        MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);
        printf("Root process scattered all data.\n");

    }else{
        // Other processes receive data from root
        MPI_Request requests[num_send]; // each process receives num_send times
        size_t offset = 0; // for local writer offset 
        for (i = rank * num_send; i < (rank + 1) * num_send; ++i) {
            MPI_Irecv(local_data + offset, partitionSizes[i], MPI_CHAR, 0, 0, MPI_COMM_WORLD, &requests[i - rank * num_send]);
            offset += partitionSizes[i];
        }
        // Wait for all non-blocking receives to complete
        MPI_Waitall(num_send, requests, MPI_STATUSES_IGNORE);
    }
    
    local_data[local_data_length] = '\0'; // Ensure local_data is null-terminated

    // Assertion -  to ensure scatter worked as expected
    if (strlen(local_data) != local_data_length) {
        fprintf(stderr, "Error: Scatter data length mismatch\n");
        fprintf(stderr, "Local data: %s\n", local_data);
        fprintf(stderr, "Expected length: %zu, Actual length: %zu\n", local_data_length, strlen(local_data));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    runtimes[2] = MPI_Wtime(); //Data distribution runtime
    

    //############ 3. Frequency calculation ############ 

    // Allocate memory for local frequency in all processes
    local_frequency = (unsigned long long int*) calloc(MAX_CHARS, sizeof(unsigned long long int));
    if (local_frequency == NULL) {
        fprintf(stderr, "Memory allocation failed for local_frequency on Rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Allocate memory for frequency in root 
    if (rank == 0){
        frequency = (unsigned long long int*) calloc(MAX_CHARS, sizeof(unsigned long long int));
        add_memeory_to_total(MAX_CHARS * sizeof(unsigned long long int));
        add_memeory_to_total(MAX_CHARS * sizeof(unsigned long long int));
        print_used_memory();
    }

    // Calculate frequency in each process
    calculate_frequency(local_data, local_frequency, local_data_length);
 
    //Gather local frequencies from all processes at root process
    MPI_Reduce(local_frequency, frequency, MAX_CHARS, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // Synchronize all processes after frequency calculation
    MPI_Barrier(MPI_COMM_WORLD);

    runtimes[3] = MPI_Wtime(); //Counting freq runtime


    //############ 4. Huffman Tree construction ############ 

    // Root process builds Huffman tree and broadcasts it
    if (rank == 0) {
        // print_frequency(frequency); // for debug

        // Build Huffman tree
        global_root = buildHuffmanTree(frequency);

        // Serialize the Huffman tree
        char serialized_tree[TREE_SERIALIZED_SIZE]; // Assuming a maximum size for the serialized tree
        int index = 0; // Start index in the serialized array
        serialize_tree(global_root, serialized_tree, &index);
        serialized_tree[index] = '\0'; // Null-terminate the serialized string
 
        // Broadcast the serialized tree to all processes
        MPI_Bcast(serialized_tree, TREE_SERIALIZED_SIZE, MPI_CHAR, 0, MPI_COMM_WORLD);
    } else {
        // Receive the serialized tree from the root process
        char serialized_tree[TREE_SERIALIZED_SIZE];
        MPI_Bcast(serialized_tree, TREE_SERIALIZED_SIZE, MPI_CHAR, 0, MPI_COMM_WORLD);
        
        // Deserialize the tree
        int deserialization_index = 0;
        global_root = deserialize_tree(serialized_tree, &deserialization_index);
    }

    // Synchronize all processes after broadcasting the Huffman tree
    MPI_Barrier(MPI_COMM_WORLD);

    runtimes[4] = MPI_Wtime(); //Huffman tree build
    

    //############ 5. Huffman code generation ############

    // Build the code dictionary from the Huffman tree
    CodeEntry* dictionary = buildCodeDictionaryFromTree(global_root, &dictionary_size);
    
    // timing info
    runtimes[5] = MPI_Wtime(); //Huffman set code


    //############ 6. Data encoding ############

    // each process encodes own local data usung huffman codes
    EncodedResult encoded_result = encodeData(local_data, local_data_length, dictionary, dictionary_size);
   
    // for debugging
    // printf("~rank: %d, local length: %zu, local bytes: %zu, encoded data len: %zu, encoded: total_bits : %zu\n", rank, strlen(local_data), local_data_length, encoded_result.total_bytes, encoded_result.total_bits);


    runtimes[6] = MPI_Wtime(); //Local data encoding
    if (rank == 0){
        add_memeory_to_total(local_data_length * sizeof(char));
        print_used_memory();
    }

    //############ 7. Encoded data gathering ############

    size_t total_bytes = 0;
    if (rank == 0) {
        recvcounts_bit = (long long int*) malloc(size * sizeof(long long int));
        recvcounts_byte = (long long int*) malloc(size * sizeof(long long int));
        if (recvcounts_bit == NULL || recvcounts_byte == NULL) {
            fprintf(stderr, "Memory allocation for recvcounts_bit or recvcounts_byte failed\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }
    MPI_Gather(&encoded_result.total_bits, 1, MPI_LONG_LONG, recvcounts_bit, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    

    if (rank == 0) {
        // Calculate total bytes required and allocate memory 
        for (i = 0; i < size; ++i) {
            recvcounts_byte[i] = (recvcounts_bit[i] + 7) / 8; // Round up to the nearest byte
            total_bytes += recvcounts_byte[i];
        }

        printf("Total_bytes: %zu\n", total_bytes); 

        gathered_encoded_data = (char*) malloc(total_bytes * sizeof(char));
        if (gathered_encoded_data == NULL) {
            fprintf(stderr, "Memory allocation for gathered_encoded_data failed\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        add_memeory_to_total(total_bytes * sizeof(char));
        print_used_memory();

        // determine partitions for receiving encoded data
        int partition_cnt[size]; 
        num_partitions = 0;
        for(i = 0; i< size; ++i){
            int k = (recvcounts_byte[i] + INT_MAX - 1) / INT_MAX;
            num_partitions += k;
            partition_cnt[i] = k; 
        }
        // Determine sizes and offsets for each partition
        size_t partition_sizes[num_partitions];
        size_t partition_offsets[num_partitions];

        int partition_index=0;
        size_t offset = 0; 
        for (i = 0; i<size; ++i){
            size_t remaining = recvcounts_byte[i];
            while(remaining > 0){
                int chunk_size = (remaining > INT_MAX) ? INT_MAX : remaining;
                partition_sizes[partition_index] = chunk_size;
                partition_offsets[partition_index] = offset;
                remaining -= chunk_size;
                offset += chunk_size;
                partition_index++;
            }
        }

        // Initiate non-blocking receives for encoded data
        partition_index = partition_cnt[0];
        MPI_Request recv_requests[num_partitions - partition_cnt[0]];
        offset = 0;
        int recv_request_count = 0;
        for (i = 1; i < size; ++i){
            for (j = 0; j<partition_cnt[i]; ++j){
                printf("----recv %i : %zu, %zu\n", i, partition_sizes[partition_index], partition_offsets[partition_index]);
                MPI_Irecv(gathered_encoded_data + partition_offsets[partition_index], partition_sizes[partition_index], MPI_CHAR, i, 0, MPI_COMM_WORLD, &recv_requests[recv_request_count]);
                
                partition_index ++; 
                recv_request_count++;
            }
        }

        // root process copies its local encoded data
        offset = 0; // for local writer offset 
        for(i = 0; i<partition_cnt[0]; ++i){
            memcpy(gathered_encoded_data + offset, encoded_result.encoded_data + partition_offsets[i], partition_sizes[i]*sizeof(char));
            offset += partition_sizes[i];
           
            printf("----send %d : %zu, %zu\n", rank, partition_sizes[i], partition_offsets[i]);
            printf("----recv %d : %zu, %zu\n", rank, partition_sizes[i], partition_offsets[i]);
        }

        // Wait for all non-blocking receives to complete
        MPI_Waitall(num_partitions - partition_cnt[0], recv_requests, MPI_STATUSES_IGNORE);

        printf("Root process gathered all encoded data.\n");
   
    } else{
        // calculate number of partitions for sending local encoded data 
        num_send = (encoded_result.total_bytes + INT_MAX - 1) / INT_MAX;

        // allocate arrays for local partition sizes and offsets
        int local_partition_sizes[num_send];
        size_t local_partition_offsets[num_send];

        // Compute sizes and offsets for each partition
        size_t offset = 0;
        size_t remaining = encoded_result.total_bytes;
        for (i = 0; i < num_send; ++i) {
            // Determine chunk size for current partition
            int chunk_size = (remaining > INT_MAX) ? INT_MAX : (int)remaining;

            // Store chunk size and offset
            local_partition_sizes[i] = chunk_size;
            local_partition_offsets[i] = offset;

            // Update offset and remaining bytes
            offset += chunk_size;
            remaining -= chunk_size;
        }

        // Initiate non-blocking sends of local encoded data to root process 
        MPI_Request send_requests[num_send];
        for (i = 0; i< num_send; i++){
            printf("----send %d, %d, %zu\n", rank, local_partition_sizes[i], local_partition_offsets[i]);
            MPI_Isend(encoded_result.encoded_data + local_partition_offsets[i], local_partition_sizes[i], MPI_CHAR,
                  0, 0, MPI_COMM_WORLD, &send_requests[i]);
        }

        // Wait for all non-blocking sends to complete
        MPI_Waitall(num_send, send_requests, MPI_STATUSES_IGNORE);
    }

    runtimes[7] = MPI_Wtime(); //Encoded data gathering
    

    //############ 8. File write  ############
    if (rank == 0) {
 
        // Calculate memory size
        printf("Memory allocated for final_sequence: %zu bytes (%.2f MB)\n", total_bytes * sizeof(char), total_bytes * sizeof(char) / (double)(1024 * 1024));
        printf("Compression ratio: %f\n",(float)total_bytes/length);
        printf("Compresison sizes: %zu, %zu\n", total_bytes, length);
        // Save Huffman encoding to file
        saveHuffmanEncodingToFile( "huffman_compressed.txt", gathered_encoded_data, total_bytes);
        // Write metadata to file 
        write_metadata("huffman_compressed_metadata.txt", recvcounts_byte, recvcounts_bit, size, frequency);

        runtimes[8] = MPI_Wtime(); //Writing file

        // Cleanup allocated memory
        free(data);
        free(frequency);
        free(recvcounts_bit);
        free(recvcounts_byte);
        free(gathered_encoded_data);
    }
 
    // Cleanup allocated memory 
    freeCodeDictionary(dictionary, dictionary_size); // Free memory for code dictionary
    free(local_data);
    free(local_frequency);
 
    // Final timing information output by rank 0
    if (rank == 0) {
        runtimes[9] = MPI_Wtime(); //end time
        
        printf("--------------------------\n");
        printf("1. Read file: %.4f seconds\n", runtimes[1] - runtimes[0]);
        printf("2. Data distribution: %.4f seconds\n", runtimes[2] - runtimes[1]);
        printf("3. Calculate freq: %.4f seconds\n", runtimes[3] - runtimes[2]);
        printf("4. Tree build: %.4f seconds\n", runtimes[4] - runtimes[3]);
        printf("5. Set Huffman codes: %.4f seconds\n", runtimes[5] - runtimes[4]);
        printf("6. Local encoding: %.4f seconds\n", runtimes[6] - runtimes[5]);
        printf("7. Encoded data gathering: %.4f seconds\n", runtimes[7] - runtimes[6]);
        printf("8. Write file: %.4f seconds\n", runtimes[8] - runtimes[7]);
        printf("---Total time w.o I/O: %.4f seconds\n", runtimes[8] - runtimes[1]);
        printf("---Total time: %.4f seconds\n", runtimes[9] - runtimes[0]);
        printf("\n\n");
    }

    MPI_Finalize();
    return 0;
}
 
