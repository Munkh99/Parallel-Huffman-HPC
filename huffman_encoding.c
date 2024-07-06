#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "huffman_encoding.h"

#define MAX_CHARS 128
#define MAX_CODE_LENGTH 32
#define TREE_SERIALIZED_SIZE (MAX_CHARS * (sizeof(unsigned long long int) + sizeof(char)) + 1)
 
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
    long long int total_bits; // This will track the number of bits used
} EncodedResult;

size_t total_allocated_mem = 0;

void print_used_memory(){
    printf("Used memory: %zu bytes (%.2f MB)\n", total_allocated_mem, total_allocated_mem / (double)(1024 * 1024));
}

void add_memeory_to_total(int new_mem){
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

Node* buildHuffmanTree(unsigned long long int* frequency) {
    //Function to build the Huffman tree
    // Create a priority queue of nodes
    int queueSize = MAX_CHARS;
    Node** queue = (Node**)malloc(queueSize * sizeof(Node*));
    if (queue == NULL) {
        fprintf(stderr, "Memory allocation failed for the priority queue\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
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


int readDataFromFile(char* filename, char** data, int* length) {
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
        fprintf(stderr, "Memory allocation for input data is failed, failed on data length:%d\n", *length);
        fclose(fp); // Close the file before exiting
        return -1;
    }

    // Read file content into the data buffer 
    fread(*data, sizeof(char), *length, fp);
    (*data)[*length] = '\0'; // Null-terminate the string 

    // Close file 
    fclose(fp);
    return 0;
}

void calculate_frequency(char *data, unsigned long long int *frequency, int length){
    int i;
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


EncodedResult encodeDataUsingDictionary(char* local_buffer, int local_size, CodeEntry* dictionary, int dictionary_size) {

    size_t allocation_size = local_size;
    // printf("Initial allocation size: %zu bytes\n", allocation_size);
    
    char* encoded_data = (char*) malloc(allocation_size * sizeof(char));
    if (encoded_data == NULL) {
        fprintf(stderr, "Memory allocation failed for encoded_data\n");
        fprintf(stderr, "for local size %d, dictionary size %d\n", local_size, dictionary_size);
        EncodedResult result = {NULL, 0};
        return result;
    }

    long long int total_bits = 0;
    int encoded_index = 0; // Tracks the byte position in encoded_data
    unsigned char current_byte = 0; // Holds the current byte being constructed
    int bit_count = 0; // Tracks the number of bits filled in the current_byte
    int i;

    // Precompute dictionary lookup
    char* lookup_table[256] = {NULL};
    int j;
    for (j = 0; j < dictionary_size; ++j) {
        lookup_table[(unsigned char)dictionary[j].character] = dictionary[j].code;
    }
  
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

    EncodedResult result = {encoded_data, total_bits};
    return result;
}

void saveHuffmanEncodingToFile(const char* filename, char* out_buffer, int buffer_length) {
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

void write_metadata(const char *filename, int *recvcounts_byte, long long int *recvcounts_bit, int num_processes, unsigned long long int *frequency) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        perror("Failed to open metadata file");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    fprintf(file, "data_lengths:\n");
    int i;
    for (i = 0; i < num_processes; i++) {
        fprintf(file, "%d ", recvcounts_byte[i]);
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
}

int main(int argc, char** argv) {
    int rank, size;
    char *data = NULL;
    char *local_data_buffer = NULL;
    int local_data_buffer_length = 0;
    int length = 0;
    int dictionary_size = 0;
    unsigned long long int *frequency = NULL;
    unsigned long long int *local_frequency = NULL;
    long long int* recvcounts_bit = NULL;
    int* recvcounts_byte = NULL;
    char* gathered_encoded_data = NULL;

    int *send_counts = NULL;
    int *displs = NULL;
    Node* global_root = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int i;
    double runtimes[10]; // Array to store runtimes of different phases

    runtimes[0] = MPI_Wtime(); // Start timing
    // Check command line arguments [file, data]
    if (argc != 2) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    // Read data into 'data' variable only on rank 0
    if (rank == 0) {
        print_used_memory();

        // Read data from file
        if (readDataFromFile(argv[1], &data, &length) != 0){
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        double size_mb = (double)length / (1024 * 1024);
        add_memeory_to_total(length);
        print_used_memory();

        printf("Size of data: %.2f MB\n", size_mb);
        printf("Length of data: %d \n", length);
        printf("Number of processes: %d \n", size); 

        runtimes[1] = MPI_Wtime(); //File reading runtime
    }

    // Allocate memory for frequency in all processes
    frequency = (unsigned long long int*) calloc(MAX_CHARS, sizeof(unsigned long long int));
    local_frequency = (unsigned long long int*) calloc(MAX_CHARS, sizeof(unsigned long long int));

    if (rank == 0){
        add_memeory_to_total(MAX_CHARS * sizeof(unsigned long long int));
        add_memeory_to_total(MAX_CHARS * sizeof(unsigned long long int));
        print_used_memory();

        int remainder = length % size; // Remainder for handling the case when length is not divisible by size
        int base_size = length / size;

        send_counts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        
        if (send_counts == NULL || displs == NULL) {
            fprintf(stderr, "Memory allocation for send_counts or displs failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        add_memeory_to_total(size * sizeof(int));
        add_memeory_to_total(size * sizeof(int));
        // print_used_memory();

        // Calculate send counts and displacements for scattering the data    
        for (i = 0; i < size; ++i) {
            send_counts[i] = base_size + (i < remainder ? 1 : 0);
            displs[i] = (i == 0) ? 0 : displs[i - 1] + send_counts[i - 1];
        }
       
    }
    // Scatter send_counts to all processes - corresponding data lengths to be received
    MPI_Scatter(send_counts, 1, MPI_INT, &local_data_buffer_length, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory for local data buffer
    local_data_buffer = (char *)malloc((local_data_buffer_length + 1) * sizeof(char)); // +1 for null terminator
    if (local_data_buffer == NULL) {
        fprintf(stderr, "Memory allocation for local_data_buffer failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0){
        add_memeory_to_total((local_data_buffer_length + 1) * sizeof(char));
        print_used_memory();
    }
    
    // Scatter data to all processes
    MPI_Scatterv(data, send_counts, displs, MPI_CHAR, local_data_buffer, local_data_buffer_length, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Null terminate the local data
    local_data_buffer[local_data_buffer_length] = '\0';

    // Debugging check to ensure scatter worked as expected
    if (strlen(local_data_buffer) != local_data_buffer_length) {
        fprintf(stderr, "Error: Scatter data length mismatch\n");
        fprintf(stderr, "Local data: %s\n", local_data_buffer);
        fprintf(stderr, "Expected length: %d, Actual length: %zu\n", local_data_buffer_length, strlen(local_data_buffer));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    
    // Calculate frequency in each process
    calculate_frequency(local_data_buffer, local_frequency, local_data_buffer_length);
 
    //Gather local frequencies from all processes
    MPI_Reduce(local_frequency, frequency, MAX_CHARS, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    
    // Print timing info
    if (rank == 0) {
        runtimes[2] = MPI_Wtime(); //Counting freq runtime
    }

    // Root process (rank 0) builds Huffman tree and broadcasts it
    if (rank == 0) {
        // print_frequency(frequency); // for debug

        // Build Huffman tree
        global_root = buildHuffmanTree(frequency);
        runtimes[3] = MPI_Wtime(); //Huffman tree build

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
    
    // Build the code dictionary from the Huffman tree
    
    CodeEntry* dictionary = buildCodeDictionaryFromTree(global_root, &dictionary_size);
    
    // Print timing info
    if (rank == 0) {
        runtimes[4] = MPI_Wtime(); //Huffman tree build
        
    }
    // for debug, printing dictionary 
    // if (rank == 0) {
    //     for (i = 0; i < dictionary_size; i++) {
    //         printf("Character: '%c', Code: %s\n", dictionary[i].character, dictionary[i].code);
    //     }
    // }

    EncodedResult encoded_result = encodeDataUsingDictionary(local_data_buffer, local_data_buffer_length, dictionary, dictionary_size);
    
    if (rank == 0){
        runtimes[5] = MPI_Wtime(); //Encoding 1 - local data encoded
        add_memeory_to_total(local_data_buffer_length * sizeof(char));
        print_used_memory();
    }
    
    if (rank == 0) {
        recvcounts_bit = (long long int*) malloc(size * sizeof(long long int));
        recvcounts_byte = (int*) malloc(size * sizeof(int));
        if (recvcounts_bit == NULL || recvcounts_byte == NULL) {
            fprintf(stderr, "Memory allocation for recvcounts_bit or recvcounts_byte failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
    }

    MPI_Gather(&encoded_result.total_bits, 1, MPI_LONG_LONG, recvcounts_bit, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    double total_bytes = 0;
    long long int total_bits = 0;

    if (rank == 0) {
        total_bits = 0;
        for (i = 0; i < size; ++i) {
            recvcounts_byte[i] = (recvcounts_bit[i] + 7) / 8; // Round up to the nearest byte
            total_bits += recvcounts_bit[i]; 
        }

        total_bytes = (total_bits + 7) / 8;
        // printf("Total encoded data size: %f bytes, %f bits \n", total_bytes, total_bits);
        
        displs[0] = 0; // Should store the cumulative byte offsets where each process's data starts in gathered_encoded_data
        for (i = 1; i < size; ++i) {
            displs[i] = displs[i - 1] + recvcounts_byte[i-1]; // Calculate displacements in bytes
        }
        gathered_encoded_data = (char*) malloc(total_bytes * sizeof(char));
        if (gathered_encoded_data == NULL) {
        fprintf(stderr, "Memory allocation for gathered_encoded_data failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    }
    
    int encoded_size_byte = (encoded_result.total_bits + 7) / 8;
    // Gather encoded data from all processes to rank 0
    MPI_Gatherv(encoded_result.encoded_data, encoded_size_byte, MPI_CHAR, gathered_encoded_data, recvcounts_byte, displs, MPI_CHAR, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        runtimes[6] = MPI_Wtime(); //Encoding 2 - gathered
       
        // Calculate memory size
        size_t final_bit_stream_size = total_bytes * sizeof(char);
        printf("Memory allocated for final_sequence: %zu bytes (%.2f MB)\n", final_bit_stream_size, final_bit_stream_size / (double)(1024 * 1024));

        // Save Huffman encoding to file
        saveHuffmanEncodingToFile( "huffman_compressed.txt", gathered_encoded_data, total_bytes);
        write_metadata("huffman_compressed_metadata.txt", recvcounts_byte, recvcounts_bit, size, frequency);

        runtimes[7] = MPI_Wtime(); //Writing file

        free(gathered_encoded_data);
        free(recvcounts_bit);
        free(recvcounts_byte);
        free(displs);
        free(send_counts);
    }
 
    // Cleanup allocated memory 
    freeCodeDictionary(dictionary, dictionary_size); // Free memory for code dictionary
    free(local_data_buffer);
    free(local_frequency);
    free(frequency);
    
  
    // Final timing info
    if (rank == 0) {
        runtimes[8] = MPI_Wtime(); //end time
        
        printf("--------------------------\n");
        printf("1. Read file: %.4f seconds\n", runtimes[1] - runtimes[0]);
        printf("2. Calculate freq: %.4f seconds\n", runtimes[2] - runtimes[1]);
        printf("3. Tree build: %.4f seconds\n", runtimes[3] - runtimes[2]);
        printf("4. set Huffman codes: %.4f seconds\n", runtimes[4] - runtimes[3]);
        printf("5. Encode: %.4f seconds\n", runtimes[6] - runtimes[4]);
        printf("6. Write file: %.4f seconds\n", runtimes[7] - runtimes[6]);
        printf("---Total time w.o I/O: %.4f seconds\n", runtimes[6] - runtimes[1]);
        printf("---Total time: %.4f seconds\n", runtimes[8] - runtimes[0]);
        printf("\n\n");
    }

    MPI_Finalize();
    return 0;
}
 
