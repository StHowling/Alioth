#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>
#include <unistd.h>

#define CACHE_LINE_SIZE 64
#define DEFAULT_BLOCK_SIZE (64) //262144
#define QUIET 0

/* allocate a test array and fill it with data
 * so as to force Linux to _really_ allocate it */
long *make_array(unsigned long long asize) {
    unsigned long long t;
    unsigned int long_size=sizeof(long);
    long *a;

    a=calloc(asize, long_size);

    if(NULL==a) {
        perror("Error allocating memory");
        exit(1);
    }

    /* make sure both arrays are allocated, fill with pattern */
    for(t=0; t<asize; t++) {
        a[t]=0xaa;
    }
    return a;
}

double worker(unsigned long long asize, long *a, long *b, unsigned long long block_size) {
    unsigned long long t;
    struct timeval starttime, endtime;
    double te;
    unsigned int long_size=sizeof(long);
    /* array size in bytes */
    unsigned long long array_bytes=asize*long_size;

    char* aa = (char*)a;
    char* bb = (char*)b;
    gettimeofday(&starttime, NULL);
    for (t=array_bytes; t >= block_size; t-=block_size, aa+=block_size){
        bb=mempcpy(bb, aa, block_size);
    }
    if(t) {
        bb=mempcpy(bb, aa, t);
    }
    gettimeofday(&endtime, NULL);


    te=((double)(endtime.tv_sec*1000000-starttime.tv_sec*1000000+endtime.tv_usec-starttime.tv_usec))/1000000;

    return te;
}

void printout(double te, double mt) {
    printf("Method: MCBLOCK\t");
    printf("Elapsed: %.5f\t", te);
    printf("MiB: %.5f\t", mt);
    printf("Copy: %.3f MiB/s\n", mt/te);
    return;
}

int main(int argc, char **argv) {
    long *a, *b; // occupied buffer
    double te, te_sum;
    unsigned long long block_size=DEFAULT_BLOCK_SIZE;
    unsigned int long_size=sizeof(long); /* the size of long on this platform */
    unsigned long long asize;
    double mt=0.;
    int i;

    if(argc >= 2) {
        mt=strtoul(argv[1], (char **)NULL, 10);
    } else {
        printf("Error: no array size given!\n");
        exit(1);
    }

    asize = 1024*1024/long_size*mt; /* how many longs then in one array? */

    if(asize*long_size < block_size) {
        printf("Error: array size larger than block size (%llu bytes)!\n", block_size);
        exit(1);
    }

    if(!QUIET) {
        printf("Long uses %d bytes. ", long_size);
        printf("Allocating 2*%lld elements = %lld bytes of memory.\n", asize, 2*asize*long_size);
        printf("Using %lld bytes as blocks for memcpy block copy test.\n", block_size);
    }

    a=make_array(asize);
    b=make_array(asize);

    for (i=0; 1; i++) {
        te=worker(asize, a, b, block_size);
        printf("%d\t", i);
        if(!QUIET) {
            printout(te, mt);
        }
    }
    return 0;
}
