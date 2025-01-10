/**

g++ codec.cpp test_codec.cpp  -g -std=c++17 &&   ./a.out
 */

#include "codec.h"

#include <set>
#include <cassert>
#include <sys/time.h>
#include <unistd.h>

double getmillisecs() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}

int main_xx() {
    uint64_t tab[] = {12351235, 49024902, 17781778, 36663666}; 
    // uint64_t tab[] = {1235, 4902, 1778, 3666}; 
    int n = 4;
    int nbits = 26;
    ANSState state; 

    compress(n, tab, state, nbits);

    printf("head=%ld\nstack=[", state.head);
    for(int i = 0; i < state.stack.size(); i++) {
        printf("%u ", state.stack[i]);
    }
    printf("]\n");

    std::vector<uint64_t> tab2(n); 
    decompress(state, n, tab2.data(), nbits); 

    printf("tab2=[");
    // for(int i = 0; i < n; i++) {
    //     printf("%u ", tab2[i]);
    // }
    printf("]\n");


    return 0;
}


int main() { 
    /*
    int n = 200; 
    int nbits = 9; 
*/

    int n = 65000; 
    int nbits = 20; 

    for (int seed = 0; seed < 10; seed++) {

        std::vector<uint64_t> data(n); 
        std::mt19937 mt(seed);

        std::set<uint64_t> seen; 

        assert(nbits < 32); 
        assert(n < (1 << nbits)); 
        for(int i = 0; i < n; i++) {
            uint64_t x;
            for(;;) {
                x = mt() & ((1 << nbits) - 1); 
                if (seen.count(x) == 0) {
                    break; 
                }
            }
            seen.insert(x); 
            data[i] = x; 
        }

        double t0 = getmillisecs(); 

        ANSState state; 
        compress(n, data.data(), state, nbits);

        double t1 = getmillisecs(); 

        size_t size = 8 + 4 * state.stack.size(); 
        std::vector<uint64_t> tab2(n); 
        decompress(state, n, tab2.data(), nbits); 

        double t2 = getmillisecs(); 

        printf("n=%d nbits=%d seed=%d encode %.3f ms decode %.3f ms size=%ld bytes (%.3f bit / id)\n", 
            n, nbits, seed, t1 - t0, t2 - t1, 
            size, size * 8 / float(n)); 

        std::set<uint64_t> seen2(tab2.begin(), tab2.end()); 
        assert(seen == seen2); 
    }

    return 0; 
}