// Exact retrograde solver for Sugar.
//
// Build and run:  gcc -O2 -o solver/solve solver/solve.c && ./solver/solve solver/solution.bin
// (about 80 s and 3.5 GB of RAM on a laptop).
//
// The opening forward-move rule is ignored here; it only affects each player's first 3 moves and
// is handled by a small search in oracle.py.
//
// State: 9 stacks x 7 bits + turn bit. Stack code = (1<<h) | colour bits (bit i = piece at level i).
// Output (binary): count, then N x uint64 state, N x int8 value, N x uint16 depth.
// value: 1 = side to move wins, -1 = side to move loses, 0 = draw (infinite play).
// depth: plies to the end under optimal play (winner fastest, loser slowest).
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static const int ADJ[9][5] = {{1,3,-1},{0,2,4,-1},{1,5,-1},{0,4,6,-1},{1,3,5,7,-1},
                              {2,4,8,-1},{3,7,-1},{4,6,8,-1},{5,7,-1}};
static int H[128];
typedef uint64_t S;
#define CAP (1ULL << 28)
static S *tab;
static uint64_t cnt = 0;

static inline uint64_t hh(S x) { x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33;
                                 x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33; return x; }
static inline int64_t find(S s) { uint64_t i = hh(s) & (CAP-1);
    while (tab[i]) { if (tab[i] == s) return i; i = (i+1) & (CAP-1); } return -1; }
static inline int insert(S s) { uint64_t i = hh(s) & (CAP-1);
    while (tab[i]) { if (tab[i] == s) return 0; i = (i+1) & (CAP-1); } tab[i] = s; cnt++; return 1; }
static inline int st(S s, int i) { return (s >> (7*i)) & 127; }
static inline S setst(S s, int i, int c) { return (s & ~((S)127 << (7*i))) | ((S)c << (7*i)); }
static inline int pop(int c) { int h = H[c]; return (c & ((1<<(h-1))-1)) | (1<<(h-1)); }
static inline int push(int c, int col) { int h = H[c]; return ((c ^ (1<<h)) | (col<<h)) | (1<<(h+1)); }

static int moves(S x, S *out) {
    int turn = x >> 63, n = 0;
    for (int a = 0; a < 9; a++) { int ca = st(x,a), ha = H[ca];
        if (!ha || ((ca >> (ha-1)) & 1) != turn) continue;
        for (int k = 0; ADJ[a][k] >= 0; k++) { int b = ADJ[a][k], cb = st(x,b), hb = H[cb];
            if (hb >= 6 || ha < hb) continue;
            S y = setst(setst(x, a, pop(ca)), b, push(cb, turn)); out[n++] = y ^ (1ULL<<63); } }
    return n;
}
// predecessors: previous mover p = 1 - turn(y) moved its top piece from a to b.
static int unmoves(S y, S *out) {
    int p = 1 - (int)(y >> 63), n = 0;
    for (int b = 0; b < 9; b++) { int cb = st(y,b), hb = H[cb];
        if (!hb || ((cb >> (hb-1)) & 1) != p) continue;
        for (int k = 0; ADJ[b][k] >= 0; k++) { int a = ADJ[b][k], ca = st(y,a), ha = H[ca];
            if (ha >= 6) continue;
            if (ha + 1 < hb - 1) continue;           // in x: h_a >= h_b
            S x = setst(setst(y, b, pop(cb)), a, push(ca, p)); out[n++] = x ^ (1ULL<<63); } }
    return n;
}

int main(int argc, char **argv) {
    for (int c = 1; c < 128; c++) { int h = 0; while (c >> (h+1)) h++; H[c] = h; }
    tab = calloc(CAP, 8);
    uint32_t *q = malloc(sizeof(uint32_t) * 100000000ULL);
    int init[9] = {4,4,4,1,1,1,7,7,7};
    S s0 = 1ULL << 63; for (int i = 0; i < 9; i++) s0 = setst(s0, i, init[i]);
    // BFS (store slot indices in q)
    insert(s0); q[0] = find(s0); uint64_t qt = 1; S buf[32];
    for (uint64_t qh = 0; qh < qt; qh++) { S x = tab[q[qh]]; int n = moves(x, buf);
        for (int i = 0; i < n; i++) if (insert(buf[i])) q[qt++] = find(buf[i]); }
    fprintf(stderr, "reachable %lu\n", cnt);
    int8_t *val = calloc(CAP, 1); uint8_t *rem = calloc(CAP, 1); uint16_t *dep = calloc(CAP, 2);
    // init: terminal positions are losses for side to move
    qt = 0;
    for (uint64_t i = 0; i < CAP; i++) if (tab[i]) { int n = moves(tab[i], buf); rem[i] = n;
        if (!n) { val[i] = -1; q[qt++] = i; } }
    fprintf(stderr, "terminal %lu\n", qt);
    for (uint64_t qh = 0; qh < qt; qh++) { uint32_t yi = q[qh]; int n = unmoves(tab[yi], buf);
        for (int i = 0; i < n; i++) { int64_t xi = find(buf[i]); if (xi < 0 || val[xi]) continue;
            if (val[yi] == -1) { val[xi] = 1; dep[xi] = dep[yi] + 1; q[qt++] = xi; }
            else if (--rem[xi] == 0) { val[xi] = -1; dep[xi] = dep[yi] + 1; q[qt++] = xi; } } }
    uint64_t w = 0, l = 0, d = 0; int maxd = 0;
    for (uint64_t i = 0; i < CAP; i++) if (tab[i]) { if (val[i] == 1) w++; else if (val[i] == -1) l++; else d++;
        if (dep[i] > maxd) maxd = dep[i]; }
    int64_t i0 = find(s0);
    fprintf(stderr, "win %lu loss %lu draw %lu maxdepth %d | start(ruleless): val %d depth %d\n",
            w, l, d, maxd, val[i0], dep[i0]);
    FILE *f = fopen(argc > 1 ? argv[1] : "solver/solution.bin", "wb");
    fwrite(&cnt, 8, 1, f);
    for (uint64_t i = 0; i < CAP; i++) if (tab[i]) fwrite(&tab[i], 8, 1, f);
    for (uint64_t i = 0; i < CAP; i++) if (tab[i]) fwrite(&val[i], 1, 1, f);
    for (uint64_t i = 0; i < CAP; i++) if (tab[i]) fwrite(&dep[i], 2, 1, f);
    fclose(f);
    return 0;
}
