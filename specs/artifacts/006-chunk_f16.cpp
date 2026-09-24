// Throwaway: does an f16-operand / fp32-accumulate chunked gated-delta-rule survive
// error accumulation across chunks? Compares against the fp32 recurrent form, which is
// what the CUDA kernel computes today. Not part of the build.
//
// rnd16() models mma.sync with half operands and fp32 accumulation: the operand is rounded
// to half, the products are exact in fp32, the accumulation is fp32. The state is the operand
// that is fed back every chunk, so this is where accumulation across chunks can blow up.

#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <random>

static const int DK = 128, DV = 128, C = 64, NT = 4096;

typedef std::vector<float> vec;

static inline float rnd16(float x) {
    // round to half precision: 1 implicit + 10 explicit mantissa bits, RN-even
    if (x == 0.0f || !std::isfinite(x)) return x;
    int   e = 0;
    float m = std::frexpf(x, &e);                 // x = m * 2^e, |m| in [0.5, 1)
    if (std::abs(m) < 0.5f) return x;             // half subnormal range, not reached here
    if (e > 16 || e < -14) return x;             // outside half normal range, not reached here
    float si = std::nearbyintf(std::ldexp(m, 11)); // |m|*2^11 in [1024, 2048)
    return std::ldexp(std::ldexp(si, -11), e);
}

static inline float dot(const float * a, const float * b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

// state layout S[i*DV + c]
static void recurrent(const vec & Q, const vec & K, const vec & V, const vec & G, const vec & B,
                      vec & S, vec & OUT, float scale) {
    for (int t = 0; t < NT; t++) {
        const float * q = &Q[t*DK], * k = &K[t*DK], * v = &V[t*DV];
        const float g = G[t], beta = B[t];
        for (int c = 0; c < DV; c++) {
            float kv = 0.0f;
            for (int i = 0; i < DK; i++) kv += S[i*DV + c] * k[i];
            float delta = beta * (v[c] - g * kv);
            for (int i = 0; i < DK; i++) S[i*DV + c] = g * S[i*DV + c] + k[i] * delta;
        }
        for (int c = 0; c < DV; c++) {
            float a = 0.0f;
            for (int i = 0; i < DK; i++) a += S[i*DV + c] * q[i];
            OUT[t*DV + c] = a * scale;
        }
    }
}

// F16: 1 = round the operands of the big matmuls to half, keep the fp32 math for the
// small triangular part. HILO: 2 = split each half operand into hi+lo halves and run the
// matmul twice, which buys back mantissa bits at 2x the mma work.
template <int F16>
static void chunked(const vec & Q, const vec & K, const vec & V, const vec & G, const vec & B,
                    vec & S, vec & OUT, float scale) {
    vec T(C*C), rhs(C*DV), del(C*DV), KS(C*DV), QS(C*DV), gam(C), Sh(DK*DV);
    vec Kh(NT*DK), Qh(NT*DK);

    for (int i = 0; i < NT*DK; i++) { Kh[i] = F16 ? rnd16(K[i]) : K[i]; Qh[i] = F16 ? rnd16(Q[i]) : Q[i]; }

    for (int t0 = 0; t0 < NT; t0 += C) {
        const int n = C;
        for (int t = 0; t < n; t++) gam[t] = t == 0 ? G[t0] : gam[t-1] * G[t0 + t];

        // the state is resident as half precision between chunks: that is the whole point
        // of using mma here, and it is what can drift
        for (int i = 0; i < DK*DV; i++) Sh[i] = F16 ? rnd16(S[i]) : S[i];

        // KS = K S, QS = Q S  -> mma with half operands
        for (int t = 0; t < n; t++) {
            for (int c = 0; c < DV; c++) {
                float a = 0.0f, b = 0.0f;
                for (int i = 0; i < DK; i++) {
                    a += Kh[(t0+t)*DK + i] * Sh[i*DV + c];
                    b += Qh[(t0+t)*DK + i] * Sh[i*DV + c];
                }
                KS[t*DV + c] = a;
                QS[t*DV + c] = b;
            }
        }

        for (int t = 0; t < n; t++) {
            for (int s = 0; s < n; s++) T[t*n + s] = 0.0f;
        }
        for (int t = 0; t < n; t++) {
            for (int s = 0; s < t; s++) {
                T[t*n + s] = B[t0+t] * dot(&Kh[(t0+t)*DK], &Kh[(t0+s)*DK], DK) * (gam[t] / gam[s]);
            }
            for (int c = 0; c < DV; c++) {
                rhs[t*DV + c] = B[t0+t] * (V[(t0+t)*DV + c] - gam[t] * KS[t*DV + c]);
            }
        }

        // unit lower triangular solve, kept in fp32 SIMT (0.52 M of 10.2 M FLOP)
        for (int c = 0; c < DV; c++) {
            for (int i = 0; i < n; i++) {
                float s = rhs[i*DV + c];
                for (int j = 0; j < i; j++) s -= T[i*n + j] * del[j*DV + c];
                del[i*DV + c] = s;
            }
        }

        vec dl(C*DV);
        if (F16) for (int i = 0; i < C*DV; i++) dl[i] = rnd16(del[i]);
        else     dl = del;

        for (int t = 0; t < n; t++) {
            for (int c = 0; c < DV; c++) {
                float a = gam[t] * QS[t*DV + c];
                for (int s = 0; s <= t; s++) {
                    a += dot(&Qh[(t0+t)*DK], &Kh[(t0+s)*DK], DK) * (gam[t]/gam[s]) * dl[s*DV + c];
                }
                OUT[(t0+t)*DV + c] = a * scale;
            }
        }

        const float gC = gam[n-1];
        for (int i = 0; i < DK; i++) {
            for (int c = 0; c < DV; c++) {
                float a = 0.0f;
                for (int s = 0; s < n; s++) {
                    a += (gC / gam[s]) * Kh[(t0+s)*DK + i] * dl[s*DV + c];
                }
                S[i*DV + c] = gC * S[i*DV + c] + a;
            }
        }
    }
}

template <int F16>
static void run(const char * label, int seed, float gscale) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nrm(0.0f, 1.0f);
    const float scale = 1.0f / sqrtf((float) DV);

    vec Q(NT*DK), K(NT*DK), V(NT*DV), G(NT), B(NT), S0(DK*DV), SA(DK*DV), SB(DK*DV);
    vec OA(NT*DV), OB(NT*DV);

    for (auto & x : V) x = nrm(rng);
    for (auto & x : S0) x = nrm(rng) * 0.1f;
    for (int t = 0; t < NT; t++) {
        G[t] = std::exp(-std::abs(nrm(rng)) * gscale);
        B[t] = 1.0f / (1.0f + std::exp(-nrm(rng)));
    }
    for (int t = 0; t < NT; t++) {
        for (int w = 0; w < 2; w++) {
            vec & X = w ? K : Q;
            float * p = &X[t*DK];
            for (int i = 0; i < DK; i++) p[i] = nrm(rng);
            float n = 0.0f;
            for (int i = 0; i < DK; i++) n += p[i]*p[i];
            n = 1.0f / std::sqrt(n);
            for (int i = 0; i < DK; i++) p[i] *= n;
        }
    }

    SA = S0; SB = S0;
    recurrent(Q, K, V, G, B, SA, OA, scale);
    chunked<F16>(Q, K, V, G, B, SB, OB, scale);

    double eo = 0.0, es = 0.0, mo = 0.0, ms = 0.0, rms = 0.0;
    for (int i = 0; i < NT*DV; i++) {
        eo = std::max(eo, (double) std::abs(OA[i] - OB[i]));
        mo = std::max(mo, (double) std::abs(OA[i]));
    }
    for (int i = 0; i < DK*DV; i++) {
        es = std::max(es, (double) std::abs(SA[i] - SB[i]));
        ms = std::max(ms, (double) std::abs(SA[i]));
    }
    printf("%-22s out: max %.3e (%.2f%% of %.4f)   state after %d chunks: max %.3e (%.2f%% of %.4f)\n",
           label, eo, 100.0*eo/mo, mo, NT/C, es, 100.0*es/ms, ms);
}

int main() {
    // gate strength sweep: g = exp(-|n| * s). Small s means slow decay, so the state lives
    // across many chunks and any operand error accumulates instead of being flushed out
    for (float s : { 0.5f, 0.05f, 0.005f, 0.0005f }) {
        printf("-- gate scale %g (mean per-token decay ~%g, chunk retention ~%g)\n",
               s, std::exp(-0.8*s), std::pow(std::exp(-0.8*s), 64.0));
        run<0>("fp32 chunked", 1, s);
        run<1>("f16 operands", 1, s);
        run<1>("f16 operands", 2, s);
    }
    return 0;
}
