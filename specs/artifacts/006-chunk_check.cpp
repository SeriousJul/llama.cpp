// Throwaway: validate the chunked gated-delta-rule derivation against the recurrent
// form the CUDA kernel implements today. Not part of the build.
//
// recurrent, per head, per token t:
//   kv_t    = S_{t-1}^T k_t            (over rows i)
//   delta_t = beta_t (v_t - g_t kv_t)
//   S_t     = g_t S_{t-1} + k_t (x) delta_t
//   out_t   = scale * S_t^T q_t
//
// chunked, chunk of C tokens, S0 = state at chunk start, gam_t = prod_{s<=t} g_s:
//   T[t][s]   = beta_t (k_t . k_s) gam_t/gam_s        for s < t, strictly lower
//   rhs_t     = beta_t v_t - beta_t gam_t (S0^T k_t)
//   delta     = (I + T)^-1 rhs
//   out_t     = scale [ gam_t (S0^T q_t) + sum_{s<=t} (q_t . k_s)(gam_t/gam_s) delta_s ]
//   S_end     = gam_C S0 + sum_s (gam_C/gam_s) k_s (x) delta_s

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <random>

static const int DK = 128, DV = 128, C = 128, NT = 512;

typedef std::vector<float> vec;

static inline float dot(const float * a, const float * b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

// state is [DK][DV], row major: S[i*DV + c]
static void recurrent(const vec & Q, const vec & K, const vec & V, const vec & G, const vec & B,
                      vec & S, vec & OUT, float scale) {
    for (int t = 0; t < NT; t++) {
        const float * q = &Q[t*DK];
        const float * k = &K[t*DK];
        const float * v = &V[t*DV];
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

// solve (I + T) x = rhs, T strictly lower triangular, C x C
static void solve_unit_lower(const float * T, const float * rhs, float * x, int n, int ld_rhs) {
    for (int c = 0; c < DV; c++) {
        for (int i = 0; i < n; i++) {
            float s = rhs[i*ld_rhs + c];
            for (int j = 0; j < i; j++) s -= T[i*n + j] * x[j*DV + c];
            x[i*DV + c] = s;   // unit diagonal
        }
    }
}

static void chunked(const vec & Q, const vec & K, const vec & V, const vec & G, const vec & B,
                    vec & S, vec & OUT, float scale) {
    std::vector<float> T(C*C), rhs(C*DV), del(C*DV), KS(C*DV), QS(C*DV), gam(C);

    for (int t0 = 0; t0 < NT; t0 += C) {
        const int n = C;   // NT is a multiple of C here

        // cumulative gate over the chunk
        for (int t = 0; t < n; t++) {
            gam[t] = t == 0 ? G[t0] : gam[t-1] * G[t0 + t];
        }

        // KS = K S0, QS = Q S0
        for (int t = 0; t < n; t++) {
            for (int c = 0; c < DV; c++) {
                float a = 0.0f, b = 0.0f;
                for (int i = 0; i < DK; i++) {
                    a += K[(t0+t)*DK + i] * S[i*DV + c];
                    b += Q[(t0+t)*DK + i] * S[i*DV + c];
                }
                KS[t*DV + c] = a;
                QS[t*DV + c] = b;
            }
        }

        // T and rhs
        for (int t = 0; t < n; t++) {
            for (int s = 0; s < n; s++) T[t*n + s] = 0.0f;
        }
        for (int t = 0; t < n; t++) {
            for (int s = 0; s < t; s++) {
                T[t*n + s] = B[t0+t] * dot(&K[(t0+t)*DK], &K[(t0+s)*DK], DK) * (gam[t] / gam[s]);
            }
            for (int c = 0; c < DV; c++) {
                rhs[t*DV + c] = B[t0+t] * (V[(t0+t)*DV + c] - gam[t] * KS[t*DV + c]);
            }
        }

        solve_unit_lower(T.data(), rhs.data(), del.data(), n, DV);

        // out_t = scale [ gam_t QS_t + sum_{s<=t} (q_t . k_s)(gam_t/gam_s) del_s ]
        for (int t = 0; t < n; t++) {
            for (int c = 0; c < DV; c++) {
                float a = gam[t] * QS[t*DV + c];
                for (int s = 0; s <= t; s++) {
                    a += dot(&Q[(t0+t)*DK], &K[(t0+s)*DK], DK) * (gam[t]/gam[s]) * del[s*DV + c];
                }
                OUT[(t0+t)*DV + c] = a * scale;
            }
        }

        // S_end = gam_C S0 + sum_s (gam_C/gam_s) k_s (x) del_s
        const float gC = gam[n-1];
        for (int i = 0; i < DK; i++) {
            for (int c = 0; c < DV; c++) {
                float a = 0.0f;
                for (int s = 0; s < n; s++) {
                    a += (gC / gam[s]) * K[(t0+s)*DK + i] * del[s*DV + c];
                }
                S[i*DV + c] = gC * S[i*DV + c] + a;
            }
        }
    }
}

int main(int argc, char ** argv) {
    std::mt19937 rng(argc > 1 ? atoi(argv[1]) : 1234);
    std::normal_distribution<float> nrm(0.0f, 1.0f);

    const float scale = 1.0f / sqrtf((float) DV);

    vec Q(NT*DK), K(NT*DK), V(NT*DV), G(NT), B(NT), S0(DK*DV), SA(DK*DV), SB(DK*DV);
    vec OA(NT*DV), OB(NT*DV);

    for (auto & x : Q) x = nrm(rng);
    for (auto & x : K) x = nrm(rng);
    // qwen35 runs build_gdn_l2_norm over q and k before the op, so |q| = |k| = 1
    for (int t = 0; t < NT; t++) {
        for (int which = 0; which < 2; which++) {
            vec & v = which ? K : Q;
            float * p2 = &v[t*DK];
            float n = 0.0f;
            for (int i = 0; i < DK; i++) n += p2[i]*p2[i];
            n = 1.0f / sqrtf(n);
            for (int i = 0; i < DK; i++) p2[i] *= n;
        }
    }
    for (auto & x : V) x = nrm(rng);
    for (auto & x : S0) x = nrm(rng) * 0.1f;
    for (int t = 0; t < NT; t++) {
        G[t] = std::exp(-std::abs(nrm(rng)) * 0.05f);   // decay in (0,1)
        B[t] = 1.0f / (1.0f + std::exp(-nrm(rng)));      // sigmoid beta
    }

    SA = S0; SB = S0;
    recurrent(Q, K, V, G, B, SA, OA, scale);
    chunked(Q, K, V, G, B, SB, OB, scale);

    double e_out = 0.0, e_state = 0.0, m_out = 0.0, m_state = 0.0;
    for (int i = 0; i < NT*DV; i++) {
        e_out = std::max(e_out, (double) std::abs(OA[i] - OB[i]));
        m_out = std::max(m_out, (double) std::abs(OA[i]));
    }
    for (int i = 0; i < DK*DV; i++) {
        e_state = std::max(e_state, (double) std::abs(SA[i] - SB[i]));
        m_state = std::max(m_state, (double) std::abs(SA[i]));
    }
    printf("max |out|        = %.6g\n", m_out);
    printf("max abs out err  = %.6g   (%.2f %% of max)\n", e_out, 100.0 * e_out / m_out);
    printf("max |state|      = %.6g\n", m_state);
    printf("max abs state er = %.6g   (%.2f %% of max)\n", e_state, 100.0 * e_state / m_state);
    return 0;
}
