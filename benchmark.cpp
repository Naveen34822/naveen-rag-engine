// benchmark.cpp — HNSW vs Brute-Force benchmark on 10K vectors
// Compile: g++ -std=c++17 -O2 -o benchmark benchmark.cpp
// Run:     ./benchmark

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <random>
#include <chrono>
#include <unordered_map>
#include <queue>
#include <functional>
#include <numeric>
#include <iomanip>
#include <set>

// =====================================================================
//  DATA TYPES (copied from main.cpp)
// =====================================================================

struct VectorItem {
    int id;
    std::string metadata;
    std::string category;
    std::vector<float> emb;
};

using DistFn = std::function<float(const std::vector<float>&, const std::vector<float>&)>;

// =====================================================================
//  DISTANCE METRICS
// =====================================================================

float cosine(const std::vector<float>& a, const std::vector<float>& b) {
    float dot=0, na=0, nb=0;
    for (int i = 0; i < (int)a.size(); i++) {
        dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i];
    }
    if (na < 1e-9f || nb < 1e-9f) return 1.0f;
    return 1.0f - dot / (std::sqrt(na) * std::sqrt(nb));
}

// =====================================================================
//  BRUTE FORCE
// =====================================================================

class BruteForce {
public:
    std::vector<VectorItem> items;

    void insert(const VectorItem& v) { items.push_back(v); }

    std::vector<std::pair<float,int>> knn(
        const std::vector<float>& q, int k, DistFn dist)
    {
        std::vector<std::pair<float,int>> r;
        r.reserve(items.size());
        for (auto& v : items) r.push_back({dist(q, v.emb), v.id});
        std::sort(r.begin(), r.end());
        if ((int)r.size() > k) r.resize(k);
        return r;
    }
};

// =====================================================================
//  HNSW — Hierarchical Navigable Small World
// =====================================================================

class HNSW {
    struct Node {
        VectorItem item;
        int maxLyr;
        std::vector<std::vector<int>> nbrs;
    };

    std::unordered_map<int, Node> G;
    int    M, M0, ef_build;
    float  mL;
    int    topLayer = -1;
    int    entryPt  = -1;
    std::mt19937 rng;

    int randLevel() {
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        return (int)std::floor(-std::log(u(rng)) * mL);
    }

    std::vector<std::pair<float,int>> searchLayer(
        const std::vector<float>& q, int ep, int ef, int lyr, DistFn dist)
    {
        std::unordered_map<int,bool> vis;
        std::priority_queue<std::pair<float,int>,
            std::vector<std::pair<float,int>>, std::greater<>> cands;
        std::priority_queue<std::pair<float,int>> found;

        float d0 = dist(q, G[ep].item.emb);
        vis[ep] = true;
        cands.push({d0, ep});
        found.push({d0, ep});

        while (!cands.empty()) {
            auto [cd, cid] = cands.top(); cands.pop();
            if ((int)found.size() >= ef && cd > found.top().first) break;
            if (lyr >= (int)G[cid].nbrs.size()) continue;
            for (int nid : G[cid].nbrs[lyr]) {
                if (vis[nid] || !G.count(nid)) continue;
                vis[nid] = true;
                float nd = dist(q, G[nid].item.emb);
                if ((int)found.size() < ef || nd < found.top().first) {
                    cands.push({nd, nid});
                    found.push({nd, nid});
                    if ((int)found.size() > ef) found.pop();
                }
            }
        }

        std::vector<std::pair<float,int>> res;
        while (!found.empty()) { res.push_back(found.top()); found.pop(); }
        std::sort(res.begin(), res.end());
        return res;
    }

    std::vector<int> selectNbrs(std::vector<std::pair<float,int>>& cands, int maxM) {
        std::vector<int> r;
        for (int i = 0; i < std::min((int)cands.size(), maxM); i++)
            r.push_back(cands[i].second);
        return r;
    }

public:
    HNSW(int m = 16, int efBuild = 200)
        : M(m), M0(2*m), ef_build(efBuild),
          mL(1.0f / std::log((float)m)), rng(42) {}

    void insert(const VectorItem& item, DistFn dist) {
        int id  = item.id;
        int lvl = randLevel();
        G[id]   = {item, lvl, std::vector<std::vector<int>>(lvl + 1)};

        if (entryPt == -1) { entryPt = id; topLayer = lvl; return; }

        int ep = entryPt;
        for (int lc = topLayer; lc > lvl; lc--) {
            if (lc < (int)G[ep].nbrs.size()) {
                auto W = searchLayer(item.emb, ep, 1, lc, dist);
                if (!W.empty()) ep = W[0].second;
            }
        }
        for (int lc = std::min(topLayer, lvl); lc >= 0; lc--) {
            auto W   = searchLayer(item.emb, ep, ef_build, lc, dist);
            int maxM = (lc == 0) ? M0 : M;
            auto sel = selectNbrs(W, maxM);
            G[id].nbrs[lc] = sel;

            for (int nid : sel) {
                if (!G.count(nid)) continue;
                if ((int)G[nid].nbrs.size() <= lc) G[nid].nbrs.resize(lc + 1);
                auto& conn = G[nid].nbrs[lc];
                conn.push_back(id);
                if ((int)conn.size() > maxM) {
                    std::vector<std::pair<float,int>> ds;
                    for (int c : conn) if (G.count(c))
                        ds.push_back({dist(G[nid].item.emb, G[c].item.emb), c});
                    std::sort(ds.begin(), ds.end());
                    conn.clear();
                    for (int i = 0; i < maxM && i < (int)ds.size(); i++)
                        conn.push_back(ds[i].second);
                }
            }
            if (!W.empty()) ep = W[0].second;
        }
        if (lvl > topLayer) { topLayer = lvl; entryPt = id; }
    }

    std::vector<std::pair<float,int>> knn(
        const std::vector<float>& q, int k, int ef, DistFn dist)
    {
        if (entryPt == -1) return {};
        int ep = entryPt;
        for (int lc = topLayer; lc > 0; lc--) {
            if (lc < (int)G[ep].nbrs.size()) {
                auto W = searchLayer(q, ep, 1, lc, dist);
                if (!W.empty()) ep = W[0].second;
            }
        }
        auto W = searchLayer(q, ep, std::max(ef, k), 0, dist);
        if ((int)W.size() > k) W.resize(k);
        return W;
    }

    size_t size() const { return G.size(); }
};

// =====================================================================
//  BENCHMARK
// =====================================================================

int main() {
    // Configuration
    const int DIMS         = 768;    // realistic embedding dimension (nomic-embed-text)
    const int N            = 10000;  // 10K document corpus
    const int NUM_CLUSTERS = 50;     // simulate 50 topics
    const int NUM_QUERIES  = 200;    // average over 200 queries
    const int K            = 5;
    const int EF_VALUES[]  = {50, 100, 200};

    std::mt19937 rng(12345);
    std::normal_distribution<float> norm(0.0f, 1.0f);
    std::uniform_int_distribution<int> clusterPick(0, NUM_CLUSTERS - 1);

    // Generate cluster centroids (simulate topic embeddings)
    std::vector<std::vector<float>> centroids;
    for (int c = 0; c < NUM_CLUSTERS; c++) {
        std::vector<float> v(DIMS);
        float mag = 0;
        for (auto& x : v) { x = norm(rng); mag += x * x; }
        mag = std::sqrt(mag);
        for (auto& x : v) x /= mag;
        centroids.push_back(v);
    }

    // Generate clustered vectors: centroid + small noise (simulates real embeddings)
    auto clusteredVec = [&]() -> std::pair<int, std::vector<float>> {
        int cluster = clusterPick(rng);
        std::vector<float> v(DIMS);
        float mag = 0;
        for (int i = 0; i < DIMS; i++) {
            v[i] = centroids[cluster][i] + norm(rng) * 0.15f;  // tight clusters
            mag += v[i] * v[i];
        }
        mag = std::sqrt(mag);
        for (auto& x : v) x /= mag;
        return {cluster, v};
    };

    std::cout << "═══════════════════════════════════════════════════════════\n";
    std::cout << "  RAG Engine Benchmark: HNSW vs Brute-Force\n";
    std::cout << "  " << N << " vectors | " << DIMS << "D | " << NUM_CLUSTERS << " topic clusters\n";
    std::cout << "  " << NUM_QUERIES << " queries | k=" << K << "\n";
    std::cout << "═══════════════════════════════════════════════════════════\n\n";

    // Build both indexes
    BruteForce bf;
    HNSW hnsw(16, 200);

    std::cout << "  Inserting " << N << " clustered vectors...";
    std::cout.flush();

    auto insertStart = std::chrono::high_resolution_clock::now();
    for (int i = 1; i <= N; i++) {
        auto [cluster, emb] = clusteredVec();
        VectorItem v{i, "doc-" + std::to_string(i), "topic-" + std::to_string(cluster), emb};
        bf.insert(v);
        hnsw.insert(v, cosine);
    }
    auto insertEnd = std::chrono::high_resolution_clock::now();
    double insertMs = std::chrono::duration<double, std::milli>(insertEnd - insertStart).count();
    std::cout << " done (" << std::fixed << std::setprecision(0) << insertMs << "ms)\n\n";

    // Generate queries from random clusters (simulate real search queries)
    std::vector<std::vector<float>> queries;
    for (int i = 0; i < NUM_QUERIES; i++) {
        auto [cluster, emb] = clusteredVec();
        queries.push_back(emb);
    }

    // Benchmark brute-force
    auto bfStart = std::chrono::high_resolution_clock::now();
    for (auto& q : queries) bf.knn(q, K, cosine);
    auto bfEnd = std::chrono::high_resolution_clock::now();
    double bfTotalUs = std::chrono::duration<double, std::micro>(bfEnd - bfStart).count();
    double bfAvgUs = bfTotalUs / NUM_QUERIES;

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "  Brute-Force avg: " << bfAvgUs << " μs/query\n\n";

    // Benchmark HNSW at different ef_search values
    std::cout << "  ┌──────────┬─────────────────┬──────────┬───────────┐\n";
    std::cout << "  │ ef_search│  HNSW avg (μs)  │ Speedup  │ Recall@"<< K <<"  │\n";
    std::cout << "  ├──────────┼─────────────────┼──────────┼───────────┤\n";

    for (int ef : EF_VALUES) {
        // Time HNSW
        auto hnswStart = std::chrono::high_resolution_clock::now();
        for (auto& q : queries) hnsw.knn(q, K, ef, cosine);
        auto hnswEnd = std::chrono::high_resolution_clock::now();
        double hnswTotalUs = std::chrono::duration<double, std::micro>(hnswEnd - hnswStart).count();
        double hnswAvgUs = hnswTotalUs / NUM_QUERIES;
        double speedup = bfAvgUs / hnswAvgUs;

        // Check recall
        int totalRecall = 0;
        for (auto& q : queries) {
            auto bfRes = bf.knn(q, K, cosine);
            auto hnswRes = hnsw.knn(q, K, ef, cosine);
            std::set<int> bfIds;
            for (auto& [d, id] : bfRes) bfIds.insert(id);
            for (auto& [d, id] : hnswRes)
                if (bfIds.count(id)) totalRecall++;
        }
        double recall = (double)totalRecall / (NUM_QUERIES * K) * 100.0;

        std::cout << "  │ " << std::setw(8) << ef
                  << " │ " << std::setw(15) << hnswAvgUs
                  << " │ " << std::setw(7) << speedup << "x"
                  << " │ " << std::setw(8) << recall << "%"
                  << " │\n";
    }

    std::cout << "  └──────────┴─────────────────┴──────────┴───────────┘\n\n";
    std::cout << "═══════════════════════════════════════════════════════════\n";
    std::cout << "  Use the best speedup with ≥90% recall for your resume!\n";
    std::cout << "═══════════════════════════════════════════════════════════\n";

    return 0;
}
