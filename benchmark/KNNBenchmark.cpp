#include <benchmark/benchmark.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <queue>
#include <cmath>
#include <map>
#include <thread>
#include <mutex>
#include <semaphore>
#include <barrier>
#include <atomic>
#include <memory>
#include <algorithm>

static const std::string FILE_PATH   = "dataset_high_dim.csv";
static const int         K           = 21;
static const int         NUM_THREADS = std::max(2u, std::thread::hardware_concurrency());

class Neighbor {
public:
    std::vector<double> values;
    std::string label;
    Neighbor(const std::vector<double>& v, const std::string& l) : values(v), label(l) {}
};

struct DistanceRecord {
    Neighbor neighbor;
    double distance;
    DistanceRecord(const Neighbor& n, double d) : neighbor(n), distance(d) {}
    bool operator<(const DistanceRecord& o) const { return distance < o.distance; }
};

struct ChunkBounds { long long startByte, endByte; };

static Neighbor* parseLine(const std::string& line) {
    std::vector<std::string> parts;
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, ',')) parts.push_back(token);
    if (parts.size() < 2) return nullptr;
    std::vector<double> values;
    try {
        for (size_t i = 0; i < parts.size() - 1; i++) {
            std::string c;
            for (char ch : parts[i]) if (!std::isspace((unsigned char)ch)) c += ch;
            values.push_back(std::stod(c));
        }
        std::string label = parts.back();
        label.erase(0, label.find_first_not_of(" \t\r\n"));
        label.erase(label.find_last_not_of(" \t\r\n") + 1);
        return new Neighbor(values, label);
    } catch (...) { return nullptr; }
}

static double euclidDist(const Neighbor& a, const Neighbor& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.values.size(); i++) {
        double d = a.values[i] - b.values[i]; sum += d * d;
    }
    return std::sqrt(sum);
}

static std::string majorityVote(std::priority_queue<DistanceRecord>& pq) {
    std::map<std::string, int> freq;
    while (!pq.empty()) { freq[pq.top().neighbor.label]++; pq.pop(); }
    return std::max_element(freq.begin(), freq.end(),
        [](const auto& a, const auto& b){ return a.second < b.second; })->first;
}

static std::vector<ChunkBounds> computeChunks(const std::string& fp, int n) {
    std::vector<ChunkBounds> chunks;
    std::ifstream file(fp, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return chunks;
    long long sz = file.tellg(); if (sz == 0) return chunks;
    file.seekg(0); std::string hdr; std::getline(file, hdr);
    long long ds = file.tellg(); if (ds >= sz) return chunks;
    long long csz = (sz - ds) / n, cs = ds;
    for (int i = 0; i < n; i++) {
        long long ce;
        if (i == n - 1) { ce = sz; }
        else {
            file.seekg(ds + (long long)(i+1) * csz);
            char c; while (file.get(c) && c != '\n') {}
            ce = file.eof() ? sz : (long long)file.tellg();
        }
        if (cs < ce) chunks.push_back({cs, ce});
        cs = ce; if (cs >= sz) break;
    }
    return chunks;
}

static std::priority_queue<DistanceRecord> processChunk(
    const std::string& fp, const ChunkBounds& chunk, const Neighbor& target, int k)
{
    std::priority_queue<DistanceRecord> local;
    std::ifstream file(fp, std::ios::binary);
    if (!file.is_open()) return local;
    file.seekg(chunk.startByte);
    std::string line;
    while (file.tellg() < chunk.endByte && std::getline(file, line)) {
        if (line.empty()) continue;
        Neighbor* cur = parseLine(line);
        if (!cur) continue;
        if (cur->values.size() != target.values.size()) { delete cur; continue; }
        double dist = euclidDist(target, *cur);
        DistanceRecord rec(*cur, dist); delete cur;
        if ((int)local.size() < k) local.push(rec);
        else if (dist < local.top().distance) { local.pop(); local.push(rec); }
    }
    return local;
}

static void mergeLocal(std::priority_queue<DistanceRecord>& global,
                       std::priority_queue<DistanceRecord>& local, int k) {
    auto copy = local;
    while (!copy.empty()) {
        auto& r = copy.top();
        if ((int)global.size() < k) global.push(r);
        else if (r.distance < global.top().distance) { global.pop(); global.push(r); }
        copy.pop();
    }
}

static Neighbor buildTarget() {
    std::ifstream file(FILE_PATH);
    std::string hdr, first;
    std::getline(file, hdr); std::getline(file, first);
    int n = 0; std::stringstream ss(first); std::string t;
    while (std::getline(ss, t, ',')) n++;
    return Neighbor(std::vector<double>(n - 1, 500.0), "Unknown");
}
static Neighbor TARGET = buildTarget();

static void BM_Serial(benchmark::State& state) {
    for (auto _ : state) {
        std::priority_queue<DistanceRecord> pq;
        std::ifstream file(FILE_PATH);
        std::string line; bool first = true;
        while (std::getline(file, line)) {
            if (first) { first = false; continue; }
            Neighbor* cur = parseLine(line);
            if (!cur) continue;
            if (cur->values.size() != TARGET.values.size()) { delete cur; continue; }
            double dist = euclidDist(TARGET, *cur);
            pq.push(DistanceRecord(*cur, dist)); delete cur;
            if ((int)pq.size() > K) pq.pop();
        }
        std::string result = majorityVote(pq);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Serial)->Name("test01_Serial")->Unit(benchmark::kSecond);

static void BM_Platform(benchmark::State& state) {
    for (auto _ : state) {
        auto chunks = computeChunks(FILE_PATH, NUM_THREADS);
        std::vector<std::priority_queue<DistanceRecord>> results(chunks.size());
        std::vector<std::thread> threads;
        for (int i = 0; i < (int)chunks.size(); i++)
            threads.emplace_back([&, i]() { results[i] = processChunk(FILE_PATH, chunks[i], TARGET, K); });
        for (auto& t : threads) if (t.joinable()) t.join();
        std::priority_queue<DistanceRecord> global;
        for (auto& local : results) mergeLocal(global, local, K);
        std::string result = majorityVote(global);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Platform)->Name("test02_Platform")->Unit(benchmark::kSecond);

static void BM_Synchronized(benchmark::State& state) {
    for (auto _ : state) {
        auto chunks = computeChunks(FILE_PATH, NUM_THREADS);
        std::priority_queue<DistanceRecord> global;
        std::mutex mtx;
        std::vector<std::thread> threads;
        for (int i = 0; i < (int)chunks.size(); i++) {
            threads.emplace_back([&, i]() {
                std::ifstream file(FILE_PATH, std::ios::binary);
                file.seekg(chunks[i].startByte);
                std::string line;
                while (file.tellg() < chunks[i].endByte && std::getline(file, line)) {
                    if (line.empty()) continue;
                    Neighbor* cur = parseLine(line);
                    if (!cur) continue;
                    if (cur->values.size() != TARGET.values.size()) { delete cur; continue; }
                    double dist = euclidDist(TARGET, *cur);
                    DistanceRecord rec(*cur, dist); delete cur;
                    std::lock_guard<std::mutex> lock(mtx);
                    if ((int)global.size() < K) global.push(rec);
                    else if (dist < global.top().distance) { global.pop(); global.push(rec); }
                }
            });
        }
        for (auto& t : threads) if (t.joinable()) t.join();
        std::string result = majorityVote(global);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Synchronized)->Name("test05_Synchronized")->Unit(benchmark::kSecond);

static void BM_Lock(benchmark::State& state) {
    for (auto _ : state) {
        auto chunks = computeChunks(FILE_PATH, NUM_THREADS);
        std::priority_queue<DistanceRecord> global;
        std::recursive_mutex rmtx;
        std::vector<std::thread> threads;
        for (int i = 0; i < (int)chunks.size(); i++) {
            threads.emplace_back([&, i]() {
                std::ifstream file(FILE_PATH, std::ios::binary);
                file.seekg(chunks[i].startByte);
                std::string line;
                while (file.tellg() < chunks[i].endByte && std::getline(file, line)) {
                    if (line.empty()) continue;
                    Neighbor* cur = parseLine(line);
                    if (!cur) continue;
                    if (cur->values.size() != TARGET.values.size()) { delete cur; continue; }
                    double dist = euclidDist(TARGET, *cur);
                    DistanceRecord rec(*cur, dist); delete cur;
                    std::unique_lock<std::recursive_mutex> guard(rmtx);
                    if ((int)global.size() < K) global.push(rec);
                    else if (dist < global.top().distance) { global.pop(); global.push(rec); }
                }
            });
        }
        for (auto& t : threads) if (t.joinable()) t.join();
        std::string result = majorityVote(global);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Lock)->Name("test06_Lock")->Unit(benchmark::kSecond);

static void BM_Semaphore(benchmark::State& state) {
    for (auto _ : state) {
        auto chunks = computeChunks(FILE_PATH, NUM_THREADS);
        std::priority_queue<DistanceRecord> global;
        std::binary_semaphore sem(1);
        std::vector<std::thread> threads;
        for (int i = 0; i < (int)chunks.size(); i++) {
            threads.emplace_back([&, i]() {
                auto local = processChunk(FILE_PATH, chunks[i], TARGET, K);
                sem.acquire();
                mergeLocal(global, local, K);
                sem.release();
            });
        }
        for (auto& t : threads) if (t.joinable()) t.join();
        std::string result = majorityVote(global);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Semaphore)->Name("test07_Semaphore")->Unit(benchmark::kSecond);

static void BM_Barrier(benchmark::State& state) {
    for (auto _ : state) {
        auto chunks = computeChunks(FILE_PATH, NUM_THREADS);
        int n = (int)chunks.size();
        std::vector<std::priority_queue<DistanceRecord>> results(n);
        std::string finalLabel = "Unknown";
        auto fn = [&]() noexcept {
            std::priority_queue<DistanceRecord> global;
            for (auto& local : results) mergeLocal(global, local, K);
            if (!global.empty()) finalLabel = majorityVote(global);
        };
        std::barrier sync(n + 1, fn);
        std::vector<std::thread> threads;
        for (int i = 0; i < n; i++) {
            threads.emplace_back([&, i]() {
                results[i] = processChunk(FILE_PATH, chunks[i], TARGET, K);
                sync.arrive_and_wait();
            });
        }
        sync.arrive_and_wait();
        for (auto& t : threads) if (t.joinable()) t.join();
        benchmark::DoNotOptimize(finalLabel);
    }
}
BENCHMARK(BM_Barrier)->Name("test08_Barrier")->Unit(benchmark::kSecond);

class ImmutableTopK {
public:
    std::vector<DistanceRecord> list;
    int k;
    explicit ImmutableTopK(int k) : k(k) {}
    ImmutableTopK(std::vector<DistanceRecord> l, int k) : list(std::move(l)), k(k) {}

    std::shared_ptr<ImmutableTopK> tryUpdate(const DistanceRecord& rec) const {
        if ((int)list.size() < k) {
            auto nl = list; nl.push_back(rec);
            std::sort(nl.begin(), nl.end());
            return std::make_shared<ImmutableTopK>(std::move(nl), k);
        }
        if (rec.distance < list.back().distance) {
            auto nl = list; nl.pop_back(); nl.push_back(rec);
            std::sort(nl.begin(), nl.end());
            return std::make_shared<ImmutableTopK>(std::move(nl), k);
        }
        return nullptr;
    }
};

static void BM_Atomic(benchmark::State& state) {
    for (auto _ : state) {
        auto chunks = computeChunks(FILE_PATH, NUM_THREADS);
        std::atomic<std::shared_ptr<ImmutableTopK>> globalTopK(
            std::make_shared<ImmutableTopK>(K)
        );
        std::vector<std::thread> threads;
        for (int i = 0; i < (int)chunks.size(); i++) {
            threads.emplace_back([&, i]() {
                std::ifstream file(FILE_PATH, std::ios::binary);
                if (!file.is_open()) return;
                file.seekg(chunks[i].startByte);
                std::string line;
                while (file.tellg() < chunks[i].endByte && std::getline(file, line)) {
                    if (line.empty()) continue;
                    Neighbor* cur = parseLine(line);
                    if (!cur) continue;
                    if (cur->values.size() != TARGET.values.size()) { delete cur; continue; }
                    double dist = euclidDist(TARGET, *cur);
                    auto snap = globalTopK.load(std::memory_order_acquire);
                    if ((int)snap->list.size() == K && dist >= snap->list.back().distance) {
                        delete cur; continue;
                    }
                    DistanceRecord rec(*cur, dist); delete cur;
                    while (true) {
                        snap = globalTopK.load(std::memory_order_acquire);
                        auto next = snap->tryUpdate(rec);
                        if (!next) break;
                        if (std::atomic_compare_exchange_strong_explicit(
                                &globalTopK, &snap, next,
                                std::memory_order_release,
                                std::memory_order_acquire)) break;
                    }
                }
            });
        }
        for (auto& t : threads) if (t.joinable()) t.join();
        auto final = globalTopK.load();
        std::string result = "Unknown";
        if (!final->list.empty()) {
            std::map<std::string, int> freq;
            for (auto& r : final->list) freq[r.neighbor.label]++;
            result = std::max_element(freq.begin(), freq.end(),
                [](const auto& a, const auto& b){ return a.second < b.second; })->first;
        }
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Atomic)->Name("test09_Atomic")->Unit(benchmark::kSecond);


BENCHMARK_MAIN();