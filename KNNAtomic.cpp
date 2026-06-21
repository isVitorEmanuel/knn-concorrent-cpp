#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <list>
#include <cmath>
#include <map>
#include <thread>
#include <atomic>
#include <memory>
#include <algorithm>

/**
 * @class Neighbor
 * @brief Represents a data point with feature values and a class label.
 */
class Neighbor {
public:
    std::vector<double> values;
    std::string label;

    Neighbor(const std::vector<double>& values, const std::string& label)
        : values(values), label(label) {}
};

/**
 * @struct DistanceRecord
 * @brief Associates a Neighbor with its computed distance to the target point.
 */
struct DistanceRecord {
    Neighbor neighbor;
    double distance;

    DistanceRecord(const Neighbor& neighbor, double distance)
        : neighbor(neighbor), distance(distance) {}

    bool operator<(const DistanceRecord& other) const {
        return this->distance < other.distance;
    }
};

/**
 * @struct ChunkBounds
 * @brief Represents the byte boundaries (start and end) of a file segment.
 */
struct ChunkBounds {
    long long startByte;
    long long endByte;
};

/**
 * @class ImmutableTopK
 * @brief Immutable snapshot of the current best K neighbors, sorted ascending by distance.
 *        Each update produces a NEW object — the original is never modified.
 *        This immutability is what makes lock-free CAS safe.
 *        Equivalent to the ImmutableTopK inner class in Java KNNAtomic.
 */
class ImmutableTopK {
public:
    std::vector<DistanceRecord> list;
    int k;

    explicit ImmutableTopK(int k) : k(k) {}

    ImmutableTopK(std::vector<DistanceRecord> list, int k)
        : list(std::move(list)), k(k) {}

    /**
     * @brief Evaluates a new record and returns a new snapshot if it qualifies.
     *        Returns nullptr if the record does not improve the current top-K.
     *        Equivalent to tryUpdate() in Java KNNAtomic.
     */
    std::shared_ptr<ImmutableTopK> tryUpdate(const DistanceRecord& record) const {
        if ((int)list.size() < k) {
            std::vector<DistanceRecord> newList = list;
            newList.push_back(record);
            std::sort(newList.begin(), newList.end());
            return std::make_shared<ImmutableTopK>(std::move(newList), k);
        }

        const DistanceRecord& worst = list.back();
        if (record.distance < worst.distance) {
            std::vector<DistanceRecord> newList = list;
            newList.pop_back();
            newList.push_back(record);
            std::sort(newList.begin(), newList.end());
            return std::make_shared<ImmutableTopK>(std::move(newList), k);
        }

        return nullptr;
    }
};

/**
 * @class KNNAtomic
 * @brief Implements KNN using parallel platform threads and a lock-free shared
 *        global state managed by std::atomic<std::shared_ptr<ImmutableTopK>>.
 *        Uses CAS (Compare-And-Swap) loops to update state without any mutex.
 *        Equivalent to the Java KNNAtomic using AtomicReference<ImmutableTopK>.
 */
class KNNAtomic {
public:

    /**
     * @brief Entry point for the lock-free atomic parallel classification.
     */
    std::string predictStream(const std::string& filePath, const Neighbor& target, int k) {
        int numThreads = std::max(2u, std::thread::hardware_concurrency());
        std::cout << "[Atomic] Using " << numThreads
                  << " platform threads with lock-free AtomicReference" << std::endl;
        return runParallel(filePath, target, k, numThreads);
    }

private:

    /**
     * @brief Manages the full lock-free parallel execution.
     *        Initializes the atomic reference, spawns threads, joins, and reads final state.
     */
    std::string runParallel(const std::string& filePath, const Neighbor& target, int k, int numThreads) {
        std::vector<ChunkBounds> chunks = computeChunks(filePath, numThreads);
        if (chunks.empty()) {
            std::cerr << "Error computing file chunks." << std::endl;
            return "Unknown";
        }

        std::cout << "[runParallel] " << chunks.size()
                  << " chunks | target dim=" << target.values.size() << std::endl;

        std::atomic<std::shared_ptr<ImmutableTopK>> globalTopK(
            std::make_shared<ImmutableTopK>(k)
        );

        std::vector<std::thread> threads;
        threads.reserve(chunks.size());

        for (int i = 0; i < (int)chunks.size(); i++) {
            threads.emplace_back([this, &filePath, &chunks, &target, &globalTopK, i, k]() {
                processChunk(filePath, chunks[i], target, k, globalTopK);
            });
        }

        for (std::thread& t : threads) {
            if (t.joinable()) t.join();
        }

        auto finalResult = globalTopK.load();
        if (finalResult->list.empty()) {
            std::cerr << "Error: no valid neighbors found." << std::endl;
            return "Unknown";
        }

        return majorityVote(finalResult->list);
    }

    /**
     * @brief Processes a single chunk using fast-path and slow-path CAS loop.
     *
     *        FAST-PATH: pure volatile read — discards records that can't qualify
     *        without touching the CAS loop. Equivalent to the fast-path in Java.
     *
     *        SLOW-PATH: optimistic CAS loop — creates a new ImmutableTopK and
     *        atomically swaps the pointer only if state hasn't changed.
     *        Equivalent to the while(true) CAS loop in Java KNNAtomic.
     */
    void processChunk(
        const std::string& filePath,
        const ChunkBounds& chunk,
        const Neighbor& target,
        int k,
        std::atomic<std::shared_ptr<ImmutableTopK>>& globalTopK)
    {
        std::ifstream file(filePath, std::ios::binary);
        if (!file.is_open()) return;

        file.seekg(chunk.startByte);

        std::string line;
        while (file.tellg() < chunk.endByte && std::getline(file, line)) {
            if (line.empty()) continue;

            Neighbor* current = parseLineToNeighbor(line);
            if (current == nullptr) continue;

            if (current->values.size() != target.values.size()) {
                delete current;
                continue;
            }

            double dist = calculateEuclideanDistance(target, *current);

            {
                auto currentTopK = globalTopK.load(std::memory_order_acquire);
                if ((int)currentTopK->list.size() == k &&
                    dist >= currentTopK->list.back().distance) {
                    delete current;
                    continue;
                }
            }

            DistanceRecord record(*current, dist);
            delete current;

            while (true) {
                auto currentTopK = globalTopK.load(std::memory_order_acquire);

                auto nextTopK = currentTopK->tryUpdate(record);

                if (nextTopK == nullptr) {
                    break;
                }

                if (std::atomic_compare_exchange_strong_explicit(
                        &globalTopK,
                        &currentTopK,
                        nextTopK,
                        std::memory_order_release,
                        std::memory_order_acquire)) {
                    break;
                }
            }
        }
    }

    /**
     * @brief Calculates balanced byte chunks ensuring line alignment.
     */
    std::vector<ChunkBounds> computeChunks(const std::string& filePath, int numChunks) {
        std::vector<ChunkBounds> chunks;

        std::ifstream file(filePath, std::ios::binary | std::ios::ate);
        if (!file.is_open()) return chunks;

        long long fileSize = file.tellg();
        if (fileSize == 0) return chunks;

        file.seekg(0);
        std::string header;
        std::getline(file, header);

        long long dataStart = file.tellg();
        if (dataStart >= fileSize) return chunks;

        long long dataSize     = fileSize - dataStart;
        long long rawChunkSize = dataSize / numChunks;
        long long chunkStart   = dataStart;

        for (int i = 0; i < numChunks; i++) {
            long long chunkEnd;

            if (i == numChunks - 1) {
                chunkEnd = fileSize;
            } else {
                long long rawEnd = dataStart + (long long)(i + 1) * rawChunkSize;
                file.seekg(rawEnd);
                char c;
                while (file.get(c) && c != '\n') {}
                chunkEnd = file.eof() ? fileSize : (long long)file.tellg();
            }

            if (chunkStart < chunkEnd)
                chunks.push_back({chunkStart, chunkEnd});

            chunkStart = chunkEnd;
            if (chunkStart >= fileSize) break;
        }

        return chunks;
    }

    /**
     * @brief Parses a CSV line into a Neighbor object.
     */
    Neighbor* parseLineToNeighbor(const std::string& line) {
        std::vector<std::string> parts;
        std::stringstream ss(line);
        std::string token;

        while (std::getline(ss, token, ',')) {
            parts.push_back(token);
        }

        if (parts.size() < 2) return nullptr;

        std::vector<double> values;
        try {
            for (size_t i = 0; i < parts.size() - 1; i++) {
                std::string cleaned;
                for (char c : parts[i]) {
                    if (!std::isspace((unsigned char)c)) cleaned += c;
                }
                values.push_back(std::stod(cleaned));
            }

            std::string label = parts.back();
            label.erase(0, label.find_first_not_of(" \t\r\n"));
            label.erase(label.find_last_not_of(" \t\r\n") + 1);

            return new Neighbor(values, label);
        } catch (...) {
            return nullptr;
        }
    }

    /**
     * @brief Computes the Euclidean distance between two Neighbor points.
     */
    double calculateEuclideanDistance(const Neighbor& target, const Neighbor& dataPoint) {
        double sum = 0.0;
        for (size_t i = 0; i < target.values.size(); i++) {
            double diff = target.values[i] - dataPoint.values[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    /**
     * @brief Counts label frequencies and returns the majority label.
     */
    std::string majorityVote(const std::vector<DistanceRecord>& topK) {
        std::map<std::string, int> freq;
        for (const auto& r : topK) {
            freq[r.neighbor.label]++;
        }
        return std::max_element(
            freq.begin(), freq.end(),
            [](const std::pair<std::string, int>& a, const std::pair<std::string, int>& b) {
                return a.second < b.second;
            }
        )->first;
    }
};