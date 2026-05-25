#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <queue>
#include <cmath>
#include <map>
#include <thread>
#include <semaphore>   // std::counting_semaphore — C++20
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
 * @class KNNSemaphore
 * @brief Implements KNN using parallel platform threads with a binary semaphore
 *        protecting only the final merge — one lock per thread, not per line.
 *        Equivalent to the Java KNNSemaphore using java.util.concurrent.Semaphore.
 */
class KNNSemaphore {
public:

    /**
     * @brief Entry point for the semaphore-based parallel classification.
     *
     * @param filePath  Path to the CSV dataset file.
     * @param target    The data point to classify.
     * @param k         Number of nearest neighbors to consider.
     * @return          Predicted class label, or "Unknown" if no valid data found.
     */
    std::string predictStream(const std::string& filePath, const Neighbor& target, int k) {
        int numThreads = std::max(2u, std::thread::hardware_concurrency());
        std::cout << "[Semaphore] Using " << numThreads << " platform threads" << std::endl;
        return runParallel(filePath, target, k, numThreads);
    }

private:

    /**
     * @brief Manages the full parallel execution lifecycle.
     *        Each thread builds a local heap freely, then acquires the semaphore
     *        only once to merge into the global heap — minimizing contention.
     *
     * @param filePath    Path to the CSV dataset file.
     * @param target      The data point to classify.
     * @param k           Number of nearest neighbors to consider.
     * @param numThreads  Number of platform threads to deploy.
     * @return            Predicted class label, or "Unknown" on failure.
     */
    std::string runParallel(const std::string& filePath, const Neighbor& target, int k, int numThreads) {
        std::vector<ChunkBounds> chunks = computeChunks(filePath, numThreads);
        if (chunks.empty()) {
            std::cerr << "Error computing file chunks." << std::endl;
            return "Unknown";
        }

        // Shared global heap — written only during the final merge per thread
        std::priority_queue<DistanceRecord> globalTopK;

        // Binary semaphore with initial value 1 — equivalent to new Semaphore(1) in Java
        // std::binary_semaphore is an alias for std::counting_semaphore<1>
        std::binary_semaphore mutex(1);

        std::vector<std::thread> threads;
        threads.reserve(chunks.size());

        for (int i = 0; i < (int)chunks.size(); i++) {
            threads.emplace_back([this, &filePath, &chunks, &target, &globalTopK, &mutex, i, k]() {
                processChunk(filePath, chunks[i], target, k, globalTopK, mutex);
            });
        }

        for (std::thread& t : threads) {
            if (t.joinable()) t.join();
        }

        if (globalTopK.empty()) {
            std::cerr << "Error: no valid neighbors found." << std::endl;
            return "Unknown";
        }

        return majorityVote(globalTopK);
    }

    /**
     * @brief Calculates balanced byte chunks across the file, ensuring line alignment.
     *
     * @param filePath   Path to the CSV file.
     * @param numChunks  Desired number of partitions.
     * @return           Vector of ChunkBounds, one per thread.
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
     * @brief Processes a single file chunk:
     *        1) Builds a local top-K heap freely — no contention
     *        2) Acquires the semaphore ONCE to merge local into global
     *        3) Releases the semaphore immediately after merge
     *        Equivalent to the acquire/release pattern in Java KNNSemaphore.
     *
     * @param filePath   Path to the CSV file.
     * @param chunk      Byte boundaries for this thread's work window.
     * @param target     The data point to classify.
     * @param k          Number of nearest neighbors to maintain.
     * @param globalTopK Shared max-heap — written only during merge.
     * @param mutex      Binary semaphore protecting the merge section.
     */
    void processChunk(
        const std::string& filePath,
        const ChunkBounds& chunk,
        const Neighbor& target,
        int k,
        std::priority_queue<DistanceRecord>& globalTopK,
        std::binary_semaphore& mutex)
    {
        // Local heap — built entirely without any lock
        // Equivalent to localTopK in Java KNNSemaphore
        std::priority_queue<DistanceRecord> localTopK;

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
            DistanceRecord record(*current, dist);
            delete current;

            if ((int)localTopK.size() < k) {
                localTopK.push(record);
            } else if (dist < localTopK.top().distance) {
                localTopK.pop();
                localTopK.push(record);
            }
        }

        // Acquire semaphore — equivalent to mutex.acquire() in Java
        // Blocks if another thread is currently merging
        mutex.acquire();

        // Critical section: merge local into global — one lock per thread total
        // Equivalent to the synchronized merge block in Java KNNSemaphore
        std::priority_queue<DistanceRecord> copy = localTopK;
        while (!copy.empty()) {
            const DistanceRecord& record = copy.top();
            if ((int)globalTopK.size() < k) {
                globalTopK.push(record);
            } else if (record.distance < globalTopK.top().distance) {
                globalTopK.pop();
                globalTopK.push(record);
            }
            copy.pop();
        }

        // Release semaphore — equivalent to mutex.release() in Java
        // In C++ any thread can release — same as Java Semaphore (unlike std::mutex)
        mutex.release();
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
    std::string majorityVote(std::priority_queue<DistanceRecord>& topK) {
        std::map<std::string, int> freq;
        while (!topK.empty()) {
            freq[topK.top().neighbor.label]++;
            topK.pop();
        }
        return std::max_element(
            freq.begin(), freq.end(),
            [](const std::pair<std::string, int>& a, const std::pair<std::string, int>& b) {
                return a.second < b.second;
            }
        )->first;
    }
};
