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
 * @class KNNLock
 * @brief Implements KNN using parallel platform threads and a std::mutex with
 *        explicit lock/unlock via std::unique_lock — equivalent to ReentrantLock in Java.
 *        Protects the shared global heap at every line insertion.
 */
class KNNLock {
public:

    /**
     * @brief Entry point for the lock-based parallel classification.
     *
     * @param filePath  Path to the CSV dataset file.
     * @param target    The data point to classify.
     * @param k         Number of nearest neighbors to consider.
     * @return          Predicted class label, or "Unknown" if no valid data found.
     */
    std::string predictStream(const std::string& filePath, const Neighbor& target, int k) {
        int numThreads = std::max(2u, std::thread::hardware_concurrency());
        std::cout << "[ReentrantLock] Using " << numThreads
                  << " platform threads with shared global queue synchronization" << std::endl;
        return runParallel(filePath, target, k, numThreads);
    }

private:

    /**
     * @brief Manages the full parallel execution lifecycle.
     *        Creates a shared global heap and a mutex, spawns threads,
     *        joins all, and returns the majority vote.
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

        // Shared global heap — protected by lock at every insertion
        // Equivalent to PriorityQueue<DistanceRecord> globalTopK in Java KNNLock
        std::priority_queue<DistanceRecord> globalTopK;

        // Explicit mutex object — equivalent to new ReentrantLock() in Java
        // Note: std::mutex is NOT reentrant by default — see notes below
        std::mutex lock;

        std::vector<std::thread> threads;
        threads.reserve(chunks.size());

        for (int i = 0; i < (int)chunks.size(); i++) {
            threads.emplace_back([this, &filePath, &chunks, &target, &globalTopK, &lock, i, k]() {
                processChunk(filePath, chunks[i], target, k, globalTopK, lock);
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
     * @brief Processes a single file chunk.
     *        Parse and distance are done outside the lock.
     *        Only the heap insertion is protected — same as Java KNNLock.
     *        Uses std::unique_lock for explicit lock/unlock with RAII safety.
     *
     * @param filePath   Path to the CSV file.
     * @param chunk      Byte boundaries for this thread.
     * @param target     The data point to classify.
     * @param k          Number of nearest neighbors to maintain.
     * @param globalTopK Shared max-heap — written under lock.
     * @param lock       Mutex protecting globalTopK.
     */
    void processChunk(
        const std::string& filePath,
        const ChunkBounds& chunk,
        const Neighbor& target,
        int k,
        std::priority_queue<DistanceRecord>& globalTopK,
        std::mutex& lock)
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

            // Parse and distance done OUTSIDE the lock — no contention on CPU work
            double dist = calculateEuclideanDistance(target, *current);
            DistanceRecord record(*current, dist);
            delete current;

            // Explicit lock — equivalent to lock.lock() in Java ReentrantLock
            // std::unique_lock gives RAII safety: if an exception occurs,
            // the destructor calls unlock() automatically — same as Java's finally block
            {
                std::unique_lock<std::mutex> guard(lock);

                // Critical section — equivalent to try { } in Java KNNLock
                if ((int)globalTopK.size() < k) {
                    globalTopK.push(record);
                } else if (dist < globalTopK.top().distance) {
                    globalTopK.pop();
                    globalTopK.push(record);
                }

            } // guard destructor calls unlock() here — equivalent to finally { lock.unlock(); }
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
