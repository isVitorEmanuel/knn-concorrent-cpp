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
 *        Used inside the max-heap priority queue to maintain the K nearest neighbors.
 */
struct DistanceRecord {
    Neighbor neighbor;
    double distance;

    DistanceRecord(const Neighbor& neighbor, double distance)
        : neighbor(neighbor), distance(distance) {}

    // Max-heap: largest distance has highest priority (to be removed first)
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
 * @class KNNSync
 * @brief Implements the K-Nearest Neighbors algorithm using parallel platform threads
 *        and a shared global priority queue protected via std::mutex.
 *        Equivalent to the Java KNNSync using synchronized blocks.
 */
class KNNSync {
public:

    /**
     * @brief Entry point for the synchronized parallel classification.
     *        Detects available hardware threads and delegates to runParallel().
     *
     * @param filePath  Path to the CSV dataset file.
     * @param target    The data point to classify.
     * @param k         Number of nearest neighbors to consider.
     * @return          Predicted class label, or "Unknown" if no valid data found.
     */
    std::string predictStream(const std::string& filePath, const Neighbor& target, int k) {
        int numThreads = std::max(2u, std::thread::hardware_concurrency());
        std::cout << "[Synchronized] Using " << numThreads
                  << " platform threads with shared global queue" << std::endl;
        return runParallel(filePath, target, k, numThreads);
    }

private:

    /**
     * @brief Manages the full parallel execution lifecycle:
     *        compute chunks → spawn threads → join → majority vote.
     *        Unlike KNNPlatform, there is no merge step — the global heap
     *        is built incrementally by all threads under mutex protection.
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

        std::cout << "[runParallel] " << chunks.size()
                  << " chunks | target dim=" << target.values.size() << std::endl;

        // Shared global heap — all threads write here under mutex protection
        // Equivalent to the shared PriorityQueue in Java KNNSync
        std::priority_queue<DistanceRecord> globalTopK;

        // Mutex protecting globalTopK — equivalent to synchronized(globalTopK) in Java
        std::mutex globalMutex;

        std::vector<std::thread> threads;
        threads.reserve(chunks.size());

        for (int i = 0; i < (int)chunks.size(); i++) {
            threads.emplace_back([this, &filePath, &chunks, &target, &globalTopK, &globalMutex, i, k]() {
                processChunk(filePath, chunks[i], target, k, globalTopK, globalMutex);
            });
        }

        // Wait for all threads to finish
        for (std::thread& t : threads) {
            if (t.joinable()) t.join();
        }

        if (globalTopK.empty()) {
            std::cerr << "Error: no valid neighbors found." << std::endl;
            return "Unknown";
        }

        // No merge needed — globalTopK already contains the final K nearest neighbors
        return majorityVote(globalTopK);
    }

    /**
     * @brief Calculates balanced byte chunks across the file, ensuring line alignment.
     *        Seeks '\n' boundaries so no line is split between two threads.
     *
     * @param filePath   Path to the CSV file.
     * @param numChunks  Desired number of partitions.
     * @return           Vector of ChunkBounds, one per thread.
     */
    std::vector<ChunkBounds> computeChunks(const std::string& filePath, int numChunks) {
        std::vector<ChunkBounds> chunks;

        std::ifstream file(filePath, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filePath << std::endl;
            return chunks;
        }

        long long fileSize = file.tellg();
        if (fileSize == 0) {
            std::cerr << "File is empty: " << filePath << std::endl;
            return chunks;
        }

        file.seekg(0);
        std::string header;
        std::getline(file, header);

        long long dataStart = file.tellg();
        if (dataStart >= fileSize) {
            std::cerr << "File contains only the header — no data." << std::endl;
            return chunks;
        }

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
     * @brief Processes a single file chunk assigned to one thread.
     *        Parse and distance calculation are done outside the lock — only the
     *        heap insertion is protected, minimizing contention.
     *        Equivalent to the synchronized(globalTopK) block in Java KNNSync.
     *
     * @param filePath     Path to the CSV file.
     * @param chunk        Byte boundaries for this thread's work window.
     * @param target       The data point to classify.
     * @param k            Number of nearest neighbors to maintain globally.
     * @param globalTopK   Shared max-heap — written under mutex protection.
     * @param globalMutex  Mutex protecting globalTopK.
     */
    void processChunk(
        const std::string& filePath,
        const ChunkBounds& chunk,
        const Neighbor& target,
        int k,
        std::priority_queue<DistanceRecord>& globalTopK,
        std::mutex& globalMutex)
    {
        std::ifstream file(filePath, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error opening file in thread." << std::endl;
            return;
        }

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

            // Parse and distance are CPU work — done OUTSIDE the lock
            // Only the heap insertion is protected — same strategy as Java KNNSync
            double dist = calculateEuclideanDistance(target, *current);
            DistanceRecord record(*current, dist);
            delete current;

            // Equivalent to synchronized(globalTopK) { ... } in Java
            {
                std::lock_guard<std::mutex> lock(globalMutex);
                if ((int)globalTopK.size() < k) {
                    globalTopK.push(record);
                } else if (dist < globalTopK.top().distance) {
                    globalTopK.pop();
                    globalTopK.push(record);
                }
            } // lock released automatically here — std::lock_guard RAII
        }
    }

    /**
     * @brief Parses a CSV line into a Neighbor object.
     *
     * @param line  A single CSV line.
     * @return      Pointer to a new Neighbor, or nullptr if parsing fails.
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
     *
     * @param target     The query point.
     * @param dataPoint  A point from the training dataset.
     * @return           Euclidean distance as a double.
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
     * @brief Counts label frequencies in the global top-K heap and returns the majority label.
     *
     * @param topK  Max-heap containing the K nearest neighbors globally.
     * @return      The label with the highest frequency.
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
