#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <queue>
#include <cmath>
#include <map>
#include <thread>
#include <future>
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

    bool operator<(const DistanceRecord& other) const {
        return this->distance < other.distance;
    }
};

/**
 * @struct ChunkBounds
 * @brief Represents the byte boundaries (start and end) of a file segment.
 *        Each async task receives one ChunkBounds to know its read window.
 */
struct ChunkBounds {
    long long startByte;
    long long endByte;
};

/**
 * @class KNNCallableFuture
 * @brief Implements the K-Nearest Neighbors algorithm using std::async + std::future.
 *        This is the C++ counterpart of {@code executor.KNNCallableFuture} (Java): instead
 *        of submitting Callable tasks to a fixed ExecutorService and blocking on each
 *        Future via future.get(), every chunk here is launched with
 *        std::async(std::launch::async, ...), which returns a std::future<T> — the same
 *        "task in, handle out, block on demand" idiom, just without an explicit thread-pool
 *        object to manage (std::async owns the thread lifecycle itself). Just like the Java
 *        version, every future produces its own local Top-K priority_queue; the merge only
 *        happens after every future has been retrieved via .get(), so there is no shared
 *        mutable state and no mutex anywhere in the parallel phase.
 */
class KNNCallableFuture {
public:

    /**
     * @brief Entry point for the Callable/Future-based classification.
     *        Detects available hardware threads and delegates to runParallel().
     *
     * @param filePath  Path to the CSV dataset file.
     * @param target    The data point to classify.
     * @param k         Number of nearest neighbors to consider.
     * @return          Predicted class label, or "Unknown" if no valid data found.
     */
    std::string predictStream(const std::string& filePath, const Neighbor& target, int k) {
        int numThreads = std::max(2u, std::thread::hardware_concurrency());
        std::cout << "[CallableFuture] Using " << numThreads << " std::async workers" << std::endl;
        return runParallel(filePath, target, k, numThreads);
    }

private:

    /**
     * @brief Manages the full parallel execution lifecycle:
     *        compute chunks → launch one std::async per chunk → future.get() each one →
     *        merge results → majority vote.
     *
     * @param filePath    Path to the CSV dataset file.
     * @param target      The data point to classify.
     * @param k           Number of nearest neighbors to consider.
     * @param numThreads  Number of chunks/futures to deploy.
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

        int numChunks = (int)chunks.size();

        // One std::future per chunk — the direct analogue of the List<Future<...>>
        // returned by ExecutorService.invokeAll() in the Java version.
        std::vector<std::future<std::priority_queue<DistanceRecord>>> futures;
        futures.reserve(numChunks);

        for (int i = 0; i < numChunks; i++) {
            futures.push_back(std::async(std::launch::async,
                [this, &filePath, &chunks, &target, i, k]() {
                    return processChunk(filePath, chunks[i], target, k);
                }));
        }

        std::priority_queue<DistanceRecord> globalTopK;

        // future.get() blocks until that specific chunk's task completes and hands back
        // its local Top-K — exactly like Future<PriorityQueue<...>>.get() in Java.
        for (auto& future : futures) {
            std::priority_queue<DistanceRecord> localTopK = future.get();
            while (!localTopK.empty()) {
                const DistanceRecord& record = localTopK.top();
                if ((int)globalTopK.size() < k) {
                    globalTopK.push(record);
                } else if (record.distance < globalTopK.top().distance) {
                    globalTopK.pop();
                    globalTopK.push(record);
                }
                localTopK.pop();
            }
        }

        if (globalTopK.empty()) {
            std::cerr << "Error: no valid neighbors found." << std::endl;
            return "Unknown";
        }

        return majorityVote(globalTopK);
    }

    /**
     * @brief Calculates balanced byte chunks across the file, ensuring line alignment.
     *        Seeks '\n' boundaries so no line is split between two tasks.
     *        Identical to KNNPlatform.cpp's computeChunks().
     *
     * @param filePath   Path to the CSV file.
     * @param numChunks  Desired number of partitions.
     * @return           Vector of ChunkBounds, one per async task.
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
                chunkEnd = (file.eof()) ? fileSize : (long long)file.tellg();
            }

            if (chunkStart < chunkEnd)
                chunks.push_back({chunkStart, chunkEnd});

            chunkStart = chunkEnd;
            if (chunkStart >= fileSize) break;
        }

        return chunks;
    }

    /**
     * @brief Processes a single file chunk assigned to one std::async task.
     *        Opens the file independently, seeks to startByte, reads until endByte,
     *        parses each line, and maintains a local max-heap of size K.
     *        Identical to KNNPlatform.cpp's processChunk().
     *
     * @param filePath  Path to the CSV file.
     * @param chunk     Byte boundaries for this task's work window.
     * @param target    The data point to classify.
     * @param k         Number of nearest neighbors to maintain locally.
     * @return          Local max-heap with the K nearest neighbors found in this chunk.
     */
    std::priority_queue<DistanceRecord> processChunk(
        const std::string& filePath,
        const ChunkBounds& chunk,
        const Neighbor& target,
        int k)
    {
        std::priority_queue<DistanceRecord> localTopK;

        std::ifstream file(filePath, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error opening file in task." << std::endl;
            return localTopK;
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

            double dist = calculateEuclideanDistance(target, *current);

            if ((int)localTopK.size() < k) {
                localTopK.push(DistanceRecord(*current, dist));
            } else if (dist < localTopK.top().distance) {
                localTopK.pop();
                localTopK.push(DistanceRecord(*current, dist));
            }

            delete current;
        }

        return localTopK;
    }

    /**
     * @brief Parses a CSV line into a Neighbor object.
     *        All columns except the last are feature values; the last is the label.
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
     *        Identical to KNNPlatform.cpp's majorityVote().
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
