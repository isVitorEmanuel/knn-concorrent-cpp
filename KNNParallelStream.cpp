#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <queue>
#include <cmath>
#include <map>
#include <thread>
#include <execution>
#include <numeric>
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
 *        Each parallel-algorithm element receives one ChunkBounds to know its read window.
 */
struct ChunkBounds {
    long long startByte;
    long long endByte;
};

/**
 * @class KNNParallelStream
 * @brief Implements the K-Nearest Neighbors algorithm using the C++17 Parallel Algorithms
 *        (<execution>, std::execution::par). This is the C++ counterpart of
 *        {@code executor.KNNParallelStream} (Java): instead of
 *        chunks.parallelStream().map(...).reduce(...), this version uses
 *        std::transform_reduce(std::execution::par, ...) — map() becomes the "unary"
 *        transform argument (turns a ChunkBounds into its local Top-K), and reduce()
 *        becomes the associative "binary" combiner (merges two local Top-Ks into one),
 *        exactly the same divide-and-combine shape as the Java version. No std::thread,
 *        no thread pool, and no mutex is ever referenced explicitly — the standard
 *        library's own parallel execution engine decides how to distribute the work.
 * <p>
 *        IMPORTANT PORTABILITY NOTE: unlike Java's parallelStream() (which always runs on
 *        the common ForkJoinPool out of the box), GCC/libstdc++'s implementation of Parallel
 *        Algorithms requires linking against Intel TBB (`-ltbb`) to actually execute in
 *        parallel — without it, the code compiles and runs correctly, but sequentially,
 *        silently. MSVC and recent libc++ (Clang) ship a parallel backend without any extra
 *        dependency. Always verify with a wall-clock/thread-count check on your target
 *        compiler before treating this as "real" parallelism.
 */
class KNNParallelStream {
public:

    /**
     * @brief Entry point for the Parallel-Algorithms-based classification.
     *        Detects available hardware threads (used only to decide how many chunks to
     *        create; the actual parallel execution is delegated to the standard library)
     *        and delegates to runParallel().
     *
     * @param filePath  Path to the CSV dataset file.
     * @param target    The data point to classify.
     * @param k         Number of nearest neighbors to consider.
     * @return          Predicted class label, or "Unknown" if no valid data found.
     */
    std::string predictStream(const std::string& filePath, const Neighbor& target, int k) {
        int numChunks = std::max(2u, std::thread::hardware_concurrency());
        std::cout << "[ParallelStream] Using std::execution::par over " << numChunks << " chunks" << std::endl;
        return runParallel(filePath, target, k, numChunks);
    }

private:

    /**
     * @brief Manages the full parallel execution lifecycle:
     *        compute chunks → std::transform_reduce(par) does map+merge in one pass →
     *        majority vote.
     *
     * @param filePath   Path to the CSV dataset file.
     * @param target     The data point to classify.
     * @param k          Number of nearest neighbors to consider.
     * @param numChunks  Number of chunks to partition the file into.
     * @return           Predicted class label, or "Unknown" on failure.
     */
    std::string runParallel(const std::string& filePath, const Neighbor& target, int k, int numChunks) {
        std::vector<ChunkBounds> chunks = computeChunks(filePath, numChunks);
        if (chunks.empty()) {
            std::cerr << "Error computing file chunks." << std::endl;
            return "Unknown";
        }

        std::cout << "[runParallel] " << chunks.size()
                  << " chunks | target dim=" << target.values.size() << std::endl;

        std::priority_queue<DistanceRecord> emptyTopK;

        // transform_reduce(par, first, last, init, binaryReduce, unaryTransform):
        //   unaryTransform  == map():    ChunkBounds -> local Top-K for that chunk
        //   binaryReduce    == reduce(): merges two local Top-Ks into one, associatively
        std::priority_queue<DistanceRecord> globalTopK = std::transform_reduce(
            std::execution::par,
            chunks.begin(), chunks.end(),
            emptyTopK,
            [this, k](std::priority_queue<DistanceRecord> a, std::priority_queue<DistanceRecord> b) {
                return mergeTopK(a, b, k);
            },
            [this, &filePath, &target, k](const ChunkBounds& chunk) {
                return processChunk(filePath, chunk, target, k);
            });

        if (globalTopK.empty()) {
            std::cerr << "Error: no valid neighbors found." << std::endl;
            return "Unknown";
        }

        return majorityVote(globalTopK);
    }

    /**
     * @brief Combines two local Top-K max-heaps into a single Top-K heap of size k.
     *        Used as the associative binary combiner passed to transform_reduce: every
     *        pairwise merge is independent of merge order, which is exactly what makes it
     *        safe for the parallel algorithm's divide-and-combine execution. Identical in
     *        spirit to KNNParallelStream.java's mergeTopK().
     *
     * @param a  First local Top-K queue.
     * @param b  Second local Top-K queue.
     * @param k  Number of nearest neighbors to keep.
     * @return   A single merged max-heap containing the closest k records.
     */
    std::priority_queue<DistanceRecord> mergeTopK(
        std::priority_queue<DistanceRecord> a,
        std::priority_queue<DistanceRecord> b,
        int k)
    {
        std::priority_queue<DistanceRecord> merged;

        for (std::priority_queue<DistanceRecord>* source : {&a, &b}) {
            while (!source->empty()) {
                const DistanceRecord& record = source->top();
                if ((int)merged.size() < k) {
                    merged.push(record);
                } else if (record.distance < merged.top().distance) {
                    merged.pop();
                    merged.push(record);
                }
                source->pop();
            }
        }

        return merged;
    }

    /**
     * @brief Calculates balanced byte chunks across the file, ensuring line alignment.
     *        Seeks '\n' boundaries so no line is split between two chunks.
     *        Identical to KNNPlatform.cpp's computeChunks().
     *
     * @param filePath   Path to the CSV file.
     * @param numChunks  Desired number of partitions.
     * @return           Vector of ChunkBounds, one per parallel-algorithm element.
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
     * @brief Processes a single file chunk, invoked once per chunk from within
     *        transform_reduce's unary transform stage.
     *        Opens the file independently, seeks to startByte, reads until endByte,
     *        parses each line, and maintains a local max-heap of size K.
     *        Identical to KNNPlatform.cpp's processChunk().
     *
     * @param filePath  Path to the CSV file.
     * @param chunk     Byte boundaries for this chunk's work window.
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
            std::cerr << "Error opening file in worker." << std::endl;
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

