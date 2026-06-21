#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <queue>
#include <cmath>
#include <map>
#include <stdexcept>
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
 * @class KNNSerial
 * @brief Implements the K-Nearest Neighbors algorithm in a serial (single-threaded) manner.
 *        (CPU-bound naive implementation for baseline comparison).
 */
class KNNSerial {
public:

    /**
     * @brief Reads the dataset line by line from a CSV file, computes distances
     *        to the target, maintains a max-heap of size K, and returns
     *        the majority label among the K nearest neighbors.
     *
     * @param filePath  Path to the CSV dataset file.
     * @param target    The data point to classify.
     * @param k         Number of nearest neighbors to consider.
     * @return          Predicted class label, or "Unknown" if no valid data found.
     */
    std::string predictStream(const std::string& filePath, const Neighbor& target, int k) {
        std::priority_queue<DistanceRecord> pq;

        std::ifstream file(filePath);
        if (!file.is_open()) {
            std::cerr << "error reading file: " << filePath << std::endl;
            return "Unknown";
        }

        std::string line;
        bool firstLine = true;

        while (std::getline(file, line)) {
            if (firstLine) {
                firstLine = false;
                continue;
            }

            Neighbor* current = parseLineToNeighbor(line);
            if (current == nullptr) continue;

            if (current->values.size() != target.values.size()) {
                delete current;
                continue;
            }

            double dist = calculateEuclideanDistance(target, *current);
            pq.push(DistanceRecord(*current, dist));
            delete current;

            if ((int)pq.size() > k) {
                pq.pop();
            }
        }

        file.close();

        if (pq.empty()) return "Unknown";

        std::map<std::string, int> labelFrequencies;
        while (!pq.empty()) {
            const std::string& label = pq.top().neighbor.label;
            labelFrequencies[label]++;
            pq.pop();
        }

        return std::max_element(
            labelFrequencies.begin(),
            labelFrequencies.end(),
            [](const std::pair<std::string, int>& a, const std::pair<std::string, int>& b) {
                return a.second < b.second;
            }
        )->first;
    }

private:

    /**
     * @brief Parses a CSV line into a Neighbor object.
     *        All columns except the last are feature values; the last is the label.
     *
     * @param line  A single line from the CSV file.
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
                    if (!std::isspace(c)) cleaned += c;
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
};