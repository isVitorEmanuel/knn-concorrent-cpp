#include <iostream>
#include <thread>
#include <mutex>
#include <semaphore>
#include <atomic>
#include <memory>
#include <string>
#include <sstream>

struct DistanceRecord {
    double distance;
    std::string label;

    DistanceRecord(double d, std::string l)
        : distance(d), label(std::move(l)) {}
};

struct TestResult {
    int total      = 0;
    int acceptable = 0;
    int bugs       = 0;
    int forbidden  = 0;

    void record(const std::string& label, double dist) {
        total++;
        std::string outcome = label + ":" + std::to_string(dist).substr(0, 3);

        bool isAcceptable = (label == "B" && dist == 3.0) ||
                            (label == "A" && dist == 5.0);
        bool isBug        = (label == "A" && dist == 3.0) ||
                            (label == "B" && dist == 5.0);

        if (isAcceptable) acceptable++;
        else if (isBug)   bugs++;
        else              forbidden++;
    }

    void print(const std::string& testName) const {
        std::cout << "\n[" << testName << "]\n";
        std::cout << "  Total iterações : " << total      << "\n";
        std::cout << "  ACCEPTABLE      : " << acceptable << "\n";
        std::cout << "  INTERESTING/BUG : " << bugs       << " ← race condition visível\n";
        std::cout << "  FORBIDDEN       : " << forbidden  << "\n";
        std::cout << "  Status          : "
                  << (bugs > 0 || forbidden > 0 ? "⚠ RACE CONDITION DETECTADA" : "✓ OK")
                  << "\n";
    }
};

void test_unsafe(int iterations) {
    TestResult result;

    for (int i = 0; i < iterations; i++) {
        double bestDist       = 10.0;
        std::string bestLabel = "None";

        std::thread t1([&]() {
            if (5.0 < bestDist) {
                bestDist  = 5.0;
                bestLabel = "A";
            }
        });

        std::thread t2([&]() {
            if (3.0 < bestDist) {
                bestDist  = 3.0;
                bestLabel = "B";
            }
        });

        t1.join();
        t2.join();

        result.record(bestLabel, bestDist);
    }

    result.print("Unsafe — sem sincronização");
}

void test_sync(int iterations) {
    TestResult result;

    for (int i = 0; i < iterations; i++) {
        double bestDist       = 10.0;
        std::string bestLabel = "None";
        std::mutex lock;

        std::thread t1([&]() {
            std::lock_guard<std::mutex> guard(lock);
            if (5.0 < bestDist) { bestDist = 5.0; bestLabel = "A"; }
        });

        std::thread t2([&]() {
            std::lock_guard<std::mutex> guard(lock);
            if (3.0 < bestDist) { bestDist = 3.0; bestLabel = "B"; }
        });

        t1.join();
        t2.join();

        result.record(bestLabel, bestDist);
    }

    result.print("Sync — std::mutex + lock_guard (KNNSync)");
}

void test_lock(int iterations) {
    TestResult result;

    for (int i = 0; i < iterations; i++) {
        double bestDist       = 10.0;
        std::string bestLabel = "None";
        std::recursive_mutex lock;

        std::thread t1([&]() {
            std::unique_lock<std::recursive_mutex> guard(lock);
            if (5.0 < bestDist) { bestDist = 5.0; bestLabel = "A"; }
        });

        std::thread t2([&]() {
            std::unique_lock<std::recursive_mutex> guard(lock);
            if (3.0 < bestDist) { bestDist = 3.0; bestLabel = "B"; }
        });

        t1.join();
        t2.join();

        result.record(bestLabel, bestDist);
    }

    result.print("Lock — std::recursive_mutex + unique_lock (KNNLock)");
}

void test_semaphore(int iterations) {
    TestResult result;

    for (int i = 0; i < iterations; i++) {
        double bestDist       = 10.0;
        std::string bestLabel = "None";
        std::binary_semaphore mutex(1);

        std::thread t1([&]() {
            mutex.acquire();
            if (5.0 < bestDist) { bestDist = 5.0; bestLabel = "A"; }
            mutex.release();
        });

        std::thread t2([&]() {
            mutex.acquire();
            if (3.0 < bestDist) { bestDist = 3.0; bestLabel = "B"; }
            mutex.release();
        });

        t1.join();
        t2.join();

        result.record(bestLabel, bestDist);
    }

    result.print("Semaphore — std::binary_semaphore (KNNSemaphore)");
}

void test_atomic(int iterations) {
    TestResult result;

    for (int i = 0; i < iterations; i++) {
        std::atomic<double> bestDist{10.0};
        std::string bestLabel = "None";
        std::mutex labelMutex;

        auto updateIfBetter = [&](double newDist, const std::string& newLabel) {
            while (true) {
                double current = bestDist.load(std::memory_order_acquire);
                if (newDist >= current) break;

                if (bestDist.compare_exchange_strong(
                        current, newDist,
                        std::memory_order_release,
                        std::memory_order_acquire)) {
                    std::lock_guard<std::mutex> guard(labelMutex);
                    bestLabel = newLabel;
                    break;
                }
            }
        };

        std::thread t1([&]() { updateIfBetter(5.0, "A"); });
        std::thread t2([&]() { updateIfBetter(3.0, "B"); });

        t1.join();
        t2.join();

        result.record(bestLabel, bestDist.load());
    }

    result.print("Atomic — std::atomic<double> CAS (KNNAtomic)");
}

int main() {
    const int ITERATIONS = 1'000'000;

    std::cout << "════════════════════════════════════════════\n";
    std::cout << " KNN Concurrency Stress Test — C++ + TSan  \n";
    std::cout << " Iterações por teste: " << ITERATIONS       << "\n";
    std::cout << "════════════════════════════════════════════\n";

    test_unsafe   (ITERATIONS);
    test_sync     (ITERATIONS);
    test_lock     (ITERATIONS);
    test_semaphore(ITERATIONS);
    test_atomic   (ITERATIONS);

    std::cout << "\n════════════════════════════════════════════\n";
    std::cout << " Verificar saída do TSan no stderr para     \n";
    std::cout << " relatório detalhado de data races.         \n";
    std::cout << "════════════════════════════════════════════\n";

    return 0;
}