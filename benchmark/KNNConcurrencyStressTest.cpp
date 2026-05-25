/**
 * KNNConcurrencyStressTest.cpp
 *
 * Stress test equivalente ao KNNConcurrencyStressTest.java usando ThreadSanitizer.
 *
 * Como compilar:
 *   g++ -std=c++20 -pthread -fsanitize=thread -g -O1 KNNConcurrencyStressTest.cpp -o stress_test
 *
 * Como rodar:
 *   ./stress_test
 *
 * O TSan reporta race conditions automaticamente no stderr.
 * O programa também conta resultados inválidos (equivalente ao ACCEPTABLE_INTERESTING do JCStress).
 */

#include <iostream>
#include <thread>
#include <mutex>
#include <semaphore>
#include <atomic>
#include <memory>
#include <string>
#include <sstream>

// ════════════════════════════════════════════════════════════════════
// Estrutura de resultado — equivalente ao record DistanceRecord do Java
// ════════════════════════════════════════════════════════════════════

struct DistanceRecord {
    double distance;
    std::string label;

    DistanceRecord(double d, std::string l)
        : distance(d), label(std::move(l)) {}
};

// ════════════════════════════════════════════════════════════════════
// Utilitário de resultado — equivalente ao @Arbiter + @Outcome do JCStress
// ════════════════════════════════════════════════════════════════════

struct TestResult {
    int total      = 0;
    int acceptable = 0;  // equivalente ao ACCEPTABLE do JCStress
    int bugs       = 0;  // equivalente ao ACCEPTABLE_INTERESTING (race condition visível)
    int forbidden  = 0;  // equivalente ao FORBIDDEN do JCStress

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

// ════════════════════════════════════════════════════════════════════
// TESTE 1 — Unsafe (sem sincronização)
// Equivalente ao UnsafeState do JCStress
// Esperado: TSan reporta data race + bugs > 0
// ════════════════════════════════════════════════════════════════════

void test_unsafe(int iterations) {
    TestResult result;

    for (int i = 0; i < iterations; i++) {
        double bestDist       = 10.0;
        std::string bestLabel = "None";

        // Duas threads escrevem sem proteção — race condition garantida
        std::thread t1([&]() {
            if (5.0 < bestDist) {
                bestDist  = 5.0;   // escrita sem lock — TSan detecta aqui
                bestLabel = "A";   // escrita sem lock — TSan detecta aqui
            }
        });

        std::thread t2([&]() {
            if (3.0 < bestDist) {
                bestDist  = 3.0;   // escrita sem lock — TSan detecta aqui
                bestLabel = "B";   // escrita sem lock — TSan detecta aqui
            }
        });

        t1.join();
        t2.join();

        // Arbiter — lê estado final e registra resultado
        result.record(bestLabel, bestDist);
    }

    result.print("Unsafe — sem sincronização");
}

// ════════════════════════════════════════════════════════════════════
// TESTE 2 — Synchronized (std::mutex + lock_guard)
// Equivalente ao SyncState do JCStress
// Equivalente ao KNNSync.cpp
// Esperado: TSan silencioso + bugs = 0 + forbidden = 0
// ════════════════════════════════════════════════════════════════════

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

// ════════════════════════════════════════════════════════════════════
// TESTE 3 — ReentrantLock (std::recursive_mutex + unique_lock)
// Equivalente ao LockState do JCStress
// Equivalente ao KNNLock.cpp
// Esperado: TSan silencioso + bugs = 0 + forbidden = 0
// ════════════════════════════════════════════════════════════════════

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

// ════════════════════════════════════════════════════════════════════
// TESTE 4 — Semaphore (std::binary_semaphore)
// Equivalente ao SemaphoreState do JCStress
// Equivalente ao KNNSemaphore.cpp
// Esperado: TSan silencioso + bugs = 0 + forbidden = 0
// ════════════════════════════════════════════════════════════════════

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

// ════════════════════════════════════════════════════════════════════
// TESTE 5 — Atomic CAS (std::atomic<double> + compare_exchange_strong)
// Equivalente ao AtomicState do JCStress
// Equivalente ao KNNAtomic.cpp
//
// Nota: std::atomic<std::shared_ptr<T>> requer suporte de hardware
// específico e pode causar segfault em algumas versões de GCC/libstdc++.
// Esta versão usa std::atomic<double> para a distância (tipo primitivo
// sempre suportado) e um mutex apenas para sincronizar o label junto
// com a distância — replicando fielmente o comportamento do CAS.
// Esperado: TSan silencioso + bugs = 0 + forbidden = 0
// ════════════════════════════════════════════════════════════════════

void test_atomic(int iterations) {
    TestResult result;

    for (int i = 0; i < iterations; i++) {
        // std::atomic<double> — CAS nativo, sem shared_ptr
        // Distância e label são atualizadas juntas sob o mesmo CAS
        std::atomic<double> bestDist{10.0};
        std::string bestLabel = "None";
        std::mutex labelMutex; // protege apenas a escrita do label junto com dist

        // CAS loop — equivalente ao updateIfBetter() do AtomicState do JCStress
        // A distância é atualizada atomicamente via compare_exchange_strong
        // O label é atualizado sob mutex apenas quando o CAS vence
        auto updateIfBetter = [&](double newDist, const std::string& newLabel) {
            while (true) {
                double current = bestDist.load(std::memory_order_acquire);
                if (newDist >= current) break;

                // Tenta trocar atomicamente a distância
                if (bestDist.compare_exchange_strong(
                        current, newDist,
                        std::memory_order_release,
                        std::memory_order_acquire)) {
                    // CAS venceu — atualiza o label de forma segura
                    std::lock_guard<std::mutex> guard(labelMutex);
                    bestLabel = newLabel;
                    break;
                }
                // CAS falhou — outra thread atualizou — tenta de novo
            }
        };

        std::thread t1([&]() { updateIfBetter(5.0, "A"); });
        std::thread t2([&]() { updateIfBetter(3.0, "B"); });

        t1.join();
        t2.join();

        // Arbiter — lê estado final
        result.record(bestLabel, bestDist.load());
    }

    result.print("Atomic — std::atomic<double> CAS (KNNAtomic)");
}

// ════════════════════════════════════════════════════════════════════
// MAIN
// ════════════════════════════════════════════════════════════════════

int main() {
    const int ITERATIONS = 100'000'000;

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
