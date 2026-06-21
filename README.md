# Guia de Execução — Stress Test e Benchmark (C++)

Este guia cobre os dois utilitários de validação de concorrência das implementações KNN em C++:

- `KNNConcurrencyStressTest.cpp` — detecção de race conditions via **ThreadSanitizer (TSan)**
- `KNNBenchmark.cpp` — medição de desempenho via **Google Benchmark**

---

## 1. Pré-requisitos

- Compilador com suporte a **C++20** (necessário para `std::barrier` e `std::counting_semaphore`)
- `g++` ≥ 11 ou `clang++` ≥ 14
- `pthread` disponível no sistema (padrão em distribuições Linux)

Verifique a versão do compilador:

```bash
g++ --version
```

---

## 2. Stress Test (ThreadSanitizer)

### O que faz

Executa 1.000.000 de iterações de cada estratégia de sincronização (Unsafe, Sync, Lock, Semaphore, Atomic) em um cenário simplificado de "duas threads competindo para atualizar o melhor resultado". O TSan instrumenta o binário e reporta **data races** diretamente no `stderr` durante a execução.

A versão *Unsafe* é esperada **falhar** (data race detectada) — ela serve como controle negativo para confirmar que o TSan está funcionando corretamente.

### Compilar

```bash
g++ -std=c++20 -pthread -fsanitize=thread -g -O1 KNNConcurrencyStressTest.cpp -o stress_test
```

| Flag | Motivo |
|---|---|
| `-std=c++20` | necessário para `std::binary_semaphore` |
| `-pthread` | suporte a `std::thread` |
| `-fsanitize=thread` | ativa o ThreadSanitizer |
| `-g` | símbolos de debug (stack traces legíveis no relatório do TSan) |
| `-O1` | otimização leve recomendada pelo próprio TSan (evita ruído de `-O0` e falsos negativos de `-O2`+) |

### Rodar

```bash
./stress_test
```

### Interpretando a saída

- No `stdout`: um resumo por teste com contagem de `ACCEPTABLE`, `INTERESTING/BUG` e `FORBIDDEN`.
- No `stderr`: relatórios do TSan (se houver) no formato `WARNING: ThreadSanitizer: data race ...`, com stack trace de ambas as threads envolvidas.

**Resultado esperado:**

| Teste | TSan | bugs/forbidden |
|---|---|---|
| Unsafe | reporta data race | `bugs > 0` |
| Sync | silencioso | `0` |
| Lock | silencioso | `0` |
| Semaphore | silencioso | `0` |
| Atomic | silencioso | `0` |

Se qualquer uma das versões sincronizadas (Sync, Lock, Semaphore, Atomic) apresentar data race reportada pelo TSan, há um bug real de concorrência na implementação correspondente.

### Salvando o relatório em arquivo

```bash
./stress_test 2> tsan_report.txt
```

---

## 3. Benchmark (Google Benchmark)

### O que faz

Mede o tempo de execução de cada implementação (`Serial`, `Platform`, `Synchronized`, `Lock`, `Semaphore`, `Barrier`, `Atomic`) rodando uma consulta KNN completa sobre o dataset `dataset_high_dim.csv`, com `K = 21` vizinhos.

> **Importante:** o arquivo `dataset_high_dim.csv` precisa estar no mesmo diretório de execução do binário, já que o caminho está fixado no código (`FILE_PATH`).

### Instalar a Google Benchmark

```bash
sudo apt update
sudo apt install libbenchmark-dev
```

### Compilar

```bash
g++ -std=c++20 -pthread -O2 KNNBenchmark.cpp -o knn_benchmark -lbenchmark -lpthread
```

| Flag | Motivo |
|---|---|
| `-O2` | otimização de produção — necessária para números de desempenho realistas |
| `-lbenchmark` | linka a biblioteca Google Benchmark |
| `-lpthread` | suporte a `std::thread` |

### Rodar (modo equivalente ao JMH: warmup 1×2s + 5 repetições)

```bash
./knn_benchmark \
  --benchmark_repetitions=5 \
  --benchmark_min_warmup_time=2 \
  --benchmark_report_aggregates_only=true
```

### Salvar resultados em JSON (equivalente ao `-rf json` do JMH)

```bash
./knn_benchmark \
  --benchmark_repetitions=5 \
  --benchmark_min_warmup_time=2 \
  --benchmark_out=results.json \
  --benchmark_out_format=json
```

### Rodar um benchmark específico

Útil para iterar rapidamente em uma única implementação:

```bash
./knn_benchmark --benchmark_filter=test07_Semaphore
```

### Outras flags úteis

| Flag | Efeito |
|---|---|
| `--benchmark_list_tests` | lista os nomes de todos os benchmarks sem executá-los |
| `--benchmark_min_time=5s` | tempo mínimo de medição por execução (alternativa a `--benchmark_repetitions`) |
| `--benchmark_format=csv` | saída em CSV no stdout |

---

## 4. Ordem recomendada de execução

1. Rodar o **stress test** primeiro para confirmar que não há data races nas versões sincronizadas.
2. Só então rodar o **benchmark**, já que números de desempenho de uma implementação com race condition não são confiáveis.

```bash
g++ -std=c++20 -pthread -fsanitize=thread -g -O1 KNNConcurrencyStressTest.cpp -o stress_test
./stress_test

g++ -std=c++20 -pthread -O2 KNNBenchmark.cpp -o knn_benchmark -lbenchmark -lpthread
./knn_benchmark --benchmark_repetitions=5 --benchmark_min_warmup_time=2 --benchmark_out=results.json --benchmark_out_format=json
```