
#include <iostream>
#include <vector>
#include <random>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <omp.h>
#include <ctime>
#include <limits>
#include <thread>

class Matr {
private:
    std::vector<std::vector<double>> data;
    size_t rows;
    size_t cols;

    // Тег для нулевой инициализации
    struct ZeroInitTag {};

    // Конструктор для нулевой инициализации
    Matr(size_t r, size_t c, ZeroInitTag) : rows(r), cols(c), data(r, std::vector<double>(c, 0.0)) {
        if (r == 0 || c == 0) {
            throw std::invalid_argument("Matrix dimensions must be positive.");
        }
    }

public:
    // Конструктор с заполнением случайными числами
    Matr(size_t r, size_t c) : rows(r), cols(c), data(r, std::vector<double>(c)) {
        if (r == 0 || c == 0) {
            throw std::invalid_argument("Matrix dimensions must be positive.");
        }

        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(0.0, 150.0);

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                data[i][j] = rand()/123;
            }
        }
    }

    // Конструктор копирования
    Matr(const Matr& other) : rows(other.rows), cols(other.cols), data(other.data) {}

    // Конструктор перемещения
    Matr(Matr&& other) noexcept : rows(other.rows), cols(other.cols), data(std::move(other.data)) {
        other.rows = 0;
        other.cols = 0;
    }

    // Оператор присваивания
    Matr& operator=(Matr other) {
        std::swap(rows, other.rows);
        std::swap(cols, other.cols);
        std::swap(data, other.data);
        return *this;
    }

    // === OpenMP: Параллельное сложение (MSVC-совместимое) ===
    Matr SumOMP(const Matr& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions must match for addition.");
        }

        // Проверка переполнения для преобразования size_t → int
        if (rows > static_cast<size_t>(INT_MAX) || cols > static_cast<size_t>(INT_MAX)) {
            throw std::overflow_error("Matrix dimensions too large for OpenMP (exceed INT_MAX).");
        }

        Matr result(rows, cols, ZeroInitTag{});

#pragma omp parallel for
        for (int i = 0; i < static_cast<int>(rows); ++i) {
            const double* __restrict a = data[i].data();
            const double* __restrict b = other.data[i].data();
            double* __restrict res = result.data[i].data();

            // Автоматическая векторизация компилятором (без #pragma omp simd)
            for (int j = 0; j < static_cast<int>(cols); ++j) {
                res[j] = a[j] + b[j];
            }
        }

        return result;
    }

    // === OpenMP: Параллельное умножение (MSVC-совместимое) ===
    // Параллелизация по строкам результата
    Matr MulOMP(const Matr& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Incompatible dimensions for multiplication.");
        }

        if (rows > static_cast<size_t>(INT_MAX) || other.cols > static_cast<size_t>(INT_MAX)) {
            throw std::overflow_error("Matrix dimensions too large for OpenMP (exceed INT_MAX).");
        }

        Matr result(rows, other.cols, ZeroInitTag{});

#pragma omp parallel for
        for (int i = 0; i < static_cast<int>(rows); ++i) {
            for (int j = 0; j < static_cast<int>(other.cols); ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < cols; ++k) {
                    sum += data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }

        return result;
    }

    // === OpenMP: Оптимизированное умножение (рекомендуется для производительности) ===
    // Перестановка циклов для лучшей локальности кэша
    Matr MulOMP_Opt(const Matr& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Incompatible dimensions for multiplication.");
        }

        if (rows > static_cast<size_t>(INT_MAX) || other.cols > static_cast<size_t>(INT_MAX)) {
            throw std::overflow_error("Matrix dimensions too large for OpenMP (exceed INT_MAX).");
        }

        Matr result(rows, other.cols, ZeroInitTag{});

#pragma omp parallel for
        for (int i = 0; i < static_cast<int>(rows); ++i) {
            // Инициализация строки нулями
            std::fill(result.data[i].begin(), result.data[i].end(), 0.0);

            for (size_t k = 0; k < cols; ++k) {
                double a_ik = data[i][k];
                const double* __restrict b_row = other.data[k].data();
                double* __restrict r_row = result.data[i].data();

                // Автоматическая векторизация при компиляции с /arch:AVX2
                for (int j = 0; j < static_cast<int>(other.cols); ++j) {
                    r_row[j] += a_ik * b_row[j];
                }
            }
        }

        return result;
    }

    // Базовые методы
    Matr Sum(const Matr& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions must match for addition.");
        }

        Matr result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return result;
    }

    Matr Mul(const Matr& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Incompatible dimensions for multiplication.");
        }

        Matr result(rows, other.cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < cols; ++k) {
                    sum += data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }

    std::string ToString() const {
        std::ostringstream oss;
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                oss.width(8);
                oss.precision(2);
                oss << std::fixed << data[i][j];
                if (j < cols - 1) oss << " ";
            }
            if (i < rows - 1) oss << "\n";
        }
        oss << "\n";
        return oss.str();
    }

    size_t GetRows() const { return rows; }
    size_t GetCols() const { return cols; }

    friend std::ostream& operator<<(std::ostream& os, const Matr& m) {
        os << m.ToString();
        return os;
    }
};

// ========================================================================
// === МЕТОД МОНТЕ-КАРЛО ДЛЯ ВЫЧИСЛЕНИЯ ЧИСЛА ПИ (вне класса) ===
// ========================================================================

// Вспомогательная структура для результатов
struct PiResult {
    double pi_estimate;
    size_t iterations;
    double error;
    double elapsed_ms;

    friend std::ostream& operator<<(std::ostream& os, const PiResult& r) {
        os << "Pi = " << std::setprecision(10) << r.pi_estimate
            << " (error: " << std::scientific << r.error
            << ", iterations: " << r.iterations
            << ", time: " << std::fixed << std::setprecision(2) << r.elapsed_ms << " ms)";
        return os;
    }
};

// === 1. Последовательный метод Монте-Карло с функцией rand() ===
PiResult EstimatePiSequential(double target_error = 1e-6, size_t batch_size = 100000) {
    const double PI_TRUE = 3.14159265358979323846;
    auto start = std::chrono::high_resolution_clock::now();

    // Инициализация генератора случайных чисел
    srand(static_cast<unsigned int>(time(nullptr)));

    size_t total_points = 0;
    size_t inside_circle = 0;
    double pi_estimate = 0.0;
    double error = std::numeric_limits<double>::max();

    // Итеративно генерируем пакеты точек до достижения точности
    while (error > target_error) {
        for (size_t i = 0; i < batch_size; ++i) {
            // Генерация случайных чисел в диапазоне [0, 1)
            double x = static_cast<double>(rand()) / RAND_MAX;
            double y = static_cast<double>(rand()) / RAND_MAX;

            if (x * x + y * y <= 1.0) {
                ++inside_circle;
            }
            ++total_points;
        }

        pi_estimate = 4.0 * inside_circle / total_points;
        error = std::fabs(pi_estimate - PI_TRUE);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    return { pi_estimate, total_points, error, elapsed_ms };
}

// === 2. Параллельный метод Монте-Карло с функцией rand() и кэшированием ===
PiResult EstimatePiParallel(double target_error = 1e-6, size_t batch_size = 1000000) {
    const double PI_TRUE = 3.14159265358979323846;
    auto start = std::chrono::high_resolution_clock::now();

    size_t total_points = 0;
    size_t inside_circle = 0;
    double pi_estimate = 0.0;
    double error = std::numeric_limits<double>::max();

    // Итеративно генерируем пакеты точек параллельно
    while (error > target_error) {
        size_t batch_inside = 0;

#pragma omp parallel
        {
            // === КЛЮЧЕВОЙ МОМЕНТ: каждый поток инициализирует свой генератор ===
            unsigned int thread_id = static_cast<unsigned int>(omp_get_thread_num());
            // Уникальное семя для каждого потока (важно!)
            srand(thread_id + static_cast<unsigned int>(time(nullptr)) + static_cast<unsigned int>(total_points));

            // Локальный счётчик для кэширования (минимизируем синхронизацию)
            size_t local_inside = 0;
            const size_t local_batch = batch_size / omp_get_num_threads();

            for (size_t i = 0; i < local_batch; ++i) {
                double x = static_cast<double>(rand()) / RAND_MAX;
                double y = static_cast<double>(rand()) / RAND_MAX;

                if (x * x + y * y <= 1.0) {
                    ++local_inside;
                }
            }

            // Атомарное обновление общего счётчика (минимальная синхронизация)
#pragma omp atomic
            batch_inside += local_inside;
        }

        // Обновляем глобальные счётчики
        inside_circle += batch_inside;
        total_points += batch_size;

        pi_estimate = 4.0 * inside_circle / total_points;
        error = std::fabs(pi_estimate - PI_TRUE);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    return { pi_estimate, total_points, error, elapsed_ms };
}

// === 3. Бонус: Параллельный метод с редукцией через критическую секцию ===
PiResult EstimatePiParallelCritical(double target_error = 1e-6, size_t batch_size = 1000000) {
    const double PI_TRUE = 3.14159265358979323846;
    auto start = std::chrono::high_resolution_clock::now();

    size_t total_points = 0;
    size_t inside_circle = 0;
    double pi_estimate = 0.0;
    double error = std::numeric_limits<double>::max();

    while (error > target_error) {
        size_t batch_inside = 0;

#pragma omp parallel
        {
            unsigned int thread_id = static_cast<unsigned int>(omp_get_thread_num());
            srand(thread_id + static_cast<unsigned int>(time(nullptr)) + static_cast<unsigned int>(total_points));

            size_t local_inside = 0;
            const size_t local_batch = batch_size / omp_get_num_threads();

            for (size_t i = 0; i < local_batch; ++i) {
                double x = static_cast<double>(rand()) / RAND_MAX;
                double y = static_cast<double>(rand()) / RAND_MAX;

                if (x * x + y * y <= 1.0) {
                    ++local_inside;
                }
            }

            // Редукция через критическую секцию
#pragma omp critical
            {
                batch_inside += local_inside;
            }
        }

        inside_circle += batch_inside;
        total_points += batch_size;
        pi_estimate = 4.0 * inside_circle / total_points;
        error = std::fabs(pi_estimate - PI_TRUE);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    return { pi_estimate, total_points, error, elapsed_ms };
}


// Вспомогательная структура для результатов сортировки
struct SortResult {
    std::vector<int> sorted_data;
    size_t comparisons;      // Количество сравнений
    size_t swaps;            // Количество обменов
    double elapsed_ms;       // Время выполнения в миллисекундах

    friend std::ostream& operator<<(std::ostream& os, const SortResult& r) {
        os << "Sorted [" << r.sorted_data.size() << " elements] "
            << "(comparisons: " << r.comparisons
            << ", swaps: " << r.swaps
            << ", time: " << std::fixed << std::setprecision(2) << r.elapsed_ms << " ms)";
        return os;
    }
};

// === 1. Последовательная сортировка простым выбором ===
SortResult SelectionSortSequential(std::vector<int> data) {
    auto start = std::chrono::high_resolution_clock::now();

    size_t comparisons = 0;
    size_t swaps = 0;
    size_t n = data.size();

    for (size_t i = 0; i < n - 1; ++i) {
        size_t min_idx = i;

        // Поиск минимального элемента в неотсортированной части
        for (size_t j = i + 1; j < n; ++j) {
            comparisons++;
            if (data[j] < data[min_idx]) {
                min_idx = j;
            }
        }

        // Обмен найденного минимума с текущим элементом
        if (min_idx != i) {
            std::swap(data[i], data[min_idx]);
            swaps++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    return { std::move(data), comparisons, swaps, elapsed_ms };
}

// === 2. Параллельная сортировка простым выбором с потоками std::thread ===
// Каждый поток ищет минимум в своей части массива
SortResult SelectionSortParallel(std::vector<int> data, size_t num_threads = 0) {
    auto start = std::chrono::high_resolution_clock::now();

    size_t comparisons = 0;
    size_t swaps = 0;
    size_t n = data.size();

    // Определяем количество потоков (не больше, чем элементов)
    if (num_threads == 0 || num_threads > n) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4; // fallback
        if (num_threads > n) num_threads = n;
    }

    // Сортировка по шагам
    for (size_t i = 0; i < n - 1; ++i) {
        size_t min_idx = i;

        // Разбиваем оставшийся массив на части для потоков
        size_t remaining = n - i;
        size_t threads_used = std::min(num_threads, remaining);

        std::vector<std::thread> threads;
        std::vector<size_t> local_mins(threads_used, i);
        std::vector<size_t> local_comparisons(threads_used, 0);

        // Запуск потоков для поиска локальных минимумов
        for (size_t t = 0; t < threads_used; ++t) {
            size_t start_idx = i + t * remaining / threads_used;
            size_t end_idx = (t == threads_used - 1) ? n : i + (t + 1) * remaining / threads_used;

            threads.emplace_back([&, t, start_idx, end_idx]() {
                size_t local_min = start_idx;
                size_t local_comp = 0;

                for (size_t j = start_idx; j < end_idx; ++j) {
                    local_comp++;
                    if (data[j] < data[local_min]) {
                        local_min = j;
                    }
                }

                local_mins[t] = local_min;
                local_comparisons[t] = local_comp;
                });
        }

        // Ожидание завершения всех потоков
        for (auto& t : threads) {
            t.join();
        }

        // Нахождение глобального минимума из локальных
        for (size_t t = 0; t < threads_used; ++t) {
            comparisons += local_comparisons[t];
            if (data[local_mins[t]] < data[min_idx]) {
                min_idx = local_mins[t];
            }
        }

        // Обмен найденного минимума с текущим элементом
        if (min_idx != i) {
            std::swap(data[i], data[min_idx]);
            swaps++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    return { std::move(data), comparisons, swaps, elapsed_ms };
}

// === 3. Параллельная сортировка простым выбором с OpenMP ===
SortResult SelectionSortParallelOMP(std::vector<int> data) {
    auto start = std::chrono::high_resolution_clock::now();

    size_t comparisons = 0;
    size_t swaps = 0;
    size_t n = data.size();

    // Сортировка по шагам
    for (size_t i = 0; i < n - 1; ++i) {
        size_t min_idx = i;

        // Параллельный поиск минимума в оставшейся части массива
#pragma omp parallel
        {
            int num_threads = omp_get_num_threads();
            int thread_id = omp_get_thread_num();

            // Разбиваем работу на части для каждого потока
            size_t remaining = n - i;
            size_t start_idx = i + thread_id * remaining / num_threads;
            size_t end_idx = (thread_id == num_threads - 1) ? n : i + (thread_id + 1) * remaining / num_threads;

            size_t local_min = start_idx;
            size_t local_comp = 0;

            // Поиск локального минимума
            for (size_t j = start_idx; j < end_idx; ++j) {
                local_comp++;
                if (data[j] < data[local_min]) {
                    local_min = j;
                }
            }

            // Критическая секция для обновления глобального минимума
#pragma omp critical
            {
                comparisons += local_comp;
                if (data[local_min] < data[min_idx]) {
                    min_idx = local_min;
                }
            }
        }

        // Обмен найденного минимума с текущим элементом
        if (min_idx != i) {
            std::swap(data[i], data[min_idx]);
            swaps++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    return { std::move(data), comparisons, swaps, elapsed_ms };
}

// === 4. Бонус: Оптимизированная параллельная сортировка с редукцией ===
SortResult SelectionSortParallelOMPOpt(std::vector<int> data) {
    auto start = std::chrono::high_resolution_clock::now();

    size_t comparisons = 0;
    size_t swaps = 0;
    size_t n = data.size();

    for (size_t i = 0; i < n - 1; ++i) {
        // Используем редукцию для поиска минимума
        size_t min_idx = i;
        int min_val = data[i];

#pragma omp parallel
        {
            int num_threads = omp_get_num_threads();
            int thread_id = omp_get_thread_num();

            size_t remaining = n - i;
            size_t start_idx = i + thread_id * remaining / num_threads;
            size_t end_idx = (thread_id == num_threads - 1) ? n : i + (thread_id + 1) * remaining / num_threads;

            size_t local_min = start_idx;
            int local_min_val = data[start_idx];
            size_t local_comp = 0;

            for (size_t j = start_idx; j < end_idx; ++j) {
                local_comp++;
                if (data[j] < local_min_val) {
                    local_min_val = data[j];
                    local_min = j;
                }
            }

            // Атомарное обновление глобального минимума
#pragma omp critical
            {
                comparisons += local_comp;
                if (local_min_val < min_val) {
                    min_val = local_min_val;
                    min_idx = local_min;
                }
            }
        }

        if (min_idx != i) {
            std::swap(data[i], data[min_idx]);
            swaps++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    return { std::move(data), comparisons, swaps, elapsed_ms };
}

// ========================================================================
// === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===
// ========================================================================

// Генерация случайного вектора int
std::vector<int> GenerateRandomVector(size_t size, int min_val = 0, int max_val = 1000) {
    std::vector<int> vec(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(min_val, max_val);

    for (auto& val : vec) {
        val = dis(gen);
    }

    return vec;
}

// Проверка корректности сортировки
bool IsSorted(const std::vector<int>& vec) {
    for (size_t i = 1; i < vec.size(); ++i) {
        if (vec[i] < vec[i - 1]) {
            return false;
        }
    }
    return true;
}

// Вывод первых и последних элементов вектора
void PrintVectorPreview(const std::vector<int>& vec, size_t preview_count = 10) {
    size_t n = vec.size();
    size_t count = std::min(preview_count, n);

    std::cout << "[";
    for (size_t i = 0; i < count; ++i) {
        std::cout << vec[i];
        if (i < count - 1) std::cout << ", ";
    }

    if (n > 2 * count) {
        std::cout << ", ..., ";
        for (size_t i = n - count; i < n; ++i) {
            std::cout << vec[i];
            if (i < n - 1) std::cout << ", ";
        }
    }
    else if (n > count) {
        for (size_t i = count; i < n; ++i) {
            std::cout << ", " << vec[i];
        }
    }

    std::cout << "]";
}

using Clock = std::chrono::high_resolution_clock;
using Miliseconds = std::chrono::milliseconds;


int main() {
    srand(time(0));

    {
        std::vector<int> test = GenerateRandomVector(100);
        std::vector<int> test_big = GenerateRandomVector(100000);

        

        std::cout << SelectionSortSequential(test) << std::endl;
        std::cout << SelectionSortParallelOMP(test) << std::endl;

        std::cout << SelectionSortSequential(test_big) << std::endl;
        std::cout << SelectionSortParallelOMP(test_big) << std::endl;
    }

    /*{
        
        std::cout << EstimatePiSequential(0.0001, 1000000) << std::endl;
        std::cout << EstimatePiParallel(0.0001, 1000000) << std::endl;
    }*/

    /*{
        Matr a(10, 10), b(10, 10);
        auto t_point_1 = Clock::now();
        a.Mul(b);
        auto t_point_2 = Clock::now();
        std::cout << std::chrono::duration_cast<Miliseconds>(t_point_2 - t_point_1).count() << std::endl;

        t_point_1 = Clock::now();
        a.MulOMP(b);
        t_point_2 = Clock::now();
        std::cout << std::chrono::duration_cast<Miliseconds>(t_point_2 - t_point_1).count() << std::endl;

        t_point_1 = Clock::now();
        a.MulOMP_Opt(b);
        t_point_2 = Clock::now();
        std::cout << std::chrono::duration_cast<Miliseconds>(t_point_2 - t_point_1).count() << std::endl;
    }

    {
        Matr a(500, 500), b(500, 500);
        auto t_point_1 = Clock::now();
        a.Mul(b);
        auto t_point_2 = Clock::now();
        std::cout << std::chrono::duration_cast<Miliseconds>(t_point_2 - t_point_1).count() << std::endl;

        t_point_1 = Clock::now();
        a.MulOMP(b);
        t_point_2 = Clock::now();
        std::cout << std::chrono::duration_cast<Miliseconds>(t_point_2 - t_point_1).count() << std::endl;

        t_point_1 = Clock::now();
        a.MulOMP_Opt(b);
        t_point_2 = Clock::now();
        std::cout << std::chrono::duration_cast<Miliseconds>(t_point_2 - t_point_1).count() << std::endl;
    }*/

}

/*class Matr {
private:
    std::vector<std::vector<double>> data;
    size_t rows;
    size_t cols;

public:
    // Конструктор с заполнением случайными числами
    Matr(size_t r, size_t c) : rows(r), cols(c), data(r, std::vector<double>(c)) {
        if (r == 0 || c == 0) {
            throw std::invalid_argument("Matrix dimensions must be positive.");
        }

        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(0.0, 99.9);

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                data[i][j] = dis(gen);
            }
        }
    }

    // Конструктор копирования (для многопоточности)
    Matr(const Matr& other) : rows(other.rows), cols(other.cols), data(other.data) {}

    // Конструктор перемещения
    Matr(Matr&& other) noexcept : rows(other.rows), cols(other.cols), data(std::move(other.data)) {
        other.rows = 0;
        other.cols = 0;
    }

    // Оператор присваивания
    Matr& operator=(Matr other) {
        std::swap(rows, other.rows);
        std::swap(cols, other.cols);
        std::swap(data, other.data);
        return *this;
    }

    // === БАЗОВЫЕ ОПЕРАЦИИ (однопоточные) ===
    Matr Sum(const Matr& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions must match for addition.");
        }

        Matr result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return result;
    }

    Matr Mul(const Matr& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Incompatible dimensions for multiplication.");
        }

        Matr result(rows, other.cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < cols; ++k) {
                    sum += data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }

    std::string ToString() const {
        std::ostringstream oss;
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                oss.width(8);
                oss.precision(2);
                oss << std::fixed << data[i][j];
                if (j < cols - 1) oss << " ";
            }
            if (i < rows - 1) oss << "\n";
        }
        oss << "\n";
        return oss.str();
    }

    // Оператор вывода
    friend std::ostream& operator<<(std::ostream& os, const Matr& m) {
        os << m.ToString();
        return os;
    }
};*/

