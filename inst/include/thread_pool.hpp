// A small persistent thread pool.
//
// ThreadPool(n) keeps n-1 worker threads alive; the calling thread acts as
// worker 0. parallel_for(n_items, fn) splits [0, n_items) into n contiguous
// chunks (the last one takes the remainder, like the original std::thread
// code) and runs fn(start, end, chunk) on each. With n <= 1 everything runs
// inline on the caller, so n_threads = 0 keeps exact serial semantics.
//
// Dispatch is lock-free: the caller publishes the task and bumps an atomic
// generation counter; workers spin on it (with a CPU pause) for a while
// before falling back to a condition variable, so the many short
// parallel_for calls of an iteration (a few ms each) do not pay a futex
// wake-up per worker per call, which limited the old sleep/wake pool to
// about 3x on 12 cores. A pool only lives for one solver call, so the
// spinning never outlasts the computation.
//
// Workers must never touch the R API. Kernels passed here are pure C++.

#ifndef RWIG_THREAD_POOL_H
#define RWIG_THREAD_POOL_H

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#define RWIG_CPU_RELAX() _mm_pause()
#else
#define RWIG_CPU_RELAX() std::this_thread::yield()
#endif

class ThreadPool {
public:
  explicit ThreadPool(int n_threads) : _n(n_threads < 1 ? 1 : n_threads) {
    for (int t = 1; t < _n; ++t) {
      _workers.emplace_back([this, t]() { this->_worker(t); });
    }
  }

  ~ThreadPool() {
    _stop.store(true, std::memory_order_relaxed);
    _publish();
    for (auto &w : _workers) w.join();
  }

  ThreadPool(const ThreadPool &) = delete;
  ThreadPool &operator=(const ThreadPool &) = delete;

  int size() const { return _n; }

  // number of chunks parallel_for(n_items, ...) runs: 1 when it falls back
  // to the caller (one thread, or fewer items than threads), else size()
  int chunks(int n_items) const { return (_n == 1 || n_items < _n) ? 1 : _n; }

  // runs fn(start, end, chunk) on every chunk, chunk in [0, chunks(n_items));
  // fn is any callable, reached through a type-erased pointer for the
  // duration of the call (no std::function, nothing allocated per call)
  template <class F> void parallel_for(int n_items, const F &fn) {
    if (chunks(n_items) == 1) {
      // serial fallback (also avoids empty chunks for tiny problems)
      fn(0, n_items, 0);
      return;
    }
    _ctx = &fn;
    _invoke = [](const void *ctx, int a, int b, int t) {
      (*static_cast<const F *>(ctx))(a, b, t);
    };
    _n_items = n_items;
    _remaining.store(_n - 1, std::memory_order_relaxed);
    _publish();
    // caller runs chunk 0, then waits for the workers
    _run_chunk(0);
    while (_remaining.load(std::memory_order_acquire) != 0) RWIG_CPU_RELAX();
    _ctx = nullptr;
  }

private:
  // spins before a worker goes to sleep: ~1 ms, longer than the serial
  // work between two parallel_for calls of the solvers
  static constexpr int SPIN_LIMIT = 50000;

  int _n;
  std::vector<std::thread> _workers;
  const void *_ctx = nullptr; // the callable of the running parallel_for
  void (*_invoke)(const void *, int, int, int) = nullptr;
  int _n_items = 0;
  // the two counters live on their own cache lines: workers spin on
  // _generation while the caller spins on _remaining
  alignas(64) std::atomic<unsigned long> _generation{0};
  alignas(64) std::atomic<int> _remaining{0};
  std::atomic<bool> _stop{false};
  std::mutex _m;
  std::condition_variable _cv;

  // bump the generation (under the mutex, so a worker that is about to
  // sleep cannot miss it) and wake any sleeping worker
  void _publish() {
    {
      std::lock_guard<std::mutex> lk(_m);
      _generation.fetch_add(1, std::memory_order_release);
    }
    _cv.notify_all();
  }

  void _run_chunk(int t) const {
    const int chunk = _n_items / _n;
    const int start = t * chunk;
    const int end = (t == _n - 1) ? _n_items : (t + 1) * chunk;
    _invoke(_ctx, start, end, t);
  }

  void _worker(int t) {
    unsigned long seen = 0;
    for (;;) {
      // wait for a new generation: spin first, then sleep
      int spins = 0;
      while (_generation.load(std::memory_order_acquire) == seen) {
        if (++spins < SPIN_LIMIT) {
          RWIG_CPU_RELAX();
        } else {
          std::unique_lock<std::mutex> lk(_m);
          _cv.wait(lk, [&]() {
            return _generation.load(std::memory_order_acquire) != seen;
          });
        }
      }
      seen = _generation.load(std::memory_order_acquire);
      if (_stop.load(std::memory_order_relaxed)) return;
      _run_chunk(t);
      _remaining.fetch_sub(1, std::memory_order_acq_rel);
    }
  }
};

#endif // RWIG_THREAD_POOL_H
