import numpy as np
from typing import List, Tuple, Dict, Optional


class MinLeakageSplitter:
    def __init__(
        self,
        distance_matrix: np.ndarray,
        desired_test_split_pct: float = 20.0,
        buffer_pct: float = 2.0,
        max_iterations: int = 5000,
        initial_temperature: float = 0.1,
        cooling_rate: float = 0.999,
        n_candidates: int = 20,
        max_no_improvement: int = 1000,
        record_interval: int = 10,
        seed: Optional[int] = None,
    ):
        mat = np.asarray(distance_matrix)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError("distance_matrix must be square (N x N).")
        self.distance_matrix = mat
        self.n_samples = mat.shape[0]

        self.desired_test_split_pct = float(desired_test_split_pct)
        self.buffer_pct = float(buffer_pct)
        self.max_iterations = int(max_iterations)
        self.n_candidates = int(n_candidates)
        self.max_no_improvement = int(max_no_improvement)
        self.record_interval = int(record_interval)

        self.rng = np.random.default_rng(seed)

        self.train_val_indices = None
        self.test_indices = None

        self.best_train_val_indices = None
        self.best_test_indices = None
        self.history: List[Dict] = []

        self.best_mean_train_val_indices = None
        self.best_mean_test_indices = None
        self.history_mean: List[Dict] = []

        self.best_min_train_val_indices = None
        self.best_min_test_indices = None
        self.history_min: List[Dict] = []
        self.best_min_tau: Optional[float] = None

    def compute_bipartite_statistics(
        self, train_val_indices: List[int], test_indices: List[int]
    ) -> Tuple[np.ndarray, Dict]:
        tv = np.asarray(train_val_indices, dtype=np.int64)
        t = np.asarray(test_indices, dtype=np.int64)
        edge = self.distance_matrix[np.ix_(tv, t)]
        if edge.size == 0:
            return edge, {"mean_distance": float("-inf"), "min_distance": float("-inf"), "n_total_edges": 0}
        return edge, {
            "mean_distance": float(edge.mean()),
            "min_distance": float(edge.min()),
            "n_total_edges": int(edge.size),
        }

    def _bounds(self, n_total: int) -> Tuple[int, int, int]:
        target = int(round(n_total * self.desired_test_split_pct / 100.0))
        min_t = int(np.floor(n_total * (self.desired_test_split_pct - self.buffer_pct) / 100.0))
        max_t = int(np.ceil(n_total * (self.desired_test_split_pct + self.buffer_pct) / 100.0))
        min_t = max(0, min_t)
        max_t = min(n_total, max_t)
        target = int(np.clip(target, min_t, max_t))
        return min_t, target, max_t

    def _mean_distance(self, cut_sum: float, n_tv: int, n_t: int) -> float:
        if n_tv <= 0 or n_t <= 0:
            return float("-inf")
        return cut_sum / (n_tv * n_t)

    def _init_state(self, tv: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        n = self.n_samples
        in_test = np.zeros(n, dtype=bool)
        in_tv = np.zeros(n, dtype=bool)
        in_test[t] = True
        in_tv[tv] = True
        cut_sum = float(self.distance_matrix[np.ix_(tv, t)].sum()) if tv.size and t.size else 0.0
        return in_tv, in_test, cut_sum

    def _cut_after_tv_to_t(self, x: int, tv: np.ndarray, t: np.ndarray, cut_sum: float) -> float:
        mat = self.distance_matrix
        tv_wo_x = tv[tv != x]
        add = float(mat[np.ix_(tv_wo_x, np.array([x], dtype=np.int64))].sum()) if tv_wo_x.size else 0.0
        sub = float(mat[np.ix_(np.array([x], dtype=np.int64), t)].sum()) if t.size else 0.0
        return cut_sum + add - sub

    def _cut_after_t_to_tv(self, y: int, tv: np.ndarray, t: np.ndarray, cut_sum: float) -> float:
        mat = self.distance_matrix
        t_wo_y = t[t != y]
        sub = float(mat[np.ix_(tv, np.array([y], dtype=np.int64))].sum()) if tv.size else 0.0
        add = float(mat[np.ix_(np.array([y], dtype=np.int64), t_wo_y)].sum()) if t_wo_y.size else 0.0
        return cut_sum - sub + add

    def _cut_after_swap(self, x: int, y: int, tv: np.ndarray, t: np.ndarray, cut_sum: float) -> float:
        mat = self.distance_matrix
        tv_wo_x = tv[tv != x]
        t_wo_y = t[t != y]

        rem1 = float(mat[np.ix_(np.array([x], dtype=np.int64), t_wo_y)].sum()) if t_wo_y.size else 0.0
        rem2 = float(mat[np.ix_(tv_wo_x, np.array([y], dtype=np.int64))].sum()) if tv_wo_x.size else 0.0
        add1 = float(mat[np.ix_(np.array([y], dtype=np.int64), t_wo_y)].sum()) if t_wo_y.size else 0.0
        add2 = float(mat[np.ix_(tv_wo_x, np.array([x], dtype=np.int64))].sum()) if tv_wo_x.size else 0.0

        return cut_sum - rem1 - rem2 - float(mat[x, y]) + add1 + add2 + float(mat[y, x])

    def _reorganize_mean_max(
        self, train_val_indices: List[int], test_indices: List[int]
    ) -> Tuple[List[int], List[int], List[Dict]]:

        tv0 = np.asarray(train_val_indices, dtype=np.int64)
        t0 = np.asarray(test_indices, dtype=np.int64)

        n_total = tv0.size + t0.size
        min_t, target_t, max_t = self._bounds(n_total)

        in_tv, in_test, cut_sum = self._init_state(tv0, t0)
        n_tv = int(in_tv.sum())
        n_t = int(in_test.sum())
        cur_md = self._mean_distance(cut_sum, n_tv, n_t)

        best_in_tv = in_tv.copy()
        best_in_test = in_test.copy()
        best_md = cur_md

        history: List[Dict] = []
        no_imp = 0

        for it in range(self.max_iterations):
            if it % self.record_interval == 0:
                history.append(
                    {
                        "iteration": it,
                        "mean_distance": cur_md,
                        "test_split_pct": (n_t / n_total) * 100.0,
                        "train_val_size": n_tv,
                        "test_size": n_t,
                    }
                )

            tv = np.flatnonzero(in_tv)
            t = np.flatnonzero(in_test)

            if n_t < min_t:
                mode = "increase_test"
            elif n_t > max_t:
                mode = "decrease_test"
            else:
                mode = "within_bounds"

            best_candidate = None
            best_cand_md = cur_md

            for _ in range(self.n_candidates):
                if mode == "increase_test":
                    if tv.size == 0:
                        continue
                    x = int(tv[self.rng.integers(tv.size)])
                    new_n_tv, new_n_t = n_tv - 1, n_t + 1
                    new_cut = self._cut_after_tv_to_t(x, tv, t, cut_sum)
                    op = ("tv_to_t", x, -1)

                elif mode == "decrease_test":
                    if t.size == 0:
                        continue
                    y = int(t[self.rng.integers(t.size)])
                    new_n_tv, new_n_t = n_tv + 1, n_t - 1
                    new_cut = self._cut_after_t_to_tv(y, tv, t, cut_sum)
                    op = ("t_to_tv", -1, y)

                else:
                    do_swap = (tv.size > 0) and (t.size > 0) and (self.rng.random() < (0.7 if n_t == target_t else 0.3))
                    if do_swap:
                        x = int(tv[self.rng.integers(tv.size)])
                        y = int(t[self.rng.integers(t.size)])
                        new_n_tv, new_n_t = n_tv, n_t
                        new_cut = self._cut_after_swap(x, y, tv, t, cut_sum)
                        op = ("swap", x, y)
                    else:
                        move_decrease = (n_t > target_t) or (n_t == target_t and self.rng.random() < 0.5)
                        if move_decrease:
                            if t.size == 0 or n_t - 1 < min_t:
                                continue
                            y = int(t[self.rng.integers(t.size)])
                            new_n_tv, new_n_t = n_tv + 1, n_t - 1
                            new_cut = self._cut_after_t_to_tv(y, tv, t, cut_sum)
                            op = ("t_to_tv", -1, y)
                        else:
                            if tv.size == 0 or n_t + 1 > max_t:
                                continue
                            x = int(tv[self.rng.integers(tv.size)])
                            new_n_tv, new_n_t = n_tv - 1, n_t + 1
                            new_cut = self._cut_after_tv_to_t(x, tv, t, cut_sum)
                            op = ("tv_to_t", x, -1)

                if not (min_t <= new_n_t <= max_t):
                    continue

                new_md = self._mean_distance(new_cut, new_n_tv, new_n_t)
                if new_md > best_cand_md:
                    best_cand_md = new_md
                    best_candidate = (op, new_cut, new_n_tv, new_n_t, new_md)

            if best_candidate is None or best_cand_md <= cur_md + 1e-15:
                no_imp += 1
                if no_imp >= self.max_no_improvement:
                    break
                continue

            op, new_cut, new_n_tv, new_n_t, new_md = best_candidate
            kind, x, y = op
            if kind == "swap":
                in_tv[x], in_test[x] = False, True
                in_tv[y], in_test[y] = True, False
            elif kind == "tv_to_t":
                in_tv[x], in_test[x] = False, True
            else:
                in_tv[y], in_test[y] = True, False

            cut_sum = float(new_cut)
            n_tv, n_t = int(new_n_tv), int(new_n_t)
            cur_md = float(new_md)

            if cur_md > best_md + 1e-15:
                best_md = cur_md
                best_in_tv = in_tv.copy()
                best_in_test = in_test.copy()
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= self.max_no_improvement:
                    break

        best_tv = np.flatnonzero(best_in_tv).tolist()
        best_t = np.flatnonzero(best_in_test).tolist()
        return best_tv, best_t, history

    def _union_find_components(self, tau: float) -> Tuple[np.ndarray, np.ndarray]:
        n = self.n_samples
        parent = np.arange(n, dtype=np.int64)
        rank = np.zeros(n, dtype=np.int8)

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1

        D = self.distance_matrix
        iu, ju = np.triu_indices(n, k=1)
        bad = D[iu, ju] < tau
        if np.any(bad):
            a = iu[bad]
            b = ju[bad]
            for i, j in zip(a.tolist(), b.tolist()):
                union(i, j)

        roots = np.fromiter((find(i) for i in range(n)), dtype=np.int64, count=n)
        uniq, inv = np.unique(roots, return_inverse=True)
        sizes = np.bincount(inv, minlength=uniq.size).astype(np.int64)
        return inv, sizes

    def _choose_components(self, sizes: np.ndarray, min_t: int, target_t: int, max_t: int) -> Optional[np.ndarray]:
        k = sizes.size
        dp = 1
        prev = [0] * (k + 1)
        prev[0] = dp
        for i, s in enumerate(sizes, start=1):
            dp = dp | (dp << int(s))
            prev[i] = dp

        def achievable(m: int) -> bool:
            return ((dp >> m) & 1) == 1

        candidates = [m for m in range(min_t, max_t + 1) if achievable(m)]
        if not candidates:
            return None

        m = min(candidates, key=lambda x: (abs(x - target_t), x))

        take = np.zeros(k, dtype=bool)
        cur = m
        for i in range(k, 0, -1):
            s = int(sizes[i - 1])
            before = prev[i - 1]
            if ((before >> cur) & 1) == 1:
                continue
            take[i - 1] = True
            cur -= s

        return take

    def _feasible_split_at_tau(self, tau: float, min_t: int, target_t: int, max_t: int):
        comp_id, sizes = self._union_find_components(tau)
        take = self._choose_components(sizes, min_t, target_t, max_t)
        if take is None:
            return None
        in_test = take[comp_id]
        test_idx = np.flatnonzero(in_test).tolist()
        tv_idx = np.flatnonzero(~in_test).tolist()
        return tv_idx, test_idx

    def _solve_min_max(self) -> Tuple[List[int], List[int], List[Dict], Optional[float]]:
        n_total = self.n_samples
        min_t, target_t, max_t = self._bounds(n_total)

        D = self.distance_matrix
        iu, ju = np.triu_indices(self.n_samples, k=1)
        vals = D[iu, ju]
        vals = np.unique(vals[np.isfinite(vals)])
        if vals.size == 0:
            idx = np.arange(self.n_samples)
            self.rng.shuffle(idx)
            t = target_t if 1 <= target_t <= self.n_samples - 1 else max(1, min(self.n_samples - 1, target_t))
            return idx[:-t].tolist(), idx[-t:].tolist(), [], None

        lo, hi = 0, vals.size - 1
        best_tv = None
        best_t = None
        best_tau = None
        history: List[Dict] = []

        while lo <= hi:
            mid = (lo + hi) // 2
            tau = float(vals[mid])
            res = self._feasible_split_at_tau(tau, min_t, target_t, max_t)
            if res is not None:
                tv_idx, t_idx = res
                edge = self.distance_matrix[np.ix_(np.asarray(tv_idx, dtype=np.int64), np.asarray(t_idx, dtype=np.int64))]
                min_cross = float(edge.min()) if edge.size else float("-inf")
                history.append(
                    {
                        "tau": tau,
                        "min_distance": min_cross,
                        "test_split_pct": (len(t_idx) / n_total) * 100.0,
                        "train_val_size": len(tv_idx),
                        "test_size": len(t_idx),
                        "feasible": True,
                    }
                )
                best_tv, best_t, best_tau = tv_idx, t_idx, tau
                lo = mid + 1
            else:
                history.append(
                    {
                        "tau": tau,
                        "min_distance": float("nan"),
                        "test_split_pct": float("nan"),
                        "train_val_size": float("nan"),
                        "test_size": float("nan"),
                        "feasible": False,
                    }
                )
                hi = mid - 1

        if best_tv is None:
            idx = np.arange(self.n_samples)
            self.rng.shuffle(idx)
            t = target_t if 1 <= target_t <= self.n_samples - 1 else max(1, min(self.n_samples - 1, target_t))
            best_tv, best_t, best_tau = idx[:-t].tolist(), idx[-t:].tolist(), float(vals[0])

        return best_tv, best_t, history, best_tau

    def split(
        self, train_val_indices: List[int] = None, test_indices: List[int] = None
    ) -> Tuple[List[int], List[int]]:

        if train_val_indices is None or test_indices is None:
            idx = np.arange(self.n_samples, dtype=np.int64)
            self.rng.shuffle(idx)
            n_total = self.n_samples
            _, target, _ = self._bounds(n_total)
            target = int(np.clip(target, 1, self.n_samples - 1))
            test_indices = idx[-target:].tolist()
            train_val_indices = idx[:-target].tolist()

        self.train_val_indices = train_val_indices
        self.test_indices = test_indices

        mean_tv, mean_t, hist_mean = self._reorganize_mean_max(train_val_indices, test_indices)
        self.best_mean_train_val_indices = mean_tv
        self.best_mean_test_indices = mean_t
        self.history_mean = hist_mean

        self.best_train_val_indices = mean_tv
        self.best_test_indices = mean_t
        self.history = hist_mean

        min_tv, min_t, hist_min, tau = self._solve_min_max()
        self.best_min_train_val_indices = min_tv
        self.best_min_test_indices = min_t
        self.history_min = hist_min
        self.best_min_tau = tau

        return self.best_train_val_indices, self.best_test_indices

