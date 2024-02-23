from typing import overload
import numpy as np
from scipy.sparse import dok_matrix
from sklearn.base import BaseEstimator, ClusterMixin

__all__ = [
    "LBSoinn",
]


class LBSoinn(BaseEstimator, ClusterMixin):
    def __init__(
        self,
        dim=2,
        max_edge_age=50,
        iteration_threshold=200,
        c1=0.001,
        c2=1.0,
        keep_node=False,
        gamma=1.01,
    ):
        self.dim = dim
        self.iteration_threshold = iteration_threshold
        self.c1 = c1
        self.c2 = c2
        self.max_edge_age = max_edge_age
        self.num_signal = 0
        self.reset_state()
        self.keep_node = keep_node
        self.gamma = gamma

    def reset_state(self):
        self.num_signal = 0
        self.nodes = np.array([], dtype=np.float64)
        self.nodes_training_index = []
        self.winning_times = []
        self.density = []
        self.N = []
        # if active
        self.won = []
        self.total_loop = 1
        self.s = []
        self.adjacent_mat = dok_matrix((0, 0), dtype=np.float64)
        self.node_labels = []
        self.labels_ = []
        self.eumin = 999999
        self.eumax = -1
        self.comin = 10
        self.comax = -1
        self.new_eumin = 999999
        self.new_eumax = -1
        self.new_comin = 10
        self.new_comax = -1
        self.G = []

    def calculate_pairwise_distance(self, p_weight, q_weight):
        eu = np.sqrt(np.sum((p_weight - q_weight) ** 2))
        co = 1 - np.dot(p_weight, q_weight) / (
            np.linalg.norm(p_weight) * np.linalg.norm(q_weight)
        )
        w = 1 / (self.gamma**self.dim)
        dist = w * (eu - self.eumin) / (1 + self.eumax - self.eumin) + (1 - w) * (
            co - self.comin
        ) / (1 + self.comax - self.comin)
        return dist, eu, co

    def calculate_qbatch_distance(self, signal, q_indexes):
        x = np.array([signal] * len(q_indexes))
        y = self.nodes[q_indexes]
        eu = np.sqrt(np.sum((x - y) ** 2, 1))
        co = 1 - np.sum(np.multiply(x, y), 1) / (
            np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
        )
        w = 1 / (self.gamma**self.dim)
        dist = w * (eu - self.eumin) / (1 + self.eumax - self.eumin) + (1 - w) * (
            co - self.comin
        ) / (1 + self.comax - self.comin)
        return dist, eu, co

    def calculate_pbatch_qbatch_distance(self, p_indexes, q_indexes):
        x = self.nodes[p_indexes]
        x = np.stack([x for _ in range(len(q_indexes))], axis=1)
        y = self.nodes[q_indexes]
        y = np.stack([y for _ in range(len(p_indexes))], axis=0)
        eu = np.sqrt(np.sum((x - y) ** 2, 2))
        co = 1 - np.sum(np.multiply(x, y), 2) / (
            np.linalg.norm(x, axis=2) * np.linalg.norm(y, axis=2)
        )
        w = 1 / (self.gamma**self.dim)
        dist = w * (eu - self.eumin) / (1 + self.eumax - self.eumin) + (1 - w) * (
            co - self.comin
        ) / (1 + self.comax - self.comin)
        return dist, eu, co

    def input_signal(
        self, signal: np.ndarray, label: str, current_training_index: tuple
    ):
        """
        Input a new signal one by one, which means training in online manner.
        fit() calls __init__() before training, which means resetting the
        state. So the function does batch training.
        :param signal: A new input signal
        :label: Text label of the input signal
        :return:
        """
        # Algorithm 3.4 (2)
        signal = self.__check_signal(signal)
        self.num_signal += 1

        # Algorithm 3.4 (1)
        if len(self.nodes) < 2:
            self.__add_node(signal, label, current_training_index)
            return
        if len(self.nodes) == 2:
            _, eu, co = self.calculate_pairwise_distance(
                self.nodes[0, :], self.nodes[1, :]
            )
            self.new_eumax = eu
            self.new_eumin = eu
            self.new_comin = co
            self.new_comin = co
        if (
            self.new_eumax != self.eumax
            or self.new_eumin != self.eumin
            or self.new_comax != self.comax
            or self.new_comin != self.comin
        ):
            self.__update_all_density(
                self.new_eumax, self.new_eumin, self.new_comax, self.new_comin
            )
            self.eumax = self.new_eumax
            self.eumin = self.new_eumin
            self.comax = self.new_comax
            self.comin = self.new_comin

        # Algorithm 3.4 (3)
        winner, dists = self.__find_nearest_nodes(2, signal)
        self.G[winner[0]] += 1
        sim_thresholds = self.__calculate_similarity_thresholds(winner)
        if (
            dists[0] > sim_thresholds[0]
            or dists[1] > sim_thresholds[1]
            or (self.node_labels[winner[0]] != label)
        ):
            _, eu, co = self.calculate_qbatch_distance(
                signal, list(range(len(self.nodes)))
            )
            self.__add_node(signal, label, current_training_index)
            self.new_eumax = max(self.new_eumax, np.amax(eu))
            self.new_eumin = min(self.new_eumin, np.amin(eu))
            self.new_comax = max(self.new_comax, np.amax(co))
            self.new_comin = min(self.new_comin, np.amin(co))
        else:
            self.nodes_training_index[winner[0]].append(current_training_index)
            # Algorithm 3.4 (4)
            self.__increment_edge_ages(winner[0])
            # Algorithm 3.4 (5)
            need_add_edge, _ = self.__need_add_edge(winner)
            if need_add_edge:
                # print("add edge")
                # Algorithm 3.4 (5)(a)
                self.__add_edge(winner)
            else:
                # Algorithm 3.4 (5)(b)
                self.__remove_edge_from_adjacent_mat(winner)

            # Algorithm 3.4 (6) checked, maybe fixed problem N
            self.__update_density(winner[0])
            # Algorithm 3.4 (7)(8)
            self.__update_winner(winner[0], signal)
            # Algorithm 3.4 (8)
            self.__update_adjacent_nodes(winner[0], signal)

            pals = self.adjacent_mat[winner[0]]
            pal_indexes = []
            for k in pals.keys():
                pal_indexes.append(k[1])
            _, eu, co = self.calculate_pbatch_qbatch_distance(
                pal_indexes + [winner[0]], list(range(len(self.nodes)))
            )
            self.new_eumax = max(self.new_eumax, np.amax(eu))
            self.new_comax = max(self.new_comax, np.amax(co))
            for i in range(eu.shape[0]):
                eu[i, i] = 999999
                co[i, i] = 999999
            self.new_eumin = min(self.new_eumin, np.amin(eu))
            self.new_comin = min(self.new_comin, np.amin(co))
            # self.new_eumax, self.new_eumin, self.new_comax, self.new_comin=self.__cal_new_eu_co_dist()

        # Algorithm 3.4 (9)
        # self.__remove_old_edges()

        # Algorithm 3.4 (10)
        if self.num_signal % self.iteration_threshold == 0 and self.num_signal > 1:
            # print(self.won)
            self.__balance_load()
            for i in range(len(self.won)):
                if self.won[i]:
                    self.N[i] += 1
            for i in range(len(self.won)):
                self.won[i] = False
            # print("Input signal amount:", self.num_signal, "nodes amount", len(self.nodes))
            # self.__separate_subclass()
            if not self.keep_node:
                self.__delete_noise_nodes()
            self.total_loop += 1
            # self.classify()
            # threading.Thread(self.plot_NN())
        assert len(self.G) == len(self.N) == len(self.nodes) == len(self.density)
        return

    def __balance_load(self):
        q = np.argmax(self.winning_times)
        class_indexes = [
            i
            for i in range(len(self.nodes))
            if self.node_labels[i] == self.node_labels[q]
        ]
        avgcq = sum([self.winning_times[i] for i in class_indexes])
        avgcq /= len(class_indexes)
        if self.winning_times[q] <= 3 * avgcq:
            return
        else:
            pals = self.adjacent_mat[q]
            pal_indexes = []
            for k in pals.keys():
                pal_indexes.append(k[1])
            if len(pal_indexes) == 0:
                return
            max_pal_wt = -1
            f = None
            for pal_index in pal_indexes:
                if self.winning_times[pal_index] > max_pal_wt:
                    f = pal_index
                    max_pal_wt = self.winning_times[pal_index]
            assert f is not None
            n = self.nodes.shape[0]
            self.nodes.resize((n + 1, self.dim))
            self.nodes[-1, :] = (self.nodes[q, :] + self.nodes[f, :]) / 2.0
            self.winning_times[q] *= 1 - 1 / (2.0 * self.dim)
            self.winning_times[f] *= 1 - 1 / (2.0 * self.dim)
            self.winning_times.append(
                0.5 * (self.winning_times[q] + self.winning_times[f])
            )
            self.adjacent_mat.resize((n + 1, n + 1))
            self.nodes_training_index.append([])
            self.N.append(1)
            self.density.append(0.5 * (self.density[q] + self.density[f]))
            self.s.append([0, 0])
            self.G.append(0.5 * (self.winning_times[q] + self.winning_times[f]))
            self.won.append(False)
            assert self.node_labels[q] == self.node_labels[f]
            self.node_labels.append(self.node_labels[q])
            self.__add_edge([n, q])
            self.__add_edge([n, f])
            self.__remove_edge_from_adjacent_mat([q, f])

    def __cal_new_eu_co_dist(self):
        dist, eu, co = self.calculate_pbatch_qbatch_distance(
            list(range(len(self.nodes))), list(range(len(self.nodes)))
        )
        eu = eu[~np.eye(eu.shape[0], dtype=bool)].reshape(dist.shape[0], -1)
        co = co[~np.eye(co.shape[0], dtype=bool)].reshape(dist.shape[0], -1)
        new_eumax = np.amax(eu)
        new_eumin = np.amin(eu)
        new_comax = np.amax(co)
        new_comin = np.amin(co)
        return new_eumax, new_eumin, new_comax, new_comin

    # checked
    def __remove_old_edges(self):
        for i in list(self.adjacent_mat.keys()):
            if self.adjacent_mat[i] > self.max_edge_age + 1:
                # print("Edge removed")
                self.adjacent_mat.pop((i[0], i[1]))

    # checked
    def __remove_edge_from_adjacent_mat(self, ids):
        if (ids[0], ids[1]) in self.adjacent_mat and (
            ids[1],
            ids[0],
        ) in self.adjacent_mat:
            self.adjacent_mat.pop((ids[0], ids[1]))
            self.adjacent_mat.pop((ids[1], ids[0]))

    # Algorithm 3.2, checked
    def __need_add_edge(self, winner):
        if self.node_labels[winner[0]] == self.node_labels[winner[1]]:
            return True, False
        else:
            return False, False

    @overload
    def __check_signal(self, signal: list) -> None:
        ...

    def __check_signal(self, signal: np.ndarray):
        """
        check type and dimensionality of an input signal.
        If signal is the first input signal, set the dimension of it as
        self.dim. So, this method have to be called before calling functions
        that use self.dim.
        :param signal: an input signal
        """
        if isinstance(signal, list):
            signal = np.array(signal)
        if not (isinstance(signal, np.ndarray)):
            print("1")
            raise TypeError()
        if len(signal.shape) != 1:
            print("2")
            raise TypeError()
        self.dim = signal.shape[0]
        if not (hasattr(self, "dim")):
            self.dim = signal.shape[0]
        else:
            if signal.shape[0] != self.dim:
                print("3")
                raise TypeError()
        return signal

    # checked
    def __add_node(self, signal: np.ndarray, label: str, current_training_index: int):
        n = self.nodes.shape[0]
        self.nodes.resize((n + 1, self.dim))
        self.nodes[-1, :] = signal
        if current_training_index:
            self.nodes_training_index.append([current_training_index])
        self.winning_times.append(1)
        self.adjacent_mat.resize((n + 1, n + 1))  # ??
        self.N.append(1)
        self.density.append(0)
        self.s.append([0, 0])
        self.G.append(1)
        self.won.append(False)
        self.node_labels.append(label)

    # checked
    def __find_nearest_nodes(self, num: int, signal: np.ndarray):
        n = self.nodes.shape[0]
        indexes = [0] * num
        nodes_dists = [0.0] * num
        D, _, _ = self.calculate_qbatch_distance(signal, list(range(n)))
        assert D.shape[0] == n
        for i in range(num):
            indexes[i] = np.nanargmin(D)
            nodes_dists[i] = D[indexes[i]]
            D[indexes[i]] = float("nan")
        return indexes, nodes_dists

    # checked
    def __calculate_similarity_thresholds(self, node_indexes):
        sim_thresholds = []
        for i in node_indexes:
            pals = self.adjacent_mat[i, :]
            if len(pals) == 0:
                idx, sq_dists = self.__find_nearest_nodes(2, self.nodes[i, :])
                sim_thresholds.append(sq_dists[1])
            else:
                q_indexes = [k[1] for k in pals.keys()]
                D, _, _ = self.calculate_qbatch_distance(self.nodes[i, :], q_indexes)
                max_dist = np.amax(D)
                sim_thresholds.append(max_dist)
        return sim_thresholds

    # checked
    def __add_edge(self, node_indexes):
        self.__set_edge_weight(node_indexes, 1)

    # checked
    def __increment_edge_ages(self, winner_index):
        for k, v in self.adjacent_mat[winner_index, :].items():
            self.__set_edge_weight((winner_index, k[1]), v + 1)

    # checked
    def __set_edge_weight(self, index, weight):
        self.adjacent_mat[index[0], index[1]] = weight
        self.adjacent_mat[index[1], index[0]] = weight

    # checked
    def __update_winner(self, winner_index, signal):
        w = self.nodes[winner_index]
        self.nodes[winner_index] = w + (signal - w) / self.winning_times[winner_index]

    # checked, maybe fixed the problem
    def __update_density(self, winner_index):
        self.winning_times[winner_index] += 1

        pals = self.adjacent_mat[winner_index]
        pal_indexes = []
        for k in pals.keys():
            pal_indexes.append(k[1])
        if len(pal_indexes) != 0:
            # print(len(pal_indexes))
            _, eu, co = self.calculate_qbatch_distance(
                self.nodes[winner_index, :], pal_indexes
            )
            sum_eu = (eu - self.eumin) / (1 + self.eumax - self.eumin)
            sum_co = (co - self.comin) / (1 + self.comax - self.comin)
            sum_eu = np.sum(sum_eu)
            sum_co = np.sum(sum_co)
            p0 = 1 - sum_eu / (len(pal_indexes) * (self.gamma**self.dim))
            p1 = 1 - (sum_co / len(pal_indexes)) * (1 - 1 / (self.gamma**self.dim))

            self.s[winner_index][0] += p0
            self.s[winner_index][1] += p1
            if self.N[winner_index] == 0:
                self.density[winner_index] = (
                    self.s[winner_index][0] + self.s[winner_index][1]
                )
            else:
                self.density[winner_index] = (
                    self.s[winner_index][0] + self.s[winner_index][1]
                ) / self.N[winner_index]

        if self.s[winner_index][0] > 0 and self.s[winner_index][1] > 0:
            self.won[winner_index] = True

    def __update_all_density(self, new_eumax, new_eumin, new_comax, new_comin):
        k = [
            (1 + self.eumax - self.eumin) / (1 + new_eumax - new_eumin),
            (1 + self.comax - self.comin) / (1 + new_comax - new_comin),
        ]
        b = [
            (self.eumin - new_eumin)
            / ((self.gamma**self.dim) * (1 + new_eumax - new_eumin)),
            (1 - (1 / (self.gamma**self.dim)))
            * (self.comin - new_comin)
            / (1 + new_comax - new_comin),
        ]
        for i in range(len(self.nodes)):
            self.s[i][0] = (
                k[0] * (self.s[i][0] - self.G[i]) - self.G[i] * b[0] + self.G[i]
            )
            self.s[i][1] = (
                k[1] * (self.s[i][1] - self.G[i]) - self.G[i] * b[1] + self.G[i]
            )
            if self.N[i] == 0:
                self.density[i] = self.s[i][0] + self.s[i][1]
            else:
                self.density[i] = (self.s[i][0] + self.s[i][1]) / self.N[i]
            if self.s[i][0] > 0 and self.s[i][1] > 0:
                self.won[i] = True

    # checked
    def __update_adjacent_nodes(self, winner_index, signal):
        pals = self.adjacent_mat[winner_index]
        for k in pals.keys():
            i = k[1]
            w = self.nodes[i]
            self.nodes[i] = w + (signal - w) / (100 * self.winning_times[i])

    # checked
    def __delete_nodes(self, indexes):
        if not indexes:
            return
        n = len(self.winning_times)
        self.nodes = np.delete(self.nodes, indexes, 0)
        remained_indexes = list(set([i for i in range(n)]) - set(indexes))
        self.winning_times = [self.winning_times[i] for i in remained_indexes]
        self.N = [self.N[i] for i in remained_indexes]
        self.G = [self.G[i] for i in remained_indexes]
        self.density = [self.density[i] for i in remained_indexes]
        self.node_labels = [self.node_labels[i] for i in remained_indexes]
        self.nodes_training_index = [
            self.nodes_training_index[i] for i in remained_indexes
        ]
        self.won = [self.won[i] for i in remained_indexes]
        self.s = [self.s[i] for i in remained_indexes]
        self.__delete_nodes_from_adjacent_mat(indexes, n, len(remained_indexes))

    # checked
    def __delete_nodes_from_adjacent_mat(self, indexes, prev_n, next_n):
        while indexes:
            next_adjacent_mat = dok_matrix((prev_n, prev_n))
            for key1, key2 in self.adjacent_mat.keys():
                if key1 == indexes[0] or key2 == indexes[0]:
                    continue
                if key1 > indexes[0]:
                    new_key1 = key1 - 1
                else:
                    new_key1 = key1
                if key2 > indexes[0]:
                    new_key2 = key2 - 1
                else:
                    new_key2 = key2
                # Because dok_matrix.__getitem__ is slow,
                # access as dictionary.
                next_adjacent_mat[new_key1, new_key2] = super(
                    dok_matrix, self.adjacent_mat
                ).__getitem__((key1, key2))
            self.adjacent_mat = next_adjacent_mat.copy()
            indexes = [i - 1 for i in indexes]
            indexes.pop(0)
        self.adjacent_mat.resize((next_n, next_n))

    # checked
    def __delete_noise_nodes(self):
        n = len(self.winning_times)
        # print(n)
        noise_indexes = []
        mean_density_all = np.mean(self.density)
        # print(mean_density_all)
        for i in range(n):
            if (
                len(self.adjacent_mat[i, :]) == 2
                and self.density[i] < self.c1 * mean_density_all
            ):
                noise_indexes.append(i)
            elif (
                len(self.adjacent_mat[i, :]) == 1
                and self.density[i] < self.c2 * mean_density_all
            ):
                noise_indexes.append(i)
            elif len(self.adjacent_mat[i, :]) == 0:
                noise_indexes.append(i)
        print("Removed noise node:", len(noise_indexes))
        self.__delete_nodes(noise_indexes)
