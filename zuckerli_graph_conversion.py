import faiss
import numpy as np
from faiss.contrib.datasets import DatasetSIFT1M
from faiss.contrib.inspect_tools import get_NSG_neighbors


def prepare_index(ds, index_string: str) -> faiss.Index:
    index = faiss.index_factory(ds.d, index_string)
    index.train(ds.get_train())
    index.build_type = 0
    index.verbose = True
    # index.train(ds.get_train())
    database = ds.get_database()
    index.add(database)
    return index


class ZuckerliGraph:
    cs: np.ndarray
    nnode: int
    lims: np.ndarray
    neighbors: np.ndarray

    @staticmethod
    def read(filename):
        zg = ZuckerliGraph()
        with open(filename, "rb") as f:
            zg.cs = np.fromfile(f, dtype="uint64", count=1)
            # print(cs)
            zg.nnode = int(np.fromfile(f, dtype="uint32", count=1)[0])
            # print(nnode)
            zg.lims = np.fromfile(f, dtype="uint64", count=zg.nnode + 1)
            # print(lims)
            zg.neighbors = np.fromfile(f, dtype="uint32", count=zg.lims[-1])
            # print(neighbors)
        return zg

    def print(self):
        for i in range(self.nnode):
            print(f"node {i}: neighbors {self.neighbors[self.lims[i]:self.lims[i+1]]}")

    def write(self, filename):
        with open(filename, "wb") as f:
            np.array([132], dtype="uint64").tofile(f)
            np.array([self.nnode], dtype="uint32").tofile(f)
            self.lims.tofile(f)
            self.neighbors.tofile(f)


def main():
    index_str = "NSG32"
    index_nsg = prepare_index(DatasetSIFT1M(), index_str)
    neigh = get_NSG_neighbors(index_nsg.nsg)

    zg = ZuckerliGraph()
    zg.nnode = len(neigh)
    lims, nl = [0], []
    for n in range(zg.nnode):
        ne = neigh[n]
        ne = ne[ne >= 0]
        ne.sort()
        lims.append(lims[-1] + len(ne))
        nl.append(ne)
    zg.lims = np.array(lims, dtype="uint64")
    zg.neighbors = np.hstack(nl).astype("uint32")

    zg.write("graph.zuckerli")


if __name__ == "__main__":
    main()
