import numpy as np


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
            zg.nnode = int(np.fromfile(f, dtype="uint32", count=1)[0])
            zg.lims = np.fromfile(f, dtype="uint64", count=zg.nnode + 1)
            zg.neighbors = np.fromfile(f, dtype="uint32", count=zg.lims[-1])
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


if __name__ == "__main__":
    with open("graph.txt", "r") as f:
        neigh = [list(map(int, line.split(" "))) for line in f.readlines()]

    zg = ZuckerliGraph()
    zg.nnode = len(neigh)
    lims, nl = [0], []
    for n in range(zg.nnode):
        ne = list(sorted(neigh[n]))
        lims.append(lims[-1] + len(ne))
        nl.append(ne)
    zg.lims = np.array(lims, dtype="uint64")
    zg.neighbors = np.hstack(nl).astype("uint32")

    zg.write("graph.zuckerli")
