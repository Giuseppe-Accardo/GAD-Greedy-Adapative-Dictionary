# GAD(Greedy Adapative Dictionary)
For dictionary-based decompositions of certain types, it has been observed that there might be a link
between sparsity in the dictionary and sparsity in the decomposition. Sparsity in the dictionary has also
been associated with the derivation of fast and efficient dictionary learning algorithms. Therefore, in this
paper we present a greedy adaptive dictionary learning algorithm that sets out to find sparse atoms for
speech signals. The algorithm learns the dictionary atoms on data frames taken from a speech signal. It
iteratively extracts the data frame with minimum sparsity index, and adds this to the dictionary matrix.
The contribution of this atom to the data frames is then removed, and the process is repeated. The
algorithm is found to yield a sparse signal decomposition, supporting the hypothesis of a link between
sparsity in the decomposition and dictionary.
