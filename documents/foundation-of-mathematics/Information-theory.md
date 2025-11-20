# Information Theory
## Self-Information and Entropy
- Self-Information: The self-information $ I(x) $ of an event $ x $ with probability $ P(x) $ is defined as:
  $$ I(x) = -\log_b(P(x)) $$
  where $ b $ is the base of the logarithm (commonly 2 for bits, e for nats).
- Entropy: The entropy $ H(X) $ of a discrete random variable $ X $ with possible outcomes $ x_1, x_2, \ldots, x_n $ and probabilities $ P(x_1), P(x_2), \ldots, P(x_n) $ is defined as:
  $$ H(X) = -\sum_{i=1}^{n} P(x_i) \log_b(P(x_i)) $$
  Entropy measures the average uncertainty or information content in the random variable.
## Encoding and Compression
- Source Coding Theorem: The average length of the optimal encoding of a source cannot be less than its entropy. This theorem provides a theoretical limit for lossless data compression.
- Huffman Coding: A widely used algorithm for lossless data compression that constructs a variable-length prefix code based on the frequencies of the symbols.
## Joint, Conditional, and Mutual Information
- Joint Entropy: The joint entropy $ H(X, Y) $ of two discrete random variables $ X $ and $ Y $ is defined as:
  $$ H(X, Y) = -\sum_{x,y} P(x,y) \log_b(P(x,y)) $$
- Conditional Entropy: The conditional entropy $ H(Y|X) $ of a random variable $ Y $ given another random variable $ X $ is defined as:
  $$ H(Y|X) = -\sum_{x,y} P(x,y) \log_b(P(y|x)) $$
- Mutual Information: The mutual information $ I(X;Y) $ between two random variables $ X $ and $ Y $ is defined as:
  $$ I(X;Y) = H(X) + H(Y) - H(X,Y) $$
  Mutual information measures the amount of information that one random variable contains about another.
## Cross-Entropy and Divergence
- Cross-Entropy: The cross-entropy $ H(P, Q) $ between two probability distributions $ P $ and $ Q $ is defined as:
  $$ H(P, Q) = -\sum_{x} P(x) \log_b(Q(x)) $$
- Kullback-Leibler Divergence: The Kullback-Leibler (KL) divergence $ D_{KL}(P || Q) $ from distribution $ Q $ to distribution $ P $ is defined as:
  $$ D_{KL}(P || Q) = \sum_{x} P(x) \log_b\left(\frac{P(x)}{Q(x)}\right) $
  KL divergence measures the difference between two probability distributions.
- Jensen-Shannon Divergence: A symmetric and finite measure of divergence between two probability distributions, defined as:
  $$ D_{JS}(P || Q) = \frac{1}{2} D_{KL}(P || M) + \frac{1}{2} D_{KL}(Q || M) $$
  where $ M = \frac{1}{2}(P + Q) $ is the average distribution.
- Wasserstein Distance: A measure of the distance between two probability distributions, defined as the minimum cost of transporting mass to transform one distribution into the other.
## Applications
- Data Compression: Techniques such as Huffman coding and arithmetic coding utilize information theory principles to reduce data size.
- Cryptography: Information theory provides a framework for understanding the security of encryption schemes and the concept of perfect secrecy.
- Machine Learning: Concepts like entropy and mutual information are used in feature selection, decision trees, and clustering algorithms.
- Communication Systems: Information theory underpins the design of efficient communication protocols and error-correcting codes.
