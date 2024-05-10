# code to handle diversity scoring of a dataset of vectors
import numpy as np
from vendi_score import vendi

class DiversityScore:
    """
    Class for computing the diversity score for a dataset via a similarity matrix.
    """
    def __init__(self, vectors):
        # check that the vectors parameter is a numpy array with two dimensions
        assert isinstance(vectors, (np.ndarray, np.matrix)), "Variable vectors is not a NumPy array or matrix"
        assert len(vectors.shape) == 2, "The dimensionality of the variable vectors is {} not 2 as expected".format(len(vectors.shape[0]))

        self.vectors = vectors
        self.similarity_matrix = None

    def __len__(self):
        return self.vectors.shape[0]

    def cosineSimilarity(self):
        """
        Compute cosine similarity between multiple vectors. Sets a class attribute.

        Returns:
        numpy.ndarray: Cosine similarity matrix.
        """
        # Compute dot product of vectors
        dot_product = np.dot(self.vectors, self.vectors.T)

        # Compute norms of vectors
        norm = np.linalg.norm(self.vectors, axis=1)
        norm = norm[:, np.newaxis]  # Convert to column vector

        # Compute cosine similarity
        self.similarity_matrix = dot_product / (norm * norm.T)

        return self.similarity_matrix

    def vendiScore(self):
        """
        Calculates the vendi score from the similarity matrix. First checks if we have already computed or assigned
        a similarity matrix and generates one if necessary first.
        :return:
        float: The Vendi score for the dataset
        """
        # check if we already computed a similarity matrix, compute one now if necessary
        if self.similarity_matrix is None:
            self.cosineSimilarity()

        score = vendi.score_K(self.similarity_matrix)

        return score

