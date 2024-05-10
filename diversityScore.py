# code to handle diversity scoring of a dataset of vectors
import numpy as np
from vendi_score import vendi
from vendi_score import image_utils
from trainers import TrainerParams
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from copy import copy


class DiversityScore:
    """
    Class for computing the diversity score for a dataset via a similarity matrix.
    """

    def __init__(self, model, params, data):
        # check that the vectors parameter is a numpy array with two dimensions
        assert isinstance(params, TrainerParams), "params is not an instance of trainerParams"
        assert isinstance(model, nn.Module), "model is not an instance of nn.Module"
        assert isinstance(data, Dataset), "train_data is not an instance of Dataset"

        self.model = model
        self.params = params
        self.data = data

        # set up a data loader
        self.data_loader = torch.utils.data.DataLoader(self.data, batch_size=self.params.batch_size,
                                                       num_workers=self.params.num_workers)

    def __len__(self):
        return self.vectors.shape[0]

    def getEmbeddings(self):
        """
        Retrieves the compressed representations of a dataset from the middle layer of an autoencoder. Loads the most
        recent model checkpoint to compute representations.
        :return:
        vectors numpy.ndarray: 2D array of vector representations of each image.
        """
        # load the model from the latest checkpoint
        checkpoint = torch.load(self.model.save_path)

        # Load the model state dictionary
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # iterate over all images in the dataset and extract representations
        for i, (images, labels) in enumerate(self.data_loader):
            preds, images_compressed = self.model(images)

            # convert to a numpy array for easier handling after detaching gradient
            images_compressed = images_compressed.detach().numpy()

            # flatten all dimensions except the first
            flattened_size = np.prod(images_compressed.shape[1:])
            images_compressed = images_compressed.reshape((images_compressed.shape[0], flattened_size))

            # stack the results from each batch until we have run over the entire dataset
            if i == 0:
                # assign the values of images compressed to output
                vectors = copy(images_compressed)
            else:
                vectors = np.vstack((vectors, images_compressed))

        assert len(vectors.shape) == 2, "The output array should have two dimensions but it has {}".format(
            len(vectors.shape))

        return vectors

    def getPixelVectors(self):
        """
        Coverts a dataset of 2D images into an array of 1D pixel vectors
        :return:
        pixel_vectors: 2D pixel vector array
        """
        for i, (images, labels) in enumerate(self.data_loader):

            # convert to a numpy array for easier handling after detaching gradient
            image = images.numpy()

            # flatten all dimensions except the first
            flattened_size = np.prod(images.shape[1:])
            images_flat = images.reshape((images.shape[0], flattened_size))

            # stack the results from each batch until we have run over the entire dataset
            if i == 0:
                # assign the values of images compressed to output
                pixel_vectors = copy(images_flat)
            else:
                pixel_vectors = np.vstack((pixel_vectors, images_flat))

        assert len(pixel_vectors.shape) == 2, "The output array should have two dimensions but it has {}".format(
            len(pixel_vectors.shape))

        return pixel_vectors

    def cosineSimilarity(self, vectors):
        """
        Compute cosine similarity between multiple vectors. Sets a class attribute.

        Returns:
        numpy.ndarray: Cosine similarity matrix.
        """

        # Compute dot product of vectors
        dot_product = np.dot(vectors, vectors.T)

        # Compute norms of vectors
        norm = np.linalg.norm(vectors, axis=1)
        norm = norm[:, np.newaxis]  # Convert to column vector

        # Compute cosine similarity
        similarity_matrix = dot_product / (norm * norm.T)

        return similarity_matrix

    def vendiScore(self):
        """
        Calculates the vendi score from the similarity matrix. First checks if we have already computed or assigned
        a similarity matrix and generates one if necessary first.
        :return:
        float: The Vendi score for the dataset
        """
        embeddings = self.getEmbeddings()

        similarity_matrix = self.cosineSimilarity(embeddings)

        score = vendi.score_K(similarity_matrix)

        return score

    def vendiScorePixel(self):
        """
        Calculates the Vendi score directly from the cosine similarity matrix of pixel values.
        :return:
        float: The Vendi score for the dataset
        """
        vectors = self.getPixelVectors()

        similarity_matrix = self.cosineSimilarity(vectors)

        score = vendi.score_K(similarity_matrix)

        return score
