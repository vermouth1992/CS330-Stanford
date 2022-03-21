"""
Classes defining user and item latent representations in
factorization models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class ZeroEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.

    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class MultiTaskNet(nn.Module):
    """
    Multitask factorization representation.

    Encodes both users and items as an embedding layer; the likelihood score
    for a user-item pair is given by the dot product of the item
    and user latent vectors. The numerical score is predicted using a small MLP.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    layer_sizes: list
        List of layer sizes to for the regression network.
    sparse: boolean, optional
        Use sparse gradients.
    embedding_sharing: boolean, optional
        Share embedding representations for both tasks.

    """

    def __init__(self, num_users, num_items, embedding_dim=32, layer_sizes=[96, 64],
                 sparse=False, embedding_sharing=True):
        super().__init__()

        self.embedding_dim = embedding_dim

        # ********************************************************
        # ******************* YOUR CODE HERE *********************
        # ********************************************************

        self.U = ScaledEmbedding(num_embeddings=num_users, embedding_dim=embedding_dim, sparse=sparse)
        self.Q = ScaledEmbedding(num_embeddings=num_items, embedding_dim=embedding_dim, sparse=sparse)
        self.A = ZeroEmbedding(num_embeddings=num_users, embedding_dim=1)
        self.B = ZeroEmbedding(num_embeddings=num_items, embedding_dim=1)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 3, layer_sizes[0]),
            nn.ReLU(),
            nn.Linear(layer_sizes[0], layer_sizes[1]),
            nn.ReLU(),
            nn.Linear(layer_sizes[1], 1)
        )
        if embedding_sharing:
            self.regression_U = self.U
            self.regression_Q = self.Q
        else:
            self.regression_U = ScaledEmbedding(num_embeddings=num_users, embedding_dim=embedding_dim, sparse=sparse)
            self.regression_Q = ScaledEmbedding(num_embeddings=num_users, embedding_dim=embedding_dim, sparse=sparse)

        # ********************************************************
        # ******************* YOUR CODE HERE *********************
        # ********************************************************

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            A tensor of integer user IDs of shape (batch,)
        item_ids: tensor
            A tensor of integer item IDs of shape (batch,)

        Returns
        -------

        predictions: tensor
            Tensor of user-item interaction predictions of 
            shape (batch,). This corresponds to p_ij in the 
            assignment.
        score: tensor
            Tensor of user-item score predictions of shape 
            (batch,). This corresponds to r_ij in the 
            assignment.
        """

        # ********************************************************
        # ******************* YOUR CODE HERE *********************
        # ********************************************************
        u = self.U(user_ids)  # (None, embedding_dim)
        q = self.Q(item_ids)  # (None, embedding_dim)
        a = self.A(user_ids)  # (None, 1)
        b = self.B(item_ids)  # (None, 1)
        predictions = torch.sum(u * q, dim=-1) + torch.squeeze(a, dim=-1) + torch.squeeze(b, dim=-1)

        u = self.regression_U(user_ids)
        q = self.regression_Q(item_ids)
        input = torch.cat([u, q, u * q], dim=-1)
        score = self.mlp(input)
        score = torch.squeeze(score, dim=-1)

        # ********************************************************
        # ********************************************************
        # ********************************************************
        return predictions, score
