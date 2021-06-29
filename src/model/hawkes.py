import torch


class MFHawkes(torch.nn.Module):
    def __init__(self, K, feature_dim=128):
        """

        :param K: number of identity
        :param feature_dim:
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.K = K

        self.identity_embedding = torch.nn.Embedding(K, feature_dim, 0)
        self.u = torch.nn.Embedding(K, 1, 0)
        self.beta = torch.nn.Parameter(torch.Tensor(1))

    def decay_function(self):
        pass

    def forward(self, identities: torch.Tensor, times, mask):
        """

        :param identities: shape(batch_size, event_length)
        :param times:  shape(batch_size, event_length)
        :param mask:  shape(batch_size, event_length)
        :return:
        """
        input_shape = times.shape
        repeated_times = times.unsqueeze(2).expand(*input_shape,
                                                   input_shape[-1])  # shape(batch_size, event_length, event_length)

        # bug warning: repeat_times should be transposed??
        time_difference = repeated_times - times  # times' shape(batch_size, event_length), can this be broadcast?

        weighted_exp = self.beta * torch.exp(
            -self.beta * time_difference)  # shape(batch_size, event_length, event_length)

        embedding = self.identity_embedding(identities)  # shape(batch_size, event_length, feature_dim)
        attention_alpha = embedding * embedding.T  # shape(batch_size, event_length, event_length)

        before_sum = torch.tril(attention_alpha, diagonal=-1)
        mask_sum = before_sum * mask  # shape(batch_size, event_length, event_length) *  shape(batch_size, event_length) broadcast ??

        u_k = self.u(identities).squeeze(-1)  # shape(batch_size, event_length)
        lambda_of_event = mask_sum.sum(-1) * u_k  # shape(batch_size, event_length)
        log_lambda = torch.log(lambda_of_event)
        return log_lambda.sum(-1)
