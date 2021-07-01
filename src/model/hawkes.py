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

    def forward(self, identities: torch.Tensor, times, mask=None):
        """

        :param identities: shape(batch_size, event_length)
        :param times:  shape(batch_size, event_length)
        :param mask:  shape(batch_size, event_length)
        :return:
        """

        times = times/10000
        input_shape = times.shape
        repeated_times = times.unsqueeze(2).expand(*input_shape,
                                                   input_shape[-1])  # shape(batch_size, event_length, event_length)

        # bug warning: repeat_times should be transposed??
        event_times = times.unsqueeze(1).expand(*input_shape, input_shape[-1])
        time_difference = repeated_times - event_times  # times' shape(batch_size, event_length), can this be broadcast?

        weighted_exp = torch.abs(self.beta) * torch.exp(
            -torch.abs(self.beta) * time_difference)  # shape(batch_size, event_length, event_length)

        embedding = self.identity_embedding(identities)  # shape(batch_size, event_length, feature_dim)
        attention_alpha = torch.matmul(embedding,
                                       embedding.transpose(-1, -2))  # shape(batch_size, event_length, event_length)
        u_k = torch.abs(self.u(identities).squeeze(-1))  # shape(batch_size, event_length)

        # negative_integral
        Th = repeated_times[:, -1:, :].expand(*input_shape, input_shape[-1])  # TODO: enable mask The largest time in the times() not the  last one
        sum_time_difference = Th - event_times
        negative_integral = -u_k * (repeated_times[:, -1, :] - repeated_times[:, 1, :]) -(attention_alpha * (1- torch.exp(-torch.abs(self.beta)*sum_time_difference))).sum(-1)



        before_sum = torch.tril(attention_alpha * weighted_exp, diagonal=-1)
        if mask is not None:
            mask_sum = before_sum * mask  # shape(batch_size, event_length, event_length) *  shape(batch_size, event_length) broadcast ??
        else:
            mask_sum = before_sum

        lambda_of_event = mask_sum.sum(-1) + u_k  # shape(batch_size, event_length)
        lambda_of_event = torch.abs(lambda_of_event) + 0.000001
        log_lambda = -torch.log(lambda_of_event) - negative_integral
        return log_lambda.sum(-1)
