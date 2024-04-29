from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import TokenClassifierOutput
from ankh import layers


class ConvBertForMultiLabelClassification(layers.BaseModule):
    def __init__(
        self,
        num_tokens: int,
        input_dim: int,
        nhead: int,
        hidden_dim: int,
        num_hidden_layers: int = 1,
        num_layers: int = 1,
        kernel_size: int = 7,
        dropout: float = 0.2,
        pooling: str = 'max',
        loss: str = 'BCE',
        weights  = None#tensor
    ):
        super(ConvBertForMultiLabelClassification, self).__init__(
            input_dim=input_dim,
            nhead=nhead,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            pooling=pooling,
        )
        """
            ConvBert model for multilabel classification task.
            Args:
                num_tokens: Integer specifying the number of tokens that should be the output of the final layer.
                input_dim: Dimension of the input embeddings.
                nhead: Integer specifying the number of heads for the `ConvBert` model.
                hidden_dim: Integer specifying the hidden dimension for the `ConvBert` model.
                num_hidden_layers: Integer specifying the number of hidden layers for the `ConvBert` model.
                num_layers: Integer specifying the number of `ConvBert` layers.
                kernel_size: Integer specifying the filter size for the `ConvBert` model. Default: 7
                dropout: Float specifying the dropout rate for the `ConvBert` model. Default: 0.2
                loss: String specifying the loss of the model.
        """

        self.model_type = "Transformer"
        self.num_labels = num_tokens
        self.decoder = nn.Linear(input_dim, num_tokens)
        self.weights = weights
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _compute_loss(self, logits, labels):
        if labels is not None and loss == 'BCE':
            loss = F.binary_cross_entropy_with_logits(
                logits.view(-1, self.num_labels), labels.view(-1, self.num_labels)
            )
        elif labels is not None and loss == 'focal':
            loss_fct = FocalLoss(weight=self.loss_weights, gamma=1.5) #manière plus propre d'implémenter les poids?
            loss = loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1, self.num_labels) 
        )
        
        else:
            loss = None
        return loss

    def forward(self, embed, labels=None):
        hidden_inputs = self.convbert_forward(embed)
        hidden_inputs = self.pooling(hidden_inputs)
        logits = self.decoder(hidden_inputs)
        loss = self._compute_loss(logits, labels)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )