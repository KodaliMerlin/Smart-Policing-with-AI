import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.explain import Explainer, GNNExplainer

class SpatioTemporalGAT(torch.nn.Module):
    def __init__(self):
        super(SpatioTemporalGAT, self).__init__()
        self.conv1 = GATConv(in_channels=3, out_channels=8, heads=8, dropout=0.6)
        self.conv2 = GATConv(in_channels=8 * 8, out_channels=2, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(F.dropout(x, p=0.6, training=self.training), edge_index))
        x = self.conv2(F.dropout(x, p=0.6, training=self.training), edge_index)
        return F.log_softmax(x, dim=1)

def train_model(model, data, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    class_weights = torch.tensor([1.0, 50.0])
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask], weight=class_weights)
        loss.backward()
        optimizer.step()
    return model

def get_explainer(model):
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        ),
    )
