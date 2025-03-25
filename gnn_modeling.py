import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, GCNConv, GINConv, SAGEConv
from torch_geometric.nn.norm import LayerNorm, BatchNorm, GraphNorm
from torch_geometric.nn.pool import global_add_pool, global_mean_pool, global_max_pool


class GNN(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        
        self.blocks = self._build_blocks()
        self.pool = self._get_pool()
        #self.head = self._get_head()
    
    def _build_blocks(self):
        blocks = nn.ModuleList()
        for i,block_cfg in enumerate(self.cfg.blocks):
            block = nn.ModuleDict()
            if block_cfg['block_type'] == 'mlp':
                block['block'] = nn.Linear(block_cfg['input_dim'],block_cfg['out_dim'])
            elif block_cfg['block_type'] == 'gcn':
                block['block'] = GCNConv(block_cfg['input_dim'],block_cfg['out_dim'])
            elif block_cfg['block_type'] == 'sage':
                block['block'] = SAGEConv(block_cfg['input_dim'],block_cfg['out_dim'],aggr=block_cfg['aggr'])
            elif block_cfg['block_type'] == 'gat':
                block['block'] = GATConv(block_cfg['input_dim'],block_cfg['out_dim'],heads=block_cfg['heads'],edge_dim=self.cfg.edge_dim)
            elif block_cfg['block_type'] == 'gatv2':
                block['block'] = GATv2Conv(block_cfg['input_dim'],block_cfg['out_dim'],heads=block_cfg['heads'],edge_dim=self.cfg.edge_dim, residual=True)
            elif block_cfg['block_type'] == 'gin':
                block['block'] = GINConv(nn.Linear(block_cfg['input_dim'],block_cfg['out_dim'],heads=block_cfg['heads']),)
                
            if block_cfg['act'] == 'GELU':
                block['act'] = nn.GELU()
            elif block_cfg['act'] == 'Identity':
                block['act'] = nn.Identity()
            if block_cfg['block_norm']:
                if block_cfg['block_norm'] == 'layer_norm':
                    block['block_norm'] = LayerNorm(block_cfg['out_dim'] * block_cfg['heads'])
                elif block_cfg['block_norm'] == 'batch_norm':
                    block['block_norm'] = BatchNorm(block_cfg['out_dim'] * block_cfg['heads'])
                elif block_cfg['block_norm'] == 'graph_norm':
                    block['block_norm'] = GraphNorm(block_cfg['out_dim'] * block_cfg['heads'])
            
            block['drop'] = nn.Dropout(block_cfg['drop_p'])
            blocks.append(block)
            
        return blocks
            
    def _get_pool(self):
        if self.cfg.pool_type == 'mean':
            return global_mean_pool
        elif self.cfg.pool_type == 'max':
            return global_max_pool
        elif self.cfg.pool_type == 'sum':
            return global_add_pool
        else:
            return nn.Identity()
    
    def _get_head(self):
        return nn.Linear(self.cfg.blocks[-1]['out_dim'],self.cfg.num_classes,bias=self.cfg.head_bias)
    
    def forward(self, x, edge_index, edge_attr, batch, cls_idxes=None,return_feats=True):
        resid_x = torch.clone(x)
        for block,block_cfg in zip(self.blocks, self.cfg.blocks):
            if block_cfg['block_type'] == 'mlp':
                x = block['block'](x)
            else:
                x = block['block'](x,edge_index.long(),edge_attr=edge_attr.long())
            
            x = block['act'](x)
            if block_cfg['block_norm']:
                if block_cfg['block_norm'] == 'batch_norm':
                    x = block['block_norm'](x)
                else:
                    x = block['block_norm'](x, batch)
            x = block['drop'](x)
        
        x += resid_x
        if self.cfg.pool_type == 'no':
            
            return x.view(batch.max()+1,self.cfg.num_nodes,-1)[:,:self.cfg.max_nodes,:]
        if self.cfg.pool_type == 'cls':
            x = x[cls_idxes]
        else:
            x = self.pool(x, batch)
        if return_feats:
            return x
        else:
            return self.head(x)
