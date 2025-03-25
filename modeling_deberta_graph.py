import os
from typing import Optional, Union, Tuple

import torch
from transformers import DebertaV2PreTrainedModel, DebertaV2Model, DebertaV2Config, SiglipVisionConfig, \
    SiglipVisionModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import MultipleChoiceModelOutput
from transformers.models.deberta_v2.modeling_deberta_v2 import ContextPooler, StableDropout, DebertaV2Embeddings, \
    DebertaV2Encoder
import torch.nn as nn

from .gnn_modeling import GNN
from .poolings import MeanPooling, AttentionPooling


class GraphConfig(PretrainedConfig):
    def __init__(self,
                 pool_type='no',
                 num_nodes=64,
                 max_nodes=64,
                 edge_dim=17,
                 blocks=({},),
                 **kwargs):
        super().__init__()
        self.pool_type = pool_type
        self.num_nodes = num_nodes
        self.max_nodes = max_nodes
        self.edge_dim = edge_dim
        self.blocks = blocks

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the text config dict if we are loading from SiglipConfig
        if config_dict.get("model_type") == "siglip":
            config_dict = config_dict["text_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class MultiModalDebertaV2Config(DebertaV2Config):
    def __init__(self,
                 vision_config: Optional[SiglipVisionConfig] = None,
                 graph_config: Optional[GraphConfig] = None,
                 **kwargs):
        super().__init__(**kwargs)
        if vision_config is not None:
            self.vision_config = SiglipVisionConfig(**vision_config)
        else:
            self.vision_config = None
        if graph_config is not None:
            self.graph_config = GraphConfig(**graph_config)
        else:
            self.graph_config = None


class DebertaGraphV2Embeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        pad_token_id = getattr(config, "pad_token_id", 0)
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embedding_size, padding_idx=pad_token_id)

        self.position_biased_input = getattr(config, "position_biased_input", True)
        if not self.position_biased_input:
            self.position_embeddings = None
        else:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.embedding_size)

        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, self.embedding_size)

        if self.embedding_size != config.hidden_size:
            self.embed_proj = nn.Linear(self.embedding_size, config.hidden_size, bias=False)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.config = config

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(self, input_ids=None, gnn_ids=None, token_type_ids=None, position_ids=None, mask=None,
                inputs_embeds=None):
        seq_length = input_ids.shape[1] + gnn_ids.shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # if token_type_ids is None:
        #     token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
            inputs_embeds = torch.cat([gnn_ids, inputs_embeds], dim=1)

        if self.position_embeddings is not None:
            position_embeddings = self.position_embeddings(position_ids.long())
        else:
            position_embeddings = torch.zeros_like(inputs_embeds)

        embeddings = inputs_embeds
        if self.position_biased_input:
            embeddings += position_embeddings
        # if self.config.type_vocab_size > 0:
        #    token_type_embeddings = self.token_type_embeddings(token_type_ids)
        #    embeddings += token_type_embeddings

        if self.embedding_size != self.config.hidden_size:
            embeddings = self.embed_proj(embeddings)

        embeddings = self.LayerNorm(embeddings)

        if mask is not None:
            if mask.dim() != embeddings.dim():
                if mask.dim() == 4:
                    mask = mask.squeeze(1).squeeze(1)
                mask = mask.unsqueeze(2)
            mask = mask.to(embeddings.dtype)

            embeddings = embeddings * mask

        embeddings = self.dropout(embeddings)
        return embeddings


class MultiModalDebertaV2ForMultipleChoice(DebertaV2Model):
    config_class = MultiModalDebertaV2Config

    def __init__(self, config: MultiModalDebertaV2Config):
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels
        self.graph_encoder = GNN(config.graph_config)
        self.vision_encoder = SiglipVisionModel(config.vision_config)
        #self.vision_proj = nn.Linear(config.vision_config.hidden_size, config.hidden_size)
        self.vision_proj = nn.Conv1d(config.vision_config.hidden_size, config.hidden_size, kernel_size=8, stride=4)
        self.modal_embeds = nn.Embedding(4, config.hidden_size)
        self.embeddings = DebertaGraphV2Embeddings(config)
        self.encoder = DebertaV2Encoder(config)
        # self.pooler = MeanPooling()
        self.pooler = AttentionPooling(config.hidden_size)
        output_dim = config.hidden_size

        self.classification_heads = nn.ModuleDict({'csqa': nn.Linear(output_dim, 1),
                                                   'dream': nn.Linear(output_dim, 1),
                                                   'vqa': nn.Linear(output_dim, 1),
                                                   'text': nn.Linear(output_dim, 1),})
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        self.init_weights()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            pixel_values: Optional[torch.Tensor] = None,
            graph_encodes=None,
            task: str = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MultipleChoiceModelOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
        graph_embeds = self.graph_encoder(
            x=graph_encodes.x,
            edge_index=graph_encodes.edge_index,
            edge_attr=graph_encodes.edge_attr,
            batch=graph_encodes.batch,
        )

        if pixel_values is not None:
            #pixel_values = pixel_values.flatten(0, 1)
            image_embeds = self.vision_encoder(pixel_values=pixel_values)[0]
            image_embeds = self.vision_proj(image_embeds.permute(0, 2, 1)).permute(0, 2, 1)
            #image_embeds = self.vision_proj(image_embeds)
            image_embeds = image_embeds[:, None, :, :].repeat(1, num_choices, 1, 1).flatten(0, 1)
            #print(image_embeds.shape, graph_embeds.shape, self.modal_embeds.weight[0].repeat(graph_embeds.size(0), 1, 1).shape)
            modality_embeds = torch.concat([
                self.modal_embeds.weight[0].repeat(graph_embeds.size(0), 1, 1),
                image_embeds,
                self.modal_embeds.weight[1].repeat(graph_embeds.size(0), 1, 1),
                self.modal_embeds.weight[2].repeat(graph_embeds.size(0), 1, 1),
                graph_embeds,
                self.modal_embeds.weight[3].repeat(graph_embeds.size(0), 1, 1)

            ], dim=1)

        else:
            #print(graph_embeds.shape, self.modal_embeds.weight[0].repeat(graph_embeds.size(0), 1, 1).shape)
            modality_embeds = torch.concat([
                self.modal_embeds.weight[2].repeat(graph_embeds.size(0), 1, 1),
                graph_embeds,
                self.modal_embeds.weight[3].repeat(graph_embeds.size(0), 1, 1)
            ], dim=1)
        #print(graph_embeds.shape, modality_embeds.shape, flat_attention_mask.shape)
        modality_mask = torch.ones(modality_embeds.shape[0:2],
                                   dtype=flat_attention_mask.dtype,
                                   device=flat_attention_mask.device)
        
        full_attention_mask = torch.concat([modality_mask, flat_attention_mask], dim=1)
        embedding_output = self.embeddings(
            input_ids=flat_input_ids,
            gnn_ids=modality_embeds,
            token_type_ids=flat_token_type_ids,
            position_ids=flat_position_ids,
            mask=full_attention_mask,
            inputs_embeds=flat_inputs_embeds,
        )
        outputs = self.encoder(
            embedding_output,
            full_attention_mask,
            output_hidden_states=False,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer, full_attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classification_heads[task](pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )