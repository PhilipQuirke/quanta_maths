import torch
import transformer_lens.utils as utils

from .useful_node import NodeLocation, UsefulNode

from .quanta_type import QuantaType


def test_questions_and_add_node_attention_tags(cfg, questions):
    cfg.useful_nodes.reset_node_tags(QuantaType.ATTENTION)

    logits, cache = cfg.main_model.run_with_cache(questions)

    all_attention_weights = []
    for layer in range(cfg.n_layers):
        attention_weights = cache["pattern", layer, "attn"]
        #print(attention_weights.shape) # 512, 4, 22, 22 = cfg.batch_size, cfg.n_heads, cfg.n_ctx, cfg.n_ctx

        average_attention_weights = attention_weights.mean(dim=0)
        #print(average_attention_weights.shape) # 4, 22, 22 = cfg.n_heads, cfg.n_ctx, cfg.n_ctx

        all_attention_weights += [average_attention_weights]


    for node in cfg.useful_nodes.nodes:
        if node.is_head:

            # Get attention weights for this token in this head
            layer_weights = all_attention_weights[node.layer]
            weights = layer_weights[node.num, node.position, :]

            top_tokens = torch.topk(weights, 4)
            total_attention = weights.sum()
            attention_percentage = top_tokens.values / total_attention * 100

            # Add up to 4 tags with percs per head
            for idx, token_idx in enumerate(top_tokens.indices):
                perc = attention_percentage[idx]
                if perc >= 1.0:
                    cfg.add_useful_node_tag( node, QuantaType.ATTENTION, f"P{token_idx}={perc:.0f}" )