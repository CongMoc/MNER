import torch
from torch import nn
from transformers import AutoModel, ViTModel
from modules.model_architecture.torchcrf import CRF


class TMP_MI(nn.Module):
    """Temporal Prompt Model for Multi-Image (TMP-MI), adapted from the
    official TPM-MI implementation (JinFish/MNER-MI, LREC-COLING 2024):
    text encoder swapped from BERT to XLM-R-large, image encoder swapped
    from ViT-base to ViT-large, matching the user's stated requirement.

    Mechanism is otherwise unchanged: each of the num_images image slots is
    encoded by ViT, tagged with a learned per-slot "temporal" embedding,
    self-attended over via a TransformerEncoderLayer, then projected into
    past_key_values (one K/V pair per text-encoder layer) that are injected
    as a soft prompt prefix into every layer of the text encoder -- not
    concatenated as extra tokens and not fused via cross-attention.
    """

    def __init__(self, bert_model_name, vit_model_name, num_labels_, num_images=4, cache_dir=None):
        super().__init__()
        self.num_labels = num_labels_
        self.num_images = num_images

        self.bert = AutoModel.from_pretrained(bert_model_name, cache_dir=cache_dir)
        self.vit = ViTModel.from_pretrained(vit_model_name, cache_dir=cache_dir, add_pooling_layer=False)

        self.config = self.bert.config
        self.n_layer = self.config.num_hidden_layers
        self.n_head = self.config.num_attention_heads
        self.n_embd = self.config.hidden_size // self.config.num_attention_heads
        vis_hidden_size = self.vit.config.hidden_size

        self.img_prompt_encoder = nn.Sequential(
            nn.Linear(in_features=vis_hidden_size, out_features=vis_hidden_size * 2),
            nn.Tanh(),
            nn.Linear(in_features=vis_hidden_size * 2, out_features=self.n_layer * 2 * self.config.hidden_size),
        )
        self.temporalEmbedding = nn.Embedding(num_images, vis_hidden_size)
        self.transformer = nn.TransformerEncoderLayer(d_model=vis_hidden_size, nhead=8)
        self.fc = nn.Linear(self.config.hidden_size, num_labels_)
        self.crf = CRF(num_labels_, batch_first=True)

    def get_prompt(self, img_features):
        bsz, img_len, _ = img_features.size()
        past_key_values = self.img_prompt_encoder(img_features)
        past_key_values = past_key_values.view(bsz, img_len, self.n_layer * 2, self.n_head, self.n_embd)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(self, input_ids, attention_mask, token_type_ids, img_feats, labels=None):
        bsz, img_len, channels, height, width = img_feats.shape
        pixel_values = img_feats.reshape(bsz * img_len, channels, height, width)
        img_features = self.vit(pixel_values=pixel_values).last_hidden_state[:, 0, :]  # CLS token
        img_features = img_features.reshape(bsz, img_len, -1)

        temp_embeddings = self.temporalEmbedding(torch.arange(img_len, device=img_features.device))
        temp_embeddings = temp_embeddings.unsqueeze(0)
        img_features = img_features + temp_embeddings
        temporal_features = self.transformer(img_features)

        past_key_values = self.get_prompt(temporal_features)
        prompt_guids_length = past_key_values[0][0].size(2)
        prompt_guids_mask = torch.ones((bsz, prompt_guids_length), device=attention_mask.device)
        prompt_attention_mask = torch.cat((prompt_guids_mask, attention_mask), dim=1)

        sequence_output = self.bert(
            input_ids=input_ids, attention_mask=prompt_attention_mask,
            token_type_ids=token_type_ids, past_key_values=past_key_values,
        ).last_hidden_state
        emissions = self.fc(sequence_output)

        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')
            return loss
        else:
            return self.crf.decode(emissions, mask=attention_mask.byte())
