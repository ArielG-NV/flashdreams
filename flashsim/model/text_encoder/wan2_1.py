import os
import torch

from transformers import AutoTokenizer, UMT5EncoderModel

from flashsim.model.text_encoder.utils import prompt_clean, str2bool


class WanTextEncoder:

    MODEL_ID_14B = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
    MODEL_ID_1_3B = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    MODEL_ID_VER_2_2_TI2V_5B_720P = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

    def __init__(self, model_id_or_local_path: str = MODEL_ID_1_3B):
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            model_id_or_local_path,
            cache_dir=os.getenv("HF_HOME", None),
            subfolder="text_encoder",
            local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
        )
        self.text_encoder.eval().requires_grad_(False)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id_or_local_path,
            cache_dir=os.getenv("HF_HOME", None),
            subfolder="tokenizer",
            local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
        )

    def encode(self, text: list[str]) -> torch.Tensor:
        assert isinstance(text, list) and len(text) > 0, "text must be a non-empty list"
        text = [prompt_clean(u) for u in text]

        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=512,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(
            text_input_ids.to(self.text_encoder.device), mask.to(self.text_encoder.device)
        ).last_hidden_state
        prompt_embeds = prompt_embeds.to(self.text_encoder.device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(512 - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        return prompt_embeds

    def to(self, *args, **kwargs):
        """
        Moves the model to the specified device.
        """
        self.text_encoder.to(*args, **kwargs)
        return self


if __name__ == "__main__":
    dtype = torch.bfloat16
    device = torch.device("cuda")
    text_encoder = WanTextEncoder().to(device=device, dtype=dtype)
    text = ["hello world"]
    text_embeddings = text_encoder.encode(text)
    print(text_embeddings.shape)
    print(text_embeddings.dtype)
    print(text_embeddings.device)
    print(text_embeddings.sum()) # 1.9766