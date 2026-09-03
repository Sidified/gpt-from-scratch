"""Model and training hyperparameters, kept in one place so experiments are reproducible."""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 65        # set from the tokenizer at runtime
    block_size: int = 128       # context length: how many chars the model can look back
    n_embd: int = 256           # embedding dimension
    n_head: int = 4             # attention heads (n_embd must divide evenly by this)
    n_layer: int = 4            # stacked transformer blocks
    dropout: float = 0.2

    # ablation switches — flipped in Block 6
    use_layernorm: bool = True
    use_residual: bool = True
    use_pos_emb: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 64
    max_iters: int = 3000
    eval_interval: int = 250
    eval_iters: int = 100
    learning_rate: float = 3e-4
    seed: int = 1337
