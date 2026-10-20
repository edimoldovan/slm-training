from mlx_lm.models.llama import Model, ModelArgs

CONFIG = {
	"model_type": "llama",
	"hidden_size": 512,
	"num_hidden_layers": 8,
	"intermediate_size": 1376,
	"num_attention_heads": 8,
	"num_key_value_heads": 8,
	"vocab_size": 8192,
	"max_position_embeddings": 1024,
	"rms_norm_eps": 1e-5,
	"rope_theta": 10000.0,
	"tie_word_embeddings": True,
	"attention_bias": False,
	"mlp_bias": False,
}

def build():
	return Model(ModelArgs.from_dict(CONFIG))
