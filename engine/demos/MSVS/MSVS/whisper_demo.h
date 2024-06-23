#pragma once

#include "../../../src/autox_engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <string.h>

#include <cmath>
#include <cstring>
#include <fstream>
#include <regex>
#include <locale>
#include <codecvt>
#include <sstream>
#include <map>
#include <set>


// available whisper models
enum e_model {
	MODEL_UNKNOWN,
	MODEL_TINY,
	MODEL_BASE,
	MODEL_SMALL,
	MODEL_MEDIUM,
	MODEL_LARGE,
};

struct whisper_context;
struct whisper_state;
struct whisper_full_params;

typedef int32_t whisper_pos;
typedef int32_t whisper_token;
typedef int32_t whisper_seq_id;

enum whisper_alignment_heads_preset {
	WHISPER_AHEADS_NONE,
	WHISPER_AHEADS_N_TOP_MOST,  // All heads from the N-top-most text-layers
	WHISPER_AHEADS_CUSTOM,
	WHISPER_AHEADS_TINY_EN,
	WHISPER_AHEADS_TINY,
	WHISPER_AHEADS_BASE_EN,
	WHISPER_AHEADS_BASE,
	WHISPER_AHEADS_SMALL_EN,
	WHISPER_AHEADS_SMALL,
	WHISPER_AHEADS_MEDIUM_EN,
	WHISPER_AHEADS_MEDIUM,
	WHISPER_AHEADS_LARGE_V1,
	WHISPER_AHEADS_LARGE_V2,
	WHISPER_AHEADS_LARGE_V3,
};

typedef struct whisper_ahead {
	int n_text_layer;
	int n_head;
} whisper_ahead;

typedef struct whisper_aheads {
	size_t n_heads;
	const whisper_ahead * heads;
} whisper_aheads;

struct whisper_context_params {
	bool  use_gpu;
	bool  flash_attn;
	int   gpu_device;  // CUDA device

	// [EXPERIMENTAL] Token-level timestamps with DTW
	bool dtw_token_timestamps;
	enum whisper_alignment_heads_preset dtw_aheads_preset;

	int dtw_n_top;
	struct whisper_aheads dtw_aheads;

	size_t dtw_mem_size; // TODO: remove
};

typedef struct whisper_token_data {
	whisper_token id;  // token id
	whisper_token tid; // forced timestamp token id

	float p;           // probability of the token
	float plog;        // log probability of the token
	float pt;          // probability of the timestamp token
	float ptsum;       // sum of probabilities of all timestamp tokens

	// token-level timestamp data
	// do not use if you haven't computed token-level timestamps
	int64_t t0;        // start time of the token
	int64_t t1;        //   end time of the token

	// [EXPERIMENTAL] Token-level timestamps with DTW
	// do not use if you haven't computed token-level timestamps with dtw
	// Roughly corresponds to the moment in audio in which the token was output
	int64_t t_dtw;

	float vlen;        // voice length of the token
} whisper_token_data;

typedef struct whisper_model_loader {
	void * context;

	size_t(*read)(void * ctx, void * output, size_t read_size);
	bool(*eof)(void * ctx);
	void(*close)(void * ctx);
} whisper_model_loader;

// grammar element type
enum whisper_gretype {
	// end of rule definition
	WHISPER_GRETYPE_END = 0,

	// start of alternate definition for rule
	WHISPER_GRETYPE_ALT = 1,

	// non-terminal element: reference to rule
	WHISPER_GRETYPE_RULE_REF = 2,

	// terminal element: character (code point)
	WHISPER_GRETYPE_CHAR = 3,

	// inverse char(s) ([^a], [^a-b] [^abc])
	WHISPER_GRETYPE_CHAR_NOT = 4,

	// modifies a preceding WHISPER_GRETYPE_CHAR or LLAMA_GRETYPE_CHAR_ALT to
	// be an inclusive range ([a-z])
	WHISPER_GRETYPE_CHAR_RNG_UPPER = 5,

	// modifies a preceding WHISPER_GRETYPE_CHAR or
	// WHISPER_GRETYPE_CHAR_RNG_UPPER to add an alternate char to match ([ab], [a-zA])
	WHISPER_GRETYPE_CHAR_ALT = 6,
};

typedef struct whisper_grammar_element {
	enum whisper_gretype type;
	uint32_t             value; // Unicode code point or rule ID
} whisper_grammar_element;

struct whisper_vocab {
	using id = int32_t;
	using token = std::string;

	int n_vocab = 51864;

	std::map<token, id> token_to_id;
	std::map<id, token> id_to_token;

	// reference: https://github.com/openai/whisper/blob/248b6cb124225dd263bb9bd32d060b6517e067f8/whisper/tokenizer.py#L334-L349
	id token_eot = 50256;
	id token_sot = 50257;
	// task tokens (used only for multilingual models)
	id token_translate = 50357;
	id token_transcribe = 50358;
	// other special tokens
	id token_solm = 50359; // [TDRZ] used by tinydiarize models to indicate speaker turn
	id token_prev = 50360;
	id token_nosp = 50361;
	id token_not = 50362; // no timestamps
	id token_beg = 50363; // begin timestamps

	bool is_multilingual() const {
		return n_vocab >= 51865;
	}

	int num_languages() const {
		return n_vocab - 51765 - (is_multilingual() ? 1 : 0);
	}
};

struct whisper_segment {
	int64_t t0;
	int64_t t1;

	std::string text;

	std::vector<whisper_token_data> tokens;

	bool speaker_turn_next;
};

struct whisper_batch {
	int32_t n_tokens;

	whisper_token  *  token;
	whisper_pos    *  pos;
	int32_t        *  n_seq_id; // always 1, here for consistency with llama.cpp
	whisper_seq_id ** seq_id;   // null terminated
	int8_t         *  logits;
};

// medium
// hparams: {
// 'n_mels': 80,
// 'n_vocab': 51864,
// 'n_audio_ctx': 1500,
// 'n_audio_state': 1024,
// 'n_audio_head': 16,
// 'n_audio_layer': 24,
// 'n_text_ctx': 448,
// 'n_text_state': 1024,
// 'n_text_head': 16,
// 'n_text_layer': 24
// }
//
// default hparams (Whisper tiny)
struct whisper_hparams {
	int32_t n_vocab = 51864;
	int32_t n_audio_ctx = 1500;
	int32_t n_audio_state = 384;
	int32_t n_audio_head = 6;
	int32_t n_audio_layer = 4;
	int32_t n_text_ctx = 448;
	int32_t n_text_state = 384;
	int32_t n_text_head = 6;
	int32_t n_text_layer = 4;
	int32_t n_mels = 80;
	int32_t ftype = 1;
	float   eps = 1e-5f;
};

// audio encoding layer
struct whisper_layer_encoder {
	// encoder.blocks.*.attn_ln
	struct ggml_tensor * attn_ln_0_w;
	struct ggml_tensor * attn_ln_0_b;

	// encoder.blocks.*.attn.out
	struct ggml_tensor * attn_ln_1_w;
	struct ggml_tensor * attn_ln_1_b;

	// encoder.blocks.*.attn.query
	struct ggml_tensor * attn_q_w;
	struct ggml_tensor * attn_q_b;

	// encoder.blocks.*.attn.key
	struct ggml_tensor * attn_k_w;

	// encoder.blocks.*.attn.value
	struct ggml_tensor * attn_v_w;
	struct ggml_tensor * attn_v_b;

	// encoder.blocks.*.mlp_ln
	struct ggml_tensor * mlp_ln_w;
	struct ggml_tensor * mlp_ln_b;

	// encoder.blocks.*.mlp.0
	struct ggml_tensor * mlp_0_w;
	struct ggml_tensor * mlp_0_b;

	// encoder.blocks.*.mlp.2
	struct ggml_tensor * mlp_1_w;
	struct ggml_tensor * mlp_1_b;
};

// token decoding layer
struct whisper_layer_decoder {
	// decoder.blocks.*.attn_ln
	struct ggml_tensor * attn_ln_0_w;
	struct ggml_tensor * attn_ln_0_b;

	// decoder.blocks.*.attn.out
	struct ggml_tensor * attn_ln_1_w;
	struct ggml_tensor * attn_ln_1_b;

	// decoder.blocks.*.attn.query
	struct ggml_tensor * attn_q_w;
	struct ggml_tensor * attn_q_b;

	// decoder.blocks.*.attn.key
	struct ggml_tensor * attn_k_w;

	// decoder.blocks.*.attn.value
	struct ggml_tensor * attn_v_w;
	struct ggml_tensor * attn_v_b;

	// decoder.blocks.*.cross_attn_ln
	struct ggml_tensor * cross_attn_ln_0_w;
	struct ggml_tensor * cross_attn_ln_0_b;

	// decoder.blocks.*.cross_attn.out
	struct ggml_tensor * cross_attn_ln_1_w;
	struct ggml_tensor * cross_attn_ln_1_b;

	// decoder.blocks.*.cross_attn.query
	struct ggml_tensor * cross_attn_q_w;
	struct ggml_tensor * cross_attn_q_b;

	// decoder.blocks.*.cross_attn.key
	struct ggml_tensor * cross_attn_k_w;

	// decoder.blocks.*.cross_attn.value
	struct ggml_tensor * cross_attn_v_w;
	struct ggml_tensor * cross_attn_v_b;

	// decoder.blocks.*.mlp_ln
	struct ggml_tensor * mlp_ln_w;
	struct ggml_tensor * mlp_ln_b;

	// decoder.blocks.*.mlp.0
	struct ggml_tensor * mlp_0_w;
	struct ggml_tensor * mlp_0_b;

	// decoder.blocks.*.mlp.2
	struct ggml_tensor * mlp_1_w;
	struct ggml_tensor * mlp_1_b;
};

struct whisper_kv_cell {
	whisper_pos pos = -1;

	std::set<whisper_seq_id> seq_id;

	bool has_seq_id(const whisper_seq_id & id) const {
		return seq_id.find(id) != seq_id.end();
	}
};

