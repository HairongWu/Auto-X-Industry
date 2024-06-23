#include "llama_demo.h"

static autox_err_t malloc_run_state(RunState* s, Config* p) {
	// we calloc instead of malloc to keep valgrind happy
	int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
	s->x = (float*)calloc(p->dim, sizeof(float));
	s->xb = (float*)calloc(p->dim, sizeof(float));
	s->xb2 = (float*)calloc(p->dim, sizeof(float));
	s->hb = (float*)calloc(p->hidden_dim, sizeof(float));
	s->hb2 = (float*)calloc(p->hidden_dim, sizeof(float));
	s->q = (float*)calloc(p->dim, sizeof(float));
	s->key_cache = (float*)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
	s->value_cache = (float*)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
	s->att = (float*)calloc(p->n_heads * p->seq_len, sizeof(float));
	s->logits = (float*)calloc(p->vocab_size, sizeof(float));
	// ensure all mallocs went fine
	if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
		|| !s->key_cache || !s->value_cache || !s->att || !s->logits) {
		return AUTOX_ERR_NO_MEM;
	}
	return AUTOX_OK;
}

static void free_run_state(RunState* s) {
	free(s->x);
	free(s->xb);
	free(s->xb2);
	free(s->hb);
	free(s->hb2);
	free(s->q);
	free(s->att);
	free(s->logits);
	free(s->key_cache);
	free(s->value_cache);
}

static void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
	sampler->vocab_size = vocab_size;
	sampler->temperature = temperature;
	sampler->topp = topp;
	sampler->rng_state = rng_seed;
	// buffer only used with nucleus sampling; may not need but it's ~small
	sampler->probindex = (ProbIndex*)malloc(sampler->vocab_size * sizeof(ProbIndex));
}

static void free_sampler(Sampler* sampler) {
	free(sampler->probindex);
}

static void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
	int head_size = p->dim / p->n_heads;
	// make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
	unsigned long long n_layers = p->n_layers;
	w->token_embedding_table = ptr;
	ptr += p->vocab_size * p->dim;
	w->rms_att_weight = ptr;
	ptr += n_layers * p->dim;
	w->wq = ptr;
	ptr += n_layers * p->dim * (p->n_heads * head_size);
	w->wk = ptr;
	ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
	w->wv = ptr;
	ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
	w->wo = ptr;
	ptr += n_layers * (p->n_heads * head_size) * p->dim;
	w->rms_ffn_weight = ptr;
	ptr += n_layers * p->dim;
	w->w1 = ptr;
	ptr += n_layers * p->dim * p->hidden_dim;
	w->w2 = ptr;
	ptr += n_layers * p->hidden_dim * p->dim;
	w->w3 = ptr;
	ptr += n_layers * p->dim * p->hidden_dim;
	w->rms_final_weight = ptr;
	ptr += p->dim;
	ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
	ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
	w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

static autox_err_t read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
	int* fd, float** data, uint32_t* file_size) {
	FILE *file;
	fopen_s(&file, checkpoint, "rb");
	if (!file) { return AUTOX_ERR_NOT_FOUND; }
	// read in the config header
	if (fread(config, sizeof(Config), 1, file) != 1) { return AUTOX_FAIL; }
	// negative vocab size is hacky way of signaling unshared weights. bit yikes.
	int shared_weights = config->vocab_size > 0 ? 1 : 0;
	config->vocab_size = abs(config->vocab_size);
	// figure out the file size
	fseek(file, 0, SEEK_END); // move file pointer to end of file
	*file_size = ftell(file); // get the file size, in bytes

	fseek(file, 0, SEEK_SET);
	*data = (float*)calloc(*file_size / 4, sizeof(float));
	for (uint32_t i = 0; i < *file_size / 4; i++)
	{
		fread(*data + i, sizeof(float), 1, file);
	}
	fclose(file);
	float* weights_ptr = *data + sizeof(Config) / sizeof(float);
	memory_map_weights(weights, config, weights_ptr, shared_weights);
	return AUTOX_OK;
}

static autox_err_t build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
	// i should have written the vocab_size into the tokenizer file... sigh
	t->vocab_size = vocab_size;
	// malloc space to hold the scores and the strings
	t->vocab = (char**)malloc(vocab_size * sizeof(char*));
	t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
	t->sorted_vocab = NULL; // initialized lazily
	for (int i = 0; i < 256; i++) {
		t->byte_pieces[i * 2] = (unsigned char)i;
		t->byte_pieces[i * 2 + 1] = '\0';
	}
	// read in the file
	FILE *file;
	fopen_s(&file, tokenizer_path, "rb");
	if (!file) { return AUTOX_ERR_NOT_FOUND; }
	if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { return AUTOX_FAIL; }
	int len;
	for (int i = 0; i < vocab_size; i++) {
		if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { return AUTOX_FAIL; }
		if (fread(&len, sizeof(int), 1, file) != 1) { return AUTOX_FAIL; }
		t->vocab[i] = (char *)malloc(len + 1);
		if (fread(t->vocab[i], len, 1, file) != 1) { return AUTOX_FAIL; }
		t->vocab[i][len] = '\0'; // add the string terminating token
	}
	fclose(file);
	return AUTOX_OK;
}

static void free_tokenizer(Tokenizer* t) {
	for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
	free(t->vocab);
	free(t->vocab_scores);
	free(t->sorted_vocab);
}
static autox_err_t build_transformer(Transformer *t, char* checkpoint_path) {
	// read in the Config and the Weights from the checkpoint
	autox_err_t err = read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
	if (err == AUTOX_OK)
	{
		// allocate the RunState buffers
		malloc_run_state(&t->state, &t->config);
		return AUTOX_OK;
	}
	return AUTOX_FAIL;
}

static  void free_transformer(Transformer* t) {
	// close the memory mapping
	if (t->data != NULL) { free(t->data); }
	// free the RunState buffers
	free_run_state(&t->state);
}

autox_err_t run_llama2(char* checkpoint_path, char* tokenizer_path, char* prompt)
{
	autox_err_t err = AUTOX_OK;
	float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
	float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
	int steps = 256;            // number of steps to run for
	unsigned long long rng_seed = 0; // seed rng with time by default

	// parameter validation/overrides
	if (temperature < 0.0) temperature = 0.0f;
	if (topp < 0.0 || 1.0 < topp) topp = 0.9f;
	if (steps < 0) steps = 0;

	// build the Transformer via the model .bin file
	Transformer transformer;
	err = build_transformer(&transformer, checkpoint_path);
	if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // override to ~max length

	// build the Tokenizer via the tokenizer .bin file
	Tokenizer tokenizer;
	err = build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

	// build the Sampler
	Sampler sampler;
	build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

	char *pieces = (char*)malloc(25600);
	err = llama2_generate(&pieces, &transformer, &tokenizer, &sampler, prompt, steps);
	printf("%s", pieces);
	// memory and file handles cleanup
	free_sampler(&sampler);
	free_tokenizer(&tokenizer);
	free_transformer(&transformer);

	free(pieces);

	return err;
}

autox_err_t run_llama3(char* checkpoint_path, char* tokenizer_path, char* prompt)
{
	autox_err_t err = AUTOX_OK;
	int max_new_tokens = 50; 

	// build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (max_new_tokens > transformer.config.seq_len)
        max_new_tokens = transformer.config.seq_len; // override to ~max length

	// build the Tokenizer via the tokenizer .bin file
	Tokenizer tokenizer;
	err = build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

	char *pieces = (char*)malloc(25600);
	err = llama3_generate(&pieces, &transformer, &tokenizer, prompt, max_new_tokens);
	printf("%s", pieces);
	// memory and file handles cleanup
	free_tokenizer(&tokenizer);
	free_transformer(&transformer);

	free(pieces);

	return err;
}