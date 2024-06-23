#include "autox_arch.h"

// ----------------------------------------------------------------------------
// generation loop

autox_err_t is_safe(char *piece) {
	// piece might be a raw byte token, and we only want to print printable chars or whitespace
	// because some of the other bytes can be various control codes, backspace, etc.
	if (piece == NULL) { return AUTOX_FAIL; }
	if (piece[0] == '\0') { return AUTOX_FAIL; }
	if (piece[1] == '\0') {
		unsigned char byte_val = piece[0];
		if (!(isprint(byte_val) || isspace(byte_val))) {
			return AUTOX_FAIL;
		}
	}
	return AUTOX_OK;
}

autox_err_t llama2_generate(char** pieces, Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, uint32_t steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
	int32_t num_prompt_tokens = 0;
	int32_t* prompt_tokens = (int32_t*)malloc((strlen(prompt)+3) * sizeof(int32_t)); // +3 for '\0', ?BOS, ?EOS
    autox_tok_encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        return AUTOX_FAIL;
    }

    // start the main loop
	int32_t next;        // will store the next token in the sequence
	int32_t token = prompt_tokens[0]; // kick off with the first token in the prompt
	uint32_t pos = 0;     // position in the sequence
	uint32_t index = 0;
    while (pos < steps) {

        // forward the transformer to get logits for the next token
        float* logits = autox_transformer(transformer, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = autox_sample(sampler, logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        char* piece = autox_tok_decode(tokenizer, token, next);
		if (is_safe(piece) == AUTOX_OK)
		{
			strcpy(*pieces + index, piece);
			index += strlen(piece);
		}
        token = next;
    }
	*(*pieces + index + 1) = '\n';
    free(prompt_tokens);
    return AUTOX_OK;
}

autox_err_t llama3_generate(char** pieces, Transformer *transformer, Tokenizer *tokenizer, char *prompt, uint32_t max_new_tokens) {
	char *empty_prompt = (char *) "";
	if (prompt == NULL) { prompt = empty_prompt; }

	// encode the (string) prompt into tokens sequence
	int num_prompt_tokens = 0;
	int *prompt_tokens = (int *)malloc((strlen(prompt) + 3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
	autox_tok_encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
	// TODO: pretty dirty monkey patch for 'I have a dream' prompt.
	if (prompt_tokens[1] == 306) prompt_tokens[1] = 76;
	if (num_prompt_tokens < 1) {
		return AUTOX_FAIL;
	}

	// start the main loop
	long start = 0;  // used to time our code, only initialized after first iteration
	int next;        // will store the next token in the sequence
	int token = prompt_tokens[0]; // kick off with the first token in the prompt
	int pos = 0;     // position in the sequence
	uint32_t index = 0;
	while (pos < max_new_tokens - 1) {
		// forward the transformer to get logits for the next token
		float *logits = autox_transformer3(transformer, token, pos);

		// advance the state machine
		if (pos < num_prompt_tokens - 1) {
			// if we are still processing the input prompt, force the next prompt token
			next = prompt_tokens[pos + 1];
		}
		else {
			autox_argmax(logits, &next, transformer->config.vocab_size);
		}
		pos++;

		// data-dependent terminating condition: the BOS (=1) token delimits sequences
		if (next == 1) { break; }

		// print the token as string, decode it with the Tokenizer object
		char *piece = autox_tok_decode(tokenizer, token, next);
		if (is_safe(piece) == AUTOX_OK)
		{
			strcpy(*pieces + index, piece);
			index += strlen(piece);
		}
		token = next;
	}
	*(*pieces + index + 1) = '\n';
	free(prompt_tokens);
	return AUTOX_OK;
}
