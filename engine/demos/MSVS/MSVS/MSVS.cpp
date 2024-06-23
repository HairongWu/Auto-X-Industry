#include "llama_demo.h"

int main(int argc, char *argv[])
{
	// default parameters
	char checkpoint_path[256] = "./stories15M.bin";
	char tokenizer_path[256] = "./tokenizer.bin";
	char prompt[256] = "One day, Lily met a Shoggoth";        // prompt string

	run_llama2(checkpoint_path, tokenizer_path, prompt);
	run_llama3(checkpoint_path, tokenizer_path, prompt);
}
