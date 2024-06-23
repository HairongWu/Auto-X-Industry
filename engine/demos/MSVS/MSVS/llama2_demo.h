#pragma once

#include "../../../src/autox_engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <string.h>

autox_err_t run_llama2(char* checkpoint_path, char* tokenizer_path, char* prompt);
autox_err_t run_llama3(char* checkpoint_path, char* tokenizer_path, char* prompt);