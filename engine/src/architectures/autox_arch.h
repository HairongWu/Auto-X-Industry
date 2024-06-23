#pragma once

#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <string.h>

#include "../layers/autox_layers.h"

typedef int autox_err_t;

/* Definitions for error constants. */
#define AUTOX_OK          0       /*!< autox_err_t value indicating success (no error) */
#define AUTOX_FAIL        -1      /*!< Generic autox_err_t code indicating failure */

#define AUTOX_ERR_NO_MEM              0x101   /*!< Out of memory */
#define AUTOX_ERR_INVALID_ARG         0x102   /*!< Invalid argument */
#define AUTOX_ERR_INVALID_STATE       0x103   /*!< Invalid state */
#define AUTOX_ERR_INVALID_SIZE        0x104   /*!< Invalid size */
#define AUTOX_ERR_NOT_FOUND           0x105   /*!< Requested resource not found */
#define AUTOX_ERR_NOT_SUPPORTED       0x106   /*!< Operation or feature not supported */
#define AUTOX_ERR_TIMEOUT             0x107   /*!< Operation timed out */
#define AUTOX_ERR_INVALID_RESPONSE    0x108   /*!< Received response was invalid */
#define AUTOX_ERR_INVALID_CRC         0x109   /*!< CRC or checksum was invalid */
#define AUTOX_ERR_INVALID_VERSION     0x10A   /*!< Version was invalid */
#define AUTOX_ERR_INVALID_MAC         0x10B   /*!< MAC address was invalid */
#define AUTOX_ERR_NOT_FINISHED        0x10C   /*!< Operation has not fully completed */
#define AUTOX_ERR_NOT_ALLOWED         0x10D   /*!< Operation is not allowed */

#define AUTOX_ERR_WIFI_BASE           0x3000  /*!< Starting number of WiFi error codes */
#define AUTOX_ERR_MESH_BASE           0x4000  /*!< Starting number of MESH error codes */
#define AUTOX_ERR_FLASH_BASE          0x6000  /*!< Starting number of flash error codes */
#define AUTOX_ERR_HW_CRYPTO_BASE      0xc000  /*!< Starting number of HW cryptography module error codes */
#define AUTOX_ERR_MEMPROT_BASE        0xd000  /*!< Starting number of Memory Protection API error codes */

autox_err_t llama2_generate(char** pieces, Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, uint32_t steps);
autox_err_t llama3_generate(char** pieces, Transformer *transformer, Tokenizer *tokenizer, char *prompt, uint32_t max_new_tokens);