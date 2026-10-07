-include Makefile.local

BUILD_DIR ?= build-sm89
BUILD_TYPE ?= Release
CUDA_ARCH ?= 89
JOBS ?= 16
CUDA_JOBS ?= $(JOBS)
BUILD_TESTING ?= ON
BUILD_APPS ?= ON
BUILD_BENCHMARKS ?= OFF
BUILD_QWEN3_8_27B ?= ON
BUILD_QWEN3_6_35B_A3B ?= OFF

CMAKE ?= cmake
CTEST ?= ctest

CMAKE_ARGS := \
	-G Ninja \
	-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
	-DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH) \
	-DBUILD_TESTING=$(BUILD_TESTING) \
	-DNINFER_BUILD_APPS=$(BUILD_APPS) \
	-DNINFER_BUILD_BENCHMARKS=$(BUILD_BENCHMARKS) \
	-DNINFER_BUILD_QWEN3_8_27B=$(BUILD_QWEN3_8_27B) \
	-DNINFER_BUILD_QWEN3_6_35B_A3B=$(BUILD_QWEN3_6_35B_A3B)

ifneq ($(strip $(FFMPEG_LIBRARIES)),)
CMAKE_ARGS += "-DNINFER_FFMPEG_LIBRARIES=$(FFMPEG_LIBRARIES)"
endif
ifneq ($(strip $(CURL_LIBRARY)),)
CMAKE_ARGS += "-DNINFER_CURL_LIBRARY=$(CURL_LIBRARY)"
endif
ifneq ($(strip $(PYTHON)),)
CMAKE_ARGS += "-DPython3_EXECUTABLE=$(PYTHON)"
endif

.DEFAULT_GOAL := build

.PHONY: configure build verbose test serving-tests decode-verbose w8-verbose clean rebuild

configure:
	$(CMAKE) -S . -B "$(BUILD_DIR)" $(CMAKE_ARGS) $(EXTRA_CMAKE_ARGS)

build: configure
	$(CMAKE) --build "$(BUILD_DIR)" --parallel "$(JOBS)"

verbose: configure
	$(CMAKE) --build "$(BUILD_DIR)" --parallel "$(JOBS)" --verbose

test: build
	$(CTEST) --test-dir "$(BUILD_DIR)" --output-on-failure --parallel "$(JOBS)"

serving-tests: configure
	$(CMAKE) --build "$(BUILD_DIR)" --parallel "$(JOBS)" --target \
		ninfer_response_store_test \
		ninfer_responses_state_test \
		ninfer_responses_transport_test \
		ninfer_responses_schema_test \
		ninfer_openai_schema_test \
		ninfer_openai_decisions_product_test \
		ninfer_systemone_product_test \
		ninfer_request_log_test \
		ninfer_serve_options_test
	$(CTEST) --test-dir "$(BUILD_DIR)" --output-on-failure -R \
		'ninfer_(response_store|responses_state|responses_transport|responses_schema|openai_schema|openai_decisions_product|systemone_product|request_log|serve_options)_test'

# Compile the token-specialized decode objects under their four-slot Ninja pool.
decode-verbose: configure
	$(CMAKE) --build "$(BUILD_DIR)" --parallel "$(CUDA_JOBS)" --verbose --target \
		ninfer_gqa_decode_t1 ninfer_gqa_decode_t2 ninfer_gqa_decode_t3 \
		ninfer_gqa_decode_t4 ninfer_gqa_decode_t5 ninfer_gqa_decode_t6

# Compile the W8 small-T geometry objects under the same bounded pool.
w8-verbose: configure
	$(CMAKE) --build "$(BUILD_DIR)" --parallel "$(CUDA_JOBS)" --verbose --target \
		ninfer_w8_small_t_p1 ninfer_w8_small_t_p2 ninfer_w8_small_t_p3 \
		ninfer_w8_small_t_p4 ninfer_w8_small_t_p5 ninfer_w8_small_t_p6 \
		ninfer_w8_small_t_p7

clean:
	@test -n "$(BUILD_DIR)" && test "$(BUILD_DIR)" != "/"
	$(CMAKE) -E remove_directory "$(BUILD_DIR)"

rebuild:
	$(MAKE) clean
	$(MAKE) build
