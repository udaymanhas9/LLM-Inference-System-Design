.PHONY: run build-cpp submodule install demo sim ollama-gemma ollama-qwen test

# ── Primary entry point ───────────────────────────────────────────────────────
run:
	-lsof -ti :8000 | xargs kill -9 2>/dev/null; true
	bash run.sh

# ── C++ inference library (requires llama.cpp submodule) ─────────────────────
submodule:
	git submodule add https://github.com/ggml-org/llama.cpp third_party/llama.cpp
	git submodule update --init --recursive

build-cpp:
	cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
	cmake --build build --target inference_lib -j$$(nproc 2>/dev/null || sysctl -n hw.logicalcpu)
	@echo ""
	@echo "✓ Built libinference.dylib (or .so)"
	@echo "  Use engine: llama.cpp:/path/to/model.gguf in the UI"

# ── Python-only helpers ───────────────────────────────────────────────────────
install:
	pip install fastapi "uvicorn[standard]" httpx pydantic

demo:
	python sim/load_sim.py 5

sim:
	python sim/load_sim.py 20

ollama-gemma:
	python sim/load_sim.py 3 --model gemma4:26b

ollama-qwen:
	python sim/load_sim.py 3 --model qwen2.5-coder:14b

test:
	python -m pytest tests/ -v
