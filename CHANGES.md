# Changelog

## [0.1.0] - 2026-05-29

### Summary

Section headers (e.g., `=== STATIC PART (CACHE-FRIENDLY) ===`) are now **optional** and controlled via the `debug_mode` flag in `OrchestratorSettings`.

### Changes

1. **OrchestratorSettings** (`prompt_orchestrator/config/settings.py`)
- Added `debug_mode: bool = False` flag
- When `False` (default): headers are excluded from prompts to save tokens
- When `True`: headers are included for debugging and visualization

2. **PromptConfig** (`prompt_orchestrator/config/prompt_config.py`)
- Modified `render_static_header(include_header: bool = False)` to accept an optional parameter
- Header is only prepended when `include_header=True`

3. **PromptBuilder** (`prompt_orchestrator/builder/builder.py`)
- Updated `build_sections()` to accept `include_headers: bool = False` parameter
- Updated `build_prompt()` to accept and forward the parameter
- Headers conditionally included in all sections (static, semi-stable, dynamic, RAG)

4. **PromptOrchestrator** (`prompt_orchestrator/orchestrator/orchestrator.py`)
- Passes `self.settings.debug_mode` to `build_sections()` call
- Respects user's debug preference automatically

5. **Simulations**
- **console_pipeline_test.py**: `build_orchestrator(debug_mode: bool)` accepts debug mode; prompts user at startup
- **conversation_simulation_test.py**: Added `--debug` CLI flag; `build_orchestrator(debug_mode: bool)` parameter

### Usage

#### Via ConfigStore/Settings
```python
settings = OrchestratorSettings(
    max_prompt_tokens=3000,
    debug_mode=True,  # Include headers
)
```

#### Via Console Pipeline
```bash
python simulations/console_pipeline_test.py
# Prompts: "Enable debug output? [y/N]: "
```

#### Via Conversation Simulation
```bash
# Without debug (default, fewer tokens)
python simulations/conversation_simulation_test.py

# With debug headers
python simulations/conversation_simulation_test.py --debug
```

### Impact

- **Production**: Smaller prompts (fewer tokens) by default
- **Development**: Full structure visibility when needed
- **Tests**: All 6 tests passing; debug_mode=True used in end-to-end test
- **Backward Compatibility**: Default is False, aligning with token efficiency

### Example Output

#### Without Debug (default)
```
System Prompt:
You are a helpful assistant.

Role:
Engineer
...
```

#### With Debug (debug_mode=True)
```
=== STATIC PART (CACHE-FRIENDLY) ===
System Prompt:
You are a helpful assistant.

Role:
Engineer
...
```
