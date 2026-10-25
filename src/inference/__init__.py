"""SPCM Indian Stock Inference Package."""
__all__ = [
    "SPCMInferencePipeline",
    "SPCMInferenceResult",
    "StockPrediction",
    "DEFAULT_MODEL_PATH",
]


def __getattr__(name: str):
    if name in __all__:
        import importlib
        mod = importlib.import_module("src.inference.spcm_inference_pipeline")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
