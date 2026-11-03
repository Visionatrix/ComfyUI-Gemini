# TEST ONLY: records a STRING into the prompt history so tests can read node output.
class TestTextSink:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "test"

    def run(self, text):
        return {"ui": {"text": [text]}}


NODE_CLASS_MAPPINGS = {"TestTextSink": TestTextSink}
