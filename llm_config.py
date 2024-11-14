from langchain.llms.llamacpp import LlamaCpp
from langchain_core.language_models.llms import LLM


class myLLM:
    def __init__(self):
        self.llm = LlamaCpp(
                    model_path="C:\\Users\\myrea\\Desktop\\Vectors\\MP-07\\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                    max_tokens=1024,
                    n_ctx=4096,
                    n_gpu_layers=-1,
                    use_mlock=True
        )

    def get_configured_llm(self) -> LLM:
        return self.llm


# Debugging
if __name__ == "__main__":
    debug = myLLM()
    llm = debug.get_configured_llm()
    print(llm.invoke('Hi'))
