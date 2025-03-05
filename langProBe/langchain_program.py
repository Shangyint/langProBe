from langchain_community.chat_models import ChatLiteLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langProBe.program_utils import DotDict


class LangProBeLangChainMetaProgram:
    def __init__(self, input_kwargs, output_kwargs):
        self.lm = None
        self.input_kwargs = input_kwargs
        self.out_kwargs = output_kwargs

    def setup_lm(self, lm: str, api_key: str = None, api_base: str = None):
        self.lm = ChatLiteLLM(model=lm, api_key=api_key, api_base=api_base)


class NaiveLangChainProgram(LangProBeLangChainMetaProgram):
    def __call__(self, **kwargs):
        if not self.lm:
            raise ValueError("Language model not initialized. Call setup_lm() first.")

        # Validate input keys
        missing_keys = [key for key in self.input_kwargs if key not in kwargs]
        if missing_keys:
            raise ValueError(f"Missing required inputs: {missing_keys}")

        # Dynamically generate prompt template
        prompt_text = "Given the following inputs:\n"
        for key in self.input_kwargs:
            prompt_text += f"- {key}: {{{key}}}\n"
        prompt_text += f"Output the following field: {self.out_kwargs[0]}. Your response should be this output field only, with no explanation and formatting.\n Your response:"

        prompt_template = PromptTemplate(
            input_variables=self.input_kwargs, template=prompt_text
        )

        # Create LLM chain
        chain = LLMChain(llm=self.lm, prompt=prompt_template)

        # Run the chain
        response = chain.run(kwargs)

        # Format output
        return DotDict({self.out_kwargs[0]: response})
