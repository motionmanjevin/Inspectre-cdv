from llm.prompts import QA_PROMPT
from transformers import pipeline

class QuestionAnswerAgent:
    def __init__(self, memory):
        self.memory = memory
        self.model = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", device=0)

    def answer(self, question):
        relevant = self.memory.search(question)
        context = "\n".join(relevant)
        prompt = QA_PROMPT.format(context=context, question=question)
        result = self.model(prompt, max_new_tokens=100, do_sample=True)
        return result[0]['generated_text']