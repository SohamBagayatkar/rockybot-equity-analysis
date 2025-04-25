import backoff
from langchain.prompts import PromptTemplate
from langchain_mistralai import ChatMistralAI
import os
from dotenv import load_dotenv

# Load API Key
load_dotenv()
mistral_api_key = os.getenv("MISTRAL_API_KEY")

if not mistral_api_key:
    raise ValueError("🚨 MISTRAL_API_KEY is missing! Check your .env file.")

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def extract_answer_from_docs(docs, question):
    """Processes the question and generates a response using Mistral Chat model."""
    question = question.strip() if question else "Summarize the article's key equity insights."
    context = "\n\n".join([f"Source: {doc.metadata['source']}\nContent: {doc.page_content}" for doc in docs])

    if not context:
        return "No relevant content found in the retrieved articles."

    chat_model = ChatMistralAI(model="mistral-small", api_key=mistral_api_key, temperature=0.05)
    prompt_template = PromptTemplate(
        input_variables=["question", "context"],
        template="""You are a financial analyst. Using the provided context, answer the question concisely, focusing on equity-related insights.

        Question: {question}
        Context: {context}

        Answer:
        """
    )
    prompt = prompt_template.format(question=question, context=context)
    response = chat_model.invoke(prompt)
    return response.content.strip()
