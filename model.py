from vllm import LLM, SamplingParams
from vector_db import retrieve_relevant_docs
from config import MODEL_PATH
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationSummaryMemory, ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain.memory import CombinedMemory
from process_books import clean_text

llm = LLM(model=MODEL_PATH, tensor_parallel_size=2,dtype="float16")

summary_memory = ConversationSummaryMemory(llm=llm, memory_key="summary", return_messages=True)
buffer_memory = ConversationBufferWindowMemory(memory_key="recent_history", return_messages=True, k=5)  # Keeps last 5 interactions
memory = CombinedMemory(memories=[summary_memory, buffer_memory])

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """You are a helpful, knowledgeable literature expert skilled in analyzing, summarizing and understanding novels. 
        Use the provided context to answer the user's query accurately. Use the chat history with the user to further personalize your response.
        Give more priority to ShortTerm Memory. Be as accurate as possible but don't make up information
        that's not from the context. Never mention that your souce is the 'context'. If need arises, use 'the book' or something similar.
        If the context lacks sufficient information, say you don't know instead of making up an answer.
        If the user's query is not at all related to the context, say so explicitly.
        If a user query contains spelling mistakes or typos, do not get confused.
        Instead, infer the most likely intended meaning and respond accordingly.
        If you are confused while interpreting the query, ask the user "Did you mean?" (add your present interpretation)

        Also, provide a confidence score (between 0-100%) indicating how sure you are about the answer.
        Format your response as:
        Answer: <Your Answer>
        Confidence: <Confidence Score>
        """
    ),
    HumanMessagePromptTemplate.from_template(
        "### LongTerm History:\n{summary}, ShortTerm History:\n{recent_history}, Context:\n{context}\n\n### Question:\n{query}\n\n### Answer:"
    ),
])


def generate_response(query,max_tokens):
    query = clean_text(query)
    relevant_docs = retrieve_relevant_docs(query)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    memory_variables = memory.load_memory_variables({})
    summary = clean_text(memory_variables.get("summary", ""))
    recent_history = clean_text(memory_variables.get("recent_history", ""))

    formatted_prompt = prompt.format(summary=summary, recent_history=recent_history, context=context, query=query)

    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=max_tokens)

    try:
        response = llm.generate(formatted_prompt, sampling_params)
    except:
        return ("I'm currently unable to retrieve an answer. Please try again later or refine your question.")
    
    response_text = response[0].response[0].text

    answer, confidence = response_text.split("Confidence: ")
    confidence= int(confidence.stripe()[:-1])

    if confidence < 30:
        return ("Im sorry, I don't know about this yet.")
    elif confidence < 70:
        return (f"I'm not really sure about this but here's what I think with confidence {confidence}%:\n {answer}")

    memory.save_context({"query": query}, {"response": response_text})

    return answer
