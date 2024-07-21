from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from tqdm import tqdm


def chunk_text(text, chunk_size=128000, overlap=5000):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if (len(text[start:end]) > 2):
            chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def summarize_chunk(chunk, llm):
    prompt_template = '''
    I am giving you a text delimited by triple backticks, which consists of various different summarries.
    Summarize all the summaries delimited by triple backticks in 1000 words.
    ```
    {text}
    ```
    Summary:'''
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = prompt | llm

    response = chain.invoke({"text": chunk})
    return response.content


def create_rolling_summary(page_summaries):

    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-1.5-flash",
    #     temperature=0,
    #     safety_settings={
    #         HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    #     },
    #     google_api_key=userdata.get('GOOGLE_API_KEY'),
    # )

    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

    combined_summary = "\n\n\n***NEXT SUMMARY***\n\n\n".join(page_summaries)

    while len(combined_summary) > 128000:
        chunks = chunk_text(combined_summary)
        chunk_summaries = [summarize_chunk(chunk, llm) for chunk in tqdm(
            chunks, desc="Re Summarizing chunks")]
        combined_summary = " ".join(chunk_summaries)

    final_summary = summarize_chunk(combined_summary, llm)
    return final_summary


# final_summary = create_rolling_summary(page_summaries, llm)
# images_and_summarize(file_path, client)
