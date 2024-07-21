from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from tqdm import tqdm


def answerQuery(query, chroma_client):

    client = chroma_client
    collection = client.get_or_create_collection(
        name='godrej_summaries',
        metadata={'hnsw:space': 'cosine', "userId": 1}
    )

    results = collection.query(
        query_texts=[query],
        n_results=5,
    )['documents'][0]

    context = '\n\n***NEXT DOCUMENT***\n\n'.join(results)

    # print("context: \n", context)

    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-1.5-flash",
    #     temperature=0,
    #     safety_settings={
    #         HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    #     },
    #     google_api_key=userdata.get('GOOGLE_API_KEY'),
    # )

    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

    prompt_template = """
  Answer the question based only on the following context. Context is delimited by triple backticks:
  ```{context}```

  - -

  Answer the question based on the above context: {question}
  """

    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = prompt | llm

    response = chain.invoke({"context": context, "question": query})
    return response.content
