from openai import OpenAI
import json

client = OpenAI(
    api_key='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',
)

# chat_completion = client.chat.completions.create(
#     messages=[
#         {
#             "role": "user",
#             "content": "Say this is a test",
#         }
#     ],
#     model="gpt-3.5-turbo",
# )

relevance_schema = {
    "type": "function",
    "function": {
        "name": "assess_relevance",
        "description": "Assess the relevance of a document to a given query",
        "parameters": {
            "type": "object",
            "properties": {
                "grade": {
                    "type": "integer",
                    "description": "The relevance grade of the document to the query, 3-scale grade 0 = non-relevant, 1 = relevant, 2 = very relevant",
                    "enum": [0, 1, 2],
                },
            },
            "required": ["grade"],
        },
    },
}

def get_relevance_grade(query, url, title, content):
    content = content[:1000]
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'user', 'content': f"Assess the relevance of the following document to this query: {query}\n Document: url: {url}, title: {title}, content: {content}"}],
        tools=[relevance_schema],
        tool_choice={"type": "function", "function": {"name": "assess_relevance"}},
    )
    arguments_str = response.choices[0].message.tool_calls[0].function.arguments
    arguments_dict = json.loads(arguments_str)
    return arguments_dict['grade']

# print(get_relevance_grade("coronavirus", "Coronavirus is a virus that causes COVID-19", "Coronavirus is a virus that causes COVID-19"))