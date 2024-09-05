from openai import OpenAI

ORGANIZATION = ""
PROJ_ID = ""
API_KEY = ""

client = OpenAI(
    organization=ORGANIZATION,
    project=PROJ_ID,
    api_key=API_KEY
)

def category(dao, content):
    q1 = f'''
    I have a new proposal from "{dao}" as follows: "{content}", among the category list: [Game Development and NFTs, Community and Social Initiatives, Technology and Development, Financial and Funding Initiatives, Governance and Structure, Election and Committee Formation, Incentives and Rewards, Security Proposals, Event Sponsorship], select the one that describe the most of the proposal and give me the choice only.
    '''
    chat_completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": q1}],
    )
    return chat_completion.choices[0].message.content

def summary(dao):
    q2 = f'''
    It is linked to the "{dao}" proposal I just asked. Give a summary of the proposal as a blockchain expert in 3 to 4 sentences. 
    '''
    chat_completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": q2}],
    )
    return chat_completion.choices[0].message.content

def attitude(dao):
    q3 = f'''
    It is linked to the "{dao}" proposal I just asked. As a community member, do you think the community leans towards accepting or rejecting the proposal? Answer “Yes” or “No”.
    '''
    chat_completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": q3}],
    )
    return chat_completion.choices[0].message.content

# def keywords(dao):
#     q4 = f'''
#     What are the five keywords in blockchain synonym that covered the most of the "{dao}" proposal I just asked?
#     '''
#     chat_completion = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "user", "content": q4}],
#     )
#     return chat_completion.choices[0].message.content


# total_output_tokens = 0

# start = time.time()

# for q in [q1, q2, q3, q4]:

#     chat_completion = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "user", "content": q}],
#     )
#     total_output_tokens += len(chat_completion.choices[0].message.content)
#     print(chat_completion.choices[0].message.content)

# end = time.time()
# elapsed = end - start
# print("Content Length: ", len(content))
# print("Total Input Tokens: ", len(q1) + len(q2) + len(q3) + len(q4))
# print("Total Output Tokens: ", total_output_tokens)
# print("Total Time: ", elapsed, "s")


