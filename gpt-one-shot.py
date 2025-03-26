import pickle
from openai import OpenAI
from utils import load_context_ikat, inference_chatgpt_all_data, parse_output_one_shot


prompt_with_query_rewrite = """You are a search quality rater evaluating the relevance of web pages. 
    Given the user query, and a web page, you must provide a score on an integer scale of 0 to 4 to indicate to what extent the given passage meets the information needs of the user. The scores have the following meanings:

    0: fails to meet 
    1: slightly meets
    2: moderately meets
    3: highly meets
    4: fully meets

    Query: {utterance}

    Passage 1: {response}
    Score: 4

    Passage 2: {passage}
    Score:
    Please only generate an int score between 0 to 4 to say to what extent the document 2 is relevant to the user question.
    """

prompt_with_context = """You are a search quality rater evaluating the relevance of passages to the user's question in the context of a conversation.
    Given the conversation context, user question, and a passage, you must provide a score on an integer scale of 0 to 4 with the following meanings:
    
    0: fails to meet 
    1: slightly meets
    2: moderately meets
    3: highly meets
    4: fully meets

    Conversation context: {context}
    Query: {utterance}

    Passage 1: {response}
    Score: 4

    Passage 2: {passage}
    Score:
    
    Please only generate an int score between 0 to 4 to say to what extent the document 2 is relevant to the user question.
    """

prompt_with_query_rewrite_with_ptkb = """You are a search quality rater evaluating the relevance of web pages. 
    Given the persona of the user, user query, persona of the user, and a web page, you must provide a score on an integer scale of 0 to 4 to indicate to what extent the given passage meets the information needs of the user. The scores have the following meanings:

    0: fails to meet 
    1: slightly meets
    2: moderately meets
    3: highly meets
    4: fully meets

    User persona: {ptkb}
    Query: {utterance}

    Passage 1: {response}
    Score: 4

    Passage 2: {passage}
    Score:
    Please only generate an int score between 0 to 4 to say to what extent the document 2 is relevant to the user question.
    """

prompt_with_context_with_ptkb = """You are a search quality rater evaluating the relevance of passages to the user's question in the context of a conversation.
    Given the conversation context, user question, persona of the user, and a passage, you must provide a score on an integer scale of 0 to 4 with the following meanings:
    
    0: fails to meet 
    1: slightly meets
    2: moderately meets
    3: highly meets
    4: fully meets

    User persona: {ptkb}
    Conversation context: {context}
    Query: {utterance}

    Passage 1: {response}
    Score: 4

    Passage 2: {passage}
    Score:
    Please only generate an int score between 0 to 4 to say to what extent the document 2 is relevant to the user question.
    """

def create_data_one_shot(data_path, prompt, dataset_name):
    flattened_inputs = []

    with open(data_path, 'r') as f:
        lines = f.readlines()

    if dataset_name == 'ikat23':
        for line in lines:
            turn_id, rewritten_utterance, response_txt, passage_id, passage_text, score, ptkb, _ = line.split('\t')
            x = {'turn_id': turn_id,
                'doc_id': passage_id,
                'passage': passage_text,
                'score': score}
            x['prompt'] = prompt.format(ptkb=ptkb, utterance=rewritten_utterance, passage=passage_text, response=response_txt)
            flattened_inputs.append(x)
    
    elif dataset_name == 'cast22':
        for line in lines:
            turn_id, rewritten_utterance, response_txt, passage_id, passage_text, score, _, _ , _ = line.split('\t')
            x = {'turn_id': turn_id,
                'doc_id': passage_id,
                'passage': passage_text,
                'score': score}
            x['prompt'] = prompt.format(utterance=rewritten_utterance, passage=passage_text, response=response_txt)
            flattened_inputs.append(x)

    return flattened_inputs

def create_data_one_shot_with_context(data_path, prompt, dataset_name, context_turn):
    flattened_inputs = []

    with open(data_path, 'r') as f:
        lines = f.readlines()

    if dataset_name == 'ikat23':
        for line in lines:
            turn_id, rewritten_utterance, response_txt, passage_id, passage_text, score, ptkb, _ = line.split('\t')
            x = {'turn_id': turn_id,
            'doc_id': passage_id,
            'passage': passage_text,
            'score': score}
            x['prompt'] = prompt.format(ptkb=ptkb, utterance=rewritten_utterance, passage=passage_text, response=response_txt, context=context_turn[turn_id])
            flattened_inputs.append(x)
    
    elif dataset_name == 'cast22':
        for line in lines:
            turn_id, rewritten_utterance, response_txt, passage_id, passage_text, score, _, context, _ = line.split('\t')
            x = {'turn_id': turn_id,
            'doc_id': passage_id,
            'passage': passage_text,
            'score': score}
            x['prompt'] = prompt.format(utterance=rewritten_utterance, passage=passage_text, response=response_txt, context=context)
            flattened_inputs.append(x)

    return flattened_inputs


DATASET_NAME = 'cast22'
FROM_CHECKPOINT= False
model_id = "gpt-3.5-turbo-0125"
API_key = "YOUR_API_KEY"
use_context = False

model_name = f'{model_id}-one-shot-{DATASET_NAME}-context-{use_context}'
client = OpenAI(api_key=API_key)
path_output_pkl = '/outputs/'+model_name+'.pkl'
output_text_path = '/outputs/'+model_name+'.txt'


flattened_inputs = []

print('================ Loading the data. ================')
if DATASET_NAME == 'cast22':
    data_path = '/inputs/cast22_splitted_data.txt'
    if use_context:

        flattened_inputs =  create_data_one_shot_with_context(data_path, prompt_with_context, DATASET_NAME, None)
    else:
        flattened_inputs = create_data_one_shot(data_path, prompt_with_query_rewrite, DATASET_NAME)


elif DATASET_NAME == 'ikat23':
    data_path = '/inputs/ikat23_splitted_data.txt'
    if use_context:
        context_turn = load_context_ikat('/inputs/ikat23_conversation_context.txt')
        flattened_inputs =  create_data_one_shot_with_context(data_path, prompt_with_context_with_ptkb, DATASET_NAME, context_turn)
    else:
        flattened_inputs = create_data_one_shot(data_path, prompt_with_query_rewrite_with_ptkb, DATASET_NAME)

if FROM_CHECKPOINT:
    print('================ Loading from checkpoint ================')
    with open(path_output_pkl, 'rb') as f:
        flattened_inputs = pickle.load(f)

print('================ Started inference. ================')
flattened_inputs = inference_chatgpt_all_data(flattened_inputs, path_output_pkl, client, model_id, 20)

print('================ Inference finished, writing the output to text file. ================')
parse_output_one_shot(flattened_inputs, output_text_path)
