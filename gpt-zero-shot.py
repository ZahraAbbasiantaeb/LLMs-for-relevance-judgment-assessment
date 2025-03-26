import pickle
from openai import OpenAI
from utils import inference_chatgpt_all_data, parse_output_zero_shot

prompt_with_query_rewrite_zero_shot_with_persona = """You are a search quality rater evaluating the relevance of web pages. 
    Given the persona of the user, user query, and a web page, you must provide a score on an integer scale of 0 to 4 to indicate to what extent the given passage meets the information needs of the user. The scores have the following meanings:

    0: fails to meet 
    1: slightly meets
    2: moderately meets
    3: highly meets
    4: fully meets

    User persona: {ptkb}
    Query: {utterance}
    Passage : {passage}
    Score:
    """

prompt_with_query_rewrite_zero_shot = """You are a search quality rater evaluating the relevance of web pages. 
    Given the user query and a web page, you must provide a score on an integer scale of 0 to 4 to indicate to what extent the given passage meets the information needs of the user. The scores have the following meanings:

    0: fails to meet 
    1: slightly meets
    2: moderately meets
    3: highly meets
    4: fully meets

    Query: {utterance}
    Passage : {passage}
    Score:
    """

def create_data_zero_shot(data_path, prompt, dataset_name):
    flattened_turn_pool_score = []

    with open(data_path, 'r') as f:
        lines = f.readlines()

    if dataset_name == 'ikat23':
        for line in lines:
            turn_id, rewritten_utterance, _, passage_id, passage_text, score, ptkb, _ = line.split('\t')
            x = {'turn_id': turn_id,
            'doc_id': passage_id,
            'passage': passage_text,
            'score': score}
            x['prompt'] = prompt.format(ptkb=ptkb, utterance=rewritten_utterance, passage=passage_text)
            flattened_turn_pool_score.append(x)
    
    elif dataset_name == 'cast22':
        for line in lines:
            turn_id, rewritten_utterance, _, passage_id, passage_text, score, _, _, _ = line.split('\t')
            x = {'turn_id': turn_id,
            'doc_id': passage_id,
            'passage': passage_text,
            'score': score}
            x['prompt'] = prompt.format(utterance=rewritten_utterance, passage=passage_text)
            flattened_turn_pool_score.append(x)

    return flattened_turn_pool_score


DATASET_NAME = 'cast22'
FROM_CHECKPOINT= False
model_id = "gpt-3.5-turbo-0125"
API_key = "YOUR_API_KEY"

model_name = f'{model_id}-zero-shot-{DATASET_NAME}'
client = OpenAI(api_key=API_key)
path_output_pkl = '/outputs/'+model_name+'.pkl'
output_text_path = '/outputs/'+model_name+'.txt'

flattened_inputs = []
print('================ Loading the data. ================')
if DATASET_NAME == 'cast22':
    data_path = '/inputs/cast22_splitted_data.txt'
    flattened_inputs =  create_data_zero_shot(data_path, prompt_with_query_rewrite_zero_shot, DATASET_NAME)


elif DATASET_NAME == 'ikat23':
    data_path = '/inputs/ikat23_splitted_data.txt'
    flattened_inputs =  create_data_zero_shot(data_path, prompt_with_query_rewrite_zero_shot_with_persona, DATASET_NAME)


if FROM_CHECKPOINT:
    print('================ Loading from checkpoint. ================')
    with open(path_output_pkl, 'rb') as f:
        flattened_inputs = pickle.load(f)

print('================ Started inference. ================')
flattened_inputs = inference_chatgpt_all_data(flattened_inputs, path_output_pkl, client, model_id, 20)

print('================ Inference finished, writing the output to text file. ================')
parse_output_zero_shot(flattened_inputs, output_text_path)

