import pickle
from openai import OpenAI
from utils import  inference_chatgpt_all_data, parse_output_two_shot, load_two_shot_examples


prompt_with_query_rewrite_two_shot_with_ptkb = """You are a search quality rater evaluating the relevance of web pages. 
    Given the persona of the user, user query, and a web page, you must provide a score on an integer scale of 0 to 4 to indicate to what extent the given passage meets the information needs of the user. The scores have the following meanings:

    0: fails to meet 
    1: slightly meets
    2: moderately meets
    3: highly meets
    4: fully meets

    User persona: {ptkb}
    Query: {utterance}

    Passage 1: {pass_1}
    Score: {score_1}

    Passage 2: {pass_2}
    Score: {score_2}

    Passage 3: {passage}
    Score:
    Please only generate an int score between 0 to 4 to say to what extent the Passage 3 is relevant to the user question. Score lower than 2 means the document is not relevant.
    """

prompt_with_query_rewrite_two_shot = """You are a search quality rater evaluating the relevance of web pages. 
    Given the user query and a web page, you must provide a score on an integer scale of 0 to 4 to indicate to what extent the given passage meets the information needs of the user. The scores have the following meanings:

    0: fails to meet 
    1: slightly meets
    2: moderately meets
    3: highly meets
    4: fully meets

    Query: {utterance}

    Passage 1: {pass_1}
    Score: {score_1}

    Passage 2: {pass_2}
    Score: {score_2}

    Passage 3: {passage}
    Score:
    Please only generate an int score between 0 to 4 to say to what extent the Passage 3 is relevant to the user question. Score lower than 2 means the document is not relevant.
    """


def create_data_two_shot(data_path, prompt, dataset_name, two_shot_examples):
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
            x['prompt'] = prompt.format(ptkb=ptkb, utterance=rewritten_utterance, passage=passage_text, response=response_txt,
              pass_1 = two_shot_examples['pass_1'], score_1 = two_shot_examples['score_1'], 
              pass_2 = two_shot_examples['pass_2'], score_2 = two_shot_examples['score_2'] )
            flattened_inputs.append(x)
    
    elif dataset_name == 'cast22':
        for line in lines:
            turn_id, rewritten_utterance, response_txt, passage_id, passage_text, score, _, _ , _ = line.split('\t')
            x = {'turn_id': turn_id,
                'doc_id': passage_id,
                'passage': passage_text,
                'score': score}
            x['prompt'] = prompt.format(utterance=rewritten_utterance, passage=passage_text, response=response_txt,
              pass_1 = two_shot_examples['pass_1'], score_1 = two_shot_examples['score_1'], 
              pass_2 = two_shot_examples['pass_2'], score_2 = two_shot_examples['score_2'])
                                                 
            flattened_inputs.append(x)

    return flattened_inputs


DATASET_NAME = 'cast22'
FROM_CHECKPOINT= False
model_id = "gpt-3.5-turbo-0125"
API_key = "YOUR_API_KEY"


model_name = f'{model_id}-two-shot-{DATASET_NAME}'
client = OpenAI(api_key=API_key)
path_output_pkl = '/outputs/'+model_name+'.pkl'
output_text_path = '/outputs/'+model_name+'.txt'

print('================ Loading the data. ================')
flattened_inputs = []

if DATASET_NAME == 'cast22':
    data_path = '/inputs/cast22_splitted_data.txt'
    two_shot_examples = load_two_shot_examples("/inputs/cast22_two_shot_examples.txt")
    flattened_inputs = create_data_two_shot(data_path, prompt_with_query_rewrite_two_shot, DATASET_NAME, two_shot_examples)


elif DATASET_NAME == 'ikat23':
    data_path = '/inputs/ikat23_splitted_data.txt'
    two_shot_examples = load_two_shot_examples("/inputs/ikat23_two_shot_examples.txt")
    flattened_inputs = create_data_two_shot(data_path, prompt_with_query_rewrite_two_shot_with_ptkb, DATASET_NAME, two_shot_examples)

if FROM_CHECKPOINT:
    print('================ Loading from checkpoint ================')
    with open(path_output_pkl, 'rb') as f:
        flattened_inputs = pickle.load(f)

print('================ Started inference. ================')
flattened_inputs = inference_chatgpt_all_data(flattened_inputs, path_output_pkl, client, model_id, 20)

print('================ Inference finished, writing the output to text file. ================')
parse_output_two_shot(flattened_inputs, output_text_path)
