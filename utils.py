from collections import defaultdict
import re
from openai import OpenAI
import pickle

def chatgpt_conversation(conversation_Log, client, model_id, max_token_length):
  response = client.chat.completions.create(
      model = model_id,
      messages= conversation_Log,
      temperature= 0,
      top_p= 1,
      n=1, 
    max_tokens=max_token_length,
  )
  
  response= response.choices[0].message.content

  return response

def run_one_sample(prompt, client, model_id, max_token_length):

    conversations = []
    conversations.append({'role': 'user', 'content': prompt})
    response = chatgpt_conversation(conversations, client, model_id, max_token_length)
    score = response.strip()

    return score

def load_context_ikat(path_to_context_ikat):
    context_turn = defaultdict(str)

    with open(path_to_context_ikat, 'r') as f:
        context_ikat = f.readlines()

    for line in context_ikat:
        turn_id, context = line.strip().split('\t')
        context_turn[turn_id] = context
    
    return context_turn

def load_two_shot_examples(sample_passage_path):

    with open(sample_passage_path, 'r') as f:
        sample_passages_data = f.readlines()
    
    sample_passage_turn = {}
    
    for line_data in sample_passages_data:
        turn_id, pass_1, score_1, pass_2, score_2 = line_data.split('\t')
        y = {'pass_1': pass_1,
             'pass_2': pass_2,
             'score_1': score_1,
             'score_2': score_2}
        sample_passage_turn[turn_id] = y
    
    return sample_passage_turn

def inference_chatgpt_all_data(input_data_list, path_output_pkl, client, model_id, max_token_length):

    index = 0
    for entry in input_data_list:
        index += 1
        print(index)
        if 'gpt3.5-score' in entry:
            print(entry['gpt3.5-score'])
            print('&&&&&&&&&&&&&&')
            continue
        
        else:
            try:
                pred_score = run_one_sample(entry['prompt'], client, model_id, max_token_length)
                pred_score = pred_score.lower().strip()
                entry['gpt3.5-score'] = pred_score
                print(entry['gpt3.5-score'])
                print('****************')

            except:
                print('unsuccessful')
                entry['gpt3.5-score'] = 'NONE'

        if index%100==99:
            print('one batch with size of 100 done.')
            with open(path_output_pkl, 'wb') as f:
                pickle.dump(input_data_list, f)


    with open(path_output_pkl, 'wb') as f:
        pickle.dump(input_data_list, f) 

    return   input_data_list

def parse_output_one_shot(flattened_inputs, output_text_path):
    
    lines = []
    for entry in flattened_inputs:
        score = 'NOT_PREDICTED_YET'

        if ('gpt3.5-score' in entry):
            score = 'NONE'
            if entry['gpt3.5-score'].strip() == '0':
                score='0'
            elif entry['gpt3.5-score'].strip() == '1':
                score='1'
            elif entry['gpt3.5-score'].strip() == '2':
                score='2'
            elif entry['gpt3.5-score'].strip() == '3':
                score='3'    
            elif entry['gpt3.5-score'].strip() == '4':  
                score='4'

            else:
                numbers = re.findall(r'\d+', entry['gpt3.5-score'].strip())
                if len(numbers)==1:
                    score = numbers[0]
                else:
                    if (('fails to meet' in entry['gpt3.5-score']) or ('does not meet' in entry['gpt3.5-score'])) :
                        score = '0'
                    elif ('slightly meets' in entry['gpt3.5-score']):
                        score = '1'
                    elif ('moderately meets' in entry['gpt3.5-score']):
                        score = '2'
                    elif ('highly meets' in entry['gpt3.5-score']):
                        score = '3'
                    elif ('fully meets' in entry['gpt3.5-score']):
                        score = '4'
                    elif (('passage 2: score 0' in entry['gpt3.5-score']) or ('passage 2: 0' in entry['gpt3.5-score']) or ('passage 2: score: 0' in entry['gpt3.5-score'])  or ('passage 2 score: 0' in entry['gpt3.5-score']) ):
                        score = '0'
                    elif (('passage 2: score 1' in entry['gpt3.5-score'] ) or ('passage 2: 1' in entry['gpt3.5-score']) or ('passage 2: score: 1' in entry['gpt3.5-score']) or ('passage 2 score: 1' in entry['gpt3.5-score'])):
                        score = '1'
                    elif (('passage 2: score 2' in entry['gpt3.5-score'] ) or ('passage 2: 2' in entry['gpt3.5-score']) or ('passage 2: score: 2' in entry['gpt3.5-score']) or ('passage 2 score: 2' in entry['gpt3.5-score'])):
                        score = '2'
                    elif (('passage 2: score 3' in entry['gpt3.5-score'] ) or ('passage 2: 3' in entry['gpt3.5-score']) or ('passage 2: score: 3' in entry['gpt3.5-score']) or ('passage 2 score: 3' in entry['gpt3.5-score']) ):
                        score = '3'
                    elif (('passage 2: score 4' in entry['gpt3.5-score'] ) or ('passage 2: 4' in entry['gpt3.5-score']) or ('passage 2: score: 4' in entry['gpt3.5-score']) or ('passage 2 score: 4' in entry['gpt3.5-score'])):
                        score = '4' 

                    else:
                        tmp = entry['gpt3.5-score'].strip().split('\n')[0]
                        numbers = re.findall(r'\d+',  tmp.strip()) 
                        if len(numbers)==1:
                            score = numbers[0]

        line = entry['turn_id'] + '\t0\t'+ entry['doc_id']+ '\t'+ str(score) + '\n'
        lines.append(line)

    with open(output_text_path, 'w') as f:
        f.writelines(lines)   

    return

def parse_output_two_shot(flattened_inputs, output_text_path):
    
    lines = []
    for entry in flattened_inputs:
        score = 'NOT_PREDICTED_YET'

        if ('gpt3.5-score' in entry):
            score = 'NONE'
            if entry['gpt3.5-score'].strip() == '0':
                score='0'
            elif entry['gpt3.5-score'].strip() == '1':
                score='1'
            elif entry['gpt3.5-score'].strip() == '2':
                score='2'
            elif entry['gpt3.5-score'].strip() == '3':
                score='3'    
            elif entry['gpt3.5-score'].strip() == '4':  
                score='4'

            else:
                numbers = re.findall(r'\d+', entry['gpt3.5-score'].strip())
                if len(numbers)==1:
                    score = numbers[0]
                else:
                    if (('fails to meet' in entry['gpt3.5-score']) or ('does not meet' in entry['gpt3.5-score'])) :
                        score = '0'
                    elif ('slightly meets' in entry['gpt3.5-score']):
                        score = '1'
                    elif ('moderately meets' in entry['gpt3.5-score']):
                        score = '2'
                    elif ('highly meets' in entry['gpt3.5-score']):
                        score = '3'
                    elif ('fully meets' in entry['gpt3.5-score']):
                        score = '4'
                    elif (('passage 3: score 0' in entry['gpt3.5-score']) or ('passage 3: 0' in entry['gpt3.5-score']) or ('passage 3: score: 0' in entry['gpt3.5-score'])  or ('passage 3 score: 0' in entry['gpt3.5-score']) ):
                        score = '0'
                    elif (('passage 3: score 1' in entry['gpt3.5-score'] ) or ('passage 3: 1' in entry['gpt3.5-score']) or ('passage 3: score: 1' in entry['gpt3.5-score']) or ('passage 3 score: 1' in entry['gpt3.5-score'])):
                        score = '1'
                    elif (('passage 3: score 2' in entry['gpt3.5-score'] ) or ('passage 3: 2' in entry['gpt3.5-score']) or ('passage 3: score: 2' in entry['gpt3.5-score']) or ('passage 3 score: 2' in entry['gpt3.5-score'])):
                        score = '2'
                    elif (('passage 3: score 3' in entry['gpt3.5-score'] ) or ('passage 3: 3' in entry['gpt3.5-score']) or ('passage 3: score: 3' in entry['gpt3.5-score']) or ('passage 3 score: 3' in entry['gpt3.5-score']) ):
                        score = '3'
                    elif (('passage 3: score 4' in entry['gpt3.5-score'] ) or ('passage 3: 4' in entry['gpt3.5-score']) or ('passage 3: score: 4' in entry['gpt3.5-score']) or ('passage 3 score: 4' in entry['gpt3.5-score'])):
                        score = '4' 

                    else:
                        tmp = entry['gpt3.5-score'].strip().split('\n')[0]
                        numbers = re.findall(r'\d+',  tmp.strip()) 
                        if len(numbers)==1:
                            score = numbers[0]

        line = entry['turn_id'] + '\t0\t'+ entry['doc_id']+ '\t'+ str(score) + '\n'
        lines.append(line)

    with open(output_text_path, 'w') as f:
        f.writelines(lines)   

    return

def parse_output_zero_shot(flattened_inputs, output_text_path):
    
    lines = []
    for entry in flattened_inputs:
        score = 'NOT_PREDICTED_YET'

        if ('gpt3.5-score' in entry):
            score = 'NONE'
            if entry['gpt3.5-score'].strip() == '0':
                score='0'
            elif entry['gpt3.5-score'].strip() == '1':
                score='1'
            elif entry['gpt3.5-score'].strip() == '2':
                score='2'
            elif entry['gpt3.5-score'].strip() == '3':
                score='3'    
            elif entry['gpt3.5-score'].strip() == '4':  
                score='4'

            else:
                numbers = re.findall(r'\d+', entry['gpt3.5-score'].strip())
                if len(numbers)==1:
                    score = numbers[0]
                else:
                    if (('fails to meet' in entry['gpt3.5-score']) or ('does not meet' in entry['gpt3.5-score'])) :
                        score = '0'
                    elif ('slightly meets' in entry['gpt3.5-score']):
                        score = '1'
                    elif ('moderately meets' in entry['gpt3.5-score']):
                        score = '2'
                    elif ('highly meets' in entry['gpt3.5-score']):
                        score = '3'
                    elif ('fully meets' in entry['gpt3.5-score']):
                        score = '4'
                    else:
                        tmp = entry['gpt3.5-score'].strip().split('\n')[0]
                        numbers = re.findall(r'\d+',  tmp.strip()) 
                        if len(numbers)==1:
                            score = numbers[0]

        line = entry['turn_id'] + '\t0\t'+ entry['doc_id']+ '\t'+ str(score) + '\n'
        lines.append(line)

    with open(output_text_path, 'w') as f:
        f.writelines(lines)   

    return


