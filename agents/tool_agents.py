import re, string, os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "tools/planner")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../tools/planner")))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import importlib
from typing import List, Dict, Any
import tiktoken
from pandas import DataFrame
from langchain_community.chat_models import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
# from langchain.llms.base import BaseLLM
# from langchain.prompts import PromptTemplate
# from langchain.schema import (
#     AIMessage,
#     HumanMessage,
#     SystemMessage
# )
from prompts import zeroshot_react_agent_prompt
from utils.func import load_line_json_data, save_file
import sys
import json
import openai
import time
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from langchain_google_genai import ChatGoogleGenerativeAI
import argparse
from datasets import load_dataset
import os
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from langchain_community.llms import HuggingFacePipeline
    import torch
except ImportError:
    pass

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']


pd.options.display.max_info_columns = 200

os.environ['TIKTOKEN_CACHE_DIR'] = './tmp'

actionMapping = {"FlightSearch":"flights","AttractionSearch":"attractions","GoogleDistanceMatrix":"googleDistanceMatrix","AccommodationSearch":"accommodations","RestaurantSearch":"restaurants","Planner":"planner","NotebookWrite":"notebook","CitySearch":"cities"}

class CityError(Exception):
    pass

class DateError(Exception):
    pass

def catch_openai_api_error():
    error = sys.exc_info()[1]
    print(f"API error: {error}")
    if error == openai.APIConnectionError:
        print("APIConnectionError")
    elif error == openai.RateLimitError:
        print("RateLimitError")
        time.sleep(60)
    elif error == openai.APIError:
        print("APIError")
    elif error == openai.AuthenticationError:
        print("AuthenticationError")
    else:
        print("API error:", error)

class ReactAgent:
    def __init__(self,
                 args,
                 mode: str = 'zero_shot',
                 tools: List[str] = None,
                 max_steps: int = 30,
                 max_retries: int = 3,
                 illegal_early_stop_patience: int = 3,
                 react_llm_name = 'gpt-3.5-turbo-1106',
                 planner_llm_name = 'gpt-3.5-turbo-1106',
                #  logs_path = '../logs/',
                 city_file_path = '../database/background/citySet.txt'
                 ) -> None: 

        self.answer = ''
        self.max_steps = max_steps
        self.mode = mode

        self.react_name = react_llm_name
        self.planner_name = planner_llm_name

        if self.mode == 'zero_shot':
            self.agent_prompt = zeroshot_react_agent_prompt

        self.json_log = []

        self.current_observation = ''
        self.current_data = None
        self.step_metrics = []
        self.react_prompt_tokens = 0
        self.react_completion_tokens = 0
        self.repeated_action_count = 0
        self.info_gathered = {
            'flights': False,
            'accommodations': False,
            'restaurants': False,
            'attractions': False,
            'cities': False,
            'distances': False
        }

        self._last_prompt_tokens = 0
        self._last_completion_tokens = 0
        self._last_step_latency = 0.0

        self.tool_call_counts = {
            'flights': 0,
            'accommodations': 0,
            'restaurants': 0,
            'attractions': 0,
            'googleDistanceMatrix': 0,
            'cities': 0,
            'notebook': 0,
            'planner': 0,
            'no_data_calls': 0
        }

        self.total_tokens_used = 0
        self.halt_reason = 'unknown'

        if 'gpt-3.5' in react_llm_name:
            stop_list = ['\n']
            self.max_token_length = 15000
            self.llm = ChatOpenAI(temperature=1,
                     max_tokens=256,
                     model_name=react_llm_name,
                     openai_api_key=OPENAI_API_KEY,
                     model_kwargs={"stop": stop_list})
            
        elif 'gpt-4' in react_llm_name:
            stop_list = ['\n']
            self.max_token_length = 30000
            self.llm = ChatOpenAI(temperature=0,
                     max_tokens=256,
                     model_name=react_llm_name,
                     openai_api_key=OPENAI_API_KEY,
                     model_kwargs={"stop": stop_list})
            
        elif react_llm_name in ['mistral-7B-32K']:
            stop_list = ['\n']
            self.max_token_length = 30000
            self.llm = ChatOpenAI(temperature=0,
                     max_tokens=256,
                     openai_api_key="EMPTY", 
                     openai_api_base="http://localhost:8301/v1", 
                     model_name="gpt-3.5-turbo",
                     model_kwargs={"stop": stop_list})
            
        elif react_llm_name in ['mixtral']:
            stop_list = ['\n']
            self.max_token_length = 30000
            self.llm = ChatOpenAI(temperature=0,
                     max_tokens=256,
                     openai_api_key="EMPTY", 
                     openai_api_base="http://localhost:8501/v1", 
                     model_name="gpt-3.5-turbo",
                     model_kwargs={"stop": stop_list})


        elif react_llm_name in ['qwen']:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            from langchain_community.llms import HuggingFacePipeline
            import torch

            model_id = "Qwen/Qwen3-8B"
            hf_token = os.environ.get("HF_API_TOKEN")

            tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
            tokenizer.chat_template = tokenizer.chat_template.replace(
                "enable_thinking=True", "enable_thinking=False"
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                token=hf_token
            )

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=256,
                do_sample=False,
                return_full_text=False,
                temperature=None,       
                top_p=None,            
            )

            self.llm = HuggingFacePipeline(pipeline=pipe)
            self.shared_pipeline = self.llm
            self.max_token_length = 4096

        elif react_llm_name in ['qwen25-7B']:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            from langchain_community.llms import HuggingFacePipeline
            import torch

            model_id = "Qwen/Qwen2.5-7B-Instruct"
            hf_token = os.environ.get("HF_API_TOKEN")

            tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                token=hf_token
            )
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=256,
                do_sample=False,
                return_full_text=False,
                temperature=None,
                top_p=None,
                batch_size=1
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)
            self.shared_pipeline = self.llm
            self.max_token_length = 30000
        
        elif react_llm_name in ['phi3-mini']:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            from langchain_community.llms import HuggingFacePipeline
            import torch

            model_id = "microsoft/Phi-3-mini-128k-instruct"
            hf_token = os.environ.get("HF_API_TOKEN")

            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                token=hf_token,
                trust_remote_code=True
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                token=hf_token,
                trust_remote_code=True
            )

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=256,
                do_sample=False,
                return_full_text=False
            )

            self.llm = HuggingFacePipeline(pipeline=pipe)
            self.shared_pipeline = self.llm
            self.max_token_length = 128000
        
        elif 'gpt-5.4-mini' in react_llm_name:
            from langchain_openai import AzureChatOpenAI
            stop_list = ['\n']
            self.max_token_length = 15000  
            self.llm = AzureChatOpenAI(
                max_tokens=4096,
                azure_deployment=os.environ['AZURE_OPENAI_DEPLOYMENT'],
                azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
                api_key=os.environ['AZURE_OPENAI_API_KEY'],
                api_version=os.environ.get('AZURE_OPENAI_API_VERSION', '2025-01-01-preview'),
            )

        elif react_llm_name in ['ChatGLM3-6B-32K']:
            stop_list = ['\n']
            self.max_token_length = 30000
            self.llm = ChatOpenAI(
                     temperature=0,
                     max_tokens=256,
                     openai_api_key="EMPTY", 
                     openai_api_base="http://localhost:8501/v1", 
                     model_name="gpt-3.5-turbo",
                     model_kwargs={"stop": stop_list})
        
        elif react_llm_name in ['gemini']:
            self.llm = ChatGoogleGenerativeAI(temperature=0,model="gemini-pro",google_api_key=GOOGLE_API_KEY)
            self.max_token_length = 30000


        self.illegal_early_stop_patience = illegal_early_stop_patience

        self.tools = self.load_tools(tools, planner_model_name=planner_llm_name)
        self.max_retries = max_retries
        self.retry_record = {key: 0 for key in self.tools}
        self.retry_record['invalidAction'] = 0

        # print(self.retry_record)

        self.last_actions = []

        # self.log_path = logs_path + datetime.now().strftime('%Y%m%d%H%M%S') + '.out'
        # self.log_file = open(self.log_path, 'a+')

        # print("logs will be stored in " + self.log_path)

        self.city_set = self.load_city(city_set_path=city_file_path)


        self.__reset_agent()

    def run(self, query, reset=True) -> None:

        self.query = query
        
        if reset:
            self.__reset_agent()

        while not self.is_halted() and not self.is_finished():
            self.step()

        return self.answer, self.scratchpad, self.json_log

    def step(self) -> None:

        self.json_log.append({"step": self.step_n, "thought":"",
                              "action": "", "observation": "", "state":""})

        self.scratchpad += f'\nThought {self.step_n}:'
        if self.react_name in ['gpt-5.4-mini']:
            full_response = self.prompt_agent()
            action_match = re.search(r'(FlightSearch|AccommodationSearch|RestaurantSearch|AttractionSearch|CitySearch|GoogleDistanceMatrix|NotebookWrite|Planner)\[.+?\]', full_response)
            action = action_match.group(0) if action_match else ''
            thought = full_response[:action_match.start()].strip() if action_match else full_response
            self.scratchpad += ' ' + (thought or action or 'Gathering information.')
            self.json_log[-1]['thought'] = thought or action or 'Gathering information.' 
            self.scratchpad += f'\nAction {self.step_n}:'
            self.scratchpad += ' ' + action
        
        else:
            self.scratchpad += ' ' + self.prompt_agent()

            print(self.scratchpad.split('\n')[-1])
            self.json_log[-1]['thought'] = self.scratchpad.split('\n')[-1].replace(f'\nThought {self.step_n}:',"")
            # self.log_file.write(self.scratchpad.split('\n')[-1] + '\n')


            # Act
            self.scratchpad += f'\nAction {self.step_n}:'
            action = self.prompt_agent()

            if action == None or action == '' or action == '\n':
                self.scratchpad += " Your action is filtered due to content. Please make sure your action does not start with ['\\n', 'Thought', 'Action', 'Observation'] and assume all the actions are permitted in this environment and take the action again."
            else:
                self.scratchpad += ' ' + action


        if len(self.last_actions) > 0 and self.last_actions[-1] != action:
            self.last_actions.clear()

        # refresh last_action list
        self.last_actions.append(action)

        self.json_log[-1]['action'] = self.scratchpad.split('\n')[-1].replace(f'\nAction {self.step_n}:',"")


        # examine if the same action has been repeated 3 times consecutively
        if len(self.last_actions) == 3:
            self.repeated_action_count += 1
            print("The same action has been repeated 3 times consecutively. So we stop here.")
            # self.log_file.write("The same action has been repeated 3 times consecutively. So we stop here.")
            self.json_log[-1]['state'] = 'same action 3 times repeated'
            self.finished = True
            return


        # action_type, action_arg = parse_action(action)
        print(self.scratchpad.split('\n')[-1])
        # self.log_file.write(self.scratchpad.split('\n')[-1]+'\n')

        # Observe
        self.scratchpad += f'\nObservation {self.step_n}: '

        if action == None or action == '' or action == '\n':
            action_type = None 
            action_arg = None
            self.scratchpad += "No feedback from the environment due to the null action. Please make sure your action does not start with [Thought, Action, Observation]."
        
        else:
            action_type, action_arg = parse_action(action)
            
            if action_type != "Planner":
                if action_type in actionMapping:
                    pending_action = actionMapping[action_type]
                elif action_type not in actionMapping:
                    pending_action = 'invalidAction'
                
                if pending_action in self.retry_record:
                    if self.retry_record[pending_action] + 1 > self.max_retries:
                        action_type = 'Planner'
                        print(f"{pending_action} early stop due to {self.max_retries} max retries.")
                        # self.log_file.write(f"{pending_action} early stop due to {self.max_retries} max retries.")
                        self.json_log[-1]['state'] = f"{pending_action} early stop due to {self.max_retries} max retries."
                        self.finished = True
                        return
                    
                elif pending_action not in self.retry_record:
                    if self.retry_record['invalidAction'] + 1 > self.max_retries:
                        action_type = 'Planner'
                        print(f"invalidAction Early stop due to {self.max_retries} max retries.")
                        # self.log_file.write(f"invalidAction early stop due to {self.max_retries} max retries.")
                        self.json_log[-1]['state'] = f"invalidAction early stop due to {self.max_retries} max retries."
                        self.finished = True
                        return

            if action_type == 'FlightSearch':
                try:
                    if validate_date_format(action_arg.split(', ')[2]) and validate_city_format(action_arg.split(', ')[0],self.city_set ) and validate_city_format(action_arg.split(', ')[1],self.city_set):
                        self.scratchpad = self.scratchpad.replace(to_string(self.current_data).strip(),'Masked due to limited length. Make sure the data has been written in Notebook.')
                        self.current_data = self.tools['flights'].run(action_arg.split(', ')[0], action_arg.split(', ')[1], action_arg.split(', ')[2])
                        self.current_observation = str(to_string(self.current_data))
                        self.scratchpad += self.current_observation 
                        self.__reset_record()
                        self.json_log[-1]['state'] = f'Successful'
                        self.tool_call_counts['flights'] += 1

                except DateError:
                    self.retry_record['flights'] += 1
                    self.current_observation = f"'{action_arg.split(', ')[2]}' is not in the format YYYY-MM-DD"
                    self.scratchpad += f"'{action_arg.split(', ')[2]}' is not in the format YYYY-MM-DD"
                    self.json_log[-1]['state'] = f'Illegal args. DateError'

                except ValueError as e:
                    self.retry_record['flights'] += 1
                    self.current_observation = str(e)
                    self.scratchpad += str(e)
                    self.json_log[-1]['state'] = f'Illegal args. City Error'

                except Exception as e:
                    print(e)
                    self.retry_record['flights'] += 1
                    self.current_observation = f'Illegal Flight Search. Please try again.'
                    self.scratchpad += f'Illegal Flight Search. Please try again.'
                    self.json_log[-1]['state'] = f'Illegal args. Other Error'

            elif action_type == 'AttractionSearch':

                try:
                    if validate_city_format(action_arg, self.city_set):
                        self.scratchpad = self.scratchpad.replace(to_string(self.current_data).strip().strip(),'Masked due to limited length. Make sure the data has been written in Notebook.')
                        self.current_data = self.tools['attractions'].run(action_arg)
                        self.current_observation = to_string(self.current_data).strip('\n').strip()
                        self.scratchpad += self.current_observation
                        self.__reset_record()
                        self.json_log[-1]['state'] = f'Successful'
                        self.tool_call_counts['attractions'] += 1
                except ValueError as e:
                    self.retry_record['attractions'] += 1
                    self.current_observation = str(e)
                    self.scratchpad += str(e)
                    self.json_log[-1]['state'] = f'Illegal args. City Error'
                except Exception as e:
                    print(e)
                    self.retry_record['attractions'] += 1
                    self.current_observation = f'Illegal Attraction Search. Please try again.'
                    self.scratchpad += f'Illegal Attraction Search. Please try again.'
                    self.json_log[-1]['state'] = f'Illegal args. Other Error'

            elif action_type == 'AccommodationSearch':

                try:
                    if validate_city_format(action_arg, self.city_set):
                        self.scratchpad = self.scratchpad.replace(to_string(self.current_data).strip().strip(),'Masked due to limited length. Make sure the data has been written in Notebook.')
                        self.current_data = self.tools['accommodations'].run(action_arg)
                        self.current_observation = to_string(self.current_data).strip('\n').strip()
                        self.scratchpad += self.current_observation
                        self.__reset_record()
                        self.json_log[-1]['state'] = f'Successful'
                        self.tool_call_counts['accommodations'] += 1 

                except ValueError as e :
                    self.retry_record['accommodations'] += 1
                    self.current_observation = str(e)
                    self.scratchpad += str(e)
                    self.json_log[-1]['state'] = f'Illegal args. City Error'
                except Exception as e:
                    print(e)
                    self.retry_record['accommodations'] += 1
                    self.current_observation = f'Illegal Accommodation Search. Please try again.'
                    self.scratchpad += f'Illegal Accommodation Search. Please try again.'
                    self.json_log[-1]['state'] = f'Illegal args. Other Error'

            elif action_type == 'RestaurantSearch':

                try:
                    if validate_city_format(action_arg, self.city_set):
                        self.scratchpad = self.scratchpad.replace(to_string(self.current_data).strip().strip(),'Masked due to limited length. Make sure the data has been written in Notebook.')
                        self.current_data = self.tools['restaurants'].run(action_arg)
                        self.current_observation = to_string(self.current_data).strip()
                        self.scratchpad += self.current_observation
                        self.__reset_record()
                        self.json_log[-1]['state'] = f'Successful'
                        self.tool_call_counts['restaurants'] += 1 

                except ValueError as e:
                    self.retry_record['restaurants'] += 1
                    self.current_observation = str(e)
                    self.scratchpad += str(e)
                    self.json_log[-1]['state'] = f'Illegal args. City Error'

                except Exception as e:
                    print(e)
                    self.retry_record['restaurants'] += 1
                    self.current_observation = f'Illegal Restaurant Search. Please try again.'
                    self.scratchpad += f'Illegal Restaurant Search. Please try again.'
                    self.json_log[-1]['state'] = 'Illegal args. Other Error'
                    
            elif action_type == "CitySearch":
                try:
                    self.scratchpad = self.scratchpad.replace(to_string(self.current_data).strip(),'Masked due to limited length. Make sure the data has been written in Notebook.')
                    # self.current_data = self.tools['cities'].run(action_arg)
                    self.current_observation = to_string(self.tools['cities'].run(action_arg)).strip()
                    self.scratchpad += self.current_observation
                    self.__reset_record()
                    self.json_log[-1]['state'] = f'Successful'
                    self.tool_call_counts['cities'] += 1 

                except ValueError as e:
                    self.retry_record['cities'] += 1
                    self.current_observation = str(e)
                    self.scratchpad += str(e)
                    self.json_log[-1]['state'] = f'Illegal args. State Error'

                except Exception as e:
                    print(e)
                    self.retry_record['cities'] += 1
                    self.current_observation = f'Illegal City Search. Please try again.'
                    self.scratchpad += f'Illegal City Search. Please try again.'
                    self.json_log[-1]['state'] = 'Illegal args. Other Error'


            elif action_type == 'GoogleDistanceMatrix':

                try:
                    self.scratchpad = self.scratchpad.replace(to_string(self.current_data).strip(),'Masked due to limited length. Make sure the data has been written in Notebook.')
                    self.current_data = self.tools['googleDistanceMatrix'].run(action_arg.split(', ')[0],action_arg.split(', ')[1],action_arg.split(', ')[2])
                    self.current_observation =  to_string(self.current_data)
                    self.scratchpad += self.current_observation 
                    self.__reset_record()
                    self.json_log[-1]['state'] = f'Successful'
                    self.tool_call_counts['googleDistanceMatrix'] += 1 

                except Exception as e:
                    print(e)
                    self.retry_record['googleDistanceMatrix'] += 1
                    self.current_observation = f'Illegal GoogleDistanceMatrix. Please try again.'
                    self.scratchpad += f'Illegal GoogleDistanceMatrix. Please try again.'
                    self.json_log[-1]['state'] = f'Illegal args. Other Error'
            
            
            elif action_type == 'NotebookWrite':
                try:
                    self.scratchpad = self.scratchpad.replace(to_string(self.current_data).strip(),'Masked due to limited length. Make sure the data has been written in Notebook.')
                    self.current_observation = str(self.tools['notebook'].write(self.current_data, action_arg))
                    self.scratchpad  +=  self.current_observation
                    self.__reset_record()
                    self.json_log[-1]['state'] = f'Successful'
                    self.tool_call_counts['notebook'] += 1

                except Exception as e:
                    print(e)
                    self.retry_record['notebook'] += 1
                    self.current_observation = f'{e}'
                    self.scratchpad += f'{e}'
                    self.json_log[-1]['state'] = f'Illegal args. Other Error'
            

            elif action_type == "Planner":
                # try:

                    self.current_observation = str(self.tools['planner'].run(str(self.tools['notebook'].list_all()),action_arg))
                    self.scratchpad  +=  self.current_observation
                    self.answer = self.current_observation
                    self.__reset_record()
                    self.json_log[-1]['state'] = f'Successful'
                    self.tool_call_counts['planner'] += 1
                    self.halt_reason = 'success'

            else:
                self.retry_record['invalidAction'] += 1
                self.current_observation = 'Invalid Action. Valid Actions are  FlightSearch[Departure City, Destination City, Date] / ' \
                                   'AccommodationSearch[City] /  RestaurantSearch[City] / NotebookWrite[Short Description] / AttractionSearch[City] / CitySearch[State] / GoogleDistanceMatrix[Origin, Destination, Mode] and Planner[Query].'
                self.scratchpad += self.current_observation
                self.json_log[-1]['state'] = f'invalidAction'

        if action == None or action == '' or action == '\n':
            print(f'Observation {self.step_n}: ' + "No feedback from the environment due to the null action.")
            # write(f'Observation {self.step_n}: ' + "Your action is filtered due to content. Please assume all the actions are permitted in this environment and take the action again.")
            self.json_log[-1]['observation'] = "No feedback from the environment due to the null action."
        else:
            no_data_phrases = ['no valid information', 'none', 'no result', 'not found']
            if any(phrase in self.current_observation.lower() for phrase in no_data_phrases):
                self.tool_call_counts['no_data_calls'] += 1
            if action_type != 'Planner' and len(self.current_observation) > 4000:
                truncated = self.current_observation[:4000] + '\n...[truncated]'
                self.scratchpad = self.scratchpad.replace(self.current_observation, truncated)
                self.current_observation = truncated
            print(f'Observation {self.step_n}: ' + self.current_observation+'\n')
            # rite(f'Observation {self.step_n}: ' + self.current_observation+'\n')
            self.json_log[-1]['observation'] = self.current_observation

        if action_type == 'FlightSearch' and self.json_log[-1]['state'] == 'Successful':
            self.info_gathered['flights'] = True
        elif action_type == 'AccommodationSearch' and self.json_log[-1]['state'] == 'Successful':
            self.info_gathered['accommodations'] = True
        elif action_type == 'RestaurantSearch' and self.json_log[-1]['state'] == 'Successful':
            self.info_gathered['restaurants'] = True
        elif action_type == 'AttractionSearch' and self.json_log[-1]['state'] == 'Successful':
            self.info_gathered['attractions'] = True
        elif action_type == 'CitySearch' and self.json_log[-1]['state'] == 'Successful':
            self.info_gathered['cities'] = True
        elif action_type == 'GoogleDistanceMatrix' and self.json_log[-1]['state'] == 'Successful':
            self.info_gathered['distances'] = True

        # record step metrics
        self.step_metrics.append({
            'step_n': self.step_n,
            'prompt_tokens': self._last_prompt_tokens,
            'completion_tokens': self._last_completion_tokens,
            'action_type': action_type if action_type else '',
            'action_success': self.json_log[-1]['state'] == 'Successful',
            'step_latency_seconds': round(self._last_step_latency, 3)
        })
        self.step_n += 1


        if action_type and action_type == 'Planner' and self.retry_record['planner']==0:
            
            self.finished = True
            self.answer = self.current_observation
            self.step_n += 1
            return

    def prompt_agent(self) -> str:
        while True:
            try:
                # self.scratchpad = truncate_scratchpad(self.scratchpad, n_tokens=1600)
                prompt = self._build_agent_prompt() 
                step_start = time.time()
                if self.react_name == 'gemini':
                    request = format_step(self.llm.invoke(prompt,stop=['\n']).content)
                elif self.react_name in ['llama', 'qwen', 'qwen25-7B']:
                    result = self.llm.invoke(prompt)
                    result = result.split('\n')[0]
                    request = format_step(result)
                elif self.react_name in ['gpt-5.4-mini']:
                    # time.sleep(3)
                    result = self.llm.invoke(prompt)
                    # gpt-5.4-mini returns reasoning tokens, not content
                    content = result.content
                    if not content and hasattr(result, 'additional_kwargs'):
                        # try reasoning content
                        reasoning = result.additional_kwargs.get('reasoning_content', '')
                        content = reasoning
                    if not content:
                        # fallback: check raw response chunks
                        content = str(result)
                    request = format_step(content)
                else:
                    print("Calling LLM...")
                    time.sleep(3) 
                    result = self.llm.invoke(prompt)
                    print(f"LLM returned: {repr(result[:100])}")
                    request = format_step(result)

                # print(request)
                step_latency = time.time() - step_start

                # count tokens
                prompt_tokens = len(prompt) // 4
                completion_tokens = len(request) // 4
                self.react_prompt_tokens += prompt_tokens
                self.react_completion_tokens += completion_tokens
                self.total_tokens_used += (prompt_tokens + completion_tokens)

                # store for step metrics
                self._last_prompt_tokens = prompt_tokens
                self._last_completion_tokens = completion_tokens
                self._last_step_latency = step_latency
                return request
            except:
                catch_openai_api_error()
                print(prompt)
                print(len(self._build_agent_prompt()) // 4)
                time.sleep(5)
                break
            
        return ''

    def _build_agent_prompt(self) -> str:
        if self.mode == "zero_shot":
            return self.agent_prompt.format(
                query=self.query,
                scratchpad=self.scratchpad)

    def is_finished(self) -> bool:
        return self.finished

    def is_halted(self) -> bool:
        if self.finished:
            return False
        if self.step_n > self.max_steps:
            self.halt_reason = 'max_steps'
            return True
        if self._last_prompt_tokens > self.max_token_length:
            self.halt_reason = 'max_tokens'
            return True
        return False

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.answer = ''
        self.scratchpad: str = ''
        self.__reset_record()
        self.json_log = []
        self.current_observation = ''
        self.current_data = None
        self.last_actions = []
        self.tool_call_counts = {k: 0 for k in self.tool_call_counts}
        self.total_tokens_used = 0
        self.halt_reason = 'unknown'
        self.step_metrics = []
        self.react_prompt_tokens = 0
        self.react_completion_tokens = 0
        self.repeated_action_count = 0
        self.info_gathered = {
            'flights': False,
            'accommodations': False,
            'restaurants': False,
            'attractions': False,
            'cities': False,
            'distances': False
        }
        self._last_prompt_tokens = 0
        self._last_completion_tokens = 0
        self._last_step_latency = 0.0

        if 'notebook' in self.tools:
            self.tools['notebook'].reset()
    
    def __reset_record(self) -> None:
        self.retry_record = {key: 0 for key in self.retry_record}
        self.retry_record['invalidAction'] = 0


    def load_tools(self, tools: List[str], planner_model_name=None) -> Dict[str, Any]:
        tools_map = {}
        for tool_name in tools:
            module = importlib.import_module("tools.{}.apis".format(tool_name))
            
            # Avoid instantiating the planner tool twice 
            if tool_name == 'planner' and planner_model_name is not None:
                tools_map[tool_name] = getattr(module, tool_name[0].upper()+tool_name[1:])(model_name=planner_model_name, shared_llm=getattr(self, 'shared_pipeline', None))
            else:
                tools_map[tool_name] = getattr(module, tool_name[0].upper()+tool_name[1:])()
        return tools_map

    def load_city(self, city_set_path: str) -> List[str]:
        city_set = []
        lines = open(city_set_path, 'r').read().strip().split('\n')
        for unit in lines:
            city_set.append(unit)
        return city_set

### String Stuff ###
gpt2_enc = tiktoken.encoding_for_model("text-davinci-003")


def parse_action(string):
    pattern = r'^(\w+)\[(.+)\]$'
    match = re.match(pattern, string)

    try:
        if match:
            action_type = match.group(1)
            action_arg = match.group(2)
            return action_type, action_arg
        else:
            return None, None
        
    except:
        return None, None

def format_step(step: str) -> str:
    step = re.sub(r'^(Action|Thought)\s+\d+:\s*', '', step.strip())
    match = re.match(r'(\w+\[.+?\])', step)
    if match:
        step = match.group(1)
    return step.strip('\n').strip().replace('\n', '')



def truncate_scratchpad(scratchpad: str, n_tokens: int = 1600, tokenizer=gpt2_enc) -> str:
    lines = scratchpad.split('\n')
    observations = filter(lambda x: x.startswith('Observation'), lines)
    observations_by_tokens = sorted(observations, key=lambda x: len(tokenizer.encode(x)))
    while len(gpt2_enc.encode('\n'.join(lines))) > n_tokens:
        largest_observation = observations_by_tokens.pop(-1)
        ind = lines.index(largest_observation)
        lines[ind] = largest_observation.split(':')[0] + ': [Masked due to limited length. Make sure the data has been written in Notebook.]'
    return '\n'.join(lines)


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the|usd)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def EM(answer, key) -> bool:
    return normalize_answer(str(answer)) == normalize_answer(str(key))


def remove_observation_lines(text, step_n):
    pattern = re.compile(rf'^Observation {step_n}.*', re.MULTILINE)
    return pattern.sub('', text)

def validate_date_format(date_str: str) -> bool:
    pattern = r'^\d{4}-\d{2}-\d{2}$'
    
    if not re.match(pattern, date_str):
        raise DateError
    return True

def validate_city_format(city_str: str, city_set: list) -> bool:
    if city_str not in city_set:
        raise ValueError(f"{city_str} is not valid city in {str(city_set)}.")
    return True

def extract_budget(query: str) -> int:
    match = re.search(r'\$([0-9,]+)', query)
    if match:
        return int(match.group(1).replace(',', ''))
    return -1

def parse_args_string(s: str) -> dict:
    # Split the string by commas
    segments = s.split(",")
    
    # Initialize an empty dictionary to store the results
    result = {}
    
    for segment in segments:
        # Check for various operators
        if "contains" in segment:
            if "~contains" in segment:
                key, value = segment.split("~contains")
                operator = "~contains"
            else:
                key, value = segment.split("contains")
                operator = "contains"
        elif "<=" in segment:
            key, value = segment.split("<=")
            operator = "<="
        elif ">=" in segment:
            key, value = segment.split(">=")
            operator = ">="
        elif "=" in segment:
            key, value = segment.split("=")
            operator = "="
        else:
            continue  # If no recognized operator is found, skip to the next segment
                
        # Strip spaces and single quotes
        key = key.strip()
        value = value.strip().strip("'")
        
        # Store the result with the operator included
        result[key] = (operator, value)
        
    return result

def to_string(data) -> str:
    if data is not None:
        if type(data) == DataFrame:
            return data.to_string(index=False)
        else:
            return str(data)
    else:
        return str(None)

if __name__ == '__main__':

    tools_list = ["notebook","flights","attractions","accommodations","restaurants","googleDistanceMatrix","planner","cities"]
    # model_name = ['gpt-3.5-turbo-1106','gpt-4-1106-preview','gemini','mistral-7B-32K','mixtral','ChatGLM3-6B-32K'][2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", type=str, default="validation")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-1106")
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--num_queries", type=int, default=None)
    parser.add_argument("--start_query", type=int, default=1)
    args = parser.parse_args()
    if args.set_type == 'validation':
        query_data_list  = load_dataset('osunlp/TravelPlanner','validation')['validation']
    elif args.set_type == 'test':
        query_data_list  = load_dataset('osunlp/TravelPlanner','test')['test']
    n = args.num_queries if args.num_queries else len(query_data_list)
    numbers = [i for i in range(args.start_query, args.start_query + n)]
    agent = ReactAgent(None, tools=tools_list,max_steps=30,react_llm_name=args.model_name,planner_llm_name=args.model_name)
    
    all_costs = [] 
    all_steps = [] 
    
    with get_openai_callback() as cb:
        
        for number in tqdm(numbers[:]):
            query = query_data_list[number-1]['query']
              # check if the directory exists
            if not os.path.exists(os.path.join(f'{args.output_dir}/{args.set_type}')):
                os.makedirs(os.path.join(f'{args.output_dir}/{args.set_type}'))
            if not os.path.exists(os.path.join(f'{args.output_dir}/{args.set_type}/generated_plan_{number}.json')):
                result =  [{}]
            else:
                result = json.load(open(os.path.join(f'{args.output_dir}/{args.set_type}/generated_plan_{number}.json')))
            
            # track runtime
            start_time = time.time()
            
            while True:
                planner_results, scratchpad, action_log  = agent.run(query)
                if planner_results != None:
                    break
            
            end_time = time.time()
            runtime = end_time - start_time

            # compute step metrics from action_log
            num_steps = len(action_log)
            num_invalid = sum(1 for s in action_log if 'invalidAction' in s.get('state','') or 'Illegal' in s.get('state',''))
            num_successful = sum(1 for s in action_log if s.get('state','') == 'Successful')
            wasted_steps = num_steps - num_successful  

            # token costs using GPT-3.5 pricing as proxy
            react_prompt_tokens = agent.react_prompt_tokens
            react_completion_tokens = agent.react_completion_tokens

            planner_pt = getattr(agent.tools['planner'], 'planner_prompt_tokens', 0)
            planner_ct = getattr(agent.tools['planner'], 'planner_completion_tokens', 0)

            
            total_prompt = react_prompt_tokens + planner_pt
            total_completion = react_completion_tokens + planner_ct
            total_tokens = total_prompt + total_completion

            # cost
            total_cost_usd = round(
                (total_prompt / 1000) * 0.0015 +
                (total_completion / 1000) * 0.002,
                5
            )

            avg_step_latency = round(
                sum(s['step_latency_seconds'] for s in agent.step_metrics) / len(agent.step_metrics)
                if agent.step_metrics else 0, 3
            )
            # store cost metrics
            cost_metrics = {
                'query_id': number,
                'query': query,
                'model_name': args.model_name,
                'react_prompt_tokens': react_prompt_tokens,
                'react_completion_tokens': react_completion_tokens,
                'planner_prompt_tokens': planner_pt,
                'planner_completion_tokens': planner_ct,
                'total_prompt_tokens': total_prompt,
                'total_completion_tokens': total_completion,
                'total_tokens': total_tokens,
                'total_cost_usd': total_cost_usd,
                'total_steps': num_steps,
                'wasted_steps': wasted_steps,
                'invalid_action_count': num_invalid,
                'repeated_action_count': agent.repeated_action_count,
                'tool_call_counts': {
                    'FlightSearch': agent.tool_call_counts['flights'],
                    'AccommodationSearch': agent.tool_call_counts['accommodations'],
                    'RestaurantSearch': agent.tool_call_counts['restaurants'],
                    'AttractionSearch': agent.tool_call_counts['attractions'],
                    'CitySearch': agent.tool_call_counts['cities'],
                    'GoogleDistanceMatrix': agent.tool_call_counts['googleDistanceMatrix'],
                    'NotebookWrite': agent.tool_call_counts['notebook'],
                    'Planner': agent.tool_call_counts['planner'],
                },
                'info_gathered': agent.info_gathered,
                'wall_clock_seconds': round(runtime, 2),
                'avg_step_latency': avg_step_latency,
                'delivered': planner_results not in [None, '', 'Max Token Length Exceeded.'],
                'halt_reason': agent.halt_reason,
                'tool_calls_no_data': agent.tool_call_counts['no_data_calls'],
                'user_budget': extract_budget(query),
            }
            
            flat_cost_metrics = {
                'query_id': number,
                'model_name': args.model_name,

                # tokens
                'react_prompt_tokens': react_prompt_tokens,
                'react_completion_tokens': react_completion_tokens,
                'planner_prompt_tokens': planner_pt,
                'planner_completion_tokens': planner_ct,
                'total_prompt_tokens': total_prompt,
                'total_completion_tokens': total_completion,
                'total_tokens': total_tokens,
                'total_cost_usd': total_cost_usd,

                # steps
                'total_steps': num_steps,
                'wasted_steps': wasted_steps,
                'invalid_action_count': num_invalid,
                'repeated_action_count': agent.repeated_action_count,

                # latency
                'wall_clock_seconds': round(runtime, 2),
                'avg_step_latency': avg_step_latency,

                # execution
                'delivered': planner_results not in [None, '', 'Max Token Length Exceeded.'],
                'halt_reason': agent.halt_reason,
                'tool_calls_no_data': agent.tool_call_counts['no_data_calls'],
                'user_budget': extract_budget(query),

                # tool counts (flattened)
                'tool_flights': agent.tool_call_counts['flights'],
                'tool_accommodations': agent.tool_call_counts['accommodations'],
                'tool_restaurants': agent.tool_call_counts['restaurants'],
                'tool_attractions': agent.tool_call_counts['attractions'],
                'tool_cities': agent.tool_call_counts['cities'],
                'tool_distances': agent.tool_call_counts['googleDistanceMatrix'],
                'tool_notebook': agent.tool_call_counts['notebook'],
                'tool_planner': agent.tool_call_counts['planner'],

                # info gathered (flattened)
                'got_flights': agent.info_gathered['flights'],
                'got_accommodations': agent.info_gathered['accommodations'],
                'got_restaurants': agent.info_gathered['restaurants'],
                'got_attractions': agent.info_gathered['attractions'],
                'got_cities': agent.info_gathered['cities'],
                'got_distances': agent.info_gathered['distances'],
            }

            all_costs.append(flat_cost_metrics)

            for step in agent.step_metrics:
                step_row = {
                    'query_id': number,
                    'model_name': args.model_name,

                    'step_n': step['step_n'],
                    'prompt_tokens': step['prompt_tokens'],
                    'completion_tokens': step['completion_tokens'],
                    'step_tokens': step['prompt_tokens'] + step['completion_tokens'],

                    'action_type': step['action_type'],
                    'action_success': step['action_success'],
                    'step_latency_seconds': step['step_latency_seconds'],

                }
                all_steps.append(step_row)

            if planner_results == 'Max Token Length Exceeded.':
                result[-1][f'{args.model_name}_two-stage_results_logs'] = scratchpad 
                result[-1][f'{args.model_name}_two-stage_results'] = 'Max Token Length Exceeded.'
                action_log[-1]['state'] = 'Max Token Length of Planner Exceeded.'
                result[-1][f'{args.model_name}_two-stage_action_logs'] = action_log
            else:
                result[-1][f'{args.model_name}_two-stage_results_logs'] = scratchpad 
                result[-1][f'{args.model_name}_two-stage_results'] = planner_results
                result[-1][f'{args.model_name}_two-stage_action_logs'] = action_log

            result[-1][f'{args.model_name}_cost_metrics'] = cost_metrics

            # write to json file
            with open(os.path.join(f'{args.output_dir}/{args.set_type}/generated_plan_{number}.json'), 'w') as f:
                json.dump(result, f, indent=4)
        
    costs_df = pd.DataFrame(all_costs)
    csv_path = os.path.join(f'{args.output_dir}/{args.set_type}/cost_metrics_{args.model_name}.csv')
    if os.path.exists(csv_path):
        costs_df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        costs_df.to_csv(csv_path, index=False)
    print(costs_df.describe())

    steps_df = pd.DataFrame(all_steps)
    steps_csv_path = os.path.join(f'{args.output_dir}/{args.set_type}/step_metrics_{args.model_name}.csv')

    if os.path.exists(steps_csv_path):
        steps_df.to_csv(steps_csv_path, mode='a', header=False, index=False)
    else:
        steps_df.to_csv(steps_csv_path, index=False)
    

