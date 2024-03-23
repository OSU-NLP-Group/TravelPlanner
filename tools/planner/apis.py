import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from langchain.prompts import PromptTemplate
from agents.prompts import planner_agent_prompt, cot_planner_agent_prompt, react_planner_agent_prompt,reflect_prompt,react_reflect_planner_agent_prompt, REFLECTION_HEADER
from langchain.chat_models import ChatOpenAI
from langchain.llms.base import BaseLLM
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from env import ReactEnv,ReactReflectEnv
import tiktoken
import re
import openai
import time
from enum import Enum
from typing import List, Union, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
import argparse


OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']


def catch_openai_api_error():
    error = sys.exc_info()[0]
    if error == openai.error.APIConnectionError:
        print("APIConnectionError")
    elif error == openai.error.RateLimitError:
        print("RateLimitError")
        time.sleep(60)
    elif error == openai.error.APIError:
        print("APIError")
    elif error == openai.error.AuthenticationError:
        print("AuthenticationError")
    else:
        print("API error:", error)


class ReflexionStrategy(Enum):
    """
    REFLEXION: Apply reflexion to the next reasoning trace 
    """
    REFLEXION = 'reflexion'


class Planner:
    def __init__(self,
                 # args,
                 agent_prompt: PromptTemplate = planner_agent_prompt,
                 model_name: str = 'gpt-3.5-turbo-1106',
                 ) -> None:

        self.agent_prompt = agent_prompt
        self.scratchpad: str = ''
        self.model_name = model_name
        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

        if model_name in  ['mistral-7B-32K']:
            self.llm = ChatOpenAI(temperature=0,
                     max_tokens=4096,
                     openai_api_key="EMPTY", 
                     openai_api_base="http://localhost:8301/v1", 
                     model_name="gpt-3.5-turbo")
        
        elif model_name in  ['ChatGLM3-6B-32K']:
            self.llm = ChatOpenAI(temperature=0,
                     max_tokens=4096,
                     openai_api_key="EMPTY", 
                     openai_api_base="http://localhost:8501/v1", 
                     model_name="gpt-3.5-turbo")
            
        elif model_name in ['mixtral']:
            self.max_token_length = 30000
            self.llm = ChatOpenAI(temperature=0,
                     max_tokens=4096,
                     openai_api_key="EMPTY", 
                     openai_api_base="http://localhost:8501/v1", 
                     model_name="YOUR/MODEL/PATH")
            
        elif model_name in ['gemini']:
            self.llm = ChatGoogleGenerativeAI(temperature=0,model="gemini-pro",google_api_key=GOOGLE_API_KEY)
        else:
            self.llm = ChatOpenAI(model_name=model_name, temperature=0, max_tokens=4096, openai_api_key=OPENAI_API_KEY)


        print(f"PlannerAgent {model_name} loaded.")

    def run(self, text, query, log_file=None) -> str:
        if log_file:
            log_file.write('\n---------------Planner\n'+self._build_agent_prompt(text, query))
        # print(self._build_agent_prompt(text, query))
        if self.model_name in ['gemini']:
            return str(self.llm.invoke(self._build_agent_prompt(text, query)).content)
        else:
            if len(self.enc.encode(self._build_agent_prompt(text, query))) > 12000:
                return 'Max Token Length Exceeded.'
            else:
                return self.llm([HumanMessage(content=self._build_agent_prompt(text, query))]).content

    def _build_agent_prompt(self, text, query) -> str:
        return self.agent_prompt.format(
            text=text,
            query=query)


class ReactPlanner:
    """
    A question answering ReAct Agent.
    """
    def __init__(self,
                 agent_prompt: PromptTemplate = react_planner_agent_prompt,
                 model_name: str = 'gpt-3.5-turbo-1106',
                 ) -> None:
        
        self.agent_prompt = agent_prompt
        self.react_llm = ChatOpenAI(model_name=model_name, temperature=0, max_tokens=1024, openai_api_key=OPENAI_API_KEY,model_kwargs={"stop": ["Action","Thought","Observation"]})
        self.env = ReactEnv()
        self.query = None
        self.max_steps = 30
        self.reset()
        self.finished = False
        self.answer = ''
        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def run(self, text, query, reset = True) -> None:

        self.query = query
        self.text = text

        if reset:
            self.reset()
        

        while not (self.is_halted() or self.is_finished()):
            self.step()
        
        return self.answer, self.scratchpad

    
    def step(self) -> None:
        # Think
        self.scratchpad += f'\nThought {self.curr_step}:'
        self.scratchpad += ' ' + self.prompt_agent()
        print(self.scratchpad.split('\n')[-1])

        # Act
        self.scratchpad += f'\nAction {self.curr_step}:'
        action = self.prompt_agent()
        self.scratchpad += ' ' + action
        print(self.scratchpad.split('\n')[-1])

        # Observe
        self.scratchpad += f'\nObservation {self.curr_step}: '

        action_type, action_arg = parse_action(action)

        if action_type == 'CostEnquiry':
            try:
                input_arg = eval(action_arg)
                if type(input_arg) != dict:
                    raise ValueError('The sub plan can not be parsed into json format, please check. Only one day plan is supported.')
                observation = f'Cost: {self.env.run(input_arg)}'
            except SyntaxError:
                observation = f'The sub plan can not be parsed into json format, please check.'
            except ValueError as e:
                observation = str(e)
        
        elif action_type == 'Finish':
            self.finished = True
            observation = f'The plan is finished.'
            self.answer = action_arg
        
        else:
            observation = f'Action {action_type} is not supported.'
        
        self.curr_step += 1

        self.scratchpad += observation
        print(self.scratchpad.split('\n')[-1])

    def prompt_agent(self) -> str:
        while True:
            try:
                return format_step(self.react_llm([HumanMessage(content=self._build_agent_prompt())]).content)
            except:
                catch_openai_api_error()
                print(self._build_agent_prompt())
                print(len(self.enc.encode(self._build_agent_prompt())))
                time.sleep(5)
    
    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
                            query = self.query,
                            text = self.text,
                            scratchpad = self.scratchpad)
    
    def is_finished(self) -> bool:
        return self.finished

    def is_halted(self) -> bool:
        return ((self.curr_step > self.max_steps) or (
                    len(self.enc.encode(self._build_agent_prompt())) > 14000)) and not self.finished

    def reset(self) -> None:
        self.scratchpad = ''
        self.answer = ''
        self.curr_step = 1
        self.finished = False


class ReactReflectPlanner:
    """
    A question answering Self-Reflecting React Agent.
    """
    def __init__(self,
                 agent_prompt: PromptTemplate = react_reflect_planner_agent_prompt,
                reflect_prompt: PromptTemplate = reflect_prompt,
                 model_name: str = 'gpt-3.5-turbo-1106',
                 ) -> None:
        
        self.agent_prompt = agent_prompt
        self.reflect_prompt = reflect_prompt
        if model_name in ['gemini']:
            self.react_llm = ChatGoogleGenerativeAI(temperature=0,model="gemini-pro",google_api_key=GOOGLE_API_KEY)
            self.reflect_llm = ChatGoogleGenerativeAI(temperature=0,model="gemini-pro",google_api_key=GOOGLE_API_KEY)
        else:
            self.react_llm = ChatOpenAI(model_name=model_name, temperature=0, max_tokens=1024, openai_api_key=OPENAI_API_KEY,model_kwargs={"stop": ["Action","Thought","Observation,'\n"]})
            self.reflect_llm = ChatOpenAI(model_name=model_name, temperature=0, max_tokens=1024, openai_api_key=OPENAI_API_KEY,model_kwargs={"stop": ["Action","Thought","Observation,'\n"]})
        self.model_name = model_name
        self.env = ReactReflectEnv()
        self.query = None
        self.max_steps = 30
        self.reset()
        self.finished = False
        self.answer = ''
        self.reflections: List[str] = []
        self.reflections_str: str = ''
        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def run(self, text, query, reset = True) -> None:

        self.query = query
        self.text = text

        if reset:
            self.reset()
        

        while not (self.is_halted() or self.is_finished()):
            self.step()
            if self.env.is_terminated and not self.finished:
                self.reflect(ReflexionStrategy.REFLEXION)

        
        return self.answer, self.scratchpad

    
    def step(self) -> None:
        # Think
        self.scratchpad += f'\nThought {self.curr_step}:'
        self.scratchpad += ' ' + self.prompt_agent()
        print(self.scratchpad.split('\n')[-1])

        # Act
        self.scratchpad += f'\nAction {self.curr_step}:'
        action = self.prompt_agent()
        self.scratchpad += ' ' + action
        print(self.scratchpad.split('\n')[-1])

        # Observe
        self.scratchpad += f'\nObservation {self.curr_step}: '

        action_type, action_arg = parse_action(action)

        if action_type == 'CostEnquiry':
            try:
                input_arg = eval(action_arg)
                if type(input_arg) != dict:
                    raise ValueError('The sub plan can not be parsed into json format, please check. Only one day plan is supported.')
                observation = f'Cost: {self.env.run(input_arg)}'
            except SyntaxError:
                observation = f'The sub plan can not be parsed into json format, please check.'
            except ValueError as e:
                observation = str(e)
        
        elif action_type == 'Finish':
            self.finished = True
            observation = f'The plan is finished.'
            self.answer = action_arg
        
        else:
            observation = f'Action {action_type} is not supported.'
        
        self.curr_step += 1

        self.scratchpad += observation
        print(self.scratchpad.split('\n')[-1])

    def reflect(self, strategy: ReflexionStrategy) -> None:
        print('Reflecting...')
        if strategy == ReflexionStrategy.REFLEXION: 
            self.reflections += [self.prompt_reflection()]
            self.reflections_str = format_reflections(self.reflections)
        else:
            raise NotImplementedError(f'Unknown reflection strategy: {strategy}')
        print(self.reflections_str)

    def prompt_agent(self) -> str:
        while True:
            try:
                if self.model_name in ['gemini']:
                    return format_step(self.react_llm.invoke(self._build_agent_prompt()).content)
                else:
                    return format_step(self.react_llm([HumanMessage(content=self._build_agent_prompt())]).content)
            except:
                catch_openai_api_error()
                print(self._build_agent_prompt())
                print(len(self.enc.encode(self._build_agent_prompt())))
                time.sleep(5)
    
    def prompt_reflection(self) -> str:
        while True:
            try:
                if self.model_name in ['gemini']:
                    return format_step(self.reflect_llm.invoke(self._build_reflection_prompt()).content)
                else:
                    return format_step(self.reflect_llm([HumanMessage(content=self._build_reflection_prompt())]).content)
            except:
                catch_openai_api_error()
                print(self._build_reflection_prompt())
                print(len(self.enc.encode(self._build_reflection_prompt())))
                time.sleep(5)
    
    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
                            query = self.query,
                            text = self.text,
                            scratchpad = self.scratchpad,
                            reflections = self.reflections_str)
    
    def _build_reflection_prompt(self) -> str:
        return self.reflect_prompt.format(
                            query = self.query,
                            text = self.text,
                            scratchpad = self.scratchpad)
    
    def is_finished(self) -> bool:
        return self.finished

    def is_halted(self) -> bool:
        return ((self.curr_step > self.max_steps) or (
                    len(self.enc.encode(self._build_agent_prompt())) > 14000)) and not self.finished

    def reset(self) -> None:
        self.scratchpad = ''
        self.answer = ''
        self.curr_step = 1
        self.finished = False
        self.reflections = []
        self.reflections_str = ''
        self.env.reset()

def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n', '')

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

def format_reflections(reflections: List[str],
                        header: str = REFLECTION_HEADER) -> str:
    if reflections == []:
        return ''
    else:
        return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])

# if __name__ == '__main__':
    