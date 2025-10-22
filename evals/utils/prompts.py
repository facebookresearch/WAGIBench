# Copyright (c) Meta Platforms, Inc. and affiliates.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

### Maestro OB2 Goal Inference Task

OB2_MCQ_PROMPT = "You are an intelligent assistant. You will be given a video and four options, each containing a digital action formatted as a dictionary. Your task is to respond with which option (A or B or C or D) is the most likely digital action that follows a human's context window of actions present in the video.\n\n"

OB2_MCQ_NOTE_PROMPT = """
NOTE:
- Your response should contain only the option letter A, B, C, or D. Only respond with one letter. Do not repeat the option.
- Wrap your final answer with <answer> and </answer> tag. For instance, an example full output should look like this: <answer>A</answer>
"""

OB2_DIGITAL_ACTION_GENERATION_SIMPLE_PROMPT = """
<DIGITAL_ACTION_PREDICTION> You are an intelligent assistant. Based on the images provided representing what the user is seeing, what digital action might the user want to do on their phones? 

These are the digital actions you can choose from: 
- search
- store_memory
- temporal_attention
- guided_activity
- control_environment
- communication
- translate
- find_directions
- shop
- entertain

NOTE:
- Output a Python dictionary
"""

OB2_DIGITAL_ACTION_EXAMPLES = """{
    “search” : {
        type: “search”,
        source: [“world”,        # general knowledge, including facts, news, weather, etc
            “timeline” ],   # about the user’s history or environment
                        # including events saved via “memory” action
        query_item: str,
        # e.g., “Red delicious apple” for objects
        # e.g., “Slicing an onion” for actions
        # e.g., “Knitting” for activities
        # e.g., “Leprous concert next week” for external entities
        query: str # e.g., “Nutritional content”
    },
    “store_memory” : {                   # Save an event to personal timeline
        type: “store_memory”,
        content: str,
        # e.g., “I took my vitamins today”
        # e.g., “Add bowtie pasta to my grocery list”
    },
    “temporal_attention” : {
        type: “temporal_attention”,
            action: [“set”, “unset”],
                time: str, # e.g., “Ten minutes”
            content: str # e.g., “Get ready for work”
    },
    “guided_activity” : {
        type: “guided_activity”,
            content: str, 
        # e.g., “guided meditation”
        # e.g., “yoga workout routine”
            time: str, # e.g., “ten minutes”
    },
    “control_environment” : {
        type: “control_environment”,
        target: str, # e.g., “smart lights in kitchen”
        value: str # e.g., “50%” or “brightness: 50%”
    },
    “communication” : {
        type: “communication”,
        action: [“message”, “share”],
        target: str # e.g., “<name of person>” or “<instagram>”,
        content: str
    },
    “translate” : {
        type: “translate”,
        modality: [“text”, “audio”],
        src_language: str # e.g., german
    },
    "find_directions"  : {
        type: "find_directions",
        to: str #.g.,
    }
    "shop" : {
        type: "shop",
        content: str # what are you looking for
    }
    “entertain”: {
        type: “entertain”,
        modality: [“audio”, “vision”],
        content: str  # e.g., “rock playlist” or “cat videos”
    }
}
"""

OB2_DIGITAL_ACTION_GENERATION_PROMPT = """
<DIGITAL_ACTION_PREDICTION> You are an intelligent agent. Your task is to answer the following question: Based on the images provided representing what the user is seeing, what digital action might they want to do on their phones?
To answer the question above, you MUST PICK A DIGITAL ACTION from the following TEMPLATE: 

{examples}

After you decide the action type, fill in other fields in the TEMPLATE according to the video contexts provided. DO NOT simply copy from templates.
""".format(
    examples=OB2_DIGITAL_ACTION_EXAMPLES
)

REFERENCE_JUDGE_PROMPT_V1 = """
You are a judge. Your task is to evaluate the digital action predicted by the model. 

Ground-truth digital action: 
{{gt}}

Model predicted digital action: 
{{prediction}}

NOTE:
- The digital action is a dictionary with the following keys: "type", "content", etc. 
- Please check each field of the predicted digital action and compare it with the ground-truth digital action. 
- Please use a scale of 1 to {max_scale}, where 1 is the worst and {max_scale} is the best, to evaluate the digital action. 
- Please only output the score and nothing else. Do not add any explanation to your final answer. 
- Wrap the final score with <score> and </score> tag. For instance, an example full output should look like this: <score>{max_scale}</score>
"""

REFERENCE_JUDGE_PROMPT_V2 = """
You are a wise human with an exceptional understanding of real-world nuance.
You must carefully determine the goal alignment between an ideal goal and a candidate goal, using a discrete scale from 1 to {max_scale}, where 1 is the worst and {max_scale} is the best. 
If the candidate goal yields the same outcome as the ideal goal, the candidate goal should be given a goal alignment of {max_scale}.
If the candidate goal yields the opposite outcome as the ideal goal, the candidate goal should be given a goal alignment of 1.
If the candidate goal is in between these two extremes, it should be given an intermediate goal alignment score.
Please check each field of the candidate digital goal and compare it with the ideal digital goal. 

Ideal digital goal: 
{{gt}}

Candidate digital goal: 
{{prediction}}

NOTE:
- Given an ideal goal and candidate goal, give me the goal alignment of the candidate goal. 
- Please only output the score and nothing else. Do not add any explanation to your final answer.
- Wrap the final score with <score> and </score> tag. For instance, an example full output should look like this: <score>{max_scale}</score>
"""

CONTEXT_REFERENCE_JUDGE_PROMPT = """
You are a judge. Your task is to evaluate the digital action predicted by the model. The digital action is is a dictionary falling into one of the following goal types: 
{examples}

Specifically, you are given a set of contexts that describe the video, as well as the predicted digital action that the user might want to take on their phones given the observed contexts.
You need to decide whether the predicted digital action is sensible given the contexts. 
To facilitate this, you are further given a reference digital action that is known to be sensible given the contexts. 

Contexts: 
{{contexts}}

Reference digital action: 
{{gt}}

Model predicted digital action: 
{{prediction}}

NOTE:
- To evaluate the predicted digital action, you should first check the completeness and conformity (e.g. check the existence and correctness of fields) to the template digital action format. 
- Then, you should check the contents to decide whether the predicted digital action is sensible given the contexts. 
- Please evaluate separately the format and contents of the prediction with a scale of 0 to {{max_scale}} for each, where 0 is the worst and {{max_scale}} is the best. 
- Please leverage the reference digital action when scoring the predicted digital action, but do not treat it as the single possible answer due to the multi-modal nature of this prediction task. 
- Please only output the score and nothing else. Do not add any explanation to your final answer. 
- Wrap the format score with <format_score> and </format_score> tag, and content score with <content_score> and </content_score> tag.
- For instance, an example full output should look like this: <format_score>{{max_scale}}</format_score>, <content_score>{{max_scale}}</content_score>
""".format(
    examples=OB2_DIGITAL_ACTION_EXAMPLES.replace("{", "{{").replace("}", "}}")
)

CONTEXT_REFERENCE_JUDGE_PROMPT_V2 = """
You are a judge. Your task is to evaluate the digital action predicted by the model. The digital action is is a dictionary falling into one of the following goal types: 
{examples}

Specifically, you are given a set of contexts that describe the video, as well as the predicted digital action that the user might want to take on their phones given the observed contexts.
You need to decide whether the predicted digital action is sensible given the contexts. 
To facilitate this, you are further given a reference digital action that is known to be sensible given the contexts. 

Contexts: 
{{contexts}}

Reference digital action: 
{{gt}}

Model predicted digital action: 
{{prediction}}

NOTE:
- To evaluate the predicted digital action, you should first check the completeness and conformity (e.g. check the existence and correctness of fields) to the template digital action format. 
- Then, you should check the contents to decide whether the predicted digital action is sensible given the contexts. 
- Please evaluate separately the format and contents of the prediction with a score that's either 0, 1 or 2.
- For the format, a score of 0 means the format is completely wrong (e.g. not even a valid dictionary), 1 means it has a partially correct format (e.g. with missing fields) and 2 means a conforming format. 
- For the contents, a score of 0 means the contents are completely wrong (e.g. non-sensible goal type), 1 means that it has partially correct contents (e.g. with sensible goal type but mistakes in other fields, or with a possible but non-convincing goal type), and 2 means all fields are sensible. 
- Please leverage the reference digital action when scoring the predicted digital action, but do not treat it as the single possible answer due to the multi-modal nature of this prediction task. 
- Please only output the score and nothing else. Do not add any explanation to your final answer. 
- Wrap the format score with <format_score> and </format_score> tag, and content score with <content_score> and </content_score> tag.
- For instance, an example full output should look like this: <format_score>2</format_score>, <content_score>2</content_score>
""".format(
    examples=OB2_DIGITAL_ACTION_EXAMPLES.replace("{", "{{").replace("}", "}}")
)

CONTEXT_REFERENCE_JUDGE_PROMPT_V3 = """
You are a judge. Your task is to evaluate the digital action predicted by the model. The digital action is is a dictionary falling into one of the following goal types: 
{examples}

Specifically, you are given a set of contexts that describe the video, as well as the predicted digital action that the user might want to take on their phones given the observed contexts.
You need to decide whether the predicted digital action is sensible given the contexts. 
To facilitate this, you are further given a reference digital action that is known to be sensible given the contexts. 

Contexts: 
{{contexts}}

Reference digital action: 
{{gt}}

Model predicted digital action: 
{{prediction}}

NOTE:
- Please evaluate the prediction with a score that's either 0, 1 or 2 that corresponds to "irrelevant", "borderline relevant", or "very relevant". 
- A score of 0 -> The digital action would probably not be useful for the person in the video. 
- A score of 1 -> The digital action might be useful for the person in the video, but you're not confident. 
- A score of 2 -> The digital action is definitely useful for the person in the video. 
- Please leverage the reference digital action when scoring the predicted digital action, but do not treat it as the single possible answer due to the multi-modal nature of this prediction task. 
- Please only output the score and nothing else. Do not add any explanation to your final answer. 
- Wrap the score with <score> and </score> tag. For instance, an example full output should look like this: <score>2</score>
""".format(
    examples=OB2_DIGITAL_ACTION_EXAMPLES.replace("{", "{{").replace("}", "}}")
)

REFORMAT_PROMPT = """
The digital action is is a dictionary falling into one of the following goal types: 
{examples}

You are given a predicted digital action that the user might want to take on their phones given the observed contexts: 
{{prediction}}

NOTE:
- Your task is to reformat the model prediction to be a valid digital action, that can be directly parsed into a Python dictionary.
- This means you need to strip out any reasoning, prefix and postfix, and also fix any errors in the format.
- Wrap the final answer with <answer> and </answer> tag. For instance, an example full output should look like this: 
<answer>
{{{{
    "type": "store_memory",
    "content": "I finished reading The Housemaid by Freida McFadden"
}}}}
</answer>
""".format(
    examples=OB2_DIGITAL_ACTION_EXAMPLES.replace("{", "{{").replace("}", "}}")
)

CONTEXT_JUDGE_PROMPT = """
You are a judge. Your task is to evaluate the digital action predicted by the model. The digital action is is a dictionary falling into one of the following goal types: 
{examples}

Specifically, you are given a set of contexts that describe the video, as well as the predicted digital action that the user might want to take on their phones given the observed contexts.
You need to decide whether the predicted digital action is sensible given the contexts. 

Contexts: 
{{contexts}}

Model predicted digital action: 
{{prediction}}

NOTE:
- Please evaluate the prediction with a score that's either 0, 1 or 2 that corresponds to "irrelevant", "borderline relevant", or "very relevant". 
- A score of 0 -> The digital action would probably not be useful for the person in the video. 
- A score of 1 -> The digital action might be useful for the person in the video, but you're not confident. 
- A score of 2 -> The digital action is definitely useful for the person in the video. 
- Please only output the score and nothing else. Do not add any explanation to your final answer. 
- Wrap the score with <score> and </score> tag. For instance, an example full output should look like this: <score>2</score>
""".format(
    examples=OB2_DIGITAL_ACTION_EXAMPLES.replace("{", "{{").replace("}", "}}")
)

REFERENCE_JUDGE_PROMPT_V3 = """
You are a judge. Your task is to evaluate the digital action predicted by the model. The digital action is is a dictionary falling into one of the following goal types: 
{examples}

Specifically, you are given the predicted digital action that the user might want to take on their phones given the observed contexts.
You need to decide whether the predicted digital action is sensible by comparing it to a reference digital action that is known to be sensible given the contexts. 

Reference digital action: 
{{gt}}

Model predicted digital action: 
{{prediction}}

NOTE:
- Please evaluate the prediction with a score that's either 0, 1 or 2 that corresponds to "irrelevant", "borderline relevant", or "very relevant". 
- A score of 0 -> The digital action would probably not be useful for the person in the video. 
- A score of 1 -> The digital action might be useful for the person in the video, but you're not confident. 
- A score of 2 -> The digital action is definitely useful for the person in the video. 
- Please only output the score and nothing else. Do not add any explanation to your final answer. 
- Wrap the score with <score> and </score> tag. For instance, an example full output should look like this: <score>2</score>
""".format(
    examples=OB2_DIGITAL_ACTION_EXAMPLES.replace("{", "{{").replace("}", "}}")
)
